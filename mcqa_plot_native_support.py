from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import mcqa_run as base_run
import torch
from mcqa_experiment.compare_runner import CompareExperimentConfig, run_comparison
from mcqa_experiment.data import canonicalize_target_var
from mcqa_experiment.ot import (
    OTConfig,
    load_prepared_alignment_artifacts,
    prepare_alignment_artifacts,
    save_prepared_alignment_artifacts,
)
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import enumerate_residual_sites
from mcqa_experiment.support import extract_ordered_site_support


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_NATIVE_RESOLUTIONS = [16, 32, 128, 256]
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)


def _synchronize_if_cuda(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _parse_csv_ints(value: str | None) -> list[int] | None:
    items = _parse_csv_strings(value)
    if items is None:
        return None
    return [int(item) for item in items]


def _parse_csv_floats(value: str | None) -> list[float] | None:
    items = _parse_csv_strings(value)
    if items is None:
        return None
    return [float(item) for item in items]


def _normalize_native_resolutions(values: list[int] | tuple[int, ...], *, hidden_size: int) -> tuple[int, ...]:
    resolved = [max(1, min(int(value), int(hidden_size))) for value in values if int(value) > 0]
    if not resolved:
        raise ValueError("Native resolutions must contain at least one positive width")
    return tuple(dict.fromkeys(int(value) for value in resolved))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical native-support PLOT OT localization for MCQA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", required=True, help="Comma-separated focused layers.")
    parser.add_argument(
        "--native-resolutions",
        default=None,
        help="Optional comma-separated native support widths. Runs the full OT sweep separately for each width.",
    )
    parser.add_argument("--ot-epsilons", help="Comma-separated OT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--ot-top-k-values", help="Comma-separated OT top-k values. Default: 1,2,4")
    parser.add_argument("--ot-lambdas", help="Comma-separated OT lambdas. Default: 0.5,1,2,4")
    parser.add_argument(
        "--calibration-family-weights",
        help="Comma-separated family weights in answerPosition,randomLetter,answerPosition_randomLetter order. Default: 1,1,1",
    )
    parser.add_argument("--support-score-slack", type=float, default=0.05)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--results-root", default="results/delta")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _configure_base_run(args: argparse.Namespace, *, sweep_root: Path, results_timestamp: str) -> None:
    base_run.DEVICE = str(args.device)
    base_run.MODEL_NAME = str(args.model_name)
    base_run.RUN_TIMESTAMP = str(results_timestamp)
    base_run.RUN_DIR = sweep_root
    base_run.OUTPUT_PATH = sweep_root / "mcqa_run_results.json"
    base_run.SUMMARY_PATH = sweep_root / "mcqa_run_summary.txt"
    base_run.SIGNATURES_DIR = Path(args.signatures_dir)
    base_run.MCQA_DATASET_PATH = str(args.dataset_path)
    base_run.MCQA_DATASET_CONFIG = args.dataset_config or None
    base_run.DATASET_SIZE = int(args.dataset_size)
    base_run.SPLIT_SEED = int(args.split_seed)
    base_run.TRAIN_POOL_SIZE = int(args.train_pool_size)
    base_run.CALIBRATION_POOL_SIZE = int(args.calibration_pool_size)
    base_run.TEST_POOL_SIZE = int(args.test_pool_size)
    base_run.BATCH_SIZE = int(args.batch_size)
    base_run.TARGET_VARS = list(DEFAULT_TARGET_VARS)
    base_run.COUNTERFACTUAL_NAMES = list(DEFAULT_COUNTERFACTUAL_NAMES)
    base_run.PROMPT_HF_LOGIN = bool(args.prompt_hf_login)
    base_run.TOKEN_POSITION_IDS = [DEFAULT_TOKEN_POSITION_ID]


def _load_existing_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _best_ot_records(ot_compare_payloads: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best_by_var: dict[str, dict[str, object]] = {}
    for compare_payload in ot_compare_payloads:
        epsilon = float(compare_payload.get("ot_epsilon", 0.0))
        for payload in compare_payload.get("method_payloads", {}).get("ot", []):
            result = payload.get("results", [{}])[0]
            target_var = str(payload.get("target_var"))
            candidate = {
                "exact_acc": float(result.get("exact_acc", 0.0)),
                "selection_score": float(result.get("selection_score", 0.0)),
                "site_label": str(result.get("site_label", "")),
                "epsilon": float(epsilon),
                "selected_hyperparameters": dict(payload.get("selected_hyperparameters", {})),
            }
            previous = best_by_var.get(target_var)
            if previous is None or (
                float(candidate["selection_score"]),
                float(candidate["exact_acc"]),
            ) > (
                float(previous["selection_score"]),
                float(previous["exact_acc"]),
            ):
                best_by_var[target_var] = candidate
    return best_by_var


def _format_summary(*, layer: int, native_resolution: int, support_by_var: dict[str, dict[str, object]], best_ot_by_var: dict[str, dict[str, object]]) -> str:
    lines = [
        "MCQA PLOT Native Support Summary",
        f"layer: {int(layer)}",
        f"token_position_id: {DEFAULT_TOKEN_POSITION_ID}",
        f"native_resolution: {int(native_resolution)}",
        "",
    ]
    for target_var in DEFAULT_TARGET_VARS:
        best = best_ot_by_var.get(str(target_var), {})
        lines.append(f"[{target_var}]")
        lines.append(
            f"best_ot exact={float(best.get('exact_acc', 0.0)):.4f} "
            f"cal={float(best.get('selection_score', 0.0)):.4f} "
            f"eps={float(best.get('epsilon', 0.0)):g} "
            f"site={best.get('site_label')}"
        )
        support_summary = support_by_var.get(str(target_var), {})
        if support_summary:
            lines.append(f"support masks={[candidate.get('name') for candidate in support_summary.get('mask_candidates', [])]}")
            lines.append(f"ranked_sites={support_summary.get('ranked_site_labels', [])[:8]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_plot_native_support"
    sweep_root.mkdir(parents=True, exist_ok=True)

    _configure_base_run(args, sweep_root=sweep_root, results_timestamp=results_timestamp)
    context = base_run.build_run_context()
    context_timing_seconds = dict(context.get("timing_seconds", {}))
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_positions = context["token_positions"]
    banks_by_split = context["banks_by_split"]
    data_metadata = context["data_metadata"]
    device = context["device"]
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    layers = tuple(_parse_csv_ints(args.layers) or [])
    if not layers:
        raise ValueError("No layers selected")
    native_resolutions = _normalize_native_resolutions(
        _parse_csv_ints(args.native_resolutions) or list(DEFAULT_NATIVE_RESOLUTIONS),
        hidden_size=int(model.config.hidden_size),
    )
    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))
    ot_top_k_values = tuple(_parse_csv_ints(args.ot_top_k_values) or list(DEFAULT_OT_TOP_K_VALUES))
    ot_lambdas = tuple(_parse_csv_floats(args.ot_lambdas) or list(DEFAULT_OT_LAMBDAS))
    calibration_family_weights = tuple(
        _parse_csv_floats(args.calibration_family_weights) or list(DEFAULT_CALIBRATION_FAMILY_WEIGHTS)
    )
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    hidden_size = int(model.config.hidden_size)

    print(
        f"[stageB native] start layers={list(int(layer) for layer in layers)} "
        f"native_resolutions={list(int(width) for width in native_resolutions)} "
        f"target_vars={list(target_vars)}"
    )
    print(
        "[stageB native] signature construction is per (layer, native_resolution) family, "
        "not per abstract variable. For each selected layer+width, one shared neural-site "
        "signature pass is built and then reused across both target variables and all epsilons."
    )
    print(
        f"[stageB native] total layer-width families={len(layers) * len(native_resolutions)} "
        f"(layers={len(layers)} x widths={len(native_resolutions)})"
    )

    train_banks = {target_var: banks_by_split["train"][target_var] for target_var in target_vars}
    all_payloads: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []

    for layer in layers:
        layer_dir = sweep_root / f"layer_{int(layer):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for native_resolution in native_resolutions:
            print("")
            print(
                f"[stageB native] layer={int(layer)} width={int(native_resolution)} "
                "begin one shared layer-width family"
            )
            width_start = perf_counter()
            sites = enumerate_residual_sites(
                num_layers=int(model.config.num_hidden_layers),
                hidden_size=hidden_size,
                token_position_ids=token_position_ids,
                resolution=int(native_resolution),
                layers=(int(layer),),
                selected_token_position_ids=(DEFAULT_TOKEN_POSITION_ID,),
            )
            cache_spec = base_run._signature_cache_spec(
                train_bank=train_banks[target_vars[0]],
                resolution=int(native_resolution),
                resolved_resolution=int(native_resolution),
                signature_mode=str(args.signature_mode),
                selected_layers=[int(layer)],
                token_position_ids=token_position_ids,
            )
            cache_path = base_run._signature_cache_path(
                resolution=int(native_resolution),
                signature_mode=str(args.signature_mode),
                cache_spec=cache_spec,
            )
            artifact_load_start = perf_counter()
            prepared_artifacts = load_prepared_alignment_artifacts(cache_path, expected_spec=cache_spec)
            artifact_prepare_load_seconds = float(perf_counter() - artifact_load_start)
            artifact_prepare_create_seconds = 0.0
            if prepared_artifacts is None:
                print(
                    f"[stageB native] signature cache miss layer={int(layer)} width={int(native_resolution)} "
                    f"sites={len(sites)}; building one shared neural-site signature pass "
                    f"for target_vars={list(target_vars)} and reusing it across epsilons={list(float(e) for e in ot_epsilons)}"
                )
                artifact_prepare_start = perf_counter()
                prepared_artifacts = prepare_alignment_artifacts(
                    model=model,
                    fit_banks_by_var=train_banks,
                    sites=sites,
                    device=device,
                    config=OTConfig(
                        method="ot",
                        batch_size=int(args.batch_size),
                        epsilon=1.0,
                        signature_mode=str(args.signature_mode),
                        top_k_values=ot_top_k_values,
                        lambda_values=ot_lambdas,
                        source_target_vars=target_vars,
                        calibration_metric=DEFAULT_CALIBRATION_METRIC,
                        calibration_family_weights=calibration_family_weights,
                        top_k_values_by_var={target_var: ot_top_k_values for target_var in target_vars},
                        lambda_values_by_var={target_var: ot_lambdas for target_var in target_vars},
                    ),
                )
                artifact_prepare_create_seconds = float(perf_counter() - artifact_prepare_start)
                save_prepared_alignment_artifacts(
                    cache_path,
                    prepared_artifacts=prepared_artifacts,
                    cache_spec=cache_spec,
                )
                print(
                    f"[stageB native] signature cache saved layer={int(layer)} width={int(native_resolution)} "
                    f"path={cache_path}"
                )
            else:
                print(
                    f"[stageB native] signature cache hit layer={int(layer)} width={int(native_resolution)} "
                    f"path={cache_path} loaded_from_disk={bool(prepared_artifacts.get('loaded_from_disk', False))}"
                )
                print(
                    f"[stageB native] reuse one shared neural-site signature family "
                    f"across target_vars={list(target_vars)} and epsilons={list(float(e) for e in ot_epsilons)}"
                )
            _synchronize_if_cuda(device)

            ot_compare_payloads: list[dict[str, object]] = []
            ot_localization_start = perf_counter()
            print(
                f"[stageB native] sweeping OT over layer={int(layer)} width={int(native_resolution)} "
                f"epsilons={list(float(e) for e in ot_epsilons)}"
            )
            for epsilon in ot_epsilons:
                output_stem = (
                    f"mcqa_plot_native_support_layer-{int(layer)}_pos-{DEFAULT_TOKEN_POSITION_ID}"
                    f"_atomic-{int(native_resolution)}_sig-{str(args.signature_mode)}_eps-{float(epsilon):g}_ot"
                )
                output_path = layer_dir / f"{output_stem}.json"
                summary_path = layer_dir / f"{output_stem}.txt"
                compare_payload = _load_existing_payload(output_path)
                if compare_payload is None:
                    compare_payload = run_comparison(
                        model=model,
                        tokenizer=tokenizer,
                        token_positions=token_positions,
                        banks_by_split=banks_by_split,
                        data_metadata=data_metadata,
                        device=device,
                        config=CompareExperimentConfig(
                            model_name=base_run.MODEL_NAME,
                            output_path=output_path,
                            summary_path=summary_path,
                            methods=("ot",),
                            target_vars=target_vars,
                            batch_size=int(args.batch_size),
                            ot_epsilon=float(epsilon),
                            signature_mode=str(args.signature_mode),
                            ot_top_k_values=ot_top_k_values,
                            ot_lambdas=ot_lambdas,
                            calibration_metric=DEFAULT_CALIBRATION_METRIC,
                            calibration_family_weights=calibration_family_weights,
                            ot_top_k_values_by_var={target_var: ot_top_k_values for target_var in target_vars},
                            ot_lambdas_by_var={target_var: ot_lambdas for target_var in target_vars},
                            resolution=int(native_resolution),
                            layers=(int(layer),),
                            token_position_ids=(DEFAULT_TOKEN_POSITION_ID,),
                        ),
                        prepared_ot_artifacts=prepared_artifacts,
                    )
                else:
                    print(
                        f"[stageB native] reusing compare payload layer={int(layer)} width={int(native_resolution)} "
                        f"epsilon={float(epsilon):g}"
                    )
                ot_compare_payloads.append(compare_payload)
            _synchronize_if_cuda(device)
            ot_localization_seconds = float(perf_counter() - ot_localization_start)

            support_start = perf_counter()
            support_by_var = extract_ordered_site_support(
                ot_run_payloads=ot_compare_payloads,
                sites=sites,
                score_slack=float(args.support_score_slack),
            )
            support_extract_seconds = float(perf_counter() - support_start)
            width_total_seconds = float(perf_counter() - width_start)
            best_ot_by_var = _best_ot_records(ot_compare_payloads)

            support_path = layer_dir / (
                f"mcqa_plot_native_support_layer-{int(layer)}_pos-{DEFAULT_TOKEN_POSITION_ID}"
                f"_atomic-{int(native_resolution)}_sig-{str(args.signature_mode)}_support.json"
            )
            summary_path = layer_dir / (
                f"mcqa_plot_native_support_layer-{int(layer)}_pos-{DEFAULT_TOKEN_POSITION_ID}"
                f"_atomic-{int(native_resolution)}_sig-{str(args.signature_mode)}_summary.txt"
            )
            payload_path = layer_dir / (
                f"mcqa_plot_native_support_layer-{int(layer)}_pos-{DEFAULT_TOKEN_POSITION_ID}"
                f"_atomic-{int(native_resolution)}_sig-{str(args.signature_mode)}.json"
            )
            write_json(support_path, support_by_var)
            write_text_report(
                summary_path,
                _format_summary(
                    layer=int(layer),
                    native_resolution=int(native_resolution),
                    support_by_var=support_by_var,
                    best_ot_by_var=best_ot_by_var,
                ),
            )
            payload = {
                "kind": "mcqa_plot_native_support_layer",
                "layer": int(layer),
                "token_position_id": DEFAULT_TOKEN_POSITION_ID,
                "signature_mode": str(args.signature_mode),
                "native_resolution": int(native_resolution),
                "native_resolutions": [int(resolution) for resolution in native_resolutions],
                "ot_epsilons": [float(epsilon) for epsilon in ot_epsilons],
                "ot_top_k_values": [int(value) for value in ot_top_k_values],
                "ot_lambdas": [float(value) for value in ot_lambdas],
                "calibration_family_weights": [float(weight) for weight in calibration_family_weights],
                "support_score_slack": float(args.support_score_slack),
                "site_labels": [site.label for site in sites],
                "support_path": str(support_path),
                "support_by_var": support_by_var,
                "method_by_var": best_ot_by_var,
                "ot_output_paths": [
                    str(
                        layer_dir
                        / (
                            f"mcqa_plot_native_support_layer-{int(layer)}_pos-{DEFAULT_TOKEN_POSITION_ID}"
                            f"_atomic-{int(native_resolution)}_sig-{str(args.signature_mode)}_eps-{float(epsilon):g}_ot.json"
                        )
                    )
                    for epsilon in ot_epsilons
                ],
                "summary_path": str(summary_path),
                "context_timing_seconds": context_timing_seconds,
                "timing_seconds": {
                    "t_model_load": float(context_timing_seconds.get("t_model_load", 0.0)),
                    "t_data_load": float(context_timing_seconds.get("t_data_load", 0.0)),
                    "t_bank_build": float(context_timing_seconds.get("t_bank_build", 0.0)),
                    "t_factual_filter": float(context_timing_seconds.get("t_factual_filter", 0.0)),
                    "t_context_total_wall": float(context_timing_seconds.get("t_context_total_wall", 0.0)),
                    "t_artifact_prepare_load": float(artifact_prepare_load_seconds),
                    "t_artifact_prepare_create": float(artifact_prepare_create_seconds),
                    "t_stageB_native_ot_localization": float(ot_localization_seconds),
                    "t_support_extract": float(support_extract_seconds),
                    "t_native_width_total_wall": float(width_total_seconds),
                },
                "artifact_cache_hit": bool(prepared_artifacts.get("loaded_from_disk", False)),
                "localization_runtime_seconds": float(width_total_seconds),
                "runtime_seconds": float(width_total_seconds),
            }
            write_json(payload_path, payload)
            all_payloads.append(payload)
            manifest_runs.append(
                {
                    "layer": int(layer),
                    "native_resolution": int(native_resolution),
                    "payload_path": str(payload_path),
                    "runtime_seconds": float(width_total_seconds),
                }
            )

    manifest_path = sweep_root / "layer_sweep_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "mcqa_plot_native_support",
            "layers": [int(layer) for layer in layers],
            "native_resolutions": [int(width) for width in native_resolutions],
            "runtime_seconds": float(perf_counter() - stage_start),
            "runs": manifest_runs,
        },
    )
    aggregate_path = sweep_root / "mcqa_run_results.json"
    write_json(aggregate_path, {"runs": all_payloads})
    print(f"Wrote native support localization manifest to {manifest_path}")


if __name__ == "__main__":
    main()
