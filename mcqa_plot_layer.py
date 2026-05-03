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
    run_bruteforce_site_pipeline,
    save_prepared_alignment_artifacts,
)
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import enumerate_residual_sites
from mcqa_experiment.support import extract_ordered_site_support


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_UOT_BETA_NEURALS = (0.1, 0.3, 1.0, 3.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_DISPLAY_TOP_LAYER_COUNT = 3
DEFAULT_STAGE_A_SINGLE_LAYER_LAMBDAS = (1.0,)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint layer-level PLOT UOT localization for MCQA.")
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
    parser.add_argument("--layers", help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument("--token-position-id", default=DEFAULT_TOKEN_POSITION_ID)
    parser.add_argument("--ot-epsilons", help="Comma-separated UOT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--uot-beta-neurals", help="Comma-separated UOT beta_neural values. Default: 0.1,0.3,1,3")
    parser.add_argument("--support-score-slack", type=float, default=0.05)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--results-root", default="results")
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
    base_run.TOKEN_POSITION_IDS = [str(args.token_position_id)]


def _load_existing_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _target_row_ranking(payload: dict[str, object], *, sites) -> list[dict[str, object]]:
    target_row_transport = payload.get("target_normalized_transport", payload.get("target_transport", []))
    if not isinstance(target_row_transport, list) or not target_row_transport:
        return []
    first_entry = target_row_transport[0]
    if isinstance(first_entry, list):
        row_transport = first_entry
    else:
        row_transport = target_row_transport
    ranking = [
        {
            "site_index": int(site_index),
            "site_label": sites[int(site_index)].label,
            "layer": int(sites[int(site_index)].layer),
            "token_position_id": str(sites[int(site_index)].token_position_id),
            "dim_start": int(sites[int(site_index)].dim_start),
            "dim_end": int(sites[int(site_index)].dim_end),
            "transport_mass": float(row_transport[site_index]),
        }
        for site_index in range(min(len(sites), len(row_transport)))
    ]
    return sorted(
        ranking,
        key=lambda entry: (
            float(entry.get("transport_mass", 0.0)),
            -int(entry.get("site_index", 10**9)),
        ),
        reverse=True,
    )


def _iter_stage_a_transport_payloads(compare_payload: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    method_payloads = compare_payload.get("method_payloads", {})
    if not isinstance(method_payloads, dict):
        return []
    payloads: list[tuple[str, dict[str, object]]] = []
    for method_name in ("uot", "ot"):
        entries = method_payloads.get(method_name, [])
        if not isinstance(entries, list):
            continue
        for payload in entries:
            if isinstance(payload, dict):
                payloads.append((str(method_name), payload))
    return payloads


def _select_layer_method_by_var(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    ot_compare_payloads: list[dict[str, object]],
    device,
    batch_size: int,
    signature_mode: str,
) -> dict[str, dict[str, object]]:
    selected_by_var: dict[str, dict[str, object]] = {}
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    for target_var in target_vars:
        best_record: dict[str, object] | None = None
        for compare_payload in ot_compare_payloads:
            epsilon = float(compare_payload.get("ot_epsilon", 0.0))
            compare_beta = compare_payload.get("uot_beta_neural")
            for method_name, payload in _iter_stage_a_transport_payloads(compare_payload):
                if str(payload.get("target_var")) != str(target_var):
                    continue
                row_ranking = _target_row_ranking(payload, sites=sites)
                top_site_index = int(row_ranking[0].get("site_index", -1)) if row_ranking else -1
                if top_site_index < 0 and sites:
                    top_site_index = 0
                candidate_sites = [sites[top_site_index]] if 0 <= int(top_site_index) < len(sites) else []
                if not candidate_sites:
                    continue
                brute_payload = run_bruteforce_site_pipeline(
                    model=model,
                    calibration_bank=banks_by_split["calibration"][str(target_var)],
                    holdout_bank=banks_by_split["test"][str(target_var)],
                    sites=candidate_sites,
                    device=device,
                    tokenizer=tokenizer,
                    config=OTConfig(
                        method="bruteforce",
                        batch_size=int(batch_size),
                        epsilon=1.0,
                        signature_mode=str(signature_mode),
                        top_k_values=(1,),
                        lambda_values=DEFAULT_STAGE_A_SINGLE_LAYER_LAMBDAS,
                        selection_verbose=True,
                        source_target_vars=target_vars,
                        calibration_metric=DEFAULT_CALIBRATION_METRIC,
                        calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                        lambda_values_by_var={str(target_var): DEFAULT_STAGE_A_SINGLE_LAYER_LAMBDAS},
                    ),
                )
                result = brute_payload.get("results", [{}])[0]
                selected_hyperparameters = dict(brute_payload.get("selected_hyperparameters", {}))
                candidate = {
                    "method": str(payload.get("method", method_name)),
                    "variable": str(target_var),
                    "exact_acc": float(result.get("exact_acc", 0.0)),
                    "selection_score": float(result.get("selection_score", 0.0)),
                    "site_label": str(result.get("site_label", selected_hyperparameters.get("site_label", ""))),
                    "layer": int(result.get("layer", candidate_sites[0].layer)),
                    "lambda": float(selected_hyperparameters.get("lambda", result.get("lambda", 0.0))),
                    "epsilon": float(epsilon),
                    "uot_beta_neural": None
                    if payload.get("uot_beta_neural", compare_beta) is None
                    else float(payload.get("uot_beta_neural", compare_beta)),
                    "candidate_site_labels": [site.label for site in candidate_sites],
                    "candidate_layers": [int(site.layer) for site in candidate_sites],
                    "coupling_argmax_site_label": candidate_sites[0].label,
                    "coupling_argmax_layer": int(candidate_sites[0].layer),
                    "coupling_top_site_labels": [str(entry.get("site_label", "")) for entry in row_ranking[:DEFAULT_DISPLAY_TOP_LAYER_COUNT]],
                    "coupling_top_layers": [int(entry.get("layer", -1)) for entry in row_ranking[:DEFAULT_DISPLAY_TOP_LAYER_COUNT]],
                    "selection_basis": "single_site_on_coupling_argmax_layer",
                    "target_row_ranking": row_ranking,
                    "payload": brute_payload,
                }
                if best_record is None or (
                    float(candidate["selection_score"]),
                    float(candidate["exact_acc"]),
                ) > (
                    float(best_record["selection_score"]),
                    float(best_record["exact_acc"]),
                ):
                    best_record = candidate
        if best_record is not None:
            selected_by_var[str(target_var)] = best_record
    return selected_by_var


def _rank_layers_from_target_row(
    *,
    selected_method_by_var: dict[str, dict[str, object]],
    runtime_seconds: float,
) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {}
    for target_var, selected in selected_method_by_var.items():
        row_ranking = [dict(entry) for entry in selected.get("target_row_ranking", []) if isinstance(entry, dict)]
        rankings[str(target_var)] = [
            {
                "variable": str(target_var),
                "layer": int(entry.get("layer", -1)),
                "selection_score": float(entry.get("transport_mass", 0.0)),
                "target_row_transport_mass": float(entry.get("transport_mass", 0.0)),
                "exact_acc": float(selected.get("exact_acc", 0.0)),
                "selection_exact_acc": float(selected.get("exact_acc", 0.0)),
                "method_selection_score": float(selected.get("selection_score", 0.0)),
                "method": selected.get("method"),
                "epsilon": float(selected.get("epsilon", 0.0)),
                "uot_beta_neural": selected.get("uot_beta_neural"),
                "site_label": entry.get("site_label"),
                "selected_lambda": float(selected.get("lambda", 0.0)),
                "selected_site_label": selected.get("site_label"),
                "runtime_seconds": float(runtime_seconds),
                "selection_basis": "target_row_mass_from_selected_single_layer_method",
            }
            for entry in row_ranking
        ]
    return rankings


def _format_summary(
    *,
    token_position_id: str,
    layers: tuple[int, ...],
    support_by_var: dict[str, dict[str, object]],
    method_by_var: dict[str, dict[str, object]],
) -> str:
    lines = [
        "MCQA PLOT Layer Summary",
        f"token_position_id: {token_position_id}",
        f"layers: {list(int(layer) for layer in layers)}",
        "",
    ]
    for target_var in DEFAULT_TARGET_VARS:
        selected = method_by_var.get(str(target_var), {})
        lines.append(f"[{target_var}]")
        if selected:
            lines.append(
                f"selected_single_layer method={selected.get('method')} "
                f"exact={float(selected.get('exact_acc', 0.0)):.4f} "
                f"cal={float(selected.get('selection_score', 0.0)):.4f} "
                f"eps={float(selected.get('epsilon', 0.0)):g} "
                f"site={selected.get('site_label')} "
                f"strength={float(selected.get('lambda', 0.0)):g} "
                + (
                    f"beta_n={float(selected.get('uot_beta_neural')):g} "
                    if selected.get("uot_beta_neural") is not None
                    else ""
                )
                + f"coupling_argmax={selected.get('coupling_argmax_site_label')}"
            )
        if selected.get("target_row_ranking"):
            lines.append(
                f"coupling_row_layers={[entry.get('site_label') for entry in selected.get('target_row_ranking', [])[:DEFAULT_DISPLAY_TOP_LAYER_COUNT]]}"
            )
        support_summary = support_by_var.get(str(target_var), {})
        if support_summary:
            lines.append(
                f"support_diagnostic_masks={[candidate.get('name') for candidate in support_summary.get('mask_candidates', [])]}"
            )
            lines.append(
                f"support_diagnostic_ranked_layers={support_summary.get('ranked_site_labels', [])}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_plot_layer"
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

    resolved_layers = tuple(
        _parse_csv_ints(args.layers) or list(range(int(model.config.num_hidden_layers)))
    )
    if not resolved_layers:
        raise ValueError("No layers selected")
    print(
        f"[stageA] start token_position={str(args.token_position_id)} "
        f"layers={len(resolved_layers)} target_vars={list(target_vars)}"
    )
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    sites = enumerate_residual_sites(
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        token_position_ids=token_position_ids,
        resolution=None,
        layers=resolved_layers,
        selected_token_position_ids=(str(args.token_position_id),),
    )

    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))
    uot_beta_neurals = tuple(_parse_csv_floats(args.uot_beta_neurals) or list(DEFAULT_UOT_BETA_NEURALS))
    train_banks = {target_var: banks_by_split["train"][target_var] for target_var in target_vars}
    cache_spec = base_run._signature_cache_spec(
        train_bank=train_banks[target_vars[0]],
        resolution=None,
        resolved_resolution=None,
        signature_mode=str(args.signature_mode),
        selected_layers=list(int(layer) for layer in resolved_layers),
        token_position_ids=token_position_ids,
    )
    cache_path = base_run._signature_cache_path(
        resolution=None,
        signature_mode=str(args.signature_mode),
        cache_spec=cache_spec,
    )
    artifact_load_start = perf_counter()
    prepared_artifacts = load_prepared_alignment_artifacts(cache_path, expected_spec=cache_spec)
    artifact_prepare_load_seconds = float(perf_counter() - artifact_load_start)
    artifact_prepare_create_seconds = 0.0
    if prepared_artifacts is None:
        print(f"[stageA] signature cache miss path={cache_path}")
        artifact_prepare_start = perf_counter()
        prepared_artifacts = prepare_alignment_artifacts(
            model=model,
            fit_banks_by_var=train_banks,
            sites=sites,
            device=device,
            config=OTConfig(
                method="uot",
                batch_size=int(args.batch_size),
                epsilon=float(ot_epsilons[0]),
                uot_beta_neural=float(uot_beta_neurals[0]),
                signature_mode=str(args.signature_mode),
                top_k_values=DEFAULT_OT_TOP_K_VALUES,
                lambda_values=DEFAULT_OT_LAMBDAS,
                source_target_vars=target_vars,
                calibration_metric=DEFAULT_CALIBRATION_METRIC,
                calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                top_k_values_by_var={target_var: DEFAULT_OT_TOP_K_VALUES for target_var in target_vars},
                lambda_values_by_var={target_var: DEFAULT_OT_LAMBDAS for target_var in target_vars},
            ),
        )
        artifact_prepare_create_seconds = float(perf_counter() - artifact_prepare_start)
        save_prepared_alignment_artifacts(
            cache_path,
            prepared_artifacts=prepared_artifacts,
            cache_spec=cache_spec,
        )
        print(f"[stageA] signature cache saved path={cache_path}")
    else:
        print(
            f"[stageA] signature cache hit path={cache_path} "
            f"loaded_from_disk={bool(prepared_artifacts.get('loaded_from_disk', False))}"
        )
    _synchronize_if_cuda(device)

    ot_compare_payloads: list[dict[str, object]] = []
    ot_localization_start = perf_counter()
    print(
        f"[stageA] running joint layer UOT epsilon_sweep={list(ot_epsilons)} "
        f"beta_neural_sweep={list(uot_beta_neurals)}"
    )
    for epsilon in ot_epsilons:
        for beta_neural in uot_beta_neurals:
            print(
                f"[stageA] epsilon={float(epsilon):g} beta_neural={float(beta_neural):g} "
                "joint layer UOT"
            )
            output_stem = (
                f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}"
                f"_eps-{float(epsilon):g}_betan-{float(beta_neural):g}_uot"
            )
            output_path = sweep_root / f"{output_stem}.json"
            summary_path = sweep_root / f"{output_stem}.txt"
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
                        methods=("uot",),
                        target_vars=target_vars,
                        batch_size=int(args.batch_size),
                        ot_epsilon=float(epsilon),
                        uot_beta_neural=float(beta_neural),
                        signature_mode=str(args.signature_mode),
                        ot_top_k_values=DEFAULT_OT_TOP_K_VALUES,
                        ot_lambdas=DEFAULT_OT_LAMBDAS,
                        calibration_metric=DEFAULT_CALIBRATION_METRIC,
                        calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                        ot_top_k_values_by_var={target_var: DEFAULT_OT_TOP_K_VALUES for target_var in target_vars},
                        ot_lambdas_by_var={target_var: DEFAULT_OT_LAMBDAS for target_var in target_vars},
                        resolution=None,
                        layers=resolved_layers,
                        token_position_ids=(str(args.token_position_id),),
                    ),
                    prepared_ot_artifacts=prepared_artifacts,
                )
            else:
                print(f"[stageA] reusing existing UOT payload path={output_path}")
            ot_compare_payloads.append(compare_payload)
    _synchronize_if_cuda(device)
    ot_localization_seconds = float(perf_counter() - ot_localization_start)

    support_start = perf_counter()
    print("[stageA] extracting UOT support diagnostics")
    support_by_var = extract_ordered_site_support(
        ot_run_payloads=ot_compare_payloads,
        sites=sites,
        score_slack=float(args.support_score_slack),
    )
    support_extract_seconds = float(perf_counter() - support_start)
    print(
        "[stageA] selecting PLOT(layer) winner by testing the argmax layer "
        "from each target-variable coupling row"
    )
    method_by_var = _select_layer_method_by_var(
        model=model,
        tokenizer=tokenizer,
        banks_by_split=banks_by_split,
        sites=sites,
        ot_compare_payloads=ot_compare_payloads,
        device=device,
        batch_size=int(args.batch_size),
        signature_mode=str(args.signature_mode),
    )
    total_seconds = float(perf_counter() - stage_start)
    rankings_by_var = _rank_layers_from_target_row(
        selected_method_by_var=method_by_var,
        runtime_seconds=total_seconds,
    )

    support_path = sweep_root / f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}_support.json"
    summary_path = sweep_root / f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}_summary.txt"
    payload_path = sweep_root / f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}.json"
    write_json(support_path, support_by_var)
    write_text_report(
        summary_path,
        _format_summary(
            token_position_id=str(args.token_position_id),
            layers=resolved_layers,
            support_by_var=support_by_var,
            method_by_var=method_by_var,
        ),
    )
    write_json(
        payload_path,
        {
            "kind": "mcqa_plot_layer",
            "token_position_id": str(args.token_position_id),
            "layers": [int(layer) for layer in resolved_layers],
            "signature_mode": str(args.signature_mode),
            "ot_epsilons": [float(epsilon) for epsilon in ot_epsilons],
            "uot_beta_neurals": [float(beta_neural) for beta_neural in uot_beta_neurals],
            "support_score_slack": float(args.support_score_slack),
            "site_labels": [site.label for site in sites],
            "support_path": str(support_path),
            "support_by_var": support_by_var,
            "rankings_by_var": rankings_by_var,
            "display_method_by_var": method_by_var,
            "method_by_var": method_by_var,
            "stage_a_transport_method": "uot",
            "ot_output_paths": [
                str(
                    sweep_root
                    / (
                        f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}"
                        f"_eps-{float(epsilon):g}_betan-{float(beta_neural):g}_uot.json"
                    )
                )
                for epsilon in ot_epsilons
                for beta_neural in uot_beta_neurals
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
                "t_stageA_ot_localization": float(ot_localization_seconds),
                "t_support_extract": float(support_extract_seconds),
                "t_stage_total_wall": float(total_seconds),
            },
            "artifact_cache_hit": bool(prepared_artifacts.get("loaded_from_disk", False)),
            "localization_runtime_seconds": float(total_seconds),
            "runtime_seconds": float(total_seconds),
        },
    )
    print(f"Wrote PLOT layer payload to {payload_path}")


if __name__ == "__main__":
    main()
