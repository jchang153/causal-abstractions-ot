from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.mcqa import mcqa_run as base_run
import torch
from experiments.mcqa.mcqa_experiment.compare_runner import CompareExperimentConfig, run_comparison
from experiments.mcqa.mcqa_experiment.das import DASConfig, run_das_pipeline
from experiments.mcqa.mcqa_experiment.data import canonicalize_target_var
from experiments.mcqa.mcqa_experiment.ot import (
    OTConfig,
    load_prepared_alignment_artifacts,
    prepare_alignment_artifacts,
    save_prepared_alignment_artifacts,
)
from experiments.mcqa.mcqa_experiment.reporting import write_text_report
from experiments.mcqa.mcqa_experiment.runtime import write_json
from experiments.mcqa.mcqa_experiment.sites import enumerate_residual_sites, site_total_width
from experiments.mcqa.mcqa_experiment.support import build_mask_sites_from_support, extract_block_mask_support


DEFAULT_LAYERS = (20, 25)
DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_IDS = ("last_token",)
DEFAULT_BLOCK_RESOLUTIONS = (128, 144, 192, 256, 288, 384, 576, 768)
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_DAS_SUBSPACE_DIMS = (
    32,
    64,
    96,
    128,
    256,
    512,
    768,
    1024,
    1536,
    2048,
    2304,
)
DEFAULT_SCREEN_MAX_EPOCHS = 25
DEFAULT_SCREEN_MIN_EPOCHS = 2
DEFAULT_FULL_MASK_LIMIT = 2


def _synchronize_if_cuda(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _guided_subspace_dims(max_width: int) -> tuple[int, ...]:
    resolved_width = max(1, int(max_width))
    raw_dims = [
        32,
        64,
        96,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2304,
    ]
    dims = tuple(int(dim) for dim in raw_dims if 1 <= int(dim) <= int(resolved_width))
    return dims or (resolved_width,)


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
    parser = argparse.ArgumentParser(description="Simplified fixed-layer last-token blockwise OT -> DAS runner.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", help="Comma-separated focused layers. Default: 20,25")
    parser.add_argument("--block-resolutions", help="Comma-separated last-token block widths. Default: 128,144,192,256,288,384,576,768")
    parser.add_argument("--include-144", action="store_true", help="Append 144-d blocks to the resolution grid.")
    parser.add_argument("--ot-epsilons", help="Comma-separated OT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--support-score-slack", type=float, default=0.05)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument(
        "--signature-ablation-modes",
        help="Comma-separated extra signature modes to run in addition to the mainline.",
    )
    parser.add_argument("--screen-max-epochs", type=int, default=DEFAULT_SCREEN_MAX_EPOCHS)
    parser.add_argument("--screen-min-epochs", type=int, default=DEFAULT_SCREEN_MIN_EPOCHS)
    parser.add_argument("--screen-restarts", type=int, default=1)
    parser.add_argument("--full-restarts", type=int, default=2)
    parser.add_argument("--full-mask-limit", type=int, default=DEFAULT_FULL_MASK_LIMIT)
    parser.add_argument(
        "--full-das-subspace-dims",
        help="Comma-separated full-layer DAS dims. Default: wide paper grid.",
    )
    parser.add_argument(
        "--guided-subspace-dims",
        default="32,64,96,128,256,512,768,1024,1536,2048,2304",
        help="Comma-separated native-guided DAS dims. Defaults to the paper master grid.",
    )
    parser.add_argument("--results-root", default="results/anvil")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _configure_base_run(args: argparse.Namespace, *, sweep_root: Path, results_timestamp: str) -> None:
    base_run.DEVICE = str(args.device)
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


def _load_existing_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_existing_runs(path: Path) -> list[dict[str, object]]:
    payload = _load_existing_payload(path)
    if not isinstance(payload, dict):
        return []
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, dict)]


def _best_record_from_compare_payload(compare_payload: dict[str, object], *, method: str) -> dict[str, dict[str, object]]:
    best_by_var: dict[str, dict[str, object]] = {}
    for payload in compare_payload.get("method_payloads", {}).get(method, []):
        result = payload.get("results", [{}])[0]
        best_by_var[str(payload.get("target_var"))] = {
            "exact_acc": float(result.get("exact_acc", 0.0)),
            "selection_score": float(result.get("selection_score", result.get("selection_exact_acc", 0.0))),
            "site_label": str(result.get("site_label")),
            "subspace_dim": result.get("subspace_dim"),
        }
    return best_by_var


def _best_screen_sites(
    screen_payload: dict[str, object],
    *,
    target_var: str,
    limit: int,
) -> list[str]:
    search_records = screen_payload.get("search_records", {}).get(str(target_var), [])
    best_by_site: dict[str, dict[str, object]] = {}
    for record in search_records:
        site_label = str(record.get("site_label"))
        previous = best_by_site.get(site_label)
        if previous is None:
            best_by_site[site_label] = record
            continue
        current_key = (
            float(record.get("selection_exact_acc", 0.0)),
            float(record.get("holdout_exact_acc", -1.0)),
            -int(record.get("site_total_dim", 0)),
        )
        previous_key = (
            float(previous.get("selection_exact_acc", 0.0)),
            float(previous.get("holdout_exact_acc", -1.0)),
            -int(previous.get("site_total_dim", 0)),
        )
        if current_key > previous_key:
            best_by_site[site_label] = record
    ranked_records = sorted(
        best_by_site.values(),
        key=lambda record: (
            float(record.get("selection_exact_acc", 0.0)),
            float(record.get("holdout_exact_acc", -1.0)),
            -int(record.get("site_total_dim", 0)),
        ),
        reverse=True,
    )
    return [str(record.get("site_label")) for record in ranked_records[: max(1, int(limit))]]


def _write_das_text_report(path: Path, *, title: str, payload: dict[str, object], extra_lines: list[str] | None = None) -> None:
    result = payload.get("results", [{}])[0]
    lines = [
        title,
        f"site: {result.get('site_label')}",
        f"exact_acc: {float(result.get('exact_acc', 0.0)):.4f}",
        f"calibration_exact_acc: {float(result.get('selection_exact_acc', result.get('calibration_exact_acc', 0.0))):.4f}",
        f"subspace_dim: {result.get('subspace_dim')}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    write_text_report(path, "\n".join(lines))


def _format_resolution_summary(
    *,
    layer: int,
    resolution: int,
    signature_mode: str,
    ot_payloads: list[dict[str, object]],
    support_by_var: dict[str, dict[str, object]],
    das_full_payload: dict[str, object],
    screen_payloads: dict[str, dict[str, object]],
    block_payloads: dict[str, dict[str, object]],
) -> str:
    lines = [
        "MCQA OT + Block DAS Summary",
        f"layer: {int(layer)}",
        f"token_position: last_token",
        f"resolution: {int(resolution)}",
        f"signature_mode: {signature_mode}",
        "",
        "best OT by epsilon:",
    ]
    best_ot_by_var: dict[str, dict[str, object]] = {}
    for compare_payload in ot_payloads:
        epsilon = float(compare_payload.get("ot_epsilon", 0.0))
        for target_var, record in _best_record_from_compare_payload(compare_payload, method="ot").items():
            previous = best_ot_by_var.get(target_var)
            if previous is None or float(record["exact_acc"]) > float(previous["exact_acc"]):
                best_ot_by_var[target_var] = {**record, "epsilon": epsilon}
    for target_var in DEFAULT_TARGET_VARS:
        record = best_ot_by_var.get(target_var)
        if record is None:
            continue
        lines.append(
            f"OT[{target_var}] exact={float(record['exact_acc']):.4f} "
            f"cal={float(record['selection_score']):.4f} eps={float(record['epsilon']):g} "
            f"site={record['site_label']}"
        )
    lines.append("")
    lines.append("pooled OT block support:")
    for target_var in DEFAULT_TARGET_VARS:
        summary = support_by_var.get(target_var, {})
        lines.append(
            f"support[{target_var}] best_score={float(summary.get('best_selection_score', 0.0)):.4f} "
            f"masks={[candidate.get('name') for candidate in summary.get('mask_candidates', [])]}"
        )
        lines.append(f"support[{target_var}] evidence={summary.get('site_evidence', {})}")
    lines.append("")
    lines.append("full last-token DAS baseline:")
    for target_var, record in _best_record_from_compare_payload(das_full_payload, method="das").items():
        lines.append(
            f"DAS[{target_var}] exact={float(record['exact_acc']):.4f} "
            f"cal={float(record['selection_score']):.4f} site={record['site_label']}"
        )
    lines.append("")
    lines.append("screened block-mask DAS:")
    for target_var in DEFAULT_TARGET_VARS:
        payload = screen_payloads.get(target_var)
        if payload is None:
            continue
        result = payload.get("results", [{}])[0]
        lines.append(
            f"DAS_SCREEN[{target_var}] exact={float(result.get('exact_acc', 0.0)):.4f} "
            f"cal={float(result.get('selection_exact_acc', result.get('calibration_exact_acc', 0.0))):.4f} "
            f"site={result.get('site_label')} dim={result.get('subspace_dim')}"
        )
    lines.append("")
    lines.append("full block-mask DAS rerun:")
    for target_var in DEFAULT_TARGET_VARS:
        payload = block_payloads.get(target_var)
        if payload is None:
            continue
        result = payload.get("results", [{}])[0]
        lines.append(
            f"DAS_BLOCK[{target_var}] exact={float(result.get('exact_acc', 0.0)):.4f} "
            f"cal={float(result.get('selection_exact_acc', result.get('calibration_exact_acc', 0.0))):.4f} "
            f"site={result.get('site_label')} dim={result.get('subspace_dim')}"
        )
    return "\n".join(lines)


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    layers = DEFAULT_LAYERS if args.layers is None else tuple(_parse_csv_ints(args.layers) or [])
    if not layers:
        raise ValueError("No layers selected")
    block_resolutions = list(
        DEFAULT_BLOCK_RESOLUTIONS if args.block_resolutions is None else tuple(_parse_csv_ints(args.block_resolutions) or [])
    )
    if args.include_144 and 144 not in block_resolutions:
        block_resolutions.append(144)
    block_resolutions = sorted(dict.fromkeys(int(resolution) for resolution in block_resolutions), reverse=True)
    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))
    full_das_subspace_dims = tuple(_parse_csv_ints(args.full_das_subspace_dims) or list(DEFAULT_DAS_SUBSPACE_DIMS))
    explicit_guided_subspace_dims = (
        None if args.guided_subspace_dims is None else tuple(_parse_csv_ints(args.guided_subspace_dims) or [])
    )
    signature_modes = [
        str(args.signature_mode),
        *[
            signature_mode
            for signature_mode in (_parse_csv_strings(args.signature_ablation_modes) or [])
            if str(signature_mode) != str(args.signature_mode)
        ],
    ]

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_ot_das_block_focus"
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
    hidden_size = int(model.config.hidden_size)
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    token_position_ids = tuple(token_position.id for token_position in token_positions)

    all_payloads: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []

    for layer in layers:
        layer_start = perf_counter()
        layer_dir = sweep_root / f"layer_{int(layer):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        base_run.TOKEN_POSITION_IDS = list(DEFAULT_TOKEN_POSITION_IDS)

        das_full_output_path = layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_das_full.json"
        das_full_summary_path = layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_das_full.txt"
        das_full_payload = _load_existing_payload(das_full_output_path)
        das_full_start = perf_counter()
        if das_full_payload is None:
            das_full_payload = run_comparison(
                model=model,
                tokenizer=tokenizer,
                token_positions=token_positions,
                banks_by_split=banks_by_split,
                data_metadata=data_metadata,
                device=device,
                config=CompareExperimentConfig(
                    model_name=base_run.MODEL_NAME,
                    output_path=das_full_output_path,
                    summary_path=das_full_summary_path,
                    methods=("das",),
                    target_vars=target_vars,
                    batch_size=int(args.batch_size),
                    signature_mode=str(args.signature_mode),
                    das_subspace_dims=full_das_subspace_dims,
                    das_store_candidate_holdout_metrics=False,
                    das_restarts=max(1, int(args.full_restarts)),
                    resolution=hidden_size,
                    layers=(int(layer),),
                    token_position_ids=DEFAULT_TOKEN_POSITION_IDS,
                    calibration_metric=DEFAULT_CALIBRATION_METRIC,
                    calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                ),
            )
        _synchronize_if_cuda(device)
        das_full_seconds = float(perf_counter() - das_full_start)

        for signature_mode in signature_modes:
            for resolution in block_resolutions:
                resolution_start = perf_counter()
                resolved_resolution = base_run._resolve_resolution(
                    resolution=resolution,
                    hidden_size=hidden_size,
                )
                resolution_tag = base_run._resolution_tag(resolution)
                ot_sites = enumerate_residual_sites(
                    num_layers=int(model.config.num_hidden_layers),
                    hidden_size=hidden_size,
                    token_position_ids=token_position_ids,
                    resolution=resolved_resolution,
                    layers=(int(layer),),
                    selected_token_position_ids=DEFAULT_TOKEN_POSITION_IDS,
                )
                train_banks = {target_var: banks_by_split["train"][target_var] for target_var in target_vars}
                cache_spec = base_run._signature_cache_spec(
                    train_bank=train_banks[target_vars[0]],
                    resolution=resolution,
                    resolved_resolution=resolved_resolution,
                    signature_mode=signature_mode,
                    selected_layers=[int(layer)],
                    token_position_ids=DEFAULT_TOKEN_POSITION_IDS,
                )
                cache_path = base_run._signature_cache_path(
                    resolution=resolution,
                    signature_mode=signature_mode,
                    cache_spec=cache_spec,
                )
                prepared_artifacts = load_prepared_alignment_artifacts(
                    cache_path,
                    expected_spec=cache_spec,
                )
                signature_build_start = perf_counter()
                if prepared_artifacts is None:
                    prepared_artifacts = prepare_alignment_artifacts(
                        model=model,
                        fit_banks_by_var=train_banks,
                        sites=ot_sites,
                        device=device,
                        config=OTConfig(
                            method="ot",
                            batch_size=int(args.batch_size),
                            epsilon=1.0,
                            signature_mode=signature_mode,
                            top_k_values=DEFAULT_OT_TOP_K_VALUES,
                            lambda_values=DEFAULT_OT_LAMBDAS,
                            source_target_vars=target_vars,
                            calibration_metric=DEFAULT_CALIBRATION_METRIC,
                            calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                            top_k_values_by_var={target_var: DEFAULT_OT_TOP_K_VALUES for target_var in target_vars},
                            lambda_values_by_var={target_var: DEFAULT_OT_LAMBDAS for target_var in target_vars},
                        ),
                    )
                    save_prepared_alignment_artifacts(
                        cache_path,
                        prepared_artifacts=prepared_artifacts,
                        cache_spec=cache_spec,
                    )
                _synchronize_if_cuda(device)
                signature_build_seconds = float(perf_counter() - signature_build_start)

                ot_compare_payloads: list[dict[str, object]] = []
                ot_fit_cal_start = perf_counter()
                for epsilon in ot_epsilons:
                    output_stem = (
                        f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_eps-{float(epsilon):g}_ot"
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
                                signature_mode=signature_mode,
                                ot_top_k_values=DEFAULT_OT_TOP_K_VALUES,
                                ot_lambdas=DEFAULT_OT_LAMBDAS,
                                calibration_metric=DEFAULT_CALIBRATION_METRIC,
                                calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                                ot_top_k_values_by_var={target_var: DEFAULT_OT_TOP_K_VALUES for target_var in target_vars},
                                ot_lambdas_by_var={target_var: DEFAULT_OT_LAMBDAS for target_var in target_vars},
                                resolution=resolved_resolution,
                                layers=(int(layer),),
                                token_position_ids=DEFAULT_TOKEN_POSITION_IDS,
                            ),
                            prepared_ot_artifacts=prepared_artifacts,
                        )
                    ot_compare_payloads.append(compare_payload)
                _synchronize_if_cuda(device)
                ot_fit_cal_seconds = float(perf_counter() - ot_fit_cal_start)

                support_start = perf_counter()
                support_by_var = extract_block_mask_support(
                    ot_run_payloads=ot_compare_payloads,
                    sites=ot_sites,
                    score_slack=float(args.support_score_slack),
                )
                support_path = layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_ot_block_support.json"
                write_json(support_path, support_by_var)
                support_extract_seconds = float(perf_counter() - support_start)

                screen_payloads: dict[str, dict[str, object]] = {}
                block_payloads: dict[str, dict[str, object]] = {}
                das_screen_seconds = 0.0
                das_block_seconds = 0.0
                for target_var in target_vars:
                    support_summary = support_by_var.get(str(target_var))
                    if support_summary is None:
                        continue
                    mask_sites = build_mask_sites_from_support(
                        support_summary=support_summary,
                        sites=ot_sites,
                    )
                    if not mask_sites:
                        continue
                    mask_max_width = max(
                        int(site_total_width(site, model_hidden_size=int(hidden_size)))
                        for site in mask_sites
                    )
                    if explicit_guided_subspace_dims is None:
                        mask_subspace_dims = _guided_subspace_dims(mask_max_width)
                    else:
                        mask_subspace_dims = tuple(
                            int(dim)
                            for dim in explicit_guided_subspace_dims
                            if 1 <= int(dim) <= int(mask_max_width)
                        )
                    if not mask_subspace_dims:
                        mask_subspace_dims = _guided_subspace_dims(mask_max_width)
                    screen_output_path = layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_{target_var}_das_block_screen.json"
                    )
                    screen_summary_path = layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_{target_var}_das_block_screen.txt"
                    )
                    screen_payload = _load_existing_payload(screen_output_path)
                    screen_start = perf_counter()
                    if screen_payload is None:
                        screen_payload = run_das_pipeline(
                            model=model,
                            train_bank=banks_by_split["train"][target_var],
                            calibration_bank=banks_by_split["calibration"][target_var],
                            holdout_bank=banks_by_split["test"][target_var],
                            sites=mask_sites,
                            device=device,
                            tokenizer=tokenizer,
                            config=DASConfig(
                                method_name="das_block_screen",
                                batch_size=int(args.batch_size),
                                max_epochs=int(args.screen_max_epochs),
                                min_epochs=int(args.screen_min_epochs),
                                plateau_patience=1,
                                plateau_rel_delta=base_run.DAS_PLATEAU_REL_DELTA,
                                learning_rate=base_run.DAS_LEARNING_RATE,
                                subspace_dims=mask_subspace_dims,
                                store_candidate_holdout_metrics=False,
                                restarts=max(1, int(args.screen_restarts)),
                                verbose=True,
                            ),
                        )
                        screen_payload["support_summary"] = support_summary
                        write_json(screen_output_path, screen_payload)
                        _write_das_text_report(
                            screen_summary_path,
                            title="MCQA Block DAS Screen Summary",
                            payload=screen_payload,
                            extra_lines=[
                                f"target_var: {target_var}",
                                f"mask_candidates: {[candidate.get('name') for candidate in support_summary.get('mask_candidates', [])]}",
                                f"subspace_dims: {list(int(dim) for dim in mask_subspace_dims)}",
                                f"restarts: {max(1, int(args.screen_restarts))}",
                            ],
                        )
                    _synchronize_if_cuda(device)
                    das_screen_seconds += float(perf_counter() - screen_start)
                    screen_payloads[str(target_var)] = screen_payload

                    selected_site_labels = _best_screen_sites(
                        screen_payload,
                        target_var=str(target_var),
                        limit=max(1, int(args.full_mask_limit)),
                    )
                    selected_sites = [site for site in mask_sites if site.label in set(selected_site_labels)]
                    if not selected_sites:
                        selected_sites = [mask_sites[0]]
                    selected_max_width = max(
                        int(site_total_width(site, model_hidden_size=int(hidden_size)))
                        for site in selected_sites
                    )
                    if explicit_guided_subspace_dims is None:
                        selected_subspace_dims = _guided_subspace_dims(selected_max_width)
                    else:
                        selected_subspace_dims = tuple(
                            int(dim)
                            for dim in explicit_guided_subspace_dims
                            if 1 <= int(dim) <= int(selected_max_width)
                        )
                    if not selected_subspace_dims:
                        selected_subspace_dims = _guided_subspace_dims(selected_max_width)

                    block_output_path = layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_{target_var}_das_block.json"
                    )
                    block_summary_path = layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_{target_var}_das_block.txt"
                    )
                    block_payload = _load_existing_payload(block_output_path)
                    block_start = perf_counter()
                    if block_payload is None:
                        block_payload = run_das_pipeline(
                            model=model,
                            train_bank=banks_by_split["train"][target_var],
                            calibration_bank=banks_by_split["calibration"][target_var],
                            holdout_bank=banks_by_split["test"][target_var],
                            sites=selected_sites,
                            device=device,
                            tokenizer=tokenizer,
                            config=DASConfig(
                                method_name="das_block",
                                batch_size=int(args.batch_size),
                                max_epochs=base_run.DAS_MAX_EPOCHS,
                                min_epochs=base_run.DAS_MIN_EPOCHS,
                                plateau_patience=base_run.DAS_PLATEAU_PATIENCE,
                                plateau_rel_delta=base_run.DAS_PLATEAU_REL_DELTA,
                                learning_rate=base_run.DAS_LEARNING_RATE,
                                subspace_dims=selected_subspace_dims,
                                store_candidate_holdout_metrics=False,
                                restarts=max(1, int(args.full_restarts)),
                                verbose=True,
                            ),
                        )
                        block_payload["support_summary"] = support_summary
                        block_payload["selected_screen_site_labels"] = selected_site_labels
                        write_json(block_output_path, block_payload)
                        _write_das_text_report(
                            block_summary_path,
                            title="MCQA Block DAS Summary",
                            payload=block_payload,
                            extra_lines=[
                                f"target_var: {target_var}",
                                f"selected_screen_site_labels: {selected_site_labels}",
                                f"subspace_dims: {list(int(dim) for dim in selected_subspace_dims)}",
                                f"restarts: {max(1, int(args.full_restarts))}",
                            ],
                        )
                    _synchronize_if_cuda(device)
                    das_block_seconds += float(perf_counter() - block_start)
                    block_payloads[str(target_var)] = block_payload
                resolution_total_seconds = float(perf_counter() - resolution_start)

                resolution_summary_path = layer_dir / (
                    f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_ot_das_block_summary.txt"
                )
                resolution_summary_text = _format_resolution_summary(
                    layer=int(layer),
                    resolution=int(resolution),
                    signature_mode=str(signature_mode),
                    ot_payloads=ot_compare_payloads,
                    support_by_var=support_by_var,
                    das_full_payload=das_full_payload,
                    screen_payloads=screen_payloads,
                    block_payloads=block_payloads,
                )
                write_text_report(resolution_summary_path, resolution_summary_text)

                resolution_payload = {
                    "payload_path": str(
                        layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_ot_das_block.json"
                    ),
                    "layer": int(layer),
                    "token_position_ids": list(DEFAULT_TOKEN_POSITION_IDS),
                    "resolution": int(resolution),
                    "signature_mode": str(signature_mode),
                    "ot_epsilons": [float(epsilon) for epsilon in ot_epsilons],
                    "support_score_slack": float(args.support_score_slack),
                    "full_das_subspace_dims": [int(dim) for dim in full_das_subspace_dims],
                    "guided_subspace_dims": None
                    if explicit_guided_subspace_dims is None
                    else [int(dim) for dim in explicit_guided_subspace_dims],
                    "support_by_var": support_by_var,
                    "context_timing_seconds": context_timing_seconds,
                    "timing_seconds": {
                        "t_model_load": float(context_timing_seconds.get("t_model_load", 0.0)),
                        "t_data_load": float(context_timing_seconds.get("t_data_load", 0.0)),
                        "t_bank_build": float(context_timing_seconds.get("t_bank_build", 0.0)),
                        "t_factual_filter": float(context_timing_seconds.get("t_factual_filter", 0.0)),
                        "t_context_total_wall": float(context_timing_seconds.get("t_context_total_wall", 0.0)),
                        "t_stageC_a_only_das": float(das_full_seconds),
                        "t_stageB_native_signature_build": float(signature_build_seconds),
                        "t_stageB_native_ot_fit_cal": float(ot_fit_cal_seconds),
                        "t_support_extract": float(support_extract_seconds),
                        "t_stageC_das_screen": float(das_screen_seconds),
                        "t_stageC_das_full": float(das_block_seconds),
                        "t_resolution_total_wall": float(resolution_total_seconds),
                        "t_layer_total_wall_so_far": float(perf_counter() - layer_start),
                        "t_stage_total_wall_so_far": float(perf_counter() - stage_start),
                    },
                    "ot_output_paths": [
                        str(
                            layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_eps-{float(epsilon):g}_ot.json"
                        )
                        for epsilon in ot_epsilons
                    ],
                    "das_full_output_path": str(das_full_output_path),
                    "screen_output_paths": {
                        str(target_var): str(
                            layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_{target_var}_das_block_screen.json"
                        )
                        for target_var in target_vars
                    },
                    "block_output_paths": {
                        str(target_var): str(
                            layer_dir / f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_{target_var}_das_block.json"
                        )
                        for target_var in target_vars
                    },
                    "summary_path": str(resolution_summary_path),
                }
                resolution_payload_path = layer_dir / (
                    f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}_sig-{signature_mode}_ot_das_block.json"
                )
                write_json(resolution_payload_path, resolution_payload)
                all_payloads.append(resolution_payload)
                manifest_runs.append(
                    {
                        "layer": int(layer),
                        "resolution": int(resolution),
                        "signature_mode": str(signature_mode),
                        "runtime_seconds": float(resolution_total_seconds),
                        "timing_seconds": resolution_payload["timing_seconds"],
                        "summary_path": str(resolution_summary_path),
                        "payload_path": str(resolution_payload_path),
                    }
                )

    manifest_path = sweep_root / "layer_sweep_manifest.json"
    existing_manifest_runs = _load_existing_runs(manifest_path)
    current_payload_paths = {str(run["payload_path"]) for run in manifest_runs}
    write_json(
        manifest_path,
        {
            "kind": "mcqa_ot_das_block_focus",
            "layers": [int(layer) for layer in layers],
            "resolutions": [int(resolution) for resolution in block_resolutions],
            "signature_modes": [str(signature_mode) for signature_mode in signature_modes],
            "full_das_subspace_dims": [int(dim) for dim in full_das_subspace_dims],
            "guided_subspace_dims": None
            if explicit_guided_subspace_dims is None
            else [int(dim) for dim in explicit_guided_subspace_dims],
            "context_timing_seconds": context_timing_seconds,
            "runtime_seconds": float(perf_counter() - stage_start),
            "runs": [
                *[run for run in existing_manifest_runs if str(run.get("payload_path", "")) not in current_payload_paths],
                *manifest_runs,
            ],
        },
    )
    aggregate_path = sweep_root / "mcqa_run_results.json"
    existing_payload_runs = _load_existing_runs(aggregate_path)
    write_json(
        aggregate_path,
        {
            "runs": [
                *[
                    run
                    for run in existing_payload_runs
                    if str(run.get("payload_path", "")) not in current_payload_paths
                ],
                *all_payloads,
            ]
        },
    )
    print(f"Wrote OT + DAS block manifest to {manifest_path}")


if __name__ == "__main__":
    main()
