from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import mcqa_run as base_run
from mcqa_experiment.das import DASConfig, run_das_pipeline
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import enumerate_residual_sites, site_total_width
from mcqa_experiment.support import build_ordered_composite_sites_from_support


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_GUIDED_MASK_NAMES = ("Top1", "Top2", "Top4", "S50", "S80")
DEFAULT_GUIDED_SUBSPACE_DIMS = (32, 64, 96, 128, 256, 512, 768, 1024, 1536, 2048, 2304)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone PLOT-DAS (native support) runner for MCQA.")
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
    parser.add_argument("--native-support-path", required=True)
    parser.add_argument("--guided-mask-names", help="Comma-separated support masks. Default: Top1,Top2,Top4,S50,S80")
    parser.add_argument(
        "--guided-subspace-dims",
        default=None,
        help="Optional comma-separated DAS dims override. By default native support uses the standard DAS grid clipped to the support width.",
    )
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--guided-restarts", type=int, default=2)
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


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload at {path}")
    return payload


def _filter_support_summary(support_summary: dict[str, object], mask_names: tuple[str, ...]) -> dict[str, object]:
    allowed = set(str(mask_name) for mask_name in mask_names)
    filtered_candidates = [
        candidate
        for candidate in support_summary.get("mask_candidates", [])
        if str(candidate.get("name")) in allowed
    ]
    if not filtered_candidates and support_summary.get("mask_candidates"):
        filtered_candidates = [support_summary["mask_candidates"][0]]
    return {
        **support_summary,
        "mask_candidates": filtered_candidates,
    }


def _native_support_subspace_dims(
    *,
    support_total_dim: int,
    explicit_dims: tuple[int, ...] | None,
) -> tuple[int, ...]:
    capped_support_total_dim = max(1, int(support_total_dim))
    candidate_dims = DEFAULT_GUIDED_SUBSPACE_DIMS if explicit_dims is None else tuple(int(dim) for dim in explicit_dims)
    filtered = tuple(int(dim) for dim in candidate_dims if 0 < int(dim) <= capped_support_total_dim)
    return filtered or (int(capped_support_total_dim),)


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    native_support_path = Path(args.native_support_path)
    native_payload = _load_json(native_support_path)
    layer = int(native_payload["layer"])
    token_position_id = str(native_payload["token_position_id"])
    native_resolution = int(native_payload.get("native_resolution", native_payload["atomic_width"]))

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_plot_das_native_support"
    sweep_root.mkdir(parents=True, exist_ok=True)

    _configure_base_run(args, sweep_root=sweep_root, results_timestamp=results_timestamp)
    context = base_run.build_run_context()
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_positions = context["token_positions"]
    banks_by_split = context["banks_by_split"]
    device = context["device"]
    hidden_size = int(model.config.hidden_size)
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    atomic_sites = enumerate_residual_sites(
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=hidden_size,
        token_position_ids=token_position_ids,
        resolution=int(native_resolution),
        layers=(int(layer),),
        selected_token_position_ids=(str(token_position_id),),
    )
    mask_names = tuple(_parse_csv_strings(args.guided_mask_names) or list(DEFAULT_GUIDED_MASK_NAMES))
    explicit_dims = _parse_csv_ints(args.guided_subspace_dims)

    layer_dir = sweep_root / f"layer_{int(layer):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    payloads_by_var: dict[str, dict[str, object]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        support_summary = native_payload.get("support_by_var", {}).get(str(target_var))
        if not isinstance(support_summary, dict):
            continue
        filtered_summary = _filter_support_summary(support_summary, mask_names)
        support_sites = build_ordered_composite_sites_from_support(
            support_summary=filtered_summary,
            sites=atomic_sites,
        )
        if not support_sites:
            continue
        subspace_dims_by_site_label = {
            site.label: _native_support_subspace_dims(
                support_total_dim=int(site_total_width(site, model_hidden_size=hidden_size)),
                explicit_dims=None if explicit_dims is None else tuple(explicit_dims),
            )
            for site in support_sites
        }
        unique_subspace_dims = tuple(
            sorted(
                {
                    int(dim)
                    for site_dims in subspace_dims_by_site_label.values()
                    for dim in site_dims
                }
            )
        )
        output_path = layer_dir / f"mcqa_plot_das_native_support_layer-{int(layer)}_{str(target_var)}.json"
        summary_path = layer_dir / f"mcqa_plot_das_native_support_layer-{int(layer)}_{str(target_var)}.txt"
        payload = _load_json(output_path) if output_path.exists() else None
        if payload is None:
            payload = run_das_pipeline(
                model=model,
                train_bank=banks_by_split["train"][str(target_var)],
                calibration_bank=banks_by_split["calibration"][str(target_var)],
                holdout_bank=banks_by_split["test"][str(target_var)],
                sites=support_sites,
                device=device,
                tokenizer=tokenizer,
                config=DASConfig(
                    method_name="das_native_support",
                    batch_size=int(args.batch_size),
                    max_epochs=int(args.guided_max_epochs),
                    min_epochs=int(args.guided_min_epochs),
                    plateau_patience=int(base_run.DAS_PLATEAU_PATIENCE),
                    plateau_rel_delta=float(base_run.DAS_PLATEAU_REL_DELTA),
                    learning_rate=float(base_run.DAS_LEARNING_RATE),
                    subspace_dims=unique_subspace_dims,
                    subspace_dims_by_site_label=subspace_dims_by_site_label,
                    store_candidate_holdout_metrics=False,
                    restarts=max(1, int(args.guided_restarts)),
                    verbose=True,
                ),
            )
            payload["support_summary"] = filtered_summary
            payload["mask_names"] = [str(mask_name) for mask_name in mask_names]
            payload["guided_subspace_dims_by_site_label"] = {
                str(label): [int(dim) for dim in dims]
                for label, dims in subspace_dims_by_site_label.items()
            }
            write_json(output_path, payload)
            write_text_report(
                summary_path,
                "\n".join(
                    [
                        "MCQA PLOT-DAS (native support) Summary",
                        f"target_var: {target_var}",
                        f"layer: {int(layer)}",
                        f"mask_names: {list(mask_names)}",
                        f"unique_subspace_dims: {list(int(dim) for dim in unique_subspace_dims)}",
                        f"subspace_dims_by_site_label: {payload['guided_subspace_dims_by_site_label']}",
                    ]
                ),
            )
        payloads_by_var[str(target_var)] = payload

    summary_payload = {
        "kind": "mcqa_plot_das_native_support",
        "layer": int(layer),
        "token_position_id": str(token_position_id),
        "native_resolution": int(native_resolution),
        "native_support_path": str(native_support_path),
        "guided_mask_names": [str(mask_name) for mask_name in mask_names],
        "payloads_by_var": payloads_by_var,
        "runtime_seconds": float(perf_counter() - stage_start),
    }
    payload_path = layer_dir / f"mcqa_plot_das_native_support_layer-{int(layer)}_summary.json"
    write_json(payload_path, summary_payload)
    print(f"Wrote native support DAS payload to {payload_path}")


if __name__ == "__main__":
    main()
