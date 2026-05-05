from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mcqa_run as base_run


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


def _parse_layer_blocks(value: str | None) -> list[list[int]] | None:
    if value is None:
        return None
    blocks: list[list[int]] = []
    for raw_block in value.split(";"):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        layers: list[int] = []
        for item in raw_block.replace("+", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                layers.extend(range(start, end + step, step))
            else:
                layers.append(int(item))
        resolved = sorted(dict.fromkeys(layers))
        if not resolved:
            raise ValueError(f"Layer block {raw_block!r} did not contain any layers")
        blocks.append(resolved)
    if not blocks:
        raise ValueError("--layer-blocks did not contain any non-empty blocks")
    return blocks


def _parse_csv_resolutions(value: str | None) -> list[int | None] | None:
    items = _parse_csv_strings(value)
    if items is None:
        return None
    parsed: list[int | None] = []
    for item in items:
        parsed.append(None if item.lower() == "full" else int(item))
    return parsed


def _parse_csv_floats(value: str | None) -> list[float] | None:
    items = _parse_csv_strings(value)
    if items is None:
        return None
    return [float(item) for item in items]


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "full":
        if args.methods is None:
            args.methods = "ot"
        if args.target_vars is None:
            args.target_vars = "answer_pointer,answer_token"
        if args.token_position_ids is None:
            args.token_position_ids = "correct_symbol,correct_symbol_period,last_token"
        if args.signature_modes is None:
            args.signature_modes = "family_slot_label_delta"
        if args.resolutions is None:
            args.resolutions = "full"
        if args.ot_epsilons is None:
            args.ot_epsilons = "0.5,1,2"
        if args.ot_top_k_values is None:
            args.ot_top_k_values = "1,2,3"
        if args.ot_lambdas is None:
            args.ot_lambdas = "0.5,1,2"
    elif args.preset == "smoke":
        if args.dataset_size is None:
            args.dataset_size = 512
        if args.train_pool_size is None:
            args.train_pool_size = 64
        if args.calibration_pool_size is None:
            args.calibration_pool_size = 32
        if args.test_pool_size is None:
            args.test_pool_size = 32
        if args.methods is None:
            args.methods = "ot"
        if args.target_vars is None:
            args.target_vars = "answer_pointer,answer_token"
        if args.token_position_ids is None:
            args.token_position_ids = "correct_symbol,correct_symbol_period,last_token"
        if args.signature_modes is None:
            args.signature_modes = "family_slot_label_delta"
        if args.resolutions is None:
            args.resolutions = "full"
        if args.ot_epsilons is None:
            args.ot_epsilons = "0.5,1,2"
        if args.ot_top_k_values is None:
            args.ot_top_k_values = "1,2,3"
        if args.ot_lambdas is None:
            args.ot_lambdas = "0.5,1,2"
    elif args.preset == "next":
        if args.dataset_size is None:
            args.dataset_size = 2000
        if args.train_pool_size is None:
            args.train_pool_size = 200
        if args.calibration_pool_size is None:
            args.calibration_pool_size = 100
        if args.test_pool_size is None:
            args.test_pool_size = 100
        if args.methods is None:
            args.methods = "ot"
        if args.target_vars is None:
            args.target_vars = "answer_pointer,answer_token"
        if args.token_position_ids is None:
            args.token_position_ids = "correct_symbol,correct_symbol_period,last_token"
        if args.signature_modes is None:
            args.signature_modes = "family_slot_label_delta_norm"
        if args.resolutions is None:
            args.resolutions = "full"
        if args.ot_epsilons is None:
            args.ot_epsilons = "0.25,0.5,1,2,4"
        if args.ot_top_k_values is None:
            args.ot_top_k_values = "1,2,3"
        if args.ot_lambdas is None:
            args.ot_lambdas = "0.25,0.5,1,2,4,8"
        if args.calibration_metric is None:
            args.calibration_metric = "family_weighted_macro_exact_acc"
        if args.calibration_family_weights is None:
            args.calibration_family_weights = "1,1.5,2"
    elif args.preset == "next_bf":
        if args.dataset_size is None:
            args.dataset_size = 2000
        if args.train_pool_size is None:
            args.train_pool_size = 200
        if args.calibration_pool_size is None:
            args.calibration_pool_size = 100
        if args.test_pool_size is None:
            args.test_pool_size = 100
        if args.methods is None:
            args.methods = "ot,bruteforce"
        if args.target_vars is None:
            args.target_vars = "answer_pointer,answer_token"
        if args.token_position_ids is None:
            args.token_position_ids = "correct_symbol,correct_symbol_period,last_token"
        if args.signature_modes is None:
            args.signature_modes = "family_slot_label_delta_norm"
        if args.resolutions is None:
            args.resolutions = "full"
        if args.ot_epsilons is None:
            args.ot_epsilons = "0.25,0.5,1,2,4"
        if args.ot_top_k_values is None:
            args.ot_top_k_values = "1,2,3"
        if args.ot_lambdas is None:
            args.ot_lambdas = "0.25,0.5,1,2,4,8"
        if args.calibration_metric is None:
            args.calibration_metric = "family_weighted_macro_exact_acc"
        if args.calibration_family_weights is None:
            args.calibration_family_weights = "1,1.5,2"
    else:
        raise ValueError(f"Unsupported preset {args.preset}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud-friendly launcher for the MCQA experiment.")
    parser.add_argument("--preset", choices=("full", "smoke", "next", "next_bf"), default="full")
    parser.add_argument("--device")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--train-pool-size", type=int)
    parser.add_argument("--calibration-pool-size", type=int)
    parser.add_argument("--test-pool-size", type=int)
    parser.add_argument("--methods", help="Comma-separated, e.g. ot,uot,das")
    parser.add_argument("--target-vars", help="Comma-separated, e.g. answer_pointer,answer_token")
    parser.add_argument(
        "--ot-source-target-vars",
        help="Comma-separated OT source rows, optionally including null/background.",
    )
    parser.add_argument(
        "--counterfactual-names",
        help="Comma-separated, e.g. answerPosition,randomLetter,answerPosition_randomLetter",
    )
    parser.add_argument("--layers", help="'auto' or comma-separated layer indices")
    parser.add_argument(
        "--layer-blocks",
        help="Semicolon-separated coarse layer blocks, e.g. '0-4;5-9;10-14;15-19;20-25'.",
    )
    parser.add_argument("--token-position-ids", help="'all' or comma-separated token position ids")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resolutions", help="Comma-separated integers or 'full'")
    parser.add_argument("--ot-epsilons", help="Comma-separated floats")
    parser.add_argument(
        "--uot-beta-neural",
        help="Comma-separated floats for the neural-side marginal KL penalty in one-sided UOT.",
    )
    parser.add_argument("--signature-modes", help="Comma-separated strings")
    parser.add_argument("--ot-top-k-values", help="Comma-separated integers")
    parser.add_argument("--ot-lambdas", help="Comma-separated floats")
    parser.add_argument("--calibration-metric", help="Calibration objective for OT/UOT selection")
    parser.add_argument("--calibration-family-weights", help="Comma-separated family weights in COUNTERFACTUAL_FAMILIES order")
    parser.add_argument("--das-max-epochs", type=int)
    parser.add_argument("--das-min-epochs", type=int)
    parser.add_argument("--das-plateau-patience", type=int)
    parser.add_argument("--das-plateau-rel-delta", type=float)
    parser.add_argument("--das-learning-rate", type=float)
    parser.add_argument("--das-restarts", type=int)
    parser.add_argument("--das-subspace-dims", help="Comma-separated integers")
    parser.add_argument("--results-root")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir")
    parser.add_argument("--layer-sweep", action="store_true", help="Run one job per layer with sites restricted to that single layer.")
    parser.add_argument(
        "--layer-indices",
        help="Comma-separated layer indices to run when using --layer-sweep. If omitted, evenly spaced layers are chosen.",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        help="Number of evenly spaced layers to select when using --layer-sweep without --layer-indices. Default: 6.",
    )
    parser.add_argument(
        "--prompt-hf-login",
        action="store_true",
        help="Allow interactive Hugging Face login if no token is present.",
    )
    return parser


def _override_base_run(
    args: argparse.Namespace,
    *,
    run_dir_override: Path | None = None,
    results_timestamp_override: str | None = None,
) -> dict[str, object]:
    results_root = Path(args.results_root) if args.results_root else Path("results")
    results_timestamp = results_timestamp_override or args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    run_dir = run_dir_override or (results_root / f"{results_timestamp}_mcqa")
    signatures_dir = Path(args.signatures_dir) if args.signatures_dir else Path("signatures")

    if args.device:
        base_run.DEVICE = args.device
    if args.model_name:
        base_run.MODEL_NAME = args.model_name
    if args.dataset_path:
        base_run.MCQA_DATASET_PATH = args.dataset_path
    if args.dataset_config is not None:
        base_run.MCQA_DATASET_CONFIG = args.dataset_config or None
    if args.dataset_size is not None:
        base_run.DATASET_SIZE = int(args.dataset_size)
    if args.split_seed is not None:
        base_run.SPLIT_SEED = int(args.split_seed)
    if args.train_pool_size is not None:
        base_run.TRAIN_POOL_SIZE = int(args.train_pool_size)
    if args.calibration_pool_size is not None:
        base_run.CALIBRATION_POOL_SIZE = int(args.calibration_pool_size)
    if args.test_pool_size is not None:
        base_run.TEST_POOL_SIZE = int(args.test_pool_size)

    methods = _parse_csv_strings(args.methods)
    if methods is not None:
        base_run.METHODS = methods
    target_vars = _parse_csv_strings(args.target_vars)
    if target_vars is not None:
        base_run.TARGET_VARS = target_vars
    ot_source_target_vars = _parse_csv_strings(args.ot_source_target_vars)
    if ot_source_target_vars is not None:
        base_run.OT_SOURCE_TARGET_VARS = ot_source_target_vars
    counterfactual_names = _parse_csv_strings(args.counterfactual_names)
    if counterfactual_names is not None:
        base_run.COUNTERFACTUAL_NAMES = counterfactual_names

    if args.layers is not None:
        base_run.LAYERS = "auto" if args.layers == "auto" else _parse_csv_ints(args.layers)
    if args.layer_blocks is not None:
        base_run.LAYER_BLOCKS = _parse_layer_blocks(args.layer_blocks)
    if args.token_position_ids is not None:
        base_run.TOKEN_POSITION_IDS = None if args.token_position_ids == "all" else _parse_csv_strings(args.token_position_ids)
    if args.batch_size is not None:
        base_run.BATCH_SIZE = int(args.batch_size)

    resolutions = _parse_csv_resolutions(args.resolutions)
    if resolutions is not None:
        base_run.RESOLUTIONS = resolutions
    ot_epsilons = _parse_csv_floats(args.ot_epsilons)
    if ot_epsilons is not None:
        base_run.OT_EPSILONS = ot_epsilons
    uot_beta_neurals = _parse_csv_floats(args.uot_beta_neural)
    if uot_beta_neurals is not None:
        base_run.UOT_BETA_NEURALS = uot_beta_neurals
    signature_modes = _parse_csv_strings(args.signature_modes)
    if signature_modes is not None:
        base_run.SIGNATURE_MODES = signature_modes
    ot_top_k_values = _parse_csv_ints(args.ot_top_k_values)
    if ot_top_k_values is not None:
        base_run.OT_TOP_K_VALUES = ot_top_k_values
    ot_lambdas = _parse_csv_floats(args.ot_lambdas)
    if ot_lambdas is not None:
        base_run.OT_LAMBDAS = ot_lambdas
    if args.calibration_metric is not None:
        base_run.CALIBRATION_METRIC = str(args.calibration_metric)
    calibration_family_weights = _parse_csv_floats(args.calibration_family_weights)
    if calibration_family_weights is not None:
        base_run.CALIBRATION_FAMILY_WEIGHTS = calibration_family_weights

    if args.das_max_epochs is not None:
        base_run.DAS_MAX_EPOCHS = int(args.das_max_epochs)
    if args.das_min_epochs is not None:
        base_run.DAS_MIN_EPOCHS = int(args.das_min_epochs)
    if args.das_plateau_patience is not None:
        base_run.DAS_PLATEAU_PATIENCE = int(args.das_plateau_patience)
    if args.das_plateau_rel_delta is not None:
        base_run.DAS_PLATEAU_REL_DELTA = float(args.das_plateau_rel_delta)
    if args.das_learning_rate is not None:
        base_run.DAS_LEARNING_RATE = float(args.das_learning_rate)
    if args.das_restarts is not None:
        base_run.DAS_RESTARTS = int(args.das_restarts)
    das_subspace_dims = _parse_csv_ints(args.das_subspace_dims)
    if das_subspace_dims is not None:
        base_run.DAS_SUBSPACE_DIMS = das_subspace_dims

    base_run.PROMPT_HF_LOGIN = bool(args.prompt_hf_login)
    base_run.RUN_TIMESTAMP = results_timestamp
    base_run.RUN_DIR = run_dir
    base_run.OUTPUT_PATH = run_dir / "mcqa_run_results.json"
    base_run.SUMMARY_PATH = run_dir / "mcqa_run_summary.txt"
    base_run.SIGNATURES_DIR = signatures_dir

    return {
        "device": base_run.DEVICE,
        "model_name": base_run.MODEL_NAME,
        "dataset_path": base_run.MCQA_DATASET_PATH,
        "dataset_config": base_run.MCQA_DATASET_CONFIG,
        "dataset_size": base_run.DATASET_SIZE,
        "split_seed": base_run.SPLIT_SEED,
        "train_pool_size": base_run.TRAIN_POOL_SIZE,
        "calibration_pool_size": base_run.CALIBRATION_POOL_SIZE,
        "test_pool_size": base_run.TEST_POOL_SIZE,
        "methods": list(base_run.METHODS),
        "target_vars": list(base_run.TARGET_VARS),
        "ot_source_target_vars": None if base_run.OT_SOURCE_TARGET_VARS is None else list(base_run.OT_SOURCE_TARGET_VARS),
        "counterfactual_names": list(base_run.COUNTERFACTUAL_NAMES),
        "layers": base_run.LAYERS,
        "layer_blocks": base_run.LAYER_BLOCKS,
        "token_position_ids": base_run.TOKEN_POSITION_IDS,
        "batch_size": base_run.BATCH_SIZE,
        "resolutions": list(base_run.RESOLUTIONS),
        "ot_epsilons": list(base_run.OT_EPSILONS),
        "uot_beta_neurals": list(base_run.UOT_BETA_NEURALS),
        "signature_modes": list(base_run.SIGNATURE_MODES),
        "ot_top_k_values": list(base_run.OT_TOP_K_VALUES),
        "ot_lambdas": list(base_run.OT_LAMBDAS),
        "calibration_metric": base_run.CALIBRATION_METRIC,
        "calibration_family_weights": list(base_run.CALIBRATION_FAMILY_WEIGHTS),
        "das_max_epochs": base_run.DAS_MAX_EPOCHS,
        "das_min_epochs": base_run.DAS_MIN_EPOCHS,
        "das_plateau_patience": base_run.DAS_PLATEAU_PATIENCE,
        "das_plateau_rel_delta": base_run.DAS_PLATEAU_REL_DELTA,
        "das_learning_rate": base_run.DAS_LEARNING_RATE,
        "das_restarts": base_run.DAS_RESTARTS,
        "das_subspace_dims": list(base_run.DAS_SUBSPACE_DIMS),
        "prompt_hf_login": base_run.PROMPT_HF_LOGIN,
        "run_dir": str(base_run.RUN_DIR),
        "signatures_dir": str(base_run.SIGNATURES_DIR),
    }


def _resolve_num_layers(model_name: str) -> int:
    from transformers import AutoConfig

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Could not resolve num_hidden_layers for model {model_name}")
    return int(config.num_hidden_layers)


def _evenly_spaced_indices(num_layers: int, count: int) -> list[int]:
    resolved_count = max(1, min(int(count), int(num_layers)))
    if resolved_count == 1:
        return [int(num_layers) - 1]
    indices = {
        int(round(position * (int(num_layers) - 1) / (resolved_count - 1)))
        for position in range(resolved_count)
    }
    return sorted(indices)


def _resolve_layer_indices(args: argparse.Namespace, model_name: str) -> list[int]:
    explicit = _parse_csv_ints(args.layer_indices)
    if explicit is not None:
        return sorted(dict.fromkeys(int(layer) for layer in explicit))
    num_layers = _resolve_num_layers(model_name)
    layer_count = 6 if args.layer_count is None else int(args.layer_count)
    return _evenly_spaced_indices(num_layers, layer_count)


def _load_existing_manifest(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, dict)]


def _write_manifest(manifest_path: Path, *, model_name: str, layer_indices: list[int], runs: list[dict[str, object]]) -> None:
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "layer_indices": layer_indices,
                "runs": runs,
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _apply_preset(args)
    if args.layer_sweep and args.layers is not None:
        raise ValueError("Use --layer-indices with --layer-sweep; do not combine --layer-sweep with --layers.")
    if args.layer_sweep and args.layer_blocks is not None:
        raise ValueError("Do not combine --layer-sweep with --layer-blocks.")

    if not args.layer_sweep:
        effective_config = _override_base_run(args)
        print("[cloud-run] effective configuration:")
        print(json.dumps(effective_config, indent=2, sort_keys=True))
        base_run.main()
        return

    model_name = args.model_name or base_run.MODEL_NAME
    layer_indices = _resolve_layer_indices(args, model_name)
    results_root = Path(args.results_root) if args.results_root else Path("results")
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_layer_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_root / "layer_sweep_manifest.json"
    manifest = _load_existing_manifest(manifest_path)
    _write_manifest(manifest_path, model_name=model_name, layer_indices=layer_indices, runs=manifest)
    _override_base_run(
        args,
        run_dir_override=sweep_root / "_bootstrap",
        results_timestamp_override=results_timestamp,
    )
    shared_context = base_run.build_run_context()
    for layer in layer_indices:
        run_dir = sweep_root / f"layer_{int(layer):02d}"
        effective_config = _override_base_run(
            args,
            run_dir_override=run_dir,
            results_timestamp_override=results_timestamp,
        )
        base_run.LAYERS = [int(layer)]
        effective_config["layer_sweep"] = True
        effective_config["layers"] = [int(layer)]
        print("[cloud-run] effective configuration:")
        print(json.dumps(effective_config, indent=2, sort_keys=True))
        base_run.execute_run_context(context=shared_context)
        current_record = {
            "layer": int(layer),
            "run_dir": str(run_dir),
            "output_path": str(base_run.OUTPUT_PATH),
            "summary_path": str(base_run.SUMMARY_PATH),
        }
        manifest = [record for record in manifest if int(record.get("layer", -1)) != int(layer)]
        manifest.append(current_record)
        manifest.sort(key=lambda record: int(record.get("layer", -1)))
        _write_manifest(manifest_path, model_name=model_name, layer_indices=layer_indices, runs=manifest)
    print(f"[cloud-run] wrote layer sweep manifest to {manifest_path}")


if __name__ == "__main__":
    main()
