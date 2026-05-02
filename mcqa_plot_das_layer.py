from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import mcqa_run as base_run
from mcqa_experiment.compare_runner import CompareExperimentConfig, run_comparison
from mcqa_experiment.data import canonicalize_target_var
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
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
    parser = argparse.ArgumentParser(description="Standalone PLOT-DAS (layer) runner for MCQA.")
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
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--token-position-id", default=DEFAULT_TOKEN_POSITION_ID)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--das-subspace-dims", help="Comma-separated DAS dims. Default: paper grid.")
    parser.add_argument("--das-max-epochs", type=int, default=100)
    parser.add_argument("--das-min-epochs", type=int, default=5)
    parser.add_argument("--das-plateau-patience", type=int, default=1)
    parser.add_argument("--das-plateau-rel-delta", type=float, default=1e-3)
    parser.add_argument("--das-learning-rate", type=float, default=1e-3)
    parser.add_argument("--das-restarts", type=int, default=2)
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
    base_run.TOKEN_POSITION_IDS = [str(args.token_position_id)]


def _load_existing_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()
    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_plot_das_layer"
    sweep_root.mkdir(parents=True, exist_ok=True)
    _configure_base_run(args, sweep_root=sweep_root, results_timestamp=results_timestamp)
    context = base_run.build_run_context()
    data_metadata = context["data_metadata"]
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_positions = context["token_positions"]
    banks_by_split = context["banks_by_split"]
    device = context["device"]
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    das_subspace_dims = tuple(_parse_csv_ints(args.das_subspace_dims) or list(DEFAULT_DAS_SUBSPACE_DIMS))

    layer_dir = sweep_root / f"layer_{int(args.layer):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    compare_output_path = layer_dir / f"mcqa_plot_das_layer_layer-{int(args.layer)}_pos-{str(args.token_position_id)}.json"
    compare_summary_path = layer_dir / f"mcqa_plot_das_layer_layer-{int(args.layer)}_pos-{str(args.token_position_id)}.txt"
    compare_payload = _load_existing_payload(compare_output_path)
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
                output_path=compare_output_path,
                summary_path=compare_summary_path,
                methods=("das",),
                target_vars=target_vars,
                batch_size=int(args.batch_size),
                signature_mode=str(args.signature_mode),
                das_max_epochs=int(args.das_max_epochs),
                das_min_epochs=int(args.das_min_epochs),
                das_plateau_patience=int(args.das_plateau_patience),
                das_plateau_rel_delta=float(args.das_plateau_rel_delta),
                das_learning_rate=float(args.das_learning_rate),
                das_subspace_dims=das_subspace_dims,
                das_restarts=max(1, int(args.das_restarts)),
                resolution=int(model.config.hidden_size),
                layers=(int(args.layer),),
                token_position_ids=(str(args.token_position_id),),
            ),
        )
    summary_payload = {
        "kind": "mcqa_plot_das_layer",
        "layer": int(args.layer),
        "token_position_id": str(args.token_position_id),
        "signature_mode": str(args.signature_mode),
        "das_subspace_dims": [int(dim) for dim in das_subspace_dims],
        "compare_output_path": str(compare_output_path),
        "method_payloads": compare_payload.get("method_payloads", {}),
        "results": compare_payload.get("results", []),
        "runtime_seconds": float(perf_counter() - stage_start),
    }
    payload_path = layer_dir / f"mcqa_plot_das_layer_layer-{int(args.layer)}_pos-{str(args.token_position_id)}_summary.json"
    write_json(payload_path, summary_payload)
    write_text_report(
        layer_dir / f"mcqa_plot_das_layer_layer-{int(args.layer)}_pos-{str(args.token_position_id)}_summary.txt",
        "\n".join(
            [
                "MCQA PLOT-DAS (layer) Summary",
                f"layer: {int(args.layer)}",
                f"token_position_id: {str(args.token_position_id)}",
                f"compare_output_path: {compare_output_path}",
            ]
        ),
    )
    print(f"Wrote layer DAS payload to {payload_path}")


if __name__ == "__main__":
    main()
