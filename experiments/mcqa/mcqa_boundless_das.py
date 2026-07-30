#!/usr/bin/env python3
"""Run Full Boundless DAS over all MCQA layers and report selected-only test metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mcqa_run as base_run
from mcqa_experiment.bdas import BoundlessDASConfig, run_boundless_das_pipeline
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import enumerate_residual_sites


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=3000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=200)
    parser.add_argument("--test-pool-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--token-position-id", default="last_token")
    parser.add_argument("--target-vars", default="answer_pointer,answer_token")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--rotation-learning-rate", type=float, default=1e-2)
    parser.add_argument("--boundary-learning-rate", type=float, default=1e-4)
    parser.add_argument("--boundary-init", type=float, default=0.5)
    parser.add_argument("--boundary-penalty", type=float, default=1.0)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=Path, default=Path("results/delta"))
    parser.add_argument("--results-timestamp", default=None)
    parser.add_argument("--signatures-dir", type=Path, default=Path("signatures"))
    parser.add_argument("--prompt-hf-login", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _configure(args: argparse.Namespace, run_dir: Path, timestamp: str) -> None:
    base_run.DEVICE = str(args.device)
    base_run.MODEL_NAME = str(args.model_name)
    base_run.MCQA_DATASET_PATH = str(args.dataset_path)
    base_run.MCQA_DATASET_CONFIG = args.dataset_config or None
    base_run.DATASET_SIZE = int(args.dataset_size)
    base_run.SPLIT_SEED = int(args.split_seed)
    base_run.TRAIN_POOL_SIZE = int(args.train_pool_size)
    base_run.CALIBRATION_POOL_SIZE = int(args.calibration_pool_size)
    base_run.TEST_POOL_SIZE = int(args.test_pool_size)
    base_run.BATCH_SIZE = int(args.batch_size)
    base_run.TARGET_VARS = [item.strip() for item in str(args.target_vars).split(",") if item.strip()]
    base_run.TOKEN_POSITION_IDS = [str(args.token_position_id)]
    base_run.PROMPT_HF_LOGIN = bool(args.prompt_hf_login)
    base_run.RUN_TIMESTAMP = timestamp
    base_run.RUN_DIR = run_dir
    base_run.OUTPUT_PATH = run_dir / "mcqa_run_results.json"
    base_run.SUMMARY_PATH = run_dir / "mcqa_run_summary.txt"
    base_run.SIGNATURES_DIR = Path(args.signatures_dir)


def main() -> None:
    args = _parser().parse_args()
    timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or f"mcqa_bdas_seed{args.split_seed}"
    run_dir = Path(args.results_root) / f"{timestamp}_mcqa_boundless_das"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "mcqa_boundless_das_results.json"
    existing_result = {}
    if args.resume and output_path.exists():
        try:
            existing_result = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_result = {}
    _configure(args, run_dir, timestamp)
    context = base_run.build_run_context()
    model = context["model"]
    tokenizer = context["tokenizer"]
    banks_by_split = context["banks_by_split"]
    device = context["device"]
    layer_ids = (
        tuple(range(int(model.config.num_hidden_layers)))
        if str(args.layers).strip().lower() == "all"
        else _csv_ints(str(args.layers))
    )
    token_position_ids = tuple(position.id for position in context["token_positions"])
    sites = enumerate_residual_sites(
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        token_position_ids=token_position_ids,
        resolution=None,
        layers=layer_ids,
        selected_token_position_ids=(str(args.token_position_id),),
    )
    target_vars = tuple(item.strip() for item in str(args.target_vars).split(",") if item.strip())
    payloads = dict(existing_result.get("payloads_by_var", {}))
    for offset, target_var in enumerate(target_vars):
        if args.resume and target_var in payloads:
            print(f"[resume] target_var={target_var}")
            continue
        payloads[target_var] = run_boundless_das_pipeline(
            model=model,
            train_bank=banks_by_split["train"][target_var],
            calibration_bank=banks_by_split["calibration"][target_var],
            holdout_bank=banks_by_split["test"][target_var],
            sites=sites,
            device=device,
            tokenizer=tokenizer,
            config=BoundlessDASConfig(
                batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                rotation_learning_rate=float(args.rotation_learning_rate),
                boundary_learning_rate=float(args.boundary_learning_rate),
                boundary_init=float(args.boundary_init),
                boundary_penalty=float(args.boundary_penalty),
                temperature_start=float(args.temperature_start),
                temperature_end=float(args.temperature_end),
                restarts=max(1, int(args.restarts)),
                seed=int(args.seed) + 104729 * offset,
            ),
        )
        write_json(output_path, {
            "kind": "mcqa_boundless_das",
            "config": {**vars(args), "results_root": str(args.results_root), "signatures_dir": str(args.signatures_dir)},
            "layers": list(layer_ids),
            "target_vars": list(target_vars),
            "payloads_by_var": payloads,
        })
    result = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
