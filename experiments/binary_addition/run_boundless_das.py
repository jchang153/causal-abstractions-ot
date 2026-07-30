#!/usr/bin/env python3
"""Run Boundless DAS on the matched ten-seed binary-addition benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition.bdas import BDASConfig, run_bdas_rows
from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.interventions import build_run_cache
from experiments.binary_addition.model import exact_accuracy
from experiments.binary_addition.run_joint_endogenous_resolution_sweep import (
    EndogenousRowSpec,
    _bank_summaries,
    _build_banks,
    _load_or_train_model,
)
from experiments.binary_addition.run_progressive_plot import _fit_records_for_row
from experiments.binary_addition.sites import FullStateSite


DEFAULT_RUN_DIR = ROOT / "results" / "7-29 boundless das" / "binary_addition_bdas_h16_10seeds"
SOURCE_POLICY = "structured_26_top3carry_c2x5_c3x7_no_random"


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_rows(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _checkpoint(seed: int, hidden_size: int) -> Path:
    if seed == 0:
        return ROOT / "eval" / f"binary_backbone_h{hidden_size}_seed0" / "gru_adder.pt"
    shared = ROOT / "eval" / "shared_checkpoints" / f"gru_h{hidden_size}_seed{seed}.pt"
    if shared.exists():
        return shared
    return (
        ROOT
        / "results"
        / "7-26 binary addition local baselines"
        / "binary_addition_mib_baselines_10seeds"
        / "checkpoints"
        / f"gru_h{hidden_size}_seed{seed}.pt"
    )


def _stats(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0,
    }


def _write_aggregate(run_dir: Path, seeds: tuple[int, ...]) -> dict[str, object]:
    results = [json.loads((run_dir / f"seed_{seed}.json").read_text(encoding="utf-8")) for seed in seeds]
    summary = {
        "method": "Boundless DAS",
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "combined": _stats([float(item["test"]["combined"]) for item in results]),
        "sensitivity": _stats([float(item["test"]["sensitivity"]) for item in results]),
        "invariance": _stats([float(item["test"]["invariance"]) for item in results]),
        "runtime_seconds": _stats([float(item["runtime_seconds"]) for item in results]),
        "selected_timesteps": {
            str(seed): {row: int(payload["selected_by_row"][row]["site_timestep"]) for row in payload["row_keys"]}
            for seed, payload in zip(seeds, results)
        },
        "selected_dimensions": {
            str(seed): {row: int(payload["selected_by_row"][row]["hard_dimension"]) for row in payload["row_keys"]}
            for seed, payload in zip(seeds, results)
        },
    }
    _atomic_json(run_dir / "aggregate_summary.json", summary)
    row = (
        "Boundless DAS & "
        f"${summary['combined']['mean']:.4f} \\pm {summary['combined']['std']:.4f}$ & "
        f"${summary['sensitivity']['mean']:.4f} \\pm {summary['sensitivity']['std']:.4f}$ & "
        f"${summary['invariance']['mean']:.4f} \\pm {summary['invariance']['std']:.4f}$ & "
        f"${summary['runtime_seconds']['mean']:.2f} \\pm {summary['runtime_seconds']['std']:.2f}$ \\\\" + "\n"
    )
    (run_dir / "table_row.tex").write_text(row, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--preset",
        choices=("released", "recommended", "recommended_batch64"),
        default="released",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--rows", default="C1,C2,C3")
    parser.add_argument("--timesteps", default="0,1,2,3")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--fit-bases", type=int, default=128)
    parser.add_argument("--calib-bases", type=int, default=64)
    parser.add_argument("--test-bases", type=int, default=64)
    parser.add_argument("--source-policy", default=SOURCE_POLICY)
    parser.add_argument("--fit-bank-mode", choices=("shared", "anchored_prefix"), default="anchored_prefix")
    parser.add_argument(
        "--train-records-per-epoch",
        type=int,
        default=None,
        help="Optional cap; by default BDAS follows the released code and uses the full fit bank.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_ints(args.seeds)
    row_keys = _parse_rows(args.rows)
    timesteps = _parse_ints(args.timesteps)
    specs = tuple(EndogenousRowSpec(key=row, kind="carry", index=int(row[1:])) for row in row_keys)
    sites = tuple(FullStateSite(timestep=timestep) for timestep in timesteps)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = BDASConfig(train_records_per_epoch=args.train_records_per_epoch)
    if args.preset in {"recommended", "recommended_batch64"}:
        base_config = replace(
            base_config,
            rotation_learning_rate=1e-2,
            boundary_learning_rate=1e-4,
            boundary_penalty=1.0,
            boundary_init=0.5,
            temperature_start=1.0,
            temperature_end=0.1,
            epochs=12,
        )
    if args.preset == "recommended_batch64":
        base_config = replace(
            base_config,
            batch_size=64,
            gradient_accumulation_steps=1,
        )
    config_payload = {
        **vars(args),
        "run_dir": str(run_dir),
        "seeds": list(seeds),
        "rows": list(row_keys),
        "timesteps": list(timesteps),
        "bdas": base_config.as_dict(),
        "lambda": 1.0,
    }
    _atomic_json(run_dir / "config.json", config_payload)
    examples = enumerate_all_examples(width=args.width)

    for seed in seeds:
        output_path = run_dir / f"seed_{seed}.json"
        if output_path.exists() and not args.no_resume:
            print(f"[resume] seed {seed}: {output_path}", flush=True)
            continue
        checkpoint = _checkpoint(seed, args.hidden_size)
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing matched seed-{seed} checkpoint: {checkpoint}")
        split = stratified_base_split(
            examples,
            fit_count=args.fit_bases,
            calib_count=args.calib_bases,
            test_count=args.test_bases,
            seed=seed,
        )
        model_args = SimpleNamespace(
            model_checkpoint=str(checkpoint),
            device=str(device),
            width=args.width,
            hidden_size=args.hidden_size,
            train_on="all",
            train_batch_size=64,
            train_epochs=120,
            train_lr=0.02,
            seed=seed,
        )
        model, model_metadata = _load_or_train_model(model_args, examples, split)
        model.eval()
        model.requires_grad_(False)
        run_cache = build_run_cache(model, examples, device=device)
        banks = _build_banks(
            split,
            specs,
            width=args.width,
            seed=seed,
            source_policy=args.source_policy,
            all_examples=examples,
        )
        fit_by_row = {
            row: _fit_records_for_row(
                banks["fit_by_row"][row],
                row_key=row,
                fit_bank_mode=args.fit_bank_mode,
            )
            for row in row_keys
        }
        print(
            f"[run] seed {seed}: "
            + ", ".join(f"{row}={len(fit_by_row[row])} fit" for row in row_keys),
            flush=True,
        )
        result = run_bdas_rows(
            model,
            fit_by_row=fit_by_row,
            calibration_positive_by_row=banks["calib_positive_by_row"],
            calibration_invariant_by_row=banks["calib_invariant_by_row"],
            test_positive_by_row=banks["test_positive_by_row"],
            test_invariant_by_row=banks["test_invariant_by_row"],
            sites=sites,
            row_keys=row_keys,
            config=replace(base_config, seed=seed),
            device=device,
            run_cache=run_cache,
        )
        result.update(
            {
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "model": model_metadata,
                "factual_exact": exact_accuracy(model, examples, device=device),
                "banks": _bank_summaries(
                    banks["fit_by_row"],
                    {row: tuple(rec for rec in banks["fit_by_row"][row] if rec.is_active) for row in row_keys},
                    {row: tuple(rec for rec in banks["fit_by_row"][row] if not rec.is_active) for row in row_keys},
                ),
                "fit_record_counts_after_mode": {row: len(fit_by_row[row]) for row in row_keys},
            }
        )
        _atomic_json(output_path, result)
        print(
            f"[done] seed {seed}: combined={result['test']['combined']:.4f}, "
            f"sens={result['test']['sensitivity']:.4f}, inv={result['test']['invariance']:.4f}, "
            f"runtime={result['runtime_seconds']:.2f}s",
            flush=True,
        )

    summary = _write_aggregate(run_dir, seeds)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
