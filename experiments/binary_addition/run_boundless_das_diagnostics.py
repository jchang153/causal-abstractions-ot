#!/usr/bin/env python3
"""One-seed diagnostic sweep for Boundless DAS hyperparameters and variants."""

from __future__ import annotations

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
from experiments.binary_addition.run_boundless_das import _atomic_json, _checkpoint
from experiments.binary_addition.run_joint_endogenous_resolution_sweep import (
    EndogenousRowSpec,
    _build_banks,
    _load_or_train_model,
)
from experiments.binary_addition.run_progressive_plot import _fit_records_for_row
from experiments.binary_addition.sites import FullStateSite


RUN_DIR = ROOT / "results" / "7-29 boundless das" / "bdas_seed0_diagnostics"
ROWS = ("C1", "C2", "C3")
SOURCE_POLICY = "structured_26_top3carry_c2x5_c3x7_no_random"


def _variants() -> list[tuple[str, dict[str, object], str]]:
    variants: list[tuple[str, dict[str, object], str]] = [("released", {}, "anchored")]
    variants += [
        (f"penalty_{value:g}", {"boundary_penalty": value}, "anchored")
        for value in (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
    ]
    variants += [
        ("penalty_quadratic_1", {"boundary_penalty_power": 2.0}, "anchored"),
        ("penalty_quadratic_0.1", {"boundary_penalty": 0.1, "boundary_penalty_power": 2.0}, "anchored"),
        ("penalty_warmup_25pct", {"boundary_penalty_warmup_fraction": 0.25}, "anchored"),
        ("penalty_warmup_50pct", {"boundary_penalty_warmup_fraction": 0.5}, "anchored"),
    ]
    variants += [
        (f"rotation_lr_{value:g}", {"rotation_learning_rate": value}, "anchored")
        for value in (3e-3, 1e-2, 3e-2)
    ]
    variants += [
        (f"boundary_lr_{value:g}", {"boundary_learning_rate": value}, "anchored")
        for value in (1e-3, 3e-3, 3e-2)
    ]
    variants += [
        ("epochs_6", {"epochs": 6}, "anchored"),
        ("epochs_12", {"epochs": 12}, "anchored"),
        ("scheduler_optimizer_horizon", {"scheduler_horizon": "optimizer_steps"}, "anchored"),
        ("scheduler_constant", {"scheduler_horizon": "constant"}, "anchored"),
    ]
    variants += [
        (f"temperature_start_{value:g}", {"temperature_start": value}, "anchored")
        for value in (0.1, 1.0, 5.0, 10.0)
    ]
    variants += [
        ("temperature_end_0.01", {"temperature_end": 0.01}, "anchored"),
        ("temperature_end_1", {"temperature_end": 1.0}, "anchored"),
    ]
    variants += [
        (f"boundary_init_{value:g}", {"boundary_init": value}, "anchored")
        for value in (0.25, 0.75, 1.0)
    ]
    variants += [
        (f"changed_bit_weight_{value:g}", {"changed_bit_weight": value}, "anchored")
        for value in (2.0, 5.0, 10.0)
    ]
    variants += [
        (f"active_record_weight_{value:g}", {"active_record_weight": value}, "anchored")
        for value in (2.0, 5.0)
    ]
    variants += [
        ("select_sensitivity", {"selection_metric": "sensitivity"}, "anchored"),
        ("fit_shared", {}, "shared"),
        ("fit_active_only", {}, "active_only"),
    ]
    variants += [
        (
            "combo_no_penalty_rot3e3_epochs12",
            {"boundary_penalty": 0.0, "rotation_learning_rate": 3e-3, "epochs": 12},
            "anchored",
        ),
        (
            "combo_no_penalty_rot1e2_epochs12",
            {"boundary_penalty": 0.0, "rotation_learning_rate": 1e-2, "epochs": 12},
            "anchored",
        ),
        (
            "combo_pen1e3_rot1e2_epochs12",
            {"boundary_penalty": 1e-3, "rotation_learning_rate": 1e-2, "epochs": 12},
            "anchored",
        ),
        (
            "combo_pen1e2_rot1e2_epochs12",
            {"boundary_penalty": 1e-2, "rotation_learning_rate": 1e-2, "epochs": 12},
            "anchored",
        ),
        (
            "combo_pen1e2_rot1e2_bound1e3_epochs12",
            {
                "boundary_penalty": 1e-2,
                "rotation_learning_rate": 1e-2,
                "boundary_learning_rate": 1e-3,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "combo_no_penalty_changed5_rot1e2_epochs12",
            {
                "boundary_penalty": 0.0,
                "changed_bit_weight": 5.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "combo_pen1e3_changed5_active2_rot1e2_epochs12",
            {
                "boundary_penalty": 1e-3,
                "changed_bit_weight": 5.0,
                "active_record_weight": 2.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "combo_pen1e2_warmup50_rot1e2_epochs12",
            {
                "boundary_penalty": 1e-2,
                "boundary_penalty_warmup_fraction": 0.5,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "combo_no_penalty_temp1_rot1e2_epochs12",
            {
                "boundary_penalty": 0.0,
                "temperature_start": 1.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "combo_no_penalty_init1_rot1e2_epochs12",
            {
                "boundary_penalty": 0.0,
                "boundary_init": 1.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "combo_pen1e3_rot1e2_epochs12_select_sens",
            {
                "boundary_penalty": 1e-3,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
                "selection_metric": "sensitivity",
            },
            "anchored",
        ),
        (
            "combo_no_penalty_rot1e2_epochs12_active_only",
            {"boundary_penalty": 0.0, "rotation_learning_rate": 1e-2, "epochs": 12},
            "active_only",
        ),
    ]
    variants += [
        (
            f"focused_no_penalty_temp{value:g}_rot1e2_epochs12",
            {
                "boundary_penalty": 0.0,
                "temperature_start": value,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        )
        for value in (0.05, 0.2, 0.5, 2.0, 5.0)
    ]
    variants += [
        (
            "focused_scaled_temperature_1.0417",
            {
                "boundary_penalty": 0.0,
                "temperature_start": 50.0 * 16.0 / 768.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        *[
            (
                f"focused_temp1_penalty_{value:g}",
                {
                    "temperature_start": 1.0,
                    "boundary_penalty": value,
                    "rotation_learning_rate": 1e-2,
                    "epochs": 12,
                },
                "anchored",
            )
            for value in (1e-3, 1e-2, 0.1, 1.0)
        ],
        *[
            (
                f"focused_temp1_pen1_boundarylr_{value:g}",
                {
                    "temperature_start": 1.0,
                    "boundary_learning_rate": value,
                    "rotation_learning_rate": 1e-2,
                    "epochs": 12,
                },
                "anchored",
            )
            for value in (1e-4, 3e-4, 1e-3, 3e-3)
        ],
        (
            "focused_temp1_pen0.1_boundarylr1e3",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.1,
                "boundary_learning_rate": 1e-3,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "focused_temp1_pen0.01_boundarylr1e3",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 1e-2,
                "boundary_learning_rate": 1e-3,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "focused_temp1_quadratic0.1_boundarylr1e3",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.1,
                "boundary_penalty_power": 2.0,
                "boundary_learning_rate": 1e-3,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "focused_temp1_no_penalty_rot3e3_epochs12",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.0,
                "rotation_learning_rate": 3e-3,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "focused_temp1_no_penalty_rot3e2_epochs12",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.0,
                "rotation_learning_rate": 3e-2,
                "epochs": 12,
            },
            "anchored",
        ),
        (
            "focused_temp1_no_penalty_rot1e2_epochs6",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 6,
            },
            "anchored",
        ),
        (
            "focused_temp1_no_penalty_rot1e2_epochs24",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 24,
            },
            "anchored",
        ),
        *[
            (
                f"focused_temp1_no_penalty_init_{value:g}",
                {
                    "temperature_start": 1.0,
                    "boundary_penalty": 0.0,
                    "boundary_init": value,
                    "rotation_learning_rate": 1e-2,
                    "epochs": 12,
                },
                "anchored",
            )
            for value in (0.25, 0.75, 1.0)
        ],
        (
            "focused_temp1_pen1_boundarylr1e3_cap640",
            {
                "temperature_start": 1.0,
                "boundary_learning_rate": 1e-3,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
                "train_records_per_epoch": 640,
            },
            "anchored",
        ),
        (
            "focused_temp1_no_penalty_batch64_accum1",
            {
                "temperature_start": 1.0,
                "boundary_penalty": 0.0,
                "rotation_learning_rate": 1e-2,
                "epochs": 12,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
            },
            "anchored",
        ),
    ]
    return variants


def _summary_row(name: str, fit_mode: str, result: dict[str, object]) -> dict[str, object]:
    selected = result["selected_by_row"]
    return {
        "name": name,
        "fit_mode": fit_mode,
        "combined": float(result["test"]["combined"]),
        "sensitivity": float(result["test"]["sensitivity"]),
        "invariance": float(result["test"]["invariance"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "dimensions": {row: int(selected[row]["hard_dimension"]) for row in ROWS},
        "timesteps": {row: int(selected[row]["site_timestep"]) for row in ROWS},
        "config": result["config"],
    }


def _write_rankings(rows: list[dict[str, object]]) -> None:
    ranked = sorted(rows, key=lambda row: (row["combined"], row["sensitivity"]), reverse=True)
    _atomic_json(RUN_DIR / "summary.json", {"seed": 0, "variant_count": len(rows), "results": ranked})
    lines = [
        "| Rank | Variant | Combined | Sensitivity | Invariance | Dimensions C1/C2/C3 | Runtime (s) |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        dims = row["dimensions"]
        lines.append(
            f"| {rank} | {row['name']} | {row['combined']:.4f} | {row['sensitivity']:.4f} | "
            f"{row['invariance']:.4f} | {dims['C1']}/{dims['C2']}/{dims['C3']} | {row['runtime_seconds']:.2f} |"
        )
    (RUN_DIR / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed = 0
    device = torch.device("cpu")
    examples = enumerate_all_examples(width=4)
    split = stratified_base_split(examples, fit_count=128, calib_count=64, test_count=64, seed=seed)
    checkpoint = _checkpoint(seed, 16)
    model_args = SimpleNamespace(
        model_checkpoint=str(checkpoint),
        device="cpu",
        width=4,
        hidden_size=16,
        train_on="all",
        train_batch_size=64,
        train_epochs=120,
        train_lr=0.02,
        seed=seed,
    )
    model, _metadata = _load_or_train_model(model_args, examples, split)
    model.eval()
    model.requires_grad_(False)
    run_cache = build_run_cache(model, examples, device=device)
    specs = tuple(EndogenousRowSpec(key=row, kind="carry", index=int(row[1:])) for row in ROWS)
    banks = _build_banks(
        split,
        specs,
        width=4,
        seed=seed,
        source_policy=SOURCE_POLICY,
        all_examples=examples,
    )
    anchored = {
        row: _fit_records_for_row(banks["fit_by_row"][row], row_key=row, fit_bank_mode="anchored_prefix")
        for row in ROWS
    }
    fit_banks = {
        "anchored": anchored,
        "shared": {row: tuple(banks["fit_by_row"][row]) for row in ROWS},
        "active_only": {row: tuple(record for record in anchored[row] if record.is_active) for row in ROWS},
    }
    sites = tuple(FullStateSite(timestep=timestep) for timestep in range(4))
    rows: list[dict[str, object]] = []
    variants = _variants()
    print(f"Running {len(variants)} seed-0 BDAS variants", flush=True)
    for index, (name, overrides, fit_mode) in enumerate(variants, start=1):
        output_path = RUN_DIR / "variants" / f"{name}.json"
        if output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(variants)} resume] {name}", flush=True)
        else:
            config = replace(BDASConfig(seed=seed), **overrides)
            result = run_bdas_rows(
                model,
                fit_by_row=fit_banks[fit_mode],
                calibration_positive_by_row=banks["calib_positive_by_row"],
                calibration_invariant_by_row=banks["calib_invariant_by_row"],
                test_positive_by_row=banks["test_positive_by_row"],
                test_invariant_by_row=banks["test_invariant_by_row"],
                sites=sites,
                row_keys=ROWS,
                config=config,
                device=device,
                run_cache=run_cache,
            )
            result["diagnostic_name"] = name
            result["fit_mode"] = fit_mode
            _atomic_json(output_path, result)
            print(
                f"[{index}/{len(variants)}] {name}: combined={result['test']['combined']:.4f}, "
                f"sens={result['test']['sensitivity']:.4f}, inv={result['test']['invariance']:.4f}",
                flush=True,
            )
        rows.append(_summary_row(name, fit_mode, result))
        _write_rankings(rows)
    print(f"Wrote {RUN_DIR / 'ranking.md'}", flush=True)


if __name__ == "__main__":
    main()
