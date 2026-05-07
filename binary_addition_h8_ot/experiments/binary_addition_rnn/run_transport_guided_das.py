from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition_rnn.interventions import build_run_cache
from experiments.binary_addition_rnn.model import exact_accuracy
from experiments.binary_addition_rnn.run_joint_endogenous_das import (
    _calibration_key,
    _exact_match_rate,
    _load_or_train_model,
    _site_dim,
    _train_rotator,
)
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import _build_banks, _row_specs, _subset_summary
from experiments.binary_addition_rnn.support import site_spec_to_key, site_spec_to_site


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run restricted DAS on supports extracted from shared OT discovery.")
    ap.add_argument("--support-summary", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--abstract-mode", type=str, default="all_endogenous", choices=["carries_only", "all_endogenous"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fit-bases", type=int, default=128)
    ap.add_argument("--calib-bases", type=int, default=64)
    ap.add_argument("--test-bases", type=int, default=64)
    ap.add_argument("--train-on", type=str, default="all", choices=["all", "fit_only"])
    ap.add_argument("--train-epochs", type=int, default=120)
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--train-lr", type=float, default=0.02)
    ap.add_argument("--model-checkpoint", type=str, default="")
    ap.add_argument("--source-policy", type=str, default="structured_26_top3carry_c2x5_c3x7_no_random")
    ap.add_argument("--fit-bank-mode", type=str, default="anchored_prefix", choices=["shared", "anchored_prefix"])
    ap.add_argument("--rows", type=str, default="C1,C2,C3")
    ap.add_argument("--mask-modes", type=str, default="StepMask,S80,S90")
    ap.add_argument("--selection-rule", type=str, default="combined", choices=["combined", "sensitivity_only", "sensitivity_then_invariance"])
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--lambda-grid", type=str, default="0.25,0.5,1,2,4,8")
    ap.add_argument("--das-subspace-dims", type=str, default="1,2,4")
    ap.add_argument("--das-lrs", type=str, default="0.01,0.003")
    ap.add_argument("--das-epochs", type=int, default=12)
    ap.add_argument("--das-train-records-per-epoch", type=int, default=256)
    ap.add_argument("--das-batch-size", type=int, default=64)
    return ap.parse_args()


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_keys(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _fit_records_for_row(all_fit_records: tuple[object, ...], *, row_key: str, fit_bank_mode: str) -> tuple[object, ...]:
    if str(fit_bank_mode) != "anchored_prefix":
        return tuple(all_fit_records)
    if row_key.startswith("C"):
        carry_index = int(row_key[1:])
        prefixes = [f"flip_A{carry_index - 1}", f"flip_B{carry_index - 1}", f"target_C{carry_index}"]
        if carry_index > 1:
            prefixes.append(f"target_C{carry_index - 1}")
    elif row_key.startswith("S"):
        sum_index = int(row_key[1:])
        prefixes = [f"flip_A{sum_index}", f"flip_B{sum_index}"]
        if sum_index > 0:
            prefixes.append(f"target_C{sum_index}")
    else:
        prefixes = []
    return tuple(
        rec
        for rec in all_fit_records
        if any(rec.family == prefix or rec.family.startswith(prefix + "_") for prefix in prefixes)
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    support_summary = json.loads(Path(args.support_summary).resolve().read_text(encoding="utf-8"))
    requested_rows = _parse_keys(args.rows)
    mask_modes = _parse_keys(args.mask_modes)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    examples = enumerate_all_examples(width=int(args.width))
    split = stratified_base_split(
        examples,
        fit_count=int(args.fit_bases),
        calib_count=int(args.calib_bases),
        test_count=int(args.test_bases),
        seed=int(args.seed),
    )
    model, train_summary = _load_or_train_model(args, examples, split)
    run_cache = build_run_cache(model, examples, device=device)

    specs = _row_specs(args.abstract_mode, int(args.width))
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(args.seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )

    lambda_grid = _parse_floats(args.lambda_grid)
    requested_dims = _parse_ints(args.das_subspace_dims)
    requested_lrs = _parse_floats(args.das_lrs)

    carry_keys = [row_key for row_key in requested_rows if row_key.startswith("C")]
    internal_carry_keys = [row_key for row_key in requested_rows if row_key.startswith("C") and row_key != f"C{int(args.width)}"]

    per_mask_mode = {}
    best_mask_mode = None
    best_mask_key = None
    for mask_mode in mask_modes:
        per_row = {}
        calib_sens = []
        calib_inv = []
        test_sens = []
        test_inv = []
        for row_key in requested_rows:
            row_support = support_summary["row_support"][row_key]
            site_spec = row_support["masks"][mask_mode]
            site = site_spec_to_site(site_spec)
            fit_records = _fit_records_for_row(
                tuple(banks["fit_by_row"][row_key]),
                row_key=row_key,
                fit_bank_mode=str(args.fit_bank_mode),
            )
            site_dim = _site_dim(site, int(args.hidden_size))
            if site_dim <= 1:
                candidate_models = [(None, None, None)]
            else:
                valid_dims = [dim for dim in requested_dims if 1 <= int(dim) <= site_dim]
                candidate_models = []
                for subspace_dim in valid_dims:
                    for lr in requested_lrs:
                        rotator = _train_rotator(
                            model,
                            fit_records,
                            site,
                            subspace_dim=int(subspace_dim),
                            lr=float(lr),
                            epochs=int(args.das_epochs),
                            train_records_per_epoch=int(args.das_train_records_per_epoch),
                            batch_size=int(args.das_batch_size),
                            device=device,
                            run_cache=run_cache,
                            seed=int(args.seed),
                        )
                        candidate_models.append((rotator, int(subspace_dim), float(lr)))

            row_best = None
            row_best_key = None
            for rotator, subspace_dim, lr in candidate_models:
                for lambda_scale in lambda_grid:
                    sens = _exact_match_rate(
                        model,
                        banks["calib_positive_by_row"][row_key],
                        site,
                        lambda_scale=float(lambda_scale),
                        device=device,
                        run_cache=run_cache,
                        rotator=rotator,
                        batch_size=int(args.das_batch_size),
                    )
                    inv = _exact_match_rate(
                        model,
                        banks["calib_invariant_by_row"][row_key],
                        site,
                        lambda_scale=float(lambda_scale),
                        device=device,
                        run_cache=run_cache,
                        rotator=rotator,
                        batch_size=int(args.das_batch_size),
                    )
                    combined = 0.5 * (float(sens) + float(inv))
                    key = _calibration_key(
                        combined=combined,
                        sensitivity=float(sens),
                        invariance=float(inv),
                        selection_rule=str(args.selection_rule),
                        invariance_floor=float(args.invariance_floor),
                    )
                    trial = {
                        "site_key": site.key(),
                        "site_spec": site_spec,
                        "lambda": float(lambda_scale),
                        "subspace_dim": int(subspace_dim) if subspace_dim is not None else int(site_dim),
                        "lr": float(lr) if lr is not None else None,
                        "rotator_state": {k: v.tolist() for k, v in rotator.state_dict().items()} if rotator is not None else None,
                        "calibration": {
                            "sensitivity": float(sens),
                            "invariance": float(inv),
                            "combined": float(combined),
                        },
                    }
                    if row_best_key is None or key > row_best_key:
                        row_best_key = key
                        row_best = trial

            if row_best is None:
                continue

            if row_best["rotator_state"] is None or site_dim <= 1:
                rotator = None
            else:
                from experiments.binary_addition_rnn.das import RotatedSubspace

                rotator = RotatedSubspace(site_dim, int(row_best["subspace_dim"]))
                rotator.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in row_best["rotator_state"].items()})

            test_eval = {
                "sensitivity": _exact_match_rate(
                    model,
                    banks["test_positive_by_row"][row_key],
                    site,
                    lambda_scale=float(row_best["lambda"]),
                    device=device,
                    run_cache=run_cache,
                    rotator=rotator,
                    batch_size=int(args.das_batch_size),
                ),
                "invariance": _exact_match_rate(
                    model,
                    banks["test_invariant_by_row"][row_key],
                    site,
                    lambda_scale=float(row_best["lambda"]),
                    device=device,
                    run_cache=run_cache,
                    rotator=rotator,
                    batch_size=int(args.das_batch_size),
                ),
            }
            test_eval["combined"] = 0.5 * (float(test_eval["sensitivity"]) + float(test_eval["invariance"]))

            per_row[row_key] = {
                "site_key": site_spec_to_key(site_spec),
                "site_spec": site_spec,
                "support_meta": {
                    "dominant_hidden_timestep": support_summary["row_support"][row_key]["dominant_hidden_timestep"],
                    "compression": support_summary["row_support"][row_key]["compression"].get(mask_mode),
                },
                "calibration": row_best["calibration"],
                "test": test_eval,
                "lambda": float(row_best["lambda"]),
                "subspace_dim": int(row_best["subspace_dim"]),
                "lr": row_best["lr"],
            }
            calib_sens.append(float(row_best["calibration"]["sensitivity"]))
            calib_inv.append(float(row_best["calibration"]["invariance"]))
            test_sens.append(float(test_eval["sensitivity"]))
            test_inv.append(float(test_eval["invariance"]))

        per_mask_mode[mask_mode] = {
            "per_row": per_row,
            "calibration": {
                "mean_sensitivity": float(sum(calib_sens) / max(1, len(calib_sens))),
                "mean_invariance": float(sum(calib_inv) / max(1, len(calib_inv))),
                "mean_combined": float((sum(calib_sens) + sum(calib_inv)) / max(1, 2 * len(calib_sens))),
            },
            "test": {
                "mean_sensitivity": float(sum(test_sens) / max(1, len(test_sens))),
                "mean_invariance": float(sum(test_inv) / max(1, len(test_inv))),
                "mean_combined": float((sum(test_sens) + sum(test_inv)) / max(1, 2 * len(test_sens))),
                "carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, carry_keys) if carry_keys else None,
                "internal_carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, internal_carry_keys) if internal_carry_keys else None,
            },
        }
        mask_key = (
            float(per_mask_mode[mask_mode]["calibration"]["mean_combined"]),
            float(per_mask_mode[mask_mode]["calibration"]["mean_sensitivity"]),
            float(per_mask_mode[mask_mode]["calibration"]["mean_invariance"]),
        )
        if best_mask_key is None or mask_key > best_mask_key:
            best_mask_key = mask_key
            best_mask_mode = str(mask_mode)

    result = {
        "config": vars(args),
        "support_summary": str(Path(args.support_summary).resolve()),
        "rows": list(requested_rows),
        "mask_modes": list(mask_modes),
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "training": train_summary,
        "per_mask_mode": per_mask_mode,
        "best_mask_mode": best_mask_mode,
        "best_result": None if best_mask_mode is None else per_mask_mode[best_mask_mode],
    }
    summary_path = out_dir / "transport_guided_das_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "mask_modes": list(per_mask_mode.keys()), "best_mask_mode": best_mask_mode}, indent=2))


if __name__ == "__main__":
    main()
