from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition_rnn.interventions import build_run_cache
from experiments.binary_addition_rnn.mib_baselines import (
    MIBBaselineConfig,
    exact_match_rate,
    fit_featurizer,
    train_dbm_mask,
)
from experiments.binary_addition_rnn.model import exact_accuracy
from experiments.binary_addition_rnn.run_joint_endogenous_das import _load_or_train_model
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import (
    _bank_summaries,
    _build_banks,
    _row_specs,
    _subset_summary,
)
from experiments.binary_addition_rnn.sites import enumerate_full_state_sites, enumerate_output_logit_sites


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MIB-style full-vector / DBM / PCA / SAE baselines on the binary addition benchmark.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--methods", type=str, default="full_vector,dbm,dbm_pca,dbm_sae")
    ap.add_argument("--abstract-mode", type=str, default="all_endogenous", choices=["carries_only", "all_endogenous"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=4)
    ap.add_argument("--timesteps", type=str, default="0,1,2,3")
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
    ap.add_argument(
        "--source-policy",
        type=str,
        default="structured_20_top3carry_no_random",
        choices=[
            "all_source",
            "structured_13",
            "structured_12_no_random",
            "structured_17_top2carry",
            "structured_20_top3carry_no_random",
            "structured_21_top3carry",
            "structured_22_top3carry_c3x5_no_random",
            "structured_24_top3carry_c2c3x5_no_random",
            "structured_24_top3carry_c3x7_no_random",
            "structured_26_top3carry_c2x5_c3x7_no_random",
        ],
    )
    ap.add_argument("--selection-rule", type=str, default="combined", choices=["combined", "sensitivity_only", "sensitivity_then_invariance"])
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--lambda-grid", type=str, default="0.5,1,2,4")
    ap.add_argument("--mask-lrs", type=str, default="0.03")
    ap.add_argument("--mask-l1s", type=str, default="0.0")
    ap.add_argument("--mask-epochs", type=int, default=16)
    ap.add_argument("--mask-batch-size", type=int, default=64)
    ap.add_argument("--mask-train-records-per-epoch", type=int, default=256)
    ap.add_argument("--sae-latent-mult", type=int, default=4)
    ap.add_argument("--sae-lr", type=float, default=0.01)
    ap.add_argument("--sae-l1", type=float, default=1e-3)
    ap.add_argument("--sae-epochs", type=int, default=250)
    return ap.parse_args()


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _calibration_key(*, combined: float, sensitivity: float, invariance: float, selection_rule: str, invariance_floor: float) -> tuple[float, float, float]:
    admissible = 1.0 if float(invariance) >= float(invariance_floor) else 0.0
    if selection_rule == "combined":
        return (admissible, float(combined), float(sensitivity))
    if selection_rule == "sensitivity_only":
        return (1.0, float(sensitivity), float(invariance))
    if selection_rule == "sensitivity_then_invariance":
        return (admissible, float(sensitivity), float(invariance))
    raise ValueError(f"unknown selection_rule: {selection_rule!r}")


def _unique_fit_examples(fit_by_row: dict[str, tuple[object, ...]], row_key: str):
    seen = {}
    for record in fit_by_row[row_key]:
        seen[(int(record.base.a), int(record.base.b))] = record.base
        seen[(int(record.source.a), int(record.source.b))] = record.source
    return tuple(seen[key] for key in sorted(seen))


def _mask_summary(mask: torch.Tensor) -> dict[str, object]:
    mask = mask.to(torch.float32)
    return {
        "feature_dim": int(mask.numel()),
        "mask_mean": float(mask.mean().item()),
        "mask_max": float(mask.max().item()),
        "mask_min": float(mask.min().item()),
        "active_fraction_ge_0_5": float((mask >= 0.5).to(torch.float32).mean().item()),
        "mask_values": [float(x) for x in mask.tolist()],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    row_keys = [spec.key for spec in specs]
    carry_keys = [f"C{i}" for i in range(1, int(args.width) + 1)]
    output_keys = [f"S{i}" for i in range(int(args.width)) if f"S{i}" in row_keys]
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(args.seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )

    timesteps = _parse_ints(args.timesteps)
    sites = tuple(site for site in enumerate_full_state_sites(width=int(args.width)) if int(site.timestep) in timesteps) + tuple(
        enumerate_output_logit_sites(output_dim=int(args.width) + 1)
    )
    fit_examples = _unique_fit_examples(banks["fit_by_row"], row_keys[0])

    config = MIBBaselineConfig(
        lambda_grid=_parse_floats(args.lambda_grid),
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        mask_lrs=_parse_floats(args.mask_lrs),
        mask_l1s=_parse_floats(args.mask_l1s),
        mask_epochs=int(args.mask_epochs),
        mask_batch_size=int(args.mask_batch_size),
        mask_train_records_per_epoch=int(args.mask_train_records_per_epoch),
        sae_latent_mult=int(args.sae_latent_mult),
        sae_lr=float(args.sae_lr),
        sae_l1=float(args.sae_l1),
        sae_epochs=int(args.sae_epochs),
        seed=int(args.seed),
    )

    results_by_method: dict[str, dict[str, object]] = {}
    requested_methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for method in requested_methods:
        method_start = time.perf_counter()
        featurizer_cache = {site.key(): fit_featurizer(method, site, fit_examples, run_cache=run_cache, hidden_size=int(args.hidden_size), config=config) for site in sites}
        method_trials = []
        per_row = {}
        calib_sens = []
        calib_inv = []
        test_sens = []
        test_inv = []
        for row_key in row_keys:
            row_best = None
            row_best_key = None
            for site in sites:
                featurizer = featurizer_cache[site.key()]
                if method == "full_vector":
                    candidates = [(torch.ones(featurizer.feature_dim, dtype=torch.float32), None, None)]
                else:
                    candidates = []
                    for lr in config.mask_lrs:
                        for l1_coeff in config.mask_l1s:
                            learned_mask = train_dbm_mask(
                                model,
                                banks["fit_by_row"][row_key],
                                site,
                                featurizer,
                                lr=float(lr),
                                l1_coeff=float(l1_coeff),
                                config=config,
                                device=device,
                                run_cache=run_cache,
                                seed=int(args.seed),
                            )
                            candidates.append((learned_mask, float(lr), float(l1_coeff)))
                for mask, lr, l1_coeff in candidates:
                    for lambda_scale in config.lambda_grid:
                        sens = exact_match_rate(
                            model,
                            banks["calib_positive_by_row"][row_key],
                            site,
                            featurizer,
                            mask=mask,
                            lambda_scale=float(lambda_scale),
                            device=device,
                            run_cache=run_cache,
                            batch_size=int(config.mask_batch_size),
                        )
                        inv = exact_match_rate(
                            model,
                            banks["calib_invariant_by_row"][row_key],
                            site,
                            featurizer,
                            mask=mask,
                            lambda_scale=float(lambda_scale),
                            device=device,
                            run_cache=run_cache,
                            batch_size=int(config.mask_batch_size),
                        )
                        combined = 0.5 * (sens + inv)
                        key = _calibration_key(
                            combined=combined,
                            sensitivity=sens,
                            invariance=inv,
                            selection_rule=str(config.selection_rule),
                            invariance_floor=float(config.invariance_floor),
                        )
                        trial = {
                            "row_key": row_key,
                            "site_key": site.key(),
                            "lambda": float(lambda_scale),
                            "mask_summary": _mask_summary(mask),
                            "mask_lr": lr,
                            "mask_l1": l1_coeff,
                            "calibration": {
                                "sensitivity": float(sens),
                                "invariance": float(inv),
                                "combined": float(combined),
                                "count_positive": int(len(banks["calib_positive_by_row"][row_key])),
                                "count_invariant": int(len(banks["calib_invariant_by_row"][row_key])),
                                "selection_rule": str(config.selection_rule),
                                "invariance_floor": float(config.invariance_floor),
                            },
                        }
                        method_trials.append(trial)
                        if row_best_key is None or key > row_best_key:
                            row_best_key = key
                            row_best = trial

            chosen_site = next(site for site in sites if site.key() == row_best["site_key"])
            chosen_mask = torch.tensor(row_best["mask_summary"]["mask_values"], dtype=torch.float32)
            test_eval = {
                "sensitivity": exact_match_rate(
                    model,
                    banks["test_positive_by_row"][row_key],
                    chosen_site,
                    featurizer_cache[chosen_site.key()],
                    mask=chosen_mask,
                    lambda_scale=float(row_best["lambda"]),
                    device=device,
                    run_cache=run_cache,
                    batch_size=int(config.mask_batch_size),
                ),
                "invariance": exact_match_rate(
                    model,
                    banks["test_invariant_by_row"][row_key],
                    chosen_site,
                    featurizer_cache[chosen_site.key()],
                    mask=chosen_mask,
                    lambda_scale=float(row_best["lambda"]),
                    device=device,
                    run_cache=run_cache,
                    batch_size=int(config.mask_batch_size),
                ),
            }
            test_eval["combined"] = 0.5 * (float(test_eval["sensitivity"]) + float(test_eval["invariance"]))
            per_row[row_key] = {
                "site_key": chosen_site.key(),
                "lambda": float(row_best["lambda"]),
                "mask_lr": row_best["mask_lr"],
                "mask_l1": row_best["mask_l1"],
                "mask_summary": row_best["mask_summary"],
                "calibration": row_best["calibration"],
                "test": test_eval,
            }
            calib_sens.append(float(row_best["calibration"]["sensitivity"]))
            calib_inv.append(float(row_best["calibration"]["invariance"]))
            test_sens.append(float(test_eval["sensitivity"]))
            test_inv.append(float(test_eval["invariance"]))

        elapsed = time.perf_counter() - method_start
        results_by_method[method] = {
            "method": method,
            "config": config.as_dict(),
            "runtime_sec": float(elapsed),
            "sites": [site.key() for site in sites],
            "trials": method_trials,
            "calibration": {
                "mean_sensitivity": float(sum(calib_sens) / max(1, len(calib_sens))),
                "mean_invariance": float(sum(calib_inv) / max(1, len(calib_inv))),
                "mean_combined": float((sum(calib_sens) + sum(calib_inv)) / max(1, 2 * len(calib_sens))),
            },
            "test": {
                "per_row": per_row,
                "mean_sensitivity": float(sum(test_sens) / max(1, len(test_sens))),
                "mean_invariance": float(sum(test_inv) / max(1, len(test_inv))),
                "mean_combined": float((sum(test_sens) + sum(test_inv)) / max(1, 2 * len(test_sens))),
                "carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, carry_keys),
                "output_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, output_keys) if output_keys else None,
            },
        }

    result = {
        "config": vars(args),
        "row_keys": row_keys,
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "bank_summaries": {
            "fit_by_row": _bank_summaries(
                banks["fit_by_row"],
                {key: tuple(rec for rec in banks["fit_by_row"][key] if rec.is_active) for key in row_keys},
                {key: tuple(rec for rec in banks["fit_by_row"][key] if not rec.is_active) for key in row_keys},
            ),
            "calib_by_row": _bank_summaries(
                {key: tuple(banks["calib_positive_by_row"][key] + banks["calib_invariant_by_row"][key]) for key in row_keys},
                banks["calib_positive_by_row"],
                banks["calib_invariant_by_row"],
            ),
            "test_by_row": _bank_summaries(
                {key: tuple(banks["test_positive_by_row"][key] + banks["test_invariant_by_row"][key]) for key in row_keys},
                banks["test_positive_by_row"],
                banks["test_invariant_by_row"],
            ),
        },
        "training": train_summary,
        "methods": results_by_method,
    }
    summary_path = out_dir / "joint_endogenous_mib_baselines_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    compact = {
        "summary": str(summary_path),
        "factual_exact_all": result["factual_exact"]["all"],
        "method_summaries": {
            method: {
                "runtime_sec": method_result["runtime_sec"],
                "carry_subset": method_result["test"]["carry_subset"],
            }
            for method, method_result in results_by_method.items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
