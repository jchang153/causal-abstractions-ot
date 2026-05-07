from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.das import RotatedSubspace
from experiments.binary_addition_rnn.interventions import RunCache, build_run_cache
from experiments.binary_addition_rnn.model import GRUAdder, TrainConfig, exact_accuracy, train_backbone
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import (
    EndogenousPairRecord,
    _bank_summaries,
    _build_banks,
    _default_resolutions,
    _row_specs,
    _subset_summary,
)
from experiments.binary_addition_rnn.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition_rnn.sites import (
    CoordinateGroupSite,
    FullStateSite,
    NeuronSite,
    OutputLogitSite,
    Site,
    enumerate_group_sites_for_timesteps,
    enumerate_output_logit_sites,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DAS baseline on the structured/endogenous binary addition benchmark.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--abstract-mode", type=str, default="all_endogenous", choices=["carries_only", "all_endogenous"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=4)
    ap.add_argument("--timesteps", type=str, default="0,1,2,3")
    ap.add_argument("--resolutions", type=str, default="")
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
        default="structured_13",
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
    ap.add_argument("--fit-bank-mode", type=str, default="shared", choices=["shared", "anchored_prefix"])
    ap.add_argument("--rows", type=str, default="")
    ap.add_argument("--selection-rule", type=str, default="combined", choices=["combined", "sensitivity_only", "sensitivity_then_invariance"])
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--lambda-grid", type=str, default="0.5,1,2,4")
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
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _load_or_train_model(
    args: argparse.Namespace,
    examples,
    split,
) -> tuple[GRUAdder, dict[str, object]]:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_text = str(getattr(args, "model_checkpoint", "")).strip()
    if checkpoint_text:
        checkpoint_path = Path(checkpoint_text).resolve()
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=device)
            train_cfg = dict(payload.get("train_config", {}))
            model = GRUAdder(
                width=int(train_cfg.get("width", args.width)),
                input_size=int(train_cfg.get("input_size", 2)),
                hidden_size=int(train_cfg.get("hidden_size", args.hidden_size)),
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            return model, {
                "loaded_checkpoint": str(checkpoint_path),
                "train_config": train_cfg,
            }

    train_examples = examples if args.train_on == "all" else split.fit
    cfg = TrainConfig(
        width=int(args.width),
        hidden_size=int(args.hidden_size),
        batch_size=int(args.train_batch_size),
        epochs=int(args.train_epochs),
        learning_rate=float(args.train_lr),
        seed=int(args.seed),
        device=str(args.device),
    )
    model, train_summary = train_backbone(cfg, train_examples=train_examples, eval_examples=examples)
    if checkpoint_text:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "train_config": cfg.as_dict()}, checkpoint_path)
        train_summary = dict(train_summary)
        train_summary["saved_checkpoint"] = str(checkpoint_path)
    return model, train_summary


def _site_dim(site: Site, hidden_size: int) -> int:
    if isinstance(site, FullStateSite):
        return int(hidden_size)
    if isinstance(site, CoordinateGroupSite):
        return int(len(site.coord_indices))
    if isinstance(site, NeuronSite):
        return 1
    if isinstance(site, OutputLogitSite):
        return 1
    raise TypeError(f"unsupported site type: {type(site)!r}")


def _site_chunk(h: torch.Tensor, site: Site) -> torch.Tensor:
    if isinstance(site, FullStateSite):
        return h
    if isinstance(site, CoordinateGroupSite):
        return h[:, list(site.coord_indices)]
    if isinstance(site, NeuronSite):
        return h[:, [int(site.neuron_index)]]
    raise TypeError(f"hidden-state chunk requested for unsupported site type: {type(site)!r}")


def _replace_site_chunk(h: torch.Tensor, site: Site, updated_chunk: torch.Tensor) -> torch.Tensor:
    if isinstance(site, FullStateSite):
        return updated_chunk
    if isinstance(site, CoordinateGroupSite):
        out = h.clone()
        out[:, list(site.coord_indices)] = updated_chunk
        return out
    if isinstance(site, NeuronSite):
        out = h.clone()
        out[:, int(site.neuron_index)] = updated_chunk[:, 0]
        return out
    raise TypeError(f"hidden-state chunk replacement requested for unsupported site type: {type(site)!r}")


def _run_das_batch(
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    site: Site,
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache,
    rotator: RotatedSubspace | None,
) -> torch.Tensor:
    if not records:
        return torch.empty((0, model.width + 1), dtype=torch.float32)

    if isinstance(site, OutputLogitSite):
        base_logits = torch.stack([run_cache.get_run(rec.base).output_logits for rec in records], dim=0).to(device=device)
        source_logits = torch.stack([run_cache.get_run(rec.source).output_logits for rec in records], dim=0).to(device=device)
        idx = int(site.output_index)
        out = base_logits.clone()
        out[:, idx] = out[:, idx] + float(lambda_scale) * (source_logits[:, idx] - out[:, idx])
        return out

    base_x = torch.cat([run_cache.get_input(rec.base) for rec in records], dim=0).to(device=device)
    source_states = torch.stack([run_cache.get_run(rec.source).hidden_states for rec in records], dim=0).to(device=device)
    h = torch.zeros(base_x.size(0), model.hidden_size, device=device, dtype=base_x.dtype)
    sum_logits = []
    target_step = int(site.timestep)
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        if step == target_step:
            base_chunk = _site_chunk(h, site)
            source_chunk = _site_chunk(source_states[:, step, :], site)
            if rotator is None:
                updated = base_chunk + float(lambda_scale) * (source_chunk - base_chunk)
            else:
                updated = rotator.intervene(base_chunk, source_chunk, lambda_scale=float(lambda_scale))
            h = _replace_site_chunk(h, site, updated)
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    return torch.cat(sum_logits + [carry_logit], dim=1)


def _exact_match_rate(
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    site: Site,
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache,
    rotator: RotatedSubspace | None,
    batch_size: int,
) -> float:
    if not records:
        return 0.0
    hits = 0
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start : start + int(batch_size)]
        logits = _run_das_batch(
            model,
            batch_records,
            site,
            lambda_scale=float(lambda_scale),
            device=device,
            run_cache=run_cache,
            rotator=rotator,
        )
        pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()
        tgt = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch_records], dtype=torch.int64)
        hits += int((pred == tgt).all(dim=1).sum().item())
    return float(hits / len(records))


def _train_rotator(
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    site: Site,
    *,
    subspace_dim: int,
    lr: float,
    epochs: int,
    train_records_per_epoch: int,
    batch_size: int,
    device: torch.device,
    run_cache: RunCache,
    seed: int,
) -> RotatedSubspace:
    site_dim = _site_dim(site, model.hidden_size)
    if site_dim <= 1 or isinstance(site, OutputLogitSite):
        raise ValueError("rotator training is only valid for hidden sites with dimension > 1")
    rotator = RotatedSubspace(site_dim, subspace_dim=int(subspace_dim)).to(device)
    optimizer = torch.optim.Adam(rotator.parameters(), lr=float(lr))
    criterion = nn.BCEWithLogitsLoss()
    rng = random.Random(int(seed) + 17 * int(subspace_dim) + int(round(lr * 1e6)))
    records = list(records)
    for _ in range(int(epochs)):
        if len(records) > int(train_records_per_epoch):
            sampled = rng.sample(records, int(train_records_per_epoch))
        else:
            sampled = records
        for start in range(0, len(sampled), int(batch_size)):
            batch = sampled[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = _run_das_batch(
                model,
                batch,
                site,
                lambda_scale=1.0,
                device=device,
                run_cache=run_cache,
                rotator=rotator,
            )
            target = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch], dtype=torch.float32, device=device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
    return rotator.cpu()


def _calibration_key(*, combined: float, sensitivity: float, invariance: float, selection_rule: str, invariance_floor: float) -> tuple[float, float, float]:
    admissible = 1.0 if float(invariance) >= float(invariance_floor) else 0.0
    if selection_rule == "combined":
        return (admissible, float(combined), float(sensitivity))
    if selection_rule == "sensitivity_only":
        return (1.0, float(sensitivity), float(invariance))
    if selection_rule == "sensitivity_then_invariance":
        return (admissible, float(sensitivity), float(invariance))
    raise ValueError(f"unknown selection_rule: {selection_rule!r}")


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
    requested_rows = _parse_keys(args.rows)
    if requested_rows:
        requested_set = set(requested_rows)
        specs = [spec for spec in specs if spec.key in requested_set]
    row_keys = [spec.key for spec in specs]
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(args.seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )

    timesteps = _parse_ints(args.timesteps)
    resolutions = _parse_ints(args.resolutions) if str(args.resolutions).strip() else _default_resolutions(int(args.hidden_size))
    lambda_grid = _parse_floats(args.lambda_grid)
    requested_dims = _parse_ints(args.das_subspace_dims)
    requested_lrs = _parse_floats(args.das_lrs)
    carry_keys = [row_key for row_key in row_keys if row_key.startswith("C")]
    internal_carry_keys = [row_key for row_key in row_keys if row_key.startswith("C") and row_key != f"C{int(args.width)}"]
    output_keys = [row_key for row_key in row_keys if row_key.startswith("S")]

    per_resolution = []
    global_best = None
    global_best_key = None
    for resolution in resolutions:
        hidden_sites = enumerate_group_sites_for_timesteps(
            timesteps=timesteps,
            hidden_size=int(args.hidden_size),
            resolution=int(resolution),
        )
        output_sites = enumerate_output_logit_sites(output_dim=int(args.width) + 1)
        sites: tuple[Site, ...] = tuple(hidden_sites) + tuple(output_sites)

        per_row = {}
        calib_sens = []
        calib_inv = []
        test_sens = []
        test_inv = []
        trial_records = []
        for row_key in row_keys:
            fit_records = banks["fit_by_row"][row_key]
            if str(args.fit_bank_mode) == "anchored_prefix":
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
                fit_records = tuple(
                    rec
                    for rec in fit_records
                    if any(rec.family == prefix or rec.family.startswith(prefix + "_") for prefix in prefixes)
                )
            row_best = None
            row_best_key = None
            for site in sites:
                site_dim = _site_dim(site, int(args.hidden_size))
                if isinstance(site, OutputLogitSite) or site_dim == 1:
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
                        combined = 0.5 * (sens + inv)
                        key = _calibration_key(
                            combined=combined,
                            sensitivity=sens,
                            invariance=inv,
                            selection_rule=str(args.selection_rule),
                            invariance_floor=float(args.invariance_floor),
                        )
                        trial = {
                            "row_key": row_key,
                            "site_key": site.key(),
                            "lambda": float(lambda_scale),
                            "subspace_dim": int(subspace_dim) if subspace_dim is not None else site_dim,
                            "lr": float(lr) if lr is not None else None,
                            "calibration": {
                                "sensitivity": float(sens),
                                "invariance": float(inv),
                                "combined": float(combined),
                                "count_positive": int(len(banks["calib_positive_by_row"][row_key])),
                                "count_invariant": int(len(banks["calib_invariant_by_row"][row_key])),
                            },
                            "rotator_state": {k: v.tolist() for k, v in rotator.state_dict().items()} if rotator is not None else None,
                        }
                        trial_records.append(trial)
                        if row_best_key is None or key > row_best_key:
                            row_best_key = key
                            row_best = trial

            chosen_site = next(site for site in sites if site.key() == row_best["site_key"])
            if row_best["rotator_state"] is None or isinstance(chosen_site, OutputLogitSite) or _site_dim(chosen_site, int(args.hidden_size)) == 1:
                rotator = None
            else:
                rotator = RotatedSubspace(_site_dim(chosen_site, int(args.hidden_size)), int(row_best["subspace_dim"]))
                rotator.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in row_best["rotator_state"].items()})
            test_eval = {
                "sensitivity": _exact_match_rate(
                    model,
                    banks["test_positive_by_row"][row_key],
                    chosen_site,
                    lambda_scale=float(row_best["lambda"]),
                    device=device,
                    run_cache=run_cache,
                    rotator=rotator,
                    batch_size=int(args.das_batch_size),
                ),
                "invariance": _exact_match_rate(
                    model,
                    banks["test_invariant_by_row"][row_key],
                    chosen_site,
                    lambda_scale=float(row_best["lambda"]),
                    device=device,
                    run_cache=run_cache,
                    rotator=rotator,
                    batch_size=int(args.das_batch_size),
                ),
            }
            test_eval["combined"] = 0.5 * (float(test_eval["sensitivity"]) + float(test_eval["invariance"]))
            per_row[row_key] = {
                "calibration": row_best["calibration"],
                "test": test_eval,
                "site_key": chosen_site.key(),
                "lambda": float(row_best["lambda"]),
                "subspace_dim": int(row_best["subspace_dim"]),
                "lr": row_best["lr"],
            }
            calib_sens.append(float(row_best["calibration"]["sensitivity"]))
            calib_inv.append(float(row_best["calibration"]["invariance"]))
            test_sens.append(float(test_eval["sensitivity"]))
            test_inv.append(float(test_eval["invariance"]))

        resolution_result = {
            "resolution": int(resolution),
            "n_hidden_sites": int(len(hidden_sites)),
            "n_output_sites": int(len(output_sites)),
            "n_sites_total": int(len(sites)),
            "sites": [site.key() for site in sites],
            "trials": trial_records,
            "test": {
                "per_row": per_row,
                "mean_sensitivity": float(sum(test_sens) / max(1, len(test_sens))),
                "mean_invariance": float(sum(test_inv) / max(1, len(test_inv))),
                "mean_combined": float((sum(test_sens) + sum(test_inv)) / max(1, 2 * len(test_sens))),
                "carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, carry_keys),
                "internal_carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, internal_carry_keys) if internal_carry_keys else None,
                "output_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, output_keys) if output_keys else None,
            },
            "calibration": {
                "mean_sensitivity": float(sum(calib_sens) / max(1, len(calib_sens))),
                "mean_invariance": float(sum(calib_inv) / max(1, len(calib_inv))),
                "mean_combined": float((sum(calib_sens) + sum(calib_inv)) / max(1, 2 * len(calib_sens))),
            },
        }
        per_resolution.append(resolution_result)
        key = (
            float(resolution_result["calibration"]["mean_combined"]),
            float(resolution_result["calibration"]["mean_sensitivity"]),
            float(resolution_result["calibration"]["mean_invariance"]),
        )
        if global_best_key is None or key > global_best_key:
            global_best_key = key
            global_best = resolution_result

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
        "per_resolution": per_resolution,
        "best_result": global_best,
    }
    summary_path = out_dir / "joint_endogenous_das_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    compact = {
        "summary": str(summary_path),
        "factual_exact_all": result["factual_exact"]["all"],
        "best_resolution": None if global_best is None else global_best["resolution"],
        "best_test": None if global_best is None else global_best["test"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
