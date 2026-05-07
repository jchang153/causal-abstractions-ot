from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.interventions import build_run_cache
from experiments.binary_addition.run_progressive_plot import (
    _build_banks,
    _checkpoint_map,
    _dedupe_sites,
    _family_order,
    _load_or_train_model,
    _make_transport_config,
    _parse_floats,
    _parse_ints,
    _parse_rows,
    _row_specs,
    _run_ot_stage,
    _summary_stats,
)
from experiments.binary_addition.sites import enumerate_group_sites_for_timesteps


ROW_KEYS_DEFAULT = "C1,C2,C3"
SUMMARY_NAME = "resolution_separate_topk_seed_summary.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run only Stage-B canonical PLOT after a cached progressive binary-addition run. "
            "Stage A is reused from the previous run; each requested resolution gets its own "
            "site menu and top-k grid, and final handles are selected per row by calibration."
        )
    )
    ap.add_argument("--base-run-dir", type=str, default=str(Path("eval") / "progressive_plot_10seed"))
    ap.add_argument("--out-dir", type=str, default=str(Path("eval") / "progressive_plot_resolution_topk"))
    ap.add_argument("--hidden-size", type=int, required=True, choices=[8, 16])
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--rows", type=str, default=ROW_KEYS_DEFAULT)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--source-policy", type=str, default="structured_26_top3carry_c2x5_c3x7_no_random")
    ap.add_argument("--stage-b-epsilons", type=str, default="0.003,0.01,0.03,0.1")
    ap.add_argument("--ot-lambda-grid", type=str, default="0.25,0.5,1,2,4,8")
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--selection-rule", type=str, default="combined")
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--fit-signature-mode", type=str, default="all")
    ap.add_argument("--fit-family-profile", type=str, default="all")
    ap.add_argument("--fit-stratify-mode", type=str, default="none")
    ap.add_argument("--cost-metric", type=str, default="sq_l2", choices=["sq_l2", "l1", "cosine"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--resolutions",
        type=str,
        default="1,2",
        help="Canonical coordinate-group resolutions to sweep separately, e.g. 1,2 or 1,2,4.",
    )
    ap.add_argument(
        "--resolution-top-k",
        type=str,
        default="",
        help=(
            "Optional per-resolution top-k map, e.g. '1:1-16;2:1-8'. "
            "Defaults to k=1..hidden_size/resolution for each resolution."
        ),
    )
    ap.add_argument(
        "--select-resolutions",
        type=str,
        default="",
        help="Optional subset of swept resolutions used for final per-row selection. Defaults to all swept resolutions.",
    )
    ap.add_argument("--checkpoint-map", type=str, default="")
    return ap.parse_args()


def _parse_checkpoint_map(text: str, hidden_size: int) -> dict[int, str]:
    if not str(text).strip():
        return _checkpoint_map(int(hidden_size))
    mapping: dict[int, str] = {}
    for chunk in str(text).split(";"):
        if not chunk.strip():
            continue
        key, value = chunk.split("=", 1)
        mapping[int(key.strip())] = value.strip()
    return mapping


def _parse_top_k_values(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_text, hi_text = chunk.split("-", 1)
            lo = int(lo_text.strip())
            hi = int(hi_text.strip())
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(chunk))
    return tuple(dict.fromkeys(values))


def _resolution_top_k_map(text: str, resolutions: Sequence[int], hidden_size: int) -> dict[int, tuple[int, ...]]:
    if not str(text).strip():
        return {
            int(resolution): tuple(range(1, int(hidden_size) // int(resolution) + 1))
            for resolution in resolutions
        }
    out: dict[int, tuple[int, ...]] = {}
    for chunk in str(text).split(";"):
        if not chunk.strip():
            continue
        key_text, value_text = chunk.split(":", 1)
        out[int(key_text.strip())] = _parse_top_k_values(value_text)
    missing = [int(resolution) for resolution in resolutions if int(resolution) not in out]
    if missing:
        raise ValueError(f"missing top-k map for resolutions: {missing}")
    return out


def _base_seed_summary_path(base_run_dir: Path, hidden_size: int, seed: int) -> Path:
    path = base_run_dir / f"h{hidden_size}" / f"seed_{seed}" / "progressive_seed_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _select_row_from_resolution_trials(
    resolution_results: dict[str, object],
    row_key: str,
    *,
    select_resolutions: set[str],
) -> tuple[str, dict[str, object], dict[str, object]]:
    best = None
    best_key = None
    for resolution, stage in resolution_results.items():
        if str(resolution) not in select_resolutions:
            continue
        for trial in stage["trials"]:
            row = trial["test"]["per_row"][row_key]
            calibration = row["calibration"]
            key = (
                float(calibration["combined"]),
                float(calibration["sensitivity"]),
                float(calibration["invariance"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (str(resolution), trial, row)
    if best is None:
        raise RuntimeError(f"no candidate found for row {row_key}")
    return best


def _run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint: str,
    base_summary_path: Path,
    out_seed_dir: Path,
    resolutions: tuple[int, ...],
    resolution_top_k: dict[int, tuple[int, ...]],
    select_resolutions: set[str],
) -> dict[str, object]:
    old = json.loads(base_summary_path.read_text(encoding="utf-8"))
    old_config = dict(old["config"])
    row_keys = _parse_rows(str(args.rows))
    stage_a_timesteps = {key: int(value) for key, value in old["stage_a_timesteps"].items() if key in row_keys}
    row_allowed_timesteps = {row_key: {int(stage_a_timesteps[row_key])} for row_key in row_keys}
    selected_timesteps = tuple(sorted(set(stage_a_timesteps.values())))
    stage_a_runtime = float(old["stages"]["stage_a"]["runtime_seconds"])

    model_args = argparse.Namespace(**old_config)
    model_args.seed = int(seed)
    model_args.model_checkpoint = str(checkpoint)
    model_args.device = str(args.device)
    model_args.hidden_size = int(args.hidden_size)
    model_args.width = int(args.width)
    model_args.rows = str(args.rows)
    model_args.source_policy = str(args.source_policy)
    model_args.selection_rule = str(args.selection_rule)
    model_args.invariance_floor = float(args.invariance_floor)
    model_args.normalize_signatures = True
    model_args.fit_signature_mode = str(args.fit_signature_mode)
    model_args.fit_family_profile = str(args.fit_family_profile)
    model_args.fit_stratify_mode = str(args.fit_stratify_mode)
    model_args.cost_metric = str(args.cost_metric)
    model_args.sinkhorn_iters = int(args.sinkhorn_iters)
    model_args.das_batch_size = int(args.batch_size)

    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    examples = enumerate_all_examples(width=int(args.width))
    split = stratified_base_split(
        examples,
        fit_count=int(model_args.fit_bases),
        calib_count=int(model_args.calib_bases),
        test_count=int(model_args.test_bases),
        seed=int(seed),
    )
    model, _train_summary = _load_or_train_model(model_args, examples, split)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    run_cache = build_run_cache(model, examples, device=device)

    specs_all = _row_specs("all_endogenous", int(args.width))
    specs = tuple(spec for spec in specs_all if spec.key in set(row_keys))
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )
    family_order = _family_order(int(args.width), str(args.source_policy))

    resolution_results = {}
    for resolution in resolutions:
        transport_cfg = _make_transport_config(
            epsilons=_parse_floats(str(args.stage_b_epsilons)),
            top_k_grid=tuple(int(k) for k in resolution_top_k[int(resolution)]),
            lambda_grid=_parse_floats(str(args.ot_lambda_grid)),
            sinkhorn_iters=int(args.sinkhorn_iters),
        )
        sites = _dedupe_sites(
            enumerate_group_sites_for_timesteps(
                timesteps=selected_timesteps,
                hidden_size=int(args.hidden_size),
                resolution=int(resolution),
            )
        )
        resolution_results[str(resolution)] = _run_ot_stage(
            stage_name=f"stage_b_canonical_r{resolution}_topk_ot_inside_cached_stage_a_timesteps",
            model=model,
            specs=specs,
            row_keys=row_keys,
            banks=banks,
            sites=sites,
            family_order=family_order,
            transport_cfg=transport_cfg,
            selection_rule=str(args.selection_rule),
            invariance_floor=float(args.invariance_floor),
            device=device,
            run_cache=run_cache,
            batch_size=int(args.batch_size),
            normalize_signatures=True,
            fit_signature_mode=str(args.fit_signature_mode),
            fit_stratify_mode=str(args.fit_stratify_mode),
            fit_family_profile=str(args.fit_family_profile),
            cost_metric=str(args.cost_metric),
            rotation_map=None,
            row_allowed_timesteps=row_allowed_timesteps,
        )

    sens_rows = []
    inv_rows = []
    selected_resolution_counts: dict[str, int] = {}
    selected_top_k_counts: dict[str, int] = {}
    selected_epsilon_counts: dict[str, int] = {}
    per_row = {}
    for row_key in row_keys:
        resolution, trial, row = _select_row_from_resolution_trials(
            resolution_results,
            row_key,
            select_resolutions=select_resolutions,
        )
        selected_resolution_counts[resolution] = selected_resolution_counts.get(resolution, 0) + 1
        selected_top_k_counts[str(row["top_k"])] = selected_top_k_counts.get(str(row["top_k"]), 0) + 1
        selected_epsilon_counts[str(trial["epsilon"])] = selected_epsilon_counts.get(str(trial["epsilon"]), 0) + 1
        sens_rows.append(float(row["test"]["sensitivity"]))
        inv_rows.append(float(row["test"]["invariance"]))
        per_row[row_key] = {
            "stage_a_timestep": int(stage_a_timesteps[row_key]),
            "resolution": int(resolution),
            "epsilon": float(trial["epsilon"]),
            "top_k": int(row["top_k"]),
            "lambda": float(row["lambda"]),
            "combined": float(row["test"]["combined"]),
            "sensitivity": float(row["test"]["sensitivity"]),
            "invariance": float(row["test"]["invariance"]),
            "selected_sites": [str(site["site_key"]) for site in row.get("selected_sites", [])],
        }

    mean_sensitivity = sum(sens_rows) / len(sens_rows)
    mean_invariance = sum(inv_rows) / len(inv_rows)
    mean_combined = 0.5 * (mean_sensitivity + mean_invariance)
    stage_b_sweep_runtime = sum(float(stage["runtime_seconds"]) for stage in resolution_results.values())
    total_runtime = stage_a_runtime + stage_b_sweep_runtime

    result = {
        "config": {
            "base_run": str(base_summary_path.parent.parent.parent),
            "hidden_size": int(args.hidden_size),
            "seed": int(seed),
            "rows": list(row_keys),
            "stage_a_source": str(base_summary_path),
            "stage_a_timesteps": stage_a_timesteps,
            "resolution_top_k": {str(k): list(v) for k, v in resolution_top_k.items()},
            "select_resolutions": sorted(select_resolutions, key=int),
            "stage_b_epsilons": list(_parse_floats(str(args.stage_b_epsilons))),
            "ot_lambda_grid": list(_parse_floats(str(args.ot_lambda_grid))),
            "sinkhorn_iters": int(args.sinkhorn_iters),
            "selection_rule": str(args.selection_rule),
            "cost_metric": str(args.cost_metric),
            "selection": "per carry across selected separate-resolution trials by calibration combined/sensitivity/invariance",
        },
        "stage_a_runtime_seconds_cached": float(stage_a_runtime),
        "stage_b_sweep_runtime_seconds": float(stage_b_sweep_runtime),
        "total_runtime_seconds_with_cached_stage_a": float(total_runtime),
        "mean_combined": float(mean_combined),
        "mean_sensitivity": float(mean_sensitivity),
        "mean_invariance": float(mean_invariance),
        "per_row": per_row,
        "selected_resolution_counts": selected_resolution_counts,
        "selected_top_k_counts": selected_top_k_counts,
        "selected_epsilon_counts": selected_epsilon_counts,
        "resolution_results": resolution_results,
    }
    out_seed_dir.mkdir(parents=True, exist_ok=True)
    (out_seed_dir / SUMMARY_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _merge_counts(items: Sequence[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            out[str(key)] = out.get(str(key), 0) + int(value)
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def main() -> None:
    args = parse_args()
    seeds = _parse_ints(str(args.seeds))
    resolutions = _parse_ints(str(args.resolutions))
    resolution_top_k = _resolution_top_k_map(str(args.resolution_top_k), resolutions, int(args.hidden_size))
    select_resolutions = (
        {str(int(value)) for value in _parse_ints(str(args.select_resolutions))}
        if str(args.select_resolutions).strip()
        else {str(int(value)) for value in resolutions}
    )

    checkpoint_map = _parse_checkpoint_map(str(args.checkpoint_map), int(args.hidden_size))
    base_run_dir = Path(args.base_run_dir).resolve()
    out_h_dir = Path(args.out_dir).resolve() / f"h{args.hidden_size}"
    out_h_dir.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    seed_results = []
    for seed in seeds:
        if int(seed) not in checkpoint_map:
            raise ValueError(f"missing checkpoint for seed {seed}")
        base_summary_path = _base_seed_summary_path(base_run_dir, int(args.hidden_size), int(seed))
        result = _run_seed(
            args,
            seed=int(seed),
            checkpoint=checkpoint_map[int(seed)],
            base_summary_path=base_summary_path,
            out_seed_dir=out_h_dir / f"seed_{seed}",
            resolutions=resolutions,
            resolution_top_k=resolution_top_k,
            select_resolutions=select_resolutions,
        )
        seed_results.append(result)
        per_seed[str(seed)] = {
            "summary_path": str(out_h_dir / f"seed_{seed}" / SUMMARY_NAME),
            "stage_a_timesteps": result["config"]["stage_a_timesteps"],
            "mean_combined": result["mean_combined"],
            "mean_sensitivity": result["mean_sensitivity"],
            "mean_invariance": result["mean_invariance"],
            "stage_b_sweep_runtime_seconds": result["stage_b_sweep_runtime_seconds"],
            "total_runtime_seconds_with_cached_stage_a": result["total_runtime_seconds_with_cached_stage_a"],
            "per_row": result["per_row"],
        }
        print(
            json.dumps(
                {
                    "seed": int(seed),
                    "accuracy": float(result["mean_combined"]),
                    "stage_b_sweep_seconds": float(result["stage_b_sweep_runtime_seconds"]),
                    "total_seconds": float(result["total_runtime_seconds_with_cached_stage_a"]),
                }
            )
        )

    aggregate = {
        "config": {
            "base_run_dir": str(base_run_dir),
            "hidden_size": int(args.hidden_size),
            "seeds": list(seeds),
            "rows": list(_parse_rows(str(args.rows))),
            "stage_a": "reused from previous run; not recomputed",
            "resolution_top_k": {str(k): list(v) for k, v in resolution_top_k.items()},
            "select_resolutions": sorted(select_resolutions, key=int),
            "stage_b_epsilons": list(_parse_floats(str(args.stage_b_epsilons))),
            "ot_lambda_grid": list(_parse_floats(str(args.ot_lambda_grid))),
            "selection_rule": str(args.selection_rule),
            "cost_metric": str(args.cost_metric),
        },
        "per_seed": per_seed,
        "accuracy": _summary_stats([float(result["mean_combined"]) for result in seed_results]),
        "sensitivity": _summary_stats([float(result["mean_sensitivity"]) for result in seed_results]),
        "invariance": _summary_stats([float(result["mean_invariance"]) for result in seed_results]),
        "stage_b_sweep_runtime_seconds": _summary_stats(
            [float(result["stage_b_sweep_runtime_seconds"]) for result in seed_results]
        ),
        "total_runtime_seconds_with_cached_stage_a": _summary_stats(
            [float(result["total_runtime_seconds_with_cached_stage_a"]) for result in seed_results]
        ),
        "selected_resolution_counts": _merge_counts(
            [result["selected_resolution_counts"] for result in seed_results]
        ),
        "selected_top_k_counts": _merge_counts([result["selected_top_k_counts"] for result in seed_results]),
        "selected_epsilon_counts": _merge_counts([result["selected_epsilon_counts"] for result in seed_results]),
    }
    aggregate_path = out_h_dir / "resolution_topk_perrow_selected_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(aggregate_path), "accuracy": aggregate["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
