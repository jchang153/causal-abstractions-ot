from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.interventions import build_run_cache
from experiments.binary_addition.model import exact_accuracy
from experiments.binary_addition.run_progressive_plot import (
    _build_banks,
    _checkpoint_map,
    _family_order,
    _load_or_train_model,
    _make_transport_config,
    _parse_checkpoint_map,
    _parse_floats,
    _parse_ints,
    _row_specs,
    _run_alignment_resolution_sweep,
    _summary_stats,
)
from experiments.binary_addition.sites import enumerate_group_sites_for_timesteps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-stage binary-addition PLOT: independently partition every hidden timestep "
            "at each resolution, form a separate coupling, select by calibration, and test only "
            "the selected handle."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--checkpoint-map", default="")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--fit-bases", type=int, default=128)
    parser.add_argument("--calib-bases", type=int, default=64)
    parser.add_argument("--test-bases", type=int, default=64)
    parser.add_argument("--train-on", choices=["all", "fit_only"], default="all")
    parser.add_argument("--train-epochs", type=int, default=120)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-lr", type=float, default=0.02)
    parser.add_argument("--source-policy", default="structured_26_top3carry_c2x5_c3x7_no_random")
    parser.add_argument("--resolutions", default="1,2,4,8,16")
    parser.add_argument("--epsilons", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--top-k-grid", default="1,2,4")
    parser.add_argument("--lambda-grid", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--sinkhorn-iters", type=int, default=80)
    parser.add_argument("--selection-rule", default="combined")
    parser.add_argument("--invariance-floor", type=float, default=0.0)
    parser.add_argument("--fit-signature-mode", default="all")
    parser.add_argument("--fit-family-profile", default="all")
    parser.add_argument("--fit-stratify-mode", default="none")
    parser.add_argument("--cost-metric", choices=["sq_l2", "l1", "cosine"], default="sq_l2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_seed(args: argparse.Namespace, *, seed: int, checkpoint: str, out_dir: Path) -> dict[str, object]:
    seed_dir = out_dir / f"h{args.hidden_size}" / f"seed_{seed}"
    summary_path = seed_dir / "single_stage_plot_seed_summary.json"
    if args.skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text())
    seed_dir.mkdir(parents=True, exist_ok=True)

    device = _device(str(args.device))
    examples = enumerate_all_examples(width=int(args.width))
    split = stratified_base_split(
        examples,
        fit_count=int(args.fit_bases),
        calib_count=int(args.calib_bases),
        test_count=int(args.test_bases),
        seed=int(seed),
    )
    model_args = argparse.Namespace(**vars(args))
    model_args.seed = int(seed)
    model_args.model_checkpoint = str(checkpoint)
    model, training = _load_or_train_model(model_args, examples, split)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    run_cache = build_run_cache(model, examples, device=device)

    row_keys = ("C1", "C2", "C3")
    specs = tuple(spec for spec in _row_specs("all_endogenous", int(args.width)) if spec.key in row_keys)
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )
    family_order = _family_order(int(args.width), str(args.source_policy))
    resolutions = _parse_ints(str(args.resolutions))
    sites_by_resolution = {
        int(resolution): enumerate_group_sites_for_timesteps(
            timesteps=tuple(range(int(args.width))),
            hidden_size=int(args.hidden_size),
            resolution=int(resolution),
        )
        for resolution in resolutions
    }
    transport = _make_transport_config(
        epsilons=_parse_floats(str(args.epsilons)),
        top_k_grid=_parse_ints(str(args.top_k_grid)),
        lambda_grid=_parse_floats(str(args.lambda_grid)),
        sinkhorn_iters=int(args.sinkhorn_iters),
    )
    stage = _run_alignment_resolution_sweep(
        stage_name="single_stage_plot_all_timesteps",
        alignment_method="ot",
        model=model,
        specs=specs,
        row_keys=row_keys,
        banks=banks,
        sites_by_resolution=sites_by_resolution,
        family_order=family_order,
        transport_cfg=transport,
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
        cosine_temperature=1.0,
        bruteforce_temperature=1.0,
        rotation_map=None,
    )
    test = stage["best_trial"]["test"]["subset"]
    result = {
        "config": {**vars(args), "seed": int(seed), "checkpoint": str(checkpoint)},
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "training": training,
        "stage": stage,
        "method": {
            "name": "PLOT (single-stage)",
            "mean_combined": float(test["mean_combined"]),
            "mean_sensitivity": float(test["mean_sensitivity"]),
            "mean_invariance": float(test["mean_invariance"]),
            "runtime_seconds": float(stage["runtime_seconds"]),
            "selected_resolution_by_row": stage["selected_resolution_by_row"],
        },
    }
    summary_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_ints(str(args.seeds))
    checkpoints = _parse_checkpoint_map(str(args.checkpoint_map), int(args.hidden_size)) if args.checkpoint_map else _checkpoint_map(int(args.hidden_size))
    results = []
    per_seed = {}
    for seed in seeds:
        result = _run_seed(args, seed=int(seed), checkpoint=str(checkpoints[int(seed)]), out_dir=out_dir)
        results.append(result)
        per_seed[str(seed)] = result["method"]
        print(json.dumps({"seed": seed, **result["method"]}))

    aggregate = {
        "config": vars(args),
        "per_seed": per_seed,
        "accuracy": _summary_stats([float(result["method"]["mean_combined"]) for result in results]),
        "sensitivity": _summary_stats([float(result["method"]["mean_sensitivity"]) for result in results]),
        "invariance": _summary_stats([float(result["method"]["mean_invariance"]) for result in results]),
        "runtime_seconds": _summary_stats([float(result["method"]["runtime_seconds"]) for result in results]),
    }
    aggregate_path = out_dir / f"h{args.hidden_size}" / "single_stage_plot_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps({"summary": str(aggregate_path), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
