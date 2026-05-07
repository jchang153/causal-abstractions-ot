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
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import (
    _build_banks,
    _default_resolutions,
    _family_order,
    _fit_cost_matrix,
    _load_or_train_model,
    _parse_floats,
    _parse_ints,
    _parse_selection_profiles,
    _row_specs,
    _subset_summary,
)
from experiments.binary_addition_rnn.sites import enumerate_group_sites_for_timesteps, enumerate_output_logit_sites
from experiments.binary_addition_rnn.transport import (
    TransportConfig,
    enumerate_transport_row_candidates,
    evaluate_single_calibrated_transport,
    select_transport_calibration_candidate,
    sinkhorn_uniform_ot,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Frozen shared OT discovery runner for transport-guided DAS.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--abstract-mode", type=str, default="all_endogenous", choices=["carries_only", "all_endogenous"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=8)
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
    ap.add_argument("--source-policy", type=str, default="structured_26_top3carry_c2x5_c3x7_no_random")
    ap.add_argument("--normalize-signatures", action="store_true")
    ap.add_argument("--fit-signature-mode", type=str, default="all", choices=["all", "active_only"])
    ap.add_argument("--fit-stratify-mode", type=str, default="none", choices=["none", "source_propagation", "row_counterfactual"])
    ap.add_argument("--fit-family-profile", type=str, default="all")
    ap.add_argument("--cost-metric", type=str, default="sq_l2", choices=["sq_l2", "l1", "cosine"])
    ap.add_argument("--ot-epsilons", type=str, default="0.01,0.03")
    ap.add_argument("--top-k-grid", type=str, default="1,2")
    ap.add_argument("--lambda-grid", type=str, default="0.5,1,2")
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--selection-profiles", type=str, default="combined:0.0")
    ap.add_argument("--batch-size", type=int, default=512)
    return ap.parse_args()


def _default_discovery_resolutions(hidden_size: int) -> tuple[int, ...]:
    out = [int(hidden_size)]
    if int(hidden_size) % 2 == 0:
        half = int(hidden_size) // 2
        if half not in out:
            out.append(half)
    return tuple(out)


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
    internal_carry_keys = [f"C{i}" for i in range(1, int(args.width)) if f"C{i}" in row_keys]
    output_keys = [f"S{i}" for i in range(int(args.width)) if f"S{i}" in row_keys]
    family_order = ("all_source",) if args.source_policy == "all_source" else _family_order(int(args.width), str(args.source_policy))
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(args.seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )

    transport_cfg = TransportConfig(
        epsilon_grid=_parse_floats(args.ot_epsilons),
        beta_grid=(0.1,),
        topk_grid=_parse_ints(args.top_k_grid),
        lambda_grid=_parse_floats(args.lambda_grid),
        sinkhorn_iters=int(args.sinkhorn_iters),
        selection_rule="combined",
        invariance_floor=0.0,
    )
    timesteps = _parse_ints(args.timesteps)
    resolutions = _parse_ints(args.resolutions) if str(args.resolutions).strip() else _default_discovery_resolutions(int(args.hidden_size))
    selection_profiles = _parse_selection_profiles(args.selection_profiles, default_rule="combined", default_floor=0.0)
    profile_keys = [f"{rule}__floor_{floor:g}" for rule, floor in selection_profiles]

    per_resolution = []
    for resolution in resolutions:
        hidden_sites = enumerate_group_sites_for_timesteps(
            timesteps=timesteps,
            hidden_size=int(args.hidden_size),
            resolution=int(resolution),
        )
        output_sites = enumerate_output_logit_sites(output_dim=int(args.width) + 1)
        sites = tuple(hidden_sites) + tuple(output_sites)
        cost, _diagnostics = _fit_cost_matrix(
            model,
            specs=specs,
            fit_by_row=banks["fit_by_row"],
            sites=sites,
            family_order=family_order,
            device=device,
            run_cache=run_cache,
            batch_size=int(args.batch_size),
            normalize_signatures=bool(args.normalize_signatures),
            fit_signature_mode=str(args.fit_signature_mode),
            fit_stratify_mode=str(args.fit_stratify_mode),
            fit_family_profile=str(args.fit_family_profile),
            cost_metric=str(args.cost_metric),
        )

        trials = []
        for epsilon in transport_cfg.epsilon_grid:
            coupling = sinkhorn_uniform_ot(
                cost,
                epsilon=float(epsilon),
                n_iter=int(transport_cfg.sinkhorn_iters),
                temperature=float(transport_cfg.temperature),
            )
            profile_results = {}
            for profile_key, (selection_rule, invariance_floor) in zip(profile_keys, selection_profiles):
                per_row = {}
                calib_sens = []
                calib_inv = []
                test_sens = []
                test_inv = []
                for row_idx, row_key in enumerate(row_keys):
                    candidates = enumerate_transport_row_candidates(
                        model,
                        coupling[row_idx],
                        sites,
                        banks["calib_positive_by_row"][row_key],
                        banks["calib_invariant_by_row"][row_key],
                        transport_cfg,
                        device=device,
                        run_cache=run_cache,
                    )
                    calibrated = select_transport_calibration_candidate(
                        candidates,
                        selection_rule=str(selection_rule),
                        invariance_floor=float(invariance_floor),
                    )
                    tested = evaluate_single_calibrated_transport(
                        model,
                        calibrated,
                        sites,
                        banks["test_positive_by_row"][row_key],
                        banks["test_invariant_by_row"][row_key],
                        device=device,
                        run_cache=run_cache,
                    )
                    per_row[row_key] = {
                        "calibration": calibrated["calibration"],
                        "test": tested,
                        "top_k": int(calibrated["top_k"]),
                        "lambda": float(calibrated["lambda"]),
                        "selected_sites": calibrated["selected_sites"],
                    }
                    calib_sens.append(float(calibrated["calibration"]["sensitivity"]))
                    calib_inv.append(float(calibrated["calibration"]["invariance"]))
                    test_sens.append(float(tested["sensitivity"]))
                    test_inv.append(float(tested["invariance"]))

                profile_results[profile_key] = {
                    "per_row": per_row,
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
                        "internal_carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, internal_carry_keys) if internal_carry_keys else None,
                        "output_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, output_keys) if output_keys else None,
                    },
                }

            trials.append(
                {
                    "config": {"method": "ot", "resolution": int(resolution), "epsilon": float(epsilon)},
                    "coupling": coupling.tolist(),
                    "profile_results": profile_results,
                }
            )

        per_resolution.append(
            {
                "resolution": int(resolution),
                "sites": [site.key() for site in sites],
                "trials": trials,
            }
        )

    result = {
        "config": vars(args),
        "row_keys": row_keys,
        "family_order": list(family_order),
        "selection_profiles": [
            {"profile_key": key, "selection_rule": rule, "invariance_floor": floor}
            for key, (rule, floor) in zip(profile_keys, selection_profiles)
        ],
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "training": train_summary,
        "per_resolution": per_resolution,
    }
    summary_path = out_dir / "shared_ot_discovery_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "factual_exact_all": result["factual_exact"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
