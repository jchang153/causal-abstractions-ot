"""
OT with PCA-rotated sites: intervene in the principal-component basis instead of
the coordinate basis, then run the same Sinkhorn OT as the shared-OT sweep.

PCA is fit on the hidden states of the fit examples at each timestep.  The
resulting rotation is used to define RotatedSite objects that replace the usual
CoordinateGroupSites.  Everything else (cost matrix, Sinkhorn, calibration,
test) is identical to the shared OT pipeline.

Timer: pca_fit_time + cost_matrix_time + trial_time for the best epsilon.
"""
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
from experiments.binary_addition_rnn.model import exact_accuracy
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import (
    _bank_summaries,
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
    _trial_grid,
)
from experiments.binary_addition_rnn.sites import enumerate_output_logit_sites, enumerate_pca_group_sites_for_timesteps
from experiments.binary_addition_rnn.transport import (
    TransportConfig,
    enumerate_transport_row_candidates,
    evaluate_single_calibrated_transport,
    select_transport_calibration_candidate,
    sinkhorn_uniform_ot,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OT with PCA-rotated hidden-state sites.")
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
    ap.add_argument("--ot-epsilons", type=str, default="0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--top-k-grid", type=str, default="1,2,4,8")
    ap.add_argument("--lambda-grid", type=str, default="0.25,0.5,1,2,4,8")
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--selection-profiles", type=str, default="combined:0.0;sensitivity_only:0.0")
    ap.add_argument("--selection-rule", type=str, default="combined")
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--source-policy",
        type=str,
        default="structured_26_top3carry_c2x5_c3x7_no_random",
    )
    ap.add_argument("--normalize-signatures", action="store_true")
    ap.add_argument("--fit-signature-mode", type=str, default="all", choices=["all", "active_only"])
    ap.add_argument("--fit-stratify-mode", type=str, default="row_counterfactual", choices=["none", "source_propagation", "row_counterfactual"])
    ap.add_argument("--fit-family-profile", type=str, default="all")
    ap.add_argument("--cost-metric", type=str, default="sq_l2", choices=["sq_l2", "l1", "cosine"])
    return ap.parse_args()


def _fit_pca_rotations(
    fit_examples,
    run_cache,
    timesteps: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """PCA at each timestep; columns of returned matrix are principal components (descending variance)."""
    rotations: dict[int, torch.Tensor] = {}
    for t in timesteps:
        states = torch.stack(
            [run_cache.get_run(ex).hidden_states[int(t)] for ex in fit_examples],
            dim=0,
        ).to(torch.float32)
        centered = states - states.mean(0, keepdim=True)
        _, _, Vt = torch.linalg.svd(centered, full_matrices=True)
        rotations[int(t)] = Vt.T.detach().cpu()  # (hidden_size, hidden_size), columns = PCs
    return rotations


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
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
    )
    timesteps = _parse_ints(args.timesteps)
    resolutions = _parse_ints(args.resolutions) if str(args.resolutions).strip() else _default_resolutions(int(args.hidden_size))
    selection_profiles = _parse_selection_profiles(
        args.selection_profiles,
        default_rule=str(args.selection_rule),
        default_floor=float(args.invariance_floor),
    )
    profile_keys = [f"{rule}__floor_{floor:g}" for rule, floor in selection_profiles]
    carry_keys = [f"C{i}" for i in range(1, int(args.width) + 1)]
    output_keys = [f"S{i}" for i in range(int(args.width)) if f"S{i}" in row_keys]
    fam_order = tuple(_family_order(int(args.width), str(args.source_policy)))

    # Fit PCA rotations — method timer starts here (includes full grid search)
    t_method_start = time.perf_counter()
    rotations = _fit_pca_rotations(split.fit, run_cache, timesteps)
    pca_fit_time = float(time.perf_counter() - t_method_start)

    output_sites = enumerate_output_logit_sites(output_dim=int(args.width) + 1)

    per_resolution = []
    best_trial_by_profile: dict[str, dict] = {}
    best_key_by_profile: dict[str, tuple] = {}

    for resolution in resolutions:
        hidden_sites = enumerate_pca_group_sites_for_timesteps(
            timesteps=timesteps,
            hidden_size=int(args.hidden_size),
            resolution=int(resolution),
            rotations=rotations,
        )
        sites = tuple(hidden_sites) + tuple(output_sites)

        t_cost = time.perf_counter()
        cost, diagnostics = _fit_cost_matrix(
            model,
            specs=specs,
            fit_by_row=banks["fit_by_row"],
            sites=sites,
            family_order=fam_order,
            device=device,
            run_cache=run_cache,
            batch_size=int(args.batch_size),
            normalize_signatures=bool(args.normalize_signatures),
            fit_signature_mode=str(args.fit_signature_mode),
            fit_stratify_mode=str(args.fit_stratify_mode),
            fit_family_profile=str(args.fit_family_profile),
            cost_metric=str(args.cost_metric),
        )
        cost_matrix_time = float(time.perf_counter() - t_cost)

        resolution_trials = []
        for trial_cfg in _trial_grid("ot", transport_cfg):
            t_trial = time.perf_counter()
            coupling = sinkhorn_uniform_ot(
                cost,
                epsilon=float(trial_cfg["epsilon"]),
                n_iter=int(transport_cfg.sinkhorn_iters),
                temperature=float(transport_cfg.temperature),
            )

            profile_row_results: dict[str, dict] = {pk: {} for pk in profile_keys}
            profile_calib_sens: dict[str, list] = {pk: [] for pk in profile_keys}
            profile_calib_inv: dict[str, list] = {pk: [] for pk in profile_keys}
            profile_test_sens: dict[str, list] = {pk: [] for pk in profile_keys}
            profile_test_inv: dict[str, list] = {pk: [] for pk in profile_keys}

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
                for pk, (sel_rule, inv_floor) in zip(profile_keys, selection_profiles):
                    calibrated = select_transport_calibration_candidate(
                        candidates,
                        selection_rule=str(sel_rule),
                        invariance_floor=float(inv_floor),
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
                    profile_row_results[pk][row_key] = {
                        "calibration": calibrated["calibration"],
                        "test": tested,
                        "top_k": int(calibrated["top_k"]),
                        "lambda": float(calibrated["lambda"]),
                        "selected_sites": calibrated["selected_sites"],
                    }
                    profile_calib_sens[pk].append(float(calibrated["calibration"]["sensitivity"]))
                    profile_calib_inv[pk].append(float(calibrated["calibration"]["invariance"]))
                    profile_test_sens[pk].append(float(tested["sensitivity"]))
                    profile_test_inv[pk].append(float(tested["invariance"]))

            trial_time = float(time.perf_counter() - t_trial)
            for pk in profile_keys:
                per_row = profile_row_results[pk]
                calib_summary = {
                    "mean_sensitivity": float(sum(profile_calib_sens[pk]) / max(1, len(profile_calib_sens[pk]))),
                    "mean_invariance": float(sum(profile_calib_inv[pk]) / max(1, len(profile_calib_inv[pk]))),
                    "mean_combined": float((sum(profile_calib_sens[pk]) + sum(profile_calib_inv[pk])) / max(1, 2 * len(profile_calib_sens[pk]))),
                }
                test_summary = {
                    "per_row": per_row,
                    "mean_sensitivity": float(sum(profile_test_sens[pk]) / max(1, len(profile_test_sens[pk]))),
                    "mean_invariance": float(sum(profile_test_inv[pk]) / max(1, len(profile_test_inv[pk]))),
                    "mean_combined": float((sum(profile_test_sens[pk]) + sum(profile_test_inv[pk])) / max(1, 2 * len(profile_test_sens[pk]))),
                    "carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, carry_keys),
                    "output_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, output_keys) if output_keys else None,
                }
                key = (
                    float(calib_summary["mean_combined"]),
                    float(calib_summary["mean_sensitivity"]),
                    float(calib_summary["mean_invariance"]),
                )
                if pk not in best_key_by_profile or key > best_key_by_profile[pk]:
                    best_key_by_profile[pk] = key
                    best_trial_by_profile[pk] = {
                        "config": {"resolution": int(resolution), **trial_cfg},
                        "profile_key": pk,
                        "calibration": calib_summary,
                        "test": test_summary,
                    }

            resolution_trials.append({
                "resolution": int(resolution),
                "config": trial_cfg,
                "trial_time_sec": float(trial_time),
            })
        per_resolution.append({
            "resolution": int(resolution),
            "sites": [site.key() for site in sites],
            "cost_matrix_time_sec": float(cost_matrix_time),
            "fit_diagnostics": diagnostics,
            "trials": resolution_trials,
        })

    primary_pk = profile_keys[0] if profile_keys else None
    best = best_trial_by_profile.get(primary_pk, {})

    result = {
        "config": vars(args),
        "row_keys": row_keys,
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "training": train_summary,
        "pca_fit_time_sec": float(pca_fit_time),
        "method_runtime_sec": float(time.perf_counter() - t_method_start),
        "best_result": best,
        "best_trial_by_profile": best_trial_by_profile,
        "per_resolution": per_resolution,
        "bank_summaries": {
            "fit_by_row": _bank_summaries(
                banks["fit_by_row"],
                {k: tuple(r for r in banks["fit_by_row"][k] if r.is_active) for k in row_keys},
                {k: tuple(r for r in banks["fit_by_row"][k] if not r.is_active) for k in row_keys},
            ),
        },
    }
    summary_path = out_dir / "joint_endogenous_pca_ot_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    compact = {
        "summary": str(summary_path),
        "factual_exact_all": result["factual_exact"]["all"],
        "best_carry_subset": best.get("test", {}).get("carry_subset", {}),
        "method_runtime_sec": result["method_runtime_sec"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
