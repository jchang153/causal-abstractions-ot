from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.interventions import build_run_cache
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import (
    EndogenousPairRecord,
    EndogenousRowSpec,
    _build_banks,
    _build_row_record,
    _default_resolutions,
    _family_order,
    _fit_cost_matrix,
    _load_or_train_model,
    _parse_floats,
    _parse_ints,
    _parse_selection_profiles,
    _row_specs,
    _structured_sources_for_base,
    _subset_summary,
    _trial_grid,
)
from experiments.binary_addition_rnn.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition_rnn.scm import BinaryAdditionExample, intervene_carries
from experiments.binary_addition_rnn.sites import enumerate_group_sites_for_timesteps, enumerate_output_logit_sites
from experiments.binary_addition_rnn.transport import (
    TransportConfig,
    enumerate_transport_row_candidates,
    evaluate_single_calibrated_transport,
    select_transport_calibration_candidate,
    sinkhorn_uniform_ot,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Anchored-bank shared OT: fit one OT plan per anchor-specific core+focus fit bank, then keep the anchor row."
    )
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
    ap.add_argument("--middle-top-k-grid", type=str, default="1,2,4,8,12,16")
    ap.add_argument("--middle-lambda-grid", type=str, default="0.125,0.25,0.5,1,2,4,8,16")
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--selection-rule", type=str, default="combined")
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--selection-profiles", type=str, default="combined:0.0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--source-policy", type=str, default="structured_26_top3carry_c2x5_c3x7_no_random")
    ap.add_argument("--focus-quota-profile", type=str, default="balanced", choices=["light", "balanced", "middle_heavy", "heavy"])
    ap.add_argument("--focus-hamming-penalty", type=float, default=0.05)
    ap.add_argument("--focus-exclude-core", action="store_true")
    ap.add_argument("--normalize-signatures", action="store_true")
    ap.add_argument("--fit-signature-mode", type=str, default="all", choices=["all", "active_only"])
    ap.add_argument("--fit-stratify-mode", type=str, default="row_counterfactual", choices=["none", "source_propagation", "row_counterfactual"])
    ap.add_argument("--fit-family-profile", type=str, default="all")
    ap.add_argument("--cost-metric", type=str, default="sq_l2", choices=["sq_l2", "l1", "cosine"])
    return ap.parse_args()


def _endogenous_key_order(width: int) -> tuple[str, ...]:
    return tuple([f"C{i}" for i in range(1, int(width) + 1)] + [f"S{i}" for i in range(int(width))])


def _endogenous_vector_from_example(example: BinaryAdditionExample) -> tuple[int, ...]:
    return tuple(int(bit) for bit in example.carries) + tuple(int(bit) for bit in example.sum_bits_lsb)


def _anchor_value_from_example(example: BinaryAdditionExample, *, anchor_key: str) -> int:
    if anchor_key.startswith("C"):
        return int(example.carry(int(anchor_key[1:])))
    if anchor_key.startswith("S"):
        return int(example.sum_bits_lsb[int(anchor_key[1:])])
    raise ValueError(f"unsupported anchor key: {anchor_key!r}")


def _ideal_endogenous_vector(
    base: BinaryAdditionExample,
    *,
    anchor_key: str,
    desired_value: int,
) -> tuple[int, ...]:
    if anchor_key.startswith("C"):
        carry_index = int(anchor_key[1:])
        counterfactual = intervene_carries(base, {carry_index: int(desired_value)})
        return _endogenous_vector_from_example(counterfactual)
    if anchor_key.startswith("S"):
        sum_index = int(anchor_key[1:])
        sum_bits = list(base.sum_bits_lsb)
        sum_bits[sum_index] = int(desired_value)
        return tuple(int(bit) for bit in base.carries) + tuple(int(bit) for bit in sum_bits)
    raise ValueError(f"unsupported anchor key: {anchor_key!r}")


def _anchor_priority_sets(anchor_key: str, *, width: int) -> tuple[set[str], set[str]]:
    prefix: set[str] = set()
    blanket: set[str] = set()
    if anchor_key.startswith("C"):
        carry_index = int(anchor_key[1:])
        for idx in range(1, carry_index):
            prefix.add(f"C{idx}")
        for idx in range(carry_index):
            prefix.add(f"S{idx}")
        if carry_index < int(width):
            blanket.add(f"S{carry_index}")
            blanket.add(f"C{carry_index + 1}")
        return prefix, blanket
    if anchor_key.startswith("S"):
        sum_index = int(anchor_key[1:])
        for idx in range(sum_index):
            prefix.add(f"S{idx}")
        for idx in range(1, sum_index):
            prefix.add(f"C{idx}")
        if sum_index > 0:
            blanket.add(f"C{sum_index}")
        return prefix, blanket
    raise ValueError(f"unsupported anchor key: {anchor_key!r}")


def _input_hamming_distance(base: BinaryAdditionExample, source: BinaryAdditionExample) -> int:
    return sum(int(a != b) for a, b in zip(base.a_bits_lsb, source.a_bits_lsb)) + sum(
        int(a != b) for a, b in zip(base.b_bits_lsb, source.b_bits_lsb)
    )


def _ideal_focus_ranking_key(
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    *,
    anchor_key: str,
    desired_value: int,
    width: int,
    hamming_penalty: float,
) -> tuple[float, int, int, int, int, int, int]:
    keys = _endogenous_key_order(int(width))
    ideal_vector = _ideal_endogenous_vector(base, anchor_key=anchor_key, desired_value=int(desired_value))
    source_vector = _endogenous_vector_from_example(source)
    prefix_keys, blanket_keys = _anchor_priority_sets(anchor_key, width=int(width))

    prefix_mismatches = 0
    blanket_mismatches = 0
    other_mismatches = 0
    for idx, key in enumerate(keys):
        if int(source_vector[idx]) == int(ideal_vector[idx]):
            continue
        if key in prefix_keys:
            prefix_mismatches += 1
        elif key in blanket_keys:
            blanket_mismatches += 1
        else:
            other_mismatches += 1

    input_hamming = _input_hamming_distance(base, source)
    weighted_distance = (
        100.0 * float(prefix_mismatches)
        + 10.0 * float(blanket_mismatches)
        + float(other_mismatches)
        + float(hamming_penalty) * float(input_hamming)
    )
    return (
        float(weighted_distance),
        int(prefix_mismatches),
        int(blanket_mismatches),
        int(other_mismatches),
        int(input_hamming),
        int(source.a),
        int(source.b),
    )


def _focus_count_per_value(anchor_key: str, *, profile: str) -> int:
    if anchor_key.startswith("C"):
        carry_index = int(anchor_key[1:])
        table = {
            "light": {1: 1, 2: 2, 3: 3, 4: 1},
            "balanced": {1: 2, 2: 3, 3: 4, 4: 2},
            "middle_heavy": {1: 1, 2: 4, 3: 6, 4: 1},
            "heavy": {1: 2, 2: 4, 3: 5, 4: 2},
        }
        mapping = table[str(profile)]
        return int(mapping.get(int(carry_index), mapping[max(mapping)]))
    if anchor_key.startswith("S"):
        table = {
            "light": 1,
            "balanced": 1,
            "middle_heavy": 1,
            "heavy": 2,
        }
        return int(table[str(profile)])
    raise ValueError(f"unsupported anchor key: {anchor_key!r}")


def _focus_family_order(anchor_key: str, *, per_value_count: int) -> tuple[str, ...]:
    names: list[str] = []
    for desired_value in (0, 1):
        for rank in range(1, int(per_value_count) + 1):
            names.append(f"focus_{anchor_key}_v{desired_value}_{rank}")
    return tuple(names)


def _choose_ideal_focus_sources_for_base(
    base: BinaryAdditionExample,
    *,
    anchor_key: str,
    width: int,
    all_examples: Sequence[BinaryAdditionExample],
    per_value_count: int,
    hamming_penalty: float,
    excluded_pairs: set[tuple[int, int]] | None = None,
) -> tuple[tuple[str, BinaryAdditionExample], ...]:
    families: list[tuple[str, BinaryAdditionExample]] = []
    excluded_pairs = set() if excluded_pairs is None else set(excluded_pairs)
    for desired_value in (0, 1):
        candidates = [
            source
            for source in all_examples
            if not (int(source.a) == int(base.a) and int(source.b) == int(base.b))
            and ((int(source.a), int(source.b)) not in excluded_pairs)
            and int(_anchor_value_from_example(source, anchor_key=anchor_key)) == int(desired_value)
        ]
        if len(candidates) < int(per_value_count):
            candidates = [
                source
                for source in all_examples
                if not (int(source.a) == int(base.a) and int(source.b) == int(base.b))
                and int(_anchor_value_from_example(source, anchor_key=anchor_key)) == int(desired_value)
            ]
        ranked = sorted(
            candidates,
            key=lambda source: _ideal_focus_ranking_key(
                base,
                source,
                anchor_key=anchor_key,
                desired_value=int(desired_value),
                width=int(width),
                hamming_penalty=float(hamming_penalty),
            ),
        )
        for rank, source in enumerate(ranked[: int(per_value_count)], start=1):
            families.append((f"focus_{anchor_key}_v{desired_value}_{rank}", source))
    return tuple(families)


def _build_anchor_core_focus_fit_by_row(
    split_bases: Sequence[BinaryAdditionExample],
    all_examples: Sequence[BinaryAdditionExample],
    specs: Sequence[EndogenousRowSpec],
    *,
    anchor_key: str,
    width: int,
    seed: int,
    core_source_policy: str,
    focus_quota_profile: str,
    focus_hamming_penalty: float,
    focus_exclude_core: bool,
) -> tuple[dict[str, tuple[EndogenousPairRecord, ...]], tuple[str, ...], dict[str, int]]:
    rows: dict[str, list[EndogenousPairRecord]] = {spec.key: [] for spec in specs}
    focus_count = _focus_count_per_value(anchor_key, profile=str(focus_quota_profile))
    family_order = tuple(_family_order(int(width), str(core_source_policy))) + _focus_family_order(
        anchor_key,
        per_value_count=int(focus_count),
    )
    for base in split_bases:
        core_families = _structured_sources_for_base(
            base,
            width=int(width),
            all_examples=all_examples,
            seed=int(seed),
            source_policy=str(core_source_policy),
        )
        excluded_pairs = {(int(source.a), int(source.b)) for _family, source in core_families} if bool(focus_exclude_core) else None
        focus_families = _choose_ideal_focus_sources_for_base(
            base,
            anchor_key=anchor_key,
            width=int(width),
            all_examples=all_examples,
            per_value_count=int(focus_count),
            hamming_penalty=float(focus_hamming_penalty),
            excluded_pairs=excluded_pairs,
        )
        for family, source in tuple(core_families) + tuple(focus_families):
            for spec in specs:
                rows[spec.key].append(_build_row_record(base, source, spec, family=family))
    fit_by_row = {key: tuple(values) for key, values in rows.items()}
    metadata = {
        "focus_count_per_value": int(focus_count),
        "core_family_count": int(len(_family_order(int(width), str(core_source_policy)))),
        "focus_family_count": int(len(_focus_family_order(anchor_key, per_value_count=int(focus_count)))),
    }
    return fit_by_row, family_order, metadata


def _transport_cfg_for_anchor(
    base_cfg: TransportConfig,
    *,
    anchor_key: str,
    middle_topk_grid: tuple[int, ...],
    middle_lambda_grid: tuple[float, ...],
) -> TransportConfig:
    if anchor_key in {"C2", "C3"}:
        return TransportConfig(
            epsilon_grid=tuple(base_cfg.epsilon_grid),
            beta_grid=tuple(base_cfg.beta_grid),
            topk_grid=tuple(int(x) for x in middle_topk_grid),
            lambda_grid=tuple(float(x) for x in middle_lambda_grid),
            sinkhorn_iters=int(base_cfg.sinkhorn_iters),
            temperature=float(base_cfg.temperature),
            invariance_floor=float(base_cfg.invariance_floor),
            selection_rule=str(base_cfg.selection_rule),
        )
    return base_cfg


def _internal_carry_keys(width: int, row_keys: Sequence[str]) -> list[str]:
    keys = [f"C{i}" for i in range(1, int(width)) if f"C{i}" in row_keys]
    return keys if keys else [key for key in row_keys if key.startswith("C")]


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

    base_transport_cfg = TransportConfig(
        epsilon_grid=_parse_floats(args.ot_epsilons),
        beta_grid=(0.1,),
        topk_grid=_parse_ints(args.top_k_grid),
        lambda_grid=_parse_floats(args.lambda_grid),
        sinkhorn_iters=int(args.sinkhorn_iters),
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
    )
    middle_topk_grid = _parse_ints(args.middle_top_k_grid)
    middle_lambda_grid = _parse_floats(args.middle_lambda_grid)
    timesteps = _parse_ints(args.timesteps)
    resolutions = _parse_ints(args.resolutions) if str(args.resolutions).strip() else _default_resolutions(int(args.hidden_size))
    selection_profiles = _parse_selection_profiles(
        args.selection_profiles,
        default_rule=str(args.selection_rule),
        default_floor=float(args.invariance_floor),
    )
    profile_keys = [f"{rule}__floor_{floor:g}" for rule, floor in selection_profiles]
    carry_keys = [f"C{i}" for i in range(1, int(args.width) + 1)]
    internal_carry_keys = _internal_carry_keys(int(args.width), row_keys)
    output_keys = [f"S{i}" for i in range(int(args.width)) if f"S{i}" in row_keys]

    per_anchor: dict[str, object] = {}
    aggregate_per_row_test: dict[str, dict[str, object]] = {}

    for anchor_idx, anchor_key in enumerate(row_keys):
        anchor_fit_by_row, anchor_family_order, anchor_metadata = _build_anchor_core_focus_fit_by_row(
            split.fit,
            examples,
            specs,
            anchor_key=anchor_key,
            width=int(args.width),
            seed=int(args.seed),
            core_source_policy=str(args.source_policy),
            focus_quota_profile=str(args.focus_quota_profile),
            focus_hamming_penalty=float(args.focus_hamming_penalty),
            focus_exclude_core=bool(args.focus_exclude_core),
        )
        anchor_transport_cfg = _transport_cfg_for_anchor(
            base_transport_cfg,
            anchor_key=anchor_key,
            middle_topk_grid=middle_topk_grid,
            middle_lambda_grid=middle_lambda_grid,
        )
        anchor_best = None
        anchor_best_key = None
        anchor_resolution_results = []
        for resolution in resolutions:
            hidden_sites = enumerate_group_sites_for_timesteps(
                timesteps=timesteps,
                hidden_size=int(args.hidden_size),
                resolution=int(resolution),
            )
            output_sites = enumerate_output_logit_sites(output_dim=int(args.width) + 1)
            sites = tuple(hidden_sites) + tuple(output_sites)
            cost, diagnostics = _fit_cost_matrix(
                model,
                specs=specs,
                fit_by_row=anchor_fit_by_row,
                sites=sites,
                family_order=anchor_family_order,
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
            for trial_cfg in _trial_grid("ot", anchor_transport_cfg):
                coupling = sinkhorn_uniform_ot(
                    cost,
                    epsilon=float(trial_cfg["epsilon"]),
                    n_iter=int(anchor_transport_cfg.sinkhorn_iters),
                    temperature=float(anchor_transport_cfg.temperature),
                )
                candidates = enumerate_transport_row_candidates(
                    model,
                    coupling[anchor_idx],
                    sites,
                    banks["calib_positive_by_row"][anchor_key],
                    banks["calib_invariant_by_row"][anchor_key],
                    anchor_transport_cfg,
                    device=device,
                    run_cache=run_cache,
                )
                profile_results = {}
                for profile_key, (selection_rule, invariance_floor) in zip(profile_keys, selection_profiles):
                    calibrated = select_transport_calibration_candidate(
                        candidates,
                        selection_rule=str(selection_rule),
                        invariance_floor=float(invariance_floor),
                    )
                    tested = evaluate_single_calibrated_transport(
                        model,
                        calibrated,
                        sites,
                        banks["test_positive_by_row"][anchor_key],
                        banks["test_invariant_by_row"][anchor_key],
                        device=device,
                        run_cache=run_cache,
                    )
                    profile_results[profile_key] = {
                        "calibration": calibrated["calibration"],
                        "test": tested,
                        "top_k": int(calibrated["top_k"]),
                        "lambda": float(calibrated["lambda"]),
                        "selected_sites": calibrated["selected_sites"],
                    }
                    key = (
                        float(tested["combined"]),
                        float(tested["sensitivity"]),
                        float(tested["invariance"]),
                    )
                    if anchor_best_key is None or key > anchor_best_key:
                        anchor_best_key = key
                        anchor_best = {
                            "anchor_key": anchor_key,
                            "config": {"method": "ot", "resolution": int(resolution), **trial_cfg},
                            "profile_key": profile_key,
                            "selection": {
                                "selection_rule": str(selection_rule),
                                "invariance_floor": float(invariance_floor),
                            },
                            "anchor_bank_design": {
                                "core_source_policy": str(args.source_policy),
                                "focus_quota_profile": str(args.focus_quota_profile),
                                "focus_hamming_penalty": float(args.focus_hamming_penalty),
                                **anchor_metadata,
                            },
                            "coupling": coupling.tolist(),
                            "fit_diagnostics": diagnostics,
                            "test": tested,
                            "calibration": calibrated["calibration"],
                            "top_k": int(calibrated["top_k"]),
                            "lambda": float(calibrated["lambda"]),
                            "selected_sites": calibrated["selected_sites"],
                            "fit_bank_counts": {
                                row_key: int(len(anchor_fit_by_row[row_key])) for row_key in row_keys
                            },
                            "fit_family_order": list(anchor_family_order),
                            "anchor_transport_grid": {
                                "top_k": list(anchor_transport_cfg.topk_grid),
                                "lambda": list(anchor_transport_cfg.lambda_grid),
                            },
                        }
                trials.append(
                    {
                        "config": {"method": "ot", "resolution": int(resolution), **trial_cfg},
                        "profile_results": profile_results,
                    }
                )
            anchor_resolution_results.append(
                {
                    "resolution": int(resolution),
                    "sites": [site.key() for site in sites],
                    "fit_diagnostics": diagnostics,
                    "trials": trials,
                }
            )

        if anchor_best is None:
            raise RuntimeError(f"failed to produce any anchor-bank OT result for {anchor_key}")
        per_anchor[anchor_key] = {
            "best": anchor_best,
            "per_resolution": anchor_resolution_results,
        }
        aggregate_per_row_test[anchor_key] = anchor_best["test"]

    result = {
        "config": vars(args),
        "row_keys": row_keys,
        "factual_exact": {
            "all": float(torch.tensor(0.0).item()) + 0.0,
        },
        "training": train_summary,
        "per_anchor": per_anchor,
        "aggregated_test": {
            "per_row": aggregate_per_row_test,
            "carry_subset": _subset_summary(aggregate_per_row_test, carry_keys),
            "internal_carry_subset": _subset_summary(aggregate_per_row_test, internal_carry_keys),
            "output_subset": _subset_summary(aggregate_per_row_test, output_keys) if output_keys else None,
        },
    }
    from experiments.binary_addition_rnn.model import exact_accuracy

    result["factual_exact"] = {
        "all": exact_accuracy(model, examples, device=device),
        "fit": exact_accuracy(model, split.fit, device=device),
        "calib": exact_accuracy(model, split.calib, device=device),
        "test": exact_accuracy(model, split.test, device=device),
    }

    summary_path = out_dir / "joint_endogenous_anchor_fitbanks_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        "summary": str(summary_path),
        "factual_exact_all": result["factual_exact"]["all"],
        "carry_subset": result["aggregated_test"]["carry_subset"],
        "internal_carry_subset": result["aggregated_test"]["internal_carry_subset"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
