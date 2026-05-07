from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.data import BaseSplit, enumerate_all_examples, stratified_base_split
from experiments.binary_addition_rnn.interventions import build_run_cache, intervene_with_site_handle_batch
from experiments.binary_addition_rnn.model import GRUAdder, TrainConfig, exact_accuracy, train_backbone
from experiments.binary_addition_rnn.pca_basis import fit_pca_rotations
from experiments.binary_addition_rnn.scm import BinaryAdditionExample, intervene_carries
from experiments.binary_addition_rnn.sites import (
    Site,
    enumerate_group_sites_for_timesteps,
    enumerate_output_logit_sites,
    enumerate_rotated_prefix_sites_for_timesteps,
    enumerate_rotated_group_sites_for_timesteps,
)
from experiments.binary_addition_rnn.transport import (
    TransportConfig,
    enumerate_transport_row_candidates,
    evaluate_single_calibrated_transport,
    select_transport_calibration_candidate,
    sinkhorn_one_sided_uot,
    sinkhorn_uniform_ot,
)


@dataclass(frozen=True)
class EndogenousRowSpec:
    key: str
    kind: str
    index: int


@dataclass(frozen=True)
class OutputCounterfactual:
    output_bits_lsb: tuple[int, ...]


@dataclass(frozen=True)
class EndogenousPairRecord:
    base: BinaryAdditionExample
    source: BinaryAdditionExample
    family: str
    row_key: str
    forced_value: int
    counterfactual: OutputCounterfactual
    is_active: bool
    base_propagation_length: int
    source_propagation_length: int
    counterfactual_propagation_length: int | None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Original-pipeline shared OT/UOT sweep over carries+outputs vs hidden-state/output-logit sites."
    )
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--abstract-mode", type=str, default="all_endogenous", choices=["carries_only", "all_endogenous"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=4)
    ap.add_argument("--timesteps", type=str, default="0,1,2,3")
    ap.add_argument("--resolutions", type=str, default="")
    ap.add_argument("--methods", type=str, default="ot,uot")
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
    ap.add_argument("--ot-epsilons", type=str, default="0.03")
    ap.add_argument("--uot-betas", type=str, default="0.1")
    ap.add_argument("--top-k-grid", type=str, default="1,2,4,8")
    ap.add_argument("--lambda-grid", type=str, default="0.5,1,2,4")
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--selection-rule", type=str, default="combined")
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--selection-profiles", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=512)
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
            "structured_24_top3carry_top1sum_no_random",
            "structured_25_top3carry_top1sum",
            "structured_24_top3carry_c2c3x5_no_random",
            "structured_24_top3carry_c3x7_no_random",
            "structured_26_top3carry_c2x5_c3x7_no_random",
            "structured_28_top3carry_top2sum_no_random",
            "structured_29_top3carry_top2sum",
            "structured_24_top3carry_top1sum_prefix_no_random",
            "structured_28_top3carry_top2sum_prefix_no_random",
        ],
    )
    ap.add_argument("--normalize-signatures", action="store_true")
    ap.add_argument("--fit-signature-mode", type=str, default="all", choices=["all", "active_only"])
    ap.add_argument(
        "--fit-stratify-mode",
        type=str,
        default="none",
        choices=["none", "source_propagation", "row_counterfactual"],
    )
    ap.add_argument(
        "--fit-family-profile",
        type=str,
        default="all",
        choices=[
            "all",
            "target_boost",
            "causal_prefix_boost",
            "target_only",
            "kind_decouple",
            "carry_prefix_no_sum",
            "row_support_only",
            "row_support_boost",
        ],
    )
    ap.add_argument("--cost-metric", type=str, default="sq_l2", choices=["sq_l2", "l1", "cosine"])
    ap.add_argument("--hidden-site-basis", type=str, default="coordinate", choices=["coordinate", "pca"])
    ap.add_argument("--pca-variant", type=str, default="uncentered", choices=["uncentered", "centered", "whitened"])
    ap.add_argument("--pca-site-menu", type=str, default="partition", choices=["partition", "top_prefix", "both"])
    return ap.parse_args()


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_selection_profiles(text: str, *, default_rule: str, default_floor: float) -> tuple[tuple[str, float], ...]:
    if not str(text).strip():
        return ((str(default_rule), float(default_floor)),)
    profiles: list[tuple[str, float]] = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        rule_text, floor_text = [part.strip() for part in chunk.split(":")]
        profiles.append((str(rule_text), float(floor_text)))
    return tuple(profiles)


def _structured_policy_spec(source_policy: str) -> tuple[int, int, bool]:
    specs = {
        "structured_13": (1, 0, True),
        "structured_12_no_random": (1, 0, False),
        "structured_17_top2carry": (2, 0, True),
        "structured_20_top3carry_no_random": (3, 0, False),
        "structured_21_top3carry": (3, 0, True),
        "structured_22_top3carry_c3x5_no_random": (3, 0, False),
        "structured_24_top3carry_top1sum_no_random": (3, 1, False),
        "structured_25_top3carry_top1sum": (3, 1, True),
        "structured_24_top3carry_c2c3x5_no_random": (3, 0, False),
        "structured_24_top3carry_c3x7_no_random": (3, 0, False),
        "structured_26_top3carry_c2x5_c3x7_no_random": (3, 0, False),
        "structured_28_top3carry_top2sum_no_random": (3, 2, False),
        "structured_29_top3carry_top2sum": (3, 2, True),
        "structured_24_top3carry_top1sum_prefix_no_random": (3, 1, False),
        "structured_28_top3carry_top2sum_prefix_no_random": (3, 2, False),
    }
    if source_policy not in specs:
        raise ValueError(f"unknown structured source policy: {source_policy!r}")
    return specs[source_policy]


def _sum_target_mode(source_policy: str) -> str:
    if source_policy in {
        "structured_24_top3carry_top1sum_prefix_no_random",
        "structured_28_top3carry_top2sum_prefix_no_random",
    }:
        return "prefix"
    return "global"


def _carry_target_counts(source_policy: str, *, width: int) -> tuple[int, ...]:
    carry_source_count, _sum_source_count, _include_random = _structured_policy_spec(source_policy)
    counts = [int(carry_source_count) for _ in range(int(width))]
    if source_policy == "structured_22_top3carry_c3x5_no_random" and int(width) >= 3:
        counts[2] = 5
    if source_policy == "structured_24_top3carry_c2c3x5_no_random" and int(width) >= 3:
        counts[1] = 5
        counts[2] = 5
    if source_policy == "structured_24_top3carry_c3x7_no_random" and int(width) >= 3:
        counts[2] = 7
    if source_policy == "structured_26_top3carry_c2x5_c3x7_no_random" and int(width) >= 3:
        counts[1] = 5
        counts[2] = 7
    return tuple(counts)


def _default_resolutions(hidden_size: int) -> tuple[int, ...]:
    candidates = [int(hidden_size)]
    if hidden_size % 2 == 0 and hidden_size // 2 not in candidates:
        candidates.append(hidden_size // 2)
    if 1 not in candidates:
        candidates.append(1)
    return tuple(candidates)


def _dedupe_sites(sites: Sequence[Site]) -> tuple[Site, ...]:
    by_key: dict[str, Site] = {}
    for site in sites:
        by_key.setdefault(site.key(), site)
    return tuple(by_key.values())


def _row_specs(mode: str, width: int) -> tuple[EndogenousRowSpec, ...]:
    specs = [EndogenousRowSpec(key=f"C{i}", kind="carry", index=i) for i in range(1, int(width) + 1)]
    if mode == "all_endogenous":
        specs.extend(EndogenousRowSpec(key=f"S{i}", kind="sum", index=i) for i in range(int(width)))
    return tuple(specs)


def _family_order(width: int, source_policy: str) -> tuple[str, ...]:
    names: list[str] = []
    for bit in range(int(width)):
        names.append(f"flip_A{bit}")
        names.append(f"flip_B{bit}")
    carry_counts = _carry_target_counts(source_policy, width=int(width))
    _carry_source_count, sum_source_count, include_random = _structured_policy_spec(source_policy)
    for carry_index in range(1, int(width) + 1):
        carry_count = int(carry_counts[carry_index - 1])
        for rank in range(carry_count):
            if carry_count == 1:
                names.append(f"target_C{carry_index}")
            else:
                names.append(f"target_C{carry_index}_{rank + 1}")
    for sum_index in range(int(width)):
        for rank in range(sum_source_count):
            if sum_source_count == 1:
                names.append(f"target_S{sum_index}")
            else:
                names.append(f"target_S{sum_index}_{rank + 1}")
    if include_random:
        names.append("random_source")
    return tuple(names)


def _input_hamming_distance(base: BinaryAdditionExample, source: BinaryAdditionExample) -> int:
    return sum(int(a != b) for a, b in zip(base.a_bits_lsb, source.a_bits_lsb)) + sum(
        int(a != b) for a, b in zip(base.b_bits_lsb, source.b_bits_lsb)
    )


def _endogenous_bit_tuple(example: BinaryAdditionExample) -> tuple[int, ...]:
    return tuple(int(bit) for bit in example.carries) + tuple(int(bit) for bit in example.sum_bits_lsb)


def _other_endogenous_diff_count(base: BinaryAdditionExample, source: BinaryAdditionExample, *, carry_index: int) -> int:
    base_bits = _endogenous_bit_tuple(base)
    source_bits = _endogenous_bit_tuple(source)
    skip = int(carry_index) - 1
    return sum(int(base_bits[idx] != source_bits[idx]) for idx in range(len(base_bits)) if idx != skip)


def _other_endogenous_diff_count_for_sum(base: BinaryAdditionExample, source: BinaryAdditionExample, *, sum_index: int) -> int:
    base_bits = _endogenous_bit_tuple(base)
    source_bits = _endogenous_bit_tuple(source)
    skip = int(base.width) + int(sum_index)
    return sum(int(base_bits[idx] != source_bits[idx]) for idx in range(len(base_bits)) if idx != skip)


def _sum_prefix_mismatch_count(base: BinaryAdditionExample, source: BinaryAdditionExample, *, carry_index: int) -> int:
    prefix_len = int(carry_index)
    return sum(
        int(base.sum_bits_lsb[idx] != source.sum_bits_lsb[idx])
        for idx in range(prefix_len)
    )


def _other_sum_mismatch_count(base: BinaryAdditionExample, source: BinaryAdditionExample, *, sum_index: int) -> int:
    return sum(
        int(base.sum_bits_lsb[idx] != source.sum_bits_lsb[idx])
        for idx in range(base.width)
        if idx != int(sum_index)
    )


def _sum_prefix_mismatch_count_for_sum(base: BinaryAdditionExample, source: BinaryAdditionExample, *, sum_index: int) -> int:
    return sum(
        int(base.sum_bits_lsb[idx] != source.sum_bits_lsb[idx])
        for idx in range(int(sum_index))
    )


def _choose_targeted_carry_sources(
    base: BinaryAdditionExample,
    *,
    carry_index: int,
    count: int,
    candidates: Sequence[BinaryAdditionExample],
) -> tuple[BinaryAdditionExample, ...]:
    valid = [source for source in candidates if int(source.carry(carry_index)) != int(base.carry(carry_index))]
    if not valid:
        raise ValueError(f"no source found that flips C{carry_index} for base ({base.a}, {base.b})")

    def key(source: BinaryAdditionExample) -> tuple[int, int, int, int, int]:
        return (
            _sum_prefix_mismatch_count(base, source, carry_index=int(carry_index)),
            _other_endogenous_diff_count(base, source, carry_index=int(carry_index)),
            _input_hamming_distance(base, source),
            int(source.a),
            int(source.b),
        )

    ranked = sorted(valid, key=key)
    return tuple(ranked[: int(count)])


def _choose_targeted_sum_sources(
    base: BinaryAdditionExample,
    *,
    sum_index: int,
    count: int,
    candidates: Sequence[BinaryAdditionExample],
    prefix_aware: bool,
) -> tuple[BinaryAdditionExample, ...]:
    valid = [source for source in candidates if int(source.sum_bits_lsb[sum_index]) != int(base.sum_bits_lsb[sum_index])]
    if not valid:
        raise ValueError(f"no source found that flips S{sum_index} for base ({base.a}, {base.b})")

    def key(source: BinaryAdditionExample) -> tuple[int, int, int, int, int]:
        prefix_mismatch = _sum_prefix_mismatch_count_for_sum(base, source, sum_index=int(sum_index)) if prefix_aware else 0
        return (
            prefix_mismatch,
            _other_sum_mismatch_count(base, source, sum_index=int(sum_index)),
            _other_endogenous_diff_count_for_sum(base, source, sum_index=int(sum_index)),
            _input_hamming_distance(base, source),
            int(source.a) * 100 + int(source.b),
        )

    ranked = sorted(valid, key=key)
    return tuple(ranked[: int(count)])


def _structured_sources_for_base(
    base: BinaryAdditionExample,
    *,
    width: int,
    all_examples: Sequence[BinaryAdditionExample],
    seed: int,
    source_policy: str,
) -> tuple[tuple[str, BinaryAdditionExample], ...]:
    by_ab = {(int(ex.a), int(ex.b)): ex for ex in all_examples}
    families: list[tuple[str, BinaryAdditionExample]] = []
    for bit in range(int(width)):
        families.append((f"flip_A{bit}", by_ab[(int(base.a) ^ (1 << bit), int(base.b))]))
        families.append((f"flip_B{bit}", by_ab[(int(base.a), int(base.b) ^ (1 << bit))]))
    carry_counts = _carry_target_counts(source_policy, width=int(width))
    _carry_source_count, sum_source_count, include_random = _structured_policy_spec(source_policy)
    sum_target_mode = _sum_target_mode(source_policy)
    for carry_index in range(1, int(width) + 1):
        targeted = _choose_targeted_carry_sources(
            base,
            carry_index=carry_index,
            count=int(carry_counts[carry_index - 1]),
            candidates=all_examples,
        )
        for rank, source in enumerate(targeted, start=1):
            family = f"target_C{carry_index}" if int(carry_counts[carry_index - 1]) == 1 else f"target_C{carry_index}_{rank}"
            families.append((family, source))
    for sum_index in range(int(width)):
        if sum_source_count <= 0:
            continue
        targeted = _choose_targeted_sum_sources(
            base,
            sum_index=sum_index,
            count=sum_source_count,
            candidates=all_examples,
            prefix_aware=(sum_target_mode == "prefix"),
        )
        for rank, source in enumerate(targeted, start=1):
            family = f"target_S{sum_index}" if sum_source_count == 1 else f"target_S{sum_index}_{rank}"
            families.append((family, source))
    if include_random:
        rng = random.Random(int(seed) * 10007 + int(base.a) * 257 + int(base.b) * 17 + 3)
        random_candidates = [ex for ex in all_examples if not (int(ex.a) == int(base.a) and int(ex.b) == int(base.b))]
        families.append(("random_source", random_candidates[rng.randrange(len(random_candidates))]))
    return tuple(families)


def _parse_target_family(family: str) -> tuple[str, int] | None:
    if family.startswith("target_C"):
        tail = family[len("target_C") :].split("_")[0]
        return ("carry", int(tail))
    if family.startswith("target_S"):
        tail = family[len("target_S") :].split("_")[0]
        return ("sum", int(tail))
    return None


def _family_weight_for_row(*, row_key: str, family: str, profile: str) -> float:
    if profile == "all":
        return 1.0
    if profile == "focus_boost_2":
        return 2.0 if family.startswith("focus_") else 1.0
    if profile == "focus_boost_4":
        return 4.0 if family.startswith("focus_") else 1.0
    if profile == "focus_boost_8":
        return 8.0 if family.startswith("focus_") else 1.0
    if profile == "focus_only":
        return 1.0 if family.startswith("focus_") else 0.0

    if row_key.startswith("C"):
        row_kind = "carry"
        row_index = int(row_key[1:])
        local_bit = row_index - 1
    elif row_key.startswith("S"):
        row_kind = "sum"
        row_index = int(row_key[1:])
        local_bit = row_index
    else:
        return 1.0

    target_info = _parse_target_family(family)
    exact_target = (
        target_info is not None
        and ((row_kind == "carry" and target_info[0] == "carry" and int(target_info[1]) == int(row_index))
             or (row_kind == "sum" and target_info[0] == "sum" and int(target_info[1]) == int(row_index)))
    )
    is_local_operand = family in {f"flip_A{local_bit}", f"flip_B{local_bit}"}

    if profile == "target_only":
        return 1.0 if (exact_target or is_local_operand) else 0.0

    if profile == "target_boost":
        if exact_target:
            return 4.0
        if is_local_operand:
            return 2.0
        return 1.0

    if profile == "causal_prefix_boost":
        if exact_target:
            return 4.0
        if row_kind == "carry":
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                if bit <= local_bit:
                    return 2.0
            if target_info is not None and target_info[0] == "carry" and int(target_info[1]) <= int(row_index):
                return 2.0
            if target_info is not None and target_info[0] == "sum" and int(target_info[1]) >= int(local_bit):
                return 1.5
            return 0.5
        if row_kind == "sum":
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                if bit <= local_bit:
                    return 2.0
            if target_info is not None and target_info[0] == "carry" and int(target_info[1]) <= int(row_index):
                return 1.5
            if target_info is not None and target_info[0] == "sum" and int(target_info[1]) <= int(row_index):
                return 2.0
            return 0.5

    if profile == "kind_decouple":
        if exact_target:
            return 4.0
        if is_local_operand:
            return 2.0
        if target_info is not None and target_info[0] != row_kind:
            return 0.0
        if target_info is not None and target_info[0] == row_kind:
            return 1.5
        if family.startswith("flip_A") or family.startswith("flip_B"):
            bit = int(family[-1])
            return 1.5 if bit <= local_bit else 0.5
        return 1.0

    if profile == "carry_prefix_no_sum":
        if exact_target:
            return 5.0
        if row_kind == "carry":
            if is_local_operand:
                return 3.0
            if target_info is not None and target_info[0] == "sum":
                return 0.0
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                return 2.0 if bit <= local_bit else 0.25
            if target_info is not None and target_info[0] == "carry":
                return 2.0 if int(target_info[1]) <= int(row_index) else 0.5
            return 0.5
        if row_kind == "sum":
            if is_local_operand:
                return 2.0
            if target_info is not None and target_info[0] == "sum":
                return 2.0 if int(target_info[1]) <= int(row_index) else 0.5
            if target_info is not None and target_info[0] == "carry":
                return 0.5
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                return 1.5 if bit <= local_bit else 0.5
            return 0.5

    if profile == "row_support_only":
        if row_kind == "carry":
            if exact_target:
                return 1.0
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                return 1.0 if bit <= local_bit else 0.0
            if target_info is not None and target_info[0] == "carry":
                return 1.0 if int(target_info[1]) <= int(row_index) else 0.0
            return 0.0
        if row_kind == "sum":
            if exact_target:
                return 1.0
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                return 1.0 if bit == local_bit else 0.0
            if target_info is not None and target_info[0] == "carry":
                return 1.0 if int(target_info[1]) <= int(row_index) else 0.0
            return 0.0

    if profile == "row_support_boost":
        if row_kind == "carry":
            if exact_target:
                return 5.0
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                return 2.0 if bit <= local_bit else 0.0
            if target_info is not None and target_info[0] == "carry":
                return 2.0 if int(target_info[1]) <= int(row_index) else 0.0
            return 0.0
        if row_kind == "sum":
            if exact_target:
                return 5.0
            if family.startswith("flip_A") or family.startswith("flip_B"):
                bit = int(family[-1])
                return 2.0 if bit == local_bit else 0.0
            if target_info is not None and target_info[0] == "carry":
                return 2.0 if int(target_info[1]) <= int(row_index) else 0.0
            return 0.0

    raise ValueError(f"unknown fit family profile: {profile!r}")


def _build_sum_counterfactual(base: BinaryAdditionExample, *, sum_index: int, forced_value: int) -> OutputCounterfactual:
    bits = list(base.output_bits_lsb)
    bits[int(sum_index)] = int(forced_value)
    return OutputCounterfactual(output_bits_lsb=tuple(bits))


def _build_row_record(
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    spec: EndogenousRowSpec,
    *,
    family: str,
) -> EndogenousPairRecord:
    cf_prop: int | None = None
    if spec.kind == "carry":
        forced_value = int(source.carry(spec.index))
        counterfactual = intervene_carries(base, {int(spec.index): forced_value})
        output_bits = tuple(int(bit) for bit in counterfactual.output_bits_lsb)
        cf_prop = int(counterfactual.propagation_length)
    elif spec.kind == "sum":
        forced_value = int(source.sum_bits_lsb[spec.index])
        counterfactual = _build_sum_counterfactual(base, sum_index=spec.index, forced_value=forced_value)
        output_bits = counterfactual.output_bits_lsb
    else:
        raise ValueError(f"unsupported row kind: {spec.kind!r}")
    is_active = tuple(output_bits) != tuple(base.output_bits_lsb)
    return EndogenousPairRecord(
        base=base,
        source=source,
        family=str(family),
        row_key=spec.key,
        forced_value=int(forced_value),
        counterfactual=OutputCounterfactual(output_bits_lsb=tuple(output_bits)),
        is_active=bool(is_active),
        base_propagation_length=int(base.propagation_length),
        source_propagation_length=int(source.propagation_length),
        counterfactual_propagation_length=cf_prop,
    )


def _build_row_records(
    split_bases: Sequence[BinaryAdditionExample],
    all_examples: Sequence[BinaryAdditionExample],
    specs: Sequence[EndogenousRowSpec],
    *,
    width: int,
    seed: int,
    source_policy: str,
) -> dict[str, tuple[EndogenousPairRecord, ...]]:
    rows: dict[str, list[EndogenousPairRecord]] = {spec.key: [] for spec in specs}
    for base in split_bases:
        if source_policy == "all_source":
            source_families = tuple(("all_source", source) for source in all_examples)
        elif source_policy in {
            "structured_13",
            "structured_12_no_random",
            "structured_17_top2carry",
            "structured_20_top3carry_no_random",
            "structured_21_top3carry",
            "structured_22_top3carry_c3x5_no_random",
            "structured_24_top3carry_top1sum_no_random",
            "structured_25_top3carry_top1sum",
            "structured_24_top3carry_c2c3x5_no_random",
            "structured_24_top3carry_c3x7_no_random",
            "structured_26_top3carry_c2x5_c3x7_no_random",
            "structured_28_top3carry_top2sum_no_random",
            "structured_29_top3carry_top2sum",
            "structured_24_top3carry_top1sum_prefix_no_random",
            "structured_28_top3carry_top2sum_prefix_no_random",
        }:
            source_families = _structured_sources_for_base(
                base,
                width=width,
                all_examples=all_examples,
                seed=seed,
                source_policy=source_policy,
            )
        else:
            raise ValueError(f"unknown source_policy: {source_policy!r}")
        for family, source in source_families:
            for spec in specs:
                rows[spec.key].append(_build_row_record(base, source, spec, family=family))
    return {key: tuple(vals) for key, vals in rows.items()}


def _partition_records(
    row_records: dict[str, tuple[EndogenousPairRecord, ...]],
    *,
    active: bool,
) -> dict[str, tuple[EndogenousPairRecord, ...]]:
    return {
        key: tuple(rec for rec in records if bool(rec.is_active) is active)
        for key, records in row_records.items()
    }


def _build_banks(
    split: BaseSplit,
    specs: Sequence[EndogenousRowSpec],
    *,
    width: int,
    seed: int,
    source_policy: str,
    all_examples: Sequence[BinaryAdditionExample],
) -> dict[str, object]:
    fit_by_row = _build_row_records(
        split.fit,
        all_examples,
        specs,
        width=width,
        seed=seed,
        source_policy=source_policy,
    )
    calib_all = _build_row_records(
        split.calib,
        all_examples,
        specs,
        width=width,
        seed=seed,
        source_policy=source_policy,
    )
    test_all = _build_row_records(
        split.test,
        all_examples,
        specs,
        width=width,
        seed=seed,
        source_policy=source_policy,
    )
    return {
        "fit_by_row": fit_by_row,
        "calib_positive_by_row": _partition_records(calib_all, active=True),
        "calib_invariant_by_row": _partition_records(calib_all, active=False),
        "test_positive_by_row": _partition_records(test_all, active=True),
        "test_invariant_by_row": _partition_records(test_all, active=False),
    }


def _bank_summaries(
    row_records: dict[str, tuple[EndogenousPairRecord, ...]],
    positive_by_row: dict[str, tuple[EndogenousPairRecord, ...]],
    invariant_by_row: dict[str, tuple[EndogenousPairRecord, ...]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row_key, records in row_records.items():
        out[row_key] = {
            "total": int(len(records)),
            "positive": int(len(positive_by_row[row_key])),
            "invariant": int(len(invariant_by_row[row_key])),
        }
    return out


def _bits_tensor(bits: Sequence[int]) -> torch.Tensor:
    return torch.tensor([float(bit) for bit in bits], dtype=torch.float32)


def _abstract_signature(record: EndogenousPairRecord) -> torch.Tensor:
    return _bits_tensor(record.counterfactual.output_bits_lsb) - _bits_tensor(record.base.output_bits_lsb)


def _aggregate_mean(signatures: Sequence[torch.Tensor]) -> torch.Tensor:
    if not signatures:
        raise ValueError("need at least one signature")
    return torch.stack([sig.to(torch.float32) for sig in signatures], dim=0).mean(dim=0)


def _aggregate_mean_or_zeros(signatures: Sequence[torch.Tensor], *, dim: int) -> torch.Tensor:
    if not signatures:
        return torch.zeros(int(dim), dtype=torch.float32)
    return _aggregate_mean(signatures)


def _fit_strata(width: int, mode: str) -> tuple[str, ...]:
    if mode == "none":
        return ("all",)
    return tuple(f"prop_{idx}" for idx in range(int(width) + 1))


def _record_fit_stratum(record: EndogenousPairRecord, mode: str) -> str:
    if mode == "none":
        return "all"
    if mode == "source_propagation":
        return f"prop_{int(record.source_propagation_length)}"
    if mode == "row_counterfactual":
        if record.counterfactual_propagation_length is not None:
            return f"prop_{int(record.counterfactual_propagation_length)}"
        return f"prop_{int(record.source_propagation_length)}"
    raise ValueError(f"unknown fit_stratify_mode: {mode!r}")


def _normalize_rows(table: torch.Tensor) -> torch.Tensor:
    norms = table.norm(dim=1, keepdim=True)
    return table / norms.clamp_min(1e-30)


def _fit_cost_matrix(
    model: GRUAdder,
    *,
    specs: Sequence[EndogenousRowSpec],
    fit_by_row: dict[str, Sequence[EndogenousPairRecord]],
    sites: Sequence[Site],
    family_order: Sequence[str],
    device: torch.device,
    run_cache,
    batch_size: int,
    normalize_signatures: bool,
    fit_signature_mode: str,
    fit_stratify_mode: str,
    fit_family_profile: str,
    cost_metric: str,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    pair_site_cache: dict[tuple[int, int, int, int, str], torch.Tensor] = {}

    unique_records: dict[tuple[int, int, int, int, str], EndogenousPairRecord] = {}
    for spec in specs:
        for rec in fit_by_row[spec.key]:
            unique_records[(rec.base.a, rec.base.b, rec.source.a, rec.source.b, rec.family)] = rec
    all_records = list(unique_records.values())

    for site in sites:
        for start in range(0, len(all_records), int(batch_size)):
            batch_records = all_records[start : start + int(batch_size)]
            batch_logits = intervene_with_site_handle_batch(
                model,
                [rec.base for rec in batch_records],
                [rec.source for rec in batch_records],
                [(site, 1.0)],
                lambda_scale=1.0,
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            batch_probs = torch.sigmoid(batch_logits)
            factual = torch.stack([run_cache.get_run(rec.base).output_probs for rec in batch_records], dim=0)
            batch_sigs = batch_probs - factual
            for rec, sig in zip(batch_records, batch_sigs):
                pair_site_cache[(rec.base.a, rec.base.b, rec.source.a, rec.source.b, site.key())] = sig.detach().cpu()

    strata = _fit_strata(int(model.width), str(fit_stratify_mode))
    bucket_order = tuple((family, stratum) for family in family_order for stratum in strata)

    abstract_rows = []
    neural_rows = []
    for spec in specs:
        family_weights = torch.tensor(
            [_family_weight_for_row(row_key=spec.key, family=family, profile=fit_family_profile) for family in family_order],
            dtype=torch.float32,
        )
        weight_vec = family_weights.repeat_interleave(len(strata) * (int(model.width) + 1))
        per_bucket: dict[tuple[str, str], list[torch.Tensor]] = {bucket: [] for bucket in bucket_order}
        neural_per_bucket_by_site: list[dict[tuple[str, str], list[torch.Tensor]]] = [
            {bucket: [] for bucket in bucket_order} for _ in sites
        ]
        for rec in fit_by_row[spec.key]:
            if fit_signature_mode == "active_only" and not rec.is_active:
                continue
            bucket = (rec.family, _record_fit_stratum(rec, fit_stratify_mode))
            per_bucket[bucket].append(_abstract_signature(rec))
            for site_idx, site in enumerate(sites):
                neural_per_bucket_by_site[site_idx][bucket].append(
                    pair_site_cache[(rec.base.a, rec.base.b, rec.source.a, rec.source.b, site.key())]
                )
        abstract_rows.append(
            torch.cat(
                [_aggregate_mean_or_zeros(per_bucket[bucket], dim=int(model.width) + 1) for bucket in bucket_order],
                dim=0,
            )
            * weight_vec
        )
        neural_site_rows = []
        for site_idx, _site in enumerate(sites):
            neural_site_rows.append(
                torch.cat(
                    [
                        _aggregate_mean_or_zeros(neural_per_bucket_by_site[site_idx][bucket], dim=int(model.width) + 1)
                        for bucket in bucket_order
                    ],
                    dim=0,
                )
                * weight_vec
            )
        neural_rows.append(torch.stack(neural_site_rows, dim=0))

    abstract_table = torch.stack(abstract_rows, dim=0)
    neural_table = torch.stack(neural_rows, dim=0)

    if normalize_signatures:
        abstract_table = _normalize_rows(abstract_table)
        neural_table = neural_table / neural_table.norm(dim=2, keepdim=True).clamp_min(1e-30)

    diffs = abstract_table[:, None, :] - neural_table
    if cost_metric == "sq_l2":
        cost = torch.sum(diffs * diffs, dim=2)
    elif cost_metric == "l1":
        cost = torch.sum(torch.abs(diffs), dim=2)
    elif cost_metric == "cosine":
        abstract_unit = abstract_table / abstract_table.norm(dim=1, keepdim=True).clamp_min(1e-30)
        neural_unit = neural_table / neural_table.norm(dim=2, keepdim=True).clamp_min(1e-30)
        cost = 1.0 - torch.sum(abstract_unit[:, None, :] * neural_unit, dim=2)
    else:
        raise ValueError(f"unknown cost_metric: {cost_metric!r}")
    cost = cost.to(torch.float32)
    diagnostics = {
        "abstract_signatures": abstract_table.tolist(),
        "neural_signatures": neural_table.tolist(),
        "cost_matrix": cost.tolist(),
        "cached_pair_site_signatures": int(len(pair_site_cache)),
        "fit_signature_mode": str(fit_signature_mode),
        "fit_stratify_mode": str(fit_stratify_mode),
        "fit_strata": list(strata),
        "fit_family_profile": str(fit_family_profile),
        "cost_metric": str(cost_metric),
    }
    return cost, diagnostics


def _load_or_train_model(
    args: argparse.Namespace,
    examples: Sequence[BinaryAdditionExample],
    split: BaseSplit,
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


def _trial_grid(method: str, config: TransportConfig) -> list[dict[str, float]]:
    if method == "ot":
        return [{"epsilon": float(eps)} for eps in config.epsilon_grid]
    if method == "uot":
        return [
            {"epsilon": float(eps), "beta_neural": float(beta)}
            for eps in config.epsilon_grid
            for beta in config.beta_grid
        ]
    raise ValueError(f"unsupported method: {method}")


def _subset_summary(per_row: dict[str, dict[str, object]], keys: Sequence[str]) -> dict[str, float]:
    sens = [float(per_row[key]["sensitivity"]) for key in keys]
    inv = [float(per_row[key]["invariance"]) for key in keys]
    return {
        "mean_sensitivity": float(sum(sens) / max(1, len(sens))),
        "mean_invariance": float(sum(inv) / max(1, len(inv))),
        "mean_combined": float((sum(sens) + sum(inv)) / max(1, 2 * len(sens))),
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
    rotation_map = None
    hidden_site_basis_diagnostics: dict[str, object] | None = None
    if str(args.hidden_site_basis) == "pca":
        rotation_map, hidden_site_basis_diagnostics = fit_pca_rotations(
            fit_examples=split.fit,
            run_cache=run_cache,
            width=int(args.width),
            hidden_size=int(args.hidden_size),
            variant=str(args.pca_variant),
        )

    specs = _row_specs(args.abstract_mode, int(args.width))
    row_keys = [spec.key for spec in specs]
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
        beta_grid=_parse_floats(args.uot_betas),
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
    requested_methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    carry_keys = [f"C{i}" for i in range(1, int(args.width) + 1)]
    output_keys = [f"S{i}" for i in range(int(args.width)) if f"S{i}" in row_keys]

    per_resolution = []
    best_by_method: dict[str, dict[str, object]] = {}
    for resolution in resolutions:
        if str(args.hidden_site_basis) == "pca":
            site_menu = str(args.pca_site_menu)
            pca_sites: list[Site] = []
            if site_menu in {"partition", "both"}:
                pca_sites.extend(
                    enumerate_rotated_group_sites_for_timesteps(
                        timesteps=timesteps,
                        hidden_size=int(args.hidden_size),
                        resolution=int(resolution),
                        basis_name="pca",
                    )
                )
            if site_menu in {"top_prefix", "both"}:
                pca_sites.extend(
                    enumerate_rotated_prefix_sites_for_timesteps(
                        timesteps=timesteps,
                        hidden_size=int(args.hidden_size),
                        resolution=int(resolution),
                        basis_name="pca",
                    )
                )
            hidden_sites = _dedupe_sites(tuple(pca_sites))
        else:
            hidden_sites = enumerate_group_sites_for_timesteps(
                timesteps=timesteps,
                hidden_size=int(args.hidden_size),
                resolution=int(resolution),
            )
        output_sites = enumerate_output_logit_sites(output_dim=int(args.width) + 1)
        sites: tuple[Site, ...] = tuple(hidden_sites) + tuple(output_sites)
        cost, diagnostics = _fit_cost_matrix(
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
            rotation_map=rotation_map,
        )
        resolution_result = {
            "resolution": int(resolution),
            "hidden_site_basis": str(args.hidden_site_basis),
            "pca_variant": None if str(args.hidden_site_basis) != "pca" else str(args.pca_variant),
            "pca_site_menu": None if str(args.hidden_site_basis) != "pca" else str(args.pca_site_menu),
            "n_hidden_sites": int(len(hidden_sites)),
            "n_output_sites": int(len(output_sites)),
            "n_sites_total": int(len(sites)),
            "sites": [site.key() for site in sites],
            "fit_diagnostics": diagnostics,
            "methods": {},
        }
        for method in requested_methods:
            method_best_trial_by_profile: dict[str, dict[str, object]] = {}
            method_best_key_by_profile: dict[str, tuple[float, float, float]] = {}
            trials = []
            for trial_cfg in _trial_grid(method, transport_cfg):
                if method == "ot":
                    coupling = sinkhorn_uniform_ot(
                        cost,
                        epsilon=float(trial_cfg["epsilon"]),
                        n_iter=int(transport_cfg.sinkhorn_iters),
                        temperature=float(transport_cfg.temperature),
                    )
                else:
                    coupling = sinkhorn_one_sided_uot(
                        cost,
                        epsilon=float(trial_cfg["epsilon"]),
                        beta_neural=float(trial_cfg["beta_neural"]),
                        n_iter=int(transport_cfg.sinkhorn_iters),
                        temperature=float(transport_cfg.temperature),
                    )

                profile_row_results: dict[str, dict[str, dict[str, object]]] = {profile_key: {} for profile_key in profile_keys}
                profile_calib_sens: dict[str, list[float]] = {profile_key: [] for profile_key in profile_keys}
                profile_calib_inv: dict[str, list[float]] = {profile_key: [] for profile_key in profile_keys}
                profile_test_sens: dict[str, list[float]] = {profile_key: [] for profile_key in profile_keys}
                profile_test_inv: dict[str, list[float]] = {profile_key: [] for profile_key in profile_keys}
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
                        rotation_map=rotation_map,
                    )
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
                            banks["test_positive_by_row"][row_key],
                            banks["test_invariant_by_row"][row_key],
                            device=device,
                            run_cache=run_cache,
                            rotation_map=rotation_map,
                        )
                        profile_row_results[profile_key][row_key] = {
                            "calibration": calibrated["calibration"],
                            "test": tested,
                            "top_k": int(calibrated["top_k"]),
                            "lambda": float(calibrated["lambda"]),
                            "selected_sites": calibrated["selected_sites"],
                        }
                        profile_calib_sens[profile_key].append(float(calibrated["calibration"]["sensitivity"]))
                        profile_calib_inv[profile_key].append(float(calibrated["calibration"]["invariance"]))
                        profile_test_sens[profile_key].append(float(tested["sensitivity"]))
                        profile_test_inv[profile_key].append(float(tested["invariance"]))

                profile_results: dict[str, dict[str, object]] = {}
                for profile_key in profile_keys:
                    per_row = profile_row_results[profile_key]
                    calibration_summary = {
                        "mean_sensitivity": float(sum(profile_calib_sens[profile_key]) / max(1, len(profile_calib_sens[profile_key]))),
                        "mean_invariance": float(sum(profile_calib_inv[profile_key]) / max(1, len(profile_calib_inv[profile_key]))),
                        "mean_combined": float(
                            (sum(profile_calib_sens[profile_key]) + sum(profile_calib_inv[profile_key]))
                            / max(1, 2 * len(profile_calib_sens[profile_key]))
                        ),
                    }
                    test_summary = {
                        "per_row": per_row,
                        "mean_sensitivity": float(sum(profile_test_sens[profile_key]) / max(1, len(profile_test_sens[profile_key]))),
                        "mean_invariance": float(sum(profile_test_inv[profile_key]) / max(1, len(profile_test_inv[profile_key]))),
                        "mean_combined": float(
                            (sum(profile_test_sens[profile_key]) + sum(profile_test_inv[profile_key]))
                            / max(1, 2 * len(profile_test_sens[profile_key]))
                        ),
                        "carry_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, carry_keys),
                        "output_subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, output_keys) if output_keys else None,
                    }
                    profile_results[profile_key] = {
                        "calibration": calibration_summary,
                        "test": test_summary,
                    }
                    key = (
                        float(calibration_summary["mean_combined"]),
                        float(calibration_summary["mean_sensitivity"]),
                        float(calibration_summary["mean_invariance"]),
                    )
                    if profile_key not in method_best_key_by_profile or key > method_best_key_by_profile[profile_key]:
                        method_best_key_by_profile[profile_key] = key
                        method_best_trial_by_profile[profile_key] = {
                            "config": {"method": method, "resolution": int(resolution), **trial_cfg},
                            "coupling": coupling.tolist(),
                            "profile_key": profile_key,
                            "profile_result": profile_results[profile_key],
                        }

                trial = {
                    "config": {"method": method, "resolution": int(resolution), **trial_cfg},
                    "coupling": coupling.tolist(),
                    "profile_results": profile_results,
                }
                if len(profile_keys) == 1:
                    only_profile_key = profile_keys[0]
                    trial["calibration"] = profile_results[only_profile_key]["calibration"]
                    trial["test"] = profile_results[only_profile_key]["test"]
                trials.append(trial)

            resolution_result["methods"][method] = {
                "trials": trials,
                "best_trial_by_profile": method_best_trial_by_profile,
            }
            if len(profile_keys) == 1 and profile_keys[0] in method_best_trial_by_profile:
                method_best_trial = method_best_trial_by_profile[profile_keys[0]]
                global_key = method_best_key_by_profile[profile_keys[0]]
                previous = best_by_method.get(method)
                if previous is None:
                    best_by_method[method] = method_best_trial
                else:
                    prev_profile_result = previous["profile_result"]
                    prev_calibration = prev_profile_result["calibration"]
                    prev_key = (
                        float(prev_calibration["mean_combined"]),
                        float(prev_calibration["mean_sensitivity"]),
                        float(prev_calibration["mean_invariance"]),
                    )
                    if global_key > prev_key:
                        best_by_method[method] = method_best_trial
        per_resolution.append(resolution_result)

    result = {
        "config": vars(args),
        "row_keys": row_keys,
        "family_order": list(family_order),
        "selection_profiles": [{"profile_key": key, "selection_rule": rule, "invariance_floor": floor} for key, (rule, floor) in zip(profile_keys, selection_profiles)],
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "bank_summaries": {
            "fit_by_row": _bank_summaries(
                banks["fit_by_row"],
                _partition_records(banks["fit_by_row"], active=True),
                _partition_records(banks["fit_by_row"], active=False),
            ),
            "calib_by_row": _bank_summaries(
                {
                    key: tuple(banks["calib_positive_by_row"][key] + banks["calib_invariant_by_row"][key])
                    for key in row_keys
                },
                banks["calib_positive_by_row"],
                banks["calib_invariant_by_row"],
            ),
            "test_by_row": _bank_summaries(
                {
                    key: tuple(banks["test_positive_by_row"][key] + banks["test_invariant_by_row"][key])
                    for key in row_keys
                },
                banks["test_positive_by_row"],
                banks["test_invariant_by_row"],
            ),
        },
        "training": train_summary,
        "hidden_site_basis_diagnostics": hidden_site_basis_diagnostics,
        "per_resolution": per_resolution,
        "best_by_method": best_by_method,
    }
    summary_path = out_dir / "joint_endogenous_resolution_sweep_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    compact = {
        "summary": str(summary_path),
        "factual_exact_all": result["factual_exact"]["all"],
        "best_ot": best_by_method.get("ot"),
        "best_uot": best_by_method.get("uot"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
