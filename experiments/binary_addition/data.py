from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

from .scm import BinaryAdditionExample, compute_example, intervene_carries


@dataclass(frozen=True)
class BaseSplit:
    fit: tuple[BinaryAdditionExample, ...]
    calib: tuple[BinaryAdditionExample, ...]
    test: tuple[BinaryAdditionExample, ...]

    def as_dict(self) -> dict[str, list[dict[str, object]]]:
        return {
            "fit": [ex.as_dict() for ex in self.fit],
            "calib": [ex.as_dict() for ex in self.calib],
            "test": [ex.as_dict() for ex in self.test],
        }


@dataclass(frozen=True)
class CarryPairRecord:
    base: BinaryAdditionExample
    source: BinaryAdditionExample
    carry_index: int
    forced_value: int
    counterfactual: BinaryAdditionExample
    is_active: bool

    def key(self) -> tuple[int, int, int, int, int]:
        return (self.base.a, self.base.b, self.source.a, self.source.b, self.carry_index)


@dataclass(frozen=True)
class BankSummary:
    total: int
    positive: int
    invariant: int

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "positive": self.positive, "invariant": self.invariant}


@dataclass(frozen=True)
class ExhaustiveBanks:
    width: int
    fit_joint: tuple[CarryPairRecord, ...]
    fit_by_carry: dict[int, tuple[CarryPairRecord, ...]]
    calib_positive_by_carry: dict[int, tuple[CarryPairRecord, ...]]
    calib_invariant_by_carry: dict[int, tuple[CarryPairRecord, ...]]
    test_positive_by_carry: dict[int, tuple[CarryPairRecord, ...]]
    test_invariant_by_carry: dict[int, tuple[CarryPairRecord, ...]]

    def summaries(self) -> dict[str, dict[str, dict[str, int]]]:
        def summarize(records: Iterable[CarryPairRecord]) -> dict[str, int]:
            seq = list(records)
            pos = sum(int(rec.is_active) for rec in seq)
            return BankSummary(total=len(seq), positive=pos, invariant=len(seq) - pos).as_dict()

        return {
            "fit_by_carry": {str(k): summarize(v) for k, v in self.fit_by_carry.items()},
            "calib_positive_by_carry": {str(k): summarize(v) for k, v in self.calib_positive_by_carry.items()},
            "calib_invariant_by_carry": {str(k): summarize(v) for k, v in self.calib_invariant_by_carry.items()},
            "test_positive_by_carry": {str(k): summarize(v) for k, v in self.test_positive_by_carry.items()},
            "test_invariant_by_carry": {str(k): summarize(v) for k, v in self.test_invariant_by_carry.items()},
        }


def sum_prefix(example: BinaryAdditionExample, *, carry_index: int) -> tuple[int, ...]:
    if carry_index < 1 or carry_index > example.width:
        raise ValueError(f"carry_index must be in [1, {example.width}]")
    return tuple(example.sum_bits_lsb[:carry_index])


def is_clean_source_pair(
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    *,
    carry_index: int,
) -> bool:
    return sum_prefix(base, carry_index=carry_index) == sum_prefix(source, carry_index=carry_index)


def enumerate_all_examples(width: int = 4) -> tuple[BinaryAdditionExample, ...]:
    max_value = 2**width
    return tuple(compute_example(a, b, width=width) for a in range(max_value) for b in range(max_value))


def stratified_base_split(
    examples: Iterable[BinaryAdditionExample],
    *,
    fit_count: int,
    calib_count: int,
    test_count: int,
    seed: int,
) -> BaseSplit:
    examples = list(examples)
    total = fit_count + calib_count + test_count
    if total > len(examples):
        raise ValueError(f"split counts sum to {total}, but only {len(examples)} examples were provided")

    rng = random.Random(seed)
    by_bucket: dict[int, list[BinaryAdditionExample]] = {}
    for ex in examples:
        by_bucket.setdefault(int(ex.propagation_length), []).append(ex)
    for bucket in by_bucket.values():
        rng.shuffle(bucket)

    ordered: list[BinaryAdditionExample] = []
    while any(by_bucket.values()):
        for key in sorted(by_bucket):
            bucket = by_bucket[key]
            if bucket:
                ordered.append(bucket.pop())

    ordered = ordered[:total]
    fit = tuple(ordered[:fit_count])
    calib = tuple(ordered[fit_count : fit_count + calib_count])
    test = tuple(ordered[fit_count + calib_count : fit_count + calib_count + test_count])
    return BaseSplit(fit=fit, calib=calib, test=test)


def build_carry_pair_records(
    bases: Iterable[BinaryAdditionExample],
    sources: Iterable[BinaryAdditionExample],
    *,
    width: int = 4,
    record_filter: Callable[[BinaryAdditionExample, BinaryAdditionExample, int], bool] | None = None,
) -> dict[int, tuple[CarryPairRecord, ...]]:
    sources = tuple(sources)
    records: dict[int, list[CarryPairRecord]] = {idx: [] for idx in range(1, width + 1)}
    for base in bases:
        for source in sources:
            for carry_index in range(1, width + 1):
                if record_filter is not None and not bool(record_filter(base, source, carry_index)):
                    continue
                forced_value = source.carry(carry_index)
                counterfactual = intervene_carries(base, {carry_index: forced_value})
                is_active = counterfactual.output_bits_lsb != base.output_bits_lsb
                records[carry_index].append(
                    CarryPairRecord(
                        base=base,
                        source=source,
                        carry_index=carry_index,
                        forced_value=forced_value,
                        counterfactual=counterfactual,
                        is_active=is_active,
                    )
                )
    return {idx: tuple(vals) for idx, vals in records.items()}


def build_exhaustive_banks(
    split: BaseSplit,
    *,
    width: int = 4,
    source_policy: str = "all_source",
) -> ExhaustiveBanks:
    all_sources = split.fit + split.calib + split.test

    if source_policy == "all_source":
        record_filter = None
    elif source_policy == "clean_source":
        record_filter = lambda base, source, carry_index: is_clean_source_pair(
            base,
            source,
            carry_index=carry_index,
        )
    else:
        raise ValueError(f"unknown source_policy: {source_policy!r}")

    fit_by_carry = build_carry_pair_records(split.fit, all_sources, width=width, record_filter=record_filter)
    fit_joint = tuple(rec for idx in range(1, width + 1) for rec in fit_by_carry[idx])

    calib_all = build_carry_pair_records(split.calib, all_sources, width=width, record_filter=record_filter)
    test_all = build_carry_pair_records(split.test, all_sources, width=width, record_filter=record_filter)

    def partition(records: dict[int, tuple[CarryPairRecord, ...]], active: bool) -> dict[int, tuple[CarryPairRecord, ...]]:
        return {
            idx: tuple(rec for rec in values if bool(rec.is_active) is active)
            for idx, values in records.items()
        }

    return ExhaustiveBanks(
        width=width,
        fit_joint=fit_joint,
        fit_by_carry=fit_by_carry,
        calib_positive_by_carry=partition(calib_all, True),
        calib_invariant_by_carry=partition(calib_all, False),
        test_positive_by_carry=partition(test_all, True),
        test_invariant_by_carry=partition(test_all, False),
    )
