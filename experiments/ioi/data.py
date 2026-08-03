"""Released MIB IOI data loading, deterministic splitting, and filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch


FAMILY_SPECS: dict[str, tuple[str, int, int]] = {
    "s1_io_flip": ("s1_io_flip_counterfactual", -1, 1),
    "s2_io_flip": ("s2_io_flip_counterfactual", -1, -1),
    "s1_ioi_flip_s2_ioi_flip": (
        "s1_ioi_flip_s2_ioi_flip_counterfactual",
        1,
        -1,
    ),
}


@dataclass(frozen=True)
class IOIExample:
    """One base/source pair from one MIB counterfactual family."""

    row_id: int
    split: str
    family: str
    base_prompt: str
    source_prompt: str
    base_io: str
    base_subject: str
    source_io: str
    source_subject: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IOIExample":
        return cls(**value)


@dataclass(frozen=True)
class IOIBanks:
    """Filtered fit/calibration/test pair banks grouped by family."""

    fit: dict[str, tuple[IOIExample, ...]]
    same_fit: tuple[IOIExample, ...]
    calibration: dict[str, tuple[IOIExample, ...]]
    test: dict[str, tuple[IOIExample, ...]]
    signature_fit: dict[str, tuple[IOIExample, ...]]
    metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "fit": {key: [item.as_dict() for item in rows] for key, rows in self.fit.items()},
            "same_fit": [item.as_dict() for item in self.same_fit],
            "calibration": {
                key: [item.as_dict() for item in rows] for key, rows in self.calibration.items()
            },
            "test": {key: [item.as_dict() for item in rows] for key, rows in self.test.items()},
            "signature_fit": {
                key: [item.as_dict() for item in rows] for key, rows in self.signature_fit.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IOIBanks":
        def restore(group: str) -> dict[str, tuple[IOIExample, ...]]:
            raw = value[group]
            return {
                key: tuple(IOIExample.from_dict(item) for item in rows)
                for key, rows in raw.items()
            }

        return cls(
            fit=restore("fit"),
            same_fit=tuple(IOIExample.from_dict(item) for item in value["same_fit"]),
            calibration=restore("calibration"),
            test=restore("test"),
            signature_fit=restore("signature_fit"),
            metadata=dict(value["metadata"]),
        )


def _correct_and_other(choices: Sequence[str], answer_key: int) -> tuple[str, str]:
    if len(choices) != 2 or int(answer_key) not in (0, 1):
        raise ValueError(f"Expected two IOI choices and answerKey in {{0,1}}, got {choices}, {answer_key}")
    answer_key = int(answer_key)
    return str(choices[answer_key]), str(choices[1 - answer_key])


def row_to_examples(row: dict[str, object], *, row_id: int, split: str) -> dict[str, IOIExample]:
    """Expand one released MIB row into the three benchmark pair families."""

    base_io, base_subject = _correct_and_other(row["choices"], int(row["answerKey"]))
    result: dict[str, IOIExample] = {}
    for family, (column, _position_signal, _token_signal) in FAMILY_SPECS.items():
        source = row[column]
        source_io, source_subject = _correct_and_other(source["choices"], int(source["answerKey"]))
        result[family] = IOIExample(
            row_id=int(row_id),
            split=str(split),
            family=family,
            base_prompt=str(row["prompt"]),
            source_prompt=str(source["prompt"]),
            base_io=base_io,
            base_subject=base_subject,
            source_io=source_io,
            source_subject=source_subject,
        )
    return result


def family_signals(family: str) -> tuple[int, int]:
    """Return ``(position_signal, token_signal)`` for a source family."""

    try:
        _, position_signal, token_signal = FAMILY_SPECS[family]
    except KeyError as exc:
        raise ValueError(f"Unknown IOI family: {family}") from exc
    return int(position_signal), int(token_signal)


def mib_name_token_id(tokenizer, name: str) -> int:
    """Reproduce MIB's operational ``tokenizer.encode(name)[0]`` convention."""

    token_ids = tokenizer.encode(str(name), add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"MIB IOI name must encode to one token: {name!r} -> {token_ids}")
    return int(token_ids[0])


def validate_name_tokens(tokenizer, examples: Iterable[IOIExample]) -> None:
    names = {
        name
        for example in examples
        for name in (
            example.base_io,
            example.base_subject,
            example.source_io,
            example.source_subject,
        )
    }
    for name in sorted(names):
        mib_name_token_id(tokenizer, name)


def iter_minibatches(
    rows: Sequence[IOIExample], batch_size: int, *, indices: Sequence[int] | None = None
) -> Iterator[list[IOIExample]]:
    ordered = list(range(len(rows))) if indices is None else [int(index) for index in indices]
    for start in range(0, len(ordered), int(batch_size)):
        yield [rows[index] for index in ordered[start : start + int(batch_size)]]


def tokenize_prompts(tokenizer, prompts: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
        add_special_tokens=False,
    )
    attention_mask = encoded["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
    }


def _greedy_predictions(model, tokenizer, prompts: Sequence[str], device: torch.device) -> list[str]:
    inputs = tokenize_prompts(tokenizer, prompts, device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, -1, :]
    token_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    return [tokenizer.decode([int(token_id)]) for token_id in token_ids]


def filter_correct_pairs(
    model,
    tokenizer,
    rows: Sequence[IOIExample],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[IOIExample, ...]:
    """Match MIB's greedy one-token base-and-source correctness filter."""

    kept: list[IOIExample] = []
    for batch in iter_minibatches(rows, batch_size):
        base_predictions = _greedy_predictions(
            model, tokenizer, [item.base_prompt for item in batch], device
        )
        source_predictions = _greedy_predictions(
            model, tokenizer, [item.source_prompt for item in batch], device
        )
        for item, base_prediction, source_prediction in zip(
            batch, base_predictions, source_predictions
        ):
            if item.base_io in base_prediction and item.source_io in source_prediction:
                kept.append(item)
    return tuple(kept)


def _expand_rows(dataset, indices: Sequence[int], split: str) -> dict[str, tuple[IOIExample, ...]]:
    groups: dict[str, list[IOIExample]] = {family: [] for family in FAMILY_SPECS}
    for row_id in indices:
        expanded = row_to_examples(dataset[int(row_id)], row_id=int(row_id), split=split)
        for family, example in expanded.items():
            groups[family].append(example)
    return {family: tuple(rows) for family, rows in groups.items()}


def _filter_groups(model, tokenizer, groups, *, device, batch_size):
    return {
        family: filter_correct_pairs(
            model, tokenizer, rows, device=device, batch_size=int(batch_size)
        )
        for family, rows in groups.items()
    }


def deterministic_public_split(
    row_count: int, *, calibration_rows: int = 2_000, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Shuffle before filtering and return disjoint calibration/test row indices."""

    if int(row_count) < 2:
        raise ValueError("At least two public-test rows are required")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(row_count)).tolist()
    calibration_count = min(int(calibration_rows), max(1, len(order) // 2))
    return order[:calibration_count], order[calibration_count:]


def as_no_change(example: IOIExample) -> IOIExample:
    """Turn a released base row into MIB's independently filtered no-change pair."""

    return IOIExample(
        row_id=example.row_id,
        split=example.split,
        family="same",
        base_prompt=example.base_prompt,
        source_prompt=example.base_prompt,
        base_io=example.base_io,
        base_subject=example.base_subject,
        source_io=example.base_io,
        source_subject=example.base_subject,
    )


def load_and_filter_banks(
    model,
    tokenizer,
    *,
    device: torch.device,
    split_seed: int = 0,
    calibration_rows: int = 2_000,
    signature_rows: int = 1_000,
    filter_batch_size: int = 64,
    dataset_name: str = "mib-bench/ioi",
    dataset_revision: str = "5024626",
    hf_token: str | None = None,
    quick_rows: int | None = None,
) -> IOIBanks:
    """Load released rows, split public test, then apply family-wise MIB filtering."""

    from datasets import load_dataset

    train_dataset = load_dataset(
        dataset_name, split="train", revision=dataset_revision, token=hf_token
    )
    test_dataset = load_dataset(
        dataset_name, split="test", revision=dataset_revision, token=hf_token
    )
    if quick_rows is None and (len(train_dataset) != 10_000 or len(test_dataset) != 10_000):
        raise ValueError(
            "Reference IOI protocol requires 10,000 released train and 10,000 public-test rows; "
            f"revision {dataset_revision!r} yielded {len(train_dataset)} and {len(test_dataset)}"
        )
    if quick_rows is not None:
        train_dataset = train_dataset.select(range(min(int(quick_rows), len(train_dataset))))
        test_dataset = test_dataset.select(range(min(2 * int(quick_rows), len(test_dataset))))

    calibration_indices, test_indices = deterministic_public_split(
        len(test_dataset), calibration_rows=calibration_rows, seed=split_seed
    )
    train_indices = list(range(len(train_dataset)))

    signature_rng = np.random.default_rng(int(split_seed))
    signature_count = min(int(signature_rows), len(train_indices))
    signature_ids = set(
        int(index)
        for index in signature_rng.choice(train_indices, size=signature_count, replace=False).tolist()
    )

    raw_fit = _expand_rows(train_dataset, train_indices, "fit")
    raw_calibration = _expand_rows(test_dataset, calibration_indices, "calibration")
    raw_test = _expand_rows(test_dataset, test_indices, "test")
    validate_name_tokens(
        tokenizer,
        (
            item
            for groups in (raw_fit, raw_calibration, raw_test)
            for rows in groups.values()
            for item in rows
        ),
    )
    fit = _filter_groups(
        model, tokenizer, raw_fit, device=device, batch_size=filter_batch_size
    )
    first_family = next(iter(raw_fit))
    same_fit = filter_correct_pairs(
        model,
        tokenizer,
        tuple(as_no_change(item) for item in raw_fit[first_family]),
        device=device,
        batch_size=filter_batch_size,
    )
    calibration = _filter_groups(
        model, tokenizer, raw_calibration, device=device, batch_size=filter_batch_size
    )
    test = _filter_groups(
        model, tokenizer, raw_test, device=device, batch_size=filter_batch_size
    )
    signature_fit = {
        family: tuple(item for item in rows if item.row_id in signature_ids)
        for family, rows in fit.items()
    }

    all_examples = list(same_fit) + [
        item
        for groups in (fit, calibration, test)
        for rows in groups.values()
        for item in rows
    ]
    validate_name_tokens(tokenizer, all_examples)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_fingerprints": {
            "train": getattr(train_dataset, "_fingerprint", None),
            "test": getattr(test_dataset, "_fingerprint", None),
        },
        "split_seed": int(split_seed),
        "raw_counts": {
            "fit": len(train_indices),
            "calibration": len(calibration_indices),
            "test": len(test_indices),
            "signature_fit": len(signature_ids),
        },
        "filtered_counts": {
            split: {family: len(rows) for family, rows in groups.items()}
            for split, groups in (
                ("fit", fit),
                ("calibration", calibration),
                ("test", test),
                ("signature_fit", signature_fit),
            )
        },
        "same_fit_filtered_count": len(same_fit),
        "calibration_row_ids": [int(index) for index in calibration_indices],
        "test_row_ids": [int(index) for index in test_indices],
        "signature_fit_row_ids": sorted(signature_ids),
    }
    return IOIBanks(
        fit=fit,
        same_fit=same_fit,
        calibration=calibration,
        test=test,
        signature_fit=signature_fit,
        metadata=metadata,
    )
