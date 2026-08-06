from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from experiments.mcqa.mcqa_experiment.data import (
    MCQACausalModel,
    MCQA_PARTITION_PROTOCOL,
    _base_group_id,
    _pair_row_id,
    _partition_pooled_rows,
)
from experiments.mcqa.mcqa_experiment.pca import (
    fit_pca_basis_from_states,
    load_pca_basis,
    save_pca_basis,
)

MCQA_DIR = Path(__file__).resolve().parent
if str(MCQA_DIR) not in sys.path:
    sys.path.insert(0, str(MCQA_DIR))

from mcqa_ot_pca_focus import _unique_prompt_records_for_all_variants


def _causal_input(group_index: int, *, pointer: int, symbols: tuple[str, ...]) -> dict[str, object]:
    answer_choice = f"color-{group_index}"
    choices = [f"distractor-{group_index}-{index}" for index in range(4)]
    choices[int(pointer)] = answer_choice
    result: dict[str, object] = {
        "question": (answer_choice, f"noun-{group_index}"),
        "raw_input": f"prompt-{group_index}-pointer-{pointer}-symbols-{''.join(symbols)}",
    }
    for index in range(4):
        result[f"choice{index}"] = choices[index]
        result[f"symbol{index}"] = symbols[index]
    return result


def _rows(group_count: int = 20) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index in range(group_count):
        base = _causal_input(group_index, pointer=0, symbols=("A", "B", "C", "D"))
        sources = (
            ("answerPosition", _causal_input(group_index, pointer=1, symbols=("A", "B", "C", "D"))),
            ("randomLetter", _causal_input(group_index, pointer=0, symbols=("B", "A", "C", "D"))),
            (
                "answerPosition_randomLetter",
                _causal_input(group_index, pointer=1, symbols=("C", "B", "A", "D")),
            ),
        )
        for family, source in sources:
            rows.append(
                {
                    "input": base,
                    "counterfactual_inputs": [source],
                    "counterfactual_family": family,
                    "base_row_id": f"synthetic:{group_index}",
                    "pair_row_id": f"synthetic:{group_index}:{family}",
                }
            )
    return rows


def test_mcqa_partition_is_exact_deterministic_and_base_group_disjoint() -> None:
    kwargs = {
        "causal_model": MCQACausalModel(),
        "target_vars": ("answer_pointer", "answer_token"),
        "split_seed": 17,
        "train_pool_size": 5,
        "calibration_pool_size": 4,
        "test_pool_size": 4,
        "pooled_total_examples": None,
    }
    train_rows, calibration_rows, test_rows, metadata = _partition_pooled_rows(_rows(), **kwargs)
    train_rows_2, calibration_rows_2, test_rows_2, metadata_2 = _partition_pooled_rows(_rows(), **kwargs)

    assert len(train_rows) == 5
    assert {target: len(rows) for target, rows in calibration_rows.items()} == {
        "answer_pointer": 4,
        "answer_token": 4,
    }
    assert {target: len(rows) for target, rows in test_rows.items()} == {
        "answer_pointer": 4,
        "answer_token": 4,
    }

    train_groups = {_base_group_id(row) for row in train_rows}
    calibration_groups = {
        _base_group_id(row) for rows in calibration_rows.values() for row in rows
    }
    test_groups = {_base_group_id(row) for rows in test_rows.values() for row in rows}
    assert train_groups.isdisjoint(calibration_groups)
    assert train_groups.isdisjoint(test_groups)
    assert calibration_groups.isdisjoint(test_groups)

    def selected_ids(rows_by_target: dict[str, list[dict[str, object]]]) -> dict[str, list[str]]:
        return {
            target: [_pair_row_id(row) for row in rows]
            for target, rows in rows_by_target.items()
        }

    assert [_pair_row_id(row) for row in train_rows] == [_pair_row_id(row) for row in train_rows_2]
    assert selected_ids(calibration_rows) == selected_ids(calibration_rows_2)
    assert selected_ids(test_rows) == selected_ids(test_rows_2)
    assert metadata == metadata_2
    assert metadata["protocol"] == MCQA_PARTITION_PROTOCOL


def test_pca_all_variants_uses_only_fit_assigned_base_groups() -> None:
    fit_base = {"raw_input": "fit-base"}
    heldout_base = {"raw_input": "heldout-base"}

    def row(base, source_prompt: str, family: str, group_id: str) -> dict[str, object]:
        return {
            "input": base,
            "counterfactual_inputs": [{"raw_input": source_prompt}],
            "counterfactual_family": family,
            "base_row_id": group_id,
        }

    filtered_datasets = {
        "answerPosition_train": [
            row(fit_base, "fit-ap", "answerPosition", "train:1"),
            row(heldout_base, "heldout-ap", "answerPosition", "test:2"),
        ],
        "randomLetter_train": [
            row(fit_base, "fit-at", "randomLetter", "train:1"),
        ],
        "answerPosition_randomLetter_validation": [
            row(fit_base, "fit-both", "answerPosition_randomLetter", "train:1"),
        ],
    }
    train_bank = SimpleNamespace(
        base_inputs=[fit_base, fit_base],
        base_group_ids=("train:1", "train:1"),
    )

    class Position:
        @staticmethod
        def resolve(input_dict, _tokenizer) -> int:
            return len(str(input_dict["raw_input"]))

    records = _unique_prompt_records_for_all_variants(
        train_bank=train_bank,
        filtered_datasets=filtered_datasets,
        token_position=Position(),
        tokenizer=None,
    )

    assert [record["raw_input"] for record in records] == [
        "fit-base",
        "fit-ap",
        "fit-at",
        "fit-both",
    ]


def test_pca_cache_preserves_recorded_fit_runtime(tmp_path: Path) -> None:
    basis = fit_pca_basis_from_states(
        states=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        layer=7,
        token_position_id="last_token",
    )
    basis = replace(basis, fit_runtime_seconds=12.5)
    cache_path = tmp_path / "basis.pt"

    save_pca_basis(cache_path, basis)
    loaded = load_pca_basis(cache_path)

    assert loaded.fit_runtime_seconds == 12.5
