from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


MCQA_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "mcqa"
sys.path.insert(0, str(MCQA_ROOT))

from mcqa_dbm_baselines import (  # noqa: E402
    build_parser,
    extract_partition_metadata,
    parse_int_grid,
    partition_sha256,
    validate_partition_reference,
    validate_sae_grid,
)
from mcqa_experiment.data import MCQA_PARTITION_PROTOCOL  # noqa: E402
from mcqa_experiment.dbm import DBMMask, IdentityBasis, PCABasis, _patch_features  # noqa: E402


def test_hard_dbm_identity_swaps_only_positive_mask_entries() -> None:
    base = torch.tensor([[1.0, 2.0, 3.0]])
    source = torch.tensor([[10.0, 20.0, 30.0]])
    mask = DBMMask(3)
    with torch.no_grad():
        mask.logits.copy_(torch.tensor([2.0, -1.0, 0.0]))
    result = _patch_features(
        base, source, basis=IdentityBasis(3), mask=mask, temperature=1.0, hard=True
    )
    assert torch.equal(result, torch.tensor([[10.0, 2.0, 3.0]]))


def test_pca_patch_preserves_base_reconstruction_residual() -> None:
    basis = PCABasis(components=torch.tensor([[1.0], [0.0]]))
    base = torch.tensor([[2.0, 7.0]])
    source = torch.tensor([[5.0, 99.0]])
    mask = DBMMask(1)
    with torch.no_grad():
        mask.logits.fill_(1.0)
    result = _patch_features(base, source, basis=basis, mask=mask, temperature=1.0, hard=True)
    assert torch.equal(result, torch.tensor([[5.0, 7.0]]))


def test_layer_grid_parser() -> None:
    assert parse_int_grid("0,2-4,3", upper_exclusive=6) == [0, 2, 3, 4]
    assert parse_int_grid("all", upper_exclusive=3) == [0, 1, 2]


def test_dbm_filter_batch_is_independent_of_evaluation_batch() -> None:
    args = build_parser().parse_args([])
    assert args.filter_batch_size == 64
    assert args.eval_batch_size == 128


def test_partition_reference_accepts_exact_full_das_partition(tmp_path: Path) -> None:
    partition = {
        "protocol": MCQA_PARTITION_PROTOCOL,
        "split_seed": 3,
        "selected_pair_rows": {
            "train": {"count": 2, "sha256": "train"},
            "calibration": {"answer_pointer": {"count": 1, "sha256": "cal"}},
            "test": {"answer_pointer": {"count": 1, "sha256": "test"}},
        },
    }
    reference = tmp_path / "full_das.json"
    reference.write_text(json.dumps({"data": {"partition": partition}}), encoding="utf-8")

    assert extract_partition_metadata({"data": {"partition": partition}}) == partition
    result = validate_partition_reference(
        actual_partition=partition,
        reference_path_template=str(reference),
        seed=3,
    )
    assert result == {
        "path": str(reference.resolve()),
        "sha256": partition_sha256(partition),
    }


def test_partition_reference_rejects_different_rows(tmp_path: Path) -> None:
    expected = {
        "protocol": MCQA_PARTITION_PROTOCOL,
        "split_seed": 0,
        "selected_pair_rows": {"train": {"count": 2, "sha256": "expected"}},
    }
    actual = {
        **expected,
        "selected_pair_rows": {"train": {"count": 2, "sha256": "actual"}},
    }
    reference = tmp_path / "reference_seed0.json"
    reference.write_text(json.dumps({"partition": expected}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the reference"):
        validate_partition_reference(
            actual_partition=actual,
            reference_path_template=str(tmp_path / "reference_seed{seed}.json"),
            seed=0,
        )


def test_all_gemma_2_layers_have_canonical_sae_aliases() -> None:
    validate_sae_grid(
        layers=list(range(26)),
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
    )
