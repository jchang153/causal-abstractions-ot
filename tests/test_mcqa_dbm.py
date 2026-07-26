from __future__ import annotations

import sys
from pathlib import Path

import torch


MCQA_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "mcqa"
sys.path.insert(0, str(MCQA_ROOT))

from mcqa_dbm_baselines import parse_int_grid, validate_sae_grid  # noqa: E402
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


def test_all_gemma_2_layers_have_canonical_sae_aliases() -> None:
    validate_sae_grid(
        layers=list(range(26)),
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
    )
