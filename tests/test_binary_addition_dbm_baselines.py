from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.binary_addition.dbm_baselines import (
    DBMMask,
    IdentityBasis,
    PCABasis,
    patch_features,
)


def test_binary_dbm_hard_identity_mask_swaps_selected_coordinates() -> None:
    base = torch.tensor([[1.0, 2.0, 3.0]])
    source = torch.tensor([[10.0, 20.0, 30.0]])
    mask = DBMMask(3)
    with torch.no_grad():
        mask.logits.copy_(torch.tensor([1.0, -1.0, 1.0]))
    result = patch_features(
        base,
        source,
        basis=IdentityBasis(3),
        gate=mask.gate(temperature=1.0, hard=True),
    )
    assert torch.equal(result, torch.tensor([[10.0, 2.0, 30.0]]))


def test_binary_pca_patch_preserves_base_complement() -> None:
    basis = PCABasis(torch.tensor([[1.0], [0.0]]))
    result = patch_features(
        torch.tensor([[2.0, 7.0]]),
        torch.tensor([[5.0, 99.0]]),
        basis=basis,
        gate=torch.ones(1),
    )
    assert torch.equal(result, torch.tensor([[5.0, 7.0]]))


def test_soft_gate_is_differentiable() -> None:
    mask = DBMMask(2)
    result = patch_features(
        torch.zeros(1, 2),
        torch.ones(1, 2),
        basis=IdentityBasis(2),
        gate=mask.gate(temperature=0.5, hard=False),
    )
    result.sum().backward()
    assert mask.logits.grad is not None
    assert bool(torch.all(mask.logits.grad > 0))
