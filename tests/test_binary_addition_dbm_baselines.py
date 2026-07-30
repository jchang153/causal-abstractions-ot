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
from experiments.binary_addition.bdas import (
    BDASConfig,
    BoundlessRotatedSubspace,
    _linear_warmup_decay,
)
from experiments.binary_addition.run_progressive_plot import parse_args as parse_progressive_args


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


def test_bdas_released_defaults() -> None:
    config = BDASConfig()
    assert config.rotation_learning_rate == 1e-3
    assert config.boundary_learning_rate == 1e-2
    assert config.boundary_init == 0.5
    assert config.boundary_penalty == 1.0
    assert config.temperature_start == 50.0
    assert config.temperature_end == 0.1
    assert config.epochs == 3
    assert config.batch_size == 16
    assert config.gradient_accumulation_steps == 4
    assert config.effective_batch_size == 64
    assert config.warmup_fraction == 0.1


def test_bdas_initial_hard_boundary_is_half_width() -> None:
    intervention = BoundlessRotatedSubspace(16, boundary_init=0.5)
    assert intervention.hard_dimension() == 8
    assert torch.equal(
        intervention.hard_mask(),
        torch.tensor([1.0] * 8 + [0.0] * 8),
    )


def test_bdas_rotation_is_orthogonal() -> None:
    intervention = BoundlessRotatedSubspace(8)
    weight = intervention.rotation_weight().detach()
    assert torch.allclose(weight @ weight.transpose(0, 1), torch.eye(8), atol=1e-5)


def test_bdas_hard_intervention_is_unit_strength_swap() -> None:
    intervention = BoundlessRotatedSubspace(4, boundary_init=1.0)
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    source = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    result = intervention.intervene(base, source, hard=True)
    assert torch.allclose(result, source, atol=1e-5)


def test_bdas_soft_boundary_receives_gradient() -> None:
    intervention = BoundlessRotatedSubspace(4, boundary_init=0.5)
    result = intervention.intervene(
        torch.zeros(1, 4),
        torch.ones(1, 4),
        hard=False,
    )
    result.sum().backward()
    assert intervention.boundary_fraction.grad is not None
    assert bool(torch.isfinite(intervention.boundary_fraction.grad).all())


def test_bdas_released_scheduler_uses_microstep_horizon() -> None:
    # The released notebook has 120 dataloader steps, 12 warmup steps, and only
    # 30 optimizer updates under four-way gradient accumulation.
    assert _linear_warmup_decay(11, warmup_steps=12, total_steps=120) == 1.0
    assert _linear_warmup_decay(12, warmup_steps=12, total_steps=120) == 1.0
    assert _linear_warmup_decay(30, warmup_steps=12, total_steps=120) == 90 / 108


def test_binary_das_defaults_match_mcqa_stopping_protocol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_progressive_plot", "--out-dir", str(tmp_path), "--hidden-size", "16"],
    )
    args = parse_progressive_args()
    assert args.das_learning_rate == 0.01
    assert args.das_restarts == 2
    assert args.das_min_epochs == 5
    assert args.das_max_epochs == 100
    assert args.das_plateau_patience == 1
    assert args.das_plateau_rel_delta == 1e-3
    assert args.das_batch_size == 64
    assert args.das_train_records_per_epoch == 0
