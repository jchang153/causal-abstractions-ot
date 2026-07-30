from __future__ import annotations

import torch

from experiments.mcqa.mcqa_experiment.bdas import BoundlessDASConfig, BoundlessDASIntervention


def test_recommended_boundless_das_defaults_match_binary_baseline() -> None:
    config = BoundlessDASConfig()
    assert config.batch_size == 64
    assert config.epochs == 12
    assert config.rotation_learning_rate == 1e-2
    assert config.boundary_learning_rate == 1e-4
    assert config.temperature_start == 1.0
    assert config.temperature_end == 0.1
    assert config.gradient_accumulation_steps == 1
    assert config.restarts == 1


def test_hard_boundary_dimension_is_clamped() -> None:
    intervention = BoundlessDASIntervention(8, boundary_init=0.5)
    assert intervention.hard_dimension() == 4
    with torch.no_grad():
        intervention.boundary_fraction.fill_(2.0)
    assert intervention.hard_dimension() == 8
    with torch.no_grad():
        intervention.boundary_fraction.fill_(-1.0)
    assert intervention.hard_dimension() == 1


def test_full_hard_boundary_is_unit_strength_source_swap() -> None:
    torch.manual_seed(0)
    intervention = BoundlessDASIntervention(4, boundary_init=1.0)
    intervention.use_hard_mask = True
    base = torch.randn(3, 4)
    source = torch.randn(3, 4)
    actual = intervention(base, source)
    assert torch.allclose(actual, source, atol=1e-5, rtol=1e-5)


def test_soft_boundary_receives_gradient() -> None:
    intervention = BoundlessDASIntervention(4, boundary_init=0.5)
    intervention.use_hard_mask = False
    base = torch.randn(2, 4)
    source = torch.randn(2, 4)
    intervention(base, source).square().mean().backward()
    assert intervention.boundary_fraction.grad is not None
    assert torch.isfinite(intervention.boundary_fraction.grad).all()
