from __future__ import annotations

from pathlib import Path
import sys

import torch

MCQA_DIR = Path(__file__).resolve().parent
if str(MCQA_DIR) not in sys.path:
    sys.path.insert(0, str(MCQA_DIR))

from experiments.mcqa.mcqa_experiment.bdas import BoundlessDASConfig, BoundlessDASIntervention
from mcqa_boundless_das import _layers_by_target


def test_recommended_boundless_das_defaults_match_binary_baseline() -> None:
    config = BoundlessDASConfig()
    assert config.batch_size == 64
    assert config.epochs == 12
    assert config.min_epochs == 5
    assert config.plateau_patience == 1
    assert config.plateau_rel_delta == 1e-3
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


def test_plot_bdas_parses_variable_specific_layers() -> None:
    assert _layers_by_target("answer_pointer:18|19,answer_token:24") == {
        "answer_pointer": (18, 19),
        "answer_token": (24,),
    }
