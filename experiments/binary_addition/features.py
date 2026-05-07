from __future__ import annotations

from typing import Sequence

import torch

from .interventions import RunCache, factual_run, evaluate_site_intervention
from .model import GRUAdder
from .scm import BinaryAdditionExample, intervene_carries
from .sites import Site


def bits_tensor(example: BinaryAdditionExample) -> torch.Tensor:
    return torch.tensor(example.output_bits_lsb, dtype=torch.float32)


def carries_tensor(example: BinaryAdditionExample) -> torch.Tensor:
    return torch.tensor(example.carries, dtype=torch.float32)


def abstract_effect_signature(base: BinaryAdditionExample, carry_index: int, forced_value: int) -> torch.Tensor:
    cf = intervene_carries(base, {int(carry_index): int(forced_value)})
    return bits_tensor(cf) - bits_tensor(base)


def abstract_carry_trace_signature(
    base: BinaryAdditionExample,
    carry_index: int,
    forced_value: int,
    *,
    local_only: bool,
) -> torch.Tensor:
    cf = intervene_carries(base, {int(carry_index): int(forced_value)})
    delta = carries_tensor(cf) - carries_tensor(base)
    if not local_only:
        return delta
    local = torch.zeros_like(delta)
    local[int(carry_index) - 1] = delta[int(carry_index) - 1]
    return local


def neural_effect_signature_for_site(
    model: GRUAdder,
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    site: Site,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    if run_cache is None:
        base_probs = factual_run(model, base, device=device).output_probs
    else:
        base_probs = run_cache.get_run(base).output_probs
    intervened_logits = evaluate_site_intervention(
        model,
        base,
        source,
        site,
        lambda_scale=1.0,
        device=device,
        run_cache=run_cache,
    )
    intervened_probs = torch.sigmoid(intervened_logits)
    return intervened_probs - base_probs


def aggregate_mean(signatures: Sequence[torch.Tensor]) -> torch.Tensor:
    if not signatures:
        raise ValueError("need at least one signature to aggregate")
    return torch.stack([sig.to(torch.float32) for sig in signatures], dim=0).mean(dim=0)
