from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .model import GRUAdder, examples_to_tensors
from .scm import BinaryAdditionExample
from .sites import CoordinateGroupSite, FullStateSite, NeuronSite, OutputLogitSite, Site


@dataclass(frozen=True)
class FactualRun:
    hidden_states: torch.Tensor
    output_logits: torch.Tensor
    output_probs: torch.Tensor


@dataclass(frozen=True)
class IntervenedRunBatch:
    hidden_states: torch.Tensor
    output_logits: torch.Tensor
    output_probs: torch.Tensor


@dataclass
class RunCache:
    runs: dict[tuple[int, int], FactualRun]
    inputs: dict[tuple[int, int], torch.Tensor]

    def key(self, example: BinaryAdditionExample) -> tuple[int, int]:
        return (int(example.a), int(example.b))

    def get_run(self, example: BinaryAdditionExample) -> FactualRun:
        return self.runs[self.key(example)]

    def get_input(self, example: BinaryAdditionExample) -> torch.Tensor:
        return self.inputs[self.key(example)]


def _single_example_tensor(example: BinaryAdditionExample, device: torch.device) -> torch.Tensor:
    x, _ = examples_to_tensors([example])
    return x.to(device)


def _stack_example_tensors(
    examples: Sequence[BinaryAdditionExample],
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    if run_cache is None:
        x, _ = examples_to_tensors(examples)
        return x.to(device)
    tensors = [run_cache.get_input(example) for example in examples]
    return torch.cat(tensors, dim=0).to(device)


def _stack_hidden_states(
    examples: Sequence[BinaryAdditionExample],
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    model: GRUAdder | None = None,
) -> torch.Tensor:
    if run_cache is None:
        if model is None:
            raise ValueError("model is required when run_cache is None")
        runs = [factual_run(model, example, device=device).hidden_states for example in examples]
    else:
        runs = [run_cache.get_run(example).hidden_states for example in examples]
    return torch.stack(runs, dim=0).to(device)


def factual_run(model: GRUAdder, example: BinaryAdditionExample, device: torch.device) -> FactualRun:
    model.eval()
    x = _single_example_tensor(example, device=device)
    with torch.no_grad():
        out = model(x)
        hidden = out["hidden_states"][0].detach().cpu()
        logits = out["output_logits"][0].detach().cpu()
        probs = torch.sigmoid(logits)
    return FactualRun(hidden_states=hidden, output_logits=logits, output_probs=probs)


def build_run_cache(
    model: GRUAdder,
    examples: Sequence[BinaryAdditionExample],
    *,
    device: torch.device,
) -> RunCache:
    unique = {(int(ex.a), int(ex.b)): ex for ex in examples}
    runs: dict[tuple[int, int], FactualRun] = {}
    inputs: dict[tuple[int, int], torch.Tensor] = {}
    for key, ex in unique.items():
        inputs[key] = _single_example_tensor(ex, device=device)
        runs[key] = factual_run(model, ex, device=device)
    return RunCache(runs=runs, inputs=inputs)


def _apply_site_delta(
    h: torch.Tensor,
    src_h: torch.Tensor,
    site: Site,
    *,
    weight: float,
    lambda_scale: float,
) -> torch.Tensor:
    if isinstance(site, FullStateSite):
        return h + float(lambda_scale) * float(weight) * (src_h - h)
    if isinstance(site, NeuronSite):
        idx = int(site.neuron_index)
        updated = h.clone()
        updated[:, idx] = updated[:, idx] + float(lambda_scale) * float(weight) * (src_h[:, idx] - updated[:, idx])
        return updated
    if isinstance(site, CoordinateGroupSite):
        updated = h.clone()
        idx = list(site.coord_indices)
        updated[:, idx] = updated[:, idx] + float(lambda_scale) * float(weight) * (src_h[:, idx] - updated[:, idx])
        return updated
    if isinstance(site, OutputLogitSite):
        return h
    raise TypeError(f"Unsupported site type: {type(site)!r}")


def intervene_with_site_handle(
    model: GRUAdder,
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    selected_sites: Sequence[tuple[Site, float]],
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    model.eval()
    if run_cache is None:
        base_x = _single_example_tensor(base, device=device)
        source_run = factual_run(model, source, device=device)
    else:
        base_x = run_cache.get_input(base)
        source_run = run_cache.get_run(source)
    source_states = source_run.hidden_states.to(device)

    by_timestep: dict[int, list[tuple[Site, float]]] = {}
    output_sites: list[tuple[OutputLogitSite, float]] = []
    for site, weight in selected_sites:
        if isinstance(site, OutputLogitSite):
            output_sites.append((site, float(weight)))
        else:
            by_timestep.setdefault(int(site.timestep), []).append((site, float(weight)))

    h = torch.zeros(1, model.hidden_size, device=device, dtype=base_x.dtype)
    sum_logits = []
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        step_sites = by_timestep.get(step, ())
        if step_sites:
            src_h = source_states[step].unsqueeze(0).to(device=device, dtype=h.dtype)
            for site, weight in step_sites:
                if abs(float(weight)) > 0.0:
                    h = _apply_site_delta(h, src_h, site, weight=float(weight), lambda_scale=float(lambda_scale))
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    output_logits = torch.cat(sum_logits + [carry_logit], dim=1)
    if output_sites:
        source_logits = source_run.output_logits.unsqueeze(0).to(device=device, dtype=output_logits.dtype)
        for site, weight in output_sites:
            idx = int(site.output_index)
            output_logits[:, idx] = output_logits[:, idx] + float(lambda_scale) * float(weight) * (
                source_logits[:, idx] - output_logits[:, idx]
            )
    output_logits = output_logits[0]
    return output_logits.detach().cpu()


def intervene_with_site_handle_batch(
    model: GRUAdder,
    bases: Sequence[BinaryAdditionExample],
    sources: Sequence[BinaryAdditionExample],
    selected_sites: Sequence[tuple[Site, float]],
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    return intervene_with_site_handle_batch_runs(
        model,
        bases,
        sources,
        selected_sites,
        lambda_scale=lambda_scale,
        device=device,
        run_cache=run_cache,
    ).output_logits


def intervene_with_site_handle_batch_runs(
    model: GRUAdder,
    bases: Sequence[BinaryAdditionExample],
    sources: Sequence[BinaryAdditionExample],
    selected_sites: Sequence[tuple[Site, float]],
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> IntervenedRunBatch:
    if len(bases) != len(sources):
        raise ValueError("bases and sources must have the same length")
    if not bases:
        empty_logits = torch.empty((0, model.width + 1), dtype=torch.float32)
        empty_hidden = torch.empty((0, model.width, model.hidden_size), dtype=torch.float32)
        return IntervenedRunBatch(
            hidden_states=empty_hidden,
            output_logits=empty_logits,
            output_probs=torch.sigmoid(empty_logits),
        )

    model.eval()
    base_x = _stack_example_tensors(bases, device=device, run_cache=run_cache)
    source_states = _stack_hidden_states(sources, device=device, run_cache=run_cache, model=model)

    by_timestep: dict[int, list[tuple[Site, float]]] = {}
    output_sites: list[tuple[OutputLogitSite, float]] = []
    for site, weight in selected_sites:
        if isinstance(site, OutputLogitSite):
            output_sites.append((site, float(weight)))
        else:
            by_timestep.setdefault(int(site.timestep), []).append((site, float(weight)))

    batch = base_x.size(0)
    h = torch.zeros(batch, model.hidden_size, device=device, dtype=base_x.dtype)
    hidden_states = []
    sum_logits = []
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        step_sites = by_timestep.get(step, ())
        if step_sites:
            src_h = source_states[:, step, :].to(device=device, dtype=h.dtype)
            for site, weight in step_sites:
                if abs(float(weight)) > 0.0:
                    h = _apply_site_delta(h, src_h, site, weight=float(weight), lambda_scale=float(lambda_scale))
        hidden_states.append(h.detach().cpu())
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    output_logits = torch.cat(sum_logits + [carry_logit], dim=1)
    if output_sites:
        if run_cache is None:
            source_logits = torch.stack([factual_run(model, source, device=device).output_logits for source in sources], dim=0)
            source_logits = source_logits.to(device=device, dtype=output_logits.dtype)
        else:
            source_logits = torch.stack([run_cache.get_run(source).output_logits for source in sources], dim=0)
            source_logits = source_logits.to(device=device, dtype=output_logits.dtype)
        for site, weight in output_sites:
            idx = int(site.output_index)
            output_logits[:, idx] = output_logits[:, idx] + float(lambda_scale) * float(weight) * (
                source_logits[:, idx] - output_logits[:, idx]
            )
    output_logits = output_logits.detach().cpu()
    hidden_tensor = torch.stack(hidden_states, dim=1)
    return IntervenedRunBatch(
        hidden_states=hidden_tensor,
        output_logits=output_logits,
        output_probs=torch.sigmoid(output_logits),
    )


def intervene_with_projection_batch_runs(
    model: GRUAdder,
    bases: Sequence[BinaryAdditionExample],
    sources: Sequence[BinaryAdditionExample],
    *,
    timestep: int,
    direction: torch.Tensor,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> IntervenedRunBatch:
    if len(bases) != len(sources):
        raise ValueError("bases and sources must have the same length")
    if not bases:
        empty_logits = torch.empty((0, model.width + 1), dtype=torch.float32)
        empty_hidden = torch.empty((0, model.width, model.hidden_size), dtype=torch.float32)
        return IntervenedRunBatch(
            hidden_states=empty_hidden,
            output_logits=empty_logits,
            output_probs=torch.sigmoid(empty_logits),
        )

    step_target = int(timestep)
    if step_target < 0 or step_target >= int(model.width):
        raise ValueError(f"timestep must be in [0, {model.width - 1}]")

    direction = direction.to(torch.float32)
    if direction.ndim != 1 or int(direction.numel()) != int(model.hidden_size):
        raise ValueError(f"direction must have shape [{model.hidden_size}], got {tuple(direction.shape)}")
    direction = direction / direction.norm().clamp_min(1e-30)

    model.eval()
    base_x = _stack_example_tensors(bases, device=device, run_cache=run_cache)
    source_states = _stack_hidden_states(sources, device=device, run_cache=run_cache, model=model)

    batch = base_x.size(0)
    h = torch.zeros(batch, model.hidden_size, device=device, dtype=base_x.dtype)
    hidden_states = []
    sum_logits = []
    direction = direction.to(device=device, dtype=h.dtype)
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        if step == step_target:
            src_h = source_states[:, step, :].to(device=device, dtype=h.dtype)
            delta = src_h - h
            coeff = torch.matmul(delta, direction).unsqueeze(1)
            h = h + float(lambda_scale) * coeff * direction.unsqueeze(0)
        hidden_states.append(h.detach().cpu())
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    output_logits = torch.cat(sum_logits + [carry_logit], dim=1).detach().cpu()
    hidden_tensor = torch.stack(hidden_states, dim=1)
    return IntervenedRunBatch(
        hidden_states=hidden_tensor,
        output_logits=output_logits,
        output_probs=torch.sigmoid(output_logits),
    )


def evaluate_site_intervention(
    model: GRUAdder,
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    site: Site,
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    return intervene_with_site_handle(
        model,
        base,
        source,
        [(site, 1.0)],
        lambda_scale=lambda_scale,
        device=device,
        run_cache=run_cache,
    )


def evaluate_topk_handle(
    model: GRUAdder,
    base: BinaryAdditionExample,
    source: BinaryAdditionExample,
    selected_sites: Sequence[tuple[Site, float]],
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    return intervene_with_site_handle(
        model,
        base,
        source,
        selected_sites,
        lambda_scale=lambda_scale,
        device=device,
        run_cache=run_cache,
    )
