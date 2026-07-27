"""MIB-style feature masking baselines for the recurrent binary adder."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from .interventions import RunCache
from .model import GRUAdder
from .run_joint_endogenous_resolution_sweep import EndogenousPairRecord


class FeatureBasis(Protocol):
    feature_dim: int

    def encode(self, vectors: torch.Tensor) -> torch.Tensor: ...

    def decode(self, features: torch.Tensor) -> torch.Tensor: ...


@dataclass
class IdentityBasis:
    feature_dim: int

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        return vectors

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features


@dataclass
class PCABasis:
    components: torch.Tensor

    @property
    def feature_dim(self) -> int:
        return int(self.components.shape[1])

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        rotation = self.components.to(device=vectors.device, dtype=vectors.dtype)
        return vectors @ rotation

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        rotation = self.components.to(device=features.device, dtype=features.dtype)
        return features @ rotation.t()

    @classmethod
    def fit(cls, observations: torch.Tensor, rank: int | None = None) -> "PCABasis":
        observations = observations.detach().cpu().float()
        if observations.ndim != 2 or observations.shape[0] < 2:
            raise ValueError(f"Expected at least two observations, got {tuple(observations.shape)}")
        std = observations.std(dim=0, unbiased=True).clamp_min(1e-6)
        standardized = (observations - observations.mean(dim=0)) / std
        max_rank = min(int(observations.shape[0]) - 1, int(observations.shape[1]))
        resolved_rank = max_rank if rank is None else min(max_rank, int(rank))
        _, _, vh = torch.linalg.svd(standardized, full_matrices=False)
        return cls(vh[:resolved_rank].t().contiguous())


class DBMMask(torch.nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(int(feature_dim), dtype=torch.float32))

    def gate(self, *, temperature: float, hard: bool) -> torch.Tensor:
        if hard:
            return (self.logits > 0).to(self.logits.dtype)
        return torch.sigmoid(self.logits / float(temperature))


def patch_features(
    base: torch.Tensor,
    source: torch.Tensor,
    *,
    basis: FeatureBasis,
    gate: torch.Tensor,
) -> torch.Tensor:
    base_features = basis.encode(base)
    source_features = basis.encode(source)
    base_error = base.to(base_features.dtype) - basis.decode(base_features)
    gate = gate.to(device=base_features.device, dtype=base_features.dtype)
    patched = (1.0 - gate) * base_features + gate * source_features
    return (basis.decode(patched) + base_error).to(base.dtype)


def _batch_inputs_and_sources(
    records: Sequence[EndogenousPairRecord],
    *,
    run_cache: RunCache,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_inputs = torch.cat([run_cache.get_input(record.base) for record in records], dim=0).to(device)
    source_states = torch.stack(
        [run_cache.get_run(record.source).hidden_states for record in records], dim=0
    ).to(device)
    return base_inputs, source_states


def run_feature_intervention(
    *,
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    timestep: int,
    basis: FeatureBasis,
    gate: torch.Tensor,
    run_cache: RunCache,
    device: torch.device,
) -> torch.Tensor:
    if not records:
        return torch.empty((0, model.width + 1), device=device)
    base_inputs, source_states = _batch_inputs_and_sources(records, run_cache=run_cache, device=device)
    h = torch.zeros(len(records), model.hidden_size, device=device, dtype=base_inputs.dtype)
    output_logits: list[torch.Tensor] = []
    for step in range(model.width):
        h = model.cell(base_inputs[:, step, :], h)
        if int(step) == int(timestep):
            h = patch_features(
                h,
                source_states[:, step, :].to(dtype=h.dtype),
                basis=basis,
                gate=gate,
            )
        output_logits.append(model.sum_head(h))
    output_logits.append(model.final_carry_head(h))
    return torch.cat(output_logits, dim=1)


def collect_pca_observations(
    records: Sequence[EndogenousPairRecord], *, timestep: int, run_cache: RunCache
) -> torch.Tensor:
    """Collect both sides of every unique fit pair at one recurrent timestep."""
    unique: dict[tuple[int, int, int, int, str], EndogenousPairRecord] = {}
    for record in records:
        key = (record.base.a, record.base.b, record.source.a, record.source.b, record.family)
        unique.setdefault(key, record)
    observations: list[torch.Tensor] = []
    for record in unique.values():
        observations.append(run_cache.get_run(record.base).hidden_states[int(timestep)])
        observations.append(run_cache.get_run(record.source).hidden_states[int(timestep)])
    return torch.stack(observations, dim=0).float()


def train_dbm(
    *,
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    timestep: int,
    basis: FeatureBasis,
    run_cache: RunCache,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 8,
    learning_rate: float = 1e-2,
    temperature_start: float = 1.0,
    temperature_end: float = 0.01,
    regularization_coefficient: float = 0.0,
    seed: int = 0,
) -> tuple[DBMMask, dict[str, object]]:
    records = tuple(records)
    if not records:
        raise ValueError("Cannot train DBM on an empty fit bank")
    torch.manual_seed(int(seed))
    mask = DBMMask(basis.feature_dim).to(device)
    optimizer = torch.optim.AdamW(mask.parameters(), lr=float(learning_rate), weight_decay=0.0)
    steps_per_epoch = (len(records) + int(batch_size) - 1) // int(batch_size)
    total_steps = max(1, int(epochs) * steps_per_epoch)
    step = 0
    history: list[dict[str, float]] = []
    started = perf_counter()
    for epoch in range(int(epochs)):
        generator = torch.Generator().manual_seed(int(seed) * 1009 + int(timestep) * 101 + epoch)
        order = torch.randperm(len(records), generator=generator).tolist()
        loss_sum = 0.0
        exact_hits = 0
        seen = 0
        for offset in range(0, len(records), int(batch_size)):
            batch = [records[index] for index in order[offset : offset + int(batch_size)]]
            progress = (step + 1) / total_steps
            temperature = float(temperature_start) + progress * (
                float(temperature_end) - float(temperature_start)
            )
            gate = mask.gate(temperature=temperature, hard=False)
            logits = run_feature_intervention(
                model=model,
                records=batch,
                timestep=timestep,
                basis=basis,
                gate=gate,
                run_cache=run_cache,
                device=device,
            )
            targets = torch.tensor(
                [record.counterfactual.output_bits_lsb for record in batch],
                dtype=torch.float32,
                device=device,
            )
            task_loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss = task_loss + float(regularization_coefficient) * gate.abs().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
            exact_hits += int((predictions == targets.to(torch.int64)).all(dim=1).sum().item())
            loss_sum += float(loss.detach()) * len(batch)
            seen += len(batch)
            step += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / max(seen, 1),
                "fit_exact_acc": exact_hits / max(seen, 1),
                "soft_mask_mean": float(torch.sigmoid(mask.logits).mean().detach().cpu()),
            }
        )
    hard_count = int((mask.logits.detach() > 0).sum().item())
    return mask, {
        "training_seconds": float(perf_counter() - started),
        "history": history,
        "selected_feature_count": hard_count,
        "feature_dim": int(basis.feature_dim),
        "selected_fraction": float(hard_count / max(int(basis.feature_dim), 1)),
        "soft_effective_size": float(torch.sigmoid(mask.logits).sum().detach().cpu()),
    }


def exact_match_rate(
    *,
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    timestep: int,
    basis: FeatureBasis,
    gate: torch.Tensor,
    run_cache: RunCache,
    device: torch.device,
    batch_size: int,
) -> float:
    if not records:
        return 0.0
    hits = 0
    with torch.no_grad():
        for start in range(0, len(records), int(batch_size)):
            batch = records[start : start + int(batch_size)]
            logits = run_feature_intervention(
                model=model,
                records=batch,
                timestep=timestep,
                basis=basis,
                gate=gate,
                run_cache=run_cache,
                device=device,
            )
            predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()
            targets = torch.tensor(
                [record.counterfactual.output_bits_lsb for record in batch], dtype=torch.int64
            )
            hits += int((predictions == targets).all(dim=1).sum().item())
    return float(hits / len(records))


def evaluate_candidate(
    *,
    model: GRUAdder,
    positive_records: Sequence[EndogenousPairRecord],
    invariant_records: Sequence[EndogenousPairRecord],
    timestep: int,
    basis: FeatureBasis,
    gate: torch.Tensor,
    run_cache: RunCache,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    started = perf_counter()
    sensitivity = exact_match_rate(
        model=model, records=positive_records, timestep=timestep, basis=basis, gate=gate,
        run_cache=run_cache, device=device, batch_size=batch_size,
    )
    invariance = exact_match_rate(
        model=model, records=invariant_records, timestep=timestep, basis=basis, gate=gate,
        run_cache=run_cache, device=device, batch_size=batch_size,
    )
    return {
        "sensitivity": float(sensitivity),
        "invariance": float(invariance),
        "combined": float(0.5 * (sensitivity + invariance)),
        "count_positive": int(len(positive_records)),
        "count_invariant": int(len(invariant_records)),
        "evaluation_seconds": float(perf_counter() - started),
    }
