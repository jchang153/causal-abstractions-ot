"""MIB-style desiderata-based masking (DBM) for the MCQA task."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch
import torch.nn.functional as F

from .data import MCQAPairBank
from .intervention import (
    _collect_source_hidden_states,
    _resolve_padded_positions,
    build_position_ids_from_left_padded_attention_mask,
    gather_last_token_logits,
    resolve_transformer_layers,
    run_soft_residual_intervention,
)
from .metrics import das_metrics_from_logits
from .sites import ResidualSite


class FeatureBasis(Protocol):
    """Feature map used by DBM; reconstruction error is preserved from the base."""

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
    """MIB-compatible PCA basis (standardize observations, then truncated SVD)."""

    components: torch.Tensor  # [hidden, features]

    @property
    def feature_dim(self) -> int:
        return int(self.components.shape[1])

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        components = self.components.to(device=vectors.device, dtype=vectors.dtype)
        return vectors @ components

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        components = self.components.to(device=features.device, dtype=features.dtype)
        return features @ components.t()

    @classmethod
    def fit(cls, observations: torch.Tensor, rank: int | None = None) -> "PCABasis":
        observations = observations.detach().to(device="cpu", dtype=torch.float32)
        if observations.ndim != 2 or observations.shape[0] < 2:
            raise ValueError(f"Expected at least two 2D observations, got {tuple(observations.shape)}")
        # Match MIB's torch.var default (sample variance) and epsilon.
        std = observations.std(dim=0, unbiased=True).clamp_min(1e-6)
        standardized = (observations - observations.mean(dim=0)) / std
        max_rank = min(int(observations.shape[0]) - 1, int(observations.shape[1]))
        resolved_rank = max_rank if rank is None else min(max_rank, int(rank))
        _, _, vh = torch.linalg.svd(standardized, full_matrices=False)
        return cls(components=vh[:resolved_rank].t().contiguous())


class SAEBasis:
    def __init__(self, sae) -> None:
        self.sae = sae
        self.feature_dim = int(getattr(sae.cfg, "d_sae"))

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        dtype = next(self.sae.parameters()).dtype
        return self.sae.encode(vectors.to(dtype=dtype))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(features)


class DBMMask(torch.nn.Module):
    """One learnable logit per feature, as in the MIB implementation."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(int(feature_dim), dtype=torch.float32))

    def gate(self, *, temperature: float, hard: bool) -> torch.Tensor:
        if hard:
            return (self.logits > 0).to(self.logits.dtype)
        return torch.sigmoid(self.logits / float(temperature))


def _patch_features(
    base_vectors: torch.Tensor,
    source_vectors: torch.Tensor,
    *,
    basis: FeatureBasis,
    mask: DBMMask,
    temperature: float,
    hard: bool,
) -> torch.Tensor:
    base_features = basis.encode(base_vectors)
    source_features = basis.encode(source_vectors)
    reconstruction_error = base_vectors.to(base_features.dtype) - basis.decode(base_features)
    gate = mask.gate(temperature=temperature, hard=hard).to(
        device=base_features.device, dtype=base_features.dtype
    )
    patched_features = (1.0 - gate) * base_features + gate * source_features
    return (basis.decode(patched_features) + reconstruction_error).to(base_vectors.dtype)


def run_dbm_intervention(
    *,
    model,
    bank: MCQAPairBank,
    indices: torch.Tensor,
    layer: int,
    basis: FeatureBasis,
    mask: DBMMask,
    temperature: float,
    hard: bool,
    device: torch.device,
) -> torch.Tensor:
    base_ids = bank.base_input_ids[indices].to(device)
    base_attention = bank.base_attention_mask[indices].to(device)
    source_ids = bank.source_input_ids[indices].to(device)
    source_attention = bank.source_attention_mask[indices].to(device)
    source_hidden = _collect_source_hidden_states(
        model=model,
        source_input_ids=source_ids,
        source_attention_mask=source_attention,
        target_layers=(int(layer),),
    )[int(layer)]
    block = resolve_transformer_layers(model)[int(layer)]

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        base_positions = _resolve_padded_positions(
            base_attention, bank.base_position_by_id["last_token"][indices].to(device)
        )
        source_positions = _resolve_padded_positions(
            source_attention, bank.source_position_by_id["last_token"][indices].to(device)
        )
        patched = _patch_features(
            hidden[batch_indices, base_positions],
            source_hidden.to(hidden.device)[batch_indices, source_positions],
            basis=basis,
            mask=mask,
            temperature=temperature,
            hard=hard,
        )
        updated = hidden.clone()
        updated[batch_indices, base_positions] = patched
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    handle = block.register_forward_hook(hook)
    try:
        outputs = model(
            input_ids=base_ids,
            attention_mask=base_attention,
            position_ids=build_position_ids_from_left_padded_attention_mask(base_attention),
            use_cache=False,
        )
    finally:
        handle.remove()
    return gather_last_token_logits(outputs.logits, base_attention)


def collect_layer_observations(
    *, model, bank: MCQAPairBank, layer: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Collect paired base/source last-token states for fitting the PCA orientation."""
    collected: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, bank.size, int(batch_size)):
            indices = torch.arange(start, min(start + int(batch_size), bank.size))
            for ids, attention, positions in (
                (bank.base_input_ids, bank.base_attention_mask, bank.base_position_by_id["last_token"]),
                (bank.source_input_ids, bank.source_attention_mask, bank.source_position_by_id["last_token"]),
            ):
                ids_batch = ids[indices].to(device)
                attention_batch = attention[indices].to(device)
                outputs = model(
                    input_ids=ids_batch,
                    attention_mask=attention_batch,
                    position_ids=build_position_ids_from_left_padded_attention_mask(attention_batch),
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[int(layer) + 1]
                padded = _resolve_padded_positions(attention_batch, positions[indices].to(device))
                rows = torch.arange(hidden.shape[0], device=device)
                collected.append(hidden[rows, padded].detach().cpu().float())
    return torch.cat(collected, dim=0)


def train_dbm(
    *,
    model,
    train_bank: MCQAPairBank,
    layer: int,
    basis: FeatureBasis,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 8,
    learning_rate: float = 1e-2,
    temperature_start: float = 1.0,
    temperature_end: float = 0.01,
    regularization_coefficient: float = 0.0,
    seed: int = 0,
) -> tuple[DBMMask, dict[str, object]]:
    torch.manual_seed(int(seed))
    mask = DBMMask(basis.feature_dim).to(device)
    optimizer = torch.optim.AdamW(mask.parameters(), lr=float(learning_rate), weight_decay=0.0)
    steps_per_epoch = (train_bank.size + int(batch_size) - 1) // int(batch_size)
    total_steps = max(1, int(epochs) * steps_per_epoch)
    step = 0
    history: list[dict[str, float]] = []
    start_time = perf_counter()
    for epoch in range(int(epochs)):
        generator = torch.Generator().manual_seed(int(seed) * 1009 + epoch)
        order = torch.randperm(train_bank.size, generator=generator)
        loss_sum = 0.0
        correct = 0
        count = 0
        for offset in range(0, train_bank.size, int(batch_size)):
            indices = order[offset : offset + int(batch_size)]
            # MIB's scheduler has already advanced to index one when the first
            # batch reads its total_steps + 1 point temperature schedule.
            progress = (step + 1) / total_steps
            temperature = float(temperature_start) + progress * (
                float(temperature_end) - float(temperature_start)
            )
            logits = run_dbm_intervention(
                model=model,
                bank=train_bank,
                indices=indices,
                layer=layer,
                basis=basis,
                mask=mask,
                temperature=temperature,
                hard=False,
                device=device,
            )
            targets = train_bank.answer_token_ids[indices].to(device)
            task_loss = F.cross_entropy(logits, targets)
            sparsity = mask.gate(temperature=temperature, hard=False).abs().mean()
            loss = task_loss + float(regularization_coefficient) * sparsity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            actual = int(indices.numel())
            loss_sum += float(loss.detach()) * actual
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            count += actual
            step += 1
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_sum / max(count, 1),
            "train_exact_acc": correct / max(count, 1),
            "soft_mask_mean": float(torch.sigmoid(mask.logits).mean().detach().cpu()),
        }
        history.append(record)
        print(
            f"[DBM] layer={layer} variable={train_bank.target_var} epoch={epoch + 1}/{epochs} "
            f"loss={record['loss']:.5f} acc={record['train_exact_acc']:.4f}"
        )
    return mask, {
        "training_seconds": float(perf_counter() - start_time),
        "history": history,
        "selected_feature_count": int((mask.logits.detach() > 0).sum().item()),
        "feature_dim": int(basis.feature_dim),
    }


def evaluate_dbm(
    *, model, bank: MCQAPairBank, layer: int, basis: FeatureBasis, mask: DBMMask,
    device: torch.device, batch_size: int, tokenizer=None,
) -> dict[str, object]:
    logits: list[torch.Tensor] = []
    start_time = perf_counter()
    with torch.no_grad():
        for start in range(0, bank.size, int(batch_size)):
            indices = torch.arange(start, min(start + int(batch_size), bank.size))
            logits.append(run_dbm_intervention(
                model=model, bank=bank, indices=indices, layer=layer, basis=basis, mask=mask,
                temperature=1.0, hard=True, device=device,
            ).detach().cpu())
    metrics = das_metrics_from_logits(torch.cat(logits), bank, tokenizer=tokenizer)
    metrics["evaluation_seconds"] = float(perf_counter() - start_time)
    return metrics


def evaluate_full_layer(
    *, model, bank: MCQAPairBank, layer: int, device: torch.device,
    batch_size: int, tokenizer=None,
) -> dict[str, object]:
    logits: list[torch.Tensor] = []
    start_time = perf_counter()
    site = ResidualSite(layer=int(layer), token_position_id="last_token", dim_start=0, dim_end=int(model.config.hidden_size))
    with torch.no_grad():
        for start in range(0, bank.size, int(batch_size)):
            end = min(start + int(batch_size), bank.size)
            logits.append(run_soft_residual_intervention(
                model=model,
                base_input_ids=bank.base_input_ids[start:end].to(device),
                base_attention_mask=bank.base_attention_mask[start:end].to(device),
                source_input_ids=bank.source_input_ids[start:end].to(device),
                source_attention_mask=bank.source_attention_mask[start:end].to(device),
                site_weights={site: 1.0}, strength=1.0,
                base_position_by_id={key: value[start:end] for key, value in bank.base_position_by_id.items()},
                source_position_by_id={key: value[start:end] for key, value in bank.source_position_by_id.items()},
            ).detach().cpu())
    metrics = das_metrics_from_logits(torch.cat(logits), bank, tokenizer=tokenizer)
    metrics["evaluation_seconds"] = float(perf_counter() - start_time)
    return metrics
