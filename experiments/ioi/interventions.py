"""Exact GPT-2 pre-``c_proj`` head interventions and joint DAS training."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .causal import LinearCausalModel
from .data import IOIExample, iter_minibatches, mib_name_token_id


Head = tuple[int, int]
ORACLE_HEADS: tuple[Head, ...] = ((7, 3), (7, 9), (8, 6), (8, 10))


def all_gpt2_heads(n_layer: int = 12, n_head: int = 12) -> tuple[Head, ...]:
    return tuple((layer, head) for layer in range(int(n_layer)) for head in range(int(n_head)))


def head_label(head: Head) -> str:
    return f"L{int(head[0])}H{int(head[1])}"


def parse_head_label(label: str) -> Head:
    layer_text, head_text = label.upper().split("H", maxsplit=1)
    return int(layer_text.removeprefix("L")), int(head_text)


def _tokenize_pair_prompts(tokenizer, base_prompts, source_prompts, device):
    count = len(base_prompts)
    encoded = tokenizer(
        list(base_prompts) + list(source_prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
        add_special_tokens=False,
    )
    attention_mask = encoded["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    result = {}
    for name, values in (
        ("input_ids", encoded["input_ids"]),
        ("attention_mask", attention_mask),
        ("position_ids", position_ids),
    ):
        values = values.to(device)
        result[f"base_{name}"] = values[:count]
        result[f"source_{name}"] = values[count:]
    return result


def _model_inputs(batch: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch[f"{prefix}_input_ids"],
        "attention_mask": batch[f"{prefix}_attention_mask"],
        "position_ids": batch[f"{prefix}_position_ids"],
    }


class HeadRotations(nn.Module):
    """One independently orthogonal low-rank DAS projector per selected head."""

    def __init__(self, heads: Sequence[Head], *, head_dim: int = 64, subspace_dim: int = 32):
        super().__init__()
        if not 0 < int(subspace_dim) <= int(head_dim):
            raise ValueError(f"subspace_dim must be in [1,{head_dim}], got {subspace_dim}")
        self.heads = tuple((int(layer), int(head)) for layer, head in heads)
        self.head_dim = int(head_dim)
        self.subspace_dim = int(subspace_dim)
        modules = {}
        for head in self.heads:
            linear = nn.Linear(self.head_dim, self.subspace_dim, bias=False)
            modules[head_label(head)] = torch.nn.utils.parametrizations.orthogonal(linear)
        self.projectors = nn.ModuleDict(modules)

    def basis(self, head: Head) -> torch.Tensor:
        return self.projectors[head_label(head)].weight

    def intervene(self, head: Head, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        basis = self.basis(head)
        compute_dtype = basis.dtype
        base_compute = base.to(compute_dtype)
        source_compute = source.to(compute_dtype)
        base_features = base_compute @ basis.t()
        source_features = source_compute @ basis.t()
        return (base_compute + (source_features - base_features) @ basis).to(base.dtype)

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def metadata(self) -> dict[str, object]:
        return {
            "heads": [head_label(head) for head in self.heads],
            "head_dim": self.head_dim,
            "subspace_dim": self.subspace_dim,
            "parameter_count": self.parameter_count,
        }


class GPT2HeadIntervention:
    """Collect and replace pre-output-projection GPT-2 head vectors."""

    def __init__(self, model, *, head_dim: int = 64):
        self.model = model
        self.head_dim = int(head_dim)

    def _projection(self, layer: int):
        return self.model.transformer.h[int(layer)].attn.c_proj

    @contextmanager
    def _collect_source(self, layers: Sequence[int]):
        collected: dict[int, torch.Tensor] = {}
        handles = []
        for layer in sorted(set(int(layer) for layer in layers)):
            def capture(_module, inputs, *, layer_index=layer):
                collected[layer_index] = inputs[0].detach()

            handles.append(self._projection(layer).register_forward_pre_hook(capture))
        try:
            yield collected
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _patch_base(
        self,
        source_by_layer: Mapping[int, torch.Tensor],
        heads: Sequence[Head],
        rotations: HeadRotations | None,
    ):
        by_layer: dict[int, list[int]] = {}
        for layer, head in heads:
            by_layer.setdefault(int(layer), []).append(int(head))
        handles = []
        for layer, layer_heads in by_layer.items():
            def patch(_module, inputs, *, layer_index=layer, selected_heads=tuple(layer_heads)):
                base = inputs[0]
                source = source_by_layer[layer_index].to(device=base.device, dtype=base.dtype)
                if source.shape != base.shape:
                    raise ValueError(
                        f"Base/source head-output shapes differ at layer {layer_index}: "
                        f"{tuple(base.shape)} vs {tuple(source.shape)}"
                    )
                updated = base.clone()
                for head_index in selected_heads:
                    start = int(head_index) * self.head_dim
                    end = start + self.head_dim
                    if rotations is None:
                        updated[..., start:end] = source[..., start:end]
                    else:
                        updated[..., start:end] = rotations.intervene(
                            (layer_index, head_index),
                            base[..., start:end],
                            source[..., start:end],
                        )
                return (updated, *inputs[1:])

            handles.append(self._projection(layer).register_forward_pre_hook(patch))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        *,
        batch_inputs: Mapping[str, torch.Tensor],
        heads: Sequence[Head],
        rotations: HeadRotations | None,
    ) -> torch.Tensor:
        layers = [layer for layer, _head in heads]
        with self._collect_source(layers) as source_by_layer:
            with torch.no_grad():
                self.model(**_model_inputs(batch_inputs, "source"), use_cache=False)
        with self._patch_base(source_by_layer, heads, rotations):
            outputs = self.model(**_model_inputs(batch_inputs, "base"), use_cache=False)
        return outputs.logits[:, -1, :]

    def factual(self, *, batch_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**_model_inputs(batch_inputs, "base"), use_cache=False).logits[:, -1, :]


def batch_logit_differences(logits: torch.Tensor, tokenizer, rows: Sequence[IOIExample]) -> torch.Tensor:
    io_ids = torch.tensor(
        [mib_name_token_id(tokenizer, item.base_io) for item in rows],
        dtype=torch.long,
        device=logits.device,
    )
    subject_ids = torch.tensor(
        [mib_name_token_id(tokenizer, item.base_subject) for item in rows],
        dtype=torch.long,
        device=logits.device,
    )
    indices = torch.arange(len(rows), device=logits.device)
    return logits[indices, io_ids] - logits[indices, subject_ids]


def intervention_logit_differences(
    model,
    tokenizer,
    rows: Sequence[IOIExample],
    *,
    heads: Sequence[Head] = (),
    rotations: HeadRotations | None = None,
    device: torch.device,
    batch_size: int,
    factual: bool = False,
) -> list[float]:
    runner = GPT2HeadIntervention(model, head_dim=int(model.config.n_embd // model.config.n_head))
    values: list[float] = []
    for batch in iter_minibatches(rows, int(batch_size)):
        inputs = _tokenize_pair_prompts(
            tokenizer,
            [item.base_prompt for item in batch],
            [item.source_prompt for item in batch],
            device,
        )
        context = torch.inference_mode() if rotations is None else torch.no_grad()
        with context:
            logits = (
                runner.factual(batch_inputs=inputs)
                if factual
                else runner.forward(batch_inputs=inputs, heads=heads, rotations=rotations)
            )
            differences = batch_logit_differences(logits, tokenizer, batch)
        values.extend(float(value) for value in differences.detach().cpu().tolist())
    return values


def grouped_examples(groups: Mapping[str, Sequence[IOIExample]]) -> list[IOIExample]:
    return [item for family in groups for item in groups[family]]


def evaluate_mse(
    model,
    tokenizer,
    groups: Mapping[str, Sequence[IOIExample]],
    *,
    variable: str,
    causal_model: LinearCausalModel,
    heads: Sequence[Head],
    rotations: HeadRotations | None,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    per_family = {}
    for family, rows in groups.items():
        values = intervention_logit_differences(
            model,
            tokenizer,
            rows,
            heads=heads,
            rotations=rotations,
            device=device,
            batch_size=batch_size,
        )
        target = causal_model.intervention_target(variable, family)
        errors = np.square(np.asarray(values, dtype=np.float64) - float(target))
        per_family[family] = {
            "mse": float(errors.mean()) if len(errors) else float("nan"),
            "count": int(len(errors)),
            "mean_logit_difference": float(np.mean(values)) if values else float("nan"),
            "target_logit_difference": float(target),
        }
    finite = [record["mse"] for record in per_family.values() if np.isfinite(record["mse"])]
    return {
        "variable": variable,
        "per_family": per_family,
        "macro_mse": float(np.mean(finite)) if finite else float("nan"),
    }


@dataclass(frozen=True)
class DASTrainConfig:
    subspace_dim: int = 32
    epochs: int = 1
    learning_rate: float = 1.0
    effective_batch_size: int = 1024
    micro_batch_size: int = 8
    seed: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def train_joint_das(
    model,
    tokenizer,
    groups: Mapping[str, Sequence[IOIExample]],
    *,
    variable: str,
    causal_model: LinearCausalModel,
    heads: Sequence[Head],
    device: torch.device,
    config: DASTrainConfig,
) -> tuple[HeadRotations, dict[str, object]]:
    """Train all selected per-head rotations jointly for one causal variable."""

    torch.manual_seed(int(config.seed))
    rows = grouped_examples(groups)
    if not rows:
        raise ValueError("Cannot train DAS with an empty fit bank")
    rotations = HeadRotations(
        heads,
        head_dim=int(model.config.n_embd // model.config.n_head),
        subspace_dim=int(config.subspace_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        rotations.parameters(), lr=float(config.learning_rate), weight_decay=0.0
    )
    runner = GPT2HeadIntervention(model, head_dim=int(model.config.n_embd // model.config.n_head))
    rng = np.random.default_rng(int(config.seed))
    epoch_losses: list[float] = []
    optimizer_steps = 0
    started = perf_counter()
    for _epoch in range(int(config.epochs)):
        order = rng.permutation(len(rows)).tolist()
        epoch_error_sum = 0.0
        epoch_count = 0
        for chunk_start in range(0, len(order), int(config.effective_batch_size)):
            chunk_indices = order[chunk_start : chunk_start + int(config.effective_batch_size)]
            optimizer.zero_grad(set_to_none=True)
            chunk_size = len(chunk_indices)
            for batch in iter_minibatches(
                rows, int(config.micro_batch_size), indices=chunk_indices
            ):
                inputs = _tokenize_pair_prompts(
                    tokenizer,
                    [item.base_prompt for item in batch],
                    [item.source_prompt for item in batch],
                    device,
                )
                logits = runner.forward(batch_inputs=inputs, heads=heads, rotations=rotations)
                actual = batch_logit_differences(logits, tokenizer, batch)
                targets = torch.tensor(
                    [causal_model.intervention_target(variable, item.family) for item in batch],
                    dtype=actual.dtype,
                    device=actual.device,
                )
                squared = torch.square(actual - targets)
                (squared.sum() / float(chunk_size)).backward()
                epoch_error_sum += float(squared.detach().sum().cpu())
                epoch_count += len(batch)
            optimizer.step()
            optimizer_steps += 1
        epoch_losses.append(float(epoch_error_sum / max(1, epoch_count)))
    return rotations, {
        "variable": variable,
        "heads": [head_label(head) for head in heads],
        "config": config.as_dict(),
        "epoch_mse": epoch_losses,
        "optimizer_steps": int(optimizer_steps),
        "runtime_seconds": float(perf_counter() - started),
        "parameter_count": rotations.parameter_count,
    }


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_gpt2(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    freeze_model(model)
    if int(model.config.n_layer) != 12 or int(model.config.n_head) != 12:
        raise ValueError("Blind IOI currently requires GPT-2 Small with 12 layers and 12 heads")
    return model, tokenizer
