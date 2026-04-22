"""PCA utilities for rotated-basis MCQA OT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data import MCQAPairBank


@dataclass(frozen=True)
class LayerPCABasis:
    """Cached PCA basis for one layer/token-position residual site."""

    basis_id: str
    layer: int
    token_position_id: str
    hidden_size: int
    rank: int
    mean: torch.Tensor
    components: torch.Tensor
    singular_values: torch.Tensor
    explained_variance: torch.Tensor
    num_fit_states: int

    def to_payload(self) -> dict[str, object]:
        return {
            "basis_id": str(self.basis_id),
            "layer": int(self.layer),
            "token_position_id": str(self.token_position_id),
            "hidden_size": int(self.hidden_size),
            "rank": int(self.rank),
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "singular_values": self.singular_values.detach().cpu(),
            "explained_variance": self.explained_variance.detach().cpu(),
            "num_fit_states": int(self.num_fit_states),
        }

    @staticmethod
    def from_payload(payload: dict[str, object]) -> "LayerPCABasis":
        return LayerPCABasis(
            basis_id=str(payload["basis_id"]),
            layer=int(payload["layer"]),
            token_position_id=str(payload["token_position_id"]),
            hidden_size=int(payload["hidden_size"]),
            rank=int(payload["rank"]),
            mean=torch.as_tensor(payload["mean"], dtype=torch.float32),
            components=torch.as_tensor(payload["components"], dtype=torch.float32),
            singular_values=torch.as_tensor(payload["singular_values"], dtype=torch.float32),
            explained_variance=torch.as_tensor(payload["explained_variance"], dtype=torch.float32),
            num_fit_states=int(payload["num_fit_states"]),
        )


def _resolve_transformer_layers(model) -> list[torch.nn.Module]:
    for path in (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        current = model
        found = True
        for attribute in path:
            if not hasattr(current, attribute):
                found = False
                break
            current = getattr(current, attribute)
        if found:
            return list(current)
    raise ValueError(f"Could not locate transformer layer stack on model type {type(model)!r}")


def _build_position_ids_from_left_padded_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _resolve_padded_positions(attention_mask: torch.Tensor, unpadded_positions: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape {tuple(attention_mask.shape)}")
    if unpadded_positions.ndim != 1:
        raise ValueError(f"Expected 1D positions, got shape {tuple(unpadded_positions.shape)}")
    if attention_mask.shape[0] != unpadded_positions.shape[0]:
        raise ValueError(
            f"Batch mismatch between attention_mask {tuple(attention_mask.shape)} "
            f"and positions {tuple(unpadded_positions.shape)}"
        )
    first_nonpad = torch.argmax(attention_mask.to(torch.long), dim=1)
    return first_nonpad + unpadded_positions.to(torch.long)


def fit_pca_basis_from_states(
    *,
    states: torch.Tensor,
    layer: int,
    token_position_id: str,
    basis_id: str | None = None,
    rank_tol: float | None = None,
) -> LayerPCABasis:
    """Fit a PCA basis from explicit residual states."""
    if states.ndim != 2:
        raise ValueError(f"Expected [num_states, hidden_size] states, got shape {tuple(states.shape)}")
    num_states, hidden_size = int(states.shape[0]), int(states.shape[1])
    if num_states < 2:
        raise ValueError(f"Need at least 2 states for PCA, got {num_states}")

    states_f32 = states.to(dtype=torch.float32)
    mean = states_f32.mean(dim=0)
    centered = states_f32 - mean
    _u, singular_values, v_h = torch.linalg.svd(centered, full_matrices=False)
    if singular_values.numel() == 0:
        raise ValueError("SVD returned no singular values")
    max_singular = float(singular_values.max().item())
    if max_singular <= 0.0:
        raise ValueError("Residual states are constant; PCA basis rank is zero")

    eps = torch.finfo(singular_values.dtype).eps
    threshold = float(rank_tol) if rank_tol is not None else max(num_states, hidden_size) * eps * max_singular
    keep_mask = singular_values > float(threshold)
    rank = int(keep_mask.sum().item())
    if rank <= 0:
        raise ValueError(
            f"No stable singular directions survived thresholding (threshold={float(threshold):.3e})"
        )
    singular_values = singular_values[:rank].contiguous()
    components = v_h[:rank].transpose(0, 1).contiguous()
    explained_variance = (singular_values.pow(2) / max(num_states - 1, 1)).contiguous()

    eye = torch.eye(rank, dtype=components.dtype)
    gram = components.transpose(0, 1) @ components
    if not torch.allclose(gram, eye, atol=1e-5, rtol=1e-4):
        raise ValueError("PCA components are not numerically orthonormal")

    resolved_basis_id = basis_id or f"L{int(layer)}:{str(token_position_id)}:pca-r{int(rank)}"
    return LayerPCABasis(
        basis_id=str(resolved_basis_id),
        layer=int(layer),
        token_position_id=str(token_position_id),
        hidden_size=int(hidden_size),
        rank=int(rank),
        mean=mean.detach().cpu(),
        components=components.detach().cpu(),
        singular_values=singular_values.detach().cpu(),
        explained_variance=explained_variance.detach().cpu(),
        num_fit_states=int(num_states),
    )


def project_centered_to_pca(vectors: torch.Tensor, basis: LayerPCABasis) -> torch.Tensor:
    """Project residual vectors into the centered PCA basis."""
    if vectors.shape[-1] != int(basis.hidden_size):
        raise ValueError(
            f"Expected vectors with hidden_size={int(basis.hidden_size)}, got shape {tuple(vectors.shape)}"
        )
    mean = basis.mean.to(device=vectors.device, dtype=vectors.dtype)
    components = basis.components.to(device=vectors.device, dtype=vectors.dtype)
    return (vectors - mean) @ components


def pca_delta_to_residual(delta_z: torch.Tensor, basis: LayerPCABasis, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Map a PCA-space delta back into residual space."""
    if delta_z.shape[-1] != int(basis.rank):
        raise ValueError(f"Expected delta_z width={int(basis.rank)}, got shape {tuple(delta_z.shape)}")
    target_dtype = dtype or delta_z.dtype
    components = basis.components.to(device=delta_z.device, dtype=delta_z.dtype)
    delta_h = delta_z @ components.transpose(0, 1)
    return delta_h.to(dtype=target_dtype)


def apply_rotated_component_update(
    *,
    base_vectors: torch.Tensor,
    source_vectors: torch.Tensor,
    basis: LayerPCABasis,
    component_segments: list[tuple[int, int, float]],
    strength: float,
) -> torch.Tensor:
    """Apply a PCA-space intervention while preserving the orthogonal complement."""
    if base_vectors.shape != source_vectors.shape:
        raise ValueError(
            f"Base/source shape mismatch: {tuple(base_vectors.shape)} vs {tuple(source_vectors.shape)}"
        )
    if base_vectors.shape[-1] != int(basis.hidden_size):
        raise ValueError(
            f"Expected vectors with hidden_size={int(basis.hidden_size)}, got shape {tuple(base_vectors.shape)}"
        )
    compute_dtype = torch.float32
    mean = basis.mean.to(device=base_vectors.device, dtype=compute_dtype)
    components = basis.components.to(device=base_vectors.device, dtype=compute_dtype)
    base_centered = base_vectors.to(compute_dtype) - mean
    source_centered = source_vectors.to(compute_dtype) - mean
    z_base = base_centered @ components
    z_source = source_centered @ components
    delta_z = torch.zeros_like(z_base)
    for component_start, component_end, weight in component_segments:
        start = int(component_start)
        end = int(component_end)
        if start < 0 or end > int(basis.rank) or start >= end:
            raise ValueError(
                f"Invalid PCA component range [{start}:{end}] for rank={int(basis.rank)}"
            )
        delta_z[:, start:end] += float(strength) * float(weight) * (z_source[:, start:end] - z_base[:, start:end])
    delta_h = delta_z @ components.transpose(0, 1)
    return base_vectors + delta_h.to(dtype=base_vectors.dtype)


def collect_layer_token_residual_states(
    *,
    model,
    bank: MCQAPairBank,
    layer: int,
    token_position_id: str,
    batch_size: int,
    device: torch.device | str,
    include_base: bool = True,
    include_source: bool = True,
) -> torch.Tensor:
    """Collect fit residual states for one layer/token position from the bank."""
    if not include_base and not include_source:
        raise ValueError("At least one of include_base/include_source must be True")
    device = torch.device(device)
    layers = _resolve_transformer_layers(model)
    target_layer = layers[int(layer)]
    collected = []

    def collect_from_inputs(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()

        handle = target_layer.register_forward_hook(hook)
        try:
            position_ids = _build_position_ids_from_left_padded_attention_mask(attention_mask)
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture layer hidden states for PCA fitting")
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        padded_positions = _resolve_padded_positions(attention_mask.to(hidden.device), positions.to(hidden.device))
        return hidden[batch_indices, padded_positions].detach().cpu()

    for start in range(0, bank.size, int(batch_size)):
        end = min(start + int(batch_size), bank.size)
        if include_base:
            collected.append(
                collect_from_inputs(
                    input_ids=bank.base_input_ids[start:end].to(device),
                    attention_mask=bank.base_attention_mask[start:end].to(device),
                    positions=bank.base_position_by_id[str(token_position_id)][start:end],
                )
            )
        if include_source:
            collected.append(
                collect_from_inputs(
                    input_ids=bank.source_input_ids[start:end].to(device),
                    attention_mask=bank.source_attention_mask[start:end].to(device),
                    positions=bank.source_position_by_id[str(token_position_id)][start:end],
                )
            )
    if not collected:
        raise ValueError("No residual states were collected for PCA fitting")
    return torch.cat(collected, dim=0)


def collect_layer_token_residual_states_from_prompt_records(
    *,
    model,
    tokenizer,
    prompt_records: list[dict[str, object]],
    layer: int,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Collect residual states from explicit prompt/position records."""
    if not prompt_records:
        raise ValueError("prompt_records must be non-empty")
    device = torch.device(device)
    layers = _resolve_transformer_layers(model)
    target_layer = layers[int(layer)]
    collected = []

    def collect_from_batch(prompts: list[str], positions: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()

        encoded = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        handle = target_layer.register_forward_hook(hook)
        try:
            position_ids = _build_position_ids_from_left_padded_attention_mask(attention_mask)
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture layer hidden states for PCA fitting")
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        padded_positions = _resolve_padded_positions(attention_mask.to(hidden.device), positions.to(hidden.device))
        return hidden[batch_indices, padded_positions].detach().cpu()

    for start in range(0, len(prompt_records), int(batch_size)):
        end = min(start + int(batch_size), len(prompt_records))
        batch_records = prompt_records[start:end]
        prompts = [str(record["raw_input"]) for record in batch_records]
        positions = torch.tensor([int(record["position"]) for record in batch_records], dtype=torch.long)
        collected.append(collect_from_batch(prompts, positions))
    return torch.cat(collected, dim=0)


def fit_layer_token_pca_basis(
    *,
    model,
    bank: MCQAPairBank,
    layer: int,
    token_position_id: str,
    batch_size: int,
    device: torch.device | str,
    basis_id: str | None = None,
    include_base: bool = True,
    include_source: bool = True,
    rank_tol: float | None = None,
) -> LayerPCABasis:
    states = collect_layer_token_residual_states(
        model=model,
        bank=bank,
        layer=layer,
        token_position_id=token_position_id,
        batch_size=batch_size,
        device=device,
        include_base=include_base,
        include_source=include_source,
    )
    return fit_pca_basis_from_states(
        states=states,
        layer=layer,
        token_position_id=token_position_id,
        basis_id=basis_id,
        rank_tol=rank_tol,
    )


def fit_layer_token_pca_basis_from_prompt_records(
    *,
    model,
    tokenizer,
    prompt_records: list[dict[str, object]],
    layer: int,
    token_position_id: str,
    batch_size: int,
    device: torch.device | str,
    basis_id: str | None = None,
    rank_tol: float | None = None,
) -> LayerPCABasis:
    states = collect_layer_token_residual_states_from_prompt_records(
        model=model,
        tokenizer=tokenizer,
        prompt_records=prompt_records,
        layer=layer,
        batch_size=batch_size,
        device=device,
    )
    return fit_pca_basis_from_states(
        states=states,
        layer=layer,
        token_position_id=token_position_id,
        basis_id=basis_id,
        rank_tol=rank_tol,
    )


def save_pca_basis(path: str | Path, basis: LayerPCABasis) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis.to_payload(), path)


def load_pca_basis(path: str | Path) -> LayerPCABasis:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid PCA basis payload at {path}")
    basis = LayerPCABasis.from_payload(payload)
    if basis.components.shape != (int(basis.hidden_size), int(basis.rank)):
        raise ValueError(
            f"Invalid PCA basis shape {tuple(basis.components.shape)} for hidden_size={int(basis.hidden_size)} rank={int(basis.rank)}"
        )
    return basis


def load_or_fit_pca_basis(
    *,
    path: str | Path,
    model,
    bank: MCQAPairBank,
    layer: int,
    token_position_id: str,
    batch_size: int,
    device: torch.device | str,
    basis_id: str | None = None,
    include_base: bool = True,
    include_source: bool = True,
    rank_tol: float | None = None,
) -> LayerPCABasis:
    path = Path(path)
    if path.exists():
        basis = load_pca_basis(path)
        if (
            int(basis.layer) == int(layer)
            and str(basis.token_position_id) == str(token_position_id)
            and int(basis.hidden_size) == int(model.config.hidden_size)
        ):
            return basis
    basis = fit_layer_token_pca_basis(
        model=model,
        bank=bank,
        layer=layer,
        token_position_id=token_position_id,
        batch_size=batch_size,
        device=device,
        basis_id=basis_id,
        include_base=include_base,
        include_source=include_source,
        rank_tol=rank_tol,
    )
    save_pca_basis(path, basis)
    return basis


def load_or_fit_pca_basis_from_prompt_records(
    *,
    path: str | Path,
    model,
    tokenizer,
    prompt_records: list[dict[str, object]],
    layer: int,
    token_position_id: str,
    batch_size: int,
    device: torch.device | str,
    basis_id: str | None = None,
    rank_tol: float | None = None,
) -> LayerPCABasis:
    path = Path(path)
    if path.exists():
        basis = load_pca_basis(path)
        if (
            int(basis.layer) == int(layer)
            and str(basis.token_position_id) == str(token_position_id)
            and int(basis.hidden_size) == int(model.config.hidden_size)
        ):
            return basis
    basis = fit_layer_token_pca_basis_from_prompt_records(
        model=model,
        tokenizer=tokenizer,
        prompt_records=prompt_records,
        layer=layer,
        token_position_id=token_position_id,
        batch_size=batch_size,
        device=device,
        basis_id=basis_id,
        rank_tol=rank_tol,
    )
    save_pca_basis(path, basis)
    return basis
