"""Residual-stream intervention helpers for Gemma-2-2B MCQA runs."""

from __future__ import annotations

import torch
import torch.nn as nn

from .pca import LayerPCABasis, apply_rotated_component_update
from .sites import (
    ResidualSite,
    RotatedBandSite,
    RotatedCompositeSite,
    SiteLike,
    site_segments,
)


def resolve_transformer_layers(model) -> list[nn.Module]:
    """Resolve the ordered transformer blocks for common causal LM wrappers."""
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


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    return len(resolve_transformer_layers(model))


def get_hidden_size(model) -> int:
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    raise ValueError(f"Could not resolve hidden_size from model config {type(model.config)!r}")


def _resolve_last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the final non-pad token for each batch element."""
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape {tuple(attention_mask.shape)}")
    reversed_mask = torch.flip(attention_mask.to(torch.long), dims=(1,))
    trailing_pad = torch.argmax(reversed_mask, dim=1)
    return attention_mask.shape[1] - 1 - trailing_pad


def _resolve_padded_positions(attention_mask: torch.Tensor, unpadded_positions: torch.Tensor) -> torch.Tensor:
    """Map positions from unpadded prompt coordinates into padded batch coordinates."""
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


def gather_last_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather next-token logits at the final non-pad token for each example."""
    last_indices = _resolve_last_nonpad_indices(attention_mask.to(logits.device))
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices]


def build_position_ids_from_left_padded_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build content-aligned position ids for left-padded causal LM batches."""
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask to have shape [batch, seq], got {tuple(attention_mask.shape)}")
    position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def forward_factual_logits(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the factual model and return last-token logits."""
    position_ids = build_position_ids_from_left_padded_attention_mask(attention_mask)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    return gather_last_token_logits(outputs.logits, attention_mask)


class DASSubspaceIntervention(nn.Module):
    """Low-rank rotated-space swap on one residual vector."""

    def __init__(self, hidden_size: int, subspace_dim: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.subspace_dim = int(subspace_dim)
        linear = nn.Linear(self.hidden_size, self.subspace_dim, bias=False)
        self.project = torch.nn.utils.parametrizations.orthogonal(linear)

    def forward(self, base_vectors: torch.Tensor, source_vectors: torch.Tensor) -> torch.Tensor:
        basis = self.project.weight
        compute_dtype = basis.dtype
        base_vectors_compute = base_vectors.to(compute_dtype)
        source_vectors_compute = source_vectors.to(compute_dtype)
        base_features = base_vectors_compute @ basis.t()
        source_features = source_vectors_compute @ basis.t()
        updated = base_vectors_compute + (source_features - base_features) @ basis
        return updated.to(base_vectors.dtype)


def _collect_source_hidden_states(
    *,
    model,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    target_layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    position_ids = build_position_ids_from_left_padded_attention_mask(source_attention_mask)
    with torch.no_grad():
        outputs = model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    return {int(layer): hidden_states[int(layer) + 1].detach() for layer in target_layers}


def run_soft_residual_intervention(
    *,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    site_weights: dict[SiteLike, float],
    strength: float,
    base_position_by_id: dict[str, torch.Tensor],
    source_position_by_id: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply a weighted residual interpolation across the selected sites."""
    if not site_weights:
        return forward_factual_logits(model=model, input_ids=base_input_ids, attention_mask=base_attention_mask)

    layers = resolve_transformer_layers(model)
    weighted_segments = [
        (segment, float(weight))
        for site, weight in site_weights.items()
        for segment in site_segments(site, model_hidden_size=get_hidden_size(model))
    ]
    target_layers = tuple(sorted({int(segment.layer) for segment, _ in weighted_segments}))
    source_hidden_by_layer = _collect_source_hidden_states(
        model=model,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        target_layers=target_layers,
    )
    handles = []

    def make_hook(layer_index: int):
        layer_sites = [
            (segment, float(weight))
            for segment, weight in weighted_segments
            if int(segment.layer) == int(layer_index)
        ]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            batch_size = hidden.shape[0]
            batch_indices = torch.arange(batch_size, device=hidden.device)
            source_hidden = source_hidden_by_layer[int(layer_index)].to(hidden.device)
            padded_base_attention_mask = base_attention_mask.to(hidden.device)
            padded_source_attention_mask = source_attention_mask.to(hidden.device)
            for site, weight in layer_sites:
                base_positions = _resolve_padded_positions(
                    padded_base_attention_mask,
                    base_position_by_id[site.token_position_id].to(hidden.device),
                )
                source_positions = _resolve_padded_positions(
                    padded_source_attention_mask,
                    source_position_by_id[site.token_position_id].to(hidden.device),
                )
                base_vectors = hidden[batch_indices, base_positions]
                source_vectors = source_hidden[batch_indices, source_positions]
                delta = float(strength) * float(weight) * (source_vectors - base_vectors)
                hidden[batch_indices, base_positions, int(site.dim_start) : int(site.dim_end)] = (
                    base_vectors[:, int(site.dim_start) : int(site.dim_end)]
                    + delta[:, int(site.dim_start) : int(site.dim_end)]
                )
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

    for layer_index in target_layers:
        handles.append(layers[int(layer_index)].register_forward_hook(make_hook(int(layer_index))))
    try:
        base_position_ids = build_position_ids_from_left_padded_attention_mask(base_attention_mask)
        outputs = model(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_ids=base_position_ids,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    return gather_last_token_logits(outputs.logits, base_attention_mask)


def run_das_residual_intervention(
    *,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    site: SiteLike,
    intervention: DASSubspaceIntervention,
    base_position_by_id: dict[str, torch.Tensor],
    source_position_by_id: dict[str, torch.Tensor],
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> torch.Tensor:
    """Apply one trainable DAS-style rotated-space swap at a residual-stream site."""
    if isinstance(site, (RotatedBandSite, RotatedCompositeSite)):
        if pca_bases_by_id is None:
            raise ValueError("pca_bases_by_id is required for rotated DAS interventions")
        return run_das_rotated_residual_intervention(
            model=model,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            site=site,
            intervention=intervention,
            base_position_by_id=base_position_by_id,
            source_position_by_id=source_position_by_id,
            pca_bases_by_id=pca_bases_by_id,
        )
    layers = resolve_transformer_layers(model)
    source_hidden_by_layer = _collect_source_hidden_states(
        model=model,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        target_layers=(site.layer,),
    )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        batch_size = hidden.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden.device)
        source_hidden = source_hidden_by_layer[int(site.layer)].to(hidden.device)
        base_attention_mask_device = base_attention_mask.to(hidden.device)
        source_attention_mask_device = source_attention_mask.to(hidden.device)
        segments = site_segments(site, model_hidden_size=int(hidden.shape[-1]))
        token_position_ids = tuple(
            dict.fromkeys(str(segment.token_position_id) for segment in segments)
        )
        base_positions_by_token = {
            token_position_id: _resolve_padded_positions(
                base_attention_mask_device,
                base_position_by_id[token_position_id].to(hidden.device),
            )
            for token_position_id in token_position_ids
        }
        source_positions_by_token = {
            token_position_id: _resolve_padded_positions(
                source_attention_mask_device,
                source_position_by_id[token_position_id].to(hidden.device),
            )
            for token_position_id in token_position_ids
        }
        base_vectors = torch.cat(
            [
                hidden[
                    batch_indices,
                    base_positions_by_token[str(segment.token_position_id)],
                    int(segment.dim_start) : int(segment.dim_end),
                ]
                for segment in segments
            ],
            dim=1,
        )
        source_vectors = torch.cat(
            [
                source_hidden[
                    batch_indices,
                    source_positions_by_token[str(segment.token_position_id)],
                    int(segment.dim_start) : int(segment.dim_end),
                ]
                for segment in segments
            ],
            dim=1,
        )
        updated_vectors = intervention(base_vectors, source_vectors)
        offset = 0
        for segment in segments:
            segment_width = int(segment.dim_end) - int(segment.dim_start)
            next_offset = offset + segment_width
            hidden[
                batch_indices,
                base_positions_by_token[str(segment.token_position_id)],
                int(segment.dim_start) : int(segment.dim_end),
            ] = updated_vectors[:, offset:next_offset]
            offset = next_offset
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = layers[int(site.layer)].register_forward_hook(hook)
    try:
        base_position_ids = build_position_ids_from_left_padded_attention_mask(base_attention_mask)
        outputs = model(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_ids=base_position_ids,
            use_cache=False,
        )
    finally:
        handle.remove()
    return gather_last_token_logits(outputs.logits, base_attention_mask)


def _rotated_site_segments(site: RotatedBandSite | RotatedCompositeSite) -> tuple[RotatedBandSite, ...]:
    if isinstance(site, RotatedCompositeSite):
        return tuple(site.segments)
    return (site,)


def _merged_rotated_site_segments(site: RotatedBandSite | RotatedCompositeSite) -> tuple[RotatedBandSite, ...]:
    segments = _rotated_site_segments(site)
    if not segments:
        return ()
    first_segment = segments[0]
    ordered = sorted(
        {
            (int(segment.component_start), int(segment.component_end))
            for segment in segments
        }
    )
    merged: list[tuple[int, int]] = []
    for component_start, component_end in ordered:
        if not merged or int(component_start) > int(merged[-1][1]):
            merged.append((int(component_start), int(component_end)))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (int(previous_start), max(int(previous_end), int(component_end)))
    return tuple(
        RotatedBandSite(
            layer=int(first_segment.layer),
            token_position_id=str(first_segment.token_position_id),
            basis_id=str(first_segment.basis_id),
            component_start=int(component_start),
            component_end=int(component_end),
        )
        for component_start, component_end in merged
    )


def apply_rotated_das_site_update(
    *,
    base_vectors: torch.Tensor,
    source_vectors: torch.Tensor,
    basis: LayerPCABasis,
    site: RotatedBandSite | RotatedCompositeSite,
    intervention: nn.Module,
) -> torch.Tensor:
    """Apply a DAS intervention inside a selected PCA span while preserving the orthogonal complement."""
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
    segments = _merged_rotated_site_segments(site)
    selected_base = torch.cat(
        [
            z_base[:, int(segment.component_start) : int(segment.component_end)]
            for segment in segments
        ],
        dim=1,
    )
    selected_source = torch.cat(
        [
            z_source[:, int(segment.component_start) : int(segment.component_end)]
            for segment in segments
        ],
        dim=1,
    )
    updated_selected = intervention(selected_base, selected_source).to(compute_dtype)
    z_updated = z_base.clone()
    offset = 0
    for segment in segments:
        segment_width = int(segment.component_end) - int(segment.component_start)
        next_offset = offset + segment_width
        z_updated[:, int(segment.component_start) : int(segment.component_end)] = updated_selected[:, offset:next_offset]
        offset = next_offset
    delta_h = (z_updated - z_base) @ components.transpose(0, 1)
    return base_vectors + delta_h.to(dtype=base_vectors.dtype)


def run_soft_rotated_residual_intervention(
    *,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    site_weights: dict[RotatedBandSite | RotatedCompositeSite, float],
    strength: float,
    base_position_by_id: dict[str, torch.Tensor],
    source_position_by_id: dict[str, torch.Tensor],
    pca_bases_by_id: dict[str, LayerPCABasis],
) -> torch.Tensor:
    """Apply a weighted PCA-band interpolation while preserving the orthogonal complement."""
    if not site_weights:
        return forward_factual_logits(model=model, input_ids=base_input_ids, attention_mask=base_attention_mask)

    layers = resolve_transformer_layers(model)
    target_layers = tuple(sorted({int(site.layer) for site in site_weights}))
    source_hidden_by_layer = _collect_source_hidden_states(
        model=model,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        target_layers=target_layers,
    )
    handles = []

    def make_hook(layer_index: int):
        layer_sites = [
            (site, float(weight))
            for site, weight in site_weights.items()
            if int(site.layer) == int(layer_index)
        ]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            batch_size = hidden.shape[0]
            batch_indices = torch.arange(batch_size, device=hidden.device)
            source_hidden = source_hidden_by_layer[int(layer_index)].to(hidden.device)
            padded_base_attention_mask = base_attention_mask.to(hidden.device)
            padded_source_attention_mask = source_attention_mask.to(hidden.device)

            grouped_sites: dict[tuple[str, str], list[tuple[RotatedBandSite | RotatedCompositeSite, float]]] = {}
            for site, weight in layer_sites:
                grouped_sites.setdefault((str(site.token_position_id), str(site.basis_id)), []).append((site, weight))

            for (token_position_id, basis_id), grouped in grouped_sites.items():
                basis = pca_bases_by_id.get(str(basis_id))
                if basis is None:
                    raise KeyError(f"Missing PCA basis for basis_id={basis_id!r}")
                if int(basis.layer) != int(layer_index):
                    raise ValueError(
                        f"PCA basis layer mismatch for basis_id={basis_id!r}: basis layer={int(basis.layer)} "
                        f"but site layer={int(layer_index)}"
                    )
                if str(basis.token_position_id) != str(token_position_id):
                    raise ValueError(
                        f"PCA basis token-position mismatch for basis_id={basis_id!r}: "
                        f"basis token_position_id={basis.token_position_id!r} vs site token_position_id={token_position_id!r}"
                    )

                base_positions = _resolve_padded_positions(
                    padded_base_attention_mask,
                    base_position_by_id[str(token_position_id)].to(hidden.device),
                )
                source_positions = _resolve_padded_positions(
                    padded_source_attention_mask,
                    source_position_by_id[str(token_position_id)].to(hidden.device),
                )
                base_vectors = hidden[batch_indices, base_positions]
                source_vectors = source_hidden[batch_indices, source_positions]
                component_segments: list[tuple[int, int, float]] = []
                for site, weight in grouped:
                    for segment in _rotated_site_segments(site):
                        component_segments.append(
                            (
                                int(segment.component_start),
                                int(segment.component_end),
                                float(weight),
                            )
                        )
                updated_vectors = apply_rotated_component_update(
                    base_vectors=base_vectors,
                    source_vectors=source_vectors,
                    basis=basis,
                    component_segments=component_segments,
                    strength=float(strength),
                )
                hidden[batch_indices, base_positions] = updated_vectors

            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

    for layer_index in target_layers:
        handles.append(layers[int(layer_index)].register_forward_hook(make_hook(int(layer_index))))
    try:
        base_position_ids = build_position_ids_from_left_padded_attention_mask(base_attention_mask)
        outputs = model(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_ids=base_position_ids,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    return gather_last_token_logits(outputs.logits, base_attention_mask)


def run_das_rotated_residual_intervention(
    *,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    site: RotatedBandSite | RotatedCompositeSite,
    intervention: DASSubspaceIntervention,
    base_position_by_id: dict[str, torch.Tensor],
    source_position_by_id: dict[str, torch.Tensor],
    pca_bases_by_id: dict[str, LayerPCABasis],
) -> torch.Tensor:
    """Apply a trainable DAS swap inside a rotated PCA span."""
    basis = pca_bases_by_id.get(str(site.basis_id))
    if basis is None:
        raise KeyError(f"Missing PCA basis for basis_id={site.basis_id!r}")
    if int(basis.layer) != int(site.layer):
        raise ValueError(
            f"PCA basis layer mismatch for basis_id={site.basis_id!r}: basis layer={int(basis.layer)} "
            f"but site layer={int(site.layer)}"
        )
    if str(basis.token_position_id) != str(site.token_position_id):
        raise ValueError(
            f"PCA basis token-position mismatch for basis_id={site.basis_id!r}: "
            f"basis token_position_id={basis.token_position_id!r} vs site token_position_id={site.token_position_id!r}"
        )

    layers = resolve_transformer_layers(model)
    source_hidden_by_layer = _collect_source_hidden_states(
        model=model,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        target_layers=(site.layer,),
    )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        batch_size = hidden.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden.device)
        source_hidden = source_hidden_by_layer[int(site.layer)].to(hidden.device)
        base_attention_mask_device = base_attention_mask.to(hidden.device)
        source_attention_mask_device = source_attention_mask.to(hidden.device)
        base_positions = _resolve_padded_positions(
            base_attention_mask_device,
            base_position_by_id[str(site.token_position_id)].to(hidden.device),
        )
        source_positions = _resolve_padded_positions(
            source_attention_mask_device,
            source_position_by_id[str(site.token_position_id)].to(hidden.device),
        )
        base_vectors = hidden[batch_indices, base_positions]
        source_vectors = source_hidden[batch_indices, source_positions]
        updated_vectors = apply_rotated_das_site_update(
            base_vectors=base_vectors,
            source_vectors=source_vectors,
            basis=basis,
            site=site,
            intervention=intervention,
        )
        hidden[batch_indices, base_positions] = updated_vectors
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = layers[int(site.layer)].register_forward_hook(hook)
    try:
        base_position_ids = build_position_ids_from_left_padded_attention_mask(base_attention_mask)
        outputs = model(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_ids=base_position_ids,
            use_cache=False,
        )
    finally:
        handle.remove()
    return gather_last_token_logits(outputs.logits, base_attention_mask)


def run_soft_site_intervention(
    *,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    site_weights: dict[SiteLike, float],
    strength: float,
    base_position_by_id: dict[str, torch.Tensor],
    source_position_by_id: dict[str, torch.Tensor],
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> torch.Tensor:
    """Dispatch soft interventions across residual or rotated MCQA site types."""
    if not site_weights:
        return forward_factual_logits(model=model, input_ids=base_input_ids, attention_mask=base_attention_mask)
    sites = tuple(site_weights.keys())
    if all(isinstance(site, (RotatedBandSite, RotatedCompositeSite)) for site in sites):
        if pca_bases_by_id is None:
            raise ValueError("pca_bases_by_id is required for rotated-site interventions")
        return run_soft_rotated_residual_intervention(
            model=model,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            site_weights=site_weights,
            strength=strength,
            base_position_by_id=base_position_by_id,
            source_position_by_id=source_position_by_id,
            pca_bases_by_id=pca_bases_by_id,
        )
    if all(not isinstance(site, (RotatedBandSite, RotatedCompositeSite)) for site in sites):
        return run_soft_residual_intervention(
            model=model,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            site_weights=site_weights,
            strength=strength,
            base_position_by_id=base_position_by_id,
            source_position_by_id=source_position_by_id,
        )
    raise ValueError("Mixed residual and rotated site types are not supported in one intervention call")
