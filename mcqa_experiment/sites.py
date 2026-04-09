"""Residual-stream site enumeration for MCQA transformer interventions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualSite:
    """One residual-stream site identified by layer and token-position id."""

    layer: int
    token_position_id: str

    @property
    def label(self) -> str:
        return f"L{int(self.layer)}:{self.token_position_id}"


def enumerate_residual_sites(
    *,
    num_layers: int,
    token_position_ids: tuple[str, ...],
    layers: tuple[int, ...] | None = None,
    selected_token_position_ids: tuple[str, ...] | None = None,
) -> list[ResidualSite]:
    """Enumerate all candidate residual-stream sites for the MCQA sweep."""
    layer_ids = tuple(range(int(num_layers))) if layers is None else tuple(int(layer) for layer in layers)
    position_ids = token_position_ids if selected_token_position_ids is None else selected_token_position_ids
    return [ResidualSite(layer=layer, token_position_id=position_id) for layer in layer_ids for position_id in position_ids]
