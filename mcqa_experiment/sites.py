"""Residual-stream site enumeration for MCQA transformer interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ResidualSite:
    """One residual-stream block identified by layer, token-position, and dim range."""

    layer: int
    token_position_id: str
    dim_start: int
    dim_end: int

    @property
    def label(self) -> str:
        return f"L{int(self.layer)}:{self.token_position_id}[{int(self.dim_start)}:{int(self.dim_end)}]"


@dataclass(frozen=True)
class ResidualCompositeSite:
    """Concatenated residual site spanning an arbitrary ordered tuple of segments."""

    layer: int
    segments: tuple[ResidualSite, ...]

    @property
    def dim_start(self) -> int:
        return 0

    @property
    def dim_end(self) -> int:
        return sum(int(segment.dim_end) - int(segment.dim_start) for segment in self.segments)

    @property
    def label(self) -> str:
        segment_labels = "+".join(
            f"{segment.token_position_id}[{int(segment.dim_start)}:{int(segment.dim_end)}]"
            for segment in self.segments
        )
        return f"L{int(self.layer)}:{segment_labels}"


@dataclass(frozen=True)
class ResidualUnionSite:
    """One layer-level site spanning multiple token positions with full residual vectors."""

    layer: int
    token_position_ids: tuple[str, ...]
    hidden_size: int

    @property
    def dim_start(self) -> int:
        return 0

    @property
    def dim_end(self) -> int:
        return int(self.hidden_size) * len(self.token_position_ids)

    @property
    def label(self) -> str:
        joined = "+".join(str(token_position_id) for token_position_id in self.token_position_ids)
        return f"L{int(self.layer)}:{joined}[{int(self.dim_start)}:{int(self.dim_end)}]"


@dataclass(frozen=True)
class RotatedBandSite:
    """One rotated-basis site identified by a contiguous PCA component band."""

    layer: int
    token_position_id: str
    basis_id: str
    component_start: int
    component_end: int

    @property
    def dim_start(self) -> int:
        return int(self.component_start)

    @property
    def dim_end(self) -> int:
        return int(self.component_end)

    @property
    def label(self) -> str:
        return f"L{int(self.layer)}:{self.token_position_id}:pc[{int(self.component_start)}:{int(self.component_end)}]"


@dataclass(frozen=True)
class RotatedCompositeSite:
    """Concatenated rotated-basis site spanning an ordered tuple of PCA bands."""

    layer: int
    token_position_id: str
    basis_id: str
    segments: tuple[RotatedBandSite, ...]

    @property
    def dim_start(self) -> int:
        return 0

    @property
    def dim_end(self) -> int:
        return sum(int(segment.component_end) - int(segment.component_start) for segment in self.segments)

    @property
    def label(self) -> str:
        segment_labels = "+".join(
            f"pc[{int(segment.component_start)}:{int(segment.component_end)}]"
            for segment in self.segments
        )
        return f"L{int(self.layer)}:{self.token_position_id}:{segment_labels}"


SiteLike = Union[
    ResidualSite,
    ResidualUnionSite,
    ResidualCompositeSite,
    RotatedBandSite,
    RotatedCompositeSite,
]


def site_segments(site: SiteLike, *, model_hidden_size: int) -> tuple[ResidualSite, ...]:
    if isinstance(site, ResidualCompositeSite):
        return tuple(site.segments)
    if isinstance(site, ResidualUnionSite):
        return tuple(
            ResidualSite(
                layer=int(site.layer),
                token_position_id=str(token_position_id),
                dim_start=0,
                dim_end=int(site.hidden_size),
            )
            for token_position_id in site.token_position_ids
        )
    return (site,)


def site_token_position_ids(site: SiteLike) -> tuple[str, ...]:
    if isinstance(site, ResidualCompositeSite):
        return tuple(dict.fromkeys(str(segment.token_position_id) for segment in site.segments))
    if isinstance(site, ResidualUnionSite):
        return tuple(str(token_position_id) for token_position_id in site.token_position_ids)
    if isinstance(site, RotatedCompositeSite):
        return (str(site.token_position_id),)
    return (str(site.token_position_id),)


def site_total_width(site: SiteLike, *, model_hidden_size: int) -> int:
    if isinstance(site, ResidualCompositeSite):
        return sum(int(segment.dim_end) - int(segment.dim_start) for segment in site.segments)
    if isinstance(site, ResidualUnionSite):
        return int(site.hidden_size) * len(site.token_position_ids)
    if isinstance(site, RotatedCompositeSite):
        return sum(int(segment.component_end) - int(segment.component_start) for segment in site.segments)
    return int(site.dim_end) - int(site.dim_start)


def enumerate_residual_sites(
    *,
    num_layers: int,
    hidden_size: int,
    token_position_ids: tuple[str, ...],
    resolution: int | None = 1,
    layers: tuple[int, ...] | None = None,
    selected_token_position_ids: tuple[str, ...] | None = None,
) -> list[ResidualSite]:
    """Enumerate residual-stream dimension blocks for the MCQA sweep."""
    layer_ids = tuple(range(int(num_layers))) if layers is None else tuple(int(layer) for layer in layers)
    position_ids = token_position_ids if selected_token_position_ids is None else selected_token_position_ids
    block_width = int(hidden_size) if resolution is None else max(1, int(resolution))
    dim_blocks = [
        (dim_start, min(dim_start + block_width, int(hidden_size)))
        for dim_start in range(0, int(hidden_size), block_width)
    ]
    return [
        ResidualSite(
            layer=layer,
            token_position_id=position_id,
            dim_start=dim_start,
            dim_end=dim_end,
        )
        for layer in layer_ids
        for position_id in position_ids
        for dim_start, dim_end in dim_blocks
    ]


def enumerate_rotated_band_sites(
    *,
    rank: int,
    num_bands: int,
    layer: int,
    token_position_id: str,
    basis_id: str,
    schedule: str = "equal",
) -> list[RotatedBandSite]:
    """Split the retained PCA spectrum into contiguous coarse bands."""
    if int(rank) <= 0:
        raise ValueError(f"rank must be > 0, got {int(rank)}")
    if int(num_bands) <= 0:
        raise ValueError(f"num_bands must be > 0, got {int(num_bands)}")
    resolved_num_bands = min(int(num_bands), int(rank))
    if str(schedule) == "equal":
        weights = [1.0] * resolved_num_bands
    elif str(schedule) == "head":
        weights = [float(index) for index in range(1, resolved_num_bands + 1)]
    else:
        raise ValueError(f"Unsupported rotated band schedule {schedule!r}")
    total_weight = float(sum(weights))
    raw_widths = [float(rank) * float(weight) / total_weight for weight in weights]
    band_widths = [max(1, int(width)) for width in raw_widths]
    allocated = sum(band_widths)
    if allocated > int(rank):
        overflow = allocated - int(rank)
        for band_index in reversed(range(len(band_widths))):
            if overflow <= 0:
                break
            reducible = max(0, band_widths[band_index] - 1)
            if reducible <= 0:
                continue
            delta = min(reducible, overflow)
            band_widths[band_index] -= delta
            overflow -= delta
    elif allocated < int(rank):
        deficit = int(rank) - allocated
        for band_index in reversed(range(len(band_widths))):
            if deficit <= 0:
                break
            band_widths[band_index] += 1
            deficit -= 1

    band_sites: list[RotatedBandSite] = []
    start = 0
    for band_width in band_widths:
        end = min(int(rank), start + int(band_width))
        band_sites.append(
            RotatedBandSite(
                layer=int(layer),
                token_position_id=str(token_position_id),
                basis_id=str(basis_id),
                component_start=int(start),
                component_end=int(end),
            )
        )
        start = end
    if start != int(rank):
        raise ValueError(f"Rotated band enumeration failed to cover the full rank={int(rank)}")
    return band_sites

