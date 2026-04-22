from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class Site(Protocol):
    def key(self) -> str:
        ...


@dataclass(frozen=True)
class FullStateSite:
    timestep: int

    def key(self) -> str:
        return f"h_{self.timestep}"


@dataclass(frozen=True)
class NeuronSite:
    timestep: int
    neuron_index: int

    def key(self) -> str:
        return f"h_{self.timestep}[{self.neuron_index}]"


@dataclass(frozen=True)
class CoordinateGroupSite:
    timestep: int
    coord_indices: tuple[int, ...]

    def key(self) -> str:
        joined = ",".join(str(i) for i in self.coord_indices)
        return f"h_{self.timestep}[{joined}]"


@dataclass(frozen=True)
class OutputLogitSite:
    output_index: int

    def key(self) -> str:
        return f"logit[{self.output_index}]"


def enumerate_full_state_sites(width: int = 4) -> tuple[FullStateSite, ...]:
    return tuple(FullStateSite(timestep=t) for t in range(width))


def enumerate_neuron_sites_for_timestep(*, timestep: int, hidden_size: int) -> tuple[NeuronSite, ...]:
    return tuple(NeuronSite(timestep=int(timestep), neuron_index=i) for i in range(int(hidden_size)))


def enumerate_all_neuron_sites(*, width: int, hidden_size: int) -> tuple[NeuronSite, ...]:
    return tuple(
        NeuronSite(timestep=timestep, neuron_index=i)
        for timestep in range(int(width))
        for i in range(int(hidden_size))
    )


def enumerate_group_sites_for_timesteps(
    *,
    timesteps: tuple[int, ...],
    hidden_size: int,
    resolution: int,
) -> tuple[CoordinateGroupSite, ...]:
    resolution = int(resolution)
    hidden_size = int(hidden_size)
    if resolution <= 0 or resolution > hidden_size:
        raise ValueError(f"resolution must be in [1, {hidden_size}], got {resolution}")
    if hidden_size % resolution != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by resolution={resolution}")
    sites = []
    for timestep in timesteps:
        for start in range(0, hidden_size, resolution):
            coords = tuple(range(start, start + resolution))
            sites.append(CoordinateGroupSite(timestep=int(timestep), coord_indices=coords))
    return tuple(sites)


class RotatedSite:
    """Site defined by a group of PCA (or any orthogonal rotation) components at a given timestep.

    The intervention projects both base and source hidden states into the rotation's
    column space, replaces the selected component coordinates, and projects back.
    """

    __slots__ = ("timestep", "coord_indices", "rotation", "_key_str")

    def __init__(self, timestep: int, coord_indices: tuple[int, ...], rotation: torch.Tensor) -> None:
        object.__setattr__(self, "timestep", int(timestep))
        object.__setattr__(self, "coord_indices", tuple(int(i) for i in coord_indices))
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "_key_str", f"pca_t{int(timestep)}_c{'_'.join(str(i) for i in coord_indices)}")

    def key(self) -> str:
        return self._key_str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RotatedSite) and self._key_str == other._key_str

    def __hash__(self) -> int:
        return hash(self._key_str)

    def __repr__(self) -> str:
        return f"RotatedSite(timestep={self.timestep}, coord_indices={self.coord_indices})"


def enumerate_pca_group_sites_for_timesteps(
    *,
    timesteps: tuple[int, ...],
    hidden_size: int,
    resolution: int,
    rotations: dict[int, torch.Tensor],
) -> tuple[RotatedSite, ...]:
    """Create RotatedSites grouping PCA components at each timestep analogously to coordinate groups."""
    sites = []
    for t in timesteps:
        rotation = rotations[int(t)]
        for start in range(0, int(hidden_size), int(resolution)):
            coords = tuple(range(start, start + int(resolution)))
            sites.append(RotatedSite(timestep=int(t), coord_indices=coords, rotation=rotation))
    return tuple(sites)


def enumerate_output_logit_sites(*, output_dim: int) -> tuple[OutputLogitSite, ...]:
    return tuple(OutputLogitSite(output_index=i) for i in range(int(output_dim)))
