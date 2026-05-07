from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


def enumerate_output_logit_sites(*, output_dim: int) -> tuple[OutputLogitSite, ...]:
    return tuple(OutputLogitSite(output_index=i) for i in range(int(output_dim)))
