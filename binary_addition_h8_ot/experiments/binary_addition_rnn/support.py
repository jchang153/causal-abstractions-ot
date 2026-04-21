from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .sites import CoordinateGroupSite, FullStateSite, OutputLogitSite, Site


_HIDDEN_FULL_RE = re.compile(r"^h_(\d+)$")
_HIDDEN_GROUP_RE = re.compile(r"^h_(\d+)\[([0-9,]+)\]$")
_LOGIT_RE = re.compile(r"^logit\[(\d+)\]$")


@dataclass(frozen=True)
class ParsedSiteKey:
    kind: str
    timestep: int | None = None
    coord_indices: tuple[int, ...] | None = None
    output_index: int | None = None


def parse_site_key(site_key: str, *, hidden_size: int | None = None) -> ParsedSiteKey:
    full_match = _HIDDEN_FULL_RE.match(str(site_key))
    if full_match:
        timestep = int(full_match.group(1))
        coords = tuple(range(int(hidden_size))) if hidden_size is not None else None
        return ParsedSiteKey(kind="full_state", timestep=timestep, coord_indices=coords)

    group_match = _HIDDEN_GROUP_RE.match(str(site_key))
    if group_match:
        timestep = int(group_match.group(1))
        coords = tuple(int(part) for part in group_match.group(2).split(",") if part)
        return ParsedSiteKey(kind="coord_group", timestep=timestep, coord_indices=coords)

    logit_match = _LOGIT_RE.match(str(site_key))
    if logit_match:
        return ParsedSiteKey(kind="output_logit", output_index=int(logit_match.group(1)))

    raise ValueError(f"unsupported site key: {site_key!r}")


def site_spec_to_site(site_spec: dict[str, object]) -> Site:
    kind = str(site_spec["kind"])
    if kind == "full_state":
        return FullStateSite(timestep=int(site_spec["timestep"]))
    if kind == "coord_group":
        return CoordinateGroupSite(
            timestep=int(site_spec["timestep"]),
            coord_indices=tuple(int(i) for i in site_spec["coord_indices"]),
        )
    if kind == "output_logit":
        return OutputLogitSite(output_index=int(site_spec["output_index"]))
    raise ValueError(f"unsupported site spec kind: {kind!r}")


def site_spec_to_key(site_spec: dict[str, object]) -> str:
    return site_spec_to_site(site_spec).key()


def full_timestep_mask(*, timestep: int) -> dict[str, object]:
    return {"kind": "full_state", "timestep": int(timestep)}


def coord_group_mask(*, timestep: int, coord_indices: Sequence[int]) -> dict[str, object]:
    coords = tuple(sorted(int(i) for i in coord_indices))
    return {"kind": "coord_group", "timestep": int(timestep), "coord_indices": list(coords)}


def output_logit_mask(*, output_index: int) -> dict[str, object]:
    return {"kind": "output_logit", "output_index": int(output_index)}
