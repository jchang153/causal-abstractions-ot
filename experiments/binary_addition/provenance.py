from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import BaseSplit


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(split: BaseSplit) -> dict[str, list[list[int]]]:
    return {
        name: [[int(example.a), int(example.b)] for example in getattr(split, name)]
        for name in ("fit", "calib", "test")
    }


def split_manifest_sha256(split: BaseSplit) -> str:
    payload = json.dumps(split_manifest(split), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def protocol_provenance(
    *,
    split: BaseSplit,
    checkpoint: str | Path,
    rows: tuple[str, ...],
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split_manifest_sha256": split_manifest_sha256(split),
        "split_sizes": {
            "fit": len(split.fit),
            "calib": len(split.calib),
            "test": len(split.test),
        },
        "rows": list(rows),
    }
