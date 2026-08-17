#!/usr/bin/env python3
"""Combine the requested 10-seed binary-addition methods into one summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing suite result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(payload: dict[str, object], key: str) -> dict[str, float]:
    value = payload[key]
    if not isinstance(value, dict) or "mean" not in value or "std" not in value:
        raise ValueError(f"expected mean/std statistics for {key!r}, got {value!r}")
    return {"mean": float(value["mean"]), "std": float(value["std"])}


def _direct_record(payload: dict[str, object]) -> dict[str, object]:
    return {
        "accuracy": _stats(payload, "accuracy"),
        "sensitivity": _stats(payload, "sensitivity"),
        "invariance": _stats(payload, "invariance"),
        "runtime_seconds": _stats(payload, "runtime_seconds"),
    }


def _progressive_record(payload: dict[str, object], method: str) -> dict[str, object]:
    methods = payload.get("aggregate_methods", {})
    if not isinstance(methods, dict) or method not in methods:
        raise KeyError(f"progressive summary is missing method {method!r}")
    record = methods[method]
    if not isinstance(record, dict):
        raise TypeError(f"invalid progressive method record for {method!r}")
    return _direct_record(record)


def _mib_record(payload: dict[str, object], method: str) -> dict[str, object]:
    record = payload.get(method)
    if not isinstance(record, dict):
        raise KeyError(f"MIB summary is missing method {method!r}")
    return {
        "accuracy": _stats(record, "combined"),
        "sensitivity": _stats(record, "sensitivity"),
        "invariance": _stats(record, "invariance"),
        "runtime_seconds": _stats(record, "runtime_seconds"),
    }


def build_summary(run_dir: Path) -> dict[str, object]:
    single = _read(run_dir / "single_stage" / "h16" / "single_stage_plot_summary.json")
    single_pca = _read(run_dir / "single_stage" / "h16" / "single_stage_plot_pca_summary.json")
    progressive = _read(run_dir / "progressive_ot" / "h16" / "progressive_summary.json")
    cosine = _read(run_dir / "progressive_cosine" / "h16" / "progressive_summary.json")
    brute_force = _read(run_dir / "progressive_bruteforce" / "h16" / "progressive_summary.json")
    mib = _read(run_dir / "mib" / "aggregate.json")

    return {
        "hidden_size": 16,
        "seeds": list(range(10)),
        "methods": {
            "PLOT (single-stage)": _direct_record(single),
            "PLOT-native (two-stage)": _progressive_record(progressive, "plot_in_timestep"),
            "PLOT-PCA (single-stage)": _direct_record(single_pca),
            "PLOT-PCA (two-stage)": _progressive_record(progressive, "plot_pca_in_timestep"),
            "PLOT-DAS": _progressive_record(progressive, "plot_guided_das_full_timestep"),
            "Full DAS": _progressive_record(progressive, "full_das"),
            "DBM (canonical)": _mib_record(mib, "dbm-canonical"),
            "DBM (PCA)": _mib_record(mib, "dbm-pca"),
            "Full-vector": _mib_record(mib, "full-state"),
            "PLOT-native-cosine": _progressive_record(cosine, "plot_in_timestep"),
            "PLOT-native-brute-force": _progressive_record(brute_force, "plot_in_timestep"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = build_summary(run_dir)
    output = run_dir / "suite_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"summary": str(output), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
