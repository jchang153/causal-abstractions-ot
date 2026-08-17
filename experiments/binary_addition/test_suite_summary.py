from __future__ import annotations

import json
from pathlib import Path

from experiments.binary_addition.summarize_10seed_suite import build_summary


STATS = {"mean": 1.0, "std": 0.1}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _direct() -> dict[str, object]:
    return {
        "accuracy": STATS,
        "sensitivity": STATS,
        "invariance": STATS,
        "runtime_seconds": STATS,
    }


def test_suite_summary_contains_exact_requested_method_set(tmp_path) -> None:
    _write(tmp_path / "single_stage/h16/single_stage_plot_summary.json", _direct())
    _write(tmp_path / "single_stage/h16/single_stage_plot_pca_summary.json", _direct())
    progressive = {
        "aggregate_methods": {
            key: _direct()
            for key in (
                "plot_in_timestep",
                "plot_pca_in_timestep",
                "plot_guided_das_full_timestep",
                "full_das",
            )
        }
    }
    _write(tmp_path / "progressive_ot/h16/progressive_summary.json", progressive)
    _write(
        tmp_path / "progressive_cosine/h16/progressive_summary.json",
        {"aggregate_methods": {"plot_in_timestep": _direct()}},
    )
    _write(
        tmp_path / "progressive_bruteforce/h16/progressive_summary.json",
        {"aggregate_methods": {"plot_in_timestep": _direct()}},
    )
    mib_record = {
        "combined": STATS,
        "sensitivity": STATS,
        "invariance": STATS,
        "runtime_seconds": STATS,
    }
    _write(
        tmp_path / "mib/aggregate.json",
        {key: mib_record for key in ("full-state", "dbm-canonical", "dbm-pca")},
    )

    summary = build_summary(tmp_path)
    assert summary["seeds"] == list(range(10))
    assert list(summary["methods"]) == [
        "PLOT (single-stage)",
        "PLOT-native (two-stage)",
        "PLOT-PCA (single-stage)",
        "PLOT-PCA (two-stage)",
        "PLOT-DAS",
        "Full DAS",
        "DBM (canonical)",
        "DBM (PCA)",
        "Full-vector",
        "PLOT-native-cosine",
        "PLOT-native-brute-force",
    ]
