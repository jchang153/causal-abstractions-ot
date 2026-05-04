"""Summarize MCQA Stage A UOT rowwise-vs-joint diagnostic sweeps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGET_VARS = ("answer_pointer", "answer_token")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _seed_from_name(name: str) -> int | None:
    match = re.search(r"_seed(\d+)_mcqa_hierarchical_sweep$", name)
    return None if match is None else int(match.group(1))


def _manifest_from_root(root: Path) -> dict[str, object] | None:
    manifest_path = root / "parallel_manifest.json"
    if not manifest_path.exists():
        return None
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    return payload


def _runtime_from_manifest(root: Path) -> tuple[float, float, str]:
    payload = _manifest_from_root(root)
    if payload is None:
        return 0.0, 0.0, "missing"
    statuses = payload.get("stage_statuses")
    if not isinstance(statuses, dict):
        return 0.0, 0.0, "malformed"
    for status in statuses.values():
        if not isinstance(status, dict):
            continue
        diagnostic = _as_float(status.get("diagnostic_runtime_seconds", status.get("runtime_seconds")))
        paper = status.get("paper_runtime_seconds")
        paper_runtime = diagnostic if paper is None else _as_float(paper)
        return paper_runtime, diagnostic, str(status.get("stage_a_runtime_policy", "unknown"))
    return 0.0, 0.0, "missing_status"


def _mode_from_root(root: Path) -> str:
    payload = _manifest_from_root(root)
    if payload is not None:
        selection = str(payload.get("stage_a_hparam_selection", ""))
        if selection in {"rowwise", "joint"}:
            return selection
    return "rowwise" if "rowwise" in root.name else "joint"


def _top_entry(rankings: dict[str, object], target_var: str) -> dict[str, object]:
    entries = rankings.get(target_var)
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        return dict(entries[0])
    return {}


def _record_from_root(root: Path) -> dict[str, object] | None:
    seed = _seed_from_name(root.name)
    if seed is None:
        return None
    rankings_path = root / "stage_a_last_token_layer_rankings.json"
    if not rankings_path.exists():
        return None
    rankings = _load_json(rankings_path)
    if not isinstance(rankings, dict):
        return None
    mode = _mode_from_root(root)
    paper_runtime, diagnostic_runtime, runtime_policy = _runtime_from_manifest(root)
    top = {var: _top_entry(rankings, var) for var in TARGET_VARS}
    cal_scores = [_as_float(top[var].get("handle_calibration_score", top[var].get("selection_score"))) for var in TARGET_VARS]
    exact_scores = [_as_float(top[var].get("exact_acc")) for var in TARGET_VARS]
    return {
        "seed": int(seed),
        "mode": mode,
        "root": str(root),
        "paper_runtime_seconds": float(paper_runtime),
        "diagnostic_runtime_seconds": float(diagnostic_runtime),
        "runtime_policy": runtime_policy,
        "joint_calibration_score": float(sum(cal_scores) / len(cal_scores)),
        "joint_exact_acc": float(sum(exact_scores) / len(exact_scores)),
        "top_by_var": {
            var: {
                "layer": top[var].get("layer"),
                "mass": _as_float(top[var].get("target_mass", top[var].get("layer_score", top[var].get("selection_score")))),
                "calibration_score": _as_float(top[var].get("handle_calibration_score", top[var].get("selection_score"))),
                "exact_acc": _as_float(top[var].get("exact_acc")),
                "epsilon": top[var].get("epsilon"),
                "method": top[var].get("method"),
                "uot_beta_neural": top[var].get("uot_beta_neural"),
                "selection_basis": top[var].get("selection_basis"),
            }
            for var in TARGET_VARS
        },
    }


def summarize(base: Path) -> dict[str, object]:
    records = [
        record
        for root in sorted(base.glob("*_mcqa_hierarchical_sweep"))
        for record in [_record_from_root(root)]
        if record is not None
    ]
    rowwise = [record for record in records if record["mode"] == "rowwise"]
    joint_candidates = [record for record in records if record["mode"] == "joint"]
    joint_selected: list[dict[str, object]] = []
    for seed in sorted({int(record["seed"]) for record in joint_candidates}):
        candidates = [record for record in joint_candidates if int(record["seed"]) == seed]
        joint_selected.append(
            max(
                candidates,
                key=lambda record: (
                    float(record["joint_calibration_score"]),
                    float(record["joint_exact_acc"]),
                    -float(record["paper_runtime_seconds"]),
                ),
            )
        )
    return {
        "kind": "mcqa_stage_a_uot_modes_summary",
        "base": str(base),
        "rowwise": rowwise,
        "joint_candidates": joint_candidates,
        "joint_selected": joint_selected,
    }


def _write_tsv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "mode",
        "seed",
        "joint_calibration_score",
        "joint_exact_acc",
        "paper_runtime_seconds",
        "diagnostic_runtime_seconds",
        "ap_layer",
        "ap_eps",
        "ap_method",
        "ap_beta",
        "at_layer",
        "at_eps",
        "at_method",
        "at_beta",
        "root",
    ]
    lines = ["\t".join(fields)]
    for record in records:
        top = record["top_by_var"]
        lines.append(
            "\t".join(
                [
                    str(record["mode"]),
                    str(record["seed"]),
                    f"{float(record['joint_calibration_score']):.6f}",
                    f"{float(record['joint_exact_acc']):.6f}",
                    f"{float(record['paper_runtime_seconds']):.2f}",
                    f"{float(record['diagnostic_runtime_seconds']):.2f}",
                    str(top["answer_pointer"]["layer"]),
                    str(top["answer_pointer"]["epsilon"]),
                    str(top["answer_pointer"]["method"]),
                    str(top["answer_pointer"]["uot_beta_neural"]),
                    str(top["answer_token"]["layer"]),
                    str(top["answer_token"]["epsilon"]),
                    str(top["answer_token"]["method"]),
                    str(top["answer_token"]["uot_beta_neural"]),
                    str(record["root"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    payload = summarize(args.base)
    out_prefix = args.out_prefix or (args.base / "stageA_uot_modes_summary")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".json")).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_tsv(out_prefix.with_suffix(".rowwise.tsv"), list(payload["rowwise"]))
    _write_tsv(out_prefix.with_suffix(".joint_selected.tsv"), list(payload["joint_selected"]))
    print(f"wrote {out_prefix.with_suffix('.json')}")
    print(f"wrote {out_prefix.with_suffix('.rowwise.tsv')}")
    print(f"wrote {out_prefix.with_suffix('.joint_selected.tsv')}")


if __name__ == "__main__":
    main()
