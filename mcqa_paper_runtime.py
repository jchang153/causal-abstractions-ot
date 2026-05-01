"""Paper-facing runtime accounting for MCQA hierarchical runs.

The hierarchy runners store many candidate payload runtimes. This module turns
those payloads into explicit selected-path runtimes for the paper table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


TARGET_VARS = ("answer_pointer", "answer_token")
NATIVE_STAGE_B_KEYS = (
    "t_stageB_native_signature_build",
    "t_stageB_native_ot_fit_cal",
)
PCA_STAGE_B_KEYS = (
    "t_stageB_pca_fit",
    "t_stageB_pca_site_build",
    "t_stageB_pca_ot_fit_cal",
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_entry(rankings: dict[str, object], target_var: str) -> dict[str, object] | None:
    entries = rankings.get(target_var)
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    return first if isinstance(first, dict) else None


def _read_rankings(sweep_root: Path, filename: str) -> dict[str, object]:
    path = sweep_root / filename
    if not path.exists():
        return {}
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _stage_a_runtime(sweep_root: Path) -> float:
    for filename in ("parallel_manifest.json", "hierarchical_sweep_manifest.json"):
        path = sweep_root / filename
        if not path.exists():
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        statuses = payload.get("stage_statuses")
        if not isinstance(statuses, dict):
            continue
        for key in ("stage_a_last_token", "stage_a"):
            status = statuses.get(key)
            if isinstance(status, dict) and status.get("runtime_seconds") is not None:
                return _as_float(status.get("runtime_seconds"))
        for key, status in statuses.items():
            if str(key).startswith("stage_a") and isinstance(status, dict):
                if status.get("runtime_seconds") is not None:
                    return _as_float(status.get("runtime_seconds"))
    return 0.0


def _task_payload_paths(sweep_root: Path, manifest_name: str) -> list[Path]:
    path = sweep_root / manifest_name
    if not path.exists():
        return []
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return []
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    paths: list[Path] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for payload_path in task.get("payload_paths", []):
            resolved = Path(str(payload_path))
            if resolved.exists():
                paths.append(resolved)
    return paths


def _payload_timing(payload: dict[str, object], keys: Iterable[str]) -> float:
    timing = payload.get("timing_seconds")
    if not isinstance(timing, dict):
        return 0.0
    return sum(_as_float(timing.get(key)) for key in keys)


def _payloads_by_layer(paths: Iterable[Path]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("layer") is None:
            continue
        grouped.setdefault(int(payload["layer"]), []).append(payload)
    return grouped


def _native_stage_b_runtime_by_layer(sweep_root: Path) -> dict[int, float]:
    payloads = _payloads_by_layer(_task_payload_paths(sweep_root, "stage_b_native_tasks.json"))
    return {
        layer: sum(_payload_timing(payload, NATIVE_STAGE_B_KEYS) for payload in layer_payloads)
        for layer, layer_payloads in payloads.items()
    }


def _pca_stage_b_runtime_by_layer(sweep_root: Path) -> dict[int, float]:
    payloads = _payloads_by_layer(_task_payload_paths(sweep_root, "stage_b_pca_tasks.json"))
    return {
        layer: sum(_payload_timing(payload, PCA_STAGE_B_KEYS) for payload in layer_payloads)
        for layer, layer_payloads in payloads.items()
    }


def _method_record(
    *,
    method: str,
    stage_a_seconds: float,
    downstream_by_var: dict[str, float],
    entries_by_var: dict[str, dict[str, object] | None],
    notes: str,
    serial_downstream_seconds: float | None = None,
    parallel_downstream_seconds: float | None = None,
    shared_runtime_seconds_by_layer: dict[int, float] | None = None,
) -> dict[str, object]:
    downstream = [_as_float(downstream_by_var.get(var)) for var in TARGET_VARS]
    serial_downstream = sum(downstream) if serial_downstream_seconds is None else float(serial_downstream_seconds)
    parallel_downstream = (
        max(downstream) if parallel_downstream_seconds is None and downstream else float(parallel_downstream_seconds or 0.0)
    )
    serial = float(stage_a_seconds) + float(serial_downstream)
    parallel = float(stage_a_seconds) + float(parallel_downstream)
    return {
        "method": method,
        "serial_runtime_seconds": serial,
        "parallel_runtime_seconds": parallel,
        "stage_a_runtime_seconds": float(stage_a_seconds),
        "downstream_runtime_seconds_by_var": {
            var: _as_float(downstream_by_var.get(var)) for var in TARGET_VARS
        },
        "shared_runtime_seconds_by_layer": {
            str(layer): float(seconds) for layer, seconds in (shared_runtime_seconds_by_layer or {}).items()
        },
        "selected_entries_by_var": entries_by_var,
        "test_used_for_selection": False,
        "runtime_notes": notes,
    }


def _unique_layer_runtime(
    entries_by_var: dict[str, dict[str, object] | None],
    runtime_by_layer: dict[int, float],
) -> tuple[dict[int, float], float, float]:
    layer_seconds: dict[int, float] = {}
    for entry in entries_by_var.values():
        if not entry or entry.get("layer") is None:
            continue
        layer = int(entry["layer"])
        layer_seconds[layer] = _as_float(runtime_by_layer.get(layer))
    values = list(layer_seconds.values())
    return layer_seconds, float(sum(values)), float(max(values) if values else 0.0)


def _iter_run_payloads(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return []
    runs = payload.get("runs")
    if isinstance(runs, list):
        return [run for run in runs if isinstance(run, dict)]
    return [payload]


def _full_das_record(full_das_outputs: list[Path]) -> dict[str, object] | None:
    if not full_das_outputs:
        return None
    runtimes_by_var: dict[str, float] = {}
    single_payload_runtimes: list[float] = []
    for path in full_das_outputs:
        for payload in _iter_run_payloads(path):
            if payload.get("runtime_seconds") is not None:
                single_payload_runtimes.append(_as_float(payload.get("runtime_seconds")))
                continue
            method_payloads = payload.get("method_payloads")
            if isinstance(method_payloads, dict):
                for method_payload in method_payloads.get("das", []):
                    if not isinstance(method_payload, dict):
                        continue
                    target_var = str(method_payload.get("target_var"))
                    if target_var in TARGET_VARS:
                        runtimes_by_var[target_var] = _as_float(runtimes_by_var.get(target_var)) + _as_float(
                            method_payload.get("runtime_seconds")
                        )

    if single_payload_runtimes:
        serial = sum(single_payload_runtimes)
        parallel = max(single_payload_runtimes)
        notes = "Using recorded joint Full DAS payload runtime; not double-counting AP/AT."
    elif runtimes_by_var:
        downstream = [_as_float(runtimes_by_var.get(var)) for var in TARGET_VARS]
        serial = sum(downstream)
        parallel = max(downstream) if downstream else 0.0
        notes = "Full DAS records per-variable payload runtimes; serial sums AP/AT, parallel takes the max."
    else:
        return None

    return {
        "method": "Full DAS",
        "serial_runtime_seconds": float(serial),
        "parallel_runtime_seconds": float(parallel),
        "stage_a_runtime_seconds": 0.0,
        "downstream_runtime_seconds_by_var": {
            var: _as_float(runtimes_by_var.get(var)) for var in TARGET_VARS
        },
        "full_das_outputs": [str(path) for path in full_das_outputs],
        "test_used_for_selection": False,
        "runtime_notes": notes,
    }


def build_paper_runtime_summary(
    *,
    sweep_root: Path,
    full_das_outputs: list[Path] | None = None,
) -> dict[str, object]:
    stage_a_seconds = _stage_a_runtime(sweep_root)
    stage_a_rankings = _read_rankings(sweep_root, "stage_a_last_token_layer_rankings.json")
    native_rankings = _read_rankings(sweep_root, "stage_b_native_rankings.json")
    pca_rankings = _read_rankings(sweep_root, "stage_b_pca_rankings.json")
    a_only_rankings = _read_rankings(sweep_root, "stage_c_a_only_rankings.json")
    native_guided_rankings = _read_rankings(sweep_root, "stage_c_native_guided_rankings.json")

    native_stage_b_by_layer = _native_stage_b_runtime_by_layer(sweep_root)
    pca_stage_b_by_layer = _pca_stage_b_runtime_by_layer(sweep_root)

    records: list[dict[str, object]] = []
    records.append(
        _method_record(
            method="PLOT (layer OT)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var={var: 0.0 for var in TARGET_VARS},
            entries_by_var={var: _first_entry(stage_a_rankings, var) for var in TARGET_VARS},
            notes="Full Stage A all-layer final-token OT discovery.",
        )
    )

    native_entries = {var: _first_entry(native_rankings, var) for var in TARGET_VARS}
    native_downstream = {
        var: native_stage_b_by_layer.get(int(entry["layer"]), 0.0) if entry else 0.0
        for var, entry in native_entries.items()
    }
    native_shared_by_layer, native_serial, native_parallel = _unique_layer_runtime(
        native_entries,
        native_stage_b_by_layer,
    )
    records.append(
        _method_record(
            method="PLOT (native OT)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=native_downstream,
            entries_by_var=native_entries,
            serial_downstream_seconds=native_serial,
            parallel_downstream_seconds=native_parallel,
            shared_runtime_seconds_by_layer=native_shared_by_layer,
            notes="Stage A plus local Stage B native search over all configured native resolutions in each unique selected layer.",
        )
    )

    pca_entries = {var: _first_entry(pca_rankings, var) for var in TARGET_VARS}
    pca_downstream = {
        var: pca_stage_b_by_layer.get(int(entry["layer"]), 0.0) if entry else 0.0
        for var, entry in pca_entries.items()
    }
    pca_shared_by_layer, pca_serial, pca_parallel = _unique_layer_runtime(
        pca_entries,
        pca_stage_b_by_layer,
    )
    records.append(
        _method_record(
            method="PLOT-PCA",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=pca_downstream,
            entries_by_var=pca_entries,
            serial_downstream_seconds=pca_serial,
            parallel_downstream_seconds=pca_parallel,
            shared_runtime_seconds_by_layer=pca_shared_by_layer,
            notes="Stage A plus local Stage B PCA search over configured PCA menus/bases in the selected layer.",
        )
    )

    a_only_entries = {var: _first_entry(a_only_rankings, var) for var in TARGET_VARS}
    a_only_downstream = {
        var: _as_float(entry.get("runtime_seconds")) if entry else 0.0
        for var, entry in a_only_entries.items()
    }
    records.append(
        _method_record(
            method="PLOT-DAS (layer only)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=a_only_downstream,
            entries_by_var=a_only_entries,
            notes="Stage A plus focused full-layer DAS for AP and AT at their selected layers.",
        )
    )

    native_guided_entries = {var: _first_entry(native_guided_rankings, var) for var in TARGET_VARS}
    native_guided_stage_b_by_layer, native_guided_stage_b_serial, _ = _unique_layer_runtime(
        native_guided_entries,
        native_stage_b_by_layer,
    )
    native_guided_downstream = {}
    native_guided_das = {}
    for var, entry in native_guided_entries.items():
        if entry:
            stage_b_seconds = native_stage_b_by_layer.get(int(entry["layer"]), 0.0)
            das_seconds = _as_float(entry.get("runtime_seconds"))
            native_guided_das[var] = das_seconds
            native_guided_downstream[var] = stage_b_seconds + das_seconds
        else:
            native_guided_das[var] = 0.0
            native_guided_downstream[var] = 0.0
    native_guided_serial = native_guided_stage_b_serial + sum(_as_float(native_guided_das.get(var)) for var in TARGET_VARS)
    native_guided_parallel = max(_as_float(native_guided_downstream.get(var)) for var in TARGET_VARS)
    records.append(
        _method_record(
            method="PLOT-DAS (native support)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=native_guided_downstream,
            entries_by_var=native_guided_entries,
            serial_downstream_seconds=native_guided_serial,
            parallel_downstream_seconds=native_guided_parallel,
            shared_runtime_seconds_by_layer=native_guided_stage_b_by_layer,
            notes="Stage A plus local Stage B native search plus guided DAS in the selected native support.",
        )
    )

    full_record = _full_das_record(full_das_outputs or [])
    if full_record is not None:
        records.append(full_record)

    return {
        "kind": "mcqa_paper_runtime_summary",
        "sweep_root": str(sweep_root),
        "stage_a_runtime_seconds": float(stage_a_seconds),
        "native_stage_b_runtime_seconds_by_layer": {str(k): float(v) for k, v in native_stage_b_by_layer.items()},
        "pca_stage_b_runtime_seconds_by_layer": {str(k): float(v) for k, v in pca_stage_b_by_layer.items()},
        "methods": records,
        "test_used_for_selection": False,
    }


def format_paper_runtime_summary(payload: dict[str, object]) -> str:
    lines = [
        "MCQA paper runtime summary",
        f"sweep_root: {payload.get('sweep_root')}",
        f"stage_a_runtime_seconds: {_as_float(payload.get('stage_a_runtime_seconds')):.2f}",
        "test_used_for_selection: false",
        "",
        "method\tserial_runtime_s\tparallel_runtime_s",
    ]
    for record in payload.get("methods", []):
        if not isinstance(record, dict):
            continue
        lines.append(
            f"{record.get('method')}\t"
            f"{_as_float(record.get('serial_runtime_seconds')):.2f}\t"
            f"{_as_float(record.get('parallel_runtime_seconds')):.2f}"
        )
    return "\n".join(lines) + "\n"


def write_paper_runtime_summary(
    *,
    sweep_root: Path,
    full_das_outputs: list[Path] | None = None,
) -> dict[str, object]:
    payload = build_paper_runtime_summary(sweep_root=sweep_root, full_das_outputs=full_das_outputs)
    _write_json(sweep_root / "paper_runtime_summary.json", payload)
    (sweep_root / "paper_runtime_summary.txt").write_text(format_paper_runtime_summary(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write paper-facing MCQA selected-path runtime summaries.")
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--full-das-output", type=Path, action="append", default=[])
    args = parser.parse_args()
    write_paper_runtime_summary(sweep_root=args.sweep_root, full_das_outputs=list(args.full_das_output))


if __name__ == "__main__":
    main()
