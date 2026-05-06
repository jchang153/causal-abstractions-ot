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


def _first_matching(
    rankings: dict[str, object],
    target_var: str,
    *,
    layer_selection_method: str | None = None,
    transport_method: str | None = None,
) -> dict[str, object] | None:
    entries = rankings.get(target_var)
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if layer_selection_method is not None and str(entry.get("layer_selection_method", "custom")) != str(layer_selection_method):
            continue
        if transport_method is not None and str(entry.get("transport_method", "ot")) != str(transport_method):
            continue
        return entry
    return None


def _first_matching_dimension(
    rankings: dict[str, object],
    target_var: str,
    *,
    layer_selection_method: str,
    stage_b_transport_method: str,
) -> dict[str, object] | None:
    entries = rankings.get(target_var)
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("layer_selection_method", "custom")) != str(layer_selection_method):
            continue
        if str(entry.get("stage_b_transport_method", entry.get("transport_method", "ot"))) != str(stage_b_transport_method):
            continue
        return entry
    return None


def _read_rankings(sweep_root: Path, filename: str) -> dict[str, object]:
    path = sweep_root / filename
    if not path.exists():
        return {}
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _task_status_runtimes(sweep_root: Path) -> dict[str, float]:
    status_dir = sweep_root / "task_status"
    if not status_dir.exists():
        return {}
    runtimes: dict[str, float] = {}
    for path in status_dir.glob("*.json"):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        task_id = payload.get("task_id")
        runtime = payload.get("runtime_seconds")
        if task_id is not None and runtime is not None:
            runtimes[str(task_id)] = _as_float(runtime)
    return runtimes


def _task_manifest_tasks(sweep_root: Path, manifest_name: str) -> list[dict[str, object]]:
    path = sweep_root / manifest_name
    if not path.exists():
        return []
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return []
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if isinstance(task, dict)]


def _task_branch(task: dict[str, object], *, default_transport: str = "ot") -> tuple[str, str]:
    return (
        str(task.get("layer_selection_method", "custom")),
        str(task.get("transport_method", default_transport)),
    )


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
            if isinstance(status, dict):
                if status.get("paper_runtime_seconds") is not None:
                    return _as_float(status.get("paper_runtime_seconds"))
                if status.get("runtime_seconds") is not None:
                    return _as_float(status.get("runtime_seconds"))
        for key, status in statuses.items():
            if str(key).startswith("stage_a") and isinstance(status, dict):
                if status.get("paper_runtime_seconds") is not None:
                    return _as_float(status.get("paper_runtime_seconds"))
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


def _task_payload_timing(task: dict[str, object], keys: Iterable[str]) -> float:
    total = 0.0
    for payload_path in task.get("payload_paths", []):
        path = Path(str(payload_path))
        if not path.exists():
            continue
        payload = _load_json(path)
        if isinstance(payload, dict):
            total += _payload_timing(payload, keys)
    return float(total)


def _stage_b_search_runtime_by_branch(
    sweep_root: Path,
    *,
    manifest_name: str,
    timing_keys: Iterable[str],
) -> dict[tuple[str, str], dict[str, float]]:
    status_runtimes = _task_status_runtimes(sweep_root)
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for task in _task_manifest_tasks(sweep_root, manifest_name):
        branch = _task_branch(task)
        seconds = _task_payload_timing(task, timing_keys)
        if seconds <= 0.0:
            seconds = _as_float(status_runtimes.get(str(task.get("task_id"))))
        if seconds <= 0.0:
            continue
        record = grouped.setdefault(branch, {"serial": 0.0, "parallel": 0.0})
        record["serial"] += float(seconds)
        record["parallel"] = max(float(record["parallel"]), float(seconds))
    return grouped


def _a_only_search_runtime_by_selection(sweep_root: Path) -> dict[str, dict[str, float]]:
    status_runtimes = _task_status_runtimes(sweep_root)
    grouped: dict[str, dict[str, float]] = {}
    for task in _task_manifest_tasks(sweep_root, "stage_c_a_only_tasks.json"):
        selection = str(task.get("layer_selection_method", "custom"))
        seconds = _as_float(status_runtimes.get(str(task.get("task_id"))))
        if seconds <= 0.0:
            for payload_path in task.get("payload_paths", []):
                path = Path(str(payload_path))
                if not path.exists():
                    continue
                payload = _load_json(path)
                if not isinstance(payload, dict):
                    continue
                for run_payload in payload.get("runs", []):
                    if not isinstance(run_payload, dict):
                        continue
                    method_payloads = run_payload.get("method_payloads", {})
                    if not isinstance(method_payloads, dict):
                        continue
                    for method_payload in method_payloads.get("das", []):
                        if isinstance(method_payload, dict):
                            seconds += _as_float(method_payload.get("runtime_seconds"))
        if seconds <= 0.0:
            continue
        record = grouped.setdefault(selection, {"serial": 0.0, "parallel": 0.0})
        record["serial"] += float(seconds)
        record["parallel"] = max(float(record["parallel"]), float(seconds))
    return grouped


def _selection_methods_from_sources(*rankings_payloads: dict[str, object], sweep_root: Path) -> tuple[str, ...]:
    methods: set[str] = set()
    for rankings in rankings_payloads:
        for entries in rankings.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    methods.add(str(entry.get("layer_selection_method", "custom")))
    for manifest_name in ("stage_b_native_tasks.json", "stage_b_pca_tasks.json", "stage_c_a_only_tasks.json"):
        for task in _task_manifest_tasks(sweep_root, manifest_name):
            methods.add(str(task.get("layer_selection_method", "custom")))
    return tuple(sorted(methods)) or ("custom",)


def _transport_methods_for_selection(*rankings_payloads: dict[str, object], selection_method: str) -> tuple[str, ...]:
    methods: set[str] = set()
    for rankings in rankings_payloads:
        for entries in rankings.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("layer_selection_method", "custom")) != str(selection_method):
                    continue
                method = str(entry.get("transport_method", "ot"))
                if method in {"ot", "uot"}:
                    methods.add(method)
    return tuple(sorted(methods)) or ("ot",)


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
    summary_payloads: list[dict[str, object]] = []
    runtimes_by_var: dict[str, float] = {}
    single_payload_runtimes: list[float] = []
    for path in full_das_outputs:
        if path.exists():
            maybe_summary = _load_json(path)
            if isinstance(maybe_summary, dict) and maybe_summary.get("method") == "Full DAS" and maybe_summary.get(
                "serial_runtime_seconds"
            ) is not None:
                summary_payloads.append(maybe_summary)
                continue
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

    if summary_payloads:
        serial = sum(_as_float(payload.get("serial_runtime_seconds")) for payload in summary_payloads)
        parallel = max(_as_float(payload.get("parallel_runtime_seconds")) for payload in summary_payloads)
        for payload in summary_payloads:
            runtime_seconds_by_var = payload.get("runtime_seconds_by_var")
            if isinstance(runtime_seconds_by_var, dict):
                for var in TARGET_VARS:
                    runtimes_by_var[var] = _as_float(runtimes_by_var.get(var)) + _as_float(runtime_seconds_by_var.get(var))
        notes = "Using layer-parallel Full DAS summary; serial sums layer payloads, parallel uses max layer payload runtime."
    elif single_payload_runtimes:
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
    native_dimension_rankings = _read_rankings(sweep_root, "stage_c_native_dimension_rankings.json")
    if not native_dimension_rankings:
        native_dimension_rankings = _read_rankings(sweep_root, "stage_c_native_dim_rankings.json")

    native_stage_b_by_layer = _native_stage_b_runtime_by_layer(sweep_root)
    pca_stage_b_by_layer = _pca_stage_b_runtime_by_layer(sweep_root)
    native_stage_b_by_branch = _stage_b_search_runtime_by_branch(
        sweep_root,
        manifest_name="stage_b_native_tasks.json",
        timing_keys=NATIVE_STAGE_B_KEYS,
    )
    pca_stage_b_by_branch = _stage_b_search_runtime_by_branch(
        sweep_root,
        manifest_name="stage_b_pca_tasks.json",
        timing_keys=PCA_STAGE_B_KEYS,
    )
    a_only_by_selection = _a_only_search_runtime_by_selection(sweep_root)
    pca_guided_rankings = _read_rankings(sweep_root, "stage_c_guided_rankings.json")

    records: list[dict[str, object]] = []
    records.append(
        _method_record(
            method="PLOT Stage A (one-sided UOT over layers)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var={var: 0.0 for var in TARGET_VARS},
            entries_by_var={var: _first_entry(stage_a_rankings, var) for var in TARGET_VARS},
            notes="Full Stage A all-layer final-token transport discovery. If stage_a_method=uot, this is one-sided UOT over layers.",
        )
    )

    selection_methods = _selection_methods_from_sources(
        native_rankings,
        pca_rankings,
        a_only_rankings,
        native_guided_rankings,
        native_dimension_rankings,
        pca_guided_rankings,
        sweep_root=sweep_root,
    )
    for selection_method in selection_methods:
        a_only_entries = {
            var: _first_matching(a_only_rankings, var, layer_selection_method=selection_method, transport_method="das")
            for var in TARGET_VARS
        }
        a_only_runtime = a_only_by_selection.get(selection_method, {"serial": 0.0, "parallel": 0.0})
        if any(a_only_entries.values()) or a_only_runtime["serial"] > 0.0:
            records.append(
                _method_record(
                    method=f"PLOT-DAS layer only ({selection_method})",
                    stage_a_seconds=stage_a_seconds,
                    downstream_by_var={
                        var: _as_float(entry.get("runtime_seconds")) if entry else 0.0
                        for var, entry in a_only_entries.items()
                    },
                    entries_by_var=a_only_entries,
                    serial_downstream_seconds=float(a_only_runtime["serial"]),
                    parallel_downstream_seconds=float(a_only_runtime["parallel"]),
                    notes="Stage A plus full-layer DAS over every layer selected by this Stage B layer-selection branch.",
                )
            )

        transport_methods = _transport_methods_for_selection(
            native_rankings,
            pca_rankings,
            native_guided_rankings,
            native_dimension_rankings,
            pca_guided_rankings,
            selection_method=selection_method,
        )
        for transport_method in transport_methods:
            native_branch_runtime = native_stage_b_by_branch.get((selection_method, transport_method), {"serial": 0.0, "parallel": 0.0})
            native_entries = {
                var: _first_matching(
                    native_rankings,
                    var,
                    layer_selection_method=selection_method,
                    transport_method=transport_method,
                )
                for var in TARGET_VARS
            }
            if any(native_entries.values()) or native_branch_runtime["serial"] > 0.0:
                records.append(
                    _method_record(
                        method=f"PLOT native transport ({selection_method}, {transport_method})",
                        stage_a_seconds=stage_a_seconds,
                        downstream_by_var={var: 0.0 for var in TARGET_VARS},
                        entries_by_var=native_entries,
                        serial_downstream_seconds=float(native_branch_runtime["serial"]),
                        parallel_downstream_seconds=float(native_branch_runtime["parallel"]),
                        notes="Stage A plus complete Stage B native search over the selected layer branch and all configured native resolutions.",
                    )
                )

            pca_branch_runtime = pca_stage_b_by_branch.get((selection_method, transport_method), {"serial": 0.0, "parallel": 0.0})
            pca_entries = {
                var: _first_matching(
                    pca_rankings,
                    var,
                    layer_selection_method=selection_method,
                    transport_method=transport_method,
                )
                for var in TARGET_VARS
            }
            if any(pca_entries.values()) or pca_branch_runtime["serial"] > 0.0:
                records.append(
                    _method_record(
                        method=f"PLOT PCA transport ({selection_method}, {transport_method})",
                        stage_a_seconds=stage_a_seconds,
                        downstream_by_var={var: 0.0 for var in TARGET_VARS},
                        entries_by_var=pca_entries,
                        serial_downstream_seconds=float(pca_branch_runtime["serial"]),
                        parallel_downstream_seconds=float(pca_branch_runtime["parallel"]),
                        notes="Stage A plus complete Stage B PCA search over the selected layer branch and configured PCA menus/bases.",
                    )
                )

            native_guided_entries = {
                var: _first_matching(
                    native_guided_rankings,
                    var,
                    layer_selection_method=selection_method,
                    transport_method=transport_method,
                )
                for var in TARGET_VARS
            }
            native_guided_das = {
                var: _as_float(entry.get("runtime_seconds")) if entry else 0.0
                for var, entry in native_guided_entries.items()
            }
            if any(native_guided_entries.values()):
                records.append(
                    _method_record(
                        method=f"PLOT-DAS native support ({selection_method}, {transport_method})",
                        stage_a_seconds=stage_a_seconds,
                        downstream_by_var={
                            var: float(native_branch_runtime["serial"]) + _as_float(native_guided_das.get(var))
                            for var in TARGET_VARS
                        },
                        entries_by_var=native_guided_entries,
                        serial_downstream_seconds=float(native_branch_runtime["serial"]) + sum(native_guided_das.values()),
                        parallel_downstream_seconds=float(native_branch_runtime["parallel"]) + max(native_guided_das.values()),
                        notes="Stage A plus complete Stage B native search plus guided DAS in the selected native support.",
                    )
                )

            native_dimension_entries = {
                var: _first_matching_dimension(
                    native_dimension_rankings,
                    var,
                    layer_selection_method=selection_method,
                    stage_b_transport_method=transport_method,
                )
                for var in TARGET_VARS
            }
            native_dimension_das = {
                var: _as_float(entry.get("runtime_seconds")) if entry else 0.0
                for var, entry in native_dimension_entries.items()
            }
            if any(native_dimension_entries.values()):
                records.append(
                    _method_record(
                        method=f"PLOT-DAS dimension ({selection_method}, {transport_method})",
                        stage_a_seconds=stage_a_seconds,
                        downstream_by_var={
                            var: float(native_branch_runtime["serial"]) + _as_float(native_dimension_das.get(var))
                            for var in TARGET_VARS
                        },
                        entries_by_var=native_dimension_entries,
                        serial_downstream_seconds=float(native_branch_runtime["serial"]) + sum(native_dimension_das.values()),
                        parallel_downstream_seconds=float(native_branch_runtime["parallel"]) + max(native_dimension_das.values()),
                        notes=(
                            "Stage A plus complete Stage B native search plus full-layer DAS on the selected "
                            "native layer with dimensions around the PLOT support width."
                        ),
                    )
                )

            pca_guided_entries = {
                var: _first_matching(
                    pca_guided_rankings,
                    var,
                    layer_selection_method=selection_method,
                    transport_method=transport_method,
                )
                for var in TARGET_VARS
            }
            pca_guided_das = {
                var: _as_float(entry.get("runtime_seconds")) if entry else 0.0
                for var, entry in pca_guided_entries.items()
            }
            if any(pca_guided_entries.values()):
                records.append(
                    _method_record(
                        method=f"PLOT-DAS PCA support ({selection_method}, {transport_method})",
                        stage_a_seconds=stage_a_seconds,
                        downstream_by_var={
                            var: float(pca_branch_runtime["serial"]) + _as_float(pca_guided_das.get(var))
                            for var in TARGET_VARS
                        },
                        entries_by_var=pca_guided_entries,
                        serial_downstream_seconds=float(pca_branch_runtime["serial"]) + sum(pca_guided_das.values()),
                        parallel_downstream_seconds=float(pca_branch_runtime["parallel"]) + max(pca_guided_das.values()),
                        notes="Stage A plus complete Stage B PCA search plus guided DAS in the selected PCA support.",
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
        "native_stage_b_search_runtime_seconds_by_branch": {
            f"{selection}:{transport}": {name: float(value) for name, value in runtime.items()}
            for (selection, transport), runtime in native_stage_b_by_branch.items()
        },
        "pca_stage_b_search_runtime_seconds_by_branch": {
            f"{selection}:{transport}": {name: float(value) for name, value in runtime.items()}
            for (selection, transport), runtime in pca_stage_b_by_branch.items()
        },
        "a_only_search_runtime_seconds_by_selection": {
            selection: {name: float(value) for name, value in runtime.items()}
            for selection, runtime in a_only_by_selection.items()
        },
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
