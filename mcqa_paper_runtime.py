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


def _signature_setup_seconds_from_payload(payload: dict[str, object]) -> float:
    direct = _as_float(payload.get("signature_prepare_runtime_seconds"))
    if direct > 0.0:
        return direct
    recorded = _as_float(payload.get("artifact_prepare_recorded_seconds"))
    if recorded > 0.0:
        return recorded
    timing_seconds = payload.get("timing_seconds", {})
    if isinstance(timing_seconds, dict):
        timed_direct = _as_float(timing_seconds.get("t_signature_prepare"))
        if timed_direct > 0.0:
            return timed_direct
        timed_recorded = _as_float(timing_seconds.get("t_artifact_prepare_recorded"))
        if timed_recorded > 0.0:
            return timed_recorded
        return (
            _as_float(timing_seconds.get("t_artifact_prepare_load"))
            + _as_float(timing_seconds.get("t_artifact_prepare_create"))
        )
    return 0.0


def _first_entry(rankings: dict[str, object], target_var: str) -> dict[str, object] | None:
    entries = rankings.get(target_var)
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    return first if isinstance(first, dict) else None


def _stage_a_display_entry(rankings: dict[str, object], target_var: str) -> dict[str, object] | None:
    display_by_var = rankings.get("_display_method_by_var")
    if isinstance(display_by_var, dict):
        entry = display_by_var.get(target_var)
        if isinstance(entry, dict):
            return entry
    return _first_entry(rankings, target_var)


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
    fallback = _read_rankings(sweep_root, "stage_a_last_token_layer_rankings.json")
    for target_var in TARGET_VARS:
        entry = _first_entry(fallback, target_var)
        if entry and entry.get("runtime_seconds") is not None:
            return _as_float(entry.get("runtime_seconds"))
    return 0.0


def _stage_a_payload_path(sweep_root: Path) -> Path | None:
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

        def _status_payload_path(status: object) -> Path | None:
            if not isinstance(status, dict):
                return None
            expected_outputs = status.get("expected_outputs")
            if not isinstance(expected_outputs, list):
                return None
            for output_path in expected_outputs:
                output = Path(str(output_path))
                if output.name.startswith("mcqa_plot_layer_pos-") and output.suffix == ".json":
                    return output
            return None

        for key in ("stage_a_last_token", "stage_a"):
            output = _status_payload_path(statuses.get(key))
            if output is not None:
                return output
        for key, status in statuses.items():
            if not str(key).startswith("stage_a"):
                continue
            output = _status_payload_path(status)
            if output is not None:
                return output
    return None


def _runtime_by_layer_from_rankings(rankings: dict[str, object]) -> dict[int, float]:
    runtime_by_layer: dict[int, float] = {}
    for target_var in TARGET_VARS:
        entries = rankings.get(target_var)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("layer") is None:
                continue
            layer = int(entry["layer"])
            runtime_by_layer[layer] = max(
                float(runtime_by_layer.get(layer, 0.0)),
                _as_float(entry.get("runtime_seconds")),
            )
    return runtime_by_layer


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


def _float_matches(lhs: object, rhs: object, *, tol: float = 1e-9) -> bool:
    return abs(_as_float(lhs) - _as_float(rhs)) <= float(tol)


def _native_selected_width_epsilon_runtime(
    *,
    rankings: dict[str, object],
    entries_by_var: dict[str, dict[str, object] | None],
    restrict_to_selected_layer: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[int, float]]]:
    runtime_grid: dict[tuple[str, int, int, float], float] = {}
    payload_paths: set[Path] = set()
    for target_var in TARGET_VARS:
        entries = rankings.get(target_var)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            payload_path = entry.get("payload_path")
            if payload_path:
                payload_paths.add(Path(str(payload_path)))

    for payload_path in payload_paths:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict):
            continue
        layer = int(payload.get("layer", -1))
        native_resolution = int(payload.get("native_resolution", payload.get("atomic_width", -1)))
        signature_setup_seconds = _signature_setup_seconds_from_payload(payload)
        for ot_path_str in payload.get("ot_output_paths", []):
            compare_payload = _load_json(Path(str(ot_path_str)))
            if not isinstance(compare_payload, dict):
                continue
            epsilon = _as_float(compare_payload.get("ot_epsilon"))
            method_payloads = compare_payload.get("method_payloads", {})
            if not isinstance(method_payloads, dict):
                continue
            for method_payload in method_payloads.get("ot", []):
                if not isinstance(method_payload, dict):
                    continue
                target_var = str(method_payload.get("target_var"))
                if target_var not in TARGET_VARS:
                    continue
                method_runtime_seconds = _as_float(
                    method_payload.get("runtime_seconds", method_payload.get("wall_runtime_seconds"))
                )
                method_signature_seconds = _as_float(method_payload.get("signature_prepare_runtime_seconds"))
                runtime_grid[(target_var, layer, native_resolution, epsilon)] = float(
                    method_runtime_seconds - method_signature_seconds + signature_setup_seconds
                )

    downstream_by_var: dict[str, float] = {}
    parallel_by_var: dict[str, float] = {}
    runtime_by_layer_by_var: dict[str, dict[int, float]] = {}
    for target_var, entry in entries_by_var.items():
        if not entry:
            downstream_by_var[target_var] = 0.0
            parallel_by_var[target_var] = 0.0
            runtime_by_layer_by_var[target_var] = {}
            continue
        selected_layer = int(entry.get("layer", -1))
        selected_width = int(entry.get("native_resolution", entry.get("atomic_width", -1)))
        selected_epsilon = _as_float(entry.get("epsilon"))
        layer_runtimes: dict[int, float] = {}
        for (row_var, layer, native_resolution, epsilon), runtime_seconds in runtime_grid.items():
            if str(row_var) != str(target_var):
                continue
            if restrict_to_selected_layer and int(layer) != int(selected_layer):
                continue
            if int(native_resolution) != int(selected_width):
                continue
            if not _float_matches(epsilon, selected_epsilon):
                continue
            layer_runtimes[int(layer)] = float(runtime_seconds)
        runtime_by_layer_by_var[str(target_var)] = layer_runtimes
        values = list(layer_runtimes.values())
        downstream_by_var[str(target_var)] = float(sum(values))
        parallel_by_var[str(target_var)] = float(max(values) if values else 0.0)
    return downstream_by_var, parallel_by_var, runtime_by_layer_by_var


def _matching_native_stage_b_entries(
    *,
    native_rankings: dict[str, object],
    guided_entries_by_var: dict[str, dict[str, object] | None],
) -> dict[str, dict[str, object] | None]:
    matched: dict[str, dict[str, object] | None] = {}
    for target_var, guided_entry in guided_entries_by_var.items():
        matched[target_var] = None
        if not guided_entry:
            continue
        selected_layer = int(guided_entry.get("layer", -1))
        selected_width = int(guided_entry.get("native_resolution", guided_entry.get("atomic_width", -1)))
        entries = native_rankings.get(target_var)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if int(entry.get("layer", -1)) != selected_layer:
                continue
            if int(entry.get("native_resolution", entry.get("atomic_width", -1))) != selected_width:
                continue
            matched[target_var] = entry
            break
    return matched


def _pca_selected_config_epsilon_runtime(
    *,
    rankings: dict[str, object],
    entries_by_var: dict[str, dict[str, object] | None],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[int, float]]]:
    runtime_grid: dict[tuple[str, int, str, str, int, float], float] = {}
    payload_paths: set[Path] = set()
    for target_var in TARGET_VARS:
        entries = rankings.get(target_var)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            payload_path = entry.get("layer_payload_path") or entry.get("payload_path")
            if payload_path:
                payload_paths.add(Path(str(payload_path)))

    for payload_path in payload_paths:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict):
            continue
        layer = int(payload.get("layer", -1))
        basis_source_mode = str(payload.get("basis_source_mode"))
        site_menu = str(payload.get("site_menu"))
        num_bands = int(payload.get("num_bands", -1))
        timing_seconds = payload.get("timing_seconds", {})
        config_setup_seconds = (
            _as_float(payload.get("pca_fit_runtime_seconds"))
            + _as_float(payload.get("pca_site_build_runtime_seconds"))
        )
        if config_setup_seconds <= 0.0 and isinstance(timing_seconds, dict):
            config_setup_seconds = (
                _as_float(timing_seconds.get("t_stageB_pca_fit"))
                + _as_float(timing_seconds.get("t_stageB_pca_site_build"))
            )
        config_setup_seconds += _signature_setup_seconds_from_payload(payload)
        for ot_path_str in payload.get("ot_output_paths", []):
            compare_payload = _load_json(Path(str(ot_path_str)))
            if not isinstance(compare_payload, dict):
                continue
            epsilon = _as_float(compare_payload.get("ot_epsilon"))
            method_payloads = compare_payload.get("method_payloads", {})
            if not isinstance(method_payloads, dict):
                continue
            for method_payload in method_payloads.get("ot", []):
                if not isinstance(method_payload, dict):
                    continue
                target_var = str(method_payload.get("target_var"))
                if target_var not in TARGET_VARS:
                    continue
                method_runtime_seconds = _as_float(
                    method_payload.get("runtime_seconds", method_payload.get("wall_runtime_seconds"))
                )
                method_signature_seconds = _as_float(method_payload.get("signature_prepare_runtime_seconds"))
                runtime_grid[(target_var, layer, basis_source_mode, site_menu, num_bands, epsilon)] = float(
                    method_runtime_seconds - method_signature_seconds + config_setup_seconds
                )

    downstream_by_var: dict[str, float] = {}
    parallel_by_var: dict[str, float] = {}
    runtime_by_layer_by_var: dict[str, dict[int, float]] = {}
    for target_var, entry in entries_by_var.items():
        if not entry:
            downstream_by_var[target_var] = 0.0
            parallel_by_var[target_var] = 0.0
            runtime_by_layer_by_var[target_var] = {}
            continue
        selected_basis = str(entry.get("basis_source_mode"))
        selected_menu = str(entry.get("site_menu"))
        selected_bands = int(entry.get("num_bands", -1))
        selected_epsilon = _as_float(entry.get("epsilon"))
        layer_runtimes: dict[int, float] = {}
        for (row_var, layer, basis_source_mode, site_menu, num_bands, epsilon), runtime_seconds in runtime_grid.items():
            if str(row_var) != str(target_var):
                continue
            if str(basis_source_mode) != selected_basis:
                continue
            if str(site_menu) != selected_menu:
                continue
            if int(num_bands) != selected_bands:
                continue
            if not _float_matches(epsilon, selected_epsilon):
                continue
            layer_runtimes[int(layer)] = float(runtime_seconds)
        runtime_by_layer_by_var[str(target_var)] = layer_runtimes
        values = list(layer_runtimes.values())
        downstream_by_var[str(target_var)] = float(sum(values))
        parallel_by_var[str(target_var)] = float(max(values) if values else 0.0)
    return downstream_by_var, parallel_by_var, runtime_by_layer_by_var


def _stage_a_selected_plan_runtime(
    *,
    sweep_root: Path,
    rankings: dict[str, object],
    entries_by_var: dict[str, dict[str, object] | None],
) -> tuple[dict[str, float], dict[str, dict[str, object]], float | None]:
    stage_a_payload_path = _stage_a_payload_path(sweep_root)
    if stage_a_payload_path is None or not stage_a_payload_path.exists():
        return ({var: 0.0 for var in TARGET_VARS}, {}, None)

    stage_a_payload = _load_json(stage_a_payload_path)
    if not isinstance(stage_a_payload, dict):
        return ({var: 0.0 for var in TARGET_VARS}, {}, None)

    selected_joint_config = stage_a_payload.get("selected_joint_config")
    if isinstance(selected_joint_config, dict):
        shared_runtime_seconds = _as_float(
            selected_joint_config.get(
                "runtime_with_signatures_seconds",
                selected_joint_config.get("runtime_seconds"),
            )
        )
        if shared_runtime_seconds > 0.0:
            downstream_by_var = {
                str(target_var): (shared_runtime_seconds if entries_by_var.get(str(target_var)) else 0.0)
                for target_var in TARGET_VARS
            }
            selected_records = {
                str(target_var): {
                    "target_var": str(target_var),
                    "method": "uot",
                    "epsilon": _as_float(selected_joint_config.get("epsilon")),
                    "uot_beta_neural": (
                        None
                        if selected_joint_config.get("uot_beta_neural") is None
                        else _as_float(selected_joint_config.get("uot_beta_neural"))
                    ),
                    "lambda": _as_float(selected_joint_config.get("lambda"), default=1.0),
                    "runtime_seconds": float(shared_runtime_seconds),
                    "payload_path": str(stage_a_payload_path),
                    "shared_joint_coupling": True,
                }
                for target_var in TARGET_VARS
                if entries_by_var.get(str(target_var))
            }
            return downstream_by_var, selected_records, float(shared_runtime_seconds)

    runtime_records: list[dict[str, object]] = []
    for compare_path_str in stage_a_payload.get("ot_output_paths", []):
        compare_payload = _load_json(Path(str(compare_path_str)))
        if not isinstance(compare_payload, dict):
            continue
        compare_epsilon = _as_float(compare_payload.get("ot_epsilon"))
        compare_beta = compare_payload.get("uot_beta_neural")
        method_payloads = compare_payload.get("method_payloads", {})
        if not isinstance(method_payloads, dict):
            continue
        for method_name, payloads in method_payloads.items():
            if not isinstance(payloads, list):
                continue
            for method_payload in payloads:
                if not isinstance(method_payload, dict):
                    continue
                target_var = str(method_payload.get("target_var"))
                if target_var not in TARGET_VARS:
                    continue
                results = method_payload.get("results", [])
                result0 = results[0] if isinstance(results, list) and results else {}
                method = str(
                    method_payload.get(
                        "method",
                        result0.get("method", method_name),
                    )
                )
                runtime_records.append(
                    {
                        "target_var": target_var,
                        "method": method,
                        "epsilon": compare_epsilon,
                        "uot_beta_neural": None
                        if method_payload.get("uot_beta_neural", compare_beta) is None
                        else _as_float(method_payload.get("uot_beta_neural", compare_beta)),
                        "runtime_seconds": _as_float(
                            method_payload.get("runtime_seconds", method_payload.get("wall_runtime_seconds"))
                        ),
                        "payload_path": str(compare_path_str),
                    }
                )

    downstream_by_var: dict[str, float] = {}
    selected_records: dict[str, dict[str, object]] = {}
    for target_var, entry in entries_by_var.items():
        if not entry:
            downstream_by_var[target_var] = 0.0
            continue
        epsilon = _as_float(entry.get("epsilon"))
        selected_beta = entry.get("uot_beta_neural")
        selected_method = entry.get("method")
        matches = [
            record
            for record in runtime_records
            if str(record.get("target_var")) == str(target_var)
            and _float_matches(record.get("epsilon"), epsilon)
        ]
        if selected_beta is not None:
            matches = [
                record
                for record in matches
                if record.get("uot_beta_neural") is not None
                and _float_matches(record.get("uot_beta_neural"), selected_beta)
            ]
        if selected_method is not None:
            matches = [record for record in matches if str(record.get("method")) == str(selected_method)]
        elif selected_beta is not None:
            uot_matches = [record for record in matches if str(record.get("method")) == "uot"]
            if uot_matches:
                matches = uot_matches
        elif len(matches) > 1:
            ot_matches = [record for record in matches if str(record.get("method")) == "ot"]
            if ot_matches:
                matches = ot_matches

        selected = matches[0] if matches else None
        downstream_by_var[target_var] = _as_float(selected.get("runtime_seconds")) if selected else 0.0
        if selected is not None:
            selected_records[target_var] = dict(selected)
    return downstream_by_var, selected_records, None


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
    native_rankings = _read_rankings(sweep_root, "stage_b_native_support_rankings.json")
    pca_rankings = _read_rankings(sweep_root, "stage_b_pca_support_rankings.json")
    layer_das_rankings = _read_rankings(sweep_root, "stage_c_layer_das_rankings.json")
    native_guided_rankings = _read_rankings(sweep_root, "stage_c_native_support_das_rankings.json")
    pca_guided_rankings = _read_rankings(sweep_root, "stage_c_pca_support_das_rankings.json")

    native_stage_b_by_layer = _runtime_by_layer_from_rankings(native_rankings)
    pca_stage_b_by_layer = _runtime_by_layer_from_rankings(pca_rankings)

    stage_a_entries = {var: _stage_a_display_entry(stage_a_rankings, var) for var in TARGET_VARS}
    stage_a_selected_runtime_by_var, stage_a_selected_runtime_records, stage_a_selected_shared_runtime = _stage_a_selected_plan_runtime(
        sweep_root=sweep_root,
        rankings=stage_a_rankings,
        entries_by_var=stage_a_entries,
    )
    records: list[dict[str, object]] = []
    records.append(
        _method_record(
            method="PLOT (layer)",
            stage_a_seconds=0.0,
            downstream_by_var=stage_a_selected_runtime_by_var,
            entries_by_var=stage_a_entries,
            serial_downstream_seconds=stage_a_selected_shared_runtime,
            parallel_downstream_seconds=stage_a_selected_shared_runtime,
            notes="Reported runtime charges only the selected Stage A shared coupling plan, including signature formation and argmax-layer evaluation, not the full Stage A sweep over coupling hyperparameters.",
        )
    )
    if stage_a_selected_runtime_records:
        records[-1]["selected_stage_a_plan_runtime_records_by_var"] = stage_a_selected_runtime_records
    if stage_a_selected_shared_runtime is not None:
        records[-1]["selected_stage_a_shared_runtime_seconds"] = float(stage_a_selected_shared_runtime)

    native_entries = {var: _first_entry(native_rankings, var) for var in TARGET_VARS}
    native_downstream, native_parallel_by_var, native_runtime_by_layer_by_var = _native_selected_width_epsilon_runtime(
        rankings=native_rankings,
        entries_by_var=native_entries,
    )
    native_shared_by_layer = {
        layer: max(layer_runtimes.get(layer, 0.0) for layer_runtimes in native_runtime_by_layer_by_var.values())
        for layer in sorted({layer for layer_runtimes in native_runtime_by_layer_by_var.values() for layer in layer_runtimes})
    }
    native_serial = float(sum(_as_float(native_downstream.get(var)) for var in TARGET_VARS))
    native_parallel = float(max((_as_float(native_parallel_by_var.get(var)) for var in TARGET_VARS), default=0.0))
    records.append(
        _method_record(
            method="PLOT (native support)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=native_downstream,
            entries_by_var=native_entries,
            serial_downstream_seconds=native_serial,
            parallel_downstream_seconds=native_parallel,
            shared_runtime_seconds_by_layer=native_shared_by_layer,
            notes="Stage A plus the selected native-support width+epsilon slice, summed across all Stage B-executed layers for each variable; width/epsilon search overhead is excluded.",
        )
    )

    pca_entries = {var: _first_entry(pca_rankings, var) for var in TARGET_VARS}
    pca_downstream, pca_parallel_by_var, pca_runtime_by_layer_by_var = _pca_selected_config_epsilon_runtime(
        rankings=pca_rankings,
        entries_by_var=pca_entries,
    )
    pca_shared_by_layer = {
        layer: max(layer_runtimes.get(layer, 0.0) for layer_runtimes in pca_runtime_by_layer_by_var.values())
        for layer in sorted({layer for layer_runtimes in pca_runtime_by_layer_by_var.values() for layer in layer_runtimes})
    }
    pca_serial = float(sum(_as_float(pca_downstream.get(var)) for var in TARGET_VARS))
    pca_parallel = float(max((_as_float(pca_parallel_by_var.get(var)) for var in TARGET_VARS), default=0.0))
    records.append(
        _method_record(
            method="PLOT (PCA support)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=pca_downstream,
            entries_by_var=pca_entries,
            serial_downstream_seconds=pca_serial,
            parallel_downstream_seconds=pca_parallel,
            shared_runtime_seconds_by_layer=pca_shared_by_layer,
            notes="Stage A plus the selected PCA-support config+epsilon slice, summed across all Stage B-executed layers for each variable; PCA config and epsilon search overhead is excluded.",
        )
    )

    layer_das_entries = {var: _first_entry(layer_das_rankings, var) for var in TARGET_VARS}
    layer_das_downstream = {
        var: _as_float(entry.get("runtime_seconds")) if entry else 0.0
        for var, entry in layer_das_entries.items()
    }
    records.append(
        _method_record(
            method="PLOT-DAS (layer)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=layer_das_downstream,
            entries_by_var=layer_das_entries,
            notes="Stage A plus standalone layer-restricted DAS over the selected layers.",
        )
    )

    native_guided_entries = {var: _first_entry(native_guided_rankings, var) for var in TARGET_VARS}
    native_guided_stage_b_entries = _matching_native_stage_b_entries(
        native_rankings=native_rankings,
        guided_entries_by_var=native_guided_entries,
    )
    (
        native_guided_stage_b_downstream,
        native_guided_stage_b_parallel_by_var,
        native_guided_stage_b_runtime_by_layer_by_var,
    ) = _native_selected_width_epsilon_runtime(
        rankings=native_rankings,
        entries_by_var=native_guided_stage_b_entries,
        restrict_to_selected_layer=True,
    )
    native_guided_stage_b_by_layer = {
        layer: max(layer_runtimes.get(layer, 0.0) for layer_runtimes in native_guided_stage_b_runtime_by_layer_by_var.values())
        for layer in sorted(
            {layer for layer_runtimes in native_guided_stage_b_runtime_by_layer_by_var.values() for layer in layer_runtimes}
        )
    }
    native_guided_downstream = {}
    native_guided_das = {}
    for var, entry in native_guided_entries.items():
        if entry:
            stage_b_seconds = _as_float(native_guided_stage_b_downstream.get(var))
            das_seconds = _as_float(entry.get("runtime_seconds"))
            native_guided_das[var] = das_seconds
            native_guided_downstream[var] = stage_b_seconds + das_seconds
        else:
            native_guided_das[var] = 0.0
            native_guided_downstream[var] = 0.0
    native_guided_serial = sum(_as_float(native_guided_downstream.get(var)) for var in TARGET_VARS)
    native_guided_parallel = float(max((_as_float(native_guided_downstream.get(var)) for var in TARGET_VARS), default=0.0))
    records.append(
        _method_record(
            method="PLOT-DAS (native support)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=native_guided_downstream,
            entries_by_var=native_guided_entries,
            serial_downstream_seconds=native_guided_serial,
            parallel_downstream_seconds=native_guided_parallel,
            shared_runtime_seconds_by_layer=native_guided_stage_b_by_layer,
            notes="Stage A plus the exact selected native-support layer+width+epsilon localization runtime and DAS over the selected native supports.",
        )
    )

    pca_guided_entries = {var: _first_entry(pca_guided_rankings, var) for var in TARGET_VARS}
    pca_guided_stage_b_by_layer, pca_guided_stage_b_serial, _ = _unique_layer_runtime(
        pca_guided_entries,
        pca_stage_b_by_layer,
    )
    pca_guided_downstream = {}
    pca_guided_das = {}
    for var, entry in pca_guided_entries.items():
        if entry:
            stage_b_seconds = pca_stage_b_by_layer.get(int(entry["layer"]), 0.0)
            das_seconds = _as_float(entry.get("runtime_seconds"))
            pca_guided_das[var] = das_seconds
            pca_guided_downstream[var] = stage_b_seconds + das_seconds
        else:
            pca_guided_das[var] = 0.0
            pca_guided_downstream[var] = 0.0
    pca_guided_serial = pca_guided_stage_b_serial + sum(_as_float(pca_guided_das.get(var)) for var in TARGET_VARS)
    pca_guided_parallel = max(_as_float(pca_guided_downstream.get(var)) for var in TARGET_VARS)
    records.append(
        _method_record(
            method="PLOT-DAS (PCA support)",
            stage_a_seconds=stage_a_seconds,
            downstream_by_var=pca_guided_downstream,
            entries_by_var=pca_guided_entries,
            serial_downstream_seconds=pca_guided_serial,
            parallel_downstream_seconds=pca_guided_parallel,
            shared_runtime_seconds_by_layer=pca_guided_stage_b_by_layer,
            notes="Stage A plus PCA-support localization plus DAS over the selected PCA supports.",
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
