from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TARGET_VARS = ("answer_pointer", "answer_token")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _iter_run_payloads(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    runs = payload.get("runs")
    if isinstance(runs, list):
        return [run for run in runs if isinstance(run, dict)]
    return [payload]


def _layer_from_path(path: Path) -> int | None:
    match = re.search(r"layer_(\d+)_full_das_timed_mcqa", str(path))
    if match:
        return int(match.group(1))
    return None


def _collect_layer_outputs(sweep_root: Path) -> list[Path]:
    return sorted(
        sweep_root.glob("layer_*_full_das_timed_mcqa/mcqa_run_results.json"),
        key=lambda path: (_layer_from_path(path) if _layer_from_path(path) is not None else 10**9, str(path)),
    )


def _selected_result(method_payload: dict[str, Any], fallback_layer: int | None) -> dict[str, Any] | None:
    results = method_payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    result = results[0]
    if not isinstance(result, dict):
        return None
    layer = result.get("layer", fallback_layer)
    return {
        "target_var": str(method_payload.get("target_var", result.get("variable", ""))),
        "layer": None if layer is None else int(layer),
        "site_label": result.get("site_label"),
        "subspace_dim": result.get("subspace_dim"),
        "selection_exact_acc": _as_float(
            result.get("selection_exact_acc", result.get("calibration_exact_acc", 0.0))
        ),
        "exact_acc": _as_float(result.get("exact_acc")),
        "runtime_seconds": _as_float(method_payload.get("runtime_seconds")),
        "output_path": None,
    }


def build_summary(sweep_root: Path, expected_layers: list[int] | None = None) -> dict[str, Any]:
    output_paths = _collect_layer_outputs(sweep_root)
    records_by_var: dict[str, list[dict[str, Any]]] = {var: [] for var in TARGET_VARS}
    runtime_by_layer: dict[int, float] = {}
    runtime_by_var: dict[str, float] = {var: 0.0 for var in TARGET_VARS}

    for output_path in output_paths:
        fallback_layer = _layer_from_path(output_path)
        layer_runtime = 0.0
        for payload in _iter_run_payloads(output_path):
            method_payloads = payload.get("method_payloads")
            if not isinstance(method_payloads, dict):
                continue
            for method_payload in method_payloads.get("das", []):
                if not isinstance(method_payload, dict):
                    continue
                selected = _selected_result(method_payload, fallback_layer)
                if selected is None:
                    continue
                selected["output_path"] = str(output_path)
                target_var = str(selected["target_var"])
                if target_var not in records_by_var:
                    continue
                records_by_var[target_var].append(selected)
                runtime = _as_float(selected.get("runtime_seconds"))
                runtime_by_var[target_var] += runtime
                layer = selected.get("layer")
                if layer is not None:
                    layer_runtime += runtime
        if fallback_layer is not None:
            runtime_by_layer[int(fallback_layer)] = runtime_by_layer.get(int(fallback_layer), 0.0) + layer_runtime

    selected_by_var: dict[str, dict[str, Any] | None] = {}
    for target_var, records in records_by_var.items():
        best: dict[str, Any] | None = None
        for record in sorted(records, key=lambda item: (int(item["layer"]), int(item.get("subspace_dim") or 0))):
            # Match the serial DAS sweep convention: calibration selects; ties keep the earlier layer.
            if best is None or _as_float(record["selection_exact_acc"]) > _as_float(best["selection_exact_acc"]):
                best = record
        selected_by_var[target_var] = best

    serial_runtime = sum(runtime_by_layer.values())
    parallel_runtime = max(runtime_by_layer.values()) if runtime_by_layer else 0.0
    selected_values = [
        _as_float(selected_by_var[var]["exact_acc"])
        for var in TARGET_VARS
        if selected_by_var.get(var) is not None
    ]
    avg_exact = sum(selected_values) / len(selected_values) if selected_values else 0.0

    completed_layers = sorted(runtime_by_layer)
    missing_layers = []
    if expected_layers is not None:
        missing_layers = [layer for layer in expected_layers if layer not in set(completed_layers)]

    return {
        "method": "Full DAS",
        "sweep_root": str(sweep_root),
        "completed_layers": completed_layers,
        "missing_layers": missing_layers,
        "num_completed_layers": len(completed_layers),
        "num_outputs": len(output_paths),
        "selected_by_var": selected_by_var,
        "avg_exact_acc": float(avg_exact),
        "serial_runtime_seconds": float(serial_runtime),
        "parallel_runtime_seconds": float(parallel_runtime),
        "runtime_seconds_by_layer": {str(layer): float(seconds) for layer, seconds in sorted(runtime_by_layer.items())},
        "runtime_seconds_by_var": {var: float(seconds) for var, seconds in runtime_by_var.items()},
        "full_das_outputs": [str(path) for path in output_paths],
        "test_used_for_selection": False,
        "runtime_notes": "Layer-parallel Full DAS: serial sums all per-layer AP/AT DAS payloads; parallel is max per-layer AP+AT payload runtime.",
    }


def format_summary(payload: dict[str, Any]) -> str:
    selected_by_var = payload.get("selected_by_var", {})
    lines = [
        "MCQA Full DAS Layer-Parallel Summary",
        f"sweep_root: {payload.get('sweep_root')}",
        f"completed_layers: {payload.get('completed_layers')}",
        f"missing_layers: {payload.get('missing_layers')}",
        "test_used_for_selection: false",
        "",
        "method\tavg_exact\tAP\tAT\tserial_runtime_s\tparallel_runtime_s",
        (
            "Full DAS\t"
            f"{_as_float(payload.get('avg_exact_acc')):.4f}\t"
            f"{_as_float((selected_by_var.get('answer_pointer') or {}).get('exact_acc')):.4f}\t"
            f"{_as_float((selected_by_var.get('answer_token') or {}).get('exact_acc')):.4f}\t"
            f"{_as_float(payload.get('serial_runtime_seconds')):.2f}\t"
            f"{_as_float(payload.get('parallel_runtime_seconds')):.2f}"
        ),
        "",
        "selected_by_var:",
    ]
    for var in TARGET_VARS:
        selected = selected_by_var.get(var)
        if not selected:
            lines.append(f"{var}\tMISSING")
            continue
        lines.append(
            f"{var}\tlayer={selected.get('layer')}\tdim={selected.get('subspace_dim')}\t"
            f"exact={_as_float(selected.get('exact_acc')):.4f}\t"
            f"cal={_as_float(selected.get('selection_exact_acc')):.4f}\t"
            f"runtime_s={_as_float(selected.get('runtime_seconds')):.2f}\t"
            f"site={selected.get('site_label')}"
        )
    return "\n".join(lines) + "\n"


def _parse_layers(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate layer-parallel MCQA Full DAS outputs.")
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--expected-layers", default=None)
    args = parser.parse_args()

    payload = build_summary(args.sweep_root, expected_layers=_parse_layers(args.expected_layers))
    args.sweep_root.mkdir(parents=True, exist_ok=True)
    json_path = args.sweep_root / "full_das_layer_parallel_summary.json"
    txt_path = args.sweep_root / "full_das_layer_parallel_summary.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    txt_path.write_text(format_summary(payload), encoding="utf-8")
    print(format_summary(payload), end="")
    print(f"Wrote Full DAS layer-parallel summary to {json_path}")


if __name__ == "__main__":
    main()
