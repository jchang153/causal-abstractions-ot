from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

from mcqa_delta_hierarchical_sweep import (
    DEFAULT_TARGET_VARS,
    _all_layer_indices,
    _build_native_block_command,
    _build_parser,
    _build_stage_a_command,
    _build_stage_b_or_c_command,
    _extract_native_rankings,
    _extract_stage_a_rankings,
    _extract_stage_b_best_configs,
    _extract_stage_c_rankings,
    _format_native_summary,
    _format_stage_a_summary,
    _format_stage_b_summary,
    _format_stage_c_summary,
    _load_json,
    _normalize_args,
    _normalize_num_bands_values,
    _select_stage_b_layers,
    _select_stage_c_configs,
    _site_catalog_tag,
    _stage_b_slug,
    _stage_output_is_valid,
    _write_json,
    _write_text,
)


TASK_MANIFEST_VERSION = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _sweep_root(args: argparse.Namespace, normalized: dict[str, object]) -> Path:
    return Path(args.results_root) / f"{str(normalized['results_timestamp'])}_mcqa_hierarchical_sweep"


def _require_hf_token(args: argparse.Namespace) -> None:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and not bool(args.prompt_hf_login):
        raise ValueError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required unless --prompt-hf-login is set.")


def _read_rankings(path: Path) -> dict[str, list[dict[str, object]]]:
    if not _stage_output_is_valid(path):
        raise FileNotFoundError(f"Missing or invalid rankings file: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed rankings file: {path}")
    return {
        str(target_var): [dict(entry) for entry in entries if isinstance(entry, dict)]
        for target_var, entries in payload.items()
        if isinstance(entries, list)
    }


def _write_task_manifest(path: Path, *, kind: str, tasks: list[dict[str, object]]) -> None:
    _write_json(
        path,
        {
            "kind": kind,
            "version": TASK_MANIFEST_VERSION,
            "created_at": datetime.now().isoformat(),
            "task_count": len(tasks),
            "tasks": tasks,
        },
    )


def _load_task_manifest(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"Malformed task manifest: {path}")
    return payload


def _task_status_path(*, sweep_root: Path, task_id: str) -> Path:
    return sweep_root / "task_status" / f"{task_id}.json"


def _mark_task(*, sweep_root: Path, task: dict[str, object], state: str, extra: dict[str, object] | None = None) -> None:
    task_id = str(task["task_id"])
    payload = {
        "task_id": task_id,
        "category": task.get("category"),
        "state": state,
        "stage_timestamp": task.get("stage_timestamp"),
        "updated_at": datetime.now().isoformat(),
        "expected_outputs": task.get("expected_outputs", []),
    }
    if extra:
        payload.update(extra)
    _write_json(_task_status_path(sweep_root=sweep_root, task_id=task_id), payload)


def _task_statuses(*, sweep_root: Path) -> list[dict[str, object]]:
    status_dir = sweep_root / "task_status"
    if not status_dir.exists():
        return []
    statuses: list[dict[str, object]] = []
    for path in sorted(status_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            statuses.append(payload)
    return statuses


def _expected_pca_payload_path(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    token_position_id: str,
    basis_source_mode: str,
    site_menu: str,
    num_bands: int,
    layer: int,
) -> Path:
    site_catalog_tag = _site_catalog_tag(
        site_menu=str(site_menu),
        num_bands=int(num_bands),
        band_scheme=str(args.pca_band_scheme),
        top_prefix_sizes=normalized["pca_top_prefix_sizes"],
    )
    return (
        Path(args.results_root)
        / f"{stage_timestamp}_mcqa_ot_pca_focus"
        / f"layer_{int(layer):02d}"
        / (
            f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
            f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_ot_pca.json"
        )
    )


def _expected_pca_guided_outputs(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    token_position_id: str,
    basis_source_mode: str,
    site_menu: str,
    num_bands: int,
    layer: int,
) -> list[str]:
    site_catalog_tag = _site_catalog_tag(
        site_menu=str(site_menu),
        num_bands=int(num_bands),
        band_scheme=str(args.pca_band_scheme),
        top_prefix_sizes=normalized["pca_top_prefix_sizes"],
    )
    layer_dir = Path(args.results_root) / f"{stage_timestamp}_mcqa_ot_pca_focus" / f"layer_{int(layer):02d}"
    outputs = [
        str(
            layer_dir
            / (
                f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_ot_pca.json"
            )
        )
    ]
    for target_var in normalized["target_vars"]:
        outputs.append(
            str(
                layer_dir
                / (
                    f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                    f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_{str(target_var)}_das_guided.json"
                )
            )
        )
    return outputs


def _expected_native_payload_paths(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    layer: int,
    resolutions: tuple[int, ...] | None = None,
) -> list[str]:
    native_root = Path(args.results_root) / f"{stage_timestamp}_mcqa_ot_das_block_focus"
    outputs = [str(native_root / "layer_sweep_manifest.json")]
    for resolution in (resolutions or normalized["native_block_resolutions"]):
        outputs.append(
            str(
                native_root
                / f"layer_{int(layer):02d}"
                / (
                    f"mcqa_layer-{int(layer)}_pos-last_token_res-{int(resolution)}"
                    f"_sig-{str(args.signature_mode)}_ot_das_block.json"
                )
            )
        )
    return outputs


def _stage_a_rankings_paths(*, sweep_root: Path, token_position_id: str) -> tuple[Path, Path]:
    return (
        sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.json",
        sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.txt",
    )


def run_stage_a(args: argparse.Namespace) -> None:
    _require_hf_token(args)
    normalized = _normalize_args(args)
    repo_root = _repo_root()
    sweep_root = _sweep_root(args, normalized)
    sweep_root.mkdir(parents=True, exist_ok=True)

    layer_indices = normalized["stage_a_layer_indices"]
    if not layer_indices:
        layer_indices = _all_layer_indices(str(args.model_name))

    stage_statuses: dict[str, dict[str, object]] = {}
    for token_position_id in normalized["stage_a_token_position_ids"]:
        stage_timestamp = f"{str(normalized['results_timestamp'])}_stageA_{str(token_position_id)}"
        run_root = Path(args.results_root) / f"{stage_timestamp}_mcqa_layer_sweep"
        stage_output = run_root / "layer_sweep_manifest.json"
        ranking_json_path, ranking_txt_path = _stage_a_rankings_paths(
            sweep_root=sweep_root,
            token_position_id=str(token_position_id),
        )

        if not _stage_output_is_valid(stage_output):
            command = _build_stage_a_command(
                args=args,
                normalized=normalized,
                stage_timestamp=stage_timestamp,
                token_position_id=str(token_position_id),
                layer_indices=tuple(int(layer) for layer in layer_indices),
            )
            print(f"[parallel-stage-a] running token_position={token_position_id} timestamp={stage_timestamp}")
            stage_start = perf_counter()
            subprocess.run(command, cwd=repo_root, check=True)
            stage_runtime_seconds = float(perf_counter() - stage_start)
        else:
            print(f"[parallel-stage-a] using existing output {stage_output}")
            stage_runtime_seconds = None

        rankings = _extract_stage_a_rankings(aggregate_path=stage_output)
        _write_json(ranking_json_path, rankings)
        _write_text(
            ranking_txt_path,
            _format_stage_a_summary(token_position_id=str(token_position_id), rankings=rankings),
        )
        stage_statuses[f"stage_a_{str(token_position_id)}"] = {
            "state": "completed",
            "stage_timestamp": stage_timestamp,
            "expected_outputs": [str(stage_output), str(ranking_json_path), str(ranking_txt_path)],
            "completed_at": datetime.now().isoformat(),
            "runtime_seconds": stage_runtime_seconds,
        }

    _write_json(
        sweep_root / "parallel_manifest.json",
        {
            "kind": "mcqa_delta_hierarchical_parallel",
            "results_timestamp": str(normalized["results_timestamp"]),
            "results_root": str(Path(args.results_root)),
            "stage_statuses": stage_statuses,
            "updated_at": datetime.now().isoformat(),
        },
    )


def _load_stage_a_rankings_by_token(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    sweep_root: Path,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    rankings_by_token: dict[str, dict[str, list[dict[str, object]]]] = {}
    for token_position_id in normalized["stage_a_token_position_ids"]:
        ranking_json_path, _ = _stage_a_rankings_paths(sweep_root=sweep_root, token_position_id=str(token_position_id))
        rankings_by_token[str(token_position_id)] = _read_rankings(ranking_json_path)
    return rankings_by_token


def plan_stage_b_tasks(args: argparse.Namespace) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    rankings_by_token = _load_stage_a_rankings_by_token(args=args, normalized=normalized, sweep_root=sweep_root)

    native_tasks: list[dict[str, object]] = []
    pca_tasks: list[dict[str, object]] = []
    for token_position_id, rankings in rankings_by_token.items():
        selected_layers = _select_stage_b_layers(
            rankings=rankings,
            top_layers_per_var=int(args.stage_b_top_layers_per_var),
            neighbor_radius=int(args.stage_b_neighbor_radius),
            max_layers_per_var=int(args.stage_b_max_layers_per_var),
        )
        if not selected_layers:
            continue

        if str(token_position_id) == "last_token":
            for layer in selected_layers:
                for resolution in normalized["native_block_resolutions"]:
                    stage_timestamp = (
                        f"{str(normalized['results_timestamp'])}_stageB_native_{token_position_id}"
                        f"_L{int(layer):02d}_res{int(resolution)}"
                    )
                    expected_outputs = _expected_native_payload_paths(
                        args=args,
                        normalized=normalized,
                        stage_timestamp=stage_timestamp,
                        layer=int(layer),
                        resolutions=(int(resolution),),
                    )
                    task_normalized = dict(normalized)
                    task_normalized["native_block_resolutions"] = (int(resolution),)
                    native_tasks.append(
                        {
                            "task_id": f"native_{token_position_id}_L{int(layer):02d}_res{int(resolution)}",
                            "category": "stage_b_native_ot",
                            "token_position_id": str(token_position_id),
                            "layer": int(layer),
                            "resolution": int(resolution),
                            "stage_timestamp": stage_timestamp,
                            "command": list(
                                _build_native_block_command(
                                    args=args,
                                    normalized=task_normalized,
                                    stage_timestamp=stage_timestamp,
                                    layers=(int(layer),),
                                )
                            ),
                            "expected_outputs": expected_outputs,
                            "payload_paths": expected_outputs[1:],
                        }
                    )

        for basis_source_mode in normalized["pca_basis_source_modes"]:
            for site_menu in normalized["pca_site_menus"]:
                for num_bands in _normalize_num_bands_values(normalized["pca_num_bands_values"], str(site_menu)):
                    base_slug = _stage_b_slug(
                        token_position_id=str(token_position_id),
                        basis_source_mode=str(basis_source_mode),
                        site_menu=str(site_menu),
                        num_bands=int(num_bands),
                    )
                    for layer in selected_layers:
                        stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_{base_slug}_L{int(layer):02d}"
                        payload_path = _expected_pca_payload_path(
                            args=args,
                            normalized=normalized,
                            stage_timestamp=stage_timestamp,
                            token_position_id=str(token_position_id),
                            basis_source_mode=str(basis_source_mode),
                            site_menu=str(site_menu),
                            num_bands=int(num_bands),
                            layer=int(layer),
                        )
                        manifest_path = Path(args.results_root) / f"{stage_timestamp}_mcqa_ot_pca_focus" / "layer_sweep_manifest.json"
                        pca_tasks.append(
                            {
                                "task_id": f"pca_{base_slug}_L{int(layer):02d}",
                                "category": "stage_b_pca_ot",
                                "token_position_id": str(token_position_id),
                                "basis_source_mode": str(basis_source_mode),
                                "site_menu": str(site_menu),
                                "num_bands": int(num_bands),
                                "layer": int(layer),
                                "stage_timestamp": stage_timestamp,
                                "command": list(
                                    _build_stage_b_or_c_command(
                                        args=args,
                                        normalized=normalized,
                                        stage_timestamp=stage_timestamp,
                                        token_position_id=str(token_position_id),
                                        layers=(int(layer),),
                                        basis_source_mode=str(basis_source_mode),
                                        site_menu=str(site_menu),
                                        num_bands=int(num_bands),
                                        guided_das=False,
                                    )
                                ),
                                "expected_outputs": [str(manifest_path), str(payload_path)],
                                "payload_paths": [str(payload_path)],
                            }
                        )

    _write_task_manifest(sweep_root / "stage_b_native_tasks.json", kind="stage_b_native_tasks", tasks=native_tasks)
    _write_task_manifest(sweep_root / "stage_b_pca_tasks.json", kind="stage_b_pca_tasks", tasks=pca_tasks)
    _write_text(
        sweep_root / "stage_b_task_plan.txt",
        "\n".join(
            [
                "MCQA hierarchical parallel Stage B task plan",
                f"native_tasks: {len(native_tasks)}",
                f"pca_tasks: {len(pca_tasks)}",
                "",
                *[f"native {task['task_id']} -> {task['stage_timestamp']}" for task in native_tasks],
                "",
                *[f"pca {task['task_id']} -> {task['stage_timestamp']}" for task in pca_tasks],
            ]
        ),
    )
    print(f"[parallel-plan-stage-b] wrote {len(native_tasks)} native tasks and {len(pca_tasks)} PCA tasks to {sweep_root}")


def _valid_payload_paths_from_tasks(tasks: Iterable[dict[str, object]], *, strict: bool) -> list[Path]:
    payload_paths: list[Path] = []
    missing: list[str] = []
    for task in tasks:
        expected_outputs = [str(path) for path in task.get("expected_outputs", [])]
        task_missing = [path for path in expected_outputs if not _stage_output_is_valid(Path(path))]
        if task_missing:
            missing.extend([f"{task.get('task_id')}: {path}" for path in task_missing])
            continue
        payload_paths.extend(Path(str(path)) for path in task.get("payload_paths", []))
    if strict and missing:
        raise RuntimeError("Missing task outputs:\n" + "\n".join(missing[:50]))
    return payload_paths


def aggregate_stage_b(args: argparse.Namespace, *, strict: bool = True, require_native: bool = True) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    native_manifest_path = sweep_root / "stage_b_native_tasks.json"
    pca_manifest_path = sweep_root / "stage_b_pca_tasks.json"

    native_payload_paths: list[Path] = []
    if _stage_output_is_valid(native_manifest_path) and require_native:
        native_manifest = _load_task_manifest(native_manifest_path)
        native_payload_paths = _valid_payload_paths_from_tasks(native_manifest["tasks"], strict=strict)
    if native_payload_paths:
        native_ot_rankings, native_guided_rankings, a_only_rankings = _extract_native_rankings(payload_paths=native_payload_paths)
        _write_json(sweep_root / "stage_b_native_rankings.json", native_ot_rankings)
        _write_text(
            sweep_root / "stage_b_native_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage B Native OT Ranking", rankings=native_ot_rankings),
        )
        _write_json(sweep_root / "stage_c_native_guided_rankings.json", native_guided_rankings)
        _write_text(
            sweep_root / "stage_c_native_guided_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage C Native Guided DAS Ranking", rankings=native_guided_rankings),
        )
        _write_json(sweep_root / "stage_c_a_only_rankings.json", a_only_rankings)
        _write_text(
            sweep_root / "stage_c_a_only_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage C A-only DAS Ranking", rankings=a_only_rankings),
        )

    if not _stage_output_is_valid(pca_manifest_path):
        if strict:
            raise FileNotFoundError(f"Missing PCA task manifest: {pca_manifest_path}")
        return
    pca_manifest = _load_task_manifest(pca_manifest_path)
    pca_payload_paths = _valid_payload_paths_from_tasks(pca_manifest["tasks"], strict=strict)
    if pca_payload_paths:
        stage_b_rankings = _extract_stage_b_best_configs(payload_paths=pca_payload_paths)
        _write_json(sweep_root / "stage_b_pca_rankings.json", stage_b_rankings)
        _write_text(sweep_root / "stage_b_pca_rankings.txt", _format_stage_b_summary(rankings=stage_b_rankings))
    print(f"[parallel-aggregate-stage-b] aggregated {len(native_payload_paths)} native payloads and {len(pca_payload_paths)} PCA payloads")


def _pca_task_lookup(tasks: Iterable[dict[str, object]]) -> dict[tuple[str, str, str, int, int], dict[str, object]]:
    lookup: dict[tuple[str, str, str, int, int], dict[str, object]] = {}
    for task in tasks:
        key = (
            str(task["token_position_id"]),
            str(task["basis_source_mode"]),
            str(task["site_menu"]),
            int(task["num_bands"]),
            int(task["layer"]),
        )
        lookup[key] = task
    return lookup


def plan_stage_c_tasks(args: argparse.Namespace) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    stage_b_rankings = _read_rankings(sweep_root / "stage_b_pca_rankings.json")
    selected_groups = _select_stage_c_configs(
        rankings=stage_b_rankings,
        top_configs_per_var=int(args.stage_c_top_configs_per_var),
    )
    pca_manifest = _load_task_manifest(sweep_root / "stage_b_pca_tasks.json")
    pca_lookup = _pca_task_lookup(pca_manifest["tasks"])
    stage_c_tasks: list[dict[str, object]] = []

    for (token_position_id, basis_source_mode, site_menu, num_bands), layers in selected_groups.items():
        for layer in layers:
            key = (str(token_position_id), str(basis_source_mode), str(site_menu), int(num_bands), int(layer))
            stage_b_task = pca_lookup.get(key)
            if stage_b_task is None:
                raise RuntimeError(f"Could not find Stage B PCA task for Stage C key={key}")
            stage_timestamp = str(stage_b_task["stage_timestamp"])
            expected_outputs = _expected_pca_guided_outputs(
                args=args,
                normalized=normalized,
                stage_timestamp=stage_timestamp,
                token_position_id=str(token_position_id),
                basis_source_mode=str(basis_source_mode),
                site_menu=str(site_menu),
                num_bands=int(num_bands),
                layer=int(layer),
            )
            stage_c_tasks.append(
                {
                    "task_id": f"stage_c_{stage_b_task['task_id']}",
                    "category": "stage_c_guided_das",
                    "token_position_id": str(token_position_id),
                    "basis_source_mode": str(basis_source_mode),
                    "site_menu": str(site_menu),
                    "num_bands": int(num_bands),
                    "layer": int(layer),
                    "stage_timestamp": stage_timestamp,
                    "command": list(
                        _build_stage_b_or_c_command(
                            args=args,
                            normalized=normalized,
                            stage_timestamp=stage_timestamp,
                            token_position_id=str(token_position_id),
                            layers=(int(layer),),
                            basis_source_mode=str(basis_source_mode),
                            site_menu=str(site_menu),
                            num_bands=int(num_bands),
                            guided_das=True,
                        )
                    ),
                    "expected_outputs": expected_outputs,
                    "payload_paths": [expected_outputs[0]],
                }
            )
    _write_task_manifest(sweep_root / "stage_c_pca_tasks.json", kind="stage_c_pca_tasks", tasks=stage_c_tasks)
    _write_text(
        sweep_root / "stage_c_task_plan.txt",
        "\n".join(
            [
                "MCQA hierarchical parallel Stage C task plan",
                f"stage_c_tasks: {len(stage_c_tasks)}",
                "",
                *[f"{task['task_id']} -> {task['stage_timestamp']}" for task in stage_c_tasks],
            ]
        ),
    )
    print(f"[parallel-plan-stage-c] wrote {len(stage_c_tasks)} Stage C tasks to {sweep_root}")


def aggregate_stage_c(args: argparse.Namespace, *, strict: bool = True) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    task_manifest_path = sweep_root / "stage_c_pca_tasks.json"
    if not _stage_output_is_valid(task_manifest_path):
        if strict:
            raise FileNotFoundError(f"Missing Stage C task manifest: {task_manifest_path}")
        return
    task_manifest = _load_task_manifest(task_manifest_path)
    stage_c_payload_paths = _valid_payload_paths_from_tasks(task_manifest["tasks"], strict=strict)
    stage_c_rankings = _extract_stage_c_rankings(payload_paths=stage_c_payload_paths)
    _write_json(sweep_root / "stage_c_guided_rankings.json", stage_c_rankings)
    _write_text(sweep_root / "stage_c_guided_rankings.txt", _format_stage_c_summary(rankings=stage_c_rankings))
    print(f"[parallel-aggregate-stage-c] aggregated {len(stage_c_payload_paths)} Stage C payloads")


def aggregate_final(args: argparse.Namespace, *, strict: bool = True) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    aggregate_stage_b(args, strict=strict, require_native=True)
    aggregate_stage_c(args, strict=strict)
    task_statuses = _task_statuses(sweep_root=sweep_root)
    completed_runtimes = [
        float(status["runtime_seconds"])
        for status in task_statuses
        if status.get("state") == "completed" and status.get("runtime_seconds") is not None
    ]
    lines = [
        "MCQA Delta Hierarchical Parallel Sweep",
        f"results_timestamp: {normalized['results_timestamp']}",
        f"results_root: {Path(args.results_root)}",
        f"task_statuses: {len(task_statuses)}",
        f"completed_task_runtime_sum_seconds: {sum(completed_runtimes):.2f}",
        f"completed_task_runtime_max_seconds: {(max(completed_runtimes) if completed_runtimes else 0.0):.2f}",
        "",
    ]
    for filename in (
        "stage_a_last_token_layer_rankings.txt",
        "stage_b_native_rankings.txt",
        "stage_c_native_guided_rankings.txt",
        "stage_c_a_only_rankings.txt",
        "stage_b_pca_rankings.txt",
        "stage_c_guided_rankings.txt",
    ):
        path = sweep_root / filename
        lines.append(f"{filename}: {'present' if path.exists() else 'missing'}")
    if task_statuses:
        lines.extend(["", "task runtimes:"])
        for status in task_statuses:
            runtime = status.get("runtime_seconds")
            runtime_text = "n/a" if runtime is None else f"{float(runtime):.2f}s"
            lines.append(f"{status.get('task_id')}: {status.get('state')} runtime={runtime_text}")
    _write_json(sweep_root / "parallel_task_statuses.json", task_statuses)
    _write_text(sweep_root / "hierarchical_parallel_summary.txt", "\n".join(lines))
    print(f"[parallel-aggregate-final] wrote {sweep_root / 'hierarchical_parallel_summary.txt'}")


def run_task(task_file: Path, task_index: int) -> None:
    manifest = _load_task_manifest(task_file)
    tasks = manifest["tasks"]
    if not isinstance(tasks, list):
        raise ValueError(f"Malformed task list in {task_file}")
    if int(task_index) < 0 or int(task_index) >= len(tasks):
        raise IndexError(f"task_index={task_index} out of range for {len(tasks)} tasks in {task_file}")
    task = dict(tasks[int(task_index)])
    sweep_root = task_file.parent
    command = [str(item) for item in task.get("command", [])]
    if not command:
        raise ValueError(f"Task {task.get('task_id')} has empty command")

    expected_outputs = [Path(str(path)) for path in task.get("expected_outputs", [])]
    if expected_outputs and all(_stage_output_is_valid(path) for path in expected_outputs):
        _mark_task(sweep_root=sweep_root, task=task, state="skipped_existing", extra={"completed_at": datetime.now().isoformat()})
        print(f"[parallel-task] skipped_existing task_id={task.get('task_id')}")
        return

    _mark_task(sweep_root=sweep_root, task=task, state="running", extra={"started_at": datetime.now().isoformat()})
    print(f"[parallel-task] running task_id={task.get('task_id')} command={' '.join(command)}")
    task_start = perf_counter()
    try:
        subprocess.run(command, cwd=_repo_root(), check=True)
    except Exception as exc:
        _mark_task(
            sweep_root=sweep_root,
            task=task,
            state="failed",
            extra={"failed_at": datetime.now().isoformat(), "error": repr(exc)},
        )
        raise
    task_runtime_seconds = float(perf_counter() - task_start)
    missing_outputs = [str(path) for path in expected_outputs if not _stage_output_is_valid(path)]
    if missing_outputs:
        _mark_task(
            sweep_root=sweep_root,
            task=task,
            state="failed_missing_outputs",
            extra={"failed_at": datetime.now().isoformat(), "missing_outputs": missing_outputs},
        )
        raise RuntimeError(f"Task {task.get('task_id')} missing outputs: {missing_outputs}")
    _mark_task(
        sweep_root=sweep_root,
        task=task,
        state="completed",
        extra={
            "completed_at": datetime.now().isoformat(),
            "runtime_seconds": float(task_runtime_seconds),
            "wall_runtime_seconds": float(task_runtime_seconds),
        },
    )
    print(f"[parallel-task] completed task_id={task.get('task_id')}")


def _parse_command() -> tuple[str, list[str]]:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: mcqa_delta_hierarchical_parallel.py "
            "{stage-a,plan-stage-b,aggregate-stage-b,plan-stage-c,aggregate-stage-c,aggregate-final,run-task} ..."
        )
    return sys.argv[1], sys.argv[2:]


def _parse_run_task_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MCQA hierarchical parallel task from a task manifest.")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-index", type=int, default=None)
    parsed = parser.parse_args(argv)
    if parsed.task_index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            raise ValueError("--task-index is required when SLURM_ARRAY_TASK_ID is unset")
        parsed.task_index = int(env_index)
    return parsed


def main() -> None:
    command, argv = _parse_command()
    if command == "run-task":
        parsed = _parse_run_task_args(argv)
        run_task(task_file=Path(parsed.task_file), task_index=int(parsed.task_index))
        return

    parser = _build_parser()
    parser.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-native-aggregation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if command == "stage-a":
        run_stage_a(args)
    elif command == "plan-stage-b":
        plan_stage_b_tasks(args)
    elif command == "aggregate-stage-b":
        aggregate_stage_b(
            args,
            strict=not bool(args.allow_partial),
            require_native=not bool(args.skip_native_aggregation),
        )
    elif command == "plan-stage-c":
        plan_stage_c_tasks(args)
    elif command == "aggregate-stage-c":
        aggregate_stage_c(args, strict=not bool(args.allow_partial))
    elif command == "aggregate-final":
        aggregate_final(args, strict=not bool(args.allow_partial))
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
