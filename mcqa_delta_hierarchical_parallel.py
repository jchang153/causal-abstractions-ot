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
    _build_stage_c_a_only_command,
    _extract_a_only_rankings,
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
    _select_stage_c_configs,
    _site_catalog_tag,
    _stage_a_output_path,
    _stage_b_slug,
    _select_stage_b_layers_for_spec,
    _stage_output_is_valid,
    _stage_b_selection_specs,
    _write_json,
    _write_text,
)
from mcqa_paper_runtime import write_paper_runtime_summary


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
        "layer_selection_method": task.get("layer_selection_method", "custom"),
        "transport_method": task.get("transport_method"),
        "token_position_id": task.get("token_position_id"),
        "layer": task.get("layer"),
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


def _expected_native_guided_outputs(
    *,
    args: argparse.Namespace,
    stage_timestamp: str,
    layer: int,
    resolution: int,
    target_vars: tuple[str, ...],
) -> list[str]:
    native_root = Path(args.results_root) / f"{stage_timestamp}_mcqa_ot_das_block_focus"
    layer_dir = native_root / f"layer_{int(layer):02d}"
    resolution_tag = str(int(resolution))
    outputs = [
        str(
            layer_dir
            / (
                f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}"
                f"_sig-{str(args.signature_mode)}_ot_das_block.json"
            )
        )
    ]
    for target_var in target_vars:
        outputs.append(
            str(
                layer_dir
                / (
                    f"mcqa_layer-{int(layer)}_pos-last_token_res-{resolution_tag}"
                    f"_sig-{str(args.signature_mode)}_{str(target_var)}_das_block.json"
                )
            )
        )
    return outputs


def _stage_a_rankings_paths(*, sweep_root: Path, token_position_id: str) -> tuple[Path, Path]:
    return (
        sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.json",
        sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.txt",
    )


def _stage_a_hparam_grid_size(normalized: dict[str, object]) -> int:
    epsilon_count = len(tuple(normalized.get("ot_epsilons", ()) or ()))
    beta_count = len(tuple(normalized.get("uot_beta_neurals", ()) or ()))
    methods = tuple(normalized.get("stage_a_methods", ()) or ())
    if not methods:
        methods = tuple(
            method.strip()
            for method in str(normalized.get("stage_a_method", "ot")).split(",")
            if method.strip()
        )
    if not methods:
        methods = ("ot",)
    grid_size = 0
    if "ot" in methods:
        grid_size += max(1, epsilon_count)
    if "uot" in methods:
        grid_size += max(1, epsilon_count) * max(1, beta_count)
    return max(1, grid_size)


def _stage_a_runtime_accounting(
    *,
    normalized: dict[str, object],
    stage_runtime_seconds: float | None,
) -> dict[str, object]:
    grid_size = _stage_a_hparam_grid_size(normalized)
    selection = str(normalized.get("stage_a_hparam_selection", "rowwise"))
    if selection == "rowwise":
        policy = "rowwise_calibrated_grid_included"
        paper_runtime_seconds = stage_runtime_seconds
    elif grid_size == 1:
        policy = "joint_single_plan"
        paper_runtime_seconds = stage_runtime_seconds
    else:
        policy = "joint_external_sweep_not_counted"
        paper_runtime_seconds = None
    return {
        "stage_a_method": str(normalized.get("stage_a_method", "ot")),
        "stage_a_hparam_selection": selection,
        "stage_a_hparam_grid_size": int(grid_size),
        "stage_a_runtime_policy": policy,
        "diagnostic_runtime_seconds": stage_runtime_seconds,
        "paper_runtime_seconds": paper_runtime_seconds,
    }


def _stage_a_rerank_output_path(*, results_root: str | Path, stage_timestamp: str) -> Path:
    return Path(results_root) / f"{stage_timestamp}_mcqa_layer_sweep" / "layer_sweep_manifest.json"


def _layer_sweep_manifest_is_complete(path: Path, layer_indices: Iterable[int]) -> bool:
    if not _stage_output_is_valid(path):
        return False
    try:
        payload = _load_json(path)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return False
    required = {int(layer) for layer in layer_indices}
    observed: set[int] = set()
    for record in runs:
        if not isinstance(record, dict):
            continue
        try:
            layer = int(record.get("layer"))
        except (TypeError, ValueError):
            continue
        output_path = record.get("output_path")
        if output_path is None:
            continue
        if _stage_output_is_valid(Path(str(output_path))):
            observed.add(layer)
    return required.issubset(observed)


def _stage_a_candidate_layers_by_var(
    rankings: dict[str, list[dict[str, object]]],
    *,
    top_k: int,
) -> dict[str, tuple[int, ...]]:
    candidates: dict[str, tuple[int, ...]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        layers: list[int] = []
        for entry in rankings.get(target_var, [])[: max(1, int(top_k))]:
            if entry.get("layer") is not None:
                layers.append(int(entry["layer"]))
        candidates[target_var] = tuple(dict.fromkeys(layers))
    return candidates


def _stage_a_entry_mass(entry: dict[str, object]) -> float:
    return float(entry.get("layer_score", entry.get("target_mass", entry.get("selection_score", 0.0))))


def _stage_a_adaptive_candidate_count(
    entries: list[dict[str, object]],
    *,
    min_k: int,
    max_k: int,
    drop_ratio: float,
) -> int:
    if not entries:
        return 0
    min_count = min(len(entries), max(1, int(min_k)))
    max_count = min(len(entries), max(min_count, int(max_k)))
    ratio = float(drop_ratio)
    if ratio <= 0.0:
        return max_count

    for index in range(min_count - 1, max_count - 1):
        current_mass = _stage_a_entry_mass(entries[index])
        next_mass = _stage_a_entry_mass(entries[index + 1])
        if current_mass > 0.0 and next_mass / current_mass < ratio:
            return index + 1
    return max_count


def _stage_a_adaptive_candidate_layers_by_var(
    rankings: dict[str, list[dict[str, object]]],
    *,
    min_k: int,
    max_k: int,
    drop_ratio: float,
) -> dict[str, tuple[int, ...]]:
    candidates: dict[str, tuple[int, ...]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        entries = [entry for entry in rankings.get(target_var, []) if entry.get("layer") is not None]
        count = _stage_a_adaptive_candidate_count(
            entries,
            min_k=int(min_k),
            max_k=int(max_k),
            drop_ratio=float(drop_ratio),
        )
        candidates[target_var] = tuple(dict.fromkeys(int(entry["layer"]) for entry in entries[:count]))
    return candidates


def _stage_a_rerank_policy(normalized: dict[str, object]) -> dict[str, object]:
    fixed_top_k = int(normalized.get("stage_a_rerank_top_k", 0) or 0)
    if fixed_top_k > 0:
        return {
            "enabled": True,
            "mode": "top_k",
            "label": f"top{fixed_top_k}",
            "top_k": fixed_top_k,
            "min_k": fixed_top_k,
            "max_k": fixed_top_k,
            "drop_ratio": 0.0,
        }

    drop_ratio = float(normalized.get("stage_a_rerank_drop_ratio", 0.0) or 0.0)
    if drop_ratio <= 0.0:
        return {"enabled": False, "mode": "disabled", "label": "none"}

    min_k = max(1, int(normalized.get("stage_a_rerank_min_k", 6) or 6))
    max_k = max(min_k, int(normalized.get("stage_a_rerank_max_k", 8) or 8))
    ratio_label = f"{drop_ratio:g}".replace(".", "p")
    return {
        "enabled": True,
        "mode": "adaptive_drop",
        "label": f"drop{ratio_label}_min{min_k}_max{max_k}",
        "top_k": 0,
        "min_k": min_k,
        "max_k": max_k,
        "drop_ratio": drop_ratio,
    }


def _build_stage_a_rerank_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    token_position_id: str,
    layer_indices: tuple[int, ...],
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "mcqa_run_cloud.py",
        "--preset",
        "full",
        "--layer-sweep",
        "--device",
        str(args.device),
        "--model-name",
        str(args.model_name),
        "--dataset-path",
        str(args.dataset_path),
        "--dataset-size",
        str(int(args.dataset_size)),
        "--split-seed",
        str(int(args.split_seed)),
        "--train-pool-size",
        str(int(args.train_pool_size)),
        "--calibration-pool-size",
        str(int(args.calibration_pool_size)),
        "--test-pool-size",
        str(int(args.test_pool_size)),
        "--batch-size",
        str(int(args.batch_size)),
        "--methods",
        "bruteforce",
        "--target-vars",
        ",".join(str(target_var) for target_var in normalized["target_vars"]),
        "--layer-indices",
        ",".join(str(layer) for layer in layer_indices),
        "--token-position-ids",
        str(token_position_id),
        "--resolutions",
        "full",
        "--ot-epsilons",
        "1",
        "--ot-lambdas",
        ",".join(f"{float(value):g}" for value in normalized["ot_lambdas"]),
        "--signature-modes",
        str(args.signature_mode),
        "--calibration-metric",
        str(args.calibration_metric),
        "--calibration-family-weights",
        ",".join(f"{float(value):g}" for value in normalized["calibration_family_weights"]),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    if args.dataset_config:
        command.extend(["--dataset-config", str(args.dataset_config)])
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _merge_stage_a_rerank_rankings(
    *,
    proposal_rankings: dict[str, list[dict[str, object]]],
    calibration_rankings: dict[str, list[dict[str, object]]],
    candidates_by_var: dict[str, tuple[int, ...]],
    selection_label: str,
) -> dict[str, list[dict[str, object]]]:
    merged: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for target_var in DEFAULT_TARGET_VARS:
        proposal_entries = proposal_rankings.get(target_var, [])
        proposal_by_layer = {
            int(entry["layer"]): (rank, entry)
            for rank, entry in enumerate(proposal_entries, start=1)
            if entry.get("layer") is not None
        }
        candidate_layers = {int(layer) for layer in candidates_by_var.get(target_var, ())}
        rows: list[dict[str, object]] = []
        for entry in calibration_rankings.get(target_var, []):
            if entry.get("layer") is None:
                continue
            layer = int(entry["layer"])
            if layer not in candidate_layers:
                continue
            proposal_rank, proposal = proposal_by_layer.get(layer, (None, {}))
            row = dict(entry)
            row["selection_basis"] = f"{selection_label}_transport_candidates_full_layer_calibration"
            row["proposal_rank"] = proposal_rank
            row["proposal_layer_score"] = proposal.get("layer_score", proposal.get("target_mass", proposal.get("selection_score")))
            row["proposal_method"] = proposal.get("method")
            row["proposal_epsilon"] = proposal.get("epsilon")
            row["proposal_uot_beta_neural"] = proposal.get("uot_beta_neural")
            row["proposal_selected_mass"] = proposal.get("selected_mass")
            rows.append(row)
        rows.sort(
            key=lambda row: (
                float(row.get("selection_score", -1.0)),
                float(row.get("exact_acc", -1.0)),
                -float(row.get("proposal_rank") or 10**9),
            ),
            reverse=True,
        )
        merged[target_var] = rows
    return merged


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
        stage_output = _stage_a_output_path(results_root=args.results_root, stage_timestamp=stage_timestamp)
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

        rankings = _extract_stage_a_rankings(
            aggregate_path=stage_output,
            hparam_selection=str(normalized.get("stage_a_hparam_selection", "rowwise")),
        )
        rerank_runtime_seconds = None
        rerank_policy = _stage_a_rerank_policy(normalized)
        rerank_top_k = int(rerank_policy.get("top_k", 0) or 0)
        candidates_by_var: dict[str, tuple[int, ...]] = {}
        if bool(rerank_policy.get("enabled", False)):
            proposal_rankings = rankings
            proposal_json_path = ranking_json_path.with_name(
                f"{ranking_json_path.stem}_transport_proposal{ranking_json_path.suffix}"
            )
            proposal_txt_path = ranking_txt_path.with_name(
                f"{ranking_txt_path.stem}_transport_proposal{ranking_txt_path.suffix}"
            )
            _write_json(proposal_json_path, proposal_rankings)
            _write_text(
                proposal_txt_path,
                _format_stage_a_summary(token_position_id=str(token_position_id), rankings=proposal_rankings),
            )
            if str(rerank_policy.get("mode")) == "top_k":
                candidates_by_var = _stage_a_candidate_layers_by_var(proposal_rankings, top_k=rerank_top_k)
            else:
                candidates_by_var = _stage_a_adaptive_candidate_layers_by_var(
                    proposal_rankings,
                    min_k=int(rerank_policy.get("min_k", 6)),
                    max_k=int(rerank_policy.get("max_k", 8)),
                    drop_ratio=float(rerank_policy.get("drop_ratio", 0.0)),
                )
            rerank_layers = tuple(
                sorted(
                    {
                        int(layer)
                        for layers in candidates_by_var.values()
                        for layer in layers
                    }
                )
            )
            if rerank_layers:
                rerank_label = str(rerank_policy.get("label", "rerank"))
                rerank_timestamp = (
                    f"{str(normalized['results_timestamp'])}_stageA_{str(token_position_id)}"
                    f"_{rerank_label}_full_layer_rerank"
                )
                rerank_output = _stage_a_rerank_output_path(
                    results_root=args.results_root,
                    stage_timestamp=rerank_timestamp,
                )
                if not _layer_sweep_manifest_is_complete(rerank_output, rerank_layers):
                    rerank_command = _build_stage_a_rerank_command(
                        args=args,
                        normalized=normalized,
                        stage_timestamp=rerank_timestamp,
                        token_position_id=str(token_position_id),
                        layer_indices=rerank_layers,
                    )
                    print(
                        "[parallel-stage-a-rerank] running "
                        f"policy={rerank_policy} candidates_by_var={candidates_by_var} "
                        f"layers={list(rerank_layers)} "
                        f"command={' '.join(rerank_command)}"
                    )
                    rerank_start = perf_counter()
                    subprocess.run(rerank_command, cwd=repo_root, check=True)
                    rerank_runtime_seconds = float(perf_counter() - rerank_start)
                else:
                    print(f"[parallel-stage-a-rerank] using existing output {rerank_output}")
                calibration_rankings = _extract_stage_a_rankings(aggregate_path=rerank_output)
                rankings = _merge_stage_a_rerank_rankings(
                    proposal_rankings=proposal_rankings,
                    calibration_rankings=calibration_rankings,
                    candidates_by_var=candidates_by_var,
                    selection_label=str(rerank_policy.get("label", "rerank")),
                )
                if stage_runtime_seconds is not None and rerank_runtime_seconds is not None:
                    stage_runtime_seconds = float(stage_runtime_seconds) + float(rerank_runtime_seconds)
                elif rerank_runtime_seconds is not None:
                    stage_runtime_seconds = float(rerank_runtime_seconds)
        _write_json(ranking_json_path, rankings)
        _write_text(
            ranking_txt_path,
            _format_stage_a_summary(token_position_id=str(token_position_id), rankings=rankings),
        )
        runtime_accounting = _stage_a_runtime_accounting(
            normalized=normalized,
            stage_runtime_seconds=stage_runtime_seconds,
        )
        stage_statuses[f"stage_a_{str(token_position_id)}"] = {
            "state": "completed",
            "stage_timestamp": stage_timestamp,
            "expected_outputs": [str(stage_output), str(ranking_json_path), str(ranking_txt_path)],
            "completed_at": datetime.now().isoformat(),
            "runtime_seconds": stage_runtime_seconds,
            "stage_a_rerank_top_k": int(rerank_top_k),
            "stage_a_rerank_policy": rerank_policy,
            "stage_a_rerank_candidate_layers_by_var": {
                str(target_var): list(layers)
                for target_var, layers in candidates_by_var.items()
            },
            "stage_a_rerank_runtime_seconds": rerank_runtime_seconds,
            **runtime_accounting,
        }

    runtime_lines = [
        "MCQA Stage A runtime summary",
        f"results_timestamp: {normalized['results_timestamp']}",
        f"stage_a_method: {normalized.get('stage_a_method')}",
        f"stage_a_hparam_selection: {normalized.get('stage_a_hparam_selection')}",
        f"stage_a_rerank_top_k: {normalized.get('stage_a_rerank_top_k', 0)}",
        f"stage_a_rerank_drop_ratio: {normalized.get('stage_a_rerank_drop_ratio', 0.0)}",
        f"stage_a_rerank_min_k: {normalized.get('stage_a_rerank_min_k', 6)}",
        f"stage_a_rerank_max_k: {normalized.get('stage_a_rerank_max_k', 8)}",
        f"stage_a_hparam_grid_size: {_stage_a_hparam_grid_size(normalized)}",
        "",
    ]
    for key, status in stage_statuses.items():
        diagnostic = status.get("diagnostic_runtime_seconds")
        paper = status.get("paper_runtime_seconds")
        runtime_lines.append(
            f"{key}: policy={status.get('stage_a_runtime_policy')} "
            f"diagnostic_runtime_s={'n/a' if diagnostic is None else f'{float(diagnostic):.2f}'} "
            f"paper_runtime_s={'n/a' if paper is None else f'{float(paper):.2f}'}"
        )
    _write_json(sweep_root / "stage_a_runtime_summary.json", stage_statuses)
    _write_text(sweep_root / "stage_a_runtime_summary.txt", "\n".join(runtime_lines) + "\n")

    _write_json(
        sweep_root / "parallel_manifest.json",
        {
            "kind": "mcqa_delta_hierarchical_parallel",
            "results_timestamp": str(normalized["results_timestamp"]),
            "results_root": str(Path(args.results_root)),
            "stage_a_method": str(normalized.get("stage_a_method", "ot")),
            "stage_a_hparam_selection": str(normalized.get("stage_a_hparam_selection", "rowwise")),
            "stage_a_hparam_grid_size": _stage_a_hparam_grid_size(normalized),
            "stage_a_rerank_top_k": int(normalized.get("stage_a_rerank_top_k", 0) or 0),
            "stage_a_rerank_drop_ratio": float(normalized.get("stage_a_rerank_drop_ratio", 0.0) or 0.0),
            "stage_a_rerank_min_k": int(normalized.get("stage_a_rerank_min_k", 6) or 6),
            "stage_a_rerank_max_k": int(normalized.get("stage_a_rerank_max_k", 8) or 8),
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
    selection_specs = _stage_b_selection_specs(args=args, normalized=normalized)
    for token_position_id, rankings in rankings_by_token.items():
        for selection_spec in selection_specs:
            selected_layers = _select_stage_b_layers_for_spec(
                rankings=rankings,
                normalized=normalized,
                selection_spec=selection_spec,
            )
            if not selected_layers:
                continue

            for transport_method in normalized["stage_b_methods"]:
                if str(token_position_id) == "last_token":
                    for layer in selected_layers:
                        for resolution in normalized["native_block_resolutions"]:
                            stage_timestamp = (
                                f"{str(normalized['results_timestamp'])}_stageB_{selection_spec.name}_native_"
                                f"{transport_method}_{token_position_id}_L{int(layer):02d}_res{int(resolution)}"
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
                                    "task_id": (
                                        f"native_{selection_spec.name}_{transport_method}_{token_position_id}"
                                        f"_L{int(layer):02d}_res{int(resolution)}"
                                    ),
                                    "category": f"stage_b_native_{selection_spec.name}_{transport_method}",
                                    "layer_selection_method": selection_spec.name,
                                    "layer_selection_top_layers_per_var": int(selection_spec.top_layers_per_var),
                                    "layer_selection_neighbor_radius": int(selection_spec.neighbor_radius),
                                    "layer_selection_max_layers_per_var": int(selection_spec.max_layers_per_var),
                                    "selected_layers": [int(selected_layer) for selected_layer in selected_layers],
                                    "transport_method": str(transport_method),
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
                                            transport_method=str(transport_method),
                                            skip_full_das=True,
                                            skip_guided_das=True,
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
                                stage_timestamp = (
                                    f"{str(normalized['results_timestamp'])}_stageB_{selection_spec.name}_"
                                    f"{transport_method}_{base_slug}_L{int(layer):02d}"
                                )
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
                                        "task_id": f"pca_{selection_spec.name}_{transport_method}_{base_slug}_L{int(layer):02d}",
                                        "category": f"stage_b_pca_{selection_spec.name}_{transport_method}",
                                        "layer_selection_method": selection_spec.name,
                                        "layer_selection_top_layers_per_var": int(selection_spec.top_layers_per_var),
                                        "layer_selection_neighbor_radius": int(selection_spec.neighbor_radius),
                                        "layer_selection_max_layers_per_var": int(selection_spec.max_layers_per_var),
                                        "selected_layers": [int(selected_layer) for selected_layer in selected_layers],
                                        "transport_method": str(transport_method),
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
                                                transport_method=str(transport_method),
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
                f"stage_b_methods: {','.join(str(method) for method in normalized['stage_b_methods'])}",
                "stage_b_selection_methods: "
                + ",".join(f"{spec.name}(top={spec.top_layers_per_var},r={spec.neighbor_radius},max={spec.max_layers_per_var})" for spec in selection_specs),
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


def _task_metadata_by_payload_path(tasks: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    metadata_by_path: dict[str, dict[str, object]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        metadata = {
            "task_id": task.get("task_id"),
            "category": task.get("category"),
            "stage_timestamp": task.get("stage_timestamp"),
            "layer_selection_method": task.get("layer_selection_method", "custom"),
            "layer_selection_top_layers_per_var": task.get("layer_selection_top_layers_per_var"),
            "layer_selection_neighbor_radius": task.get("layer_selection_neighbor_radius"),
            "layer_selection_max_layers_per_var": task.get("layer_selection_max_layers_per_var"),
        }
        for payload_path in task.get("payload_paths", []):
            path = Path(str(payload_path))
            metadata_by_path[str(path)] = dict(metadata)
            try:
                metadata_by_path[str(path.resolve())] = dict(metadata)
            except OSError:
                pass
    return metadata_by_path


def aggregate_stage_b(args: argparse.Namespace, *, strict: bool = True, require_native: bool = True) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    native_manifest_path = sweep_root / "stage_b_native_tasks.json"
    pca_manifest_path = sweep_root / "stage_b_pca_tasks.json"

    native_payload_paths: list[Path] = []
    native_metadata_by_path: dict[str, dict[str, object]] = {}
    if _stage_output_is_valid(native_manifest_path) and require_native:
        native_manifest = _load_task_manifest(native_manifest_path)
        native_payload_paths = _valid_payload_paths_from_tasks(native_manifest["tasks"], strict=strict)
        native_metadata_by_path = _task_metadata_by_payload_path(native_manifest["tasks"])
    if native_payload_paths:
        native_ot_rankings, native_guided_rankings, a_only_rankings = _extract_native_rankings(
            payload_paths=native_payload_paths,
            metadata_by_payload_path=native_metadata_by_path,
        )
        _write_json(sweep_root / "stage_b_native_rankings.json", native_ot_rankings)
        _write_text(
            sweep_root / "stage_b_native_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage B Native Transport Ranking", rankings=native_ot_rankings),
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
        stage_b_rankings = _extract_stage_b_best_configs(
            payload_paths=pca_payload_paths,
            metadata_by_payload_path=_task_metadata_by_payload_path(pca_manifest["tasks"]),
        )
        _write_json(sweep_root / "stage_b_pca_rankings.json", stage_b_rankings)
        _write_text(sweep_root / "stage_b_pca_rankings.txt", _format_stage_b_summary(rankings=stage_b_rankings))
    print(f"[parallel-aggregate-stage-b] aggregated {len(native_payload_paths)} native payloads and {len(pca_payload_paths)} PCA payloads")


def _pca_task_lookup(tasks: Iterable[dict[str, object]]) -> dict[tuple[str, str, str, str, str, int, int], dict[str, object]]:
    lookup: dict[tuple[str, str, str, str, str, int, int], dict[str, object]] = {}
    for task in tasks:
        key = (
            str(task.get("layer_selection_method", "custom")),
            str(task.get("transport_method", "ot")),
            str(task["token_position_id"]),
            str(task["basis_source_mode"]),
            str(task["site_menu"]),
            int(task["num_bands"]),
            int(task["layer"]),
        )
        lookup[key] = task
    return lookup


def _native_task_lookup(tasks: Iterable[dict[str, object]]) -> dict[tuple[str, str, str, int, int], dict[str, object]]:
    lookup: dict[tuple[str, str, str, int, int], dict[str, object]] = {}
    for task in tasks:
        key = (
            str(task.get("layer_selection_method", "custom")),
            str(task.get("transport_method", "ot")),
            str(task.get("token_position_id", "last_token")),
            int(task["layer"]),
            int(task["resolution"]),
        )
        lookup[key] = task
    return lookup


def _select_stage_c_native_configs(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_configs_per_var: int,
) -> dict[tuple[str, str, str, int, int], tuple[str, ...]]:
    grouped: dict[tuple[str, str, str, int, int], list[str]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        entries_by_method: dict[tuple[str, str], list[dict[str, object]]] = {}
        for entry in rankings.get(target_var, []):
            key = (
                str(entry.get("layer_selection_method", "custom")),
                str(entry.get("transport_method", "ot")),
            )
            entries_by_method.setdefault(key, []).append(entry)
        for (layer_selection_method, transport_method), method_entries in entries_by_method.items():
            for entry in method_entries[: max(1, int(top_configs_per_var))]:
                key = (
                    str(layer_selection_method),
                    str(transport_method),
                    str(entry.get("token_position_id", "last_token")),
                    int(entry["layer"]),
                    int(entry["resolution"]),
                )
                grouped.setdefault(key, []).append(str(target_var))
    return {
        key: tuple(dict.fromkeys(target_vars))
        for key, target_vars in grouped.items()
    }


def plan_stage_c_tasks(args: argparse.Namespace) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    rankings_by_token = _load_stage_a_rankings_by_token(args=args, normalized=normalized, sweep_root=sweep_root)
    native_rankings_path = sweep_root / "stage_b_native_rankings.json"
    native_task_manifest_path = sweep_root / "stage_b_native_tasks.json"
    native_rankings: dict[str, list[dict[str, object]]] = {}
    selected_native_groups: dict[tuple[str, str, str, int, int], tuple[str, ...]] = {}
    native_lookup: dict[tuple[str, str, str, int, int], dict[str, object]] = {}
    if _stage_output_is_valid(native_rankings_path) and _stage_output_is_valid(native_task_manifest_path):
        native_rankings = _read_rankings(native_rankings_path)
        selected_native_groups = _select_stage_c_native_configs(
            rankings=native_rankings,
            top_configs_per_var=int(args.stage_c_top_configs_per_var),
        )
        native_manifest = _load_task_manifest(native_task_manifest_path)
        native_lookup = _native_task_lookup(native_manifest["tasks"])
    stage_b_rankings = _read_rankings(sweep_root / "stage_b_pca_rankings.json")
    selected_groups = _select_stage_c_configs(
        rankings=stage_b_rankings,
        top_configs_per_var=int(args.stage_c_top_configs_per_var),
    )
    pca_manifest = _load_task_manifest(sweep_root / "stage_b_pca_tasks.json")
    pca_lookup = _pca_task_lookup(pca_manifest["tasks"])
    stage_c_native_tasks: list[dict[str, object]] = []
    stage_c_tasks: list[dict[str, object]] = []
    a_only_tasks: list[dict[str, object]] = []
    selection_specs = _stage_b_selection_specs(args=args, normalized=normalized)

    for token_position_id, rankings in rankings_by_token.items():
        for selection_spec in selection_specs:
            selected_layers = _select_stage_b_layers_for_spec(
                rankings=rankings,
                normalized=normalized,
                selection_spec=selection_spec,
            )
            for layer in selected_layers:
                stage_timestamp = (
                    f"{str(normalized['results_timestamp'])}_stageC_{selection_spec.name}_a_only_"
                    f"{token_position_id}_L{int(layer):02d}"
                )
                output_path = _stage_a_output_path(results_root=args.results_root, stage_timestamp=stage_timestamp)
                a_only_tasks.append(
                    {
                        "task_id": f"stage_c_a_only_{selection_spec.name}_{token_position_id}_L{int(layer):02d}",
                        "category": f"stage_c_a_only_das_{selection_spec.name}",
                        "layer_selection_method": selection_spec.name,
                        "layer_selection_top_layers_per_var": int(selection_spec.top_layers_per_var),
                        "layer_selection_neighbor_radius": int(selection_spec.neighbor_radius),
                        "layer_selection_max_layers_per_var": int(selection_spec.max_layers_per_var),
                        "selected_layers": [int(selected_layer) for selected_layer in selected_layers],
                        "transport_method": "das",
                        "token_position_id": str(token_position_id),
                        "layer": int(layer),
                        "stage_timestamp": stage_timestamp,
                        "command": list(
                            _build_stage_c_a_only_command(
                                args=args,
                                normalized=normalized,
                                stage_timestamp=stage_timestamp,
                                token_position_id=str(token_position_id),
                                layers=(int(layer),),
                            )
                        ),
                        "expected_outputs": [str(output_path)],
                        "payload_paths": [str(output_path)],
                    }
                )

    for (layer_selection_method, transport_method, token_position_id, layer, resolution), target_vars in selected_native_groups.items():
        key = (
            str(layer_selection_method),
            str(transport_method),
            str(token_position_id),
            int(layer),
            int(resolution),
        )
        stage_b_task = native_lookup.get(key)
        if stage_b_task is None:
            raise RuntimeError(f"Could not find Stage B native task for Stage C key={key}")
        stage_timestamp = str(stage_b_task["stage_timestamp"])
        expected_outputs = _expected_native_guided_outputs(
            args=args,
            stage_timestamp=stage_timestamp,
            layer=int(layer),
            resolution=int(resolution),
            target_vars=tuple(str(target_var) for target_var in normalized["target_vars"]),
        )
        task_normalized = dict(normalized)
        task_normalized["native_block_resolutions"] = (int(resolution),)
        stage_c_native_tasks.append(
            {
                "task_id": f"stage_c_{stage_b_task['task_id']}",
                "category": "stage_c_native_guided_das",
                "layer_selection_method": str(layer_selection_method),
                "layer_selection_top_layers_per_var": stage_b_task.get("layer_selection_top_layers_per_var"),
                "layer_selection_neighbor_radius": stage_b_task.get("layer_selection_neighbor_radius"),
                "layer_selection_max_layers_per_var": stage_b_task.get("layer_selection_max_layers_per_var"),
                "selected_layers": stage_b_task.get("selected_layers", []),
                "transport_method": str(transport_method),
                "token_position_id": str(token_position_id),
                "layer": int(layer),
                "resolution": int(resolution),
                "stage_timestamp": stage_timestamp,
                "stage_c_target_vars": list(target_vars),
                "stage_c_selection_basis": "top_stage_b_plot_calibration_score_per_variable",
                "guided_mask_names": [str(mask_name) for mask_name in normalized["guided_mask_names"]],
                "command": list(
                    _build_native_block_command(
                        args=args,
                        normalized=task_normalized,
                        stage_timestamp=stage_timestamp,
                        layers=(int(layer),),
                        transport_method=str(transport_method),
                        skip_full_das=True,
                        skip_guided_das=False,
                        guided_mask_names=tuple(str(mask_name) for mask_name in normalized["guided_mask_names"]),
                    )
                ),
                "expected_outputs": expected_outputs,
                "payload_paths": [expected_outputs[0]],
            }
        )

    for (layer_selection_method, transport_method, token_position_id, basis_source_mode, site_menu, num_bands), layers in selected_groups.items():
        for layer in layers:
            key = (
                str(layer_selection_method),
                str(transport_method),
                str(token_position_id),
                str(basis_source_mode),
                str(site_menu),
                int(num_bands),
                int(layer),
            )
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
                    "layer_selection_method": str(layer_selection_method),
                    "layer_selection_top_layers_per_var": stage_b_task.get("layer_selection_top_layers_per_var"),
                    "layer_selection_neighbor_radius": stage_b_task.get("layer_selection_neighbor_radius"),
                    "layer_selection_max_layers_per_var": stage_b_task.get("layer_selection_max_layers_per_var"),
                    "selected_layers": stage_b_task.get("selected_layers", []),
                    "transport_method": str(transport_method),
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
                            transport_method=str(transport_method),
                        )
                    ),
                    "expected_outputs": expected_outputs,
                    "payload_paths": [expected_outputs[0]],
                }
            )
    _write_task_manifest(sweep_root / "stage_c_native_tasks.json", kind="stage_c_native_tasks", tasks=stage_c_native_tasks)
    _write_task_manifest(sweep_root / "stage_c_pca_tasks.json", kind="stage_c_pca_tasks", tasks=stage_c_tasks)
    _write_task_manifest(sweep_root / "stage_c_a_only_tasks.json", kind="stage_c_a_only_tasks", tasks=a_only_tasks)
    _write_text(
        sweep_root / "stage_c_task_plan.txt",
        "\n".join(
            [
                "MCQA hierarchical parallel Stage C task plan",
                "stage_b_selection_methods: "
                + ",".join(f"{spec.name}(top={spec.top_layers_per_var},r={spec.neighbor_radius},max={spec.max_layers_per_var})" for spec in selection_specs),
                f"stage_c_native_tasks: {len(stage_c_native_tasks)}",
                f"stage_c_pca_tasks: {len(stage_c_tasks)}",
                f"stage_c_a_only_tasks: {len(a_only_tasks)}",
                "",
                *[f"{task['task_id']} -> {task['stage_timestamp']}" for task in a_only_tasks],
                "",
                *[f"{task['task_id']} -> {task['stage_timestamp']}" for task in stage_c_native_tasks],
                "",
                *[f"{task['task_id']} -> {task['stage_timestamp']}" for task in stage_c_tasks],
            ]
        ),
    )
    print(
        f"[parallel-plan-stage-c] wrote {len(stage_c_native_tasks)} native Stage C tasks, "
        f"{len(stage_c_tasks)} PCA Stage C tasks, and {len(a_only_tasks)} A-only tasks to {sweep_root}"
    )


def aggregate_stage_c(args: argparse.Namespace, *, strict: bool = True) -> None:
    normalized = _normalize_args(args)
    sweep_root = _sweep_root(args, normalized)
    native_task_manifest_path = sweep_root / "stage_c_native_tasks.json"
    task_manifest_path = sweep_root / "stage_c_pca_tasks.json"
    a_only_manifest_path = sweep_root / "stage_c_a_only_tasks.json"
    if not _stage_output_is_valid(task_manifest_path):
        if strict:
            raise FileNotFoundError(f"Missing Stage C task manifest: {task_manifest_path}")
        return
    native_stage_c_payload_paths: list[Path] = []
    if _stage_output_is_valid(native_task_manifest_path):
        native_task_manifest = _load_task_manifest(native_task_manifest_path)
        native_stage_c_payload_paths = _valid_payload_paths_from_tasks(native_task_manifest["tasks"], strict=strict)
        if native_stage_c_payload_paths:
            _, native_guided_rankings, _ = _extract_native_rankings(
                payload_paths=native_stage_c_payload_paths,
                metadata_by_payload_path=_task_metadata_by_payload_path(native_task_manifest["tasks"]),
            )
            _write_json(sweep_root / "stage_c_native_guided_rankings.json", native_guided_rankings)
            _write_text(
                sweep_root / "stage_c_native_guided_rankings.txt",
                _format_native_summary(title="MCQA Hierarchical Stage C Native Guided DAS Ranking", rankings=native_guided_rankings),
            )
    task_manifest = _load_task_manifest(task_manifest_path)
    stage_c_payload_paths = _valid_payload_paths_from_tasks(task_manifest["tasks"], strict=strict)
    stage_c_rankings = _extract_stage_c_rankings(
        payload_paths=stage_c_payload_paths,
        metadata_by_payload_path=_task_metadata_by_payload_path(task_manifest["tasks"]),
    )
    _write_json(sweep_root / "stage_c_guided_rankings.json", stage_c_rankings)
    _write_text(sweep_root / "stage_c_guided_rankings.txt", _format_stage_c_summary(rankings=stage_c_rankings))
    a_only_payload_paths: list[Path] = []
    if _stage_output_is_valid(a_only_manifest_path):
        a_only_manifest = _load_task_manifest(a_only_manifest_path)
        a_only_payload_paths = _valid_payload_paths_from_tasks(a_only_manifest["tasks"], strict=strict)
        a_only_rankings = _extract_a_only_rankings(
            payload_paths=a_only_payload_paths,
            metadata_by_payload_path=_task_metadata_by_payload_path(a_only_manifest["tasks"]),
        )
        _write_json(sweep_root / "stage_c_a_only_rankings.json", a_only_rankings)
        _write_text(
            sweep_root / "stage_c_a_only_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage C A-only DAS Ranking", rankings=a_only_rankings),
        )
    print(
        f"[parallel-aggregate-stage-c] aggregated {len(native_stage_c_payload_paths)} native Stage C payloads, "
        f"{len(stage_c_payload_paths)} PCA Stage C payloads "
        f"and {len(a_only_payload_paths)} A-only payloads"
    )


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
    write_paper_runtime_summary(sweep_root=sweep_root, full_das_outputs=list(getattr(args, "full_das_output", []) or []))
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
