from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import mcqa_run as base_run
import torch
from mcqa_experiment.data import canonicalize_target_var
from mcqa_experiment.ot import (
    OTConfig,
    _evaluate_single_site_intervention,
    adjust_runtime_for_cached_signatures,
    load_prepared_alignment_artifacts,
    normalize_transport_rows,
    prepare_alignment_artifacts,
    resolve_recorded_artifact_prepare_seconds,
    save_prepared_alignment_artifacts,
    solve_ot_transport,
    solve_uot_transport,
)
from mcqa_experiment.metrics import build_variable_signature
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import enumerate_residual_sites
from mcqa_experiment.support import extract_ordered_site_support


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_UOT_BETA_NEURALS = (0.1, 0.3, 1.0, 3.0)
DEFAULT_OT_LAMBDAS = (1.0,)
DEFAULT_DISPLAY_TOP_LAYER_COUNT = 3
DEFAULT_STAGE_A_INTERVENTION_STRENGTH = 1.0
DEFAULT_STAGE_A_ROW_TOP_K = 6
DEFAULT_STAGE_A_TRANSPORT_METHODS = ("uot",)
SUPPORTED_STAGE_A_TRANSPORT_METHODS = ("uot", "ot")


def _synchronize_if_cuda(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _parse_csv_ints(value: str | None) -> list[int] | None:
    items = _parse_csv_strings(value)
    if items is None:
        return None
    return [int(item) for item in items]


def _parse_csv_floats(value: str | None) -> list[float] | None:
    items = _parse_csv_strings(value)
    if items is None:
        return None
    return [float(item) for item in items]


def _parse_stage_a_transport_methods(value: str | None) -> tuple[str, ...]:
    raw_methods = _parse_csv_strings(value) or list(DEFAULT_STAGE_A_TRANSPORT_METHODS)
    methods = tuple(dict.fromkeys(str(method).strip().lower() for method in raw_methods if str(method).strip()))
    unsupported = sorted(set(methods) - set(SUPPORTED_STAGE_A_TRANSPORT_METHODS))
    if unsupported:
        raise ValueError(
            f"Unsupported Stage A transport methods: {unsupported}. "
            f"Supported methods are {list(SUPPORTED_STAGE_A_TRANSPORT_METHODS)}."
        )
    if not methods:
        raise ValueError("At least one Stage A transport method is required.")
    return methods


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint layer-level PLOT transport localization for MCQA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument("--token-position-id", default=DEFAULT_TOKEN_POSITION_ID)
    parser.add_argument("--ot-epsilons", help="Comma-separated Stage A OT/UOT epsilons. Default: 0.5,1,2,4")
    parser.add_argument(
        "--stage-a-transport-methods",
        default=",".join(DEFAULT_STAGE_A_TRANSPORT_METHODS),
        help="Comma-separated Stage A transport methods to sweep: uot,ot. Default: uot",
    )
    parser.add_argument("--uot-beta-neurals", help="Comma-separated UOT beta_neural values. Ignored for balanced OT. Default: 0.1,0.3,1,3")
    parser.add_argument(
        "--ot-lambdas",
        help="Deprecated compatibility flag. Stage A now uses a fixed full-strength intervention and ignores lambda sweeps.",
    )
    parser.add_argument(
        "--stage-a-row-top-k",
        type=int,
        default=DEFAULT_STAGE_A_ROW_TOP_K,
        help="Number of top-ranked layers per transport row to calibrate directly before selecting the best layer. Default: 6",
    )
    parser.add_argument(
        "--calibration-family-weights",
        help="Comma-separated family weights in answerPosition,randomLetter,answerPosition_randomLetter order. Default: 1,1,1",
    )
    parser.add_argument("--support-score-slack", type=float, default=0.05)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _configure_base_run(args: argparse.Namespace, *, sweep_root: Path, results_timestamp: str) -> None:
    base_run.DEVICE = str(args.device)
    base_run.MODEL_NAME = str(args.model_name)
    base_run.RUN_TIMESTAMP = str(results_timestamp)
    base_run.RUN_DIR = sweep_root
    base_run.OUTPUT_PATH = sweep_root / "mcqa_run_results.json"
    base_run.SUMMARY_PATH = sweep_root / "mcqa_run_summary.txt"
    base_run.SIGNATURES_DIR = Path(args.signatures_dir)
    base_run.MCQA_DATASET_PATH = str(args.dataset_path)
    base_run.MCQA_DATASET_CONFIG = args.dataset_config or None
    base_run.DATASET_SIZE = int(args.dataset_size)
    base_run.SPLIT_SEED = int(args.split_seed)
    base_run.TRAIN_POOL_SIZE = int(args.train_pool_size)
    base_run.CALIBRATION_POOL_SIZE = int(args.calibration_pool_size)
    base_run.TEST_POOL_SIZE = int(args.test_pool_size)
    base_run.BATCH_SIZE = int(args.batch_size)
    base_run.TARGET_VARS = list(DEFAULT_TARGET_VARS)
    base_run.COUNTERFACTUAL_NAMES = list(DEFAULT_COUNTERFACTUAL_NAMES)
    base_run.PROMPT_HF_LOGIN = bool(args.prompt_hf_login)
    base_run.TOKEN_POSITION_IDS = [str(args.token_position_id)]


def _load_existing_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _target_row_ranking(payload: dict[str, object], *, sites) -> list[dict[str, object]]:
    target_row_transport = payload.get("target_normalized_transport", payload.get("target_transport", []))
    if not isinstance(target_row_transport, list) or not target_row_transport:
        return []
    first_entry = target_row_transport[0]
    if isinstance(first_entry, list):
        row_transport = first_entry
    else:
        row_transport = target_row_transport
    ranking = [
        {
            "site_index": int(site_index),
            "site_label": sites[int(site_index)].label,
            "layer": int(sites[int(site_index)].layer),
            "token_position_id": str(sites[int(site_index)].token_position_id),
            "dim_start": int(sites[int(site_index)].dim_start),
            "dim_end": int(sites[int(site_index)].dim_end),
            "transport_mass": float(row_transport[site_index]),
        }
        for site_index in range(min(len(sites), len(row_transport)))
    ]
    return sorted(
        ranking,
        key=lambda entry: (
            float(entry.get("transport_mass", 0.0)),
            -int(entry.get("site_index", 10**9)),
        ),
        reverse=True,
    )


def _iter_stage_a_transport_payloads(compare_payload: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    method_payloads = compare_payload.get("method_payloads", {})
    if not isinstance(method_payloads, dict):
        return []
    payloads: list[tuple[str, dict[str, object]]] = []
    for method_name in ("uot", "ot"):
        entries = method_payloads.get(method_name, [])
        if not isinstance(entries, list):
            continue
        for payload in entries:
            if isinstance(payload, dict):
                payloads.append((str(method_name), payload))
    return payloads


def _build_transport_only_compare_payload(
    *,
    method: str,
    target_vars: tuple[str, ...],
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures_by_var: dict[str, torch.Tensor],
    epsilon: float,
    beta_neural: float | None,
    signature_mode: str,
    calibration_family_weights: tuple[float, ...],
) -> dict[str, object]:
    method = str(method).lower()
    if method not in SUPPORTED_STAGE_A_TRANSPORT_METHODS:
        raise ValueError(f"Unsupported Stage A transport method {method}")
    resolved_beta_neural = None if method == "ot" else float(beta_neural)
    solve_start = perf_counter()
    config = OTConfig(
        method=method,
        batch_size=1,
        epsilon=float(epsilon),
        uot_beta_neural=1.0 if resolved_beta_neural is None else float(resolved_beta_neural),
        signature_mode=str(signature_mode),
        selection_verbose=True,
        source_target_vars=target_vars,
        calibration_metric=DEFAULT_CALIBRATION_METRIC,
        calibration_family_weights=tuple(float(weight) for weight in calibration_family_weights),
    )
    if method == "ot":
        transport, transport_meta = solve_ot_transport(
            variable_signatures_by_var,
            site_signatures_by_var,
            config,
        )
    else:
        transport, transport_meta = solve_uot_transport(
            variable_signatures_by_var,
            site_signatures_by_var,
            config,
        )
    runtime_seconds = float(perf_counter() - solve_start)
    normalized_transport = normalize_transport_rows(transport)
    method_payloads: list[dict[str, object]] = []
    for target_row_index, target_var in enumerate(target_vars):
        method_payloads.append(
            {
                "kind": "mcqa_plot_layer_transport_target_row",
                "method": method,
                "target_var": str(target_var),
                "ot_epsilon": float(epsilon),
                "uot_beta_neural": resolved_beta_neural,
                "source_target_vars": [str(var) for var in target_vars],
                "target_var_row_index": int(target_row_index),
                "transport": transport.tolist(),
                "target_transport": transport[target_row_index : target_row_index + 1].tolist(),
                "target_normalized_transport": normalized_transport[target_row_index : target_row_index + 1].tolist(),
                "transport_meta": dict(transport_meta),
                "runtime_seconds": float(runtime_seconds),
                "wall_runtime_seconds": float(runtime_seconds),
                "results": [],
            }
        )
    return {
        "kind": "mcqa_plot_layer_transport_only_compare",
        "method_payloads": {method: method_payloads},
        "stage_a_transport_method": method,
        "ot_epsilon": float(epsilon),
        "uot_beta_neural": resolved_beta_neural,
        "source_target_vars": [str(var) for var in target_vars],
        "transport_meta": dict(transport_meta),
        "runtime_seconds": float(runtime_seconds),
        "wall_runtime_seconds": float(runtime_seconds),
    }


def _sanitize_stage_a_eval_result(record: dict[str, object]) -> dict[str, object]:
    cleaned = dict(record)
    cleaned.pop("top_k", None)
    cleaned.pop("lambda", None)
    return cleaned


def _stage_a_calibration_score(
    *,
    result: dict[str, object],
    calibration_family_weights: tuple[float, ...],
) -> float:
    exact_acc = float(result.get("exact_acc", 0.0))
    family_exact_accs = result.get("family_exact_accs", {})
    if not isinstance(family_exact_accs, dict):
        return exact_acc
    weighted_sum = 0.0
    total_weight = 0.0
    for family_name, weight in zip(DEFAULT_COUNTERFACTUAL_NAMES, calibration_family_weights):
        if family_name not in family_exact_accs:
            continue
        weighted_sum += float(weight) * float(family_exact_accs[family_name])
        total_weight += float(weight)
    return exact_acc if total_weight <= 0.0 else float(weighted_sum / total_weight)


def _evaluate_fixed_single_layer_calibration_only(
    *,
    model,
    tokenizer,
    calibration_bank,
    site,
    site_index: int,
    device,
    batch_size: int,
    strength: float,
    calibration_family_weights: tuple[float, ...],
) -> dict[str, object]:
    eval_start = perf_counter()
    calibration_result, calibration_ranking = _evaluate_single_site_intervention(
        model=model,
        bank=calibration_bank,
        site=site,
        site_index=int(site_index),
        strength=float(strength),
        batch_size=int(batch_size),
        device=device,
        tokenizer=tokenizer,
        include_details=True,
    )
    calibration_result = _sanitize_stage_a_eval_result(calibration_result)
    calibration_score = _stage_a_calibration_score(
        result=calibration_result,
        calibration_family_weights=calibration_family_weights,
    )
    runtime_seconds = float(perf_counter() - eval_start)
    return {
        "target_var": str(calibration_bank.target_var),
        "selected_site_label": str(site.label),
        "selected_layer": int(site.layer),
        "intervention_strength": float(strength),
        "runtime_seconds": float(runtime_seconds),
        "calibration_score": float(calibration_score),
        "selected_calibration_result": calibration_result,
        "selected_calibration_ranking": calibration_ranking,
    }


def _evaluate_fixed_single_layer_holdout_only(
    *,
    model,
    tokenizer,
    holdout_bank,
    site,
    site_index: int,
    device,
    batch_size: int,
    strength: float,
    calibration_score: float,
    calibration_exact_acc: float,
) -> dict[str, object]:
    eval_start = perf_counter()
    holdout_result, holdout_ranking = _evaluate_single_site_intervention(
        model=model,
        bank=holdout_bank,
        site=site,
        site_index=int(site_index),
        strength=float(strength),
        batch_size=int(batch_size),
        device=device,
        tokenizer=tokenizer,
        include_details=True,
    )
    holdout_result = _sanitize_stage_a_eval_result(holdout_result)
    runtime_seconds = float(perf_counter() - eval_start)
    holdout_result["method"] = "single_layer_full_swap"
    holdout_result["selection_score"] = float(calibration_score)
    holdout_result["selection_exact_acc"] = float(calibration_exact_acc)
    holdout_result["calibration_exact_acc"] = float(calibration_exact_acc)
    holdout_result["lambda"] = float(strength)
    return {
        "target_var": str(holdout_bank.target_var),
        "selected_site_label": str(site.label),
        "selected_layer": int(site.layer),
        "intervention_strength": float(strength),
        "runtime_seconds": float(runtime_seconds),
        "ranking": holdout_ranking,
        "results": [holdout_result],
    }


def _evaluate_stage_a_config(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    compare_payload: dict[str, object],
    device,
    batch_size: int,
    signature_mode: str,
    calibration_family_weights: tuple[float, ...],
    row_top_k: int,
) -> dict[str, object]:
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    epsilon = float(compare_payload.get("ot_epsilon", 0.0))
    compare_beta = compare_payload.get("uot_beta_neural")
    compare_method = str(compare_payload.get("stage_a_transport_method", "uot")).lower()
    row_info_by_var: dict[str, dict[str, object]] = {}
    for target_var in target_vars:
        for method_name, payload in _iter_stage_a_transport_payloads(compare_payload):
            if str(payload.get("target_var")) != str(target_var):
                continue
            row_ranking = _target_row_ranking(payload, sites=sites)
            shortlisted_entries: list[dict[str, object]] = []
            for rank_index, entry in enumerate(row_ranking[: max(1, int(row_top_k))]):
                site_index = int(entry.get("site_index", -1))
                if not (0 <= site_index < len(sites)):
                    continue
                shortlisted_entries.append(
                    {
                        "rank_index": int(rank_index),
                        "site_index": int(site_index),
                        "site": sites[site_index],
                        "site_label": str(entry.get("site_label", sites[site_index].label)),
                        "layer": int(entry.get("layer", sites[site_index].layer)),
                        "transport_mass": float(entry.get("transport_mass", 0.0)),
                    }
                )
            if not shortlisted_entries and sites:
                shortlisted_entries.append(
                    {
                        "rank_index": 0,
                        "site_index": 0,
                        "site": sites[0],
                        "site_label": str(sites[0].label),
                        "layer": int(sites[0].layer),
                        "transport_mass": 0.0,
                    }
                )
            if not shortlisted_entries:
                continue
            row_info_by_var[str(target_var)] = {
                "method_name": str(method_name),
                "payload": payload,
                "row_ranking": row_ranking,
                "shortlisted_entries": shortlisted_entries,
            }
            break

    calibration_sweep_runtime_seconds = 0.0
    per_var_records: dict[str, dict[str, object]] = {}
    for target_var in target_vars:
        row_info = row_info_by_var.get(str(target_var))
        if not isinstance(row_info, dict):
            continue
        payload = row_info["payload"]
        row_ranking = row_info["row_ranking"]
        shortlisted_entries = row_info["shortlisted_entries"]
        candidate_records: list[dict[str, object]] = []
        for shortlisted_entry in shortlisted_entries:
            candidate_site = shortlisted_entry["site"]
            calibration_eval_payload = _evaluate_fixed_single_layer_calibration_only(
                model=model,
                tokenizer=tokenizer,
                calibration_bank=banks_by_split["calibration"][str(target_var)],
                site=candidate_site,
                site_index=int(shortlisted_entry["site_index"]),
                device=device,
                batch_size=int(batch_size),
                strength=float(DEFAULT_STAGE_A_INTERVENTION_STRENGTH),
                calibration_family_weights=calibration_family_weights,
            )
            calibration_sweep_runtime_seconds += float(calibration_eval_payload.get("runtime_seconds", 0.0))
            calibration_result = calibration_eval_payload.get("selected_calibration_result", {})
            calibration_score = float(
                calibration_eval_payload.get(
                    "calibration_score",
                    calibration_result.get("selection_score", calibration_result.get("exact_acc", 0.0)),
                )
            )
            candidate_records.append(
                {
                    "rank_index": int(shortlisted_entry["rank_index"]),
                    "site_index": int(shortlisted_entry["site_index"]),
                    "site": candidate_site,
                    "site_label": str(shortlisted_entry["site_label"]),
                    "layer": int(shortlisted_entry["layer"]),
                    "transport_mass": float(shortlisted_entry["transport_mass"]),
                    "runtime_seconds": float(calibration_eval_payload.get("runtime_seconds", 0.0)),
                    "calibration_score": float(calibration_score),
                    "calibration_exact_acc": float(calibration_result.get("exact_acc", 0.0)),
                    "calibration_payload": calibration_eval_payload,
                }
            )
        best_candidate = max(
            candidate_records,
            key=lambda record: (
                float(record.get("calibration_score", 0.0)),
                float(record.get("calibration_exact_acc", 0.0)),
                float(record.get("transport_mass", 0.0)),
                -int(record.get("rank_index", 10**9)),
            ),
        )
        calibration_score = float(best_candidate["calibration_score"])
        record = {
            "method": str(payload.get("method", row_info["method_name"])),
            "variable": str(target_var),
            "exact_acc": None,
            "selection_score": float(calibration_score),
            "selection_exact_acc": float(best_candidate["calibration_exact_acc"]),
            "calibration_exact_acc": float(best_candidate["calibration_exact_acc"]),
            "site_label": str(best_candidate["site_label"]),
            "layer": int(best_candidate["layer"]),
            "epsilon": float(epsilon),
            "uot_beta_neural": None
            if payload.get("uot_beta_neural", compare_beta) is None
            else float(payload.get("uot_beta_neural", compare_beta)),
            "intervention_strength": float(DEFAULT_STAGE_A_INTERVENTION_STRENGTH),
            "candidate_site_labels": [str(entry["site_label"]) for entry in shortlisted_entries],
            "candidate_layers": [int(entry["layer"]) for entry in shortlisted_entries],
            "coupling_argmax_site_label": str(shortlisted_entries[0]["site_label"]),
            "coupling_argmax_layer": int(shortlisted_entries[0]["layer"]),
            "coupling_top_site_labels": [str(entry.get("site_label", "")) for entry in row_ranking[:DEFAULT_DISPLAY_TOP_LAYER_COUNT]],
            "coupling_top_layers": [int(entry.get("layer", -1)) for entry in row_ranking[:DEFAULT_DISPLAY_TOP_LAYER_COUNT]],
            "selection_basis": "best_calibrated_single_site_within_coupling_topk",
            "selected_candidate_rank": int(best_candidate["rank_index"]),
            "selected_candidate_transport_mass": float(best_candidate["transport_mass"]),
            "candidate_records": [
                {
                    "rank_index": int(candidate["rank_index"]),
                    "site_index": int(candidate["site_index"]),
                    "site_label": str(candidate["site_label"]),
                    "layer": int(candidate["layer"]),
                    "transport_mass": float(candidate["transport_mass"]),
                    "calibration_score": float(candidate["calibration_score"]),
                    "calibration_exact_acc": float(candidate["calibration_exact_acc"]),
                    "runtime_seconds": float(candidate["runtime_seconds"]),
                }
                for candidate in candidate_records
            ],
            "target_row_ranking": row_ranking,
            "runtime_seconds": float(sum(candidate["runtime_seconds"] for candidate in candidate_records)),
            "holdout_runtime_seconds": 0.0,
            "calibration_payload": best_candidate["calibration_payload"],
            "payload": best_candidate["calibration_payload"],
        }
        per_var_records[str(target_var)] = record

    calibration_scores = [
        float(record.get("selection_score", 0.0))
        for target_var in target_vars
        for record in [per_var_records.get(str(target_var))]
        if isinstance(record, dict)
    ]
    calibration_exact_scores = [
        float(record.get("selection_exact_acc", record.get("calibration_exact_acc", 0.0)))
        for target_var in target_vars
        for record in [per_var_records.get(str(target_var))]
        if isinstance(record, dict)
    ]

    if not per_var_records:
        return {
            "method": compare_method,
            "epsilon": float(epsilon),
            "uot_beta_neural": None if compare_beta is None else float(compare_beta),
            "intervention_strength": float(DEFAULT_STAGE_A_INTERVENTION_STRENGTH),
            "per_var_records": {},
            "mean_calibration_score": 0.0,
            "mean_calibration_exact_acc": 0.0,
            "calibration_sweep_runtime_seconds": 0.0,
            "row_top_k": int(row_top_k),
        }

    for target_var, record in per_var_records.items():
        row_info = row_info_by_var.get(str(target_var))
        if not isinstance(row_info, dict):
            continue
        payload = row_info["payload"]
        payload["results"] = [
            {
                "selection_score": float(record["selection_score"]),
                "exact_acc": float(
                    record.get(
                        "selection_exact_acc",
                        record.get("calibration_exact_acc", 0.0),
                    )
                ),
                "site_label": str(record["site_label"]),
                "layer": int(record["layer"]),
                "intervention_strength": float(DEFAULT_STAGE_A_INTERVENTION_STRENGTH),
                "runtime_seconds": float(record.get("runtime_seconds", 0.0)),
                "variable": str(target_var),
            }
        ]
    return {
        "method": compare_method,
        "epsilon": float(epsilon),
        "uot_beta_neural": None if compare_beta is None else float(compare_beta),
        "intervention_strength": float(DEFAULT_STAGE_A_INTERVENTION_STRENGTH),
        "per_var_records": per_var_records,
        "mean_calibration_score": float(sum(calibration_scores) / len(calibration_scores)) if calibration_scores else 0.0,
        "mean_calibration_exact_acc": (
            float(sum(calibration_exact_scores) / len(calibration_exact_scores))
            if calibration_exact_scores else 0.0
        ),
        "calibration_sweep_runtime_seconds": float(calibration_sweep_runtime_seconds),
        "row_top_k": int(row_top_k),
    }


def _format_stage_a_config_line(
    *,
    candidate_config: dict[str, object],
    runtime_with_signatures_seconds: float,
) -> str:
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    per_var_records = candidate_config.get("per_var_records", {})
    chosen_layers = {
        str(target_var): int(record.get("layer", -1))
        for target_var in target_vars
        for record in [per_var_records.get(str(target_var))]
        if isinstance(record, dict)
    }
    calibration_scores = {
        str(target_var): float(record.get("selection_score", 0.0))
        for target_var in target_vars
        for record in [per_var_records.get(str(target_var))]
        if isinstance(record, dict)
    }
    calibration_exact_scores = {
        str(target_var): float(record.get("selection_exact_acc", record.get("calibration_exact_acc", 0.0)))
        for target_var in target_vars
        for record in [per_var_records.get(str(target_var))]
        if isinstance(record, dict)
    }
    return (
        f"[stageA] coupling_result method={candidate_config.get('method', 'transport')} "
        f"eps={float(candidate_config.get('epsilon', 0.0)):g} "
        + (
            f"beta_neural={float(candidate_config.get('uot_beta_neural')):g} "
            if candidate_config.get("uot_beta_neural") is not None else ""
        )
        + f"row_top_k={int(candidate_config.get('row_top_k', DEFAULT_STAGE_A_ROW_TOP_K))} "
        + f"layers={chosen_layers} "
        + f"cal={{{', '.join(f'{target_var}: {score:.4f}' for target_var, score in calibration_scores.items())}}} "
        + f"cal_exact={{{', '.join(f'{target_var}: {score:.4f}' for target_var, score in calibration_exact_scores.items())}}} "
        + f"avg_cal={float(candidate_config.get('mean_calibration_score', 0.0)):.4f} "
        + f"avg_cal_exact={float(candidate_config.get('mean_calibration_exact_acc', 0.0)):.4f} "
        + f"runtime_with_signatures={float(runtime_with_signatures_seconds):.2f}s"
    )


def _select_joint_layer_config(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    ot_compare_payloads: list[dict[str, object]],
    device,
    batch_size: int,
    signature_mode: str,
    calibration_family_weights: tuple[float, ...],
    signature_prepare_runtime_seconds: float,
    row_top_k: int,
) -> dict[str, object]:
    best_config: dict[str, object] | None = None
    best_config_by_method: dict[str, dict[str, object]] = {}
    candidate_configs: list[dict[str, object]] = []
    for compare_payload in ot_compare_payloads:
        candidate_config = _evaluate_stage_a_config(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=sites,
            compare_payload=compare_payload,
            device=device,
            batch_size=batch_size,
            signature_mode=signature_mode,
            calibration_family_weights=calibration_family_weights,
            row_top_k=row_top_k,
        )
        shortlist_runtime_seconds = float(candidate_config.get("calibration_sweep_runtime_seconds", 0.0))
        candidate_config["signature_prepare_runtime_seconds"] = float(signature_prepare_runtime_seconds)
        candidate_config["transport_solve_runtime_seconds"] = float(compare_payload.get("runtime_seconds", 0.0))
        candidate_config["direct_eval_runtime_seconds"] = float(shortlist_runtime_seconds)
        candidate_config["runtime_with_signatures_seconds"] = float(
            signature_prepare_runtime_seconds
            + float(compare_payload.get("runtime_seconds", 0.0))
            + shortlist_runtime_seconds
        )
        for record in candidate_config.get("per_var_records", {}).values():
            if isinstance(record, dict):
                record["runtime_with_signatures_seconds"] = float(candidate_config["runtime_with_signatures_seconds"])
        print(
            _format_stage_a_config_line(
                candidate_config=candidate_config,
                runtime_with_signatures_seconds=float(candidate_config["runtime_with_signatures_seconds"]),
            )
        )
        candidate_configs.append(dict(candidate_config))
        candidate_method = str(candidate_config.get("method", "transport"))
        method_best = best_config_by_method.get(candidate_method)
        if method_best is None or (
            float(candidate_config["mean_calibration_score"]),
            float(candidate_config.get("mean_calibration_exact_acc", 0.0)),
        ) > (
            float(method_best["mean_calibration_score"]),
            float(method_best.get("mean_calibration_exact_acc", 0.0)),
        ):
            best_config_by_method[candidate_method] = dict(candidate_config)
        if best_config is None or (
            float(candidate_config["mean_calibration_score"]),
            float(candidate_config.get("mean_calibration_exact_acc", 0.0)),
        ) > (
            float(best_config["mean_calibration_score"]),
            float(best_config.get("mean_calibration_exact_acc", 0.0)),
        ):
            best_config = candidate_config
    if best_config is None:
        return {}
    selected = dict(best_config)
    selected["candidate_configs"] = candidate_configs
    selected["best_config_by_method"] = best_config_by_method
    return selected


def _evaluate_selected_stage_a_holdout(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    device,
    batch_size: int,
    selected_config: dict[str, object],
) -> dict[str, object]:
    if not isinstance(selected_config, dict):
        return {}
    per_var_records = selected_config.get("per_var_records", {})
    if not isinstance(per_var_records, dict):
        return dict(selected_config)
    updated = dict(selected_config)
    updated_records: dict[str, dict[str, object]] = {}
    holdout_exact_scores: list[float] = []
    total_holdout_runtime_seconds = 0.0
    for target_var, record in per_var_records.items():
        if not isinstance(record, dict):
            continue
        site_label = str(record.get("site_label", ""))
        matching_site_index = next(
            (
                site_index
                for site_index, site in enumerate(sites)
                if str(site.label) == site_label
            ),
            None,
        )
        if matching_site_index is None:
            updated_records[str(target_var)] = dict(record)
            continue
        holdout_eval_payload = _evaluate_fixed_single_layer_holdout_only(
            model=model,
            tokenizer=tokenizer,
            holdout_bank=banks_by_split["test"][str(target_var)],
            site=sites[int(matching_site_index)],
            site_index=int(matching_site_index),
            device=device,
            batch_size=int(batch_size),
            strength=float(record.get("intervention_strength", DEFAULT_STAGE_A_INTERVENTION_STRENGTH)),
            calibration_score=float(record.get("selection_score", 0.0)),
            calibration_exact_acc=float(record.get("selection_exact_acc", record.get("calibration_exact_acc", 0.0))),
        )
        holdout_result = holdout_eval_payload.get("results", [{}])[0]
        updated_record = dict(record)
        updated_record["exact_acc"] = float(holdout_result.get("exact_acc", 0.0))
        updated_record["holdout_runtime_seconds"] = float(holdout_eval_payload.get("runtime_seconds", 0.0))
        updated_record["payload"] = holdout_eval_payload
        updated_record["ranking"] = holdout_eval_payload.get("ranking", [])
        updated_records[str(target_var)] = updated_record
        holdout_exact_scores.append(float(updated_record["exact_acc"]))
        total_holdout_runtime_seconds += float(updated_record["holdout_runtime_seconds"])
    updated["per_var_records"] = updated_records
    updated["mean_exact_acc"] = (
        float(sum(holdout_exact_scores) / len(holdout_exact_scores))
        if holdout_exact_scores else 0.0
    )
    updated["holdout_eval_runtime_seconds"] = float(total_holdout_runtime_seconds)
    return updated


def _rank_layers_from_target_row(
    *,
    selected_method_by_var: dict[str, dict[str, object]],
    runtime_seconds: float,
) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {}
    for target_var, selected in selected_method_by_var.items():
        row_ranking = [dict(entry) for entry in selected.get("target_row_ranking", []) if isinstance(entry, dict)]
        selected_layer = int(selected.get("layer", -1))
        selected_transport_mass = float(selected.get("selected_candidate_transport_mass", 0.0))
        selected_entry = {
            "variable": str(target_var),
            "layer": int(selected_layer),
            "selection_score": float(selected.get("selection_score", 0.0)),
            "target_row_transport_mass": float(selected_transport_mass),
            "exact_acc": float(selected.get("exact_acc", 0.0)),
            "selection_exact_acc": float(selected.get("selection_exact_acc", 0.0)),
            "method_selection_score": float(selected.get("selection_score", 0.0)),
            "method": selected.get("method"),
            "epsilon": float(selected.get("epsilon", 0.0)),
            "uot_beta_neural": selected.get("uot_beta_neural"),
            "site_label": selected.get("site_label"),
            "selected_site_label": selected.get("site_label"),
            "runtime_seconds": float(runtime_seconds),
            "selection_basis": "selected_calibrated_single_site_within_coupling_topk",
        }
        seen_layers = {int(selected_layer)}
        remaining_entries = []
        for entry in row_ranking:
            layer = int(entry.get("layer", -1))
            if layer in seen_layers:
                continue
            remaining_entries.append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "selection_score": float(entry.get("transport_mass", 0.0)),
                    "target_row_transport_mass": float(entry.get("transport_mass", 0.0)),
                    "exact_acc": float(selected.get("exact_acc", 0.0)),
                    "selection_exact_acc": float(selected.get("selection_exact_acc", 0.0)),
                    "method_selection_score": float(selected.get("selection_score", 0.0)),
                    "method": selected.get("method"),
                    "epsilon": float(selected.get("epsilon", 0.0)),
                    "uot_beta_neural": selected.get("uot_beta_neural"),
                    "site_label": entry.get("site_label"),
                    "selected_site_label": selected.get("site_label"),
                    "runtime_seconds": float(runtime_seconds),
                    "selection_basis": "target_row_mass_from_selected_single_layer_method",
                }
            )
        rankings[str(target_var)] = [selected_entry, *remaining_entries]
    return rankings


def _format_summary(
    *,
    token_position_id: str,
    layers: tuple[int, ...],
    support_by_var: dict[str, dict[str, object]],
    selected_config: dict[str, object],
) -> str:
    lines = [
        "MCQA PLOT Layer Summary",
        f"token_position_id: {token_position_id}",
        f"layers: {list(int(layer) for layer in layers)}",
        "",
    ]
    per_var_records = selected_config.get("per_var_records", {})
    if isinstance(per_var_records, dict) and per_var_records:
        selected_method = str(selected_config.get("method", "transport"))
        chosen_layers = {
            str(target_var): int(record.get("layer", -1))
            for target_var, record in per_var_records.items()
            if isinstance(record, dict)
        }
        calibration_scores = {
            str(target_var): float(record.get("selection_score", 0.0))
            for target_var, record in per_var_records.items()
            if isinstance(record, dict)
        }
        exact_scores = {
            str(target_var): float(record.get("exact_acc", 0.0))
            for target_var, record in per_var_records.items()
            if isinstance(record, dict)
        }
        lines.extend(
            [
                f"[selected_{selected_method}_coupling]",
                f"eps={float(selected_config.get('epsilon', 0.0)):g}",
                (
                    f"beta_n={float(selected_config.get('uot_beta_neural')):g}"
                    if selected_config.get("uot_beta_neural") is not None else "beta_n=NA"
                ),
                f"strength={float(selected_config.get('intervention_strength', DEFAULT_STAGE_A_INTERVENTION_STRENGTH)):g}",
                f"row_top_k={int(selected_config.get('row_top_k', DEFAULT_STAGE_A_ROW_TOP_K))}",
                f"layers={chosen_layers}",
                f"calibration_by_var={calibration_scores}",
                f"test_by_var={exact_scores}",
                f"avg_cal={float(selected_config.get('mean_calibration_score', 0.0)):.4f}",
                f"avg_test={float(selected_config.get('mean_exact_acc', 0.0)):.4f}",
                f"runtime_with_signatures={float(selected_config.get('runtime_with_signatures_seconds', 0.0)):.2f}s",
                "",
            ]
        )
        best_config_by_method = selected_config.get("best_config_by_method", {})
        if isinstance(best_config_by_method, dict) and best_config_by_method:
            lines.append("[best_by_method]")
            for method, method_config in sorted(best_config_by_method.items()):
                if not isinstance(method_config, dict):
                    continue
                method_records = method_config.get("per_var_records", {})
                if not isinstance(method_records, dict):
                    continue
                method_layers = {
                    str(target_var): int(record.get("layer", -1))
                    for target_var, record in method_records.items()
                    if isinstance(record, dict)
                }
                method_tests = {
                    str(target_var): float(record.get("exact_acc", 0.0))
                    for target_var, record in method_records.items()
                    if isinstance(record, dict)
                }
                lines.append(
                    f"{method}: eps={float(method_config.get('epsilon', 0.0)):g} "
                    + (
                        f"beta_n={float(method_config.get('uot_beta_neural')):g} "
                        if method_config.get("uot_beta_neural") is not None else ""
                    )
                    + f"layers={method_layers} test_by_var={method_tests} "
                    + f"avg_test={float(method_config.get('mean_exact_acc', 0.0)):.4f}"
                )
            lines.append("")
    for target_var in DEFAULT_TARGET_VARS:
        selected = per_var_records.get(str(target_var), {}) if isinstance(per_var_records, dict) else {}
        lines.append(f"[{target_var}]")
        if selected.get("target_row_ranking"):
            lines.append(
                f"coupling_row_layers={[entry.get('site_label') for entry in selected.get('target_row_ranking', [])[:DEFAULT_DISPLAY_TOP_LAYER_COUNT]]}"
            )
        support_summary = support_by_var.get(str(target_var), {})
        if support_summary:
            lines.append(
                f"support_diagnostic_masks={[candidate.get('name') for candidate in support_summary.get('mask_candidates', [])]}"
            )
            lines.append(
                f"support_diagnostic_ranked_layers={support_summary.get('ranked_site_labels', [])}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_plot_layer"
    sweep_root.mkdir(parents=True, exist_ok=True)

    _configure_base_run(args, sweep_root=sweep_root, results_timestamp=results_timestamp)
    context = base_run.build_run_context()
    context_timing_seconds = dict(context.get("timing_seconds", {}))
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_positions = context["token_positions"]
    banks_by_split = context["banks_by_split"]
    data_metadata = context["data_metadata"]
    device = context["device"]
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)

    resolved_layers = tuple(
        _parse_csv_ints(args.layers) or list(range(int(model.config.num_hidden_layers)))
    )
    if not resolved_layers:
        raise ValueError("No layers selected")
    print(
        f"[stageA] start token_position={str(args.token_position_id)} "
        f"layers={len(resolved_layers)} target_vars={list(target_vars)}"
    )
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    sites = enumerate_residual_sites(
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        token_position_ids=token_position_ids,
        resolution=None,
        layers=resolved_layers,
        selected_token_position_ids=(str(args.token_position_id),),
    )

    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))
    stage_a_transport_methods = _parse_stage_a_transport_methods(args.stage_a_transport_methods)
    uot_beta_neurals = tuple(_parse_csv_floats(args.uot_beta_neurals) or list(DEFAULT_UOT_BETA_NEURALS))
    stage_a_row_top_k = max(1, int(args.stage_a_row_top_k))
    calibration_family_weights = tuple(
        _parse_csv_floats(args.calibration_family_weights) or list(DEFAULT_CALIBRATION_FAMILY_WEIGHTS)
    )
    train_banks = {target_var: banks_by_split["train"][target_var] for target_var in target_vars}
    cache_spec = base_run._signature_cache_spec(
        train_bank=train_banks[target_vars[0]],
        resolution=None,
        resolved_resolution=None,
        signature_mode=str(args.signature_mode),
        selected_layers=list(int(layer) for layer in resolved_layers),
        token_position_ids=token_position_ids,
    )
    cache_path = base_run._signature_cache_path(
        resolution=None,
        signature_mode=str(args.signature_mode),
        cache_spec=cache_spec,
    )
    artifact_load_start = perf_counter()
    prepared_artifacts = load_prepared_alignment_artifacts(cache_path, expected_spec=cache_spec)
    artifact_prepare_load_seconds = float(perf_counter() - artifact_load_start)
    artifact_prepare_create_seconds = 0.0
    if prepared_artifacts is None:
        print(f"[stageA] signature cache miss path={cache_path}")
        artifact_prepare_start = perf_counter()
        prepared_artifacts = prepare_alignment_artifacts(
            model=model,
            fit_banks_by_var=train_banks,
            sites=sites,
            device=device,
            config=OTConfig(
                method=str(stage_a_transport_methods[0]),
                batch_size=int(args.batch_size),
                epsilon=float(ot_epsilons[0]),
                uot_beta_neural=float(uot_beta_neurals[0]),
                signature_mode=str(args.signature_mode),
                source_target_vars=target_vars,
                calibration_metric=DEFAULT_CALIBRATION_METRIC,
                calibration_family_weights=calibration_family_weights,
            ),
        )
        artifact_prepare_create_seconds = float(perf_counter() - artifact_prepare_start)
        save_prepared_alignment_artifacts(
            cache_path,
            prepared_artifacts=prepared_artifacts,
            cache_spec=cache_spec,
        )
        print(f"[stageA] signature cache saved path={cache_path}")
    else:
        print(
            f"[stageA] signature cache hit path={cache_path} "
            f"loaded_from_disk={bool(prepared_artifacts.get('loaded_from_disk', False))}"
        )
    _synchronize_if_cuda(device)
    artifact_prepare_recorded_seconds = resolve_recorded_artifact_prepare_seconds(
        prepared_artifacts,
        artifact_prepare_create_seconds=artifact_prepare_create_seconds,
    )
    variable_signatures_by_var = {
        str(target_var): build_variable_signature(
            banks_by_split["train"][str(target_var)],
            str(args.signature_mode),
        )
        for target_var in target_vars
    }
    site_signatures_by_var = {
        str(target_var): tensor
        for target_var, tensor in prepared_artifacts["site_signatures_by_var"].items()
    }

    ot_compare_payloads: list[dict[str, object]] = []
    transport_output_paths: list[str] = []
    ot_localization_start = perf_counter()
    print(
        f"[stageA] running joint layer transport methods={list(stage_a_transport_methods)} "
        f"epsilon_sweep={list(ot_epsilons)} "
        f"beta_neural_sweep={list(uot_beta_neurals)} "
        f"row_top_k={int(stage_a_row_top_k)}"
    )
    for method in stage_a_transport_methods:
        if method == "ot":
            method_grid = [(epsilon, None) for epsilon in ot_epsilons]
        else:
            method_grid = [
                (epsilon, beta_neural)
                for epsilon in ot_epsilons
                for beta_neural in uot_beta_neurals
            ]
        for epsilon, beta_neural in method_grid:
            beta_suffix = "" if beta_neural is None else f"_betan-{float(beta_neural):g}"
            beta_log = "" if beta_neural is None else f" beta_neural={float(beta_neural):g}"
            print(
                f"[stageA] method={method} epsilon={float(epsilon):g}{beta_log} "
                f"joint layer {method.upper()}"
            )
            output_stem = (
                f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}"
                f"_eps-{float(epsilon):g}{beta_suffix}_{method}"
            )
            output_path = sweep_root / f"{output_stem}.json"
            summary_path = sweep_root / f"{output_stem}.txt"
            compare_payload = _load_existing_payload(output_path)
            if compare_payload is not None and str(compare_payload.get("kind")) != "mcqa_plot_layer_transport_only_compare":
                print(f"[stageA] ignoring legacy non-transport-only payload path={output_path}")
                compare_payload = None
            if compare_payload is None:
                print(
                    f"[stageA] transport-only {method.upper()} solve eps={float(epsilon):g}{beta_log}"
                )
                compare_payload = _build_transport_only_compare_payload(
                    method=method,
                    target_vars=target_vars,
                    variable_signatures_by_var=variable_signatures_by_var,
                    site_signatures_by_var=site_signatures_by_var,
                    epsilon=float(epsilon),
                    beta_neural=None if beta_neural is None else float(beta_neural),
                    signature_mode=str(args.signature_mode),
                    calibration_family_weights=calibration_family_weights,
                )
                write_json(output_path, compare_payload)
                report_lines = [
                    f"MCQA PLOT Layer Transport-Only {method.upper()}",
                    f"token_position_id: {str(args.token_position_id)}",
                    f"epsilon: {float(epsilon):g}",
                    f"layers: {list(int(layer) for layer in resolved_layers)}",
                ]
                if beta_neural is not None:
                    report_lines.insert(3, f"uot_beta_neural: {float(beta_neural):g}")
                write_text_report(summary_path, "\n".join(report_lines))
            else:
                print(f"[stageA] reusing existing {method.upper()} payload path={output_path}")
            ot_compare_payloads.append(compare_payload)
            transport_output_paths.append(str(output_path))
    _synchronize_if_cuda(device)
    ot_localization_seconds = float(perf_counter() - ot_localization_start)

    signature_prepare_runtime_seconds = float(
        resolve_recorded_artifact_prepare_seconds(
            prepared_artifacts,
            artifact_prepare_create_seconds=artifact_prepare_create_seconds,
        )
    )
    print(
        "[stageA] selecting one shared PLOT(layer) transport coupling by averaging the "
        "best shortlisted-layer calibration accuracies across abstract variables"
    )
    selected_config = _select_joint_layer_config(
        model=model,
        tokenizer=tokenizer,
        banks_by_split=banks_by_split,
        sites=sites,
        ot_compare_payloads=ot_compare_payloads,
        device=device,
        batch_size=int(args.batch_size),
        signature_mode=str(args.signature_mode),
        calibration_family_weights=calibration_family_weights,
        signature_prepare_runtime_seconds=signature_prepare_runtime_seconds,
        row_top_k=stage_a_row_top_k,
    )
    selected_config = _evaluate_selected_stage_a_holdout(
        model=model,
        tokenizer=tokenizer,
        banks_by_split=banks_by_split,
        sites=sites,
        device=device,
        batch_size=int(args.batch_size),
        selected_config=selected_config,
    )
    best_config_by_method = selected_config.get("best_config_by_method", {})
    if isinstance(best_config_by_method, dict):
        selected_config["best_config_by_method"] = {
            str(method): _evaluate_selected_stage_a_holdout(
                model=model,
                tokenizer=tokenizer,
                banks_by_split=banks_by_split,
                sites=sites,
                device=device,
                batch_size=int(args.batch_size),
                selected_config=method_config,
            )
            for method, method_config in best_config_by_method.items()
            if isinstance(method_config, dict)
        }
    selected_config["row_top_k"] = int(stage_a_row_top_k)
    method_by_var = {
        str(target_var): dict(record)
        for target_var, record in selected_config.get("per_var_records", {}).items()
        if isinstance(record, dict)
    }
    support_start = perf_counter()
    print("[stageA] extracting transport support diagnostics")
    support_by_var = extract_ordered_site_support(
        ot_run_payloads=ot_compare_payloads,
        sites=sites,
        score_slack=float(args.support_score_slack),
    )
    support_extract_seconds = float(perf_counter() - support_start)
    total_seconds = float(perf_counter() - stage_start)
    effective_total_seconds = adjust_runtime_for_cached_signatures(
        wall_runtime_seconds=total_seconds,
        artifact_prepare_load_seconds=artifact_prepare_load_seconds,
        artifact_prepare_create_seconds=artifact_prepare_create_seconds,
        artifact_prepare_recorded_seconds=artifact_prepare_recorded_seconds,
    )
    rankings_by_var = _rank_layers_from_target_row(
        selected_method_by_var=method_by_var,
        runtime_seconds=float(selected_config.get("runtime_with_signatures_seconds", total_seconds)),
    )

    support_path = sweep_root / f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}_support.json"
    summary_path = sweep_root / f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}_summary.txt"
    payload_path = sweep_root / f"mcqa_plot_layer_pos-{str(args.token_position_id)}_sig-{str(args.signature_mode)}.json"
    write_json(support_path, support_by_var)
    write_text_report(
        summary_path,
        _format_summary(
            token_position_id=str(args.token_position_id),
            layers=resolved_layers,
            support_by_var=support_by_var,
            selected_config=selected_config,
        ),
    )
    write_json(
        payload_path,
        {
            "kind": "mcqa_plot_layer",
            "token_position_id": str(args.token_position_id),
            "layers": [int(layer) for layer in resolved_layers],
            "signature_mode": str(args.signature_mode),
            "ot_epsilons": [float(epsilon) for epsilon in ot_epsilons],
            "stage_a_transport_methods": [str(method) for method in stage_a_transport_methods],
            "uot_beta_neurals": [float(beta_neural) for beta_neural in uot_beta_neurals],
            "intervention_strength": float(DEFAULT_STAGE_A_INTERVENTION_STRENGTH),
            "stage_a_row_top_k": int(stage_a_row_top_k),
            "support_score_slack": float(args.support_score_slack),
            "calibration_metric": DEFAULT_CALIBRATION_METRIC,
            "calibration_family_weights": [float(weight) for weight in calibration_family_weights],
            "site_labels": [site.label for site in sites],
            "support_path": str(support_path),
            "support_by_var": support_by_var,
            "rankings_by_var": rankings_by_var,
            "selected_joint_config": selected_config,
            "display_method_by_var": method_by_var,
            "method_by_var": method_by_var,
            "stage_a_transport_method": str(selected_config.get("method", stage_a_transport_methods[0])),
            "ot_output_paths": transport_output_paths,
            "summary_path": str(summary_path),
            "context_timing_seconds": context_timing_seconds,
            "artifact_prepare_recorded_seconds": float(artifact_prepare_recorded_seconds),
            "signature_prepare_runtime_seconds": float(signature_prepare_runtime_seconds),
            "timing_seconds": {
                "t_model_load": float(context_timing_seconds.get("t_model_load", 0.0)),
                "t_data_load": float(context_timing_seconds.get("t_data_load", 0.0)),
                "t_bank_build": float(context_timing_seconds.get("t_bank_build", 0.0)),
                "t_factual_filter": float(context_timing_seconds.get("t_factual_filter", 0.0)),
                "t_context_total_wall": float(context_timing_seconds.get("t_context_total_wall", 0.0)),
                "t_artifact_prepare_load": float(artifact_prepare_load_seconds),
                "t_artifact_prepare_create": float(artifact_prepare_create_seconds),
                "t_artifact_prepare_recorded": float(artifact_prepare_recorded_seconds),
                "t_signature_prepare": float(signature_prepare_runtime_seconds),
                "t_stageA_ot_localization": float(ot_localization_seconds),
                "t_support_extract": float(support_extract_seconds),
                "t_stage_total_wall": float(total_seconds),
            },
            "artifact_cache_hit": bool(prepared_artifacts.get("loaded_from_disk", False)),
            "wall_runtime_seconds": float(total_seconds),
            "localization_runtime_seconds": float(effective_total_seconds),
            "runtime_seconds": float(effective_total_seconds),
        },
    )
    print(f"Wrote PLOT layer payload to {payload_path}")


if __name__ == "__main__":
    main()
