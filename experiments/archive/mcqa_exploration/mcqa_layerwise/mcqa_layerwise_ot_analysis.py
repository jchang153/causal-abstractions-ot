from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from experiments.mcqa import mcqa_run as base_run
from experiments.mcqa.mcqa_experiment.data import canonicalize_target_var
from experiments.mcqa.mcqa_experiment.ot import (
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
from experiments.mcqa.mcqa_experiment.metrics import build_variable_signature
from experiments.mcqa.mcqa_experiment.reporting import write_text_report
from experiments.mcqa.mcqa_experiment.runtime import write_json
from experiments.mcqa.mcqa_experiment.sites import enumerate_residual_sites


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_METHODS = ("ot", "uot")
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_UOT_BETA_NEURALS = (0.1, 0.3, 1.0, 3.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_SINGLE_LAYER_LAMBDAS = (1.0,)
DEFAULT_ROW_TOP_K = 6
DEFAULT_COMPARE_TOP_K = 3


def _synchronize_if_cuda(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",")]
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone MCQA layerwise OT/UOT analysis with whole-layer sites and top-k shortlisted layer calibration."
    )
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
    parser.add_argument("--methods", default="ot,uot")
    parser.add_argument(
        "--skip-transport",
        action="store_true",
        help="Skip OT/UOT transport analysis.",
    )
    parser.add_argument("--ot-epsilons", help="Comma-separated OT/UOT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--uot-beta-neurals", help="Comma-separated neural-side UOT penalties. Default: 0.1,0.3,1,3")
    parser.add_argument(
        "--row-top-k",
        type=int,
        default=DEFAULT_ROW_TOP_K,
        help="Per transport row, shortlist the top-k layers by row mass, then pick the best calibrated layer. Default: 6",
    )
    parser.add_argument(
        "--ot-top-k-values",
        help="Compatibility flag. Layer selection no longer calibrates transport rows over top-k handles.",
    )
    parser.add_argument(
        "--ot-lambdas",
        help="Compatibility flag. Layer selection no longer calibrates transport rows over lambda handles.",
    )
    parser.add_argument(
        "--single-layer-lambdas",
        help="Single fixed intervention strength for per-layer single-site evaluation. Default: 1.0",
    )
    parser.add_argument(
        "--single-layer-results-path",
        help="Compatibility flag. Layerwise selection no longer precomputes all brute-force layers.",
    )
    parser.add_argument(
        "--calibration-family-weights",
        help="Comma-separated family weights in answerPosition,randomLetter,answerPosition_randomLetter order. Default: 1,1,1",
    )
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_single_layer_results_payload(path: Path) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, list[dict[str, object]]]]]:
    payload = _load_existing_payload(path)
    if payload is None:
        raise FileNotFoundError(f"Could not load single-layer results payload from {path}")
    single_layer_by_var = payload.get("single_layer_by_var")
    if not isinstance(single_layer_by_var, dict):
        raise ValueError(f"single_layer_by_var missing or invalid in {path}")
    normalized_single_layer_by_var: dict[str, list[dict[str, object]]] = {}
    for target_var, entries in single_layer_by_var.items():
        if not isinstance(entries, list):
            raise ValueError(f"single_layer_by_var[{target_var!r}] is not a list in {path}")
        normalized_single_layer_by_var[str(target_var)] = [
            dict(entry) for entry in entries if isinstance(entry, dict)
        ]
    single_layer_rankings_by_var = payload.get("single_layer_rankings_by_var")
    if isinstance(single_layer_rankings_by_var, dict):
        normalized_rankings_by_var: dict[str, dict[str, list[dict[str, object]]]] = {}
        for target_var, ranking_payload in single_layer_rankings_by_var.items():
            if not isinstance(ranking_payload, dict):
                continue
            normalized_rankings_by_var[str(target_var)] = {
                str(label): [dict(entry) for entry in entries if isinstance(entry, dict)]
                for label, entries in ranking_payload.items()
                if isinstance(entries, list)
            }
    else:
        normalized_rankings_by_var = _build_single_layer_rankings(normalized_single_layer_by_var)
    return normalized_single_layer_by_var, normalized_rankings_by_var


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


def _format_strength_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def _target_row_ranking(payload: dict[str, object], *, sites) -> list[dict[str, object]]:
    target_row_transport = payload.get("target_normalized_transport", payload.get("target_transport", []))
    if not isinstance(target_row_transport, list) or not target_row_transport:
        return []
    first_entry = target_row_transport[0]
    row_transport = first_entry if isinstance(first_entry, list) else target_row_transport
    ranking = [
        {
            "site_index": int(site_index),
            "site_label": sites[int(site_index)].label,
            "layer": int(sites[int(site_index)].layer),
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


def _score_to_rank(scores_by_layer: dict[int, float]) -> dict[int, int]:
    ordered_layers = sorted(
        scores_by_layer,
        key=lambda layer: (
            float(scores_by_layer[layer]),
            -int(layer),
        ),
        reverse=True,
    )
    return {int(layer): int(index + 1) for index, layer in enumerate(ordered_layers)}


def _spearman_rank_corr(coupling_scores_by_layer: dict[int, float], actual_scores_by_layer: dict[int, float]) -> float | None:
    common_layers = sorted(set(coupling_scores_by_layer) & set(actual_scores_by_layer))
    if len(common_layers) < 2:
        return None
    coupling_ranks = _score_to_rank({layer: coupling_scores_by_layer[layer] for layer in common_layers})
    actual_ranks = _score_to_rank({layer: actual_scores_by_layer[layer] for layer in common_layers})
    n = len(common_layers)
    d2 = sum((coupling_ranks[layer] - actual_ranks[layer]) ** 2 for layer in common_layers)
    return float(1.0 - (6.0 * d2) / (n * (n**2 - 1)))


def _ranking_comparison(
    *,
    coupling_ranking: list[dict[str, object]],
    actual_entries: list[dict[str, object]],
    actual_metric_key: str,
    top_k: int = DEFAULT_COMPARE_TOP_K,
) -> dict[str, object]:
    coupling_order = [int(entry["layer"]) for entry in coupling_ranking]
    coupling_scores_by_layer = {
        int(entry["layer"]): float(entry.get("transport_mass", 0.0))
        for entry in coupling_ranking
    }
    actual_scores_by_layer = {
        int(entry["layer"]): float(entry.get(actual_metric_key, 0.0))
        for entry in actual_entries
    }
    actual_order = [
        int(entry["layer"])
        for entry in sorted(
            actual_entries,
            key=lambda entry: (
                float(entry.get(actual_metric_key, 0.0)),
                -int(entry["layer"]),
            ),
            reverse=True,
        )
    ]
    coupling_rank_by_layer = {int(layer): int(index + 1) for index, layer in enumerate(coupling_order)}
    actual_best_layer = int(actual_order[0]) if actual_order else None
    coupling_top = [int(layer) for layer in coupling_order[:top_k]]
    actual_top = [int(layer) for layer in actual_order[:top_k]]
    overlap = [layer for layer in coupling_top if layer in set(actual_top)]
    rank_deltas = []
    actual_rank_by_layer = {int(layer): int(index + 1) for index, layer in enumerate(actual_order)}
    for layer in sorted(set(coupling_rank_by_layer) & set(actual_rank_by_layer)):
        rank_deltas.append(abs(coupling_rank_by_layer[layer] - actual_rank_by_layer[layer]))
    return {
        "actual_metric": str(actual_metric_key),
        "coupling_top_layers": coupling_top,
        "actual_top_layers": actual_top,
        "top1_match": bool(coupling_top[:1] == actual_top[:1]),
        "top_overlap_count": int(len(overlap)),
        "top_overlap_layers": overlap,
        "best_actual_layer": actual_best_layer,
        "best_actual_layer_coupling_rank": None if actual_best_layer is None else int(coupling_rank_by_layer.get(actual_best_layer, 10**9)),
        "spearman_rank_correlation": _spearman_rank_corr(coupling_scores_by_layer, actual_scores_by_layer),
        "mean_absolute_rank_error": (
            None if not rank_deltas else float(sum(rank_deltas) / len(rank_deltas))
        ),
    }


def _evaluate_single_layer_calibration(
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
    calibration_result, _ = _evaluate_single_site_intervention(
        model=model,
        bank=calibration_bank,
        site=site,
        site_index=int(site_index),
        strength=float(strength),
        batch_size=int(batch_size),
        device=device,
        tokenizer=tokenizer,
        include_details=False,
    )
    runtime_seconds = float(perf_counter() - eval_start)
    calibration_score = _stage_a_calibration_score(
        result=calibration_result,
        calibration_family_weights=calibration_family_weights,
    )
    return {
        "target_var": str(calibration_bank.target_var),
        "layer": int(site.layer),
        "site_label": str(site.label),
        "intervention_strength": float(strength),
        "calibration_score": float(calibration_score),
        "calibration_exact_acc": float(calibration_result.get("exact_acc", 0.0)),
        "runtime_seconds": float(runtime_seconds),
        "calibration_result": dict(calibration_result),
    }


def _rank_single_layer_entries(
    entries: list[dict[str, object]],
    *,
    metric_key: str,
    secondary_metric_key: str,
) -> list[dict[str, object]]:
    ranked = sorted(
        entries,
        key=lambda entry: (
            float(entry.get(metric_key, 0.0)),
            float(entry.get(secondary_metric_key, 0.0)),
            -int(entry["layer"]),
        ),
        reverse=True,
    )
    return [
        {
            **dict(entry),
            "rank": int(rank + 1),
            "ranking_metric": str(metric_key),
        }
        for rank, entry in enumerate(ranked)
    ]


def _build_single_layer_rankings(
    single_layer_by_var: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, list[dict[str, object]]]]:
    rankings: dict[str, dict[str, list[dict[str, object]]]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        entries = [dict(entry) for entry in single_layer_by_var.get(str(target_var), [])]
        rankings[str(target_var)] = {
            "by_test": _rank_single_layer_entries(
                entries,
                metric_key="test_exact_acc",
                secondary_metric_key="calibration_exact_acc",
            ),
            "by_calibration": _rank_single_layer_entries(
                entries,
                metric_key="calibration_exact_acc",
                secondary_metric_key="test_exact_acc",
            ),
        }
    return rankings


def _transport_configs(
    *,
    methods: tuple[str, ...],
    ot_epsilons: tuple[float, ...],
    uot_beta_neurals: tuple[float, ...],
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for method in methods:
        if method == "ot":
            for epsilon in ot_epsilons:
                configs.append(
                    {
                        "method": "ot",
                        "ot_epsilon": float(epsilon),
                        "uot_beta_neural": None,
                    }
                )
        elif method == "uot":
            for epsilon in ot_epsilons:
                for beta_neural in uot_beta_neurals:
                    configs.append(
                        {
                            "method": "uot",
                            "ot_epsilon": float(epsilon),
                            "uot_beta_neural": float(beta_neural),
                        }
                    )
        else:
            raise ValueError(f"Unsupported transport method {method}")
    return configs


def _transport_output_name(config: dict[str, object]) -> str:
    method = str(config["method"])
    epsilon = float(config["ot_epsilon"])
    if method == "ot":
        return f"ot_eps-{epsilon:g}_transport_only.json"
    return f"uot_eps-{epsilon:g}_betan-{float(config['uot_beta_neural']):g}_transport_only.json"


def _transport_config_label(config: dict[str, object]) -> str:
    method = str(config.get("method", "unknown")).upper()
    epsilon = float(config.get("ot_epsilon", 0.0))
    beta_neural = config.get("uot_beta_neural")
    if beta_neural is None:
        return f"{method} eps={epsilon:g}"
    return f"{method} eps={epsilon:g} beta_n={float(beta_neural):g}"


def _single_layer_entry_by_layer(entries: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {
        int(entry["layer"]): entry
        for entry in entries
        if "layer" in entry
    }


def _evaluate_shortlisted_layer_calibrations(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    device,
    batch_size: int,
    calibration_family_weights: tuple[float, ...],
    intervention_strength: float,
    target_var: str,
    ranking: list[dict[str, object]],
    row_top_k: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None, float]:
    shortlisted_entries = ranking[: max(1, int(row_top_k))]
    evaluated_candidates: list[dict[str, object]] = []
    calibration_runtime_seconds = 0.0
    for rank_index, ranking_entry in enumerate(shortlisted_entries):
        site_index = int(ranking_entry.get("site_index", -1))
        if not (0 <= site_index < len(sites)):
            continue
        site = sites[site_index]
        cached = _evaluate_single_layer_calibration(
            model=model,
            tokenizer=tokenizer,
            calibration_bank=banks_by_split["calibration"][str(target_var)],
            site=site,
            site_index=site_index,
            device=device,
            batch_size=int(batch_size),
            strength=float(intervention_strength),
            calibration_family_weights=calibration_family_weights,
        )
        evaluated_candidate = {
            "rank_index": int(rank_index),
            "site_index": int(site_index),
            "layer": int(site.layer),
            "site_label": str(site.label),
            "transport_mass": float(ranking_entry.get("transport_mass", 0.0)),
            "calibration_score": float(cached.get("calibration_score", 0.0)),
            "calibration_exact_acc": float(cached.get("calibration_exact_acc", 0.0)),
            "intervention_strength": float(cached.get("intervention_strength", intervention_strength)),
            "runtime_seconds": float(cached.get("runtime_seconds", 0.0)),
        }
        evaluated_candidates.append(evaluated_candidate)
        calibration_runtime_seconds += float(evaluated_candidate["runtime_seconds"])
    candidate_lookup = _single_layer_entry_by_layer(evaluated_candidates)
    shortlisted_candidates, best_candidate = _select_best_shortlisted_single_layer(
        ranking=ranking,
        candidate_lookup=candidate_lookup,
        row_top_k=row_top_k,
    )
    return shortlisted_candidates, best_candidate, float(calibration_runtime_seconds)


def _select_best_shortlisted_single_layer(
    *,
    ranking: list[dict[str, object]],
    candidate_lookup: dict[int, dict[str, object]],
    row_top_k: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    shortlisted_candidates: list[dict[str, object]] = []
    for rank_index, ranking_entry in enumerate(ranking[: max(1, int(row_top_k))]):
        layer = int(ranking_entry["layer"])
        candidate_entry = candidate_lookup.get(layer)
        if not isinstance(candidate_entry, dict):
            continue
        shortlisted_candidates.append(
            {
                "rank_index": int(rank_index),
                "layer": int(layer),
                "site_label": str(ranking_entry.get("site_label", candidate_entry.get("site_label", ""))),
                "transport_mass": float(ranking_entry.get("transport_mass", 0.0)),
                "calibration_score": float(candidate_entry.get("calibration_score", 0.0)),
                "calibration_exact_acc": float(candidate_entry.get("calibration_exact_acc", 0.0)),
                "intervention_strength": float(
                    candidate_entry.get(
                        "intervention_strength",
                        candidate_entry.get("selected_lambda", 0.0),
                    )
                ),
                "runtime_seconds": float(candidate_entry.get("runtime_seconds", 0.0)),
            }
        )
    if not shortlisted_candidates:
        return [], None
    best_candidate = max(
        shortlisted_candidates,
        key=lambda entry: (
            float(entry.get("calibration_score", 0.0)),
            float(entry.get("calibration_exact_acc", 0.0)),
            float(entry.get("transport_mass", 0.0)),
            -int(entry.get("rank_index", 10**9)),
        ),
    )
    return shortlisted_candidates, best_candidate


def _run_transport_sweeps(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    device,
    batch_size: int,
    signature_mode: str,
    prepared_artifacts: dict[str, object],
    output_root: Path,
    methods: tuple[str, ...],
    ot_epsilons: tuple[float, ...],
    uot_beta_neurals: tuple[float, ...],
) -> list[dict[str, object]]:
    configs = _transport_configs(
        methods=methods,
        ot_epsilons=ot_epsilons,
        uot_beta_neurals=uot_beta_neurals,
    )
    runs: list[dict[str, object]] = []
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    variable_signatures_by_var = {
        str(target_var): build_variable_signature(
            banks_by_split["train"][str(target_var)],
            str(signature_mode),
        )
        for target_var in target_vars
    }
    site_signatures_by_var = {
        str(target_var): tensor
        for target_var, tensor in prepared_artifacts["site_signatures_by_var"].items()
    }
    progress = tqdm(total=len(configs), desc="Transport config sweep", leave=True) if tqdm is not None else None
    for config_index, transport_config in enumerate(configs, start=1):
        method = str(transport_config["method"])
        epsilon = float(transport_config["ot_epsilon"])
        beta_neural = transport_config.get("uot_beta_neural")
        config_label = _transport_config_label(transport_config)
        if progress is not None:
            progress.set_description(f"Transport {config_index}/{len(configs)} {config_label}")
        output_path = output_root / _transport_output_name(transport_config)
        payload = _load_existing_payload(output_path)
        print(f"[layerwise] transport_config_start {config_label}")
        if payload is None:
            print(
                f"[layerwise] transport-only method={method} epsilon={epsilon:g}"
                + ("" if beta_neural is None else f" uot_beta_neural={float(beta_neural):g}")
            )
            config = OTConfig(
                method=method,
                batch_size=int(batch_size),
                epsilon=float(epsilon),
                uot_beta_neural=1.0 if beta_neural is None else float(beta_neural),
                signature_mode=str(signature_mode),
                top_k_values=(1,),
                lambda_values=(1.0,),
                selection_verbose=False,
                source_target_vars=target_vars,
                calibration_metric=DEFAULT_CALIBRATION_METRIC,
                calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
            )
            solve_start = perf_counter()
            if method == "ot":
                transport, transport_meta = solve_ot_transport(
                    variable_signatures_by_var,
                    site_signatures_by_var,
                    config,
                )
            elif method == "uot":
                transport, transport_meta = solve_uot_transport(
                    variable_signatures_by_var,
                    site_signatures_by_var,
                    config,
                )
            else:
                raise ValueError(f"Unsupported transport method {method}")
            transport_runtime_seconds = float(perf_counter() - solve_start)
            normalized_transport = normalize_transport_rows(transport)
            per_var_payloads: dict[str, dict[str, object]] = {}
            for target_row_index, target_var in enumerate(target_vars):
                target_transport = transport[target_row_index : target_row_index + 1]
                target_normalized_transport = normalized_transport[target_row_index : target_row_index + 1]
                per_var_payloads[str(target_var)] = {
                    "kind": "mcqa_layerwise_transport_target_row",
                    "method": method,
                    "target_var": str(target_var),
                    "ot_epsilon": float(epsilon),
                    "uot_beta_neural": None if beta_neural is None else float(beta_neural),
                    "target_row_index": int(target_row_index),
                    "target_transport": target_transport.tolist(),
                    "target_normalized_transport": target_normalized_transport.tolist(),
                    "transport_meta": dict(transport_meta),
                    "runtime_seconds": float(transport_runtime_seconds),
                    "selection_basis": "coupling_row_topk_then_best_calibrated_single_layer_lookup",
                }
            payload = {
                "kind": "mcqa_layerwise_transport_analysis_run_transport_only",
                "method": method,
                "ot_epsilon": float(epsilon),
                "uot_beta_neural": None if beta_neural is None else float(beta_neural),
                "target_vars": list(target_vars),
                "selection_basis": "coupling_row_topk_then_best_calibrated_single_layer_lookup",
                "transport_meta": dict(transport_meta),
                "runtime_seconds": float(transport_runtime_seconds),
                "by_var": per_var_payloads,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, payload)
            payload_source = "computed"
        else:
            payload_source = "cache"
        print(
            f"[layerwise] transport_config_done {config_label} "
            f"transport_runtime={float(payload.get('runtime_seconds', 0.0)):.2f}s "
            f"source={payload_source}"
        )
        runs.append(payload)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return runs


def _summarize_transport_runs(
    *,
    transport_runs: list[dict[str, object]],
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    device,
    batch_size: int,
    calibration_family_weights: tuple[float, ...],
    intervention_strength: float,
    row_top_k: int,
    signature_prepare_runtime_seconds: float,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    progress = tqdm(total=len(transport_runs), desc="Shortlist calibration sweep", leave=True) if tqdm is not None else None
    for config_index, run_payload in enumerate(transport_runs, start=1):
        method = str(run_payload.get("method"))
        epsilon = float(run_payload.get("ot_epsilon", 0.0))
        beta_neural = run_payload.get("uot_beta_neural")
        config_label = _transport_config_label(run_payload)
        if progress is not None:
            progress.set_description(f"Shortlist {config_index}/{len(transport_runs)} {config_label}")
        by_var = run_payload.get("by_var", {})
        run_summary: dict[str, object] = {
            "method": method,
            "ot_epsilon": float(epsilon),
            "uot_beta_neural": None if beta_neural is None else float(beta_neural),
            "by_var": {},
        }
        shortlist_union_layers: set[int] = set()
        per_var_runtime_by_var: dict[str, float] = {}
        selected_scores: list[float] = []
        selected_exact_accs: list[float] = []
        for target_var in DEFAULT_TARGET_VARS:
            payload = by_var.get(str(target_var)) if isinstance(by_var, dict) else None
            if not isinstance(payload, dict):
                continue
            ranking = _target_row_ranking(payload, sites=sites)
            argmax_entry = ranking[0] if ranking else None
            shortlisted_candidates, selected_shortlist_candidate, calibration_runtime_seconds = _evaluate_shortlisted_layer_calibrations(
                model=model,
                tokenizer=tokenizer,
                banks_by_split=banks_by_split,
                sites=sites,
                device=device,
                batch_size=int(batch_size),
                calibration_family_weights=calibration_family_weights,
                intervention_strength=float(intervention_strength),
                target_var=str(target_var),
                ranking=ranking,
                row_top_k=row_top_k,
            )
            shortlist_union_layers.update(
                int(candidate.get("layer", -1))
                for candidate in shortlisted_candidates
                if int(candidate.get("layer", -1)) >= 0
            )
            per_var_runtime_by_var[str(target_var)] = float(calibration_runtime_seconds)
            if isinstance(selected_shortlist_candidate, dict):
                selected_scores.append(float(selected_shortlist_candidate.get("calibration_score", 0.0)))
                selected_exact_accs.append(float(selected_shortlist_candidate.get("calibration_exact_acc", 0.0)))
            run_summary["by_var"][str(target_var)] = {
                "coupling_top_layers": [
                    {
                        "layer": int(entry["layer"]),
                        "site_label": entry.get("site_label"),
                        "transport_mass": float(entry.get("transport_mass", 0.0)),
                    }
                    for entry in ranking[:10]
                ],
                "row_top_k": int(row_top_k),
                "coupling_argmax_layer": None if not isinstance(argmax_entry, dict) else int(argmax_entry["layer"]),
                "coupling_argmax_site_label": None if not isinstance(argmax_entry, dict) else str(argmax_entry.get("site_label")),
                "coupling_argmax_transport_mass": 0.0 if not isinstance(argmax_entry, dict) else float(argmax_entry.get("transport_mass", 0.0)),
                "shortlisted_candidates": shortlisted_candidates,
                "selected_shortlist_layer": None if not isinstance(selected_shortlist_candidate, dict) else int(selected_shortlist_candidate["layer"]),
                "selected_shortlist_site_label": None if not isinstance(selected_shortlist_candidate, dict) else str(selected_shortlist_candidate.get("site_label")),
                "selected_shortlist_rank": None if not isinstance(selected_shortlist_candidate, dict) else int(selected_shortlist_candidate.get("rank_index", 0)) + 1,
                "selected_shortlist_transport_mass": 0.0 if not isinstance(selected_shortlist_candidate, dict) else float(selected_shortlist_candidate.get("transport_mass", 0.0)),
                "selected_shortlist_layer_calibration_exact_acc": 0.0 if not isinstance(selected_shortlist_candidate, dict) else float(selected_shortlist_candidate.get("calibration_exact_acc", 0.0)),
                "selected_shortlist_layer_calibration_score": 0.0 if not isinstance(selected_shortlist_candidate, dict) else float(selected_shortlist_candidate.get("calibration_score", 0.0)),
                "selected_shortlist_layer_strength": 0.0 if not isinstance(selected_shortlist_candidate, dict) else float(selected_shortlist_candidate.get("intervention_strength", 0.0)),
                "selected_transport_method_calibration_exact_acc": 0.0 if not isinstance(selected_shortlist_candidate, dict) else float(selected_shortlist_candidate.get("calibration_exact_acc", 0.0)),
                "selected_transport_method_calibration_score": 0.0 if not isinstance(selected_shortlist_candidate, dict) else float(selected_shortlist_candidate.get("calibration_score", 0.0)),
                "selection_basis": "coupling_row_topk_then_best_calibrated_single_layer_lookup",
                "calibration_runtime_seconds": float(calibration_runtime_seconds),
                "matched_mass": float(payload.get("transport_meta", {}).get("matched_mass", 1.0)),
                "transport_meta": dict(payload.get("transport_meta", {})),
            }
        run_summary["shortlist_union_layers"] = sorted(int(layer) for layer in shortlist_union_layers)
        run_summary["shortlist_union_size"] = int(len(shortlist_union_layers))
        run_summary["per_var_calibration_runtime_seconds"] = per_var_runtime_by_var
        run_summary["shortlist_calibration_runtime_seconds"] = float(sum(per_var_runtime_by_var.values()))
        run_summary["signature_prepare_runtime_seconds"] = float(signature_prepare_runtime_seconds)
        run_summary["transport_solve_runtime_seconds"] = float(run_payload.get("transport_meta", {}).get("runtime_seconds", run_payload.get("runtime_seconds", 0.0)))
        run_summary["runtime_with_signatures_seconds"] = float(
            signature_prepare_runtime_seconds
            + float(run_summary.get("transport_solve_runtime_seconds", 0.0))
            + float(run_summary.get("shortlist_calibration_runtime_seconds", 0.0))
        )
        run_summary["mean_selected_calibration_score"] = (
            float(sum(selected_scores) / len(selected_scores)) if selected_scores else 0.0
        )
        run_summary["mean_selected_calibration_exact_acc"] = (
            float(sum(selected_exact_accs) / len(selected_exact_accs)) if selected_exact_accs else 0.0
        )
        chosen_layers = {
            str(target_var): int(entry.get("selected_shortlist_layer", -1))
            for target_var in DEFAULT_TARGET_VARS
            for entry in [run_summary.get("by_var", {}).get(str(target_var))]
            if isinstance(entry, dict)
        }
        print(
            f"[layerwise] config_eval_done {config_label} "
            f"transport_runtime={float(run_summary.get('transport_solve_runtime_seconds', 0.0)):.2f}s "
            f"shortlist_runtime={float(run_summary.get('shortlist_calibration_runtime_seconds', 0.0)):.2f}s "
            f"total_config_runtime={float(run_summary.get('runtime_with_signatures_seconds', 0.0)):.2f}s "
            f"avg_cal_score={float(run_summary.get('mean_selected_calibration_score', 0.0)):.4f} "
            f"chosen_layers={chosen_layers}"
        )
        summaries.append(run_summary)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return summaries


def _format_transport_summary(transport_summaries: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for run_summary in transport_summaries:
        method = str(run_summary.get("method"))
        epsilon = float(run_summary.get("ot_epsilon", 0.0))
        beta_neural = run_summary.get("uot_beta_neural")
        header = f"[{method.upper()} eps={epsilon:g}"
        if beta_neural is not None:
            header += f" beta_n={float(beta_neural):g}"
        header += "]"
        lines.append(header)
        lines.append(
            "  "
            + " ".join(
                [
                    f"avg_cal_score={float(run_summary.get('mean_selected_calibration_score', 0.0)):.4f}",
                    f"avg_cal_exact={float(run_summary.get('mean_selected_calibration_exact_acc', 0.0)):.4f}",
                    f"transport_runtime={float(run_summary.get('transport_solve_runtime_seconds', 0.0)):.2f}s",
                    f"shortlist_runtime={float(run_summary.get('shortlist_calibration_runtime_seconds', 0.0)):.2f}s",
                    f"total_runtime={float(run_summary.get('runtime_with_signatures_seconds', 0.0)):.2f}s",
                    f"shortlist_union_layers={run_summary.get('shortlist_union_layers', [])}",
                ]
            )
        )
        by_var = run_summary.get("by_var", {})
        for target_var in DEFAULT_TARGET_VARS:
            entry = by_var.get(str(target_var)) if isinstance(by_var, dict) else None
            if not isinstance(entry, dict):
                continue
            lines.append(f"  {target_var}:")
            top_layers = entry.get("coupling_top_layers", [])
            lines.append(
                f"    coupling_top5={[item.get('layer') for item in top_layers[:5]]} "
                f"matched_mass={float(entry.get('matched_mass', 1.0)):.4f} "
                f"argmax_layer={entry.get('coupling_argmax_layer')} "
                f"selected_layer={entry.get('selected_shortlist_layer')} "
                f"selected_rank={entry.get('selected_shortlist_rank')} "
                f"strength={float(entry.get('selected_shortlist_layer_strength', 0.0)):g} "
                f"cal_score={float(entry.get('selected_shortlist_layer_calibration_score', 0.0)):.4f} "
                f"cal_exact={float(entry.get('selected_shortlist_layer_calibration_exact_acc', 0.0)):.4f} "
                f"row_runtime={float(entry.get('calibration_runtime_seconds', 0.0)):.2f}s"
            )
            shortlisted_candidates = entry.get("shortlisted_candidates", [])
            if isinstance(shortlisted_candidates, list) and shortlisted_candidates:
                lines.append(
                    "    shortlisted_topk="
                    + str(
                        [
                            {
                                "layer": int(candidate.get("layer", -1)),
                                "rank": int(candidate.get("rank_index", 0)) + 1,
                                "mass": round(float(candidate.get("transport_mass", 0.0)), 4),
                                "cal_score": round(float(candidate.get("calibration_score", 0.0)), 4),
                                "cal": round(float(candidate.get("calibration_exact_acc", 0.0)), 4),
                            }
                            for candidate in shortlisted_candidates
                        ]
                    )
                )
        lines.append("")
    return lines


def _best_transport_summary(transport_summaries: list[dict[str, object]]) -> dict[str, object] | None:
    if not transport_summaries:
        return None
    return max(
        transport_summaries,
        key=lambda summary: (
            float(summary.get("mean_selected_calibration_score", 0.0)),
            float(summary.get("mean_selected_calibration_exact_acc", 0.0)),
        ),
    )


def _format_summary(
    *,
    token_position_id: str,
    layers: tuple[int, ...],
    row_top_k: int,
    transport_summaries: list[dict[str, object]],
) -> str:
    lines = [
        "MCQA Layerwise OT/UOT Analysis",
        f"token_position_id: {token_position_id}",
        f"layers: {list(int(layer) for layer in layers)}",
        "",
        "This analysis compares joint 26-layer coupling rankings against per-layer single-site intervention accuracies.",
        (
            f"Reported method accuracy does not calibrate transport rows themselves. For each target-variable row, "
            f"it shortlists the top {int(row_top_k)} layers by row mass, then selects the best calibrated "
            "fixed-strength single-layer intervention within that shortlist."
        ),
        "",
    ]
    best_summary = _best_transport_summary(transport_summaries)
    if isinstance(best_summary, dict):
        lines.extend(
            [
                "[best_selected_coupling]",
                f"method={best_summary.get('method')}",
                f"eps={float(best_summary.get('ot_epsilon', 0.0)):g}",
                (
                    f"beta_n={float(best_summary.get('uot_beta_neural')):g}"
                    if best_summary.get("uot_beta_neural") is not None else "beta_n=NA"
                ),
                f"row_top_k={int(row_top_k)}",
                f"avg_cal_score={float(best_summary.get('mean_selected_calibration_score', 0.0)):.4f}",
                f"avg_cal_exact={float(best_summary.get('mean_selected_calibration_exact_acc', 0.0)):.4f}",
                f"total_runtime={float(best_summary.get('runtime_with_signatures_seconds', 0.0)):.2f}s",
                f"shortlist_union_layers={best_summary.get('shortlist_union_layers', [])}",
            ]
        )
        by_var = best_summary.get("by_var", {})
        for target_var in DEFAULT_TARGET_VARS:
            entry = by_var.get(str(target_var)) if isinstance(by_var, dict) else None
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"{target_var}: chosen_layer={entry.get('selected_shortlist_layer')} "
                f"argmax_layer={entry.get('coupling_argmax_layer')} "
                f"cal_score={float(entry.get('selected_shortlist_layer_calibration_score', 0.0)):.4f} "
                f"cal_exact={float(entry.get('selected_shortlist_layer_calibration_exact_acc', 0.0)):.4f}"
            )
        lines.append("")
    lines.extend(_format_transport_summary(transport_summaries))
    return "\n".join(lines)


def main() -> None:
    overall_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_layerwise_ot_analysis"
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
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    sites = enumerate_residual_sites(
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        token_position_ids=token_position_ids,
        resolution=None,
        layers=resolved_layers,
        selected_token_position_ids=(str(args.token_position_id),),
    )
    methods = tuple(_parse_csv_strings(args.methods) or list(DEFAULT_METHODS))
    if bool(args.skip_transport):
        methods = ()
    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))
    uot_beta_neurals = tuple(_parse_csv_floats(args.uot_beta_neurals) or list(DEFAULT_UOT_BETA_NEURALS))
    row_top_k = max(1, int(args.row_top_k))
    calibration_family_weights = tuple(
        _parse_csv_floats(args.calibration_family_weights) or list(DEFAULT_CALIBRATION_FAMILY_WEIGHTS)
    )
    single_layer_lambdas = tuple(_parse_csv_floats(args.single_layer_lambdas) or list(DEFAULT_SINGLE_LAYER_LAMBDAS))
    if len(single_layer_lambdas) != 1:
        raise ValueError(
            "Shortlisted layer evaluation uses one fixed intervention strength. "
            f"Received {list(single_layer_lambdas)}"
        )
    intervention_strength = float(single_layer_lambdas[0])
    if intervention_strength != 1.0:
        print(
            f"[layerwise] warning fixed shortlisted-layer intervention strength is {intervention_strength:g}; "
            "the intended Stage A compatibility setting is 1.0"
        )
    if args.single_layer_results_path:
        print(
            "[layerwise] ignoring compatibility --single-layer-results-path; "
            "this run evaluates only shortlisted layers per coupling"
        )

    print(
        f"[layerwise] start token_position={str(args.token_position_id)} "
        f"layers={len(sites)} methods={list(methods)} row_top_k={int(row_top_k)} "
        f"intervention_strength={intervention_strength:g}"
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
        print(f"[layerwise] signature cache miss path={cache_path}")
        artifact_prepare_start = perf_counter()
        prepared_artifacts = prepare_alignment_artifacts(
            model=model,
            fit_banks_by_var=train_banks,
            sites=sites,
            device=device,
            config=OTConfig(
                method="ot",
                batch_size=int(args.batch_size),
                epsilon=float(ot_epsilons[0]),
                signature_mode=str(args.signature_mode),
                top_k_values=(1,),
                lambda_values=(1.0,),
                selection_verbose=False,
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
        print(f"[layerwise] signature cache saved path={cache_path}")
    else:
        print(
            f"[layerwise] signature cache hit path={cache_path} "
            f"loaded_from_disk={bool(prepared_artifacts.get('loaded_from_disk', False))}"
        )
    _synchronize_if_cuda(device)
    artifact_prepare_recorded_seconds = resolve_recorded_artifact_prepare_seconds(
        prepared_artifacts,
        artifact_prepare_create_seconds=artifact_prepare_create_seconds,
    )
    signature_prepare_runtime_seconds = float(artifact_prepare_recorded_seconds)

    transport_root = sweep_root / "transport"
    transport_runs: list[dict[str, object]] = []
    transport_summaries: list[dict[str, object]] = []
    if methods:
        transport_runs = _run_transport_sweeps(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=sites,
            device=device,
            batch_size=int(args.batch_size),
            signature_mode=str(args.signature_mode),
            prepared_artifacts=prepared_artifacts,
            output_root=transport_root,
            methods=methods,
            ot_epsilons=ot_epsilons,
            uot_beta_neurals=uot_beta_neurals,
        )
        transport_summaries = _summarize_transport_runs(
            transport_runs=transport_runs,
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            sites=sites,
            device=device,
            batch_size=int(args.batch_size),
            calibration_family_weights=calibration_family_weights,
            intervention_strength=intervention_strength,
            row_top_k=row_top_k,
            signature_prepare_runtime_seconds=signature_prepare_runtime_seconds,
        )

    summary_text = _format_summary(
        token_position_id=str(args.token_position_id),
        layers=resolved_layers,
        row_top_k=row_top_k,
        transport_summaries=transport_summaries,
    )
    summary_path = sweep_root / "layerwise_ot_summary.txt"
    payload_path = sweep_root / "layerwise_ot_results.json"
    best_transport_summary = _best_transport_summary(transport_summaries)
    total_wall_seconds = float(perf_counter() - overall_start)
    effective_total_seconds = adjust_runtime_for_cached_signatures(
        wall_runtime_seconds=total_wall_seconds,
        artifact_prepare_load_seconds=artifact_prepare_load_seconds,
        artifact_prepare_create_seconds=artifact_prepare_create_seconds,
        artifact_prepare_recorded_seconds=artifact_prepare_recorded_seconds,
    )
    write_text_report(summary_path, summary_text)
    write_json(
        payload_path,
        {
            "kind": "mcqa_layerwise_ot_analysis",
            "token_position_id": str(args.token_position_id),
            "layers": [int(layer) for layer in resolved_layers],
            "target_vars": list(target_vars),
            "signature_mode": str(args.signature_mode),
            "methods": list(methods),
            "ot_epsilons": [float(value) for value in ot_epsilons],
            "uot_beta_neurals": [float(value) for value in uot_beta_neurals],
            "row_top_k": int(row_top_k),
            "calibration_family_weights": [float(value) for value in calibration_family_weights],
            "single_layer_lambdas": [float(value) for value in single_layer_lambdas],
            "site_labels": [site.label for site in sites],
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
                "t_total_wall": float(total_wall_seconds),
                "t_total_effective": float(effective_total_seconds),
            },
            "artifact_cache_hit": bool(prepared_artifacts.get("loaded_from_disk", False)),
            "wall_runtime_seconds": float(total_wall_seconds),
            "runtime_seconds": float(effective_total_seconds),
            "transport_runs": transport_runs,
            "transport_summaries": transport_summaries,
            "best_transport_summary": best_transport_summary,
            "data": data_metadata,
            "summary_path": str(summary_path),
        },
    )
    if isinstance(best_transport_summary, dict):
        chosen_layers = {
            str(target_var): int(entry.get("selected_shortlist_layer", -1))
            for target_var in DEFAULT_TARGET_VARS
            for entry in [best_transport_summary.get("by_var", {}).get(str(target_var))]
            if isinstance(entry, dict)
        }
        print(
            "[layerwise] best_selected_coupling "
            + " ".join(
                [
                    f"method={best_transport_summary.get('method')}",
                    f"epsilon={float(best_transport_summary.get('ot_epsilon', 0.0)):g}",
                    (
                        f"beta_n={float(best_transport_summary.get('uot_beta_neural')):g}"
                        if best_transport_summary.get("uot_beta_neural") is not None else "beta_n=NA"
                    ),
                    f"avg_cal_score={float(best_transport_summary.get('mean_selected_calibration_score', 0.0)):.4f}",
                    f"avg_cal_exact={float(best_transport_summary.get('mean_selected_calibration_exact_acc', 0.0)):.4f}",
                    f"chosen_layers={chosen_layers}",
                ]
            )
        )
    print(f"Wrote layerwise OT analysis payload to {payload_path}")
    print(f"Wrote layerwise OT analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
