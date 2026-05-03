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

import mcqa_run as base_run
from mcqa_experiment.data import canonicalize_target_var
from mcqa_experiment.ot import (
    OTConfig,
    load_prepared_alignment_artifacts,
    normalize_transport_rows,
    prepare_alignment_artifacts,
    run_bruteforce_site_pipeline,
    save_prepared_alignment_artifacts,
    solve_ot_transport,
    solve_uot_transport,
)
from mcqa_experiment.metrics import build_variable_signature
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import enumerate_residual_sites


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
        description="Standalone MCQA layerwise OT/UOT analysis with whole-layer sites and per-layer brute-force comparisons."
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
        help="Only run the full-strength single-layer brute-force sweep; skip OT/UOT transport analysis.",
    )
    parser.add_argument("--ot-epsilons", help="Comma-separated OT/UOT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--uot-beta-neurals", help="Comma-separated neural-side UOT penalties. Default: 0.1,0.3,1,3")
    parser.add_argument("--ot-top-k-values", help="Comma-separated OT/UOT top-k calibration grid.")
    parser.add_argument("--ot-lambdas", help="Comma-separated OT/UOT lambda calibration grid.")
    parser.add_argument(
        "--single-layer-lambdas",
        help="Single fixed intervention strength for per-layer single-site evaluation. Default: 1.0",
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


def _evaluate_all_layers(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    device,
    batch_size: int,
    signature_mode: str,
    lambda_values: tuple[float, ...],
    output_root: Path,
) -> dict[str, list[dict[str, object]]]:
    results_by_var: dict[str, list[dict[str, object]]] = {}
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    fixed_strength = float(lambda_values[0])
    strength_tag = _format_strength_tag(fixed_strength)
    for target_var in target_vars:
        entries: list[dict[str, object]] = []
        site_iterator = sites
        if tqdm is not None:
            site_iterator = tqdm(
                sites,
                desc=f"Single-layer eval ({target_var})",
                leave=False,
            )
        print(
            f"[single-layer] target={target_var} evaluating {len(sites)} whole-layer sites "
            f"with fixed intervention strength={fixed_strength:g}"
        )
        for site in site_iterator:
            output_path = (
                output_root
                / target_var
                / f"strength_{strength_tag}"
                / f"layer_{int(site.layer):02d}.json"
            )
            payload = _load_existing_payload(output_path)
            if payload is None:
                payload = run_bruteforce_site_pipeline(
                    model=model,
                    calibration_bank=banks_by_split["calibration"][str(target_var)],
                    holdout_bank=banks_by_split["test"][str(target_var)],
                    sites=[site],
                    device=device,
                    tokenizer=tokenizer,
                    config=OTConfig(
                        method="bruteforce",
                        batch_size=int(batch_size),
                        epsilon=1.0,
                        signature_mode=str(signature_mode),
                        top_k_values=(1,),
                        lambda_values=lambda_values,
                        selection_verbose=False,
                        source_target_vars=target_vars,
                        calibration_metric=DEFAULT_CALIBRATION_METRIC,
                        calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                        lambda_values_by_var={str(target_var): lambda_values},
                    ),
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(output_path, payload)
            result = payload.get("results", [{}])[0]
            calibration_result = payload.get("selected_calibration_result", {})
            selected_hyperparameters = dict(payload.get("selected_hyperparameters", {}))
            entries.append(
                {
                    "layer": int(site.layer),
                    "site_label": site.label,
                    "calibration_exact_acc": float(
                        result.get("calibration_exact_acc", calibration_result.get("exact_acc", 0.0))
                    ),
                    "calibration_score": float(
                        result.get("selection_score", calibration_result.get("exact_acc", 0.0))
                    ),
                    "test_exact_acc": float(result.get("exact_acc", 0.0)),
                    "selected_lambda": fixed_strength,
                    "intervention_strength": fixed_strength,
                    "output_path": str(output_path),
                }
            )
        results_by_var[str(target_var)] = sorted(entries, key=lambda entry: int(entry["layer"]))
    return results_by_var


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


def _single_layer_entry_by_layer(entries: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {
        int(entry["layer"]): entry
        for entry in entries
        if "layer" in entry
    }


def _run_transport_sweeps(
    *,
    model,
    tokenizer,
    banks_by_split: dict[str, dict[str, object]],
    sites,
    device,
    batch_size: int,
    signature_mode: str,
    top_k_values: tuple[int, ...],
    lambda_values: tuple[float, ...],
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
    iterator = configs
    if tqdm is not None:
        iterator = tqdm(configs, desc="Transport sweeps", leave=False)
    for transport_config in iterator:
        method = str(transport_config["method"])
        epsilon = float(transport_config["ot_epsilon"])
        beta_neural = transport_config.get("uot_beta_neural")
        output_path = output_root / _transport_output_name(transport_config)
        payload = _load_existing_payload(output_path)
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
                top_k_values=top_k_values,
                lambda_values=lambda_values,
                selection_verbose=True,
                source_target_vars=target_vars,
                calibration_metric=DEFAULT_CALIBRATION_METRIC,
                calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                top_k_values_by_var={str(target_var): top_k_values for target_var in target_vars},
                lambda_values_by_var={str(target_var): lambda_values for target_var in target_vars},
            )
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
                    "selection_basis": "coupling_row_argmax_then_fixed_strength_single_layer_lookup",
                }
            payload = {
                "kind": "mcqa_layerwise_transport_analysis_run_transport_only",
                "method": method,
                "ot_epsilon": float(epsilon),
                "uot_beta_neural": None if beta_neural is None else float(beta_neural),
                "target_vars": list(target_vars),
                "selection_basis": "coupling_row_argmax_then_fixed_strength_single_layer_lookup",
                "transport_meta": dict(transport_meta),
                "by_var": per_var_payloads,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, payload)
        runs.append(payload)
    return runs


def _summarize_transport_runs(
    *,
    transport_runs: list[dict[str, object]],
    single_layer_by_var: dict[str, list[dict[str, object]]],
    sites,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for run_payload in transport_runs:
        method = str(run_payload.get("method"))
        epsilon = float(run_payload.get("ot_epsilon", 0.0))
        beta_neural = run_payload.get("uot_beta_neural")
        by_var = run_payload.get("by_var", {})
        run_summary: dict[str, object] = {
            "method": method,
            "ot_epsilon": float(epsilon),
            "uot_beta_neural": None if beta_neural is None else float(beta_neural),
            "by_var": {},
        }
        for target_var in DEFAULT_TARGET_VARS:
            payload = by_var.get(str(target_var)) if isinstance(by_var, dict) else None
            if not isinstance(payload, dict):
                continue
            ranking = _target_row_ranking(payload, sites=sites)
            argmax_entry = ranking[0] if ranking else None
            single_layer_lookup = _single_layer_entry_by_layer(single_layer_by_var.get(str(target_var), []))
            selected_single_layer = (
                single_layer_lookup.get(int(argmax_entry["layer"]))
                if isinstance(argmax_entry, dict) and "layer" in argmax_entry else None
            )
            calibration_compare = _ranking_comparison(
                coupling_ranking=ranking,
                actual_entries=single_layer_by_var.get(str(target_var), []),
                actual_metric_key="calibration_exact_acc",
            )
            test_compare = _ranking_comparison(
                coupling_ranking=ranking,
                actual_entries=single_layer_by_var.get(str(target_var), []),
                actual_metric_key="test_exact_acc",
            )
            run_summary["by_var"][str(target_var)] = {
                "coupling_top_layers": [
                    {
                        "layer": int(entry["layer"]),
                        "site_label": entry.get("site_label"),
                        "transport_mass": float(entry.get("transport_mass", 0.0)),
                    }
                    for entry in ranking[:10]
                ],
                "selected_coupling_layer": None if not isinstance(argmax_entry, dict) else int(argmax_entry["layer"]),
                "selected_coupling_site_label": None if not isinstance(argmax_entry, dict) else str(argmax_entry.get("site_label")),
                "selected_coupling_transport_mass": 0.0 if not isinstance(argmax_entry, dict) else float(argmax_entry.get("transport_mass", 0.0)),
                "selected_coupling_layer_test_exact_acc": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("test_exact_acc", 0.0)),
                "selected_coupling_layer_calibration_exact_acc": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("calibration_exact_acc", 0.0)),
                "selected_coupling_layer_calibration_score": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("calibration_score", 0.0)),
                "selected_coupling_layer_lambda": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("selected_lambda", 0.0)),
                "selected_coupling_layer_output_path": None if not isinstance(selected_single_layer, dict) else str(selected_single_layer.get("output_path", "")),
                "selected_transport_method_test_exact_acc": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("test_exact_acc", 0.0)),
                "selected_transport_method_calibration_exact_acc": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("calibration_exact_acc", 0.0)),
                "selected_transport_method_calibration_score": 0.0 if not isinstance(selected_single_layer, dict) else float(selected_single_layer.get("calibration_score", 0.0)),
                "matched_mass": float(payload.get("transport_meta", {}).get("matched_mass", 1.0)),
                "transport_meta": dict(payload.get("transport_meta", {})),
                "calibration_ranking_comparison": calibration_compare,
                "test_ranking_comparison": test_compare,
            }
        summaries.append(run_summary)
    return summaries


def _format_single_layer_summary(
    single_layer_by_var: dict[str, list[dict[str, object]]],
    single_layer_rankings_by_var: dict[str, dict[str, list[dict[str, object]]]],
) -> list[str]:
    lines: list[str] = []
    for target_var in DEFAULT_TARGET_VARS:
        entries = single_layer_by_var.get(str(target_var), [])
        lines.append(f"[single-layer {target_var}]")
        by_test = single_layer_rankings_by_var.get(str(target_var), {}).get("by_test", [])
        by_calibration = single_layer_rankings_by_var.get(str(target_var), {}).get("by_calibration", [])
        if entries and by_test:
            best_test = by_test[0]
            lines.append(
                "  best_by_test: "
                + " ".join(
                    [
                        f"layer={int(best_test['layer'])}",
                        f"cal={float(best_test.get('calibration_exact_acc', 0.0)):.4f}",
                        f"test={float(best_test.get('test_exact_acc', 0.0)):.4f}",
                        f"strength={float(best_test.get('intervention_strength', 0.0)):g}",
                        f"site={best_test.get('site_label')}",
                    ]
                )
            )
        if entries and by_calibration:
            best_calibration = by_calibration[0]
            lines.append(
                "  best_by_calibration: "
                + " ".join(
                    [
                        f"layer={int(best_calibration['layer'])}",
                        f"cal={float(best_calibration.get('calibration_exact_acc', 0.0)):.4f}",
                        f"test={float(best_calibration.get('test_exact_acc', 0.0)):.4f}",
                        f"strength={float(best_calibration.get('intervention_strength', 0.0)):g}",
                        f"site={best_calibration.get('site_label')}",
                    ]
                )
            )
        lines.append("  full_test_ranking:")
        for entry in by_test:
            lines.append(
                "    "
                + " ".join(
                    [
                        f"rank={int(entry['rank'])}",
                        f"layer={int(entry['layer'])}",
                        f"test={float(entry.get('test_exact_acc', 0.0)):.4f}",
                        f"cal={float(entry.get('calibration_exact_acc', 0.0)):.4f}",
                        f"strength={float(entry.get('intervention_strength', 0.0)):g}",
                        f"site={entry.get('site_label')}",
                    ]
                )
            )
        lines.append("  full_calibration_ranking:")
        for entry in by_calibration:
            lines.append(
                "    "
                + " ".join(
                    [
                        f"rank={int(entry['rank'])}",
                        f"layer={int(entry['layer'])}",
                        f"cal={float(entry.get('calibration_exact_acc', 0.0)):.4f}",
                        f"test={float(entry.get('test_exact_acc', 0.0)):.4f}",
                        f"strength={float(entry.get('intervention_strength', 0.0)):g}",
                        f"site={entry.get('site_label')}",
                    ]
                )
            )
        lines.append("")
    return lines


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
                f"selected_layer={entry.get('selected_coupling_layer')} "
                f"strength={float(entry.get('selected_coupling_layer_lambda', 0.0)):g} "
                f"cal={float(entry.get('selected_coupling_layer_calibration_exact_acc', 0.0)):.4f} "
                f"test={float(entry.get('selected_coupling_layer_test_exact_acc', 0.0)):.4f}"
            )
            for label in ("calibration_ranking_comparison", "test_ranking_comparison"):
                compare = entry.get(label, {})
                if not isinstance(compare, dict):
                    continue
                metric_name = "calibration" if label.startswith("calibration") else "test"
                lines.append(
                    "    "
                    + f"{metric_name}: "
                    + " ".join(
                        [
                            f"best_actual={compare.get('best_actual_layer')}",
                            f"best_actual_rank_in_coupling={compare.get('best_actual_layer_coupling_rank')}",
                            f"top1_match={compare.get('top1_match')}",
                            f"top3_overlap={compare.get('top_overlap_count')}/{DEFAULT_COMPARE_TOP_K}",
                            f"spearman={float(compare.get('spearman_rank_correlation')):.4f}"
                            if compare.get("spearman_rank_correlation") is not None else "spearman=None",
                            f"mean_abs_rank_err={float(compare.get('mean_absolute_rank_error')):.4f}"
                            if compare.get("mean_absolute_rank_error") is not None else "mean_abs_rank_err=None",
                        ]
                    )
                )
        lines.append("")
    return lines


def _format_summary(
    *,
    token_position_id: str,
    layers: tuple[int, ...],
    single_layer_by_var: dict[str, list[dict[str, object]]],
    single_layer_rankings_by_var: dict[str, dict[str, list[dict[str, object]]]],
    transport_summaries: list[dict[str, object]],
) -> str:
    lines = [
        "MCQA Layerwise OT/UOT Analysis",
        f"token_position_id: {token_position_id}",
        f"layers: {list(int(layer) for layer in layers)}",
        "",
        "This analysis compares joint 26-layer coupling rankings against per-layer single-site intervention accuracies.",
        "Reported method accuracy uses the argmax layer from each target-variable coupling row, evaluated as a fixed-strength single-site intervention.",
        "",
    ]
    lines.extend(_format_single_layer_summary(single_layer_by_var, single_layer_rankings_by_var))
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
    ot_top_k_values = tuple(_parse_csv_ints(args.ot_top_k_values) or list(DEFAULT_OT_TOP_K_VALUES))
    ot_lambdas = tuple(_parse_csv_floats(args.ot_lambdas) or list(DEFAULT_OT_LAMBDAS))
    single_layer_lambdas = tuple(_parse_csv_floats(args.single_layer_lambdas) or list(DEFAULT_SINGLE_LAYER_LAMBDAS))
    if len(single_layer_lambdas) != 1:
        raise ValueError(
            "Single-layer brute-force evaluation now uses one fixed intervention strength. "
            f"Received {list(single_layer_lambdas)}"
        )

    print(
        f"[layerwise] start token_position={str(args.token_position_id)} "
        f"layers={len(sites)} methods={list(methods)}"
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
                top_k_values=ot_top_k_values,
                lambda_values=ot_lambdas,
                selection_verbose=True,
                source_target_vars=target_vars,
                calibration_metric=DEFAULT_CALIBRATION_METRIC,
                calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                top_k_values_by_var={target_var: ot_top_k_values for target_var in target_vars},
                lambda_values_by_var={target_var: ot_lambdas for target_var in target_vars},
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

    single_layer_root = sweep_root / "single_layer"
    transport_root = sweep_root / "transport"
    single_layer_by_var = _evaluate_all_layers(
        model=model,
        tokenizer=tokenizer,
        banks_by_split=banks_by_split,
        sites=sites,
        device=device,
        batch_size=int(args.batch_size),
        signature_mode=str(args.signature_mode),
        lambda_values=single_layer_lambdas,
        output_root=single_layer_root,
    )
    single_layer_rankings_by_var = _build_single_layer_rankings(single_layer_by_var)
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
            top_k_values=ot_top_k_values,
            lambda_values=ot_lambdas,
            prepared_artifacts=prepared_artifacts,
            output_root=transport_root,
            methods=methods,
            ot_epsilons=ot_epsilons,
            uot_beta_neurals=uot_beta_neurals,
        )
        transport_summaries = _summarize_transport_runs(
            transport_runs=transport_runs,
            single_layer_by_var=single_layer_by_var,
            sites=sites,
        )

    summary_text = _format_summary(
        token_position_id=str(args.token_position_id),
        layers=resolved_layers,
        single_layer_by_var=single_layer_by_var,
        single_layer_rankings_by_var=single_layer_rankings_by_var,
        transport_summaries=transport_summaries,
    )
    summary_path = sweep_root / "layerwise_ot_summary.txt"
    payload_path = sweep_root / "layerwise_ot_results.json"
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
            "ot_top_k_values": [int(value) for value in ot_top_k_values],
            "ot_lambdas": [float(value) for value in ot_lambdas],
            "single_layer_lambdas": [float(value) for value in single_layer_lambdas],
            "site_labels": [site.label for site in sites],
            "context_timing_seconds": context_timing_seconds,
            "timing_seconds": {
                "t_model_load": float(context_timing_seconds.get("t_model_load", 0.0)),
                "t_data_load": float(context_timing_seconds.get("t_data_load", 0.0)),
                "t_bank_build": float(context_timing_seconds.get("t_bank_build", 0.0)),
                "t_factual_filter": float(context_timing_seconds.get("t_factual_filter", 0.0)),
                "t_context_total_wall": float(context_timing_seconds.get("t_context_total_wall", 0.0)),
                "t_artifact_prepare_load": float(artifact_prepare_load_seconds),
                "t_artifact_prepare_create": float(artifact_prepare_create_seconds),
                "t_total_wall": float(perf_counter() - overall_start),
            },
            "artifact_cache_hit": bool(prepared_artifacts.get("loaded_from_disk", False)),
            "single_layer_by_var": single_layer_by_var,
            "single_layer_rankings_by_var": single_layer_rankings_by_var,
            "transport_runs": transport_runs,
            "transport_summaries": transport_summaries,
            "data": data_metadata,
            "summary_path": str(summary_path),
        },
    )
    print(f"Wrote layerwise OT analysis payload to {payload_path}")
    print(f"Wrote layerwise OT analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
