from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import mcqa_run as base_run
import torch
from mcqa_experiment.das import DASConfig, run_das_pipeline
from mcqa_experiment.data import COUNTERFACTUAL_FAMILIES, canonicalize_target_var
from mcqa_experiment.ot import OTConfig, prepare_alignment_artifacts, run_alignment_pipeline
from mcqa_experiment.pca import (
    LayerPCABasis,
    load_or_fit_pca_basis,
    load_or_fit_pca_basis_from_prompt_records,
)
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import (
    RotatedBandSite,
    enumerate_rotated_band_sites,
    enumerate_rotated_top_prefix_sites,
    site_total_width,
)
from mcqa_experiment.support import build_rotated_span_sites_from_support, extract_ordered_site_support


DEFAULT_LAYERS = (20, 25)
DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_NUM_BANDS = 8
DEFAULT_SITE_MENU = "partition"
DEFAULT_TOP_PREFIX_SIZES = (8, 16, 32, 64)
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.5, 2.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_BAND_SCHEME = "equal"
DEFAULT_BASIS_SOURCE_MODE = "all_variants"
DEFAULT_SCREEN_MAX_EPOCHS = 25
DEFAULT_SCREEN_MIN_EPOCHS = 2
DEFAULT_SCREEN_MASK_NAMES = ("Top1", "Top2")
DEFAULT_GUIDED_DAS_MAX_EPOCHS = 100
DEFAULT_GUIDED_DAS_MIN_EPOCHS = 5
DEFAULT_GUIDED_DAS_MASK_NAMES = ("Top1", "Top2", "Top4", "S50", "S80")


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-layer PCA-site OT focus runner for MCQA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", help="Comma-separated focused layers. Default: 20,25")
    parser.add_argument("--token-position-id", default=DEFAULT_TOKEN_POSITION_ID)
    parser.add_argument("--site-menu", default=DEFAULT_SITE_MENU, choices=("partition", "mixed"))
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument("--band-scheme", default=DEFAULT_BAND_SCHEME, choices=("equal", "head"))
    parser.add_argument(
        "--top-prefix-sizes",
        help="Comma-separated top-prefix sizes for mixed PCA menus. Default: 8,16,32,64",
    )
    parser.add_argument("--basis-source-mode", default=DEFAULT_BASIS_SOURCE_MODE, choices=("pair_bank", "all_variants"))
    parser.add_argument("--ot-epsilons", help="Comma-separated OT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--support-score-slack", type=float, default=0.05)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--screen-das", action="store_true")
    parser.add_argument(
        "--screen-mask-names",
        help="Comma-separated PCA support masks for the tiny DAS screen. Default: Top1,Top2",
    )
    parser.add_argument("--screen-max-epochs", type=int, default=DEFAULT_SCREEN_MAX_EPOCHS)
    parser.add_argument("--screen-min-epochs", type=int, default=DEFAULT_SCREEN_MIN_EPOCHS)
    parser.add_argument("--screen-restarts", type=int, default=1)
    parser.add_argument("--guided-das", action="store_true")
    parser.add_argument(
        "--guided-mask-names",
        help="Comma-separated PCA support masks for the guided DAS sweep. Default: Top1,Top2,Top4,S50,S80",
    )
    parser.add_argument(
        "--guided-subspace-dims",
        default="32,64,96,128,256,512,768,1024,1536,2048,2304",
        help="Comma-separated subspace dims for the guided DAS sweep. Defaults to the paper master grid.",
    )
    parser.add_argument("--guided-max-epochs", type=int, default=DEFAULT_GUIDED_DAS_MAX_EPOCHS)
    parser.add_argument("--guided-min-epochs", type=int, default=DEFAULT_GUIDED_DAS_MIN_EPOCHS)
    parser.add_argument("--guided-restarts", type=int, default=2)
    parser.add_argument("--results-root", default="results/anvil")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _configure_base_run(args: argparse.Namespace, *, sweep_root: Path, results_timestamp: str) -> None:
    base_run.DEVICE = str(args.device)
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


def _load_existing_runs(path: Path) -> list[dict[str, object]]:
    payload = _load_existing_payload(path)
    if not isinstance(payload, dict):
        return []
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, dict)]


def _merge_component_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted({(int(start), int(end)) for start, end in ranges})
    merged: list[tuple[int, int]] = []
    for component_start, component_end in ordered:
        if not merged or int(component_start) > int(merged[-1][1]):
            merged.append((int(component_start), int(component_end)))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (int(previous_start), max(int(previous_end), int(component_end)))
    return merged


def _site_component_ranges(site: RotatedBandSite) -> list[tuple[int, int]]:
    return [(int(site.component_start), int(site.component_end))]


def _rotated_site_metrics(
    *,
    basis: LayerPCABasis,
    sites: list[RotatedBandSite],
) -> list[dict[str, object]]:
    total_variance = float(basis.explained_variance.sum().item())
    metrics: list[dict[str, object]] = []
    for site in sites:
        component_ranges = _merge_component_ranges(_site_component_ranges(site))
        variance_share = 0.0
        site_width = 0
        for component_start, component_end in component_ranges:
            variance_share += float(
                basis.explained_variance[int(component_start) : int(component_end)].sum().item()
            )
            site_width += int(component_end) - int(component_start)
        metrics.append(
            {
                "site_label": site.label,
                "width": int(site_width),
                "variance_share": 0.0 if total_variance <= 0.0 else float(variance_share / total_variance),
            }
        )
    return metrics


def _unique_prompt_records_for_all_variants(
    *,
    train_bank,
    filtered_datasets: dict[str, list[dict[str, object]]],
    token_position,
    tokenizer,
) -> list[dict[str, object]]:
    grouped_rows: dict[str, dict[str, object]] = {}
    for row in filtered_datasets.get("train", []):
        base_input = row.get("input")
        if not isinstance(base_input, dict):
            continue
        base_prompt = str(base_input.get("raw_input", ""))
        if not base_prompt:
            continue
        family = str(row.get("counterfactual_family", ""))
        entry = grouped_rows.setdefault(base_prompt, {"base_input": base_input, "family_sources": {}})
        if isinstance(row.get("counterfactual_inputs"), list) and row["counterfactual_inputs"]:
            entry["family_sources"][family] = row["counterfactual_inputs"][0]

    ordered_base_prompts: list[str] = []
    seen_base_prompts: set[str] = set()
    for base_input in train_bank.base_inputs:
        base_prompt = str(base_input.get("raw_input", ""))
        if base_prompt and base_prompt not in seen_base_prompts:
            seen_base_prompts.add(base_prompt)
            ordered_base_prompts.append(base_prompt)

    prompt_records: list[dict[str, object]] = []
    seen_prompt_strings: set[str] = set()
    for base_prompt in ordered_base_prompts:
        grouped = grouped_rows.get(base_prompt)
        base_input = grouped.get("base_input") if isinstance(grouped, dict) else None
        if not isinstance(base_input, dict):
            base_input = next(
                (
                    candidate
                    for candidate in train_bank.base_inputs
                    if str(candidate.get("raw_input", "")) == base_prompt
                ),
                None,
            )
        if isinstance(base_input, dict):
            prompt_string = str(base_input.get("raw_input", ""))
            if prompt_string and prompt_string not in seen_prompt_strings:
                seen_prompt_strings.add(prompt_string)
                prompt_records.append(
                    {
                        "raw_input": prompt_string,
                        "position": int(token_position.resolve(base_input, tokenizer)),
                    }
                )
        family_sources = grouped.get("family_sources", {}) if isinstance(grouped, dict) else {}
        for family_name in COUNTERFACTUAL_FAMILIES:
            source_input = family_sources.get(str(family_name))
            if not isinstance(source_input, dict):
                continue
            prompt_string = str(source_input.get("raw_input", ""))
            if not prompt_string or prompt_string in seen_prompt_strings:
                continue
            seen_prompt_strings.add(prompt_string)
            prompt_records.append(
                {
                    "raw_input": prompt_string,
                    "position": int(token_position.resolve(source_input, tokenizer)),
                }
            )
    return prompt_records


def _enumerate_pca_sites(
    *,
    basis: LayerPCABasis,
    token_position_id: str,
    layer: int,
    site_menu: str,
    num_bands: int,
    band_scheme: str,
    top_prefix_sizes: tuple[int, ...],
) -> list[RotatedBandSite]:
    partition_sites = enumerate_rotated_band_sites(
        rank=int(basis.rank),
        num_bands=int(num_bands),
        layer=int(layer),
        token_position_id=str(token_position_id),
        basis_id=str(basis.basis_id),
        schedule=str(band_scheme),
    )
    if str(site_menu) == "partition":
        return partition_sites
    if str(site_menu) != "mixed":
        raise ValueError(f"Unsupported site_menu {site_menu!r}")
    prefix_sites = enumerate_rotated_top_prefix_sites(
        rank=int(basis.rank),
        prefix_sizes=top_prefix_sizes,
        layer=int(layer),
        token_position_id=str(token_position_id),
        basis_id=str(basis.basis_id),
    )
    ordered_sites: list[RotatedBandSite] = []
    seen_labels: set[str] = set()
    for site in [*partition_sites, *prefix_sites]:
        if site.label in seen_labels:
            continue
        seen_labels.add(site.label)
        ordered_sites.append(site)
    return ordered_sites


def _site_catalog_tag(
    *,
    site_menu: str,
    num_bands: int,
    band_scheme: str,
    top_prefix_sizes: tuple[int, ...],
) -> str:
    tag = f"menu-{str(site_menu)}-bands-{int(num_bands)}-scheme-{str(band_scheme)}"
    if str(site_menu) == "mixed":
        tag += f"-top-{'-'.join(str(size) for size in top_prefix_sizes)}"
    return tag


def _best_ot_records(ot_compare_payloads: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best_by_var: dict[str, dict[str, object]] = {}
    for compare_payload in ot_compare_payloads:
        epsilon = float(compare_payload.get("ot_epsilon", 0.0))
        for payload in compare_payload.get("method_payloads", {}).get("ot", []):
            result = payload.get("results", [{}])[0]
            target_var = str(payload.get("target_var"))
            record = {
                "exact_acc": float(result.get("exact_acc", 0.0)),
                "selection_score": float(result.get("selection_score", 0.0)),
                "site_label": str(result.get("site_label")),
                "epsilon": float(epsilon),
            }
            previous = best_by_var.get(target_var)
            if previous is None or (
                float(record["exact_acc"]),
                float(record["selection_score"]),
            ) > (
                float(previous["exact_acc"]),
                float(previous["selection_score"]),
            ):
                best_by_var[target_var] = record
    return best_by_var


def _filter_support_summary_by_mask_names(
    *,
    support_summary: dict[str, object],
    mask_names: tuple[str, ...],
) -> dict[str, object]:
    filtered_candidates = [
        candidate
        for candidate in support_summary.get("mask_candidates", [])
        if str(candidate.get("name")) in set(str(mask_name) for mask_name in mask_names)
    ]
    if not filtered_candidates and support_summary.get("mask_candidates"):
        filtered_candidates = [support_summary["mask_candidates"][0]]
    return {
        **support_summary,
        "mask_candidates": filtered_candidates,
    }


def _screen_subspace_dims(max_width: int) -> tuple[int, ...]:
    resolved_width = max(1, int(max_width))
    if resolved_width <= 8:
        return tuple(range(1, resolved_width + 1))
    dims = [dim for dim in (4, 8, 16, 32, 64) if int(dim) < resolved_width]
    dims.append(int(resolved_width))
    return tuple(dict.fromkeys(int(dim) for dim in dims))


def _guided_subspace_dims(max_width: int) -> tuple[int, ...]:
    resolved_width = max(1, int(max_width))
    raw_dims = [
        32,
        64,
        96,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2304,
    ]
    dims = tuple(int(dim) for dim in raw_dims if 1 <= int(dim) <= int(resolved_width))
    return dims or (resolved_width,)


def _write_das_text_report(path: Path, *, title: str, payload: dict[str, object], extra_lines: list[str] | None = None) -> None:
    result = payload.get("results", [{}])[0]
    lines = [
        title,
        f"site: {result.get('site_label')}",
        f"exact_acc: {float(result.get('exact_acc', 0.0)):.4f}",
        f"calibration_exact_acc: {float(result.get('selection_exact_acc', result.get('calibration_exact_acc', 0.0))):.4f}",
        f"subspace_dim: {result.get('subspace_dim')}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    write_text_report(path, "\n".join(lines))


def _run_pca_das_from_support(
    *,
    model,
    tokenizer,
    banks_by_split,
    device,
    layer_dir: Path,
    layer: int,
    token_position_id: str,
    site_catalog_tag: str,
    basis_source_mode: str,
    signature_mode: str,
    target_vars: tuple[str, ...],
    support_by_var: dict[str, dict[str, object]],
    pca_sites: list[RotatedBandSite],
    pca_bases_by_id: dict[str, LayerPCABasis],
    model_hidden_size: int,
    batch_size: int,
    enabled: bool,
    method_suffix: str,
    method_name: str,
    title: str,
    mask_names: tuple[str, ...],
    max_epochs: int,
    min_epochs: int,
    plateau_patience: int,
    plateau_rel_delta: float,
    learning_rate: float,
    explicit_subspace_dims: tuple[int, ...] | None,
    subspace_dim_resolver,
    restarts: int,
) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    if not enabled:
        return payloads
    for target_var in target_vars:
        support_summary = support_by_var.get(str(target_var))
        if support_summary is None:
            continue
        filtered_summary = _filter_support_summary_by_mask_names(
            support_summary=support_summary,
            mask_names=mask_names,
        )
        span_sites = build_rotated_span_sites_from_support(
            support_summary=filtered_summary,
            sites=pca_sites,
        )
        if not span_sites:
            continue
        if explicit_subspace_dims is None:
            subspace_dims = subspace_dim_resolver(
                max(
                    int(site_total_width(site, model_hidden_size=int(model_hidden_size)))
                    for site in span_sites
                )
            )
        else:
            max_width = max(
                int(site_total_width(site, model_hidden_size=int(model_hidden_size)))
                for site in span_sites
            )
            filtered_dims = tuple(
                int(dim) for dim in explicit_subspace_dims if 0 < int(dim) <= int(max_width)
            )
            subspace_dims = filtered_dims or (int(max_width),)
        output_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
            f"_basis-{str(basis_source_mode)}_sig-{str(signature_mode)}_{str(target_var)}_{method_suffix}.json"
        )
        summary_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
            f"_basis-{str(basis_source_mode)}_sig-{str(signature_mode)}_{str(target_var)}_{method_suffix}.txt"
        )
        payload = _load_existing_payload(output_path)
        if payload is None:
            payload = run_das_pipeline(
                model=model,
                train_bank=banks_by_split["train"][target_var],
                calibration_bank=banks_by_split["calibration"][target_var],
                holdout_bank=banks_by_split["test"][target_var],
                sites=span_sites,
                device=device,
                tokenizer=tokenizer,
                config=DASConfig(
                    method_name=str(method_name),
                    batch_size=int(batch_size),
                    max_epochs=int(max_epochs),
                    min_epochs=int(min_epochs),
                    plateau_patience=int(plateau_patience),
                    plateau_rel_delta=float(plateau_rel_delta),
                    learning_rate=float(learning_rate),
                    subspace_dims=subspace_dims,
                    store_candidate_holdout_metrics=False,
                    restarts=max(1, int(restarts)),
                    verbose=True,
                ),
                pca_bases_by_id=pca_bases_by_id,
            )
            payload["support_summary"] = filtered_summary
            payload["mask_names"] = list(str(mask_name) for mask_name in mask_names)
            write_json(output_path, payload)
            _write_das_text_report(
                summary_path,
                title=str(title),
                payload=payload,
                extra_lines=[
                    f"target_var: {target_var}",
                    f"mask_names: {list(str(mask_name) for mask_name in mask_names)}",
                    f"subspace_dims: {list(int(dim) for dim in subspace_dims)}",
                    f"restarts: {max(1, int(restarts))}",
                ],
            )
        payloads[str(target_var)] = payload
    return payloads


def _format_layer_summary(
    *,
    layer: int,
    token_position_id: str,
    signature_mode: str,
    basis_source_mode: str,
    site_menu: str,
    num_bands: int,
    band_scheme: str,
    top_prefix_sizes: tuple[int, ...],
    basis: LayerPCABasis,
    site_metrics: list[dict[str, object]],
    ot_compare_payloads: list[dict[str, object]],
    support_by_var: dict[str, dict[str, object]],
    screen_payloads: dict[str, dict[str, object]],
    guided_payloads: dict[str, dict[str, object]],
) -> str:
    lines = [
        "MCQA PCA OT Summary",
        f"layer: {int(layer)}",
        f"token_position: {token_position_id}",
        f"signature_mode: {signature_mode}",
        f"basis_source_mode: {basis_source_mode}",
        f"site_menu: {site_menu}",
        f"num_bands: {int(num_bands)}",
        f"band_scheme: {band_scheme}",
        f"top_prefix_sizes: {list(int(size) for size in top_prefix_sizes)}",
        f"basis_id: {basis.basis_id}",
        f"rank: {int(basis.rank)}",
        f"num_fit_states: {int(basis.num_fit_states)}",
        "",
        "pca sites:",
    ]
    for metric in site_metrics:
        lines.append(
            f"{metric['site_label']} width={int(metric['width'])} "
            f"var={float(metric['variance_share']):.4f}"
        )
    lines.append("")
    lines.append("best OT by epsilon:")
    for target_var, record in _best_ot_records(ot_compare_payloads).items():
        lines.append(
            f"OT[{target_var}] exact={float(record['exact_acc']):.4f} "
            f"cal={float(record['selection_score']):.4f} "
            f"eps={float(record['epsilon']):g} site={record['site_label']}"
        )
    lines.append("")
    lines.append("pooled PCA support:")
    for target_var in DEFAULT_TARGET_VARS:
        summary = support_by_var.get(str(target_var), {})
        if not summary:
            continue
        lines.append(
            f"support[{target_var}] best_score={float(summary.get('best_selection_score', 0.0)):.4f} "
            f"masks={[candidate.get('name') for candidate in summary.get('mask_candidates', [])]}"
        )
        lines.append(f"support[{target_var}] evidence={summary.get('site_evidence', {})}")
    if screen_payloads:
        lines.append("")
        lines.append("tiny PCA-span DAS screen:")
        for target_var in DEFAULT_TARGET_VARS:
            payload = screen_payloads.get(str(target_var))
            if payload is None:
                continue
            result = payload.get("results", [{}])[0]
            lines.append(
                f"DAS_SCREEN[{target_var}] exact={float(result.get('exact_acc', 0.0)):.4f} "
                f"cal={float(result.get('selection_exact_acc', result.get('calibration_exact_acc', 0.0))):.4f} "
                f"site={result.get('site_label')} dim={result.get('subspace_dim')}"
            )
    if guided_payloads:
        lines.append("")
        lines.append("guided PCA-span DAS sweep:")
        for target_var in DEFAULT_TARGET_VARS:
            payload = guided_payloads.get(str(target_var))
            if payload is None:
                continue
            result = payload.get("results", [{}])[0]
            lines.append(
                f"DAS_GUIDED[{target_var}] exact={float(result.get('exact_acc', 0.0)):.4f} "
                f"cal={float(result.get('selection_exact_acc', result.get('calibration_exact_acc', 0.0))):.4f} "
                f"site={result.get('site_label')} dim={result.get('subspace_dim')}"
            )
    return "\n".join(lines)


def _write_epsilon_summary(path: Path, *, payload: dict[str, object]) -> None:
    lines = [
        "MCQA PCA OT Epsilon Summary",
        f"layer: {int(payload['layer'])}",
        f"token_position: {payload['token_position_id']}",
        f"site_menu: {payload['site_menu']}",
        f"num_bands: {int(payload['num_bands'])}",
        f"band_scheme: {payload['band_scheme']}",
        f"basis_source_mode: {payload['basis_source_mode']}",
        f"ot_epsilon: {float(payload['ot_epsilon']):g}",
        f"signature_mode: {payload['signature_mode']}",
        f"basis_id: {payload['basis']['basis_id']}",
        "",
    ]
    for result_payload in payload.get("method_payloads", {}).get("ot", []):
        result = result_payload.get("results", [{}])[0]
        lines.append(
            f"OT[{result_payload.get('target_var')}] exact={float(result.get('exact_acc', 0.0)):.4f} "
            f"cal={float(result.get('selection_score', 0.0)):.4f} "
            f"site={result.get('site_label')}"
        )
    write_text_report(path, "\n".join(lines))


def main() -> None:
    stage_start = perf_counter()
    parser = _build_parser()
    args = parser.parse_args()

    layers = DEFAULT_LAYERS if args.layers is None else tuple(_parse_csv_ints(args.layers) or [])
    if not layers:
        raise ValueError("No layers selected")
    if int(args.num_bands) <= 0:
        raise ValueError("num_bands must be > 0")
    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))
    top_prefix_sizes = tuple(_parse_csv_ints(args.top_prefix_sizes) or list(DEFAULT_TOP_PREFIX_SIZES))
    screen_mask_names = tuple(_parse_csv_strings(args.screen_mask_names) or list(DEFAULT_SCREEN_MASK_NAMES))
    guided_mask_names = tuple(_parse_csv_strings(args.guided_mask_names) or list(DEFAULT_GUIDED_DAS_MASK_NAMES))
    guided_subspace_dims = None if args.guided_subspace_dims is None else tuple(_parse_csv_ints(args.guided_subspace_dims) or [])

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_ot_pca_focus"
    sweep_root.mkdir(parents=True, exist_ok=True)

    _configure_base_run(args, sweep_root=sweep_root, results_timestamp=results_timestamp)
    context = base_run.build_run_context()
    context_timing_seconds = dict(context.get("timing_seconds", {}))
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_positions = context["token_positions"]
    filtered_datasets = context["filtered_datasets"]
    banks_by_split = context["banks_by_split"]
    data_metadata = context["data_metadata"]
    device = context["device"]
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in DEFAULT_TARGET_VARS)
    fit_bank_for_basis = banks_by_split["train"][target_vars[0]]
    token_position_by_id = {
        str(token_position.id): token_position
        for token_position in token_positions
    }
    selected_token_position = token_position_by_id.get(str(args.token_position_id))
    if selected_token_position is None:
        raise ValueError(f"Unknown token_position_id {args.token_position_id!r}")

    all_payloads: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []
    site_catalog_tag = _site_catalog_tag(
        site_menu=str(args.site_menu),
        num_bands=int(args.num_bands),
        band_scheme=str(args.band_scheme),
        top_prefix_sizes=top_prefix_sizes,
    )

    for layer in layers:
        layer_start = perf_counter()
        layer_dir = sweep_root / f"layer_{int(layer):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        basis_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_{str(args.token_position_id)}_basis-{str(args.basis_source_mode)}_pca_basis.pt"
        )
        prompt_records = None
        pca_fit_start = perf_counter()
        if str(args.basis_source_mode) == "all_variants":
            prompt_records = _unique_prompt_records_for_all_variants(
                train_bank=fit_bank_for_basis,
                filtered_datasets=filtered_datasets,
                token_position=selected_token_position,
                tokenizer=tokenizer,
            )
            basis = load_or_fit_pca_basis_from_prompt_records(
                path=basis_path,
                model=model,
                tokenizer=tokenizer,
                prompt_records=prompt_records,
                layer=int(layer),
                token_position_id=str(args.token_position_id),
                batch_size=int(args.batch_size),
                device=device,
                basis_id=f"L{int(layer)}:{str(args.token_position_id)}:pca-{str(args.basis_source_mode)}",
            )
        else:
            basis = load_or_fit_pca_basis(
                path=basis_path,
                model=model,
                bank=fit_bank_for_basis,
                layer=int(layer),
                token_position_id=str(args.token_position_id),
                batch_size=int(args.batch_size),
                device=device,
                basis_id=f"L{int(layer)}:{str(args.token_position_id)}:pca-{str(args.basis_source_mode)}",
            )
        _synchronize_if_cuda(device)
        pca_fit_seconds = float(perf_counter() - pca_fit_start)
        site_build_start = perf_counter()
        pca_sites = _enumerate_pca_sites(
            basis=basis,
            token_position_id=str(args.token_position_id),
            layer=int(layer),
            site_menu=str(args.site_menu),
            num_bands=int(args.num_bands),
            band_scheme=str(args.band_scheme),
            top_prefix_sizes=top_prefix_sizes,
        )
        site_metrics = _rotated_site_metrics(basis=basis, sites=pca_sites)
        pca_bases_by_id = {str(basis.basis_id): basis}
        site_build_seconds = float(perf_counter() - site_build_start)

        prepared_artifacts = None
        ot_compare_payloads: list[dict[str, object]] = []
        ot_fit_cal_start = perf_counter()
        for epsilon in ot_epsilons:
            output_stem = (
                f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
                f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_eps-{float(epsilon):g}_ot"
            )
            output_path = layer_dir / f"{output_stem}.json"
            summary_path = layer_dir / f"{output_stem}.txt"
            compare_payload = _load_existing_payload(output_path)
            if compare_payload is None:
                ot_config = OTConfig(
                    method="ot",
                    batch_size=int(args.batch_size),
                    epsilon=float(epsilon),
                    signature_mode=str(args.signature_mode),
                    top_k_values=DEFAULT_OT_TOP_K_VALUES,
                    lambda_values=DEFAULT_OT_LAMBDAS,
                    source_target_vars=target_vars,
                    calibration_metric=DEFAULT_CALIBRATION_METRIC,
                    calibration_family_weights=DEFAULT_CALIBRATION_FAMILY_WEIGHTS,
                    top_k_values_by_var={target_var: DEFAULT_OT_TOP_K_VALUES for target_var in target_vars},
                    lambda_values_by_var={target_var: DEFAULT_OT_LAMBDAS for target_var in target_vars},
                )
                if prepared_artifacts is None:
                    prepared_artifacts = prepare_alignment_artifacts(
                        model=model,
                        fit_banks_by_var={target_var: banks_by_split["train"][target_var] for target_var in target_vars},
                        sites=pca_sites,
                        device=device,
                        config=ot_config,
                        pca_bases_by_id=pca_bases_by_id,
                    )
                method_payloads = []
                for target_var in target_vars:
                    method_payloads.append(
                        run_alignment_pipeline(
                            model=model,
                            fit_banks_by_var={source_var: banks_by_split["train"][source_var] for source_var in target_vars},
                            calibration_bank=banks_by_split["calibration"][target_var],
                            holdout_bank=banks_by_split["test"][target_var],
                            sites=pca_sites,
                            device=device,
                            tokenizer=tokenizer,
                            config=ot_config,
                            prepared_artifacts=prepared_artifacts,
                            pca_bases_by_id=pca_bases_by_id,
                        )
                    )
                compare_payload = {
                    "kind": "mcqa_ot_pca_focus_epsilon",
                    "layer": int(layer),
                    "token_position_id": str(args.token_position_id),
                    "site_menu": str(args.site_menu),
                    "num_bands": int(args.num_bands),
                    "band_scheme": str(args.band_scheme),
                    "top_prefix_sizes": [int(size) for size in top_prefix_sizes],
                    "basis_source_mode": str(args.basis_source_mode),
                    "ot_epsilon": float(epsilon),
                    "signature_mode": str(args.signature_mode),
                    "model_name": base_run.MODEL_NAME,
                    "basis": {
                        "basis_id": str(basis.basis_id),
                        "rank": int(basis.rank),
                        "hidden_size": int(basis.hidden_size),
                        "num_fit_states": int(basis.num_fit_states),
                        "path": str(basis_path),
                        "fit_prompt_record_count": 0 if prompt_records is None else len(prompt_records),
                    },
                    "site_labels": [site.label for site in pca_sites],
                    "site_metrics": site_metrics,
                    "data": data_metadata,
                    "method_payloads": {"ot": method_payloads},
                }
                write_json(output_path, compare_payload)
                _write_epsilon_summary(summary_path, payload=compare_payload)
            ot_compare_payloads.append(compare_payload)
        _synchronize_if_cuda(device)
        ot_fit_cal_seconds = float(perf_counter() - ot_fit_cal_start)

        support_start = perf_counter()
        support_by_var = extract_ordered_site_support(
            ot_run_payloads=ot_compare_payloads,
            sites=pca_sites,
            score_slack=float(args.support_score_slack),
            prefix_sizes=(1, 2, 4),
            coverage_specs=(("S50", 0.50), ("S80", 0.80)),
        )
        support_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
            f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_support.json"
        )
        write_json(support_path, support_by_var)
        support_extract_seconds = float(perf_counter() - support_start)

        das_screen_start = perf_counter()
        screen_payloads = _run_pca_das_from_support(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            device=device,
            layer_dir=layer_dir,
            layer=int(layer),
            token_position_id=str(args.token_position_id),
            site_catalog_tag=site_catalog_tag,
            basis_source_mode=str(args.basis_source_mode),
            signature_mode=str(args.signature_mode),
            target_vars=target_vars,
            support_by_var=support_by_var,
            pca_sites=pca_sites,
            pca_bases_by_id=pca_bases_by_id,
            model_hidden_size=int(model.config.hidden_size),
            batch_size=int(args.batch_size),
            enabled=bool(args.screen_das),
            method_suffix="das_screen",
            method_name="das_pca_screen",
            title="MCQA PCA DAS Screen Summary",
            mask_names=screen_mask_names,
            max_epochs=int(args.screen_max_epochs),
            min_epochs=int(args.screen_min_epochs),
            plateau_patience=1,
            plateau_rel_delta=base_run.DAS_PLATEAU_REL_DELTA,
            learning_rate=base_run.DAS_LEARNING_RATE,
            explicit_subspace_dims=None,
            subspace_dim_resolver=_screen_subspace_dims,
            restarts=int(args.screen_restarts),
        )
        _synchronize_if_cuda(device)
        das_screen_seconds = float(perf_counter() - das_screen_start)
        das_guided_start = perf_counter()
        guided_payloads = _run_pca_das_from_support(
            model=model,
            tokenizer=tokenizer,
            banks_by_split=banks_by_split,
            device=device,
            layer_dir=layer_dir,
            layer=int(layer),
            token_position_id=str(args.token_position_id),
            site_catalog_tag=site_catalog_tag,
            basis_source_mode=str(args.basis_source_mode),
            signature_mode=str(args.signature_mode),
            target_vars=target_vars,
            support_by_var=support_by_var,
            pca_sites=pca_sites,
            pca_bases_by_id=pca_bases_by_id,
            model_hidden_size=int(model.config.hidden_size),
            batch_size=int(args.batch_size),
            enabled=bool(args.guided_das),
            method_suffix="das_guided",
            method_name="das_pca_guided",
            title="MCQA PCA Guided DAS Summary",
            mask_names=guided_mask_names,
            max_epochs=int(args.guided_max_epochs),
            min_epochs=int(args.guided_min_epochs),
            plateau_patience=int(base_run.DAS_PLATEAU_PATIENCE),
            plateau_rel_delta=base_run.DAS_PLATEAU_REL_DELTA,
            learning_rate=base_run.DAS_LEARNING_RATE,
            explicit_subspace_dims=guided_subspace_dims,
            subspace_dim_resolver=_guided_subspace_dims,
            restarts=int(args.guided_restarts),
        )
        _synchronize_if_cuda(device)
        das_guided_seconds = float(perf_counter() - das_guided_start)
        layer_total_seconds = float(perf_counter() - layer_start)

        layer_summary_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
            f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_summary.txt"
        )
        write_text_report(
            layer_summary_path,
            _format_layer_summary(
                layer=int(layer),
                token_position_id=str(args.token_position_id),
                signature_mode=str(args.signature_mode),
                basis_source_mode=str(args.basis_source_mode),
                site_menu=str(args.site_menu),
                num_bands=int(args.num_bands),
                band_scheme=str(args.band_scheme),
                top_prefix_sizes=top_prefix_sizes,
                basis=basis,
                site_metrics=site_metrics,
                ot_compare_payloads=ot_compare_payloads,
                support_by_var=support_by_var,
                screen_payloads=screen_payloads,
                guided_payloads=guided_payloads,
            ),
        )

        layer_payload = {
            "kind": "mcqa_ot_pca_focus_layer",
            "layer": int(layer),
            "token_position_id": str(args.token_position_id),
            "site_menu": str(args.site_menu),
            "num_bands": int(args.num_bands),
            "band_scheme": str(args.band_scheme),
            "top_prefix_sizes": [int(size) for size in top_prefix_sizes],
            "basis_source_mode": str(args.basis_source_mode),
            "signature_mode": str(args.signature_mode),
            "ot_epsilons": [float(epsilon) for epsilon in ot_epsilons],
            "basis_path": str(basis_path),
            "basis": basis.to_payload(),
            "fit_prompt_record_count": 0 if prompt_records is None else len(prompt_records),
            "site_labels": [site.label for site in pca_sites],
            "site_metrics": site_metrics,
            "support_score_slack": float(args.support_score_slack),
            "support_by_var": support_by_var,
            "screen_mask_names": [str(mask_name) for mask_name in screen_mask_names],
            "screen_das_enabled": bool(args.screen_das),
            "guided_mask_names": [str(mask_name) for mask_name in guided_mask_names],
            "guided_subspace_dims": None if guided_subspace_dims is None else [int(dim) for dim in guided_subspace_dims],
            "guided_das_enabled": bool(args.guided_das),
            "context_timing_seconds": context_timing_seconds,
            "timing_seconds": {
                "t_model_load": float(context_timing_seconds.get("t_model_load", 0.0)),
                "t_data_load": float(context_timing_seconds.get("t_data_load", 0.0)),
                "t_bank_build": float(context_timing_seconds.get("t_bank_build", 0.0)),
                "t_factual_filter": float(context_timing_seconds.get("t_factual_filter", 0.0)),
                "t_context_total_wall": float(context_timing_seconds.get("t_context_total_wall", 0.0)),
                "t_stageB_pca_fit": float(pca_fit_seconds),
                "t_stageB_pca_site_build": float(site_build_seconds),
                "t_stageB_pca_ot_fit_cal": float(ot_fit_cal_seconds),
                "t_support_extract": float(support_extract_seconds),
                "t_stageC_das_screen": float(das_screen_seconds),
                "t_stageC_das_full": float(das_guided_seconds),
                "t_layer_total_wall": float(layer_total_seconds),
                "t_stage_total_wall_so_far": float(perf_counter() - stage_start),
            },
            "screen_output_paths": {
                str(target_var): str(
                    layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
                        f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_{str(target_var)}_das_screen.json"
                    )
                )
                for target_var in screen_payloads
            },
            "guided_output_paths": {
                str(target_var): str(
                    layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
                        f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_{str(target_var)}_das_guided.json"
                    )
                )
                for target_var in guided_payloads
            },
            "ot_output_paths": [
                str(
                    layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
                        f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_eps-{float(epsilon):g}_ot.json"
                    )
                )
                for epsilon in ot_epsilons
            ],
            "summary_path": str(layer_summary_path),
        }
        layer_payload_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-{site_catalog_tag}"
            f"_basis-{str(args.basis_source_mode)}_sig-{str(args.signature_mode)}_ot_pca.json"
        )
        write_json(layer_payload_path, layer_payload)
        all_payloads.append(layer_payload)
        manifest_runs.append(
            {
                "layer": int(layer),
                "token_position_id": str(args.token_position_id),
                "site_menu": str(args.site_menu),
                "num_bands": int(args.num_bands),
                "band_scheme": str(args.band_scheme),
                "top_prefix_sizes": [int(size) for size in top_prefix_sizes],
                "basis_source_mode": str(args.basis_source_mode),
                "signature_mode": str(args.signature_mode),
                "screen_das_enabled": bool(args.screen_das),
                "guided_das_enabled": bool(args.guided_das),
                "runtime_seconds": float(layer_total_seconds),
                "timing_seconds": layer_payload["timing_seconds"],
                "summary_path": str(layer_summary_path),
                "payload_path": str(layer_payload_path),
            }
        )

    manifest_path = sweep_root / "layer_sweep_manifest.json"
    existing_manifest_runs = _load_existing_runs(manifest_path)
    current_payload_paths = {str(run["payload_path"]) for run in manifest_runs}
    write_json(
        manifest_path,
        {
            "kind": "mcqa_ot_pca_focus",
            "layers": [int(layer) for layer in layers],
            "token_position_id": str(args.token_position_id),
            "site_menu": str(args.site_menu),
            "num_bands": int(args.num_bands),
            "band_scheme": str(args.band_scheme),
            "top_prefix_sizes": [int(size) for size in top_prefix_sizes],
            "basis_source_mode": str(args.basis_source_mode),
            "signature_mode": str(args.signature_mode),
            "screen_das_enabled": bool(args.screen_das),
            "screen_mask_names": [str(mask_name) for mask_name in screen_mask_names],
            "guided_das_enabled": bool(args.guided_das),
            "guided_mask_names": [str(mask_name) for mask_name in guided_mask_names],
            "guided_subspace_dims": None if guided_subspace_dims is None else [int(dim) for dim in guided_subspace_dims],
            "context_timing_seconds": context_timing_seconds,
            "runtime_seconds": float(perf_counter() - stage_start),
            "runs": [
                *[run for run in existing_manifest_runs if str(run.get("payload_path", "")) not in current_payload_paths],
                *manifest_runs,
            ],
        },
    )
    aggregate_path = sweep_root / "mcqa_run_results.json"
    existing_payload_runs = _load_existing_runs(aggregate_path)
    write_json(
        aggregate_path,
        {
            "runs": [
                *[
                    run
                    for run in existing_payload_runs
                    if str(run.get("payload_path", "")) not in current_payload_paths
                ],
                *all_payloads,
            ]
        },
    )
    print(f"Wrote PCA OT manifest to {manifest_path}")


if __name__ == "__main__":
    main()
