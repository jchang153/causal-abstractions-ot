from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mcqa_run as base_run
from mcqa_experiment.data import COUNTERFACTUAL_FAMILIES
from mcqa_experiment.data import canonicalize_target_var
from mcqa_experiment.ot import OTConfig, prepare_alignment_artifacts, run_alignment_pipeline
from mcqa_experiment.pca import (
    LayerPCABasis,
    load_or_fit_pca_basis,
    load_or_fit_pca_basis_from_prompt_records,
)
from mcqa_experiment.reporting import write_text_report
from mcqa_experiment.runtime import write_json
from mcqa_experiment.sites import RotatedBandSite, enumerate_rotated_band_sites
from mcqa_experiment.support import extract_ordered_site_support


DEFAULT_LAYERS = (20, 25)
DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_COUNTERFACTUAL_NAMES = ("answerPosition", "randomLetter", "answerPosition_randomLetter")
DEFAULT_TOKEN_POSITION_ID = "last_token"
DEFAULT_NUM_BANDS = 8
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.5, 2.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_BAND_SCHEME = "equal"
DEFAULT_BASIS_SOURCE_MODE = "all_variants"


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
    parser = argparse.ArgumentParser(description="Fixed-layer PCA-band OT focus runner for MCQA.")
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
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument("--band-scheme", default=DEFAULT_BAND_SCHEME, choices=("equal", "head"))
    parser.add_argument("--basis-source-mode", default=DEFAULT_BASIS_SOURCE_MODE, choices=("pair_bank", "all_variants"))
    parser.add_argument("--ot-epsilons", help="Comma-separated OT epsilons. Default: 0.5,1,2,4")
    parser.add_argument("--support-score-slack", type=float, default=0.05)
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
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


def _basis_metrics_by_band(basis: LayerPCABasis, band_sites: list[RotatedBandSite]) -> list[dict[str, object]]:
    total_variance = float(basis.explained_variance.sum().item())
    cumulative = 0.0
    metrics: list[dict[str, object]] = []
    for site in band_sites:
        band_variance = float(
            basis.explained_variance[int(site.component_start) : int(site.component_end)].sum().item()
        )
        variance_share = 0.0 if total_variance <= 0.0 else float(band_variance / total_variance)
        cumulative += variance_share
        metrics.append(
            {
                "site_label": site.label,
                "component_start": int(site.component_start),
                "component_end": int(site.component_end),
                "width": int(site.component_end) - int(site.component_start),
                "variance_share": float(variance_share),
                "cumulative_variance_share": float(cumulative),
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
    for rows in filtered_datasets.values():
        for row in rows:
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


def _format_layer_summary(
    *,
    layer: int,
    token_position_id: str,
    signature_mode: str,
    basis_source_mode: str,
    band_scheme: str,
    basis: LayerPCABasis,
    band_metrics: list[dict[str, object]],
    ot_compare_payloads: list[dict[str, object]],
    support_by_var: dict[str, dict[str, object]],
) -> str:
    lines = [
        "MCQA PCA OT Summary",
        f"layer: {int(layer)}",
        f"token_position: {token_position_id}",
        f"signature_mode: {signature_mode}",
        f"basis_source_mode: {basis_source_mode}",
        f"band_scheme: {band_scheme}",
        f"basis_id: {basis.basis_id}",
        f"rank: {int(basis.rank)}",
        f"num_fit_states: {int(basis.num_fit_states)}",
        "",
        "pca bands:",
    ]
    for metric in band_metrics:
        lines.append(
            f"{metric['site_label']} width={int(metric['width'])} "
            f"var={float(metric['variance_share']):.4f} "
            f"cum_var={float(metric['cumulative_variance_share']):.4f}"
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
    return "\n".join(lines)


def _write_epsilon_summary(path: Path, *, payload: dict[str, object]) -> None:
    lines = [
        "MCQA PCA OT Epsilon Summary",
        f"layer: {int(payload['layer'])}",
        f"token_position: {payload['token_position_id']}",
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
    parser = _build_parser()
    args = parser.parse_args()

    layers = DEFAULT_LAYERS if args.layers is None else tuple(_parse_csv_ints(args.layers) or [])
    if not layers:
        raise ValueError("No layers selected")
    if int(args.num_bands) <= 0:
        raise ValueError("num_bands must be > 0")
    ot_epsilons = tuple(_parse_csv_floats(args.ot_epsilons) or list(DEFAULT_OT_EPSILONS))

    results_root = Path(args.results_root)
    results_timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or base_run.RUN_TIMESTAMP
    sweep_root = results_root / f"{results_timestamp}_mcqa_ot_pca_focus"
    sweep_root.mkdir(parents=True, exist_ok=True)

    _configure_base_run(args, sweep_root=sweep_root, results_timestamp=results_timestamp)
    context = base_run.build_run_context()
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

    for layer in layers:
        layer_dir = sweep_root / f"layer_{int(layer):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        basis_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_{str(args.token_position_id)}_basis-{str(args.basis_source_mode)}_pca_basis.pt"
        )
        prompt_records = None
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
        band_sites = enumerate_rotated_band_sites(
            rank=int(basis.rank),
            num_bands=int(args.num_bands),
            layer=int(layer),
            token_position_id=str(args.token_position_id),
            basis_id=str(basis.basis_id),
            schedule=str(args.band_scheme),
        )
        band_metrics = _basis_metrics_by_band(basis, band_sites)
        pca_bases_by_id = {str(basis.basis_id): basis}

        prepared_artifacts = None
        ot_compare_payloads: list[dict[str, object]] = []
        for epsilon in ot_epsilons:
            output_stem = (
                f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-bands-{int(args.num_bands)}"
                f"_scheme-{str(args.band_scheme)}_basis-{str(args.basis_source_mode)}"
                f"_sig-{str(args.signature_mode)}_eps-{float(epsilon):g}_ot"
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
                        sites=band_sites,
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
                            sites=band_sites,
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
                    "num_bands": int(args.num_bands),
                    "band_scheme": str(args.band_scheme),
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
                    "band_sites": [site.label for site in band_sites],
                    "band_metrics": band_metrics,
                    "data": data_metadata,
                    "method_payloads": {"ot": method_payloads},
                }
                write_json(output_path, compare_payload)
                _write_epsilon_summary(summary_path, payload=compare_payload)
            ot_compare_payloads.append(compare_payload)

        support_by_var = extract_ordered_site_support(
            ot_run_payloads=ot_compare_payloads,
            sites=band_sites,
            score_slack=float(args.support_score_slack),
            prefix_sizes=(1, 2, 4),
            coverage_specs=(("S50", 0.50), ("S80", 0.80)),
        )
        support_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-bands-{int(args.num_bands)}"
            f"_scheme-{str(args.band_scheme)}_basis-{str(args.basis_source_mode)}"
            f"_sig-{str(args.signature_mode)}_support.json"
        )
        write_json(support_path, support_by_var)

        layer_summary_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-bands-{int(args.num_bands)}"
            f"_scheme-{str(args.band_scheme)}_basis-{str(args.basis_source_mode)}"
            f"_sig-{str(args.signature_mode)}_summary.txt"
        )
        write_text_report(
            layer_summary_path,
            _format_layer_summary(
                layer=int(layer),
                token_position_id=str(args.token_position_id),
                signature_mode=str(args.signature_mode),
                basis_source_mode=str(args.basis_source_mode),
                band_scheme=str(args.band_scheme),
                basis=basis,
                band_metrics=band_metrics,
                ot_compare_payloads=ot_compare_payloads,
                support_by_var=support_by_var,
            ),
        )

        layer_payload = {
            "kind": "mcqa_ot_pca_focus_layer",
            "layer": int(layer),
            "token_position_id": str(args.token_position_id),
            "num_bands": int(args.num_bands),
            "band_scheme": str(args.band_scheme),
            "basis_source_mode": str(args.basis_source_mode),
            "signature_mode": str(args.signature_mode),
            "ot_epsilons": [float(epsilon) for epsilon in ot_epsilons],
            "basis_path": str(basis_path),
            "basis": basis.to_payload(),
            "fit_prompt_record_count": 0 if prompt_records is None else len(prompt_records),
            "band_sites": [site.label for site in band_sites],
            "band_metrics": band_metrics,
            "support_score_slack": float(args.support_score_slack),
            "support_by_var": support_by_var,
            "ot_output_paths": [
                str(
                    layer_dir / (
                        f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-bands-{int(args.num_bands)}"
                        f"_scheme-{str(args.band_scheme)}_basis-{str(args.basis_source_mode)}"
                        f"_sig-{str(args.signature_mode)}_eps-{float(epsilon):g}_ot.json"
                    )
                )
                for epsilon in ot_epsilons
            ],
            "summary_path": str(layer_summary_path),
        }
        layer_payload_path = layer_dir / (
            f"mcqa_layer-{int(layer)}_pos-{str(args.token_position_id)}_pca-bands-{int(args.num_bands)}"
            f"_scheme-{str(args.band_scheme)}_basis-{str(args.basis_source_mode)}"
            f"_sig-{str(args.signature_mode)}_ot_pca.json"
        )
        write_json(layer_payload_path, layer_payload)
        all_payloads.append(layer_payload)
        manifest_runs.append(
            {
                "layer": int(layer),
                "token_position_id": str(args.token_position_id),
                "num_bands": int(args.num_bands),
                "band_scheme": str(args.band_scheme),
                "basis_source_mode": str(args.basis_source_mode),
                "signature_mode": str(args.signature_mode),
                "summary_path": str(layer_summary_path),
                "payload_path": str(layer_payload_path),
            }
        )

    manifest_path = sweep_root / "layer_sweep_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "mcqa_ot_pca_focus",
            "layers": [int(layer) for layer in layers],
            "token_position_id": str(args.token_position_id),
            "num_bands": int(args.num_bands),
            "band_scheme": str(args.band_scheme),
            "basis_source_mode": str(args.basis_source_mode),
            "signature_mode": str(args.signature_mode),
            "runs": manifest_runs,
        },
    )
    aggregate_path = sweep_root / "mcqa_run_results.json"
    write_json(aggregate_path, {"runs": all_payloads})
    print(f"Wrote PCA OT manifest to {manifest_path}")


if __name__ == "__main__":
    main()
