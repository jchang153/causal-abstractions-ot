"""Shared comparison runner for MCQA OT/UOT/DAS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .das import DASConfig, run_das_pipeline
from .data import canonicalize_target_var
from .ot import (
    OTConfig,
    canonicalize_source_target_var,
    is_null_source_target_var,
    prepare_alignment_artifacts,
    run_alignment_pipeline,
    run_bruteforce_site_pipeline,
)
from .reporting import format_summary, print_results_table, summarize_method_records, write_text_report
from .runtime import write_json
from .sites import enumerate_layer_block_sites, enumerate_residual_sites


@dataclass(frozen=True)
class CompareExperimentConfig:
    """Config controlling one end-to-end MCQA comparison run."""

    model_name: str
    output_path: Path
    summary_path: Path
    methods: tuple[str, ...] = ("ot",)
    target_vars: tuple[str, ...] = ("answer_pointer", "answer_token")
    batch_size: int = 16
    ot_epsilon: float = 1.0
    uot_beta_neural: float = 1.0
    signature_mode: str = "family_slot_label_delta"
    ot_top_k_values: tuple[int, ...] | None = None
    ot_lambdas: tuple[float, ...] = (1.0,)
    calibration_metric: str = "exact_acc"
    calibration_family_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    ot_top_k_values_by_var: dict[str, tuple[int, ...]] | None = None
    ot_lambdas_by_var: dict[str, tuple[float, ...]] | None = None
    das_max_epochs: int = 5
    das_min_epochs: int = 1
    das_plateau_patience: int = 2
    das_plateau_rel_delta: float = 5e-3
    das_learning_rate: float = 1e-3
    das_subspace_dims: tuple[int, ...] | None = None
    das_store_candidate_holdout_metrics: bool = False
    das_restarts: int = 1
    resolution: int | None = None
    layers: tuple[int, ...] | None = None
    token_position_ids: tuple[str, ...] | None = ("correct_symbol", "correct_symbol_period", "last_token")
    layer_blocks: tuple[tuple[int, ...], ...] | None = None
    ot_source_target_vars: tuple[str, ...] | None = None


def run_comparison(
    *,
    model,
    tokenizer,
    token_positions,
    banks_by_split,
    data_metadata: dict[str, object],
    device,
    config: CompareExperimentConfig,
    prepared_ot_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    target_vars = tuple(canonicalize_target_var(target_var) for target_var in config.target_vars)
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    if config.layer_blocks:
        ot_sites = enumerate_layer_block_sites(
            hidden_size=int(model.config.hidden_size),
            token_position_ids=token_position_ids,
            layer_blocks=tuple(tuple(int(layer) for layer in block) for block in config.layer_blocks),
            selected_token_position_ids=config.token_position_ids,
        )
    else:
        ot_sites = enumerate_residual_sites(
            num_layers=int(model.config.num_hidden_layers),
            hidden_size=int(model.config.hidden_size),
            token_position_ids=token_position_ids,
            resolution=config.resolution,
            layers=config.layers,
            selected_token_position_ids=config.token_position_ids,
        )
    das_sites = enumerate_residual_sites(
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        token_position_ids=token_position_ids,
        resolution=int(model.config.hidden_size),
        layers=config.layers,
        selected_token_position_ids=config.token_position_ids,
    )
    method_payloads: dict[str, list[dict[str, object]]] = {method: [] for method in config.methods}
    all_records: list[dict[str, object]] = []
    for method in config.methods:
        print(f"[method] start method={method} targets={list(target_vars)}")
        prepared_artifacts = None
        ot_config = None
        if method in {"ot", "uot"} and target_vars:
            source_target_vars = tuple(
                canonicalize_source_target_var(target_var)
                for target_var in (config.ot_source_target_vars or target_vars)
            )
            ot_config = OTConfig(
                method=method,
                batch_size=config.batch_size,
                epsilon=config.ot_epsilon,
                uot_beta_neural=config.uot_beta_neural,
                signature_mode=config.signature_mode,
                top_k_values=config.ot_top_k_values,
                lambda_values=config.ot_lambdas,
                source_target_vars=source_target_vars,
                calibration_metric=config.calibration_metric,
                calibration_family_weights=config.calibration_family_weights,
                top_k_values_by_var=config.ot_top_k_values_by_var,
                lambda_values_by_var=config.ot_lambdas_by_var,
            )
            prepared_artifacts = prepared_ot_artifacts
            if prepared_artifacts is None:
                fit_banks_by_var = {
                    target_var: banks_by_split["train"][target_var]
                    for target_var in source_target_vars
                    if not is_null_source_target_var(target_var)
                }
                prepared_artifacts = prepare_alignment_artifacts(
                    model=model,
                    fit_banks_by_var=fit_banks_by_var,
                    sites=ot_sites,
                    device=device,
                    config=ot_config,
                )
        for target_var in target_vars:
            start = perf_counter()
            print(f"[method] method={method} target={target_var}")
            train_bank = banks_by_split["train"][target_var]
            calibration_bank = banks_by_split["calibration"][target_var]
            test_bank = banks_by_split["test"][target_var]
            if method in {"ot", "uot"}:
                current_ot_config = ot_config or OTConfig(
                    method=method,
                    batch_size=config.batch_size,
                    epsilon=config.ot_epsilon,
                    uot_beta_neural=config.uot_beta_neural,
                    signature_mode=config.signature_mode,
                    top_k_values=config.ot_top_k_values,
                    lambda_values=config.ot_lambdas,
                    source_target_vars=tuple(
                        canonicalize_source_target_var(target_var)
                        for target_var in (config.ot_source_target_vars or target_vars)
                    ),
                    calibration_metric=config.calibration_metric,
                    calibration_family_weights=config.calibration_family_weights,
                    top_k_values_by_var=config.ot_top_k_values_by_var,
                    lambda_values_by_var=config.ot_lambdas_by_var,
                )
                payload = run_alignment_pipeline(
                    model=model,
                    fit_banks_by_var={
                        source_var: banks_by_split["train"][source_var]
                        for source_var in current_ot_config.source_target_vars
                        if not is_null_source_target_var(source_var)
                    },
                    calibration_bank=calibration_bank,
                    holdout_bank=test_bank,
                    sites=ot_sites,
                    device=device,
                    tokenizer=tokenizer,
                    config=current_ot_config,
                    prepared_artifacts=prepared_artifacts,
                )
            elif method == "bruteforce":
                current_ot_config = OTConfig(
                    method=method,
                    batch_size=config.batch_size,
                    epsilon=config.ot_epsilon,
                    uot_beta_neural=config.uot_beta_neural,
                    signature_mode=config.signature_mode,
                    top_k_values=config.ot_top_k_values,
                    lambda_values=config.ot_lambdas,
                    source_target_vars=target_vars,
                    calibration_metric=config.calibration_metric,
                    calibration_family_weights=config.calibration_family_weights,
                    top_k_values_by_var=config.ot_top_k_values_by_var,
                    lambda_values_by_var=config.ot_lambdas_by_var,
                )
                payload = run_bruteforce_site_pipeline(
                    model=model,
                    calibration_bank=calibration_bank,
                    holdout_bank=test_bank,
                    sites=ot_sites,
                    device=device,
                    tokenizer=tokenizer,
                    config=current_ot_config,
                )
            elif method == "das":
                payload = run_das_pipeline(
                    model=model,
                    train_bank=train_bank,
                    calibration_bank=calibration_bank,
                    holdout_bank=test_bank,
                    sites=das_sites,
                    device=device,
                    tokenizer=tokenizer,
                    config=DASConfig(
                        batch_size=config.batch_size,
                        max_epochs=config.das_max_epochs,
                        min_epochs=config.das_min_epochs,
                        plateau_patience=config.das_plateau_patience,
                        plateau_rel_delta=config.das_plateau_rel_delta,
                        learning_rate=config.das_learning_rate,
                        subspace_dims=config.das_subspace_dims,
                        store_candidate_holdout_metrics=config.das_store_candidate_holdout_metrics,
                        restarts=config.das_restarts,
                    ),
                )
            else:
                raise ValueError(f"Unsupported method {method}")
            wall_runtime_seconds = perf_counter() - start
            signature_prepare_runtime_seconds = 0.0
            if method in {"ot", "uot"} and prepared_artifacts is not None:
                signature_prepare_runtime_seconds = float(prepared_artifacts.get("prepare_runtime_seconds", 0.0))
            reported_runtime_seconds = float(wall_runtime_seconds) + float(signature_prepare_runtime_seconds)
            payload["signature_prepare_runtime_seconds"] = float(signature_prepare_runtime_seconds)
            payload["wall_runtime_seconds"] = float(wall_runtime_seconds)
            payload["runtime_seconds"] = float(reported_runtime_seconds)
            method_payloads[method].append(payload)
            all_records.extend(payload["results"])
            print(
                f"[method] done method={method} target={target_var} "
                f"runtime={float(reported_runtime_seconds):.2f}s "
                f"(wall={float(wall_runtime_seconds):.2f}s, signatures={float(signature_prepare_runtime_seconds):.2f}s)"
            )
    summary_records = summarize_method_records(all_records)
    summary_text = format_summary(
        model_name=config.model_name,
        data_metadata=data_metadata,
        method_payloads=method_payloads,
        summary_records=summary_records,
    )
    config_payload = {
        **asdict(config),
        "output_path": str(config.output_path),
        "summary_path": str(config.summary_path),
        "target_vars": list(target_vars),
    }
    payload = {
        "config": config_payload,
        "model_name": config.model_name,
        "methods": list(config.methods),
        "target_vars": list(target_vars),
        "signature_mode": config.signature_mode,
        "ot_epsilon": float(config.ot_epsilon),
        "uot_beta_neural": float(config.uot_beta_neural),
        "resolution": None if config.resolution is None else int(config.resolution),
        "num_candidate_sites": len(ot_sites),
        "candidate_sites": [site.label for site in ot_sites],
        "num_das_candidate_sites": len(das_sites),
        "das_candidate_sites": [site.label for site in das_sites],
        "data": data_metadata,
        "method_payloads": method_payloads,
        "results": all_records,
        "method_summary": summary_records,
    }
    write_json(config.output_path, payload)
    write_text_report(config.summary_path, summary_text)
    print_results_table(all_records, "MCQA Counterfactual Test Results")
    print_results_table(summary_records, "MCQA Method Average Summary")
    print(f"Wrote comparison results to {Path(config.output_path).resolve()}")
    print(f"Wrote comparison summary to {Path(config.summary_path).resolve()}")
    return payload
