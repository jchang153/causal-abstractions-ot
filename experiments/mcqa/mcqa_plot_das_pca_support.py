#!/usr/bin/env python3
"""Run PLOT-PCA-DAS directly from one cached Stage B PCA support payload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import mcqa_run as base_run
from mcqa_experiment.data import MCQA_PARTITION_PROTOCOL
from mcqa_experiment.pca import load_pca_basis
from mcqa_experiment.runtime import write_json
from mcqa_ot_pca_focus import (
    _enumerate_pca_sites,
    _guided_subspace_dims,
    _pca_effective_dims,
    _run_pca_das_from_support,
    _site_catalog_tag,
)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=200)
    parser.add_argument("--test-pool-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pca-support-path", type=Path, required=True)
    parser.add_argument("--target-vars", default="answer_pointer,answer_token")
    parser.add_argument("--guided-subspace-dims", default=None)
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--guided-restarts", type=int, default=1)
    parser.add_argument("--results-root", type=Path, default=Path("results/delta"))
    parser.add_argument("--results-timestamp", default=None)
    parser.add_argument("--signatures-dir", type=Path, default=Path("signatures"))
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _load_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict payload at {path}")
    return payload


def _resolve_basis_path(payload: dict[str, object], support_path: Path) -> Path:
    recorded = Path(str(payload["basis_path"]))
    if recorded.exists():
        return recorded
    local = support_path.parent / recorded.name
    if local.exists():
        return local
    raise FileNotFoundError(
        f"Cached PCA basis is unavailable at recorded={recorded} or adjacent={local}"
    )


def main() -> None:
    args = _parser().parse_args()
    stage_wall_start = perf_counter()
    support_payload = _load_payload(args.pca_support_path)
    support_by_var = support_payload.get("support_by_var")
    if not isinstance(support_by_var, dict):
        raise ValueError(f"PCA support payload lacks support_by_var: {args.pca_support_path}")

    layer = int(support_payload["layer"])
    token_position_id = str(support_payload["token_position_id"])
    site_menu = str(support_payload["site_menu"])
    num_bands = int(support_payload["num_bands"])
    band_scheme = str(support_payload["band_scheme"])
    basis_source_mode = str(support_payload["basis_source_mode"])
    signature_mode = str(support_payload["signature_mode"])
    basis_path = _resolve_basis_path(support_payload, args.pca_support_path)
    basis = load_pca_basis(basis_path)
    pca_sites = _enumerate_pca_sites(
        basis=basis,
        token_position_id=token_position_id,
        layer=layer,
        site_menu=site_menu,
        num_bands=num_bands,
        band_scheme=band_scheme,
    )
    pca_bases_by_id = {str(basis.basis_id): basis}
    effective_dim_by_var = _pca_effective_dims(
        support_by_var,
        rank=int(basis.rank),
        num_bands=int(num_bands),
    )

    timestamp = args.results_timestamp or os.environ.get("RESULTS_TIMESTAMP") or "mcqa_plot_das_pca_support"
    run_root = args.results_root / f"{timestamp}_mcqa_plot_das_pca_support"
    layer_dir = run_root / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    base_run.DEVICE = str(args.device)
    base_run.MODEL_NAME = str(args.model_name)
    base_run.MCQA_DATASET_PATH = str(args.dataset_path)
    base_run.MCQA_DATASET_CONFIG = args.dataset_config or None
    base_run.DATASET_SIZE = int(args.dataset_size)
    base_run.SPLIT_SEED = int(args.split_seed)
    base_run.TRAIN_POOL_SIZE = int(args.train_pool_size)
    base_run.CALIBRATION_POOL_SIZE = int(args.calibration_pool_size)
    base_run.TEST_POOL_SIZE = int(args.test_pool_size)
    base_run.BATCH_SIZE = int(args.batch_size)
    base_run.TARGET_VARS = ["answer_pointer", "answer_token"]
    base_run.TOKEN_POSITION_IDS = [token_position_id]
    base_run.PROMPT_HF_LOGIN = bool(args.prompt_hf_login)
    base_run.RUN_TIMESTAMP = str(timestamp)
    base_run.RUN_DIR = run_root
    base_run.OUTPUT_PATH = run_root / "mcqa_run_results.json"
    base_run.SUMMARY_PATH = run_root / "mcqa_run_summary.txt"
    base_run.SIGNATURES_DIR = Path(args.signatures_dir)

    context = base_run.build_run_context()
    support_data = support_payload.get("data", {})
    support_partition = support_data.get("partition", {}) if isinstance(support_data, dict) else {}
    if support_partition != context["data_metadata"].get("partition", {}):
        raise ValueError(
            f"PCA support payload {args.pca_support_path} uses a different or legacy MCQA data partition"
        )
    model = context["model"]
    tokenizer = context["tokenizer"]
    banks_by_split = context["banks_by_split"]
    device = context["device"]
    requested_targets = _csv_strings(args.target_vars)
    target_vars = tuple(target for target in requested_targets if target in support_by_var)
    if not target_vars:
        raise ValueError(
            f"None of requested targets {requested_targets} exist in cached support {args.pca_support_path}"
        )
    site_catalog_tag = _site_catalog_tag(
        site_menu=site_menu,
        num_bands=num_bands,
        band_scheme=band_scheme,
    )
    explicit_dims = _csv_ints(args.guided_subspace_dims)
    guided_payloads = _run_pca_das_from_support(
        model=model,
        tokenizer=tokenizer,
        banks_by_split=banks_by_split,
        device=device,
        layer_dir=layer_dir,
        layer=layer,
        token_position_id=token_position_id,
        site_catalog_tag=site_catalog_tag,
        basis_source_mode=basis_source_mode,
        signature_mode=signature_mode,
        target_vars=target_vars,
        support_by_var=support_by_var,
        pca_sites=pca_sites,
        pca_bases_by_id=pca_bases_by_id,
        model_hidden_size=int(model.config.hidden_size),
        batch_size=int(args.batch_size),
        enabled=True,
        method_suffix="das_guided",
        method_name="das_pca_guided",
        title="MCQA Cached PLOT-PCA Guided DAS Summary",
        mask_names=("Selected",),
        max_epochs=int(args.guided_max_epochs),
        min_epochs=int(args.guided_min_epochs),
        plateau_patience=int(base_run.DAS_PLATEAU_PATIENCE),
        plateau_rel_delta=float(base_run.DAS_PLATEAU_REL_DELTA),
        learning_rate=float(base_run.DAS_LEARNING_RATE),
        explicit_subspace_dims=explicit_dims,
        subspace_dim_resolver=_guided_subspace_dims,
        restarts=max(1, int(args.guided_restarts)),
        full_pca_basis=True,
        effective_dim_by_var=effective_dim_by_var,
    )
    guided_output_paths = {
        target_var: str(
            layer_dir
            / (
                f"mcqa_layer-{layer}_pos-{token_position_id}_pca-{site_catalog_tag}"
                f"_basis-{basis_source_mode}_sig-{signature_mode}_{target_var}_das_guided.json"
            )
        )
        for target_var in guided_payloads
    }
    core_method_runtime_seconds = float(
        sum(float(payload.get("runtime_seconds", 0.0)) for payload in guided_payloads.values())
    )
    summary = {
        "kind": "mcqa_plot_das_pca_support_cached",
        "layer": layer,
        "token_position_id": token_position_id,
        "basis_source_mode": basis_source_mode,
        "site_menu": site_menu,
        "num_bands": num_bands,
        "band_scheme": band_scheme,
        "target_vars": list(target_vars),
        "pca_support_path": str(args.pca_support_path),
        "basis_path": str(basis_path),
        "pca_rank": int(basis.rank),
        "effective_dim_by_var": effective_dim_by_var,
        "das_search_space": "full_pca_basis",
        "localization_recomputed": False,
        "guided_output_paths": guided_output_paths,
        "data": context["data_metadata"],
        "partition_protocol": MCQA_PARTITION_PROTOCOL,
        "runtime_seconds": core_method_runtime_seconds,
        "core_method_runtime_seconds": core_method_runtime_seconds,
        "stage_wall_runtime_seconds": float(perf_counter() - stage_wall_start),
        "runtime_accounting": (
            "sum of per-target guided DAS method runtimes; excludes model loading, dataset "
            "loading/filtering, pair-bank construction, cached support/PCA loading, and wrapper setup"
        ),
    }
    summary_path = layer_dir / f"mcqa_plot_das_pca_support_layer-{layer}_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote cached PCA-support DAS payload to {summary_path}")


if __name__ == "__main__":
    main()
