from __future__ import annotations

from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import os
from time import perf_counter

from huggingface_hub import login as hf_login
import torch

from mcqa_experiment.compare_runner import CompareExperimentConfig, run_comparison
import mcqa_experiment.data as mcqa_data
from mcqa_experiment.data import build_pair_banks, canonicalize_target_var, load_filtered_mcqa_pipeline
from mcqa_experiment.ot import (
    OTConfig,
    load_prepared_alignment_artifacts,
    prepare_alignment_artifacts,
    save_prepared_alignment_artifacts,
)
from mcqa_experiment.runtime import resolve_device
from mcqa_experiment.sites import enumerate_residual_sites


DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_mcqa"
OUTPUT_PATH = RUN_DIR / "mcqa_run_results.json"
SUMMARY_PATH = RUN_DIR / "mcqa_run_summary.txt"
SIGNATURES_DIR = Path("signatures")
SPLIT_PRINT_ORDER = ("train", "calibration", "test")

MODEL_NAME = "google/gemma-2-2b"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
PROMPT_HF_LOGIN = True

# Data
MCQA_DATASET_PATH = "jchang153/copycolors_mcqa"
MCQA_DATASET_CONFIG = None
# MCQA_DATASET_PATH = "mib-bench/copycolors_mcqa"
# MCQA_DATASET_CONFIG = "4_answer_choices"
DATASET_SIZE = 2000  # Cap raw rows loaded from the dataset before factual filtering.
SPLIT_SEED = 0
TRAIN_POOL_SIZE = 200
CALIBRATION_POOL_SIZE = 100
TEST_POOL_SIZE = 100

# Experiment
METHODS = ["ot"]
TARGET_VARS = ["answer_pointer", "answer_token"]
COUNTERFACTUAL_NAMES = ["answerPosition", "randomLetter", "answerPosition_randomLetter"]

LAYERS = "auto"
TOKEN_POSITION_IDS = ["correct_symbol", "correct_symbol_period", "last_token"]

BATCH_SIZE = 64 

RESOLUTIONS = [None]
OT_EPSILONS = [0.5, 1.0, 2.0]
UOT_BETA_NEURALS = [0.1, 0.3, 1.0, 3.0]
SIGNATURE_MODES = ["family_slot_label_delta"]
OT_TOP_K_VALUES = [1, 2, 3]
OT_LAMBDAS = [0.5, 1.0, 2.0]
CALIBRATION_METRIC = "exact_acc"
CALIBRATION_FAMILY_WEIGHTS = [1.0, 1.0, 1.0]

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-3
DAS_LEARNING_RATE = 1e-3
DAS_RESTARTS = 2
DAS_SUBSPACE_DIMS = [
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


def _resolution_tag(resolution: int | None) -> str:
    return "full" if resolution is None else str(int(resolution))


def _resolve_resolution(*, resolution: int | None, hidden_size: int) -> int | None:
    return None if resolution is None else max(1, min(int(resolution), int(hidden_size)))


def _signature_cache_spec(
    *,
    train_bank,
    resolution: int | None,
    resolved_resolution: int | None,
    signature_mode: str,
    selected_layers: list[int],
    token_position_ids: tuple[str, ...],
) -> dict[str, object]:
    train_rows_digest = hashlib.sha256(
        "\n".join(
            f"{base.get('raw_input', '')}|||{source.get('raw_input', '')}"
            for base, source in zip(train_bank.base_inputs, train_bank.source_inputs)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "mcqa_alignment_signatures",
        "model_name": MODEL_NAME,
        "dataset_path": MCQA_DATASET_PATH,
        "dataset_config": MCQA_DATASET_CONFIG,
        "counterfactual_names": list(COUNTERFACTUAL_NAMES),
        "split_seed": int(SPLIT_SEED),
        "resolution": _resolution_tag(resolution),
        "resolved_resolution": int(resolved_resolution) if resolved_resolution is not None else "full",
        "signature_mode": str(signature_mode),
        "train_pool_size": int(TRAIN_POOL_SIZE),
        "train_bank": train_bank.metadata(),
        "source_target_vars": [canonicalize_target_var(target_var) for target_var in TARGET_VARS],
        "train_rows_digest": train_rows_digest,
        "selected_layers": [int(layer) for layer in selected_layers],
        "token_position_ids": list(token_position_ids if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS)),
        "batch_size": int(BATCH_SIZE),
    }


def _signature_cache_path(*, resolution: int | None, signature_mode: str, cache_spec: dict[str, object]) -> Path:
    spec_json = json.dumps(cache_spec, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()[:12]
    stem = (
        f"mcqa_res-{_resolution_tag(resolution)}_sig-{str(signature_mode)}"
        f"_train-{int(cache_spec['train_pool_size'])}_{spec_hash}.pt"
    )
    return SIGNATURES_DIR / stem


def ensure_hf_login(token: str | None, prompt_login: bool) -> str | None:
    if token:
        return token
    if prompt_login:
        hf_login(add_to_git_credential=False)
    return token


def _load_existing_run_payload(output_path: Path) -> dict[str, object] | None:
    if not output_path.exists():
        return None
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def build_run_context() -> dict[str, object]:
    context_start = perf_counter()
    device = resolve_device(DEVICE)
    hf_token = ensure_hf_login(HF_TOKEN, PROMPT_HF_LOGIN)
    print(f"[run] starting MCQA run model={MODEL_NAME} device={device}")
    load_start = perf_counter()
    model, tokenizer, causal_model, token_positions, filtered_datasets = load_filtered_mcqa_pipeline(
        model_name=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        dataset_size=DATASET_SIZE,
        hf_token=hf_token,
        dataset_path=MCQA_DATASET_PATH,
        dataset_name=MCQA_DATASET_CONFIG,
    )
    load_pipeline_seconds = float(perf_counter() - load_start)
    pipeline_timing_seconds = dict(getattr(mcqa_data, "LAST_PIPELINE_TIMING_SECONDS", {}))
    print("[run] building pair banks")
    bank_start = perf_counter()
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
        datasets_by_name=filtered_datasets,
        counterfactual_names=tuple(COUNTERFACTUAL_NAMES),
        target_vars=tuple(TARGET_VARS),
        split_seed=SPLIT_SEED,
        train_pool_size=TRAIN_POOL_SIZE,
        calibration_pool_size=CALIBRATION_POOL_SIZE,
        test_pool_size=TEST_POOL_SIZE,
    )
    bank_build_seconds = float(perf_counter() - bank_start)
    total_context_seconds = float(perf_counter() - context_start)
    print(f"[run] built splits={list(banks_by_split.keys())}")
    for split in SPLIT_PRINT_ORDER:
        split_metadata = data_metadata.get(split)
        if isinstance(split_metadata, dict) and split_metadata:
            dataset_names = next(iter(split_metadata.values())).get("dataset_names", [])
            print(f"{split} pair bank | datasets={dataset_names}")
            for variable, variable_stats in sorted(split_metadata.items()):
                total_pairs = int(variable_stats.get("size", 0))
                changed_count = int(variable_stats.get("changed_count", 0))
                unchanged_count = max(0, total_pairs - changed_count)
                print(
                    f"{split} pair bank [{variable}] | changed={changed_count} "
                    f"| unchanged={unchanged_count} "
                    f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f}"
                )
    return {
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
        "causal_model": causal_model,
        "token_positions": token_positions,
        "filtered_datasets": filtered_datasets,
        "banks_by_split": banks_by_split,
        "data_metadata": data_metadata,
        "timing_seconds": {
            **pipeline_timing_seconds,
            "t_model_data_filter_load": load_pipeline_seconds,
            "t_bank_build": bank_build_seconds,
            "t_context_total_wall": total_context_seconds,
        },
    }


def execute_run_context(*, context: dict[str, object]) -> None:
    device = context["device"]
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_positions = context["token_positions"]
    banks_by_split = context["banks_by_split"]
    data_metadata = context["data_metadata"]
    selected_layers = list(range(int(model.config.num_hidden_layers))) if LAYERS == "auto" else list(LAYERS)
    target_vars = [canonicalize_target_var(target_var) for target_var in TARGET_VARS]
    print(f"[run] selected_layers={selected_layers}")
    layer_suffix = ""
    if len(selected_layers) == 1:
        layer_suffix = f"_layer-{int(selected_layers[0])}"
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    all_payloads = []

    def run_or_resume(config: CompareExperimentConfig, prepared_artifacts):
        existing_payload = _load_existing_run_payload(Path(config.output_path))
        if existing_payload is not None:
            print(f"[resume] reusing existing output {Path(config.output_path).resolve()}")
            return existing_payload
        return run_comparison(
            model=model,
            tokenizer=tokenizer,
            token_positions=token_positions,
            banks_by_split=banks_by_split,
            data_metadata=data_metadata,
            device=device,
            config=config,
            prepared_ot_artifacts=prepared_artifacts,
        )

    transport_methods = tuple(method for method in METHODS if method in {"ot", "uot"})
    nontransport_methods = tuple(method for method in METHODS if method not in {"ot", "uot"})
    for signature_mode in SIGNATURE_MODES:
        for resolution in RESOLUTIONS:
            resolved_resolution = _resolve_resolution(
                resolution=resolution,
                hidden_size=int(model.config.hidden_size),
            )
            resolution_tag = _resolution_tag(resolution)
            prepared_ot_artifacts = None
            if transport_methods and TARGET_VARS:
                ot_sites = enumerate_residual_sites(
                    num_layers=int(model.config.num_hidden_layers),
                    hidden_size=int(model.config.hidden_size),
                    token_position_ids=token_position_ids,
                    resolution=resolved_resolution,
                    layers=tuple(selected_layers),
                    selected_token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                )
                train_banks = {target_var: banks_by_split["train"][target_var] for target_var in target_vars}
                cache_spec = _signature_cache_spec(
                    train_bank=train_banks[target_vars[0]],
                    resolution=resolution,
                    resolved_resolution=resolved_resolution,
                    signature_mode=signature_mode,
                    selected_layers=selected_layers,
                    token_position_ids=token_position_ids,
                )
                cache_path = _signature_cache_path(
                    resolution=resolution,
                    signature_mode=signature_mode,
                    cache_spec=cache_spec,
                )
                prepared_ot_artifacts = load_prepared_alignment_artifacts(
                    cache_path,
                    expected_spec=cache_spec,
                )
                if prepared_ot_artifacts is not None:
                    print(
                        f"[signatures] loaded cache path={cache_path} "
                        f"prepare_time={float(prepared_ot_artifacts.get('prepare_runtime_seconds', 0.0)):.2f}s"
                    )
                else:
                    prepared_ot_artifacts = prepare_alignment_artifacts(
                        model=model,
                        fit_banks_by_var=train_banks,
                        sites=ot_sites,
                        device=device,
                        config=OTConfig(
                            method="ot",
                            batch_size=BATCH_SIZE,
                            epsilon=1.0,
                            signature_mode=signature_mode,
                            top_k_values=tuple(OT_TOP_K_VALUES),
                            lambda_values=tuple(OT_LAMBDAS),
                            source_target_vars=tuple(TARGET_VARS),
                            calibration_metric=CALIBRATION_METRIC,
                            calibration_family_weights=tuple(CALIBRATION_FAMILY_WEIGHTS),
                        ),
                    )
                    prepared_ot_artifacts["cache_spec"] = cache_spec
                    prepared_ot_artifacts["cache_path"] = str(cache_path)
                    save_prepared_alignment_artifacts(
                        cache_path,
                        prepared_artifacts=prepared_ot_artifacts,
                        cache_spec=cache_spec,
                    )
                    print(
                        f"[signatures] saved cache path={cache_path} "
                        f"prepare_time={float(prepared_ot_artifacts.get('prepare_runtime_seconds', 0.0)):.2f}s"
                    )

            for method in transport_methods:
                epsilon_values = OT_EPSILONS
                for epsilon in epsilon_values:
                    if method == "uot":
                        for beta_neural in UOT_BETA_NEURALS:
                            uot_config = CompareExperimentConfig(
                                model_name=MODEL_NAME,
                                output_path=RUN_DIR / (
                                    f"mcqa_uot{layer_suffix}_res-{resolution_tag}_sig-{signature_mode}_eps-{float(epsilon):g}_"
                                    f"bn-{beta_neural:g}.json"
                                ),
                                summary_path=RUN_DIR / (
                                    f"mcqa_uot{layer_suffix}_res-{resolution_tag}_sig-{signature_mode}_eps-{float(epsilon):g}_"
                                    f"bn-{beta_neural:g}.txt"
                                ),
                                methods=("uot",),
                                target_vars=tuple(target_vars),
                                batch_size=BATCH_SIZE,
                                ot_epsilon=float(epsilon),
                                uot_beta_neural=float(beta_neural),
                                signature_mode=signature_mode,
                                ot_top_k_values=tuple(OT_TOP_K_VALUES),
                                ot_lambdas=tuple(OT_LAMBDAS),
                                resolution=resolved_resolution,
                                layers=tuple(selected_layers),
                                token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                                calibration_metric=CALIBRATION_METRIC,
                                calibration_family_weights=tuple(CALIBRATION_FAMILY_WEIGHTS),
                            )
                            all_payloads.append(
                                run_or_resume(
                                    uot_config,
                                    prepared_ot_artifacts,
                                )
                            )
                    else:
                        output_stem = f"mcqa{layer_suffix}_res-{resolution_tag}_sig-{signature_mode}_eps-{float(epsilon):g}"
                        config = CompareExperimentConfig(
                            model_name=MODEL_NAME,
                            output_path=RUN_DIR / f"{output_stem}.json",
                            summary_path=RUN_DIR / f"{output_stem}.txt",
                            methods=("ot",),
                            target_vars=tuple(target_vars),
                            batch_size=BATCH_SIZE,
                            ot_epsilon=float(epsilon),
                            signature_mode=signature_mode,
                            ot_top_k_values=tuple(OT_TOP_K_VALUES),
                            ot_lambdas=tuple(OT_LAMBDAS),
                            das_max_epochs=DAS_MAX_EPOCHS,
                            das_min_epochs=DAS_MIN_EPOCHS,
                            das_plateau_patience=DAS_PLATEAU_PATIENCE,
                            das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
                            das_learning_rate=DAS_LEARNING_RATE,
                            das_subspace_dims=tuple(DAS_SUBSPACE_DIMS),
                            resolution=resolved_resolution,
                            layers=tuple(selected_layers),
                            token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                            calibration_metric=CALIBRATION_METRIC,
                            calibration_family_weights=tuple(CALIBRATION_FAMILY_WEIGHTS),
                        )
                        all_payloads.append(
                            run_or_resume(
                                config,
                                prepared_ot_artifacts,
                            )
                        )

            for method in nontransport_methods:
                output_stem = f"mcqa_{method}{layer_suffix}_res-{resolution_tag}_sig-{signature_mode}"
                config = CompareExperimentConfig(
                    model_name=MODEL_NAME,
                    output_path=RUN_DIR / f"{output_stem}.json",
                    summary_path=RUN_DIR / f"{output_stem}.txt",
                    methods=(method,),
                    target_vars=tuple(target_vars),
                    batch_size=BATCH_SIZE,
                    ot_epsilon=1.0,
                    signature_mode=signature_mode,
                    ot_top_k_values=tuple(OT_TOP_K_VALUES),
                    ot_lambdas=tuple(OT_LAMBDAS),
                    das_max_epochs=DAS_MAX_EPOCHS,
                    das_min_epochs=DAS_MIN_EPOCHS,
                    das_plateau_patience=DAS_PLATEAU_PATIENCE,
                    das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
                    das_learning_rate=DAS_LEARNING_RATE,
                    das_restarts=DAS_RESTARTS,
                    das_subspace_dims=tuple(DAS_SUBSPACE_DIMS),
                    resolution=resolved_resolution,
                    layers=tuple(selected_layers),
                    token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                    calibration_metric=CALIBRATION_METRIC,
                    calibration_family_weights=tuple(CALIBRATION_FAMILY_WEIGHTS),
                )
                all_payloads.append(
                    run_or_resume(
                        config,
                        None,
                    )
                )
    from mcqa_experiment.runtime import write_json

    write_json(OUTPUT_PATH, {"runs": all_payloads})
    print(f"Wrote aggregate MCQA run payload to {OUTPUT_PATH.resolve()}")


def main() -> None:
    context = build_run_context()
    execute_run_context(context=context)


if __name__ == "__main__":
    main()
