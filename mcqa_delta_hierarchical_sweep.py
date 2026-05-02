from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

from mcqa_paper_runtime import write_paper_runtime_summary


DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_STAGE_A_TOKEN_POSITION_IDS = ("last_token",)
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.5, 2.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_PCA_SITE_MENUS = ("partition", "mixed")
DEFAULT_PCA_BASIS_SOURCE_MODES = ("pair_bank", "all_variants")
DEFAULT_PCA_NUM_BANDS_VALUES = (8, 16)
DEFAULT_PCA_BAND_SCHEME = "equal"
DEFAULT_PCA_TOP_PREFIX_SIZES = (8, 16, 32, 64)
DEFAULT_GUIDED_MASK_NAMES = ("Top1", "Top2", "Top4", "S50", "S80")
DEFAULT_NATIVE_BLOCK_RESOLUTIONS = (128, 144, 192, 256, 288, 384, 576, 768)
DEFAULT_DAS_SUBSPACE_DIMS = (
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
)
DEFAULT_STAGES = (
    "stage_a_plot_layer",
    "stage_b_plot_native_support",
    "stage_b_plot_pca_support",
    "stage_c_plot_das_layer",
    "stage_c_plot_das_native_support",
    "stage_c_plot_das_pca_support",
)


@dataclass(frozen=True)
class SweepStage:
    name: str
    category: str
    description: str
    stage_timestamp: str
    command: tuple[str, ...]
    expected_outputs: tuple[str, ...]


def _parse_csv_strings(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _parse_csv_ints(value: str | None) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv_strings(value))


def _parse_csv_floats(value: str | None) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_csv_strings(value))


def _append_optional_arg(args: list[str], name: str, value: str | None) -> None:
    if value is None or value == "":
        return
    args.extend([name, value])


def _score_tuple(entry: dict[str, object]) -> tuple[float, float]:
    return (
        float(entry.get("exact_acc", -1.0)),
        float(entry.get("selection_score", entry.get("cal", -1.0))),
    )


def _sort_best_first(entries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        entries,
        key=lambda entry: (
            float(entry.get("selection_score", entry.get("cal", -1.0))),
            float(entry.get("exact_acc", -1.0)),
            -int(entry.get("layer", 10**9)),
        ),
        reverse=True,
    )


def _selection_score(record: dict[str, object]) -> float:
    if "selection_score" in record and record["selection_score"] is not None:
        return float(record["selection_score"])
    if "selection_exact_acc" in record and record["selection_exact_acc"] is not None:
        return float(record["selection_exact_acc"])
    if "calibration_exact_acc" in record and record["calibration_exact_acc"] is not None:
        return float(record["calibration_exact_acc"])
    return -1.0


def _best_result_record(payloads: Iterable[dict[str, object]]) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for payload in payloads:
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            continue
        result = results[0]
        if not isinstance(result, dict):
            continue
        candidate = {
            "epsilon": float(payload.get("_ot_epsilon", payload.get("ot_epsilon", -1.0))),
            "selection_score": _selection_score(result),
            "exact_acc": float(result.get("exact_acc", -1.0)),
            "site_label": result.get("site_label"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "wall_runtime_seconds": payload.get("wall_runtime_seconds"),
            "signature_prepare_runtime_seconds": payload.get("signature_prepare_runtime_seconds"),
            "result": result,
            "payload": payload,
        }
        if best is None or (
            float(candidate["selection_score"]),
            float(candidate["exact_acc"]),
        ) > (
            float(best["selection_score"]),
            float(best["exact_acc"]),
        ):
            best = candidate
    return best


def _site_catalog_tag(*, site_menu: str, num_bands: int, band_scheme: str, top_prefix_sizes: tuple[int, ...]) -> str:
    tag = f"menu-{str(site_menu)}-bands-{int(num_bands)}-scheme-{str(band_scheme)}"
    if str(site_menu) == "mixed":
        tag += f"-top-{'-'.join(str(size) for size in top_prefix_sizes)}"
    return tag


def _stage_b_slug(*, token_position_id: str, basis_source_mode: str, site_menu: str, num_bands: int) -> str:
    return f"{str(token_position_id)}_{str(basis_source_mode)}_{str(site_menu)}_{int(num_bands)}b"


def _normalize_num_bands_values(values: tuple[int, ...], site_menu: str) -> tuple[int, ...]:
    resolved = tuple(sorted(dict.fromkeys(int(value) for value in values)))
    if str(site_menu) == "mixed":
        return tuple(value for value in resolved if int(value) == 8) or (resolved[0],)
    return resolved


def _layer_from_site_label(label: object) -> int | None:
    text = str(label)
    if not text.startswith("L") or ":" not in text:
        return None
    try:
        return int(text[1 : text.index(":")])
    except ValueError:
        return None


def _load_json(path: Path) -> dict[str, object] | list[object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _stage_output_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if path.suffix.lower() != ".json":
        return True
    try:
        payload = _load_json(path)
    except Exception:
        return False
    return isinstance(payload, (dict, list))


def _resolve_num_layers(model_name: str) -> int:
    from transformers import AutoConfig

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Could not resolve num_hidden_layers for model {model_name}")
    return int(config.num_hidden_layers)


def _all_layer_indices(model_name: str) -> tuple[int, ...]:
    num_layers = _resolve_num_layers(model_name)
    return tuple(range(int(num_layers)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hierarchical MCQA sweep for PLOT/PLOT-DAS method families.")
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
    parser.add_argument("--results-root", default="results/delta")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument(
        "--stages",
        default="stage_a_plot_layer,stage_b_plot_native_support,stage_b_plot_pca_support,stage_c_plot_das_layer,stage_c_plot_das_native_support,stage_c_plot_das_pca_support",
    )
    parser.add_argument("--stage-a-token-position-ids", default="last_token")
    parser.add_argument("--stage-a-layer-indices", default=None, help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument("--target-vars", default="answer_pointer,answer_token")
    parser.add_argument("--ot-epsilons", default="0.5,1,2,4")
    parser.add_argument("--ot-top-k-values", default="1,2,4")
    parser.add_argument("--ot-lambdas", default="0.5,1,2,4")
    parser.add_argument("--calibration-metric", default=DEFAULT_CALIBRATION_METRIC)
    parser.add_argument("--calibration-family-weights", default="1,1.5,2")
    parser.add_argument("--stage-b-top-layers-per-var", type=int, default=1)
    parser.add_argument("--stage-b-neighbor-radius", type=int, default=0)
    parser.add_argument("--stage-b-max-layers-per-var", type=int, default=1)
    parser.add_argument("--native-block-resolutions", default="128,144,192,256,288,384,576,768")
    parser.add_argument("--pca-site-menus", default="partition,mixed")
    parser.add_argument("--pca-basis-source-modes", default="pair_bank,all_variants")
    parser.add_argument("--pca-num-bands-values", default="8,16")
    parser.add_argument("--pca-band-scheme", default=DEFAULT_PCA_BAND_SCHEME, choices=("equal", "head"))
    parser.add_argument("--pca-top-prefix-sizes", default="8,16,32,64")
    parser.add_argument("--stage-c-top-configs-per-var", type=int, default=2)
    parser.add_argument("--guided-mask-names", default="Top1,Top2,Top4,S50,S80")
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--screen-restarts", type=int, default=1)
    parser.add_argument("--guided-restarts", type=int, default=2)
    parser.add_argument("--guided-subspace-dims", default="32,64,96,128,256,512,768,1024,1536,2048,2304")
    parser.add_argument("--full-das-output", type=Path, action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--regular-das-subspace-dims",
        default="32,64,96,128,256,512,768,1024,1536,2048,2304",
        help="Comma-separated DAS dimensions for full-layer/A-only DAS baselines.",
    )
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _normalize_args(args: argparse.Namespace) -> dict[str, object]:
    stages = _parse_csv_strings(args.stages) or DEFAULT_STAGES
    unsupported_stage_names = sorted(set(stages) - set(DEFAULT_STAGES))
    if unsupported_stage_names:
        raise ValueError(f"Unsupported stage names: {unsupported_stage_names}")
    target_vars = _parse_csv_strings(args.target_vars) or DEFAULT_TARGET_VARS
    stage_a_token_position_ids = _parse_csv_strings(args.stage_a_token_position_ids) or DEFAULT_STAGE_A_TOKEN_POSITION_IDS
    stage_a_layer_indices = _parse_csv_ints(args.stage_a_layer_indices) if args.stage_a_layer_indices is not None else ()
    pca_site_menus = _parse_csv_strings(args.pca_site_menus) or DEFAULT_PCA_SITE_MENUS
    pca_basis_source_modes = _parse_csv_strings(args.pca_basis_source_modes) or DEFAULT_PCA_BASIS_SOURCE_MODES
    pca_num_bands_values = _parse_csv_ints(args.pca_num_bands_values) or DEFAULT_PCA_NUM_BANDS_VALUES
    pca_top_prefix_sizes = _parse_csv_ints(args.pca_top_prefix_sizes) or DEFAULT_PCA_TOP_PREFIX_SIZES
    native_block_resolutions = _parse_csv_ints(args.native_block_resolutions) or DEFAULT_NATIVE_BLOCK_RESOLUTIONS
    ot_epsilons = _parse_csv_floats(args.ot_epsilons) or DEFAULT_OT_EPSILONS
    ot_top_k_values = _parse_csv_ints(args.ot_top_k_values) or DEFAULT_OT_TOP_K_VALUES
    ot_lambdas = _parse_csv_floats(args.ot_lambdas) or DEFAULT_OT_LAMBDAS
    calibration_family_weights = _parse_csv_floats(args.calibration_family_weights) or DEFAULT_CALIBRATION_FAMILY_WEIGHTS
    guided_mask_names = _parse_csv_strings(args.guided_mask_names) or DEFAULT_GUIDED_MASK_NAMES
    guided_subspace_dims = None
    if args.guided_subspace_dims is not None:
        guided_subspace_dims = _parse_csv_ints(args.guided_subspace_dims)
    regular_das_subspace_dims = _parse_csv_ints(args.regular_das_subspace_dims) or DEFAULT_DAS_SUBSPACE_DIMS
    results_timestamp = (
        args.results_timestamp
        or os.environ.get("RESULTS_TIMESTAMP")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return {
        "stages": tuple(str(stage) for stage in stages),
        "target_vars": tuple(str(target_var) for target_var in target_vars),
        "stage_a_token_position_ids": tuple(str(token_position_id) for token_position_id in stage_a_token_position_ids),
        "stage_a_layer_indices": tuple(int(layer) for layer in stage_a_layer_indices),
        "pca_site_menus": tuple(str(site_menu) for site_menu in pca_site_menus),
        "pca_basis_source_modes": tuple(str(mode) for mode in pca_basis_source_modes),
        "pca_num_bands_values": tuple(int(value) for value in pca_num_bands_values),
        "pca_top_prefix_sizes": tuple(int(size) for size in pca_top_prefix_sizes),
        "native_block_resolutions": tuple(int(resolution) for resolution in native_block_resolutions),
        "ot_epsilons": tuple(float(epsilon) for epsilon in ot_epsilons),
        "ot_top_k_values": tuple(int(value) for value in ot_top_k_values),
        "ot_lambdas": tuple(float(value) for value in ot_lambdas),
        "calibration_family_weights": tuple(float(weight) for weight in calibration_family_weights),
        "guided_mask_names": tuple(str(mask_name) for mask_name in guided_mask_names),
        "guided_subspace_dims": None if guided_subspace_dims is None else tuple(int(dim) for dim in guided_subspace_dims),
        "regular_das_subspace_dims": tuple(int(dim) for dim in regular_das_subspace_dims),
        "results_timestamp": str(results_timestamp),
    }


def _build_stage_a_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    token_position_id: str,
    layer_indices: tuple[int, ...],
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "mcqa_plot_layer.py",
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
        "--layers",
        ",".join(str(layer) for layer in layer_indices),
        "--token-position-id",
        str(token_position_id),
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--signature-mode",
        str(args.signature_mode),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _build_stage_b_or_c_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    token_position_id: str,
    layers: tuple[int, ...],
    basis_source_mode: str,
    site_menu: str,
    num_bands: int,
    guided_das: bool,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "mcqa_ot_pca_focus.py",
        "--device",
        str(args.device),
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
        "--layers",
        ",".join(str(layer) for layer in layers),
        "--token-position-id",
        str(token_position_id),
        "--site-menu",
        str(site_menu),
        "--num-bands",
        str(int(num_bands)),
        "--band-scheme",
        str(args.pca_band_scheme),
        "--top-prefix-sizes",
        ",".join(str(size) for size in normalized["pca_top_prefix_sizes"]),
        "--basis-source-mode",
        str(basis_source_mode),
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--signature-mode",
        str(args.signature_mode),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    if guided_das:
        command.extend(
            [
                "--guided-das",
                "--guided-mask-names",
                ",".join(str(mask_name) for mask_name in normalized["guided_mask_names"]),
                "--guided-max-epochs",
                str(int(args.guided_max_epochs)),
                "--guided-min-epochs",
                str(int(args.guided_min_epochs)),
                "--guided-restarts",
                str(max(1, int(args.guided_restarts))),
            ]
        )
        if normalized["guided_subspace_dims"] is not None:
            command.extend(
                [
                    "--guided-subspace-dims",
                    ",".join(str(dim) for dim in normalized["guided_subspace_dims"]),
                ]
            )
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _build_native_block_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    layers: tuple[int, ...],
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "mcqa_plot_native_support.py",
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
        "--layers",
        ",".join(str(layer) for layer in layers),
        "--block-resolutions",
        ",".join(str(resolution) for resolution in normalized["native_block_resolutions"]),
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--signature-mode",
        str(args.signature_mode),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _build_layer_das_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    token_position_id: str,
    layer: int,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "mcqa_plot_das_layer.py",
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
        "--layer",
        str(int(layer)),
        "--token-position-id",
        str(token_position_id),
        "--signature-mode",
        str(args.signature_mode),
        "--das-subspace-dims",
        ",".join(str(dim) for dim in normalized["regular_das_subspace_dims"]),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _build_native_support_das_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    native_support_path: Path,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "mcqa_plot_das_native_support.py",
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
        "--native-support-path",
        str(native_support_path),
        "--guided-mask-names",
        ",".join(str(mask_name) for mask_name in normalized["guided_mask_names"]),
        "--guided-subspace-dims",
        ",".join(str(dim) for dim in normalized["guided_subspace_dims"] or DEFAULT_DAS_SUBSPACE_DIMS),
        "--guided-max-epochs",
        str(int(args.guided_max_epochs)),
        "--guided-min-epochs",
        str(int(args.guided_min_epochs)),
        "--guided-restarts",
        str(max(1, int(args.guided_restarts))),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _run_stage_command(*, stage: SweepStage, repo_root: Path) -> float:
    start = perf_counter()
    subprocess.run(stage.command, cwd=repo_root, check=True)
    return float(perf_counter() - start)


def _iter_ot_payloads_from_run_payload(run_payload: dict[str, object]) -> Iterable[dict[str, object]]:
    compare_runs = run_payload.get("runs", [])
    if not isinstance(compare_runs, list):
        return
    for compare_payload in compare_runs:
        if not isinstance(compare_payload, dict):
            continue
        method_payloads = compare_payload.get("method_payloads", {})
        if not isinstance(method_payloads, dict):
            continue
        for payload in method_payloads.get("ot", []):
            if not isinstance(payload, dict):
                continue
            enriched = dict(payload)
            enriched["_candidate_sites"] = compare_payload.get("candidate_sites", [])
            enriched["_ot_epsilon"] = float(compare_payload.get("ot_epsilon", -1.0))
            yield enriched


def _stage_a_rankings_from_layer_sweep(*, manifest_path: Path, manifest_payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    manifest_runs = manifest_payload.get("runs", [])
    if not isinstance(manifest_runs, list):
        raise ValueError(f"Malformed Stage A layer-sweep manifest at {manifest_path}")

    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for manifest_record in manifest_runs:
        if not isinstance(manifest_record, dict):
            continue
        layer = int(manifest_record.get("layer", -1))
        output_path = manifest_record.get("output_path")
        if layer < 0 or not output_path:
            continue
        raw_run_path = Path(str(output_path))
        candidate_paths = [raw_run_path]
        if not raw_run_path.is_absolute():
            candidate_paths.extend(
                [
                    manifest_path.parent / raw_run_path.name,
                    manifest_path.parent.parent / raw_run_path,
                ]
            )
        run_path = next((path for path in candidate_paths if _stage_output_is_valid(path)), raw_run_path)
        if not _stage_output_is_valid(run_path):
            continue
        run_payload = _load_json(run_path)
        if not isinstance(run_payload, dict):
            continue
        grouped_payloads: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
        for payload in _iter_ot_payloads_from_run_payload(run_payload):
            target_var = str(payload.get("target_var"))
            grouped_payloads.setdefault(target_var, []).append(payload)
        for target_var, payloads in grouped_payloads.items():
            best = _best_result_record(payloads)
            if best is None:
                continue
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "selection_score": float(best["selection_score"]),
                    "exact_acc": float(best["exact_acc"]),
                    "epsilon": float(best["epsilon"]),
                    "site_label": best.get("site_label"),
                    "runtime_seconds": best.get("runtime_seconds"),
                    "wall_runtime_seconds": best.get("wall_runtime_seconds"),
                    "signature_prepare_runtime_seconds": best.get("signature_prepare_runtime_seconds"),
                    "selection_basis": "calibration",
                    "source_path": str(run_path),
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _stage_a_rankings_from_joint_run(*, aggregate_path: Path, aggregate_payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    # Fallback for older/non-sweep outputs. This records transport mass as diagnostic evidence,
    # but Stage B selection should use the layer-sweep path above whenever possible.
    grouped_payloads: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for payload in _iter_ot_payloads_from_run_payload(aggregate_payload):
        target_var = str(payload.get("target_var"))
        grouped_payloads.setdefault(target_var, []).append(payload)

    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for target_var, payloads in grouped_payloads.items():
        if not payloads:
            continue
        best = _best_result_record(payloads)
        if best is None:
            continue
        evidence_by_layer: dict[int, float] = {}
        target_mass_by_layer: dict[int, float] = {}
        for payload in payloads:
            candidate_sites = list(payload.get("_candidate_sites", []))
            transport = payload.get("normalized_transport", payload.get("transport", []))
            if not isinstance(transport, list) or not transport:
                continue
            source_target_vars = tuple(str(var) for var in payload.get("source_target_vars", [])) or (str(target_var),)
            try:
                target_row_index = int(payload.get("target_var_row_index", source_target_vars.index(str(target_var))))
            except ValueError:
                target_row_index = 0
            row_count = len(transport)
            for site_index, label in enumerate(candidate_sites):
                layer = _layer_from_site_label(label)
                if layer is None:
                    continue
                try:
                    target_mass = float(transport[target_row_index][site_index])
                except (IndexError, TypeError, ValueError):
                    continue
                competitor_mass = 0.0
                for other_row_index in range(row_count):
                    if int(other_row_index) == int(target_row_index):
                        continue
                    try:
                        competitor_mass = max(competitor_mass, float(transport[other_row_index][site_index]))
                    except (IndexError, TypeError, ValueError):
                        continue
                evidence_by_layer[layer] = evidence_by_layer.get(layer, 0.0) + max(target_mass - competitor_mass, 0.0)
                target_mass_by_layer[layer] = target_mass_by_layer.get(layer, 0.0) + target_mass
        normalizer = max(1, len(payloads))
        mean_target_mass_by_layer = {
            int(layer): float(value) / float(normalizer)
            for layer, value in target_mass_by_layer.items()
        }
        total_positive_mass = sum(max(float(value), 0.0) for value in mean_target_mass_by_layer.values())
        for layer in sorted(set(evidence_by_layer) | set(mean_target_mass_by_layer)):
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "selection_score": float(best["selection_score"]),
                    "exact_acc": float(best["exact_acc"]),
                    "epsilon": float(best["epsilon"]),
                    "site_label": best.get("site_label"),
                    "runtime_seconds": best.get("runtime_seconds"),
                    "wall_runtime_seconds": best.get("wall_runtime_seconds"),
                    "signature_prepare_runtime_seconds": best.get("signature_prepare_runtime_seconds"),
                    "row_dominant_mass": float(evidence_by_layer.get(int(layer), 0.0)),
                    "mean_target_mass": float(mean_target_mass_by_layer.get(int(layer), 0.0)),
                    "mass_share": (
                        max(float(mean_target_mass_by_layer.get(int(layer), 0.0)), 0.0) / total_positive_mass
                        if total_positive_mass > 0.0 else 0.0
                    ),
                    "selection_basis": "joint_calibration_with_mass_diagnostic",
                    "source_path": str(aggregate_path),
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = sorted(
            rankings[target_var],
            key=lambda entry: (
                float(entry.get("selection_score", -1.0)),
                float(entry.get("exact_acc", -1.0)),
                float(entry.get("row_dominant_mass", 0.0)),
                -int(entry.get("layer", 10**9)),
            ),
            reverse=True,
        )
    return rankings


def _extract_stage_a_rankings(*, aggregate_path: Path) -> dict[str, list[dict[str, object]]]:
    aggregate_payload = _load_json(aggregate_path)
    if not isinstance(aggregate_payload, dict):
        raise ValueError(f"Unexpected Stage A payload at {aggregate_path}")
    if str(aggregate_payload.get("kind")) == "mcqa_plot_layer":
        rankings = aggregate_payload.get("rankings_by_var", {})
        if not isinstance(rankings, dict):
            raise ValueError(f"Malformed PLOT layer payload at {aggregate_path}")
        return {
            str(target_var): [dict(entry) for entry in entries if isinstance(entry, dict)]
            for target_var, entries in rankings.items()
            if isinstance(entries, list)
        }
    if str(aggregate_path.name) == "layer_sweep_manifest.json":
        return _stage_a_rankings_from_layer_sweep(manifest_path=aggregate_path, manifest_payload=aggregate_payload)
    return _stage_a_rankings_from_joint_run(aggregate_path=aggregate_path, aggregate_payload=aggregate_payload)


def _format_stage_a_summary(*, token_position_id: str, rankings: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        "MCQA Hierarchical Stage A Layer Discovery",
        f"token_position_id: {token_position_id}",
        "",
    ]
    for target_var in DEFAULT_TARGET_VARS:
        lines.append(f"[{target_var}]")
        for entry in rankings.get(target_var, []):
            parts = [
                f"layer={int(entry['layer'])}",
                f"exact={float(entry.get('exact_acc', -1.0)):.4f}",
                f"cal={float(entry.get('selection_score', -1.0)):.4f}",
            ]
            if "epsilon" in entry:
                parts.append(f"eps={float(entry.get('epsilon', -1.0)):g}")
            if entry.get("site_label") is not None:
                parts.append(f"site={entry.get('site_label')}")
            if entry.get("runtime_seconds") is not None:
                parts.append(f"runtime={float(entry['runtime_seconds']):.2f}s")
            if "row_dominant_mass" in entry:
                parts.append(f"row_dom={float(entry.get('row_dominant_mass', 0.0)):.4f}")
            if "mass_share" in entry:
                parts.append(f"mass_share={float(entry.get('mass_share', 0.0)):.4f}")
            lines.append("  - " + " ".join(parts))
        lines.append("")
    return "\n".join(lines)


def _select_stage_b_layers(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_layers_per_var: int,
    neighbor_radius: int = 1,
    max_layers_per_var: int = 5,
) -> tuple[int, ...]:
    selected: list[int] = []
    for target_var in DEFAULT_TARGET_VARS:
        row_entries = rankings.get(target_var, [])
        row_selected: list[int] = [
            int(entry["layer"])
            for entry in row_entries[: max(1, int(top_layers_per_var))]
        ]
        if row_entries:
            best_layer = int(row_entries[0]["layer"])
            for offset in range(-max(0, int(neighbor_radius)), max(0, int(neighbor_radius)) + 1):
                row_selected.append(int(best_layer) + int(offset))
        row_selected = [
            layer
            for layer in sorted(dict.fromkeys(row_selected))
            if layer >= 0
        ][: max(1, int(max_layers_per_var))]
        selected.extend(row_selected)
    return tuple(sorted(dict.fromkeys(selected)))


def _extract_stage_b_best_configs(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int, str, str, int], dict[str, object]] = {}
    for payload_path in payload_paths:
        layer_payload = _load_json(payload_path)
        if not isinstance(layer_payload, dict):
            continue
        localization_runtime_seconds = float(
            layer_payload.get("localization_runtime_seconds", layer_payload.get("runtime_seconds", 0.0))
        )
        layer = int(layer_payload.get("layer"))
        token_position_id = str(layer_payload.get("token_position_id"))
        basis_source_mode = str(layer_payload.get("basis_source_mode"))
        site_menu = str(layer_payload.get("site_menu"))
        num_bands = int(layer_payload.get("num_bands"))
        for ot_path_str in layer_payload.get("ot_output_paths", []):
            compare_payload = _load_json(Path(str(ot_path_str)))
            if not isinstance(compare_payload, dict):
                continue
            epsilon = float(compare_payload.get("ot_epsilon", -1.0))
            method_payloads = compare_payload.get("method_payloads", {}).get("ot", [])
            for method_payload in method_payloads:
                if not isinstance(method_payload, dict):
                    continue
                results = method_payload.get("results", [])
                if not results:
                    continue
                result = results[0]
                if not isinstance(result, dict):
                    continue
                target_var = str(method_payload.get("target_var") or result.get("variable"))
                key = (target_var, token_position_id, layer, basis_source_mode, site_menu, num_bands)
                entry = {
                    "variable": target_var,
                    "token_position_id": token_position_id,
                    "layer": layer,
                    "basis_source_mode": basis_source_mode,
                    "site_menu": site_menu,
                    "num_bands": num_bands,
                    "exact_acc": float(result.get("exact_acc", -1.0)),
                    "selection_score": _selection_score(result),
                    "epsilon": epsilon,
                    "site_label": result.get("site_label"),
                    "runtime_seconds": float(localization_runtime_seconds),
                    "wall_runtime_seconds": float(localization_runtime_seconds),
                    "signature_prepare_runtime_seconds": method_payload.get("signature_prepare_runtime_seconds"),
                    "layer_payload_path": str(payload_path),
                }
                current = grouped.get(key)
                if current is None or _score_tuple(entry) > _score_tuple(current):
                    grouped[key] = entry
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for entry in grouped.values():
        rankings.setdefault(str(entry["variable"]), []).append(entry)
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _format_stage_b_summary(*, rankings: dict[str, list[dict[str, object]]]) -> str:
    lines = ["MCQA Hierarchical Stage B PCA OT Ranking", ""]
    for target_var in DEFAULT_TARGET_VARS:
        lines.append(f"[{target_var}]")
        for entry in rankings.get(target_var, []):
            parts = [
                f"layer={int(entry['layer'])}",
                f"pos={entry['token_position_id']}",
                f"basis={entry['basis_source_mode']}",
                f"menu={entry['site_menu']}",
                f"bands={int(entry['num_bands'])}",
                f"exact={float(entry['exact_acc']):.4f}",
                f"cal={float(entry['selection_score']):.4f}",
                f"eps={float(entry['epsilon']):g}",
                f"site={entry.get('site_label')}",
            ]
            if entry.get("runtime_seconds") is not None:
                parts.append(f"runtime={float(entry['runtime_seconds']):.2f}s")
            lines.append("  - " + " ".join(parts))
        lines.append("")
    return "\n".join(lines)


def _extract_native_support_rankings(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for payload_path in payload_paths:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict) or str(payload.get("kind")) != "mcqa_plot_native_support_layer":
            continue
        layer = int(payload.get("layer"))
        atomic_width = int(payload.get("atomic_width"))
        runtime_seconds = float(payload.get("localization_runtime_seconds", payload.get("runtime_seconds", 0.0)))
        method_by_var = payload.get("method_by_var", {})
        support_by_var = payload.get("support_by_var", {})
        for target_var, method_summary in method_by_var.items():
            if not isinstance(method_summary, dict):
                continue
            support_summary = support_by_var.get(str(target_var), {}) if isinstance(support_by_var, dict) else {}
            top_site_label = None
            if isinstance(support_summary, dict):
                ranked_site_labels = support_summary.get("ranked_site_labels", [])
                if isinstance(ranked_site_labels, list) and ranked_site_labels:
                    top_site_label = ranked_site_labels[0]
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "atomic_width": int(atomic_width),
                    "exact_acc": float(method_summary.get("exact_acc", 0.0)),
                    "selection_score": float(method_summary.get("selection_score", 0.0)),
                    "epsilon": float(method_summary.get("epsilon", 0.0)),
                    "site_label": top_site_label or method_summary.get("site_label"),
                    "runtime_seconds": float(runtime_seconds),
                    "payload_path": str(payload_path),
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _extract_layer_das_rankings(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for payload_path in payload_paths:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict) or str(payload.get("kind")) != "mcqa_plot_das_layer":
            continue
        layer = int(payload.get("layer"))
        runtime_seconds = float(payload.get("runtime_seconds", 0.0))
        method_payloads = payload.get("method_payloads", {}).get("das", [])
        for method_payload in method_payloads:
            if not isinstance(method_payload, dict):
                continue
            results = method_payload.get("results", [])
            if not results or not isinstance(results[0], dict):
                continue
            result = results[0]
            target_var = str(method_payload.get("target_var") or result.get("variable"))
            rankings.setdefault(target_var, []).append(
                {
                    "variable": target_var,
                    "layer": int(layer),
                    "exact_acc": float(result.get("exact_acc", 0.0)),
                    "selection_score": _selection_score(result),
                    "site_label": result.get("site_label"),
                    "subspace_dim": result.get("subspace_dim"),
                    "site_total_dim": result.get("site_total_dim"),
                    "runtime_seconds": float(runtime_seconds),
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _extract_native_support_das_rankings(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for payload_path in payload_paths:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict) or str(payload.get("kind")) != "mcqa_plot_das_native_support":
            continue
        layer = int(payload.get("layer"))
        for target_var, das_payload in payload.get("payloads_by_var", {}).items():
            if not isinstance(das_payload, dict):
                continue
            results = das_payload.get("results", [])
            if not results or not isinstance(results[0], dict):
                continue
            result = results[0]
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "exact_acc": float(result.get("exact_acc", 0.0)),
                    "selection_score": _selection_score(result),
                    "site_label": result.get("site_label"),
                    "subspace_dim": result.get("subspace_dim"),
                    "site_total_dim": result.get("site_total_dim"),
                    "runtime_seconds": float(das_payload.get("runtime_seconds", payload.get("runtime_seconds", 0.0))),
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _format_native_summary(*, title: str, rankings: dict[str, list[dict[str, object]]]) -> str:
    lines = [title, ""]
    for target_var in DEFAULT_TARGET_VARS:
        lines.append(f"[{target_var}]")
        for entry in rankings.get(target_var, []):
            width_value = entry.get("resolution", entry.get("atomic_width"))
            parts = [
                f"layer={int(entry['layer'])}",
                f"width={width_value}",
                f"exact={float(entry['exact_acc']):.4f}",
                f"cal={float(entry['selection_score']):.4f}",
            ]
            if entry.get("epsilon") is not None:
                parts.append(f"eps={float(entry['epsilon']):g}")
            if entry.get("site_label") is not None:
                parts.append(f"site={entry.get('site_label')}")
            if entry.get("subspace_dim") is not None:
                parts.append(f"dim={entry.get('subspace_dim')}")
            if entry.get("site_total_dim") is not None:
                parts.append(f"width={entry.get('site_total_dim')}")
            if entry.get("runtime_seconds") is not None:
                parts.append(f"runtime={float(entry['runtime_seconds']):.2f}s")
            lines.append("  - " + " ".join(parts))
        lines.append("")
    return "\n".join(lines)


def _select_stage_c_configs(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_configs_per_var: int,
) -> dict[tuple[str, str, str, int], tuple[int, ...]]:
    grouped: dict[tuple[str, str, str, int], list[int]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        for entry in rankings.get(target_var, [])[: max(1, int(top_configs_per_var))]:
            key = (
                str(entry["token_position_id"]),
                str(entry["basis_source_mode"]),
                str(entry["site_menu"]),
                int(entry["num_bands"]),
            )
            grouped.setdefault(key, []).append(int(entry["layer"]))
    return {
        key: tuple(sorted(dict.fromkeys(layers)))
        for key, layers in grouped.items()
    }


def _extract_stage_c_rankings(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for payload_path in payload_paths:
        layer_payload = _load_json(payload_path)
        if not isinstance(layer_payload, dict):
            continue
        guided_summary_runtime = float(layer_payload.get("runtime_seconds", 0.0))
        layer = int(layer_payload.get("layer"))
        token_position_id = str(layer_payload.get("token_position_id"))
        basis_source_mode = str(layer_payload.get("basis_source_mode"))
        site_menu = str(layer_payload.get("site_menu"))
        num_bands = int(layer_payload.get("num_bands"))
        guided_output_paths = layer_payload.get("guided_output_paths", {})
        if not isinstance(guided_output_paths, dict):
            continue
        for target_var, guided_path_str in guided_output_paths.items():
            guided_payload = _load_json(Path(str(guided_path_str)))
            if not isinstance(guided_payload, dict):
                continue
            results = guided_payload.get("results", [])
            if not results:
                continue
            result = results[0]
            if not isinstance(result, dict):
                continue
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "token_position_id": token_position_id,
                    "layer": layer,
                    "basis_source_mode": basis_source_mode,
                    "site_menu": site_menu,
                    "num_bands": num_bands,
                    "exact_acc": float(result.get("exact_acc", -1.0)),
                    "selection_score": _selection_score(result),
                    "site_label": result.get("site_label"),
                    "subspace_dim": result.get("subspace_dim"),
                    "site_total_dim": result.get("site_total_dim"),
                    "runtime_seconds": float(guided_payload.get("runtime_seconds", guided_summary_runtime)),
                    "layer_payload_path": str(payload_path),
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _format_stage_c_summary(*, rankings: dict[str, list[dict[str, object]]]) -> str:
    lines = ["MCQA Hierarchical Stage C Guided DAS Ranking", ""]
    for target_var in DEFAULT_TARGET_VARS:
        lines.append(f"[{target_var}]")
        for entry in rankings.get(target_var, []):
            parts = [
                f"layer={int(entry['layer'])}",
                f"pos={entry['token_position_id']}",
                f"basis={entry['basis_source_mode']}",
                f"menu={entry['site_menu']}",
                f"bands={int(entry['num_bands'])}",
                f"exact={float(entry['exact_acc']):.4f}",
                f"cal={float(entry['selection_score']):.4f}",
                f"site={entry.get('site_label')}",
                f"dim={entry.get('subspace_dim')}",
                f"width={entry.get('site_total_dim')}",
            ]
            if entry.get("runtime_seconds") is not None:
                parts.append(f"runtime={float(entry['runtime_seconds']):.2f}s")
            lines.append("  - " + " ".join(parts))
        lines.append("")
    return "\n".join(lines)


def _load_status(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    try:
        payload = _load_json(manifest_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_status(
    manifest_path: Path,
    *,
    repo_root: Path,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_statuses: dict[str, dict[str, object]],
) -> None:
    payload = {
        "kind": "mcqa_delta_hierarchical_sweep",
        "repo_root": str(repo_root),
        "results_root": str(Path(args.results_root)),
        "results_timestamp": str(normalized["results_timestamp"]),
        "args": {
            "device": str(args.device),
            "dataset_path": str(args.dataset_path),
            "dataset_config": args.dataset_config,
            "dataset_size": int(args.dataset_size),
            "split_seed": int(args.split_seed),
            "train_pool_size": int(args.train_pool_size),
            "calibration_pool_size": int(args.calibration_pool_size),
            "test_pool_size": int(args.test_pool_size),
            "batch_size": int(args.batch_size),
            "target_vars": list(normalized["target_vars"]),
            "stage_a_token_position_ids": list(normalized["stage_a_token_position_ids"]),
            "stage_a_layer_indices": list(normalized["stage_a_layer_indices"]),
            "stage_b_top_layers_per_var": int(args.stage_b_top_layers_per_var),
            "stage_b_neighbor_radius": int(args.stage_b_neighbor_radius),
            "stage_b_max_layers_per_var": int(args.stage_b_max_layers_per_var),
            "stage_c_top_configs_per_var": int(args.stage_c_top_configs_per_var),
            "native_block_resolutions": list(normalized["native_block_resolutions"]),
            "pca_site_menus": list(normalized["pca_site_menus"]),
            "pca_basis_source_modes": list(normalized["pca_basis_source_modes"]),
            "pca_num_bands_values": list(normalized["pca_num_bands_values"]),
            "guided_mask_names": list(normalized["guided_mask_names"]),
            "guided_subspace_dims": None
            if normalized["guided_subspace_dims"] is None
            else list(normalized["guided_subspace_dims"]),
            "regular_das_subspace_dims": list(normalized["regular_das_subspace_dims"]),
        },
        "stage_statuses": stage_statuses,
        "updated_at": datetime.now().isoformat(),
    }
    _write_json(manifest_path, payload)


def _mark_stage(
    *,
    manifest_path: Path,
    repo_root: Path,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_statuses: dict[str, dict[str, object]],
    stage_name: str,
    payload: dict[str, object],
) -> None:
    stage_statuses[stage_name] = payload
    _write_status(
        manifest_path,
        repo_root=repo_root,
        args=args,
        normalized=normalized,
        stage_statuses=stage_statuses,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    normalized = _normalize_args(args)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and not bool(args.prompt_hf_login):
        raise ValueError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required unless --prompt-hf-login is set.")

    repo_root = Path(__file__).resolve().parent
    results_root = Path(args.results_root)
    sweep_root = results_root / f"{str(normalized['results_timestamp'])}_mcqa_hierarchical_sweep"
    manifest_path = sweep_root / "hierarchical_sweep_manifest.json"
    existing_status = _load_status(manifest_path).get("stage_statuses", {})
    stage_statuses: dict[str, dict[str, object]] = {
        str(name): dict(payload) for name, payload in existing_status.items() if isinstance(payload, dict)
    }
    _write_status(
        manifest_path,
        repo_root=repo_root,
        args=args,
        normalized=normalized,
        stage_statuses=stage_statuses,
    )

    layer_indices = normalized["stage_a_layer_indices"]
    if not layer_indices:
        layer_indices = _all_layer_indices(str(args.model_name))

    stage_a_rankings_by_token: dict[str, dict[str, list[dict[str, object]]]] = {}
    stage_a_summary_paths: list[Path] = []
    if "stage_a_plot_layer" in normalized["stages"]:
        for token_position_id in normalized["stage_a_token_position_ids"]:
            stage_name = f"stage_a_{str(token_position_id)}"
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageA_{str(token_position_id)}"
            run_root = results_root / f"{stage_timestamp}_mcqa_plot_layer"
            stage_output = run_root / f"mcqa_plot_layer_pos-{str(token_position_id)}_sig-{str(args.signature_mode)}.json"
            ranking_json_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.json"
            ranking_txt_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.txt"
            if _stage_output_is_valid(stage_output) and _stage_output_is_valid(ranking_json_path):
                rankings = _load_json(ranking_json_path)
                if not isinstance(rankings, dict):
                    raise ValueError(f"Malformed stage A ranking payload at {ranking_json_path}")
                stage_a_rankings_by_token[str(token_position_id)] = rankings
                stage_a_summary_paths.extend([ranking_json_path, ranking_txt_path])
                _mark_stage(
                    manifest_path=manifest_path,
                    repo_root=repo_root,
                    args=args,
                    normalized=normalized,
                    stage_statuses=stage_statuses,
                    stage_name=stage_name,
                    payload={
                        "state": "skipped_existing",
                        "stage_timestamp": stage_timestamp,
                        "expected_outputs": [str(stage_output), str(ranking_json_path), str(ranking_txt_path)],
                        "completed_at": datetime.now().isoformat(),
                    },
                )
                continue
            stage = SweepStage(
                name=stage_name,
                category="stage_a_plot_layer",
                description=f"Joint layer-level OT discovery at token position {token_position_id}.",
                stage_timestamp=stage_timestamp,
                command=_build_stage_a_command(
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                    token_position_id=str(token_position_id),
                    layer_indices=tuple(int(layer) for layer in layer_indices),
                ),
                expected_outputs=(str(stage_output),),
            )
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=stage.name,
                payload={
                    "state": "running",
                    "stage_timestamp": stage_timestamp,
                    "expected_outputs": [str(stage_output), str(ranking_json_path), str(ranking_txt_path)],
                    "started_at": datetime.now().isoformat(),
                },
            )
            stage_runtime_seconds = _run_stage_command(stage=stage, repo_root=repo_root)
            rankings = _extract_stage_a_rankings(aggregate_path=stage_output)
            _write_json(ranking_json_path, rankings)
            _write_text(
                ranking_txt_path,
                _format_stage_a_summary(token_position_id=str(token_position_id), rankings=rankings),
            )
            stage_a_rankings_by_token[str(token_position_id)] = rankings
            stage_a_summary_paths.extend([ranking_json_path, ranking_txt_path])
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=stage.name,
                payload={
                    "state": "completed",
                    "stage_timestamp": stage_timestamp,
                    "expected_outputs": [str(stage_output), str(ranking_json_path), str(ranking_txt_path)],
                    "runtime_seconds": float(stage_runtime_seconds),
                    "wall_runtime_seconds": float(stage_runtime_seconds),
                    "completed_at": datetime.now().isoformat(),
                },
            )
    else:
        for token_position_id in normalized["stage_a_token_position_ids"]:
            ranking_json_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.json"
            if _stage_output_is_valid(ranking_json_path):
                rankings = _load_json(ranking_json_path)
                if isinstance(rankings, dict):
                    stage_a_rankings_by_token[str(token_position_id)] = rankings
                    stage_a_summary_paths.append(ranking_json_path)

    native_payload_paths: list[Path] = []
    if "stage_b_plot_native_support" in normalized["stages"]:
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            if str(token_position_id) != "last_token":
                continue
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            if not selected_layers:
                continue
            stage_name = f"stage_b_native_{str(token_position_id)}"
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_native_{str(token_position_id)}"
            native_root = results_root / f"{stage_timestamp}_mcqa_plot_native_support"
            expected_outputs = [str(native_root / "layer_sweep_manifest.json")]
            for layer in selected_layers:
                native_widths = tuple(int(width) for width in normalized["native_block_resolutions"])
                atomic_width = native_widths[0]
                for width in native_widths[1:]:
                    atomic_width = math.gcd(int(atomic_width), int(width))
                expected_outputs.append(
                    str(
                        native_root
                        / f"layer_{int(layer):02d}"
                        / (
                            f"mcqa_plot_native_support_layer-{int(layer)}_pos-last_token"
                            f"_atomic-{int(atomic_width if atomic_width > 0 else 1)}_sig-{str(args.signature_mode)}.json"
                        )
                    )
                )
            if all(_stage_output_is_valid(Path(path)) for path in expected_outputs):
                native_payload_paths.extend(Path(path) for path in expected_outputs[1:])
                _mark_stage(
                    manifest_path=manifest_path,
                    repo_root=repo_root,
                    args=args,
                    normalized=normalized,
                    stage_statuses=stage_statuses,
                    stage_name=stage_name,
                    payload={
                        "state": "skipped_existing",
                        "stage_timestamp": stage_timestamp,
                        "expected_outputs": expected_outputs,
                        "completed_at": datetime.now().isoformat(),
                    },
                )
                continue
            stage = SweepStage(
                name=stage_name,
                category="stage_b_plot_native_support",
                description=f"Native support localization for selected {token_position_id} layers {list(selected_layers)}.",
                stage_timestamp=stage_timestamp,
                command=_build_native_block_command(
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                    layers=selected_layers,
                ),
                expected_outputs=tuple(expected_outputs),
            )
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=stage_name,
                payload={
                    "state": "running",
                    "stage_timestamp": stage_timestamp,
                    "expected_outputs": expected_outputs,
                    "started_at": datetime.now().isoformat(),
                },
            )
            stage_runtime_seconds = _run_stage_command(stage=stage, repo_root=repo_root)
            missing_outputs = [path for path in expected_outputs if not _stage_output_is_valid(Path(path))]
            if missing_outputs:
                raise RuntimeError(f"Stage {stage_name} missing outputs: {missing_outputs}")
            native_payload_paths.extend(Path(path) for path in expected_outputs[1:])
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=stage_name,
                payload={
                    "state": "completed",
                    "stage_timestamp": stage_timestamp,
                    "expected_outputs": expected_outputs,
                    "runtime_seconds": float(stage_runtime_seconds),
                    "wall_runtime_seconds": float(stage_runtime_seconds),
                    "completed_at": datetime.now().isoformat(),
                },
            )
        native_ot_rankings = _extract_native_support_rankings(payload_paths=native_payload_paths)
        _write_json(sweep_root / "stage_b_native_support_rankings.json", native_ot_rankings)
        _write_text(
            sweep_root / "stage_b_native_support_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage B Native Support Ranking", rankings=native_ot_rankings),
        )

    if "stage_c_plot_das_layer" in normalized["stages"]:
        layer_das_payload_paths: list[Path] = []
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            for layer in selected_layers:
                stage_name = f"stage_c_layer_{str(token_position_id)}_L{int(layer):02d}"
                stage_timestamp = f"{str(normalized['results_timestamp'])}_stageC_layer_{str(token_position_id)}_L{int(layer):02d}"
                run_root = results_root / f"{stage_timestamp}_mcqa_plot_das_layer"
                payload_path = run_root / f"layer_{int(layer):02d}" / f"mcqa_plot_das_layer_layer-{int(layer)}_pos-{str(token_position_id)}_summary.json"
                if not _stage_output_is_valid(payload_path):
                    stage = SweepStage(
                        name=stage_name,
                        category="stage_c_plot_das_layer",
                        description=f"Standalone layer DAS for token position {token_position_id} layer {int(layer)}.",
                        stage_timestamp=stage_timestamp,
                        command=_build_layer_das_command(
                            args=args,
                            normalized=normalized,
                            stage_timestamp=stage_timestamp,
                            token_position_id=str(token_position_id),
                            layer=int(layer),
                        ),
                        expected_outputs=(str(payload_path),),
                    )
                    _run_stage_command(stage=stage, repo_root=repo_root)
                if _stage_output_is_valid(payload_path):
                    layer_das_payload_paths.append(payload_path)
        layer_das_rankings = _extract_layer_das_rankings(payload_paths=layer_das_payload_paths)
        _write_json(sweep_root / "stage_c_layer_das_rankings.json", layer_das_rankings)
        _write_text(
            sweep_root / "stage_c_layer_das_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage C Layer DAS Ranking", rankings=layer_das_rankings),
        )

    if "stage_c_plot_das_native_support" in normalized["stages"] and native_payload_paths:
        native_das_payload_paths: list[Path] = []
        for native_payload_path in native_payload_paths:
            native_payload = _load_json(native_payload_path)
            if not isinstance(native_payload, dict):
                continue
            layer = int(native_payload.get("layer"))
            stage_name = f"stage_c_native_L{int(layer):02d}"
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageC_native_L{int(layer):02d}"
            run_root = results_root / f"{stage_timestamp}_mcqa_plot_das_native_support"
            payload_path = run_root / f"layer_{int(layer):02d}" / f"mcqa_plot_das_native_support_layer-{int(layer)}_summary.json"
            if not _stage_output_is_valid(payload_path):
                stage = SweepStage(
                    name=stage_name,
                    category="stage_c_plot_das_native_support",
                    description=f"Standalone native-support DAS for layer {int(layer)}.",
                    stage_timestamp=stage_timestamp,
                    command=_build_native_support_das_command(
                        args=args,
                        normalized=normalized,
                        stage_timestamp=stage_timestamp,
                        native_support_path=native_payload_path,
                    ),
                    expected_outputs=(str(payload_path),),
                )
                _run_stage_command(stage=stage, repo_root=repo_root)
            if _stage_output_is_valid(payload_path):
                native_das_payload_paths.append(payload_path)
        native_das_rankings = _extract_native_support_das_rankings(payload_paths=native_das_payload_paths)
        _write_json(sweep_root / "stage_c_native_support_das_rankings.json", native_das_rankings)
        _write_text(
            sweep_root / "stage_c_native_support_das_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage C Native Support DAS Ranking", rankings=native_das_rankings),
        )

    stage_b_payload_paths: list[Path] = []
    stage_b_rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    if "stage_b_plot_pca_support" in normalized["stages"]:
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            if not selected_layers:
                continue
            for basis_source_mode in normalized["pca_basis_source_modes"]:
                for site_menu in normalized["pca_site_menus"]:
                    for num_bands in _normalize_num_bands_values(normalized["pca_num_bands_values"], str(site_menu)):
                        stage_slug = _stage_b_slug(
                            token_position_id=str(token_position_id),
                            basis_source_mode=str(basis_source_mode),
                            site_menu=str(site_menu),
                            num_bands=int(num_bands),
                        )
                        stage_name = f"stage_b_{stage_slug}"
                        stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_{stage_slug}"
                        sweep_run_root = results_root / f"{stage_timestamp}_mcqa_ot_pca_focus"
                        site_catalog_tag = _site_catalog_tag(
                            site_menu=str(site_menu),
                            num_bands=int(num_bands),
                            band_scheme=str(args.pca_band_scheme),
                            top_prefix_sizes=normalized["pca_top_prefix_sizes"],
                        )
                        expected_outputs = [str(sweep_run_root / "layer_sweep_manifest.json")]
                        for layer in selected_layers:
                            expected_outputs.append(
                                str(
                                    sweep_run_root
                                    / f"layer_{int(layer):02d}"
                                    / (
                                        f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                                        f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_plot_pca_support.json"
                                    )
                                )
                            )
                        if all(_stage_output_is_valid(Path(path)) for path in expected_outputs):
                            stage_b_payload_paths.extend(Path(path) for path in expected_outputs[1:])
                            _mark_stage(
                                manifest_path=manifest_path,
                                repo_root=repo_root,
                                args=args,
                                normalized=normalized,
                                stage_statuses=stage_statuses,
                                stage_name=stage_name,
                                payload={
                                    "state": "skipped_existing",
                                    "stage_timestamp": stage_timestamp,
                                    "expected_outputs": expected_outputs,
                                    "completed_at": datetime.now().isoformat(),
                                },
                            )
                            continue
                        stage = SweepStage(
                            name=stage_name,
                            category="stage_b_plot_pca_support",
                            description=(
                                f"PCA OT refinement at token position {token_position_id}, "
                                f"basis={basis_source_mode}, menu={site_menu}, bands={num_bands}, "
                                f"layers={list(selected_layers)}."
                            ),
                            stage_timestamp=stage_timestamp,
                            command=_build_stage_b_or_c_command(
                                args=args,
                                normalized=normalized,
                                stage_timestamp=stage_timestamp,
                                token_position_id=str(token_position_id),
                                layers=selected_layers,
                                basis_source_mode=str(basis_source_mode),
                                site_menu=str(site_menu),
                                num_bands=int(num_bands),
                                guided_das=False,
                            ),
                            expected_outputs=tuple(expected_outputs),
                        )
                        _mark_stage(
                            manifest_path=manifest_path,
                            repo_root=repo_root,
                            args=args,
                            normalized=normalized,
                            stage_statuses=stage_statuses,
                            stage_name=stage_name,
                            payload={
                                "state": "running",
                                "stage_timestamp": stage_timestamp,
                                "expected_outputs": expected_outputs,
                                "started_at": datetime.now().isoformat(),
                            },
                        )
                        stage_runtime_seconds = _run_stage_command(stage=stage, repo_root=repo_root)
                        missing_outputs = [path for path in expected_outputs if not _stage_output_is_valid(Path(path))]
                        if missing_outputs:
                            raise RuntimeError(f"Stage {stage_name} missing outputs: {missing_outputs}")
                        stage_b_payload_paths.extend(Path(path) for path in expected_outputs[1:])
                        _mark_stage(
                            manifest_path=manifest_path,
                            repo_root=repo_root,
                            args=args,
                            normalized=normalized,
                            stage_statuses=stage_statuses,
                            stage_name=stage_name,
                            payload={
                                "state": "completed",
                                "stage_timestamp": stage_timestamp,
                                "expected_outputs": expected_outputs,
                                "runtime_seconds": float(stage_runtime_seconds),
                                "wall_runtime_seconds": float(stage_runtime_seconds),
                                "completed_at": datetime.now().isoformat(),
                            },
                        )
        stage_b_rankings = _extract_stage_b_best_configs(payload_paths=stage_b_payload_paths)
        stage_b_json_path = sweep_root / "stage_b_pca_support_rankings.json"
        stage_b_txt_path = sweep_root / "stage_b_pca_support_rankings.txt"
        _write_json(stage_b_json_path, stage_b_rankings)
        _write_text(stage_b_txt_path, _format_stage_b_summary(rankings=stage_b_rankings))
    else:
        stage_b_json_path = sweep_root / "stage_b_pca_support_rankings.json"
        if _stage_output_is_valid(stage_b_json_path):
            payload = _load_json(stage_b_json_path)
            if isinstance(payload, dict):
                stage_b_rankings = payload

    if "stage_c_plot_das_pca_support" in normalized["stages"]:
        selected_config_groups = _select_stage_c_configs(
            rankings=stage_b_rankings,
            top_configs_per_var=int(args.stage_c_top_configs_per_var),
        )
        stage_c_payload_paths: list[Path] = []
        for (token_position_id, basis_source_mode, site_menu, num_bands), layers in selected_config_groups.items():
            stage_slug = _stage_b_slug(
                token_position_id=str(token_position_id),
                basis_source_mode=str(basis_source_mode),
                site_menu=str(site_menu),
                num_bands=int(num_bands),
            )
            stage_name = f"stage_c_{stage_slug}"
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_{stage_slug}"
            sweep_run_root = results_root / f"{stage_timestamp}_mcqa_ot_pca_focus"
            site_catalog_tag = _site_catalog_tag(
                site_menu=str(site_menu),
                num_bands=int(num_bands),
                band_scheme=str(args.pca_band_scheme),
                top_prefix_sizes=normalized["pca_top_prefix_sizes"],
            )
            expected_outputs: list[str] = []
            for layer in layers:
                expected_outputs.append(
                    str(
                        sweep_run_root
                        / f"layer_{int(layer):02d}"
                        / (
                            f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                            f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_plot_pca_support.json"
                        )
                    )
                )
                expected_outputs.append(
                    str(
                        sweep_run_root
                        / f"layer_{int(layer):02d}"
                        / (
                            f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                            f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_plot_das_pca_support.json"
                        )
                    )
                )
                for target_var in normalized["target_vars"]:
                    expected_outputs.append(
                        str(
                            sweep_run_root
                            / f"layer_{int(layer):02d}"
                            / (
                                f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                                f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_{str(target_var)}_das_guided.json"
                            )
                        )
                    )
            if all(_stage_output_is_valid(Path(path)) for path in expected_outputs):
                stage_c_payload_paths.extend(Path(path) for path in expected_outputs if path.endswith("_plot_das_pca_support.json"))
                _mark_stage(
                    manifest_path=manifest_path,
                    repo_root=repo_root,
                    args=args,
                    normalized=normalized,
                    stage_statuses=stage_statuses,
                    stage_name=stage_name,
                    payload={
                        "state": "skipped_existing",
                        "stage_timestamp": stage_timestamp,
                        "expected_outputs": expected_outputs,
                        "completed_at": datetime.now().isoformat(),
                    },
                )
                continue
            stage = SweepStage(
                name=stage_name,
                category="stage_c_plot_das_pca_support",
                description=(
                    f"Guided DAS on selected PCA OT config token_position={token_position_id}, "
                    f"basis={basis_source_mode}, menu={site_menu}, bands={num_bands}, layers={list(layers)}."
                ),
                stage_timestamp=stage_timestamp,
                command=_build_stage_b_or_c_command(
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                    token_position_id=str(token_position_id),
                    layers=layers,
                    basis_source_mode=str(basis_source_mode),
                    site_menu=str(site_menu),
                    num_bands=int(num_bands),
                    guided_das=True,
                ),
                expected_outputs=tuple(expected_outputs),
            )
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=stage_name,
                payload={
                    "state": "running",
                    "stage_timestamp": stage_timestamp,
                    "expected_outputs": expected_outputs,
                    "started_at": datetime.now().isoformat(),
                },
            )
            stage_runtime_seconds = _run_stage_command(stage=stage, repo_root=repo_root)
            missing_outputs = [path for path in expected_outputs if not _stage_output_is_valid(Path(path))]
            if missing_outputs:
                raise RuntimeError(f"Stage {stage_name} missing outputs: {missing_outputs}")
            stage_c_payload_paths.extend(Path(path) for path in expected_outputs if path.endswith("_plot_das_pca_support.json"))
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=stage_name,
                payload={
                    "state": "completed",
                    "stage_timestamp": stage_timestamp,
                    "expected_outputs": expected_outputs,
                    "runtime_seconds": float(stage_runtime_seconds),
                    "wall_runtime_seconds": float(stage_runtime_seconds),
                    "completed_at": datetime.now().isoformat(),
                },
            )
        stage_c_rankings = _extract_stage_c_rankings(payload_paths=stage_c_payload_paths)
        stage_c_json_path = sweep_root / "stage_c_pca_support_das_rankings.json"
        stage_c_txt_path = sweep_root / "stage_c_pca_support_das_rankings.txt"
        _write_json(stage_c_json_path, stage_c_rankings)
        _write_text(stage_c_txt_path, _format_stage_c_summary(rankings=stage_c_rankings))

    write_paper_runtime_summary(sweep_root=sweep_root, full_das_outputs=list(args.full_das_output or []))

    summary_path = sweep_root / "hierarchical_sweep_summary.txt"
    lines = [
        "MCQA Delta Hierarchical Sweep",
        f"results_timestamp: {normalized['results_timestamp']}",
        f"results_root: {results_root}",
        "",
    ]
    for stage_name in sorted(stage_statuses):
        status = stage_statuses[stage_name]
        lines.append(f"{stage_name}: {status.get('state', 'unknown')}")
        if status.get("runtime_seconds") is not None:
            lines.append(f"  runtime_seconds: {float(status['runtime_seconds']):.2f}")
        for path in status.get("expected_outputs", []):
            lines.append(f"  {path}")
        lines.append("")
    _write_text(summary_path, "\n".join(lines))
    print(f"Wrote hierarchical sweep manifest to {manifest_path}")
    print(f"Wrote hierarchical sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
