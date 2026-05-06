from __future__ import annotations

import argparse
import json
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
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_STAGE_A_UOT_BETA_NEURALS = (0.1, 0.3, 1.0, 3.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 3, 4, 5)
DEFAULT_OT_LAMBDAS = (
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    27.0,
    28.0,
    29.0,
    30.0,
)
DEFAULT_PCA_SITE_MENUS = ("partition",)
DEFAULT_PCA_BASIS_SOURCE_MODES = ("pair_bank", "all_variants")
DEFAULT_PCA_NUM_BANDS_VALUES = (8, 16)
DEFAULT_PCA_BAND_SCHEME = "equal"
DEFAULT_GUIDED_MASK_NAMES = ("Selected",)
DEFAULT_GUIDED_SUPPORT_DIM_COUNT = 10
DEFAULT_DIM_HINT_SCALE_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_NATIVE_RESOLUTIONS = [16, 32, 128, 256]
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
    "stage_c_plot_das_dimension",
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


def _parse_stage_a_fixed_layers(value: str | None) -> dict[str, tuple[int, ...]]:
    if value is None or str(value).strip() == "":
        return {}
    resolved: dict[str, tuple[int, ...]] = {}
    for item in _parse_csv_strings(value):
        if ":" not in item:
            raise ValueError(
                "--stage-a-fixed-layers entries must look like "
                "'answer_pointer:17|18|19,answer_token:24|23|22'"
            )
        target_var, layer_text = item.split(":", 1)
        target_var = str(target_var).strip()
        if target_var not in DEFAULT_TARGET_VARS:
            raise ValueError(
                f"Unsupported target var in --stage-a-fixed-layers: {target_var}. "
                f"Expected one of {list(DEFAULT_TARGET_VARS)}."
            )
        if target_var in resolved:
            raise ValueError(f"Duplicate target var in --stage-a-fixed-layers: {target_var}")
        layer_values = tuple(
            int(layer.strip())
            for layer in str(layer_text).split("|")
            if str(layer).strip() != ""
        )
        if not layer_values:
            raise ValueError(
                f"--stage-a-fixed-layers must provide at least one layer for {target_var}"
            )
        deduped = tuple(dict.fromkeys(int(layer) for layer in layer_values if int(layer) >= 0))
        if not deduped:
            raise ValueError(
                f"--stage-a-fixed-layers must provide at least one non-negative layer for {target_var}"
            )
        resolved[target_var] = deduped
    missing = [target_var for target_var in DEFAULT_TARGET_VARS if target_var not in resolved]
    if missing:
        raise ValueError(
            "--stage-a-fixed-layers must specify every target var. "
            f"Missing: {missing}"
        )
    return resolved


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


def _normalize_native_resolutions(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, set)):
        resolved = [int(item) for item in value]
    else:
        resolved = [int(value)]
    filtered = [int(item) for item in resolved if int(item) > 0]
    if not filtered:
        raise ValueError("DEFAULT_NATIVE_RESOLUTIONS must contain at least one positive width")
    return tuple(dict.fromkeys(filtered))


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


def _site_catalog_tag(*, site_menu: str, num_bands: int, band_scheme: str) -> str:
    return f"menu-{str(site_menu)}-bands-{int(num_bands)}-scheme-{str(band_scheme)}"


def _target_file_suffix(target_var: str) -> str:
    return f"_target-{str(target_var)}"


def _target_vars_from_target_suffixed_path(path: Path, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    stem = path.stem
    for target_var in DEFAULT_TARGET_VARS:
        suffix = _target_file_suffix(str(target_var))
        if stem.endswith(suffix) or f"{suffix}_" in stem:
            return (str(target_var),)
    return tuple(str(target_var) for target_var in fallback)


def _stage_b_slug(*, token_position_id: str, basis_source_mode: str, site_menu: str, num_bands: int) -> str:
    return f"{str(token_position_id)}_{str(basis_source_mode)}_{str(site_menu)}_{int(num_bands)}b"


def _normalize_num_bands_values(values: tuple[int, ...], site_menu: str) -> tuple[int, ...]:
    return tuple(sorted(dict.fromkeys(int(value) for value in values)))


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
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument(
        "--stages",
        default="stage_a_plot_layer,stage_b_plot_native_support,stage_b_plot_pca_support,stage_c_plot_das_layer,stage_c_plot_das_native_support,stage_c_plot_das_dimension,stage_c_plot_das_pca_support",
    )
    parser.add_argument("--stage-a-token-position-ids", default="last_token")
    parser.add_argument("--stage-a-layer-indices", default=None, help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument(
        "--stage-a-fixed-layers",
        default=None,
        help=(
            "Optional fixed Stage A layer override like "
            "'answer_pointer:17|18|19,answer_token:24|23|22'. When set, Stage A "
            "transport is skipped and downstream stages use these ordered layers."
        ),
    )
    parser.add_argument("--target-vars", default="answer_pointer,answer_token")
    parser.add_argument("--ot-epsilons", default="0.5,1,2,4")
    parser.add_argument("--stage-a-uot-beta-neurals", default="0.1,0.3,1,3")
    parser.add_argument("--stage-a-row-top-k", type=int, default=6)
    parser.add_argument(
        "--stage-a-ot-lambdas",
        default=None,
        help="Deprecated compatibility flag. Stage A now uses a fixed full-strength intervention and ignores lambda sweeps.",
    )
    parser.add_argument("--ot-top-k-values", default="1,2,3,4,5")
    parser.add_argument(
        "--ot-lambdas",
        default="10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
    )
    parser.add_argument("--calibration-metric", default=DEFAULT_CALIBRATION_METRIC)
    parser.add_argument("--calibration-family-weights", default="1,1,1")
    parser.add_argument("--stage-b-top-layers-per-var", type=int, default=1)
    parser.add_argument("--stage-b-neighbor-radius", type=int, default=0)
    parser.add_argument("--stage-b-max-layers-per-var", type=int, default=1)
    parser.add_argument(
        "--native-resolutions",
        default=None,
        help="Comma-separated native OT widths. Default: values from DEFAULT_NATIVE_RESOLUTIONS.",
    )
    parser.add_argument("--pca-site-menus", default="partition")
    parser.add_argument("--pca-basis-source-modes", default="pair_bank,all_variants")
    parser.add_argument("--pca-num-bands-values", default="8,16")
    parser.add_argument("--pca-band-scheme", default=DEFAULT_PCA_BAND_SCHEME, choices=("equal", "head"))
    parser.add_argument(
        "--pca-support-extraction-mode",
        default="selected_only",
        choices=("ranked", "selected_only"),
        help=(
            "Deprecated compatibility flag. PCA support now always keeps only the best calibrated PLOT handle."
        ),
    )
    parser.add_argument(
        "--pca-cache-signatures",
        action="store_true",
        help="Persist PLOT-PCA signature caches. Disabled by default to reduce disk use.",
    )
    parser.add_argument(
        "--pca-write-epsilon-artifacts",
        action="store_true",
        help="Write per-epsilon PLOT-PCA OT artifacts. Disabled by default to reduce disk use.",
    )
    parser.add_argument(
        "--pca-write-support-artifact",
        action="store_true",
        help="Write a separate PLOT-PCA support JSON. Disabled by default because support is embedded.",
    )
    parser.add_argument("--stage-c-top-configs-per-var", type=int, default=1)
    parser.add_argument(
        "--guided-mask-names",
        default="Selected",
        help="Deprecated for native-support DAS; native support always uses the exact Selected PLOT handle.",
    )
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--screen-restarts", type=int, default=1)
    parser.add_argument("--guided-restarts", type=int, default=2)
    parser.add_argument("--guided-support-dim-count", type=int, default=DEFAULT_GUIDED_SUPPORT_DIM_COUNT)
    parser.add_argument(
        "--guided-subspace-dims",
        default=None,
        help="Deprecated for native-support DAS. Native support defaults to evenly spaced dims up to the selected support width.",
    )
    parser.add_argument(
        "--stage-c-dim-hint-scale-factors",
        default="0.5,0.75,1,1.25,1.5",
        help="Comma-separated multiplicative full-layer DAS dimension factors around the selected native support width.",
    )
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
    stage_a_fixed_layers = _parse_stage_a_fixed_layers(args.stage_a_fixed_layers)
    pca_site_menus = _parse_csv_strings(args.pca_site_menus) or DEFAULT_PCA_SITE_MENUS
    unsupported_pca_site_menus = sorted(set(str(site_menu) for site_menu in pca_site_menus) - {"partition"})
    if unsupported_pca_site_menus:
        raise ValueError(f"Unsupported PCA site menus: {unsupported_pca_site_menus}. PCA support is partition-only.")
    pca_basis_source_modes = _parse_csv_strings(args.pca_basis_source_modes) or DEFAULT_PCA_BASIS_SOURCE_MODES
    pca_num_bands_values = _parse_csv_ints(args.pca_num_bands_values) or DEFAULT_PCA_NUM_BANDS_VALUES
    native_resolutions = _normalize_native_resolutions(
        _parse_csv_ints(args.native_resolutions) or DEFAULT_NATIVE_RESOLUTIONS
    )
    ot_epsilons = _parse_csv_floats(args.ot_epsilons) or DEFAULT_OT_EPSILONS
    stage_a_uot_beta_neurals = _parse_csv_floats(args.stage_a_uot_beta_neurals) or DEFAULT_STAGE_A_UOT_BETA_NEURALS
    ot_top_k_values = _parse_csv_ints(args.ot_top_k_values) or DEFAULT_OT_TOP_K_VALUES
    ot_lambdas = _parse_csv_floats(args.ot_lambdas) or DEFAULT_OT_LAMBDAS
    calibration_family_weights = _parse_csv_floats(args.calibration_family_weights) or DEFAULT_CALIBRATION_FAMILY_WEIGHTS
    guided_mask_names = DEFAULT_GUIDED_MASK_NAMES
    guided_subspace_dims = None
    if args.guided_subspace_dims is not None:
        guided_subspace_dims = _parse_csv_ints(args.guided_subspace_dims)
    dim_hint_scale_factors = _parse_csv_floats(args.stage_c_dim_hint_scale_factors) or DEFAULT_DIM_HINT_SCALE_FACTORS
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
        "stage_a_fixed_layers": {
            str(target_var): tuple(int(layer) for layer in layers)
            for target_var, layers in stage_a_fixed_layers.items()
        },
        "pca_site_menus": tuple(str(site_menu) for site_menu in pca_site_menus),
        "pca_basis_source_modes": tuple(str(mode) for mode in pca_basis_source_modes),
        "pca_num_bands_values": tuple(int(value) for value in pca_num_bands_values),
        "pca_support_extraction_mode": "selected_only",
        "pca_cache_signatures": bool(args.pca_cache_signatures),
        "pca_write_epsilon_artifacts": bool(args.pca_write_epsilon_artifacts),
        "pca_write_support_artifact": bool(args.pca_write_support_artifact),
        "native_resolutions": tuple(int(resolution) for resolution in native_resolutions),
        "ot_epsilons": tuple(float(epsilon) for epsilon in ot_epsilons),
        "stage_a_uot_beta_neurals": tuple(float(beta) for beta in stage_a_uot_beta_neurals),
        "stage_a_row_top_k": max(1, int(args.stage_a_row_top_k)),
        "ot_top_k_values": tuple(int(value) for value in ot_top_k_values),
        "ot_lambdas": tuple(float(value) for value in ot_lambdas),
        "calibration_family_weights": tuple(float(weight) for weight in calibration_family_weights),
        "guided_mask_names": tuple(str(mask_name) for mask_name in guided_mask_names),
        "guided_support_dim_count": max(1, int(args.guided_support_dim_count)),
        "guided_subspace_dims": None if guided_subspace_dims is None else tuple(int(dim) for dim in guided_subspace_dims),
        "dim_hint_scale_factors": tuple(float(value) for value in dim_hint_scale_factors),
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
        "--uot-beta-neurals",
        ",".join(
            str(beta).rstrip("0").rstrip(".") if "." in str(beta) else str(beta)
            for beta in normalized["stage_a_uot_beta_neurals"]
        ),
        "--stage-a-row-top-k",
        str(int(normalized["stage_a_row_top_k"])),
        "--calibration-family-weights",
        ",".join(
            str(weight).rstrip("0").rstrip(".") if "." in str(weight) else str(weight)
            for weight in normalized["calibration_family_weights"]
        ),
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
    requested_target_vars: tuple[str, ...],
    layer_target_vars: dict[int, tuple[str, ...]] | None,
    basis_source_mode: str,
    site_menu: str,
    num_bands: int,
    num_bands_values: tuple[int, ...] | None = None,
    guided_das: bool = False,
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
        "--target-vars",
        ",".join(str(target_var) for target_var in requested_target_vars),
        "--site-menu",
        str(site_menu),
        "--band-scheme",
        str(args.pca_band_scheme),
        "--basis-source-mode",
        str(basis_source_mode),
        "--support-extraction-mode",
        str(normalized["pca_support_extraction_mode"]),
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--ot-top-k-values",
        ",".join(str(value) for value in normalized["ot_top_k_values"]),
        "--ot-lambdas",
        ",".join(str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value) for value in normalized["ot_lambdas"]),
        "--calibration-family-weights",
        ",".join(
            str(weight).rstrip("0").rstrip(".") if "." in str(weight) else str(weight)
            for weight in normalized["calibration_family_weights"]
        ),
        "--signature-mode",
        str(args.signature_mode),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    if num_bands_values:
        command.extend(
            [
                "--num-bands-values",
                ",".join(str(int(value)) for value in num_bands_values),
            ]
        )
    else:
        command.extend(["--num-bands", str(int(num_bands))])
    _append_optional_arg(
        command,
        "--layer-target-vars",
        _encode_layer_target_vars_arg(target_vars_by_layer=layer_target_vars or {}),
    )
    if bool(normalized["pca_cache_signatures"]):
        command.append("--cache-signatures")
    if bool(normalized["pca_write_epsilon_artifacts"]):
        command.append("--write-epsilon-artifacts")
    if bool(normalized["pca_write_support_artifact"]):
        command.append("--write-support-artifact")
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
    layer_target_vars: dict[int, tuple[str, ...]] | None,
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
        "--native-resolutions",
        ",".join(str(resolution) for resolution in normalized["native_resolutions"]),
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--ot-top-k-values",
        ",".join(str(value) for value in normalized["ot_top_k_values"]),
        "--ot-lambdas",
        ",".join(str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value) for value in normalized["ot_lambdas"]),
        "--calibration-family-weights",
        ",".join(
            str(weight).rstrip("0").rstrip(".") if "." in str(weight) else str(weight)
            for weight in normalized["calibration_family_weights"]
        ),
        "--signature-mode",
        str(args.signature_mode),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
    ]
    _append_optional_arg(
        command,
        "--layer-target-vars",
        _encode_layer_target_vars_arg(target_vars_by_layer=layer_target_vars or {}),
    )
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
    target_vars: tuple[str, ...],
    das_subspace_dims: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    resolved_das_subspace_dims = (
        tuple(int(dim) for dim in normalized["regular_das_subspace_dims"])
        if das_subspace_dims is None
        else tuple(int(dim) for dim in das_subspace_dims)
    )
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
        "--target-vars",
        ",".join(str(target_var) for target_var in target_vars),
        "--signature-mode",
        str(args.signature_mode),
        "--das-subspace-dims",
        ",".join(str(dim) for dim in resolved_das_subspace_dims),
        "--das-max-epochs",
        str(int(args.guided_max_epochs)),
        "--das-min-epochs",
        str(int(args.guided_min_epochs)),
        "--das-restarts",
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


def _build_native_support_das_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    native_support_path: Path,
    target_vars: tuple[str, ...],
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
        "--target-vars",
        ",".join(str(target_var) for target_var in target_vars),
        "--guided-mask-names",
        ",".join(str(mask_name) for mask_name in normalized["guided_mask_names"]),
        "--guided-support-dim-count",
        str(int(normalized["guided_support_dim_count"])),
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
        resolved = {
            str(target_var): [dict(entry) for entry in entries if isinstance(entry, dict)]
            for target_var, entries in rankings.items()
            if isinstance(entries, list)
        }
        display_method_by_var = aggregate_payload.get("display_method_by_var", {})
        if isinstance(display_method_by_var, dict):
            resolved["_display_method_by_var"] = {
                str(target_var): dict(entry)
                for target_var, entry in display_method_by_var.items()
                if isinstance(entry, dict)
            }
        return resolved
    if str(aggregate_path.name) == "layer_sweep_manifest.json":
        return _stage_a_rankings_from_layer_sweep(manifest_path=aggregate_path, manifest_payload=aggregate_payload)
    return _stage_a_rankings_from_joint_run(aggregate_path=aggregate_path, aggregate_payload=aggregate_payload)


def _format_stage_a_summary(*, token_position_id: str, rankings: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        "MCQA Hierarchical Stage A Layer Discovery",
        f"token_position_id: {token_position_id}",
        "",
    ]
    display_method_by_var = rankings.get("_display_method_by_var", {}) if isinstance(rankings, dict) else {}
    for target_var in DEFAULT_TARGET_VARS:
        lines.append(f"[{target_var}]")
        display_entry = display_method_by_var.get(target_var) if isinstance(display_method_by_var, dict) else None
        if isinstance(display_entry, dict):
            exact_acc = display_entry.get("exact_acc")
            selection_score = display_entry.get("selection_score")
            parts = [
                f"method={display_entry.get('method')}",
                f"layer={int(display_entry.get('layer', -1))}",
                f"exact={float(exact_acc):.4f}" if exact_acc is not None else "exact=NA",
                f"cal={float(selection_score):.4f}" if selection_score is not None else "cal=NA",
                (
                    f"eps={float(display_entry.get('epsilon', 0.0)):g}"
                    if display_entry.get("epsilon") is not None else "eps=NA"
                ),
                f"site={display_entry.get('site_label')}",
                f"candidates={display_entry.get('candidate_site_labels', [])}",
            ]
            if display_entry.get("uot_beta_neural") is not None:
                parts.append(f"beta_n={float(display_entry.get('uot_beta_neural')):g}")
            runtime_value = display_entry.get("runtime_with_signatures_seconds", display_entry.get("runtime_seconds"))
            if runtime_value is not None:
                parts.append(f"runtime={float(runtime_value):.2f}s")
            lines.append("  method " + " ".join(parts))
        for entry in rankings.get(target_var, []):
            parts = [
                f"layer={int(entry['layer'])}",
                f"row_mass={float(entry.get('target_row_transport_mass', entry.get('support_evidence', entry.get('selection_score', 0.0)))):.4f}",
            ]
            if "epsilon" in entry:
                parts.append(f"eps={float(entry.get('epsilon', -1.0)):g}")
            if entry.get("site_label") is not None:
                parts.append(f"site={entry.get('site_label')}")
            if entry.get("runtime_seconds") is not None:
                parts.append(f"runtime={float(entry['runtime_seconds']):.2f}s")
            if "mean_target_site_mass" in entry:
                parts.append(f"mean_mass={float(entry.get('mean_target_site_mass', 0.0)):.4f}")
            if "selected_site_labels" in entry:
                parts.append(f"selected_handle={entry.get('selected_site_labels')}")
            lines.append("  - " + " ".join(parts))
        lines.append("")
    return "\n".join(lines)


def _fixed_stage_a_rankings(
    *,
    token_position_id: str,
    fixed_layers_by_var: dict[str, tuple[int, ...]],
) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {}
    display_method_by_var: dict[str, dict[str, object]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        layers = tuple(int(layer) for layer in fixed_layers_by_var[str(target_var)])
        rankings[str(target_var)] = []
        for index, layer in enumerate(layers):
            score = float(len(layers) - index)
            site_label = f"L{int(layer)}:{str(token_position_id)}"
            rankings[str(target_var)].append(
                {
                    "layer": int(layer),
                    "selection_score": score,
                    "target_row_transport_mass": score,
                    "site_label": site_label,
                    "runtime_seconds": 0.0,
                    "selection_basis": "manual_fixed_stage_a_layer_override",
                }
            )
        lead_layer = int(layers[0])
        display_method_by_var[str(target_var)] = {
            "method": "fixed_layer_override",
            "layer": int(lead_layer),
            "exact_acc": None,
            "selection_score": None,
            "epsilon": None,
            "site_label": f"L{int(lead_layer)}:{str(token_position_id)}",
            "candidate_site_labels": [f"L{int(layer)}:{str(token_position_id)}" for layer in layers],
            "selection_basis": "manual_fixed_stage_a_layer_override",
        }
    rankings["_display_method_by_var"] = display_method_by_var
    return rankings


def _print_stage_a_fixed_layer_override(*, token_position_id: str, fixed_layers_by_var: dict[str, tuple[int, ...]]) -> None:
    print("")
    print(f"[stageA] using fixed layer override for token_position={token_position_id}")
    for target_var in DEFAULT_TARGET_VARS:
        print("")
        print(f"  [{target_var}]")
        print(f"  fixed_layers={list(int(layer) for layer in fixed_layers_by_var[str(target_var)])}")
    print("")


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


def _select_stage_b_layers_by_var(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_layers_per_var: int,
    neighbor_radius: int = 1,
    max_layers_per_var: int = 5,
) -> dict[str, tuple[int, ...]]:
    selected_by_var: dict[str, tuple[int, ...]] = {}
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
        selected_by_var[str(target_var)] = tuple(int(layer) for layer in row_selected)
    return selected_by_var


def _print_stage_b_layer_selection(
    *,
    token_position_id: str,
    selected_by_var: dict[str, tuple[int, ...]],
    selected_layers: tuple[int, ...],
) -> None:
    print("")
    print(f"[stageB] selected Stage A layers for token_position={token_position_id}")
    for target_var in DEFAULT_TARGET_VARS:
        layers = list(int(layer) for layer in selected_by_var.get(str(target_var), ()))
        print("")
        print(f"  [{target_var}]")
        print(f"  selected_layers={layers}")
    print("")
    print(f"  union_layers={list(int(layer) for layer in selected_layers)}")
    print("")


def _selected_target_vars_by_layer(
    *,
    selected_by_var: dict[str, tuple[int, ...]],
) -> dict[int, tuple[str, ...]]:
    resolved: dict[int, list[str]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        for layer in selected_by_var.get(str(target_var), ()):
            resolved.setdefault(int(layer), []).append(str(target_var))
    return {
        int(layer): tuple(dict.fromkeys(str(target_var) for target_var in target_vars))
        for layer, target_vars in resolved.items()
    }


def _encode_layer_target_vars_arg(*, target_vars_by_layer: dict[int, tuple[str, ...]]) -> str | None:
    if not target_vars_by_layer:
        return None
    parts: list[str] = []
    for layer in sorted(int(layer) for layer in target_vars_by_layer):
        target_vars = tuple(str(target_var) for target_var in target_vars_by_layer.get(int(layer), ()))
        if not target_vars:
            continue
        parts.append(f"{int(layer)}:{'|'.join(target_vars)}")
    if not parts:
        return None
    return ";".join(parts)


def _stage_b_payload_matches_target_vars(
    *,
    payload_path: Path,
    expected_target_vars: tuple[str, ...],
    expected_transport_target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    expected_support_extraction_mode: str | None = None,
) -> bool:
    if not _stage_output_is_valid(payload_path):
        return False
    payload = _load_json(payload_path)
    if not isinstance(payload, dict):
        return False
    actual_target_vars = payload.get("target_vars")
    if not isinstance(actual_target_vars, list):
        return False
    if tuple(str(target_var) for target_var in actual_target_vars) != tuple(
        str(target_var) for target_var in expected_target_vars
    ):
        return False
    actual_transport_target_vars = payload.get("transport_target_vars")
    if not isinstance(actual_transport_target_vars, list):
        return False
    if tuple(str(target_var) for target_var in actual_transport_target_vars) != tuple(
        str(target_var) for target_var in expected_transport_target_vars
    ):
        return False
    support_by_var = payload.get("support_by_var")
    kind = str(payload.get("kind"))
    if (
        kind == "mcqa_plot_pca_support_layer"
        and expected_support_extraction_mode is not None
        and str(payload.get("support_extraction_mode", "ranked")) != str(expected_support_extraction_mode)
    ):
        return False
    if kind in {"mcqa_plot_native_support_layer", "mcqa_plot_pca_support_layer"} and not isinstance(support_by_var, dict):
        return False
    if isinstance(support_by_var, dict):
        for target_var in expected_target_vars:
            support_summary = support_by_var.get(str(target_var))
            if not isinstance(support_summary, dict):
                return False
            mask_candidates = support_summary.get("mask_candidates", [])
            if not isinstance(mask_candidates, list):
                return False
            selected_candidate = next(
                (
                    candidate
                    for candidate in mask_candidates
                    if isinstance(candidate, dict) and str(candidate.get("name")) == "Selected"
                ),
                None,
            )
            if not isinstance(selected_candidate, dict) or not selected_candidate.get("site_indices"):
                return False
    return True


def _das_payload_matches_target_vars(*, payload_path: Path, expected_target_vars: tuple[str, ...]) -> bool:
    if not _stage_output_is_valid(payload_path):
        return False
    payload = _load_json(payload_path)
    if not isinstance(payload, dict):
        return False
    expected = tuple(str(target_var) for target_var in expected_target_vars)
    kind = str(payload.get("kind"))
    if kind == "mcqa_plot_das_native_support":
        if str(payload.get("support_mode")) != "selected_plot_handle":
            return False
        if payload.get("guided_support_dim_count") is None:
            return False
        payloads_by_var = payload.get("payloads_by_var", {})
        if not isinstance(payloads_by_var, dict):
            return False
        return tuple(str(target_var) for target_var in payloads_by_var.keys()) == expected
    actual = payload.get("target_vars")
    if isinstance(actual, list):
        return tuple(str(target_var) for target_var in actual) == expected
    if kind == "mcqa_plot_das_pca_support_layer":
        if isinstance(actual, list):
            return tuple(str(target_var) for target_var in actual) == expected
        guided_output_paths = payload.get("guided_output_paths", {})
        if not isinstance(guided_output_paths, dict):
            return False
        return tuple(str(target_var) for target_var in guided_output_paths.keys()) == expected
    if kind == "mcqa_plot_das_layer":
        method_payloads = payload.get("method_payloads", {}).get("das", [])
        actual_target_vars: list[str] = []
        for method_payload in method_payloads:
            if not isinstance(method_payload, dict):
                continue
            target_var = method_payload.get("target_var")
            if target_var is None:
                results = method_payload.get("results", [])
                if results and isinstance(results[0], dict):
                    target_var = results[0].get("variable")
            if target_var is not None:
                actual_target_vars.append(str(target_var))
        return tuple(dict.fromkeys(actual_target_vars)) == expected
    return False


def _expected_stage_b_native_payload_paths(
    *,
    native_root: Path,
    selected_layers: tuple[int, ...],
    selected_target_vars_by_layer: dict[int, tuple[str, ...]],
    native_resolutions: tuple[int, ...],
    signature_mode: str,
) -> list[Path]:
    payload_paths: list[Path] = []
    for layer in selected_layers:
        target_vars = selected_target_vars_by_layer.get(
            int(layer),
            tuple(str(target_var) for target_var in DEFAULT_TARGET_VARS),
        )
        for native_resolution in native_resolutions:
            for target_var in target_vars:
                payload_paths.append(
                    native_root
                    / f"layer_{int(layer):02d}"
                    / (
                        f"mcqa_plot_native_support_layer-{int(layer)}_pos-last_token"
                        f"_atomic-{int(native_resolution)}_sig-{str(signature_mode)}"
                        f"{_target_file_suffix(str(target_var))}.json"
                    )
                )
    return payload_paths


def _extract_stage_b_best_configs(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int, str, str, int], dict[str, object]] = {}
    def _stage_b_config_score(entry: dict[str, object]) -> tuple[float, float]:
        return (
            float(entry.get("selection_score", entry.get("cal", -1.0))),
            float(entry.get("exact_acc", -1.0)),
        )

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
        method_by_var = layer_payload.get("method_by_var", {})
        if isinstance(method_by_var, dict):
            for target_var, method_summary in method_by_var.items():
                if not isinstance(method_summary, dict):
                    continue
                selected_hyperparameters = method_summary.get("selected_hyperparameters", {})
                if not isinstance(selected_hyperparameters, dict):
                    selected_hyperparameters = {}
                target_var = str(target_var)
                key = (target_var, token_position_id, layer, basis_source_mode, site_menu, num_bands)
                entry = {
                    "variable": target_var,
                    "token_position_id": token_position_id,
                    "layer": layer,
                    "basis_source_mode": basis_source_mode,
                    "site_menu": site_menu,
                    "num_bands": num_bands,
                    "exact_acc": float(method_summary.get("exact_acc", -1.0)),
                    "selection_score": float(method_summary.get("selection_score", -1.0)),
                    "epsilon": float(method_summary.get("epsilon", -1.0)),
                    "site_label": method_summary.get("site_label"),
                    "selected_top_k": selected_hyperparameters.get("top_k"),
                    "selected_lambda": selected_hyperparameters.get("lambda"),
                    "runtime_seconds": float(localization_runtime_seconds),
                    "wall_runtime_seconds": float(localization_runtime_seconds),
                    "signature_prepare_runtime_seconds": layer_payload.get("signature_prepare_runtime_seconds"),
                    "layer_payload_path": str(payload_path),
                }
                current = grouped.get(key)
                if current is None or _stage_b_config_score(entry) > _stage_b_config_score(current):
                    grouped[key] = entry
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
                selected_hyperparameters = method_payload.get("selected_hyperparameters", {})
                if not isinstance(selected_hyperparameters, dict):
                    selected_hyperparameters = {}
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
                    "selected_top_k": selected_hyperparameters.get("top_k"),
                    "selected_lambda": selected_hyperparameters.get("lambda"),
                    "runtime_seconds": float(localization_runtime_seconds),
                    "wall_runtime_seconds": float(localization_runtime_seconds),
                    "signature_prepare_runtime_seconds": method_payload.get("signature_prepare_runtime_seconds"),
                    "layer_payload_path": str(payload_path),
                }
                current = grouped.get(key)
                if current is None or _stage_b_config_score(entry) > _stage_b_config_score(current):
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
            if entry.get("selected_top_k") is not None:
                parts.append(f"k={int(entry['selected_top_k'])}")
            if entry.get("selected_lambda") is not None:
                parts.append(f"lambda={float(entry['selected_lambda']):g}")
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
        native_resolution = int(payload.get("native_resolution", payload.get("atomic_width")))
        runtime_seconds = float(payload.get("localization_runtime_seconds", payload.get("runtime_seconds", 0.0)))
        method_by_var = payload.get("method_by_var", {})
        support_by_var = payload.get("support_by_var", {})
        for target_var, method_summary in method_by_var.items():
            if not isinstance(method_summary, dict):
                continue
            support_summary = support_by_var.get(str(target_var), {}) if isinstance(support_by_var, dict) else {}
            top_site_label = None
            selected_site_total_dim = None
            selected_site_labels = None
            selected_top_k = None
            selected_lambda = None
            if isinstance(support_summary, dict):
                ranked_site_labels = support_summary.get("ranked_site_labels", [])
                if isinstance(ranked_site_labels, list) and ranked_site_labels:
                    top_site_label = ranked_site_labels[0]
                mask_candidates = support_summary.get("mask_candidates", [])
                if isinstance(mask_candidates, list):
                    selected_mask = next(
                        (
                            candidate
                            for candidate in mask_candidates
                            if isinstance(candidate, dict) and str(candidate.get("name")) == "Selected"
                        ),
                        None,
                    )
                    if isinstance(selected_mask, dict):
                        if selected_mask.get("site_total_dim") is not None:
                            selected_site_total_dim = int(selected_mask.get("site_total_dim", 0))
                        if isinstance(selected_mask.get("site_labels"), list):
                            selected_site_labels = [str(label) for label in selected_mask.get("site_labels", [])]
                selected_trial = support_summary.get("selected_trial", {})
                if isinstance(selected_trial, dict):
                    if selected_trial.get("top_k") is not None:
                        selected_top_k = int(selected_trial.get("top_k"))
                    if selected_trial.get("lambda") is not None:
                        selected_lambda = float(selected_trial.get("lambda"))
            selected_hyperparameters = method_summary.get("selected_hyperparameters", {})
            if isinstance(selected_hyperparameters, dict):
                if selected_top_k is None and selected_hyperparameters.get("top_k") is not None:
                    selected_top_k = int(selected_hyperparameters.get("top_k"))
                if selected_lambda is None and selected_hyperparameters.get("lambda") is not None:
                    selected_lambda = float(selected_hyperparameters.get("lambda"))
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "native_resolution": int(native_resolution),
                    "exact_acc": float(method_summary.get("exact_acc", 0.0)),
                    "selection_score": float(method_summary.get("selection_score", 0.0)),
                    "epsilon": float(method_summary.get("epsilon", 0.0)),
                    "site_label": top_site_label or method_summary.get("site_label"),
                    "selected_top_k": selected_top_k,
                    "selected_lambda": selected_lambda,
                    "selected_site_total_dim": selected_site_total_dim,
                    "selected_site_labels": selected_site_labels,
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
        native_resolution = int(payload.get("native_resolution", payload.get("atomic_width")))
        native_support_payload = None
        native_support_path = payload.get("native_support_path")
        if native_support_path is not None and Path(str(native_support_path)).exists():
            maybe_native_support_payload = _load_json(Path(str(native_support_path)))
            if isinstance(maybe_native_support_payload, dict):
                native_support_payload = maybe_native_support_payload
        for target_var, das_payload in payload.get("payloads_by_var", {}).items():
            if not isinstance(das_payload, dict):
                continue
            results = das_payload.get("results", [])
            if not results or not isinstance(results[0], dict):
                continue
            result = results[0]
            selected_metadata: dict[str, object] = {}
            if isinstance(native_support_payload, dict):
                support_summary = native_support_payload.get("support_by_var", {}).get(str(target_var), {})
                if isinstance(support_summary, dict):
                    selected_trial = support_summary.get("selected_trial", {})
                    if isinstance(selected_trial, dict):
                        if selected_trial.get("top_k") is not None:
                            selected_metadata["selected_top_k"] = int(selected_trial.get("top_k"))
                        if selected_trial.get("lambda") is not None:
                            selected_metadata["selected_lambda"] = float(selected_trial.get("lambda"))
                    for candidate in support_summary.get("mask_candidates", []):
                        if not isinstance(candidate, dict) or str(candidate.get("name")) != "Selected":
                            continue
                        if candidate.get("site_total_dim") is not None:
                            selected_metadata["selected_site_total_dim"] = int(candidate.get("site_total_dim", 0))
                        if isinstance(candidate.get("site_labels"), list):
                            selected_metadata["selected_site_labels"] = [str(label) for label in candidate.get("site_labels", [])]
                        break
            rankings.setdefault(str(target_var), []).append(
                {
                    "variable": str(target_var),
                    "layer": int(layer),
                    "native_resolution": int(native_resolution),
                    "exact_acc": float(result.get("exact_acc", 0.0)),
                    "selection_score": _selection_score(result),
                    "site_label": result.get("site_label"),
                    "subspace_dim": result.get("subspace_dim"),
                    "site_total_dim": result.get("site_total_dim"),
                    "runtime_seconds": float(das_payload.get("runtime_seconds", payload.get("runtime_seconds", 0.0))),
                    **selected_metadata,
                }
            )
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _extract_dimension_das_rankings(
    *,
    payload_records: Iterable[tuple[Path, dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    for payload_path, metadata in payload_records:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict) or str(payload.get("kind")) != "mcqa_plot_das_layer":
            continue
        target_var = str(metadata.get("variable"))
        layer = int(metadata.get("layer", payload.get("layer", -1)))
        native_resolution = int(metadata.get("native_resolution", metadata.get("atomic_width", -1)))
        method_payloads = payload.get("method_payloads", {}).get("das", [])
        for method_payload in method_payloads:
            if not isinstance(method_payload, dict):
                continue
            results = method_payload.get("results", [])
            if not results or not isinstance(results[0], dict):
                continue
            result = results[0]
            result_target_var = str(method_payload.get("target_var") or result.get("variable"))
            if result_target_var != target_var:
                continue
            rankings.setdefault(target_var, []).append(
                {
                    "variable": target_var,
                    "layer": int(layer),
                    "native_resolution": int(native_resolution),
                    "exact_acc": float(result.get("exact_acc", 0.0)),
                    "selection_score": _selection_score(result),
                    "site_label": result.get("site_label"),
                    "subspace_dim": result.get("subspace_dim"),
                    "site_total_dim": result.get("site_total_dim"),
                    "runtime_seconds": float(payload.get("runtime_seconds", 0.0)),
                    "payload_path": str(payload_path),
                    "native_support_payload_path": metadata.get("payload_path"),
                    "dim_hint_effective_dim": metadata.get("dim_hint_effective_dim"),
                    "dim_hint_subspace_dims": metadata.get("dim_hint_subspace_dims"),
                    "selected_top_k": metadata.get("selected_top_k"),
                    "selected_lambda": metadata.get("selected_lambda"),
                    "selected_site_total_dim": metadata.get("selected_site_total_dim"),
                    "selected_site_labels": metadata.get("selected_site_labels"),
                    "epsilon": metadata.get("epsilon"),
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
            width_value = entry.get("native_resolution", entry.get("resolution", entry.get("atomic_width")))
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
            if entry.get("selected_top_k") is not None:
                parts.append(f"k={entry.get('selected_top_k')}")
            if entry.get("selected_lambda") is not None:
                parts.append(f"lambda={float(entry.get('selected_lambda')):g}")
            if entry.get("selected_site_total_dim") is not None:
                parts.append(f"selected_width={entry.get('selected_site_total_dim')}")
            if entry.get("subspace_dim") is not None:
                parts.append(f"dim={entry.get('subspace_dim')}")
            if entry.get("site_total_dim") is not None:
                parts.append(f"width={entry.get('site_total_dim')}")
            if entry.get("dim_hint_effective_dim") is not None:
                parts.append(f"dim_hint_width={entry.get('dim_hint_effective_dim')}")
            if entry.get("dim_hint_subspace_dims") is not None:
                parts.append(f"dim_grid={entry.get('dim_hint_subspace_dims')}")
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


def _select_stage_c_config_targets(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_configs_per_var: int,
) -> dict[tuple[str, str, str, int], dict[int, tuple[str, ...]]]:
    grouped: dict[tuple[str, str, str, int], dict[int, list[str]]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        for entry in rankings.get(target_var, [])[: max(1, int(top_configs_per_var))]:
            key = (
                str(entry["token_position_id"]),
                str(entry["basis_source_mode"]),
                str(entry["site_menu"]),
                int(entry["num_bands"]),
            )
            layer_group = grouped.setdefault(key, {})
            layer_group.setdefault(int(entry["layer"]), []).append(str(target_var))
    return {
        key: {
            int(layer): tuple(dict.fromkeys(str(target_var) for target_var in target_vars))
            for layer, target_vars in layers.items()
        }
        for key, layers in grouped.items()
    }


def _select_stage_c_config_entries(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_configs_per_var: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for target_var in DEFAULT_TARGET_VARS:
        for entry in rankings.get(target_var, [])[: max(1, int(top_configs_per_var))]:
            record = dict(entry)
            record["variable"] = str(target_var)
            selected.append(record)
    return selected


def _select_stage_c_native_support_groups(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_configs_per_var: int,
) -> dict[Path, tuple[str, ...]]:
    grouped: dict[Path, list[str]] = {}
    for target_var in DEFAULT_TARGET_VARS:
        for entry in rankings.get(target_var, [])[: max(1, int(top_configs_per_var))]:
            payload_path = entry.get("payload_path")
            if payload_path is None:
                continue
            grouped.setdefault(Path(str(payload_path)), []).append(str(target_var))
    return {
        payload_path: tuple(dict.fromkeys(target_vars))
        for payload_path, target_vars in grouped.items()
    }


def _select_stage_c_native_support_entries(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_configs_per_var: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for target_var in DEFAULT_TARGET_VARS:
        for entry in rankings.get(target_var, [])[: max(1, int(top_configs_per_var))]:
            record = dict(entry)
            record["variable"] = str(target_var)
            selected.append(record)
    return selected


def _dim_hint_subspace_dims(
    effective_dim: int,
    *,
    min_dim: int = 32,
    max_dim: int = 2304,
    scale_factors: Iterable[float] = DEFAULT_DIM_HINT_SCALE_FACTORS,
) -> tuple[int, ...]:
    dims: list[int] = []
    for factor in scale_factors:
        dim = int(round(float(effective_dim) * float(factor)))
        dim = max(int(min_dim), min(int(max_dim), dim))
        dims.append(dim)
    dims.append(max(int(min_dim), min(int(max_dim), int(effective_dim))))
    return tuple(sorted(dict.fromkeys(dims)))


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
            "stage_a_uot_beta_neurals": list(normalized["stage_a_uot_beta_neurals"]),
            "stage_a_row_top_k": int(normalized["stage_a_row_top_k"]),
            "stage_b_top_layers_per_var": int(args.stage_b_top_layers_per_var),
            "stage_b_neighbor_radius": int(args.stage_b_neighbor_radius),
            "stage_b_max_layers_per_var": int(args.stage_b_max_layers_per_var),
            "stage_c_top_configs_per_var": int(args.stage_c_top_configs_per_var),
            "native_resolutions": list(normalized["native_resolutions"]),
            "pca_site_menus": list(normalized["pca_site_menus"]),
            "pca_basis_source_modes": list(normalized["pca_basis_source_modes"]),
            "pca_num_bands_values": list(normalized["pca_num_bands_values"]),
            "pca_support_extraction_mode": str(normalized["pca_support_extraction_mode"]),
            "pca_cache_signatures": bool(normalized["pca_cache_signatures"]),
            "pca_write_epsilon_artifacts": bool(normalized["pca_write_epsilon_artifacts"]),
            "pca_write_support_artifact": bool(normalized["pca_write_support_artifact"]),
            "guided_mask_names": list(normalized["guided_mask_names"]),
            "guided_support_dim_count": int(normalized["guided_support_dim_count"]),
            "guided_subspace_dims": None
            if normalized["guided_subspace_dims"] is None
            else list(normalized["guided_subspace_dims"]),
            "stage_c_dim_hint_scale_factors": list(normalized["dim_hint_scale_factors"]),
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
    fixed_stage_a_layers = normalized["stage_a_fixed_layers"]

    stage_a_rankings_by_token: dict[str, dict[str, list[dict[str, object]]]] = {}
    stage_a_summary_paths: list[Path] = []
    if fixed_stage_a_layers:
        for token_position_id in normalized["stage_a_token_position_ids"]:
            ranking_json_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.json"
            ranking_txt_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.txt"
            rankings = _fixed_stage_a_rankings(
                token_position_id=str(token_position_id),
                fixed_layers_by_var={
                    str(target_var): tuple(int(layer) for layer in layers)
                    for target_var, layers in fixed_stage_a_layers.items()
                },
            )
            _write_json(ranking_json_path, rankings)
            _write_text(
                ranking_txt_path,
                _format_stage_a_summary(token_position_id=str(token_position_id), rankings=rankings),
            )
            _print_stage_a_fixed_layer_override(
                token_position_id=str(token_position_id),
                fixed_layers_by_var={
                    str(target_var): tuple(int(layer) for layer in layers)
                    for target_var, layers in fixed_stage_a_layers.items()
                },
            )
            stage_a_rankings_by_token[str(token_position_id)] = rankings
            stage_a_summary_paths.extend([ranking_json_path, ranking_txt_path])
            _mark_stage(
                manifest_path=manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                stage_statuses=stage_statuses,
                stage_name=f"stage_a_{str(token_position_id)}",
                payload={
                    "state": "skipped_fixed_layers",
                    "stage_timestamp": None,
                    "expected_outputs": [str(ranking_json_path), str(ranking_txt_path)],
                    "runtime_seconds": 0.0,
                    "wall_runtime_seconds": 0.0,
                    "fixed_layers_by_var": {
                        str(target_var): [int(layer) for layer in layers]
                        for target_var, layers in fixed_stage_a_layers.items()
                    },
                    "completed_at": datetime.now().isoformat(),
                },
            )
    elif "stage_a_plot_layer" in normalized["stages"]:
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
    native_ot_rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    stage_b_selection_logged: set[str] = set()
    if "stage_b_plot_native_support" in normalized["stages"]:
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            if str(token_position_id) != "last_token":
                continue
            selected_layers_by_var = _select_stage_b_layers_by_var(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_target_vars_by_layer = _selected_target_vars_by_layer(selected_by_var=selected_layers_by_var)
            if not selected_layers:
                continue
            _print_stage_b_layer_selection(
                token_position_id=str(token_position_id),
                selected_by_var=selected_layers_by_var,
                selected_layers=selected_layers,
            )
            stage_b_selection_logged.add(str(token_position_id))
            stage_name = f"stage_b_native_{str(token_position_id)}"
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_native_{str(token_position_id)}"
            native_root = results_root / f"{stage_timestamp}_mcqa_plot_native_support"
            native_resolutions = tuple(int(width) for width in normalized["native_resolutions"])
            manifest_output_path = native_root / "layer_sweep_manifest.json"
            payload_output_paths = _expected_stage_b_native_payload_paths(
                native_root=native_root,
                selected_layers=selected_layers,
                selected_target_vars_by_layer=selected_target_vars_by_layer,
                native_resolutions=native_resolutions,
                signature_mode=str(args.signature_mode),
            )
            expected_outputs = [str(manifest_output_path), *[str(path) for path in payload_output_paths]]
            if all(
                _stage_b_payload_matches_target_vars(
                    payload_path=path,
                    expected_target_vars=_target_vars_from_target_suffixed_path(
                        path,
                        fallback=selected_target_vars_by_layer.get(
                            int(path.parent.name.split("_")[1]),
                            tuple(str(target_var) for target_var in DEFAULT_TARGET_VARS),
                        ),
                    ),
                )
                for path in payload_output_paths
            ):
                native_payload_paths.extend(payload_output_paths)
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
                description=(
                    f"Native support localization for selected {token_position_id} layers {list(selected_layers)} "
                    f"across widths {list(native_resolutions)}."
                ),
                stage_timestamp=stage_timestamp,
                command=_build_native_block_command(
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                    layers=selected_layers,
                    layer_target_vars=selected_target_vars_by_layer,
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
            missing_outputs = [str(path) for path in payload_output_paths if not _stage_output_is_valid(path)]
            if missing_outputs:
                raise RuntimeError(f"Stage {stage_name} missing outputs: {missing_outputs}")
            native_payload_paths.extend(payload_output_paths)
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
    elif (
        "stage_c_plot_das_native_support" in normalized["stages"]
        or "stage_c_plot_das_dimension" in normalized["stages"]
    ):
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            if str(token_position_id) != "last_token":
                continue
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_layers_by_var = _select_stage_b_layers_by_var(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_target_vars_by_layer = _selected_target_vars_by_layer(selected_by_var=selected_layers_by_var)
            if not selected_layers:
                continue
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_native_{str(token_position_id)}"
            native_root = results_root / f"{stage_timestamp}_mcqa_plot_native_support"
            native_resolutions = tuple(int(width) for width in normalized["native_resolutions"])
            native_payload_paths.extend(
                path
                for path in _expected_stage_b_native_payload_paths(
                    native_root=native_root,
                    selected_layers=selected_layers,
                    selected_target_vars_by_layer=selected_target_vars_by_layer,
                    native_resolutions=native_resolutions,
                    signature_mode=str(args.signature_mode),
                )
                if _stage_output_is_valid(path)
            )
        if native_payload_paths:
            native_ot_rankings = _extract_native_support_rankings(payload_paths=native_payload_paths)
        else:
            native_rankings_path = sweep_root / "stage_b_native_support_rankings.json"
            if _stage_output_is_valid(native_rankings_path):
                maybe_rankings = _load_json(native_rankings_path)
                if isinstance(maybe_rankings, dict):
                    native_ot_rankings = {
                        str(target_var): [dict(entry) for entry in entries if isinstance(entry, dict)]
                        for target_var, entries in maybe_rankings.items()
                        if isinstance(entries, list)
                    }

    if "stage_c_plot_das_layer" in normalized["stages"]:
        layer_das_payload_paths: list[Path] = []
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            selected_layers_by_var = _select_stage_b_layers_by_var(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_target_vars_by_layer = _selected_target_vars_by_layer(selected_by_var=selected_layers_by_var)
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            for layer in selected_layers:
                target_vars = selected_target_vars_by_layer.get(int(layer), tuple(str(target_var) for target_var in DEFAULT_TARGET_VARS))
                stage_name = f"stage_c_layer_{str(token_position_id)}_L{int(layer):02d}"
                stage_timestamp = f"{str(normalized['results_timestamp'])}_stageC_layer_{str(token_position_id)}_L{int(layer):02d}"
                run_root = results_root / f"{stage_timestamp}_mcqa_plot_das_layer"
                payload_path = run_root / f"layer_{int(layer):02d}" / f"mcqa_plot_das_layer_layer-{int(layer)}_pos-{str(token_position_id)}_summary.json"
                if not _das_payload_matches_target_vars(payload_path=payload_path, expected_target_vars=target_vars):
                    stage = SweepStage(
                        name=stage_name,
                        category="stage_c_plot_das_layer",
                        description=(
                            f"Standalone layer DAS for token position {token_position_id} layer {int(layer)} "
                            f"target_vars={list(target_vars)}."
                        ),
                        stage_timestamp=stage_timestamp,
                        command=_build_layer_das_command(
                            args=args,
                            normalized=normalized,
                            stage_timestamp=stage_timestamp,
                            token_position_id=str(token_position_id),
                            layer=int(layer),
                            target_vars=target_vars,
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

    if "stage_c_plot_das_native_support" in normalized["stages"] and native_ot_rankings:
        native_das_payload_paths: list[Path] = []
        native_support_entries = _select_stage_c_native_support_entries(
            rankings=native_ot_rankings,
            top_configs_per_var=1,
        )
        for entry in native_support_entries:
            target_var = str(entry.get("variable"))
            if target_var not in DEFAULT_TARGET_VARS:
                continue
            native_payload_path = Path(str(entry.get("payload_path", "")))
            if not _stage_output_is_valid(native_payload_path):
                continue
            native_payload = _load_json(native_payload_path)
            if not isinstance(native_payload, dict):
                continue
            layer = int(native_payload.get("layer"))
            native_resolution = int(native_payload.get("native_resolution", native_payload.get("atomic_width")))
            target_vars = (target_var,)
            stage_name = f"stage_c_native_{target_var}_L{int(layer):02d}_W{int(native_resolution):04d}"
            stage_timestamp = (
                f"{str(normalized['results_timestamp'])}_stageC_native_{target_var}"
                f"_L{int(layer):02d}_W{int(native_resolution):04d}"
            )
            run_root = results_root / f"{stage_timestamp}_mcqa_plot_das_native_support"
            payload_path = run_root / f"layer_{int(layer):02d}" / f"mcqa_plot_das_native_support_layer-{int(layer)}_summary.json"
            if not _das_payload_matches_target_vars(payload_path=payload_path, expected_target_vars=target_vars):
                stage = SweepStage(
                    name=stage_name,
                    category="stage_c_plot_das_native_support",
                    description=(
                        f"Standalone native-support DAS for layer {int(layer)} width {int(native_resolution)} "
                        f"target_vars={list(target_vars)}."
                    ),
                    stage_timestamp=stage_timestamp,
                    command=_build_native_support_das_command(
                        args=args,
                        normalized=normalized,
                        stage_timestamp=stage_timestamp,
                        native_support_path=native_payload_path,
                        target_vars=target_vars,
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

    if "stage_c_plot_das_dimension" in normalized["stages"] and native_ot_rankings:
        dimension_payload_records: list[tuple[Path, dict[str, object]]] = []
        selected_entries = _select_stage_c_native_support_entries(
            rankings=native_ot_rankings,
            top_configs_per_var=1,
        )
        for entry in selected_entries:
            target_var = str(entry.get("variable"))
            if target_var not in DEFAULT_TARGET_VARS:
                continue
            effective_dim = int(entry.get("selected_site_total_dim") or 0)
            if effective_dim <= 0:
                continue
            native_support_path = entry.get("payload_path")
            if native_support_path is None or not _stage_output_is_valid(Path(str(native_support_path))):
                continue
            layer = int(entry["layer"])
            native_resolution = int(entry.get("native_resolution", entry.get("atomic_width", -1)))
            das_subspace_dims = _dim_hint_subspace_dims(
                effective_dim,
                scale_factors=normalized["dim_hint_scale_factors"],
            )
            stage_name = f"stage_c_dimension_{target_var}_L{int(layer):02d}_W{int(native_resolution):04d}"
            stage_timestamp = (
                f"{str(normalized['results_timestamp'])}_stageC_dimension_{target_var}"
                f"_L{int(layer):02d}_W{int(native_resolution):04d}"
            )
            run_root = results_root / f"{stage_timestamp}_mcqa_plot_das_layer"
            payload_path = run_root / f"layer_{int(layer):02d}" / (
                f"mcqa_plot_das_layer_layer-{int(layer)}_pos-last_token_summary.json"
            )
            if not _das_payload_matches_target_vars(payload_path=payload_path, expected_target_vars=(target_var,)):
                stage = SweepStage(
                    name=stage_name,
                    category="stage_c_plot_das_dimension",
                    description=(
                        f"Full-layer DAS around selected native support width for {target_var}: "
                        f"layer {int(layer)} width {int(native_resolution)} dims={list(das_subspace_dims)}."
                    ),
                    stage_timestamp=stage_timestamp,
                    command=_build_layer_das_command(
                        args=args,
                        normalized=normalized,
                        stage_timestamp=stage_timestamp,
                        token_position_id="last_token",
                        layer=int(layer),
                        target_vars=(target_var,),
                        das_subspace_dims=das_subspace_dims,
                    ),
                    expected_outputs=(str(payload_path),),
                )
                _run_stage_command(stage=stage, repo_root=repo_root)
            if _stage_output_is_valid(payload_path):
                metadata = dict(entry)
                metadata["dim_hint_effective_dim"] = int(effective_dim)
                metadata["dim_hint_subspace_dims"] = list(das_subspace_dims)
                dimension_payload_records.append((payload_path, metadata))
        dimension_das_rankings = _extract_dimension_das_rankings(payload_records=dimension_payload_records)
        _write_json(sweep_root / "stage_c_dimension_das_rankings.json", dimension_das_rankings)
        _write_text(
            sweep_root / "stage_c_dimension_das_rankings.txt",
            _format_native_summary(title="MCQA Hierarchical Stage C Dimension DAS Ranking", rankings=dimension_das_rankings),
        )

    stage_b_payload_paths: list[Path] = []
    stage_b_rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    if "stage_b_plot_pca_support" in normalized["stages"]:
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            selected_layers_by_var = _select_stage_b_layers_by_var(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
                neighbor_radius=int(args.stage_b_neighbor_radius),
                max_layers_per_var=int(args.stage_b_max_layers_per_var),
            )
            selected_target_vars_by_layer = _selected_target_vars_by_layer(selected_by_var=selected_layers_by_var)
            if not selected_layers:
                continue
            if str(token_position_id) not in stage_b_selection_logged:
                _print_stage_b_layer_selection(
                    token_position_id=str(token_position_id),
                    selected_by_var=selected_layers_by_var,
                    selected_layers=selected_layers,
                )
                stage_b_selection_logged.add(str(token_position_id))
            for basis_source_mode in normalized["pca_basis_source_modes"]:
                for site_menu in normalized["pca_site_menus"]:
                    num_bands_values = _normalize_num_bands_values(normalized["pca_num_bands_values"], str(site_menu))
                    if not num_bands_values:
                        continue
                    band_slug = "bands-" + "-".join(str(int(value)) for value in num_bands_values)
                    stage_slug = (
                        f"{str(token_position_id)}_{str(basis_source_mode)}_{str(site_menu)}_{band_slug}"
                    )
                    stage_name = f"stage_b_{stage_slug}"
                    stage_timestamp = f"{str(normalized['results_timestamp'])}_stageB_{stage_slug}"
                    sweep_run_root = results_root / f"{stage_timestamp}_mcqa_ot_pca_focus"
                    expected_outputs = [str(sweep_run_root / "layer_sweep_manifest.json")]
                    for layer in selected_layers:
                        target_vars = selected_target_vars_by_layer.get(
                            int(layer),
                            tuple(str(target_var) for target_var in DEFAULT_TARGET_VARS),
                        )
                        for num_bands in num_bands_values:
                            site_catalog_tag = _site_catalog_tag(
                                site_menu=str(site_menu),
                                num_bands=int(num_bands),
                                band_scheme=str(args.pca_band_scheme),
                            )
                            for target_var in target_vars:
                                expected_outputs.append(
                                    str(
                                        sweep_run_root
                                        / f"layer_{int(layer):02d}"
                                        / (
                                            f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                                            f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}"
                                            f"{_target_file_suffix(str(target_var))}_plot_pca_support.json"
                                        )
                                    )
                                )
                    if _stage_output_is_valid(Path(expected_outputs[0])) and all(
                        _stage_b_payload_matches_target_vars(
                            payload_path=Path(path),
                            expected_target_vars=_target_vars_from_target_suffixed_path(
                                Path(path),
                                fallback=selected_target_vars_by_layer.get(
                                    int(Path(path).parent.name.split("_")[1]),
                                    tuple(str(target_var) for target_var in DEFAULT_TARGET_VARS),
                                ),
                            ),
                            expected_support_extraction_mode=str(normalized["pca_support_extraction_mode"]),
                        )
                        for path in expected_outputs[1:]
                    ):
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
                            f"basis={basis_source_mode}, menu={site_menu}, bands={list(num_bands_values)}, "
                            f"layers={list(selected_layers)}."
                        ),
                        stage_timestamp=stage_timestamp,
                        command=_build_stage_b_or_c_command(
                            args=args,
                            normalized=normalized,
                            stage_timestamp=stage_timestamp,
                            token_position_id=str(token_position_id),
                            layers=selected_layers,
                            requested_target_vars=tuple(str(target_var) for target_var in normalized["target_vars"]),
                            layer_target_vars=selected_target_vars_by_layer,
                            basis_source_mode=str(basis_source_mode),
                            site_menu=str(site_menu),
                            num_bands=int(num_bands_values[0]),
                            num_bands_values=tuple(int(value) for value in num_bands_values),
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
        selected_config_entries = _select_stage_c_config_entries(
            rankings=stage_b_rankings,
            top_configs_per_var=1,
        )
        stage_c_payload_paths: list[Path] = []
        for entry in selected_config_entries:
            target_var = str(entry.get("variable"))
            if target_var not in DEFAULT_TARGET_VARS:
                continue
            token_position_id = str(entry["token_position_id"])
            basis_source_mode = str(entry["basis_source_mode"])
            site_menu = str(entry["site_menu"])
            num_bands = int(entry["num_bands"])
            layer = int(entry["layer"])
            layers = (int(layer),)
            layer_target_vars = {int(layer): (target_var,)}
            stage_slug = _stage_b_slug(
                token_position_id=str(token_position_id),
                basis_source_mode=str(basis_source_mode),
                site_menu=str(site_menu),
                num_bands=int(num_bands),
            )
            stage_name = f"stage_c_{target_var}_{stage_slug}_L{int(layer):02d}"
            stage_timestamp = (
                f"{str(normalized['results_timestamp'])}_stageC_pca_{target_var}"
                f"_{stage_slug}_L{int(layer):02d}"
            )
            sweep_run_root = results_root / f"{stage_timestamp}_mcqa_ot_pca_focus"
            site_catalog_tag = _site_catalog_tag(
                site_menu=str(site_menu),
                num_bands=int(num_bands),
                band_scheme=str(args.pca_band_scheme),
            )
            target_suffix = _target_file_suffix(target_var)
            plot_payload_path = (
                sweep_run_root
                / f"layer_{int(layer):02d}"
                / (
                    f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                    f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}"
                    f"{target_suffix}_plot_pca_support.json"
                )
            )
            das_payload_path = (
                sweep_run_root
                / f"layer_{int(layer):02d}"
                / (
                    f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                    f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}"
                    f"{target_suffix}_plot_das_pca_support.json"
                )
            )
            guided_payload_path = (
                sweep_run_root
                / f"layer_{int(layer):02d}"
                / (
                    f"mcqa_layer-{int(layer)}_pos-{str(token_position_id)}_pca-{site_catalog_tag}"
                    f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_{target_var}_das_guided.json"
                )
            )
            expected_outputs = [str(plot_payload_path), str(das_payload_path), str(guided_payload_path)]
            if (
                all(_stage_output_is_valid(Path(path)) for path in expected_outputs)
                and _stage_b_payload_matches_target_vars(
                    payload_path=plot_payload_path,
                    expected_target_vars=(target_var,),
                    expected_support_extraction_mode=str(normalized["pca_support_extraction_mode"]),
                )
                and _das_payload_matches_target_vars(
                    payload_path=das_payload_path,
                    expected_target_vars=(target_var,),
                )
            ):
                stage_c_payload_paths.append(das_payload_path)
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
                    f"Guided DAS on selected PCA OT coupling for {target_var}: token_position={token_position_id}, "
                    f"basis={basis_source_mode}, menu={site_menu}, bands={num_bands}, layer={int(layer)}."
                ),
                stage_timestamp=stage_timestamp,
                command=_build_stage_b_or_c_command(
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                    token_position_id=str(token_position_id),
                    layers=layers,
                    requested_target_vars=(target_var,),
                    layer_target_vars=layer_target_vars,
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
            stage_c_payload_paths.append(das_payload_path)
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
