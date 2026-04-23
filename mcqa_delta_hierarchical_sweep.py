from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


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
DEFAULT_PCA_NUM_BANDS_VALUES = (8,)
DEFAULT_PCA_BAND_SCHEME = "equal"
DEFAULT_PCA_TOP_PREFIX_SIZES = (8, 16, 32, 64)
DEFAULT_GUIDED_MASK_NAMES = ("Top1", "Top2", "Top4", "S50", "S80")
DEFAULT_STAGES = ("stage_a_layer_ot", "stage_b_pca_ot", "stage_c_guided_das")


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
            float(entry.get("exact_acc", -1.0)),
            float(entry.get("selection_score", entry.get("cal", -1.0))),
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
        return resolved
    return resolved


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
    parser = argparse.ArgumentParser(description="Hierarchical Delta sweep for MCQA layer discovery -> PCA OT -> guided DAS.")
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
    parser.add_argument("--stages", default="stage_a_layer_ot,stage_b_pca_ot,stage_c_guided_das")
    parser.add_argument("--stage-a-token-position-ids", default="last_token")
    parser.add_argument("--stage-a-layer-indices", default=None, help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument("--target-vars", default="answer_pointer,answer_token")
    parser.add_argument("--ot-epsilons", default="0.5,1,2,4")
    parser.add_argument("--ot-top-k-values", default="1,2,4")
    parser.add_argument("--ot-lambdas", default="0.5,1,2,4")
    parser.add_argument("--calibration-metric", default=DEFAULT_CALIBRATION_METRIC)
    parser.add_argument("--calibration-family-weights", default="1,1.5,2")
    parser.add_argument("--stage-b-top-layers-per-var", type=int, default=3)
    parser.add_argument("--pca-site-menus", default="partition,mixed")
    parser.add_argument("--pca-basis-source-modes", default="pair_bank,all_variants")
    parser.add_argument("--pca-num-bands-values", default="8")
    parser.add_argument("--pca-band-scheme", default=DEFAULT_PCA_BAND_SCHEME, choices=("equal", "head"))
    parser.add_argument("--pca-top-prefix-sizes", default="8,16,32,64")
    parser.add_argument("--stage-c-top-configs-per-var", type=int, default=2)
    parser.add_argument("--guided-mask-names", default="Top1,Top2,Top4,S50,S80")
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--guided-subspace-dims", default=None)
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
    ot_epsilons = _parse_csv_floats(args.ot_epsilons) or DEFAULT_OT_EPSILONS
    ot_top_k_values = _parse_csv_ints(args.ot_top_k_values) or DEFAULT_OT_TOP_K_VALUES
    ot_lambdas = _parse_csv_floats(args.ot_lambdas) or DEFAULT_OT_LAMBDAS
    calibration_family_weights = _parse_csv_floats(args.calibration_family_weights) or DEFAULT_CALIBRATION_FAMILY_WEIGHTS
    guided_mask_names = _parse_csv_strings(args.guided_mask_names) or DEFAULT_GUIDED_MASK_NAMES
    guided_subspace_dims = None
    if args.guided_subspace_dims is not None:
        guided_subspace_dims = _parse_csv_ints(args.guided_subspace_dims)
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
        "ot_epsilons": tuple(float(epsilon) for epsilon in ot_epsilons),
        "ot_top_k_values": tuple(int(value) for value in ot_top_k_values),
        "ot_lambdas": tuple(float(value) for value in ot_lambdas),
        "calibration_family_weights": tuple(float(weight) for weight in calibration_family_weights),
        "guided_mask_names": tuple(str(mask_name) for mask_name in guided_mask_names),
        "guided_subspace_dims": None if guided_subspace_dims is None else tuple(int(dim) for dim in guided_subspace_dims),
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
        "mcqa_run_cloud.py",
        "--preset",
        "full",
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
        "--methods",
        "ot",
        "--target-vars",
        ",".join(str(target_var) for target_var in normalized["target_vars"]),
        "--token-position-ids",
        str(token_position_id),
        "--resolutions",
        "full",
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--ot-top-k-values",
        ",".join(str(value) for value in normalized["ot_top_k_values"]),
        "--ot-lambdas",
        ",".join(str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value) for value in normalized["ot_lambdas"]),
        "--signature-modes",
        str(args.signature_mode),
        "--calibration-metric",
        str(args.calibration_metric),
        "--calibration-family-weights",
        ",".join(str(weight).rstrip("0").rstrip(".") if "." in str(weight) else str(weight) for weight in normalized["calibration_family_weights"]),
        "--results-root",
        str(args.results_root),
        "--results-timestamp",
        str(stage_timestamp),
        "--signatures-dir",
        str(args.signatures_dir),
        "--layer-sweep",
        "--layer-indices",
        ",".join(str(layer) for layer in layer_indices),
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


def _run_stage_command(*, stage: SweepStage, repo_root: Path) -> None:
    subprocess.run(stage.command, cwd=repo_root, check=True)


def _extract_stage_a_rankings(*, manifest_path: Path) -> dict[str, list[dict[str, object]]]:
    manifest_payload = _load_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Unexpected layer sweep manifest payload at {manifest_path}")
    rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    runs = manifest_payload.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"Malformed layer sweep manifest at {manifest_path}")
    for run in runs:
        if not isinstance(run, dict):
            continue
        layer = int(run.get("layer"))
        output_path = Path(str(run.get("output_path")))
        aggregate_payload = _load_json(output_path)
        if not isinstance(aggregate_payload, dict):
            continue
        compare_runs = aggregate_payload.get("runs", [])
        layer_best: dict[tuple[str, str], dict[str, object]] = {}
        for compare_payload in compare_runs:
            if not isinstance(compare_payload, dict):
                continue
            epsilon = float(compare_payload.get("ot_epsilon", -1.0))
            config = compare_payload.get("config", {})
            token_position_ids = config.get("token_position_ids") or ()
            token_position_id = str(token_position_ids[0]) if token_position_ids else ""
            for result in compare_payload.get("results", []):
                if not isinstance(result, dict):
                    continue
                if str(result.get("method")) != "ot":
                    continue
                target_var = str(result.get("variable"))
                entry = {
                    "variable": target_var,
                    "token_position_id": token_position_id,
                    "layer": layer,
                    "exact_acc": float(result.get("exact_acc", -1.0)),
                    "selection_score": _selection_score(result),
                    "epsilon": epsilon,
                    "site_label": result.get("site_label"),
                    "source_path": str(output_path),
                }
                key = (target_var, token_position_id)
                current = layer_best.get(key)
                if current is None or _score_tuple(entry) > _score_tuple(current):
                    layer_best[key] = entry
        for (target_var, _token_position_id), entry in layer_best.items():
            rankings.setdefault(target_var, []).append(entry)
    for target_var in list(rankings):
        rankings[target_var] = _sort_best_first(rankings[target_var])
    return rankings


def _format_stage_a_summary(*, token_position_id: str, rankings: dict[str, list[dict[str, object]]]) -> str:
    lines = [
        "MCQA Hierarchical Stage A Layer Discovery",
        f"token_position_id: {token_position_id}",
        "",
    ]
    for target_var in DEFAULT_TARGET_VARS:
        lines.append(f"[{target_var}]")
        for entry in rankings.get(target_var, []):
            lines.append(
                "  - "
                f"layer={int(entry['layer'])} "
                f"exact={float(entry['exact_acc']):.4f} "
                f"cal={float(entry['selection_score']):.4f} "
                f"eps={float(entry['epsilon']):g} "
                f"site={entry.get('site_label')}"
            )
        lines.append("")
    return "\n".join(lines)


def _select_stage_b_layers(
    *,
    rankings: dict[str, list[dict[str, object]]],
    top_layers_per_var: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    for target_var in DEFAULT_TARGET_VARS:
        for entry in rankings.get(target_var, [])[: max(1, int(top_layers_per_var))]:
            selected.append(int(entry["layer"]))
    return tuple(sorted(dict.fromkeys(selected)))


def _extract_stage_b_best_configs(*, payload_paths: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int, str, str, int], dict[str, object]] = {}
    for payload_path in payload_paths:
        layer_payload = _load_json(payload_path)
        if not isinstance(layer_payload, dict):
            continue
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
            lines.append(
                "  - "
                f"layer={int(entry['layer'])} "
                f"pos={entry['token_position_id']} "
                f"basis={entry['basis_source_mode']} "
                f"menu={entry['site_menu']} "
                f"bands={int(entry['num_bands'])} "
                f"exact={float(entry['exact_acc']):.4f} "
                f"cal={float(entry['selection_score']):.4f} "
                f"eps={float(entry['epsilon']):g} "
                f"site={entry.get('site_label')}"
            )
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
            lines.append(
                "  - "
                f"layer={int(entry['layer'])} "
                f"pos={entry['token_position_id']} "
                f"basis={entry['basis_source_mode']} "
                f"menu={entry['site_menu']} "
                f"bands={int(entry['num_bands'])} "
                f"exact={float(entry['exact_acc']):.4f} "
                f"cal={float(entry['selection_score']):.4f} "
                f"site={entry.get('site_label')} "
                f"dim={entry.get('subspace_dim')}"
            )
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
            "stage_c_top_configs_per_var": int(args.stage_c_top_configs_per_var),
            "pca_site_menus": list(normalized["pca_site_menus"]),
            "pca_basis_source_modes": list(normalized["pca_basis_source_modes"]),
            "pca_num_bands_values": list(normalized["pca_num_bands_values"]),
            "guided_mask_names": list(normalized["guided_mask_names"]),
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
    if "stage_a_layer_ot" in normalized["stages"]:
        for token_position_id in normalized["stage_a_token_position_ids"]:
            stage_name = f"stage_a_{str(token_position_id)}"
            stage_timestamp = f"{str(normalized['results_timestamp'])}_stageA_{str(token_position_id)}"
            run_root = results_root / f"{stage_timestamp}_mcqa_layer_sweep"
            stage_manifest = run_root / "layer_sweep_manifest.json"
            ranking_json_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.json"
            ranking_txt_path = sweep_root / f"stage_a_{str(token_position_id)}_layer_rankings.txt"
            if _stage_output_is_valid(stage_manifest) and _stage_output_is_valid(ranking_json_path):
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
                        "expected_outputs": [str(stage_manifest), str(ranking_json_path), str(ranking_txt_path)],
                        "completed_at": datetime.now().isoformat(),
                    },
                )
                continue
            stage = SweepStage(
                name=stage_name,
                category="stage_a_layer_ot",
                description=f"All-layer full-vector OT discovery at token position {token_position_id}.",
                stage_timestamp=stage_timestamp,
                command=_build_stage_a_command(
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                    token_position_id=str(token_position_id),
                    layer_indices=tuple(int(layer) for layer in layer_indices),
                ),
                expected_outputs=(str(stage_manifest),),
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
                    "expected_outputs": [str(stage_manifest), str(ranking_json_path), str(ranking_txt_path)],
                    "started_at": datetime.now().isoformat(),
                },
            )
            _run_stage_command(stage=stage, repo_root=repo_root)
            rankings = _extract_stage_a_rankings(manifest_path=stage_manifest)
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
                    "expected_outputs": [str(stage_manifest), str(ranking_json_path), str(ranking_txt_path)],
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

    stage_b_payload_paths: list[Path] = []
    stage_b_rankings: dict[str, list[dict[str, object]]] = {target_var: [] for target_var in DEFAULT_TARGET_VARS}
    if "stage_b_pca_ot" in normalized["stages"]:
        for token_position_id, rankings in stage_a_rankings_by_token.items():
            selected_layers = _select_stage_b_layers(
                rankings=rankings,
                top_layers_per_var=int(args.stage_b_top_layers_per_var),
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
                                        f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_ot_pca.json"
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
                            category="stage_b_pca_ot",
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
                        _run_stage_command(stage=stage, repo_root=repo_root)
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
                                "completed_at": datetime.now().isoformat(),
                            },
                        )
        stage_b_rankings = _extract_stage_b_best_configs(payload_paths=stage_b_payload_paths)
        stage_b_json_path = sweep_root / "stage_b_pca_rankings.json"
        stage_b_txt_path = sweep_root / "stage_b_pca_rankings.txt"
        _write_json(stage_b_json_path, stage_b_rankings)
        _write_text(stage_b_txt_path, _format_stage_b_summary(rankings=stage_b_rankings))
    else:
        stage_b_json_path = sweep_root / "stage_b_pca_rankings.json"
        if _stage_output_is_valid(stage_b_json_path):
            payload = _load_json(stage_b_json_path)
            if isinstance(payload, dict):
                stage_b_rankings = payload

    if "stage_c_guided_das" in normalized["stages"]:
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
                            f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_ot_pca.json"
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
                stage_c_payload_paths.extend(Path(path) for path in expected_outputs if path.endswith("_ot_pca.json"))
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
                category="stage_c_guided_das",
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
            _run_stage_command(stage=stage, repo_root=repo_root)
            missing_outputs = [path for path in expected_outputs if not _stage_output_is_valid(Path(path))]
            if missing_outputs:
                raise RuntimeError(f"Stage {stage_name} missing outputs: {missing_outputs}")
            stage_c_payload_paths.extend(Path(path) for path in expected_outputs if path.endswith("_ot_pca.json"))
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
                    "completed_at": datetime.now().isoformat(),
                },
            )
        stage_c_rankings = _extract_stage_c_rankings(payload_paths=stage_c_payload_paths)
        stage_c_json_path = sweep_root / "stage_c_guided_rankings.json"
        stage_c_txt_path = sweep_root / "stage_c_guided_rankings.txt"
        _write_json(stage_c_json_path, stage_c_rankings)
        _write_text(stage_c_txt_path, _format_stage_c_summary(rankings=stage_c_rankings))

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
        for path in status.get("expected_outputs", []):
            lines.append(f"  {path}")
        lines.append("")
    _write_text(summary_path, "\n".join(lines))
    print(f"Wrote hierarchical sweep manifest to {manifest_path}")
    print(f"Wrote hierarchical sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
