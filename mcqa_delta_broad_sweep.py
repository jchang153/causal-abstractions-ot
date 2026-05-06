from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_LAYERS = (20, 25)
DEFAULT_TARGET_VARS = ("answer_pointer", "answer_token")
DEFAULT_FULL_TOKEN_POSITION_IDS = ("correct_symbol", "correct_symbol_period", "last_token")
DEFAULT_PCA_TOKEN_POSITION_ID = "last_token"
DEFAULT_SIGNATURE_MODE = "family_label_delta_norm"
DEFAULT_CALIBRATION_METRIC = "family_weighted_macro_exact_acc"
DEFAULT_CALIBRATION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_OT_EPSILONS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_OT_TOP_K_VALUES = (1, 2, 4)
DEFAULT_OT_LAMBDAS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_REGULAR_DAS_SUBSPACE_DIMS = (
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
DEFAULT_PCA_SITE_MENUS = ("partition",)
DEFAULT_PCA_BASIS_SOURCE_MODES = ("all_variants",)
DEFAULT_PCA_NUM_BANDS = 8
DEFAULT_PCA_BAND_SCHEME = "equal"
DEFAULT_GUIDED_PCA_CONFIGS = ("all_variants:partition",)
DEFAULT_GUIDED_MASK_NAMES = ("Selected",)
DEFAULT_STAGES = ("vanilla_ot", "pca_ot", "pca_guided_das", "regular_das")


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


def _site_catalog_tag(*, site_menu: str, num_bands: int, band_scheme: str) -> str:
    return f"menu-{str(site_menu)}-bands-{int(num_bands)}-scheme-{str(band_scheme)}"


def _pca_config_slug(*, basis_source_mode: str, site_menu: str) -> str:
    return f"{str(basis_source_mode)}_{str(site_menu)}"


def _canonical_pca_basis_source_mode(value: str) -> str:
    mode = str(value)
    if mode == "pair_bank":
        return "all_variants"
    if mode != "all_variants":
        raise ValueError(
            f"Unsupported PCA basis source mode {value!r}; PCA uses the canonical broad all_variants point cloud."
        )
    return mode


def _append_optional_arg(args: list[str], name: str, value: str | None) -> None:
    if value is None or value == "":
        return
    args.extend([name, value])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable Delta broad sweep for MCQA OT/PCA/DAS experiments.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", default="20,25")
    parser.add_argument("--results-root", default="results/delta")
    parser.add_argument("--results-timestamp")
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--signature-mode", default=DEFAULT_SIGNATURE_MODE)
    parser.add_argument("--stages", default="vanilla_ot,pca_ot,pca_guided_das,regular_das")
    parser.add_argument("--full-token-position-ids", default="correct_symbol,correct_symbol_period,last_token")
    parser.add_argument("--ot-epsilons", default="0.5,1,2,4")
    parser.add_argument("--ot-top-k-values", default="1,2,4")
    parser.add_argument("--ot-lambdas", default="0.5,1,2,4")
    parser.add_argument("--calibration-metric", default=DEFAULT_CALIBRATION_METRIC)
    parser.add_argument("--calibration-family-weights", default="1,1,1")
    parser.add_argument(
        "--regular-das-subspace-dims",
        default="32,64,96,128,256,512,768,1024,1536,2048,2304",
    )
    parser.add_argument("--regular-das-max-epochs", type=int, default=100)
    parser.add_argument("--regular-das-min-epochs", type=int, default=5)
    parser.add_argument("--regular-das-plateau-patience", type=int, default=2)
    parser.add_argument("--regular-das-plateau-rel-delta", type=float, default=1e-3)
    parser.add_argument("--regular-das-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pca-token-position-id", default=DEFAULT_PCA_TOKEN_POSITION_ID)
    parser.add_argument("--pca-site-menus", default="partition")
    parser.add_argument(
        "--pca-basis-source-modes",
        default="all_variants",
        help="Comma-separated PCA basis source modes. pair_bank is treated as all_variants.",
    )
    parser.add_argument("--pca-num-bands", type=int, default=DEFAULT_PCA_NUM_BANDS)
    parser.add_argument("--pca-band-scheme", default=DEFAULT_PCA_BAND_SCHEME, choices=("equal", "head"))
    parser.add_argument("--guided-pca-configs", default="all_variants:partition")
    parser.add_argument(
        "--guided-mask-names",
        default="Selected",
        help="Deprecated compatibility flag. Guided DAS now uses the exact Selected PLOT handle.",
    )
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--guided-subspace-dims", default=None)
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def _normalize_args(args: argparse.Namespace) -> dict[str, object]:
    layers = _parse_csv_ints(args.layers) or DEFAULT_LAYERS
    stages = _parse_csv_strings(args.stages) or DEFAULT_STAGES
    full_token_position_ids = _parse_csv_strings(args.full_token_position_ids) or DEFAULT_FULL_TOKEN_POSITION_IDS
    ot_epsilons = _parse_csv_floats(args.ot_epsilons) or DEFAULT_OT_EPSILONS
    ot_top_k_values = _parse_csv_ints(args.ot_top_k_values) or DEFAULT_OT_TOP_K_VALUES
    ot_lambdas = _parse_csv_floats(args.ot_lambdas) or DEFAULT_OT_LAMBDAS
    calibration_family_weights = _parse_csv_floats(args.calibration_family_weights) or DEFAULT_CALIBRATION_FAMILY_WEIGHTS
    regular_das_subspace_dims = _parse_csv_ints(args.regular_das_subspace_dims) or DEFAULT_REGULAR_DAS_SUBSPACE_DIMS
    pca_site_menus = _parse_csv_strings(args.pca_site_menus) or DEFAULT_PCA_SITE_MENUS
    unsupported_pca_site_menus = sorted(set(str(site_menu) for site_menu in pca_site_menus) - {"partition"})
    if unsupported_pca_site_menus:
        raise ValueError(f"Unsupported PCA site menus: {unsupported_pca_site_menus}. PCA support is partition-only.")
    pca_basis_source_modes = tuple(
        dict.fromkeys(
            _canonical_pca_basis_source_mode(mode)
            for mode in (_parse_csv_strings(args.pca_basis_source_modes) or DEFAULT_PCA_BASIS_SOURCE_MODES)
        )
    )
    guided_pca_configs = _parse_csv_strings(args.guided_pca_configs) or DEFAULT_GUIDED_PCA_CONFIGS
    guided_mask_names = DEFAULT_GUIDED_MASK_NAMES
    guided_subspace_dims = None
    if args.guided_subspace_dims is not None:
        guided_subspace_dims = _parse_csv_ints(args.guided_subspace_dims)
    unsupported_stage_names = sorted(set(stages) - set(DEFAULT_STAGES))
    if unsupported_stage_names:
        raise ValueError(f"Unsupported stage names: {unsupported_stage_names}")
    guided_config_pairs: list[tuple[str, str]] = []
    for value in guided_pca_configs:
        if ":" not in value:
            raise ValueError(f"guided-pca-config entry must be basis:menu, got {value!r}")
        basis_source_mode, site_menu = (part.strip() for part in value.split(":", 1))
        if not basis_source_mode or not site_menu:
            raise ValueError(f"guided-pca-config entry must include both basis and menu, got {value!r}")
        guided_config_pairs.append((_canonical_pca_basis_source_mode(basis_source_mode), site_menu))
    guided_config_pairs = list(dict.fromkeys(guided_config_pairs))
    results_timestamp = (
        args.results_timestamp
        or os.environ.get("RESULTS_TIMESTAMP")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return {
        "layers": tuple(int(layer) for layer in layers),
        "stages": tuple(str(stage) for stage in stages),
        "full_token_position_ids": tuple(str(token_position_id) for token_position_id in full_token_position_ids),
        "ot_epsilons": tuple(float(epsilon) for epsilon in ot_epsilons),
        "ot_top_k_values": tuple(int(value) for value in ot_top_k_values),
        "ot_lambdas": tuple(float(value) for value in ot_lambdas),
        "calibration_family_weights": tuple(float(weight) for weight in calibration_family_weights),
        "regular_das_subspace_dims": tuple(int(dim) for dim in regular_das_subspace_dims),
        "pca_site_menus": tuple(str(site_menu) for site_menu in pca_site_menus),
        "pca_basis_source_modes": tuple(str(mode) for mode in pca_basis_source_modes),
        "guided_pca_configs": tuple((str(mode), str(menu)) for mode, menu in guided_config_pairs),
        "guided_mask_names": tuple(str(mask_name) for mask_name in guided_mask_names),
        "guided_subspace_dims": None if guided_subspace_dims is None else tuple(int(dim) for dim in guided_subspace_dims),
        "results_timestamp": str(results_timestamp),
    }


def _stage_output_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if path.suffix.lower() != ".json":
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, (dict, list))


def _build_cloud_compare_command(
    *,
    script_name: str,
    method: str,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        script_name,
        "--preset",
        "full",
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
        "--methods",
        str(method),
        "--layers",
        ",".join(str(layer) for layer in normalized["layers"]),
        "--token-position-ids",
        ",".join(str(token_position_id) for token_position_id in normalized["full_token_position_ids"]),
        "--resolutions",
        "full",
        "--signature-modes",
        str(args.signature_mode),
        "--ot-epsilons",
        ",".join(str(epsilon).rstrip("0").rstrip(".") if "." in str(epsilon) else str(epsilon) for epsilon in normalized["ot_epsilons"]),
        "--ot-top-k-values",
        ",".join(str(value) for value in normalized["ot_top_k_values"]),
        "--ot-lambdas",
        ",".join(str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value) for value in normalized["ot_lambdas"]),
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
    ]
    if method == "das":
        command.extend(
            [
                "--das-max-epochs",
                str(int(args.regular_das_max_epochs)),
                "--das-min-epochs",
                str(int(args.regular_das_min_epochs)),
                "--das-plateau-patience",
                str(int(args.regular_das_plateau_patience)),
                "--das-plateau-rel-delta",
                str(float(args.regular_das_plateau_rel_delta)),
                "--das-learning-rate",
                str(float(args.regular_das_learning_rate)),
                "--das-subspace-dims",
                ",".join(str(dim) for dim in normalized["regular_das_subspace_dims"]),
            ]
        )
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def _build_pca_command(
    *,
    args: argparse.Namespace,
    normalized: dict[str, object],
    stage_timestamp: str,
    basis_source_mode: str,
    site_menu: str,
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
        ",".join(str(layer) for layer in normalized["layers"]),
        "--token-position-id",
        str(args.pca_token_position_id),
        "--site-menu",
        str(site_menu),
        "--num-bands",
        str(int(args.pca_num_bands)),
        "--band-scheme",
        str(args.pca_band_scheme),
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
    _append_optional_arg(command, "--dataset-config", args.dataset_config)
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
    if bool(args.prompt_hf_login):
        command.append("--prompt-hf-login")
    return tuple(command)


def build_broad_sweep_plan(*, repo_root: Path, args: argparse.Namespace, normalized: dict[str, object]) -> list[SweepStage]:
    results_root = Path(args.results_root)
    stages: list[SweepStage] = []
    timestamp_root = str(normalized["results_timestamp"])
    if "vanilla_ot" in normalized["stages"]:
        stage_timestamp = f"{timestamp_root}_vanilla_ot"
        run_dir = results_root / f"{stage_timestamp}_mcqa"
        stages.append(
            SweepStage(
                name="vanilla_ot_full",
                category="vanilla_ot",
                description="Full-vector vanilla OT sweep across selected layers and token positions.",
                stage_timestamp=stage_timestamp,
                command=_build_cloud_compare_command(
                    script_name="mcqa_run_cloud.py",
                    method="ot",
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                ),
                expected_outputs=(
                    str(run_dir / "mcqa_run_results.json"),
                ),
            )
        )
    if "pca_ot" in normalized["stages"]:
        for basis_source_mode in normalized["pca_basis_source_modes"]:
            for site_menu in normalized["pca_site_menus"]:
                stage_timestamp = f"{timestamp_root}_pca_{_pca_config_slug(basis_source_mode=basis_source_mode, site_menu=site_menu)}"
                sweep_root = results_root / f"{stage_timestamp}_mcqa_ot_pca_focus"
                site_catalog_tag = _site_catalog_tag(
                    site_menu=str(site_menu),
                    num_bands=int(args.pca_num_bands),
                    band_scheme=str(args.pca_band_scheme),
                )
                expected_outputs = [
                    str(sweep_root / "layer_sweep_manifest.json"),
                ]
                for layer in normalized["layers"]:
                    expected_outputs.append(
                        str(
                            sweep_root
                            / f"layer_{int(layer):02d}"
                            / (
                                f"mcqa_layer-{int(layer)}_pos-{str(args.pca_token_position_id)}_pca-{site_catalog_tag}"
                                f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_ot_pca.json"
                            )
                        )
                    )
                stages.append(
                    SweepStage(
                        name=f"pca_ot_{_pca_config_slug(basis_source_mode=basis_source_mode, site_menu=site_menu)}",
                        category="pca_ot",
                        description=(
                            f"PCA OT sweep for basis_source_mode={basis_source_mode}, site_menu={site_menu}, "
                            f"token_position={args.pca_token_position_id}."
                        ),
                        stage_timestamp=stage_timestamp,
                        command=_build_pca_command(
                            args=args,
                            normalized=normalized,
                            stage_timestamp=stage_timestamp,
                            basis_source_mode=str(basis_source_mode),
                            site_menu=str(site_menu),
                            guided_das=False,
                        ),
                        expected_outputs=tuple(expected_outputs),
                    )
                )
    if "pca_guided_das" in normalized["stages"]:
        for basis_source_mode, site_menu in normalized["guided_pca_configs"]:
            stage_timestamp = f"{timestamp_root}_pca_{_pca_config_slug(basis_source_mode=basis_source_mode, site_menu=site_menu)}"
            sweep_root = results_root / f"{stage_timestamp}_mcqa_ot_pca_focus"
            site_catalog_tag = _site_catalog_tag(
                site_menu=str(site_menu),
                num_bands=int(args.pca_num_bands),
                band_scheme=str(args.pca_band_scheme),
            )
            expected_outputs = []
            for layer in normalized["layers"]:
                for target_var in DEFAULT_TARGET_VARS:
                    expected_outputs.append(
                        str(
                            sweep_root
                            / f"layer_{int(layer):02d}"
                            / (
                                f"mcqa_layer-{int(layer)}_pos-{str(args.pca_token_position_id)}_pca-{site_catalog_tag}"
                                f"_basis-{str(basis_source_mode)}_sig-{str(args.signature_mode)}_{str(target_var)}_das_guided.json"
                            )
                        )
                    )
            stages.append(
                SweepStage(
                    name=f"pca_guided_das_{_pca_config_slug(basis_source_mode=basis_source_mode, site_menu=site_menu)}",
                    category="pca_guided_das",
                    description=(
                        f"OT-guided DAS sweep for basis_source_mode={basis_source_mode}, site_menu={site_menu}, "
                        f"masks={list(normalized['guided_mask_names'])}."
                    ),
                    stage_timestamp=stage_timestamp,
                    command=_build_pca_command(
                        args=args,
                        normalized=normalized,
                        stage_timestamp=stage_timestamp,
                        basis_source_mode=str(basis_source_mode),
                        site_menu=str(site_menu),
                        guided_das=True,
                    ),
                    expected_outputs=tuple(expected_outputs),
                )
            )
    if "regular_das" in normalized["stages"]:
        stage_timestamp = f"{timestamp_root}_regular_das"
        run_dir = results_root / f"{stage_timestamp}_mcqa"
        stages.append(
            SweepStage(
                name="regular_das_full",
                category="regular_das",
                description="Full-vector regular DAS baseline across selected layers and token positions.",
                stage_timestamp=stage_timestamp,
                command=_build_cloud_compare_command(
                    script_name="mcqa_run_cloud.py",
                    method="das",
                    args=args,
                    normalized=normalized,
                    stage_timestamp=stage_timestamp,
                ),
                expected_outputs=(
                    str(run_dir / "mcqa_run_results.json"),
                ),
            )
        )
    return stages


def _load_status(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_status(
    manifest_path: Path,
    *,
    repo_root: Path,
    args: argparse.Namespace,
    normalized: dict[str, object],
    plan: list[SweepStage],
    stage_statuses: dict[str, dict[str, object]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "mcqa_delta_broad_sweep",
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
            "layers": [int(layer) for layer in normalized["layers"]],
            "signature_mode": str(args.signature_mode),
            "stages": list(normalized["stages"]),
            "full_token_position_ids": list(normalized["full_token_position_ids"]),
            "pca_token_position_id": str(args.pca_token_position_id),
            "pca_site_menus": list(normalized["pca_site_menus"]),
            "pca_basis_source_modes": list(normalized["pca_basis_source_modes"]),
            "guided_pca_configs": [
                {"basis_source_mode": str(mode), "site_menu": str(menu)}
                for mode, menu in normalized["guided_pca_configs"]
            ],
            "guided_mask_names": list(normalized["guided_mask_names"]),
        },
        "plan": [asdict(stage) for stage in plan],
        "stage_statuses": stage_statuses,
        "updated_at": datetime.now().isoformat(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_stage(*, stage: SweepStage, repo_root: Path) -> None:
    subprocess.run(stage.command, cwd=repo_root, check=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    normalized = _normalize_args(args)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and not bool(args.prompt_hf_login):
        raise ValueError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required unless --prompt-hf-login is set.")

    repo_root = Path(__file__).resolve().parent
    results_root = Path(args.results_root)
    sweep_root = results_root / f"{str(normalized['results_timestamp'])}_mcqa_broad_sweep"
    manifest_path = sweep_root / "broad_sweep_manifest.json"
    plan = build_broad_sweep_plan(repo_root=repo_root, args=args, normalized=normalized)

    existing_status = _load_status(manifest_path).get("stage_statuses", {})
    stage_statuses: dict[str, dict[str, object]] = {
        str(name): dict(payload) for name, payload in existing_status.items() if isinstance(payload, dict)
    }
    _write_status(
        manifest_path,
        repo_root=repo_root,
        args=args,
        normalized=normalized,
        plan=plan,
        stage_statuses=stage_statuses,
    )

    for stage in plan:
        expected_paths = [Path(path) for path in stage.expected_outputs]
        if all(_stage_output_is_valid(path) for path in expected_paths):
            stage_statuses[stage.name] = {
                "state": "skipped_existing",
                "stage_timestamp": stage.stage_timestamp,
                "expected_outputs": [str(path) for path in expected_paths],
                "completed_at": datetime.now().isoformat(),
            }
            _write_status(
                manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                plan=plan,
                stage_statuses=stage_statuses,
            )
            continue
        stage_statuses[stage.name] = {
            "state": "running",
            "stage_timestamp": stage.stage_timestamp,
            "expected_outputs": [str(path) for path in expected_paths],
            "started_at": datetime.now().isoformat(),
        }
        _write_status(
            manifest_path,
            repo_root=repo_root,
            args=args,
            normalized=normalized,
            plan=plan,
            stage_statuses=stage_statuses,
        )
        _run_stage(stage=stage, repo_root=repo_root)
        missing_outputs = [str(path) for path in expected_paths if not _stage_output_is_valid(path)]
        if missing_outputs:
            stage_statuses[stage.name] = {
                "state": "failed_validation",
                "stage_timestamp": stage.stage_timestamp,
                "expected_outputs": [str(path) for path in expected_paths],
                "missing_outputs": missing_outputs,
                "completed_at": datetime.now().isoformat(),
            }
            _write_status(
                manifest_path,
                repo_root=repo_root,
                args=args,
                normalized=normalized,
                plan=plan,
                stage_statuses=stage_statuses,
            )
            raise RuntimeError(f"Stage {stage.name} finished but expected outputs were missing or invalid: {missing_outputs}")
        stage_statuses[stage.name] = {
            "state": "completed",
            "stage_timestamp": stage.stage_timestamp,
            "expected_outputs": [str(path) for path in expected_paths],
            "completed_at": datetime.now().isoformat(),
        }
        _write_status(
            manifest_path,
            repo_root=repo_root,
            args=args,
            normalized=normalized,
            plan=plan,
            stage_statuses=stage_statuses,
        )

    summary_path = sweep_root / "broad_sweep_summary.txt"
    lines = [
        "MCQA Delta Broad Sweep",
        f"results_timestamp: {normalized['results_timestamp']}",
        f"results_root: {results_root}",
        "",
    ]
    for stage in plan:
        status = stage_statuses.get(stage.name, {})
        lines.append(f"[{stage.category}] {stage.name}: {status.get('state', 'unknown')}")
        for path in stage.expected_outputs:
            lines.append(f"  {path}")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote broad sweep manifest to {manifest_path}")
    print(f"Wrote broad sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
