from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_STAGES = (
    "stage_a_plot_layer",
    "stage_b_plot_native_support",
    "stage_b_plot_pca_support",
    "stage_c_plot_das_layer",
    "stage_c_plot_das_native_support",
)
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


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv_strings(value))


def _format_shell(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run(command: list[str], *, cwd: Path) -> None:
    print("")
    print(f"[overnight] running: {_format_shell(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full 5-seed MCQA overnight sweep: hierarchical PLOT/PLOT-DAS plus Full DAS."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--train-pool-size", type=int, default=200)
    parser.add_argument("--calibration-pool-size", type=int, default=100)
    parser.add_argument("--test-pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--signatures-dir", default="signatures")
    parser.add_argument("--signature-mode", default="family_label_delta_norm")
    parser.add_argument("--target-vars", default="answer_pointer,answer_token")
    parser.add_argument("--stage-a-token-position-ids", default="last_token")
    parser.add_argument("--ot-epsilons", default="0.5,1,2")
    parser.add_argument("--stage-a-uot-beta-neurals", default="0.03,0.3,1")
    parser.add_argument("--stage-a-row-top-k", type=int, default=6)
    parser.add_argument(
        "--stage-a-ot-lambdas",
        default="0.5,1,2",
        help="Deprecated compatibility flag. Stage A now uses a fixed full-strength intervention and ignores lambda sweeps.",
    )
    parser.add_argument("--ot-top-k-values", default="1,2,4")
    parser.add_argument(
        "--ot-lambdas",
        default="0.5,1,2,4",
        help="Downstream Stage B OT lambdas.",
    )
    parser.add_argument("--calibration-metric", default="family_weighted_macro_exact_acc")
    parser.add_argument("--calibration-family-weights", default="1,1.5,2")
    parser.add_argument("--stage-b-top-layers-per-var", type=int, default=3)
    parser.add_argument("--stage-b-neighbor-radius", type=int, default=0)
    parser.add_argument("--stage-b-max-layers-per-var", type=int, default=3)
    parser.add_argument("--native-resolutions", default="32,64,128,256")
    parser.add_argument("--pca-site-menus", default="partition")
    parser.add_argument("--pca-basis-source-modes", default="all_variants")
    parser.add_argument("--pca-num-bands-values", default="8,16")
    parser.add_argument("--pca-band-scheme", default="equal")
    parser.add_argument("--guided-mask-names", default="Selected")
    parser.add_argument("--guided-max-epochs", type=int, default=100)
    parser.add_argument("--guided-min-epochs", type=int, default=5)
    parser.add_argument("--guided-restarts", type=int, default=2)
    parser.add_argument(
        "--regular-das-subspace-dims",
        default="32,64,96,128,256,512,768,1024,1536,2048,2304",
    )
    parser.add_argument("--full-das-max-epochs", type=int, default=100)
    parser.add_argument("--full-das-min-epochs", type=int, default=5)
    parser.add_argument("--full-das-restarts", type=int, default=2)
    parser.add_argument("--prompt-hf-login", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    run_prefix = args.run_prefix or datetime.now().strftime("%Y%m%d_%H%M%S_mcqa_overnight")
    seeds = _parse_csv_ints(args.seeds) or DEFAULT_SEEDS
    regular_das_subspace_dims = _parse_csv_ints(args.regular_das_subspace_dims) or DEFAULT_REGULAR_DAS_SUBSPACE_DIMS

    manifest_path = results_root / f"{run_prefix}_mcqa_overnight_manifest.json"
    manifest: dict[str, object] = {
        "run_prefix": str(run_prefix),
        "results_root": str(results_root),
        "device": str(args.device),
        "model_name": str(args.model_name),
        "dataset_path": str(args.dataset_path),
        "dataset_config": args.dataset_config,
        "dataset_size": int(args.dataset_size),
        "train_pool_size": int(args.train_pool_size),
        "calibration_pool_size": int(args.calibration_pool_size),
        "test_pool_size": int(args.test_pool_size),
        "batch_size": int(args.batch_size),
        "stage_a_grid": {
            "ot_epsilons": str(args.ot_epsilons),
            "uot_beta_neurals": str(args.stage_a_uot_beta_neurals),
            "row_top_k": int(args.stage_a_row_top_k),
        },
        "downstream_ot_grid": {
            "ot_top_k_values": str(args.ot_top_k_values),
            "ot_lambdas": str(args.ot_lambdas),
        },
        "calibration_family_weights": str(args.calibration_family_weights),
        "stage_b_top_layers_per_var": int(args.stage_b_top_layers_per_var),
        "stage_b_neighbor_radius": int(args.stage_b_neighbor_radius),
        "stage_b_max_layers_per_var": int(args.stage_b_max_layers_per_var),
        "native_resolutions": str(args.native_resolutions),
        "seeds": [int(seed) for seed in seeds],
        "seed_runs": [],
    }

    for seed in seeds:
        seed_prefix = f"{run_prefix}_seed{int(seed)}"
        hierarchy_timestamp = str(seed_prefix)
        full_das_timestamp = f"{seed_prefix}_full_das"
        sweep_root = results_root / f"{hierarchy_timestamp}_mcqa_hierarchical_sweep"
        full_das_root = results_root / f"{full_das_timestamp}_mcqa"
        full_das_output_path = full_das_root / "mcqa_run_results.json"

        hierarchy_command = [
            sys.executable,
            "mcqa_delta_hierarchical_sweep.py",
            "--device",
            str(args.device),
            "--model-name",
            str(args.model_name),
            "--dataset-path",
            str(args.dataset_path),
            "--dataset-size",
            str(int(args.dataset_size)),
            "--split-seed",
            str(int(seed)),
            "--train-pool-size",
            str(int(args.train_pool_size)),
            "--calibration-pool-size",
            str(int(args.calibration_pool_size)),
            "--test-pool-size",
            str(int(args.test_pool_size)),
            "--batch-size",
            str(int(args.batch_size)),
            "--results-root",
            str(results_root),
            "--results-timestamp",
            str(hierarchy_timestamp),
            "--signatures-dir",
            str(args.signatures_dir),
            "--signature-mode",
            str(args.signature_mode),
            "--stages",
            ",".join(DEFAULT_STAGES),
            "--stage-a-token-position-ids",
            str(args.stage_a_token_position_ids),
            "--target-vars",
            str(args.target_vars),
            "--ot-epsilons",
            str(args.ot_epsilons),
            "--stage-a-uot-beta-neurals",
            str(args.stage_a_uot_beta_neurals),
            "--stage-a-row-top-k",
            str(int(args.stage_a_row_top_k)),
            "--ot-top-k-values",
            str(args.ot_top_k_values),
            "--ot-lambdas",
            str(args.ot_lambdas),
            "--calibration-metric",
            str(args.calibration_metric),
            "--calibration-family-weights",
            str(args.calibration_family_weights),
            "--stage-b-top-layers-per-var",
            str(int(args.stage_b_top_layers_per_var)),
            "--stage-b-neighbor-radius",
            str(int(args.stage_b_neighbor_radius)),
            "--stage-b-max-layers-per-var",
            str(int(args.stage_b_max_layers_per_var)),
            "--native-resolutions",
            str(args.native_resolutions),
            "--pca-site-menus",
            str(args.pca_site_menus),
            "--pca-basis-source-modes",
            str(args.pca_basis_source_modes),
            "--pca-num-bands-values",
            str(args.pca_num_bands_values),
            "--pca-band-scheme",
            str(args.pca_band_scheme),
            "--guided-mask-names",
            str(args.guided_mask_names),
            "--guided-max-epochs",
            str(int(args.guided_max_epochs)),
            "--guided-min-epochs",
            str(int(args.guided_min_epochs)),
            "--guided-restarts",
            str(int(args.guided_restarts)),
            "--regular-das-subspace-dims",
            ",".join(str(dim) for dim in regular_das_subspace_dims),
        ]
        if args.dataset_config:
            hierarchy_command.extend(["--dataset-config", str(args.dataset_config)])
        if args.prompt_hf_login:
            hierarchy_command.append("--prompt-hf-login")

        full_das_command = [
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
            str(int(seed)),
            "--train-pool-size",
            str(int(args.train_pool_size)),
            "--calibration-pool-size",
            str(int(args.calibration_pool_size)),
            "--test-pool-size",
            str(int(args.test_pool_size)),
            "--methods",
            "das",
            "--target-vars",
            str(args.target_vars),
            "--counterfactual-names",
            "answerPosition,randomLetter,answerPosition_randomLetter",
            "--layers",
            "auto",
            "--token-position-ids",
            "last_token",
            "--batch-size",
            str(int(args.batch_size)),
            "--resolutions",
            "full",
            "--signature-modes",
            str(args.signature_mode),
            "--ot-top-k-values",
            str(args.ot_top_k_values),
            "--ot-lambdas",
            str(args.ot_lambdas),
            "--calibration-metric",
            str(args.calibration_metric),
            "--calibration-family-weights",
            str(args.calibration_family_weights),
            "--das-max-epochs",
            str(int(args.full_das_max_epochs)),
            "--das-min-epochs",
            str(int(args.full_das_min_epochs)),
            "--das-restarts",
            str(max(1, int(args.full_das_restarts))),
            "--das-subspace-dims",
            ",".join(str(dim) for dim in regular_das_subspace_dims),
            "--results-root",
            str(results_root),
            "--results-timestamp",
            str(full_das_timestamp),
            "--signatures-dir",
            str(args.signatures_dir),
        ]
        if args.dataset_config:
            full_das_command.extend(["--dataset-config", str(args.dataset_config)])
        if args.prompt_hf_login:
            full_das_command.append("--prompt-hf-login")

        paper_runtime_command = [
            sys.executable,
            "mcqa_paper_runtime.py",
            str(sweep_root),
            "--full-das-output",
            str(full_das_output_path),
        ]

        _run(hierarchy_command, cwd=repo_root)
        _run(full_das_command, cwd=repo_root)
        _run(paper_runtime_command, cwd=repo_root)

        manifest["seed_runs"].append(
            {
                "seed": int(seed),
                "hierarchy_timestamp": str(hierarchy_timestamp),
                "hierarchy_sweep_root": str(sweep_root),
                "full_das_timestamp": str(full_das_timestamp),
                "full_das_output_path": str(full_das_output_path),
                "paper_runtime_summary_json": str(sweep_root / "paper_runtime_summary.json"),
                "paper_runtime_summary_txt": str(sweep_root / "paper_runtime_summary.txt"),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("")
    print(f"[overnight] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
