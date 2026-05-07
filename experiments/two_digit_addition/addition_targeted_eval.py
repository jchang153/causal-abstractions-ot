import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from addition_experiment.backbone import load_backbone
from addition_experiment.das import DASConfig, run_das_search_for_variable
from addition_experiment.pair_bank import build_pair_bank, build_structured_pair_bank
from addition_experiment.runtime import resolve_device, write_json
from addition_experiment.scm import load_addition_problem


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_addition_targeted_eval"
CHECKPOINT_PATH = Path("models/addition_mlp_seed42.pt")
TARGET_VARS = ("C1", "C2")


@dataclass(frozen=True)
class TargetedAdditionConfig:
    train_pair_size: int = 1000
    calibration_pair_size: int = 1000
    test_pair_size: int = 2000
    train_pair_policy: str = "mixed"
    train_pair_policy_target: str = "any"
    train_mixed_positive_fraction: float = 1.0
    train_pair_pool_size: int | None = 1024
    train_source_strategy: str = "structured"
    evaluation_pair_policy: str = "mixed"
    evaluation_mixed_positive_fraction: float = 1.0
    evaluation_pair_pool_size: int | None = 1024
    evaluation_source_strategy: str = "structured"
    batch_size: int = 128
    das_max_epochs: int = 1000
    das_min_epochs: int = 5
    das_plateau_patience: int = 2
    das_plateau_rel_delta: float = 5e-3
    das_learning_rate: float = 1e-3
    das_subspace_dims: tuple[int, ...] = (1, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192)
    das_layers: tuple[int, ...] | None = None


def _build_bank(
    problem,
    *,
    size: int,
    seed: int,
    split: str,
    pair_policy: str,
    pair_policy_target: str,
    mixed_positive_fraction: float,
    pair_pool_size: int | None,
    source_strategy: str,
):
    builder = build_pair_bank if source_strategy == "random" else build_structured_pair_bank
    if source_strategy not in {"random", "structured"}:
        raise ValueError(f"Unsupported source_strategy={source_strategy}")
    return builder(
        problem,
        size,
        seed,
        split,
        target_vars=tuple(TARGET_VARS),
        pair_policy=pair_policy,
        pair_policy_target=pair_policy_target,
        mixed_positive_fraction=mixed_positive_fraction,
        pair_pool_size=pair_pool_size,
    )


def _build_eval_bank(problem, *, size: int, seed: int, split: str, variable: str, config: TargetedAdditionConfig):
    return _build_bank(
        problem,
        size=size,
        seed=seed,
        split=split,
        pair_policy=config.evaluation_pair_policy,
        pair_policy_target=variable,
        mixed_positive_fraction=config.evaluation_mixed_positive_fraction,
        pair_pool_size=config.evaluation_pair_pool_size,
        source_strategy=config.evaluation_source_strategy,
    )


def main() -> None:
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    config = TargetedAdditionConfig()

    model, model_config, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    train_bank = _build_bank(
        problem,
        size=config.train_pair_size,
        seed=SEED + 201,
        split="train",
        pair_policy=config.train_pair_policy,
        pair_policy_target=config.train_pair_policy_target,
        mixed_positive_fraction=config.train_mixed_positive_fraction,
        pair_pool_size=config.train_pair_pool_size,
        source_strategy=config.train_source_strategy,
    )

    calibration_banks = {
        variable: _build_eval_bank(
            problem,
            size=config.calibration_pair_size,
            seed=SEED + 301 + index,
            split=f"calibration_{variable}",
            variable=variable,
            config=config,
        )
        for index, variable in enumerate(TARGET_VARS)
    }
    test_banks = {
        variable: _build_eval_bank(
            problem,
            size=config.test_pair_size,
            seed=SEED + 401 + index,
            split=f"test_{variable}",
            variable=variable,
            config=config,
        )
        for index, variable in enumerate(TARGET_VARS)
    }

    das_config = DASConfig(
        batch_size=config.batch_size,
        max_epochs=config.das_max_epochs,
        min_epochs=config.das_min_epochs,
        learning_rate=config.das_learning_rate,
        subspace_dims=config.das_subspace_dims,
        search_layers=config.das_layers,
        target_vars=tuple(TARGET_VARS),
        plateau_patience=config.das_plateau_patience,
        plateau_rel_delta=config.das_plateau_rel_delta,
        verbose=True,
        progress_interval=25,
    )

    variable_results = {}
    search_records = {}
    t0 = perf_counter()
    for variable in TARGET_VARS:
        result_record, all_records = run_das_search_for_variable(
            model=model,
            variable=variable,
            train_bank=train_bank,
            calibration_bank=calibration_banks[variable],
            holdout_bank=test_banks[variable],
            device=device,
            config=das_config,
        )
        variable_results[variable] = result_record
        search_records[variable] = all_records
    runtime_sec = perf_counter() - t0

    avg_exact = sum(float(variable_results[v]["exact_acc"]) for v in TARGET_VARS) / len(TARGET_VARS)
    avg_shared = sum(float(variable_results[v]["mean_shared_digits"]) for v in TARGET_VARS) / len(TARGET_VARS)

    payload = {
        "seed": SEED,
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_config": model_config.to_dict(),
        "backbone_meta": backbone_meta,
        "targeted_config": asdict(config),
        "train_bank": train_bank.metadata(),
        "calibration_banks": {k: v.metadata() for k, v in calibration_banks.items()},
        "test_banks": {k: v.metadata() for k, v in test_banks.items()},
        "das": {
            "per_variable": variable_results,
            "avg_exact_acc": avg_exact,
            "avg_mean_shared_digits": avg_shared,
            "runtime_sec": runtime_sec,
        },
        "search_records": search_records,
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RUN_DIR / "addition_targeted_eval_results.json"
    write_json(json_path, payload)
    (RUN_DIR / "addition_targeted_eval_summary.txt").write_text(
        "\n".join(
            [
                "Addition Targeted DAS Eval",
                f"checkpoint: {CHECKPOINT_PATH}",
                f"device: {device}",
                f"avg_exact_acc: {avg_exact:.4f}",
                f"avg_mean_shared_digits: {avg_shared:.4f}",
                f"runtime_sec: {runtime_sec:.2f}",
                *[
                    (
                        f"{variable}: site={variable_results[variable]['site_label']} "
                        f"| calib_exact={float(variable_results[variable]['calibration_exact_acc']):.4f} "
                        f"| test_exact={float(variable_results[variable]['exact_acc']):.4f} "
                        f"| test_shared={float(variable_results[variable]['mean_shared_digits']):.4f}"
                    )
                    for variable in TARGET_VARS
                ],
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path.resolve()), "avg_exact_acc": avg_exact, "avg_mean_shared_digits": avg_shared}, indent=2))


if __name__ == "__main__":
    main()
