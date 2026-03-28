import os
from datetime import datetime
from pathlib import Path

import numpy as np

from equality_experiment.backbone import EqualityTrainConfig, load_backbone, train_backbone
from equality_experiment.compare_runner import CompareExperimentConfig, run_comparison_with_banks
from equality_experiment.reporting import write_text_report
from equality_experiment.runtime import resolve_device, write_json
from equality_experiment.scm import load_equality_problem
from equality_experiment.pair_bank import build_pair_bank


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality_lambda_extension"
CHECKPOINT_PATH = Path(f"models/equality_mlp_seed{SEED}.pt")
RETRAIN_BACKBONE = False

TARGET_VARS = ("WX", "YZ")
NUM_ENTITIES = 100
EMBEDDING_DIM = 4

FACTUAL_TRAIN_SIZE = 1048576
FACTUAL_VALIDATION_SIZE = 10000
HIDDEN_DIMS = (16, 16, 16)
LEARNING_RATE = 1e-3
EPOCHS = 3
TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024

TRAIN_PAIR_SIZE = 1000
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 1000
TRAIN_PAIR_POLICY = "mixed"
TRAIN_PAIR_POLICY_TARGET = "any"
TRAIN_MIXED_POSITIVE_FRACTION = 0.5
TRAIN_PAIR_POOL_SIZE = 2048
CALIBRATION_PAIR_POLICY = "mixed"
CALIBRATION_PAIR_POLICY_TARGET = "WX"
CALIBRATION_MIXED_POSITIVE_FRACTION = 1.0
CALIBRATION_PAIR_POOL_SIZE = 2048
TEST_PAIR_POLICY = "mixed"
TEST_PAIR_POLICY_TARGET = "WX"
TEST_MIXED_POSITIVE_FRACTION = 1.0
TEST_PAIR_POOL_SIZE = 2048

BATCH_SIZE = 128
RESOLUTION = 1
OT_EPSILONS = (0.005, 0.01, 0.03)
OT_TAUS = (0.1, 0.25, 0.5, 1.0)
UOT_REG_MS = (0.1, 0.25, 0.5, 1.0, 2.0)
OT_TOP_K_VALUES = tuple(range(1, 21))
OT_LAMBDAS = tuple(np.arange(0.1, 8.0 + 1e-9, 0.1))
SIGNATURE_MODE = "prob_delta"


def build_train_config() -> EqualityTrainConfig:
    return EqualityTrainConfig(
        seed=SEED,
        n_train=FACTUAL_TRAIN_SIZE,
        n_validation=FACTUAL_VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
    )


def build_compare_config(run_dir: Path, methods: tuple[str, ...], ot_epsilon: float, ot_tau: float, uot_reg_m: float) -> CompareExperimentConfig:
    return CompareExperimentConfig(
        seed=SEED,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=run_dir / "equality_run_results.json",
        summary_path=run_dir / "equality_run_summary.txt",
        methods=tuple(methods),
        factual_validation_size=FACTUAL_VALIDATION_SIZE,
        train_pair_size=TRAIN_PAIR_SIZE,
        calibration_pair_size=CALIBRATION_PAIR_SIZE,
        test_pair_size=TEST_PAIR_SIZE,
        target_vars=tuple(TARGET_VARS),
        train_pair_policy=TRAIN_PAIR_POLICY,
        train_pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        train_mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        train_pair_pool_size=TRAIN_PAIR_POOL_SIZE,
        calibration_pair_policy=CALIBRATION_PAIR_POLICY,
        calibration_pair_policy_target=CALIBRATION_PAIR_POLICY_TARGET,
        calibration_mixed_positive_fraction=CALIBRATION_MIXED_POSITIVE_FRACTION,
        calibration_pair_pool_size=CALIBRATION_PAIR_POOL_SIZE,
        test_pair_policy=TEST_PAIR_POLICY,
        test_pair_policy_target=TEST_PAIR_POLICY_TARGET,
        test_mixed_positive_fraction=TEST_MIXED_POSITIVE_FRACTION,
        test_pair_pool_size=TEST_PAIR_POOL_SIZE,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        ot_epsilon=float(ot_epsilon),
        ot_tau=float(ot_tau),
        uot_reg_m=float(uot_reg_m),
        signature_mode=SIGNATURE_MODE,
        ot_top_k_values=OT_TOP_K_VALUES,
        ot_lambdas=tuple(OT_LAMBDAS),
    )


def sweep_record(method: str, epsilon: float, tau: float, uot_reg_m: float, comparison: dict[str, object], run_dir: Path) -> dict[str, object]:
    results = comparison["results"]
    by_var = {record["variable"]: float(record["exact_acc"]) for record in results}
    return {
        "method": method,
        "epsilon": float(epsilon),
        "tau": float(tau),
        "uot_reg_m": float(uot_reg_m),
        "signature_mode": SIGNATURE_MODE,
        "average_exact_acc": float(sum(by_var.values()) / len(by_var)),
        "per_variable_exact_acc": by_var,
        "selected_hyperparameters": comparison["method_selections"][method]["selected_hyperparameters"],
        "transport_meta": comparison["method_selections"][method]["transport_meta"],
        "runtime_seconds": float(comparison["method_runtime_seconds"][method]),
        "run_dir": str(run_dir),
    }


def main() -> None:
    problem = load_equality_problem(
        run_checks=True,
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
        seed=SEED,
    )
    device = resolve_device(DEVICE)
    train_config = build_train_config()

    if RETRAIN_BACKBONE or not CHECKPOINT_PATH.exists():
        model, _, backbone_meta = train_backbone(
            problem=problem,
            train_config=train_config,
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
        )
    else:
        model, _, backbone_meta = load_backbone(
            problem=problem,
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            train_config=train_config,
        )

    train_bank = build_pair_bank(
        problem, TRAIN_PAIR_SIZE, SEED + 201, "train",
        target_vars=tuple(TARGET_VARS),
        pair_policy=TRAIN_PAIR_POLICY,
        pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        pair_pool_size=TRAIN_PAIR_POOL_SIZE,
    )
    calibration_bank = build_pair_bank(
        problem, CALIBRATION_PAIR_SIZE, SEED + 301, "calibration",
        target_vars=tuple(TARGET_VARS),
        pair_policy=CALIBRATION_PAIR_POLICY,
        pair_policy_target=CALIBRATION_PAIR_POLICY_TARGET,
        mixed_positive_fraction=CALIBRATION_MIXED_POSITIVE_FRACTION,
        pair_pool_size=CALIBRATION_PAIR_POOL_SIZE,
    )
    test_bank = build_pair_bank(
        problem, TEST_PAIR_SIZE, SEED + 401, "test",
        target_vars=tuple(TARGET_VARS),
        pair_policy=TEST_PAIR_POLICY,
        pair_policy_target=TEST_PAIR_POLICY_TARGET,
        mixed_positive_fraction=TEST_MIXED_POSITIVE_FRACTION,
        pair_pool_size=TEST_PAIR_POOL_SIZE,
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    transport_records = []
    best_by_method = {}
    sweep_index = 0
    sweep_methods = (
        ("ot", (UOT_REG_MS[0],)),
        ("uot", UOT_REG_MS),
    )
    total = len(OT_EPSILONS) * len(OT_TAUS) * sum(len(reg_ms) for _, reg_ms in sweep_methods)
    for epsilon in OT_EPSILONS:
        for tau in OT_TAUS:
            for method, reg_ms in sweep_methods:
                for uot_reg_m in reg_ms:
                    sweep_index += 1
                    tag = f"{method}_eps{str(epsilon).replace('.', 'p')}_tau{str(tau).replace('.', 'p')}_m{str(uot_reg_m).replace('.', 'p')}"
                    run_dir = RUN_DIR / tag
                    print(f"[{sweep_index}/{total}] method={method} epsilon={epsilon} tau={tau} uot_reg_m={uot_reg_m}", flush=True)
                    comparison = run_comparison_with_banks(
                        model=model,
                        backbone_meta=backbone_meta,
                        device=device,
                        config=build_compare_config(run_dir, (method,), epsilon, tau, uot_reg_m),
                        train_bank=train_bank,
                        calibration_bank=calibration_bank,
                        test_bank=test_bank,
                    )
                    record = sweep_record(method, epsilon, tau, uot_reg_m, comparison, run_dir)
                    transport_records.append(record)
                    incumbent = best_by_method.get(method)
                    if incumbent is None or float(record["average_exact_acc"]) > float(incumbent["average_exact_acc"]):
                        best_by_method[method] = record

    payload = {
        "seed": SEED,
        "device": str(device),
        "target_vars": list(TARGET_VARS),
        "transport_sweep": transport_records,
        "best_by_method": best_by_method,
    }
    write_json(RUN_DIR / "equality_lambda_extension_results.json", payload)

    summary_lines = [
        "Equality Lambda Extension Summary",
        f"run_dir: {RUN_DIR}",
        f"seed: {SEED}",
        f"target_vars: {', '.join(TARGET_VARS)}",
        f"signature_mode: {SIGNATURE_MODE}",
        "",
        "Best transport results:",
    ]
    for method in ("ot", "uot"):
        rec = best_by_method.get(method)
        if rec is None:
            continue
        summary_lines.append(
            f"{method.upper()}: avg={float(rec['average_exact_acc']):.4f} | "
            f"WX={float(rec['per_variable_exact_acc']['WX']):.4f} | "
            f"YZ={float(rec['per_variable_exact_acc']['YZ']):.4f} | "
            f"eps={float(rec['epsilon']):.4f} | tau={float(rec['tau']):.4f} | "
            f"reg_m={float(rec['uot_reg_m']):.4f}"
        )
    write_text_report(RUN_DIR / "equality_lambda_extension_summary.txt", "\n".join(summary_lines))


if __name__ == "__main__":
    main()
