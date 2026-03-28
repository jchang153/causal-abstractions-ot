import os
from datetime import datetime
from pathlib import Path

from equality_experiment.backbone import EqualityTrainConfig, load_backbone, train_backbone
from equality_experiment.das import DASConfig, run_das_pipeline
from equality_experiment.ot import OTConfig, run_alignment_pipeline
from equality_experiment.pair_bank import build_pair_bank
from equality_experiment.runtime import resolve_device, write_json
from equality_experiment.scm import load_equality_problem


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality_targeted_eval"
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
TARGETED_PAIR_POLICY = "mixed"
TARGETED_MIXED_POSITIVE_FRACTION = 1.0
TARGETED_PAIR_POOL_SIZE = 2048

BATCH_SIZE = 128
RESOLUTION = 1
SIGNATURE_MODE = "prob_delta"

OT_EPSILONS = (0.005, 0.01, 0.03)
OT_TAUS = (0.1, 0.25, 0.5, 1.0)
UOT_REG_MS = (0.1, 0.25, 0.5, 1.0, 2.0)
OT_TOP_K_VALUES = tuple(range(1, 21))
OT_LAMBDAS = tuple(round(x * 0.1, 1) for x in range(1, 81))

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 4, 8, 12, 16)


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


def build_targeted_banks(problem):
    train_bank = build_pair_bank(
        problem,
        TRAIN_PAIR_SIZE,
        SEED + 201,
        "train",
        target_vars=tuple(TARGET_VARS),
        pair_policy=TRAIN_PAIR_POLICY,
        pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        pair_pool_size=TRAIN_PAIR_POOL_SIZE,
    )
    calibration_banks = {}
    test_banks = {}
    for idx, variable in enumerate(TARGET_VARS):
        calibration_banks[variable] = build_pair_bank(
            problem,
            CALIBRATION_PAIR_SIZE,
            SEED + 301 + idx,
            "calibration",
            target_vars=tuple(TARGET_VARS),
            pair_policy=TARGETED_PAIR_POLICY,
            pair_policy_target=variable,
            mixed_positive_fraction=TARGETED_MIXED_POSITIVE_FRACTION,
            pair_pool_size=TARGETED_PAIR_POOL_SIZE,
        )
        test_banks[variable] = build_pair_bank(
            problem,
            TEST_PAIR_SIZE,
            SEED + 401 + idx,
            "test",
            target_vars=tuple(TARGET_VARS),
            pair_policy=TARGETED_PAIR_POLICY,
            pair_policy_target=variable,
            mixed_positive_fraction=TARGETED_MIXED_POSITIVE_FRACTION,
            pair_pool_size=TARGETED_PAIR_POOL_SIZE,
        )
    return train_bank, calibration_banks, test_banks


def record_from_payload(method, payload, epsilon=None, tau=None, reg_m=None):
    by_var = {record["variable"]: float(record["exact_acc"]) for record in payload["results"]}
    return {
        "method": method,
        "epsilon": epsilon,
        "tau": tau,
        "uot_reg_m": reg_m,
        "average_exact_acc": float(sum(by_var.values()) / len(by_var)),
        "per_variable_exact_acc": by_var,
        "selected_hyperparameters": payload.get("selected_hyperparameters"),
        "transport_meta": payload.get("transport_meta"),
        "runtime_seconds": float(payload.get("runtime_seconds", 0.0)),
        "results": payload["results"],
    }


def main():
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

    train_bank, calibration_banks, test_banks = build_targeted_banks(problem)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    das_payload = run_das_pipeline(
        model=model,
        train_bank=train_bank,
        calibration_bank=calibration_banks,
        holdout_bank=test_banks,
        device=device,
        config=DASConfig(
            batch_size=BATCH_SIZE,
            max_epochs=DAS_MAX_EPOCHS,
            min_epochs=DAS_MIN_EPOCHS,
            plateau_patience=DAS_PLATEAU_PATIENCE,
            plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
            learning_rate=DAS_LEARNING_RATE,
            subspace_dims=DAS_SUBSPACE_DIMS,
            search_layers=None,
            target_vars=tuple(TARGET_VARS),
        ),
    )
    das_record = record_from_payload("das", das_payload)

    transport_records = []
    best_by_method = {}
    total = len(OT_EPSILONS) * len(OT_TAUS) * (1 + len(UOT_REG_MS))
    sweep_index = 0
    for epsilon in OT_EPSILONS:
        for tau in OT_TAUS:
            for method, reg_ms in (("ot", (UOT_REG_MS[0],)), ("uot", UOT_REG_MS)):
                for reg_m in reg_ms:
                    sweep_index += 1
                    print(f"[{sweep_index}/{total}] method={method} eps={epsilon} tau={tau} reg_m={reg_m}", flush=True)
                    payload = run_alignment_pipeline(
                        model=model,
                        fit_bank=train_bank,
                        calibration_bank=calibration_banks,
                        holdout_bank=test_banks,
                        device=device,
                        config=OTConfig(
                            method=method,
                            batch_size=BATCH_SIZE,
                            resolution=RESOLUTION,
                            epsilon=epsilon,
                            tau=tau,
                            uot_reg_m=reg_m,
                            target_vars=tuple(TARGET_VARS),
                            top_k_values=OT_TOP_K_VALUES,
                            lambda_values=OT_LAMBDAS,
                            signature_mode=SIGNATURE_MODE,
                        ),
                    )
                    rec = record_from_payload(method, payload, epsilon=epsilon, tau=tau, reg_m=reg_m)
                    transport_records.append(rec)
                    incumbent = best_by_method.get(method)
                    if incumbent is None or rec["average_exact_acc"] > incumbent["average_exact_acc"]:
                        best_by_method[method] = rec

    payload = {
        "seed": SEED,
        "device": str(device),
        "target_vars": list(TARGET_VARS),
        "banks": {
            "train": train_bank.metadata(),
            "calibration": {k: v.metadata() for k, v in calibration_banks.items()},
            "test": {k: v.metadata() for k, v in test_banks.items()},
        },
        "das": das_record,
        "transport_sweep": transport_records,
        "best_by_method": best_by_method,
    }
    write_json(RUN_DIR / "equality_targeted_eval_results.json", payload)
    print("done", RUN_DIR, flush=True)


if __name__ == "__main__":
    main()
