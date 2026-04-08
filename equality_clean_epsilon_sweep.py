import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from equality_calibration_strategy_sweep import (
    build_fixed_test_banks,
    build_fixed_train_bank,
    build_strategy_calibration_bank,
    ensure_backbone,
)
from equality_experiment.ot import OTConfig, run_alignment_pipeline
from equality_experiment.runtime import resolve_device, write_json
from equality_experiment.scm import load_equality_problem


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality_clean_epsilon_sweep"
OUTPUT_PATH = RUN_DIR / "equality_clean_epsilon_sweep.json"
SUMMARY_PATH = RUN_DIR / "equality_clean_epsilon_sweep.txt"

METHODS = ("ot", "uot")
TARGET_VARS = ("WX", "YZ")

EPSILON_VALUES = tuple(float(2.0**power) for power in range(-10, 8))
TAU = 1.0
UOT_REG_M = 1.0

BATCH_SIZE = 128
RESOLUTION = 1
SIGNATURE_MODE = "prob_delta"
TOP_K_VALUES = tuple(range(1, 21))
LAMBDAS = tuple(np.arange(0.1, 8.0 + 1e-9, 0.1))

CALIBRATION_STRATEGY = "shared_balanced_wx_yz_only"


def record_from_payload(method: str, epsilon: float, payload: dict[str, object], runtime_seconds: float) -> dict[str, object]:
    by_var = {record["variable"]: float(record["exact_acc"]) for record in payload["results"]}
    return {
        "method": method,
        "epsilon": float(epsilon),
        "average_exact_acc": float(sum(by_var.values()) / len(by_var)),
        "per_variable_exact_acc": by_var,
        "selected_hyperparameters": payload.get("selected_hyperparameters"),
        "transport_meta": payload.get("transport_meta"),
        "runtime_seconds": float(runtime_seconds),
        "results": payload["results"],
    }


def format_summary(payload: dict[str, object]) -> str:
    lines = [
        "HEQ Clean Epsilon Sweep",
        f"seed: {payload['seed']}",
        f"device: {payload['device']}",
        f"calibration_strategy: {payload['calibration_strategy']}",
        f"regularization_source: {payload['regularization_source']}",
        f"retry_multipliers: {payload['retry_multipliers']}",
        f"tau_fixed: {float(payload['tau_fixed']):.6f}",
        f"uot_reg_m_fixed: {float(payload['uot_reg_m_fixed']):.6f}",
        f"factual_validation_exact_acc: {float(payload['backbone']['factual_validation_metrics']['exact_acc']):.4f}",
        "",
    ]
    for method in METHODS:
        lines.append(f"{method.upper()} sweep:")
        method_records = [record for record in payload["sweep_records"] if record["method"] == method]
        for record in method_records:
            lines.append(
                f"  eps={float(record['epsilon']):.6f} "
                f"| avg={float(record['average_exact_acc']):.4f} "
                f"| WX={float(record['per_variable_exact_acc']['WX']):.4f} "
                f"| YZ={float(record['per_variable_exact_acc']['YZ']):.4f} "
                f"| reg_used={float(record['transport_meta']['regularization_used']):.6f}"
            )
        best = payload["best_by_method"][method]
        lines.extend(
            [
                "",
                f"  best {method.upper()}: eps={float(best['epsilon']):.6f} "
                f"| avg={float(best['average_exact_acc']):.4f} "
                f"| WX={float(best['per_variable_exact_acc']['WX']):.4f} "
                f"| YZ={float(best['per_variable_exact_acc']['YZ']):.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    device = resolve_device(DEVICE)
    problem = load_equality_problem(seed=SEED)
    model, _, backbone_meta = ensure_backbone(problem, device)

    train_bank = build_fixed_train_bank(problem)
    calibration_bank = build_strategy_calibration_bank(problem, CALIBRATION_STRATEGY)
    test_banks = build_fixed_test_banks(problem)

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    sweep_records = []
    best_by_method: dict[str, dict[str, object]] = {}
    total = len(METHODS) * len(EPSILON_VALUES)
    run_index = 0
    for method in METHODS:
        for epsilon in EPSILON_VALUES:
            run_index += 1
            print(
                f"[{run_index}/{total}] method={method} epsilon={float(epsilon):.6f} "
                f"| tau={float(TAU):.6f} | reg_m={float(UOT_REG_M):.6f}",
                flush=True,
            )
            start = perf_counter()
            payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_banks,
                device=device,
                config=OTConfig(
                    method=method,
                    batch_size=BATCH_SIZE,
                    resolution=RESOLUTION,
                    epsilon=float(epsilon),
                    tau=float(TAU),
                    uot_reg_m=float(UOT_REG_M),
                    target_vars=tuple(TARGET_VARS),
                    top_k_values=TOP_K_VALUES,
                    lambda_values=LAMBDAS,
                    signature_mode=SIGNATURE_MODE,
                    selection_verbose=True,
                    regularization_source="epsilon",
                    epsilon_retry_multipliers=(1.0,),
                ),
            )
            runtime_seconds = perf_counter() - start
            record = record_from_payload(method, epsilon, payload, runtime_seconds)
            sweep_records.append(record)
            incumbent = best_by_method.get(method)
            if incumbent is None or float(record["average_exact_acc"]) > float(incumbent["average_exact_acc"]):
                best_by_method[method] = record

    summary_payload = {
        "seed": SEED,
        "device": str(device),
        "calibration_strategy": CALIBRATION_STRATEGY,
        "regularization_source": "epsilon",
        "retry_multipliers": [1.0],
        "tau_fixed": float(TAU),
        "uot_reg_m_fixed": float(UOT_REG_M),
        "epsilon_values": list(EPSILON_VALUES),
        "banks": {
            "train": train_bank.metadata(),
            "calibration": calibration_bank.metadata() if not isinstance(calibration_bank, dict) else {
                variable: bank.metadata() for variable, bank in calibration_bank.items()
            },
            "test": {variable: bank.metadata() for variable, bank in test_banks.items()},
        },
        "backbone": backbone_meta,
        "sweep_records": sweep_records,
        "best_by_method": best_by_method,
    }
    write_json(OUTPUT_PATH, summary_payload)
    SUMMARY_PATH.write_text(format_summary(summary_payload), encoding="utf-8")
    print(f"Saved sweep JSON to {OUTPUT_PATH}")
    print(f"Saved sweep summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
