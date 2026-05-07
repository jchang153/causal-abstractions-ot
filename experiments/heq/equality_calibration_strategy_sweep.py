import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

try:
    from .equality_experiment.backbone import EqualityTrainConfig, load_backbone, train_backbone
    from .equality_experiment.das import DASConfig, run_das_pipeline
    from .equality_experiment.ot import OTConfig, run_alignment_pipeline
    from .equality_experiment.pair_bank import PairBank, build_pair_bank
    from .equality_experiment.runtime import resolve_device, write_json
    from .equality_experiment.scm import load_equality_problem
except ImportError:  # pragma: no cover - supports direct script execution.
    from equality_experiment.backbone import EqualityTrainConfig, load_backbone, train_backbone
    from equality_experiment.das import DASConfig, run_das_pipeline
    from equality_experiment.ot import OTConfig, run_alignment_pipeline
    from equality_experiment.pair_bank import PairBank, build_pair_bank
    from equality_experiment.runtime import resolve_device, write_json
    from equality_experiment.scm import load_equality_problem


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality_calibration_strategy_sweep"
OUTPUT_PATH = RUN_DIR / "equality_calibration_strategy_sweep.json"
SUMMARY_PATH = RUN_DIR / "equality_calibration_strategy_sweep.txt"
CHECKPOINT_PATH = Path("models/equality_mlp_seed42.pt")

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
PAIR_POOL_SIZE = 2048

BATCH_SIZE = 128
RESOLUTION = 1
SIGNATURE_MODE = "prob_delta"

OT_EPSILON = 0.005
OT_TAU = 0.1
UOT_TAU = 0.25
UOT_REG_M = 1.0
TOP_K_VALUES = tuple(range(1, 21))
LAMBDAS = tuple(np.arange(0.1, 8.0 + 1e-9, 0.1))

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 4, 8, 12, 16)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str


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
        verbose=True,
    )


def ensure_backbone(problem, device: torch.device):
    train_config = build_train_config()
    if CHECKPOINT_PATH.exists():
        try:
            return load_backbone(
                problem,
                checkpoint_path=CHECKPOINT_PATH,
                device=device,
                train_config=train_config,
            )
        except Exception:
            pass
    return train_backbone(
        problem,
        train_config=train_config,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )


def _pair_stats_from_bank(bank: PairBank) -> dict[str, object]:
    per_variable = {}
    for variable in bank.pair_policy_vars:
        changed = int(bank.changed_by_var[variable].to(torch.int64).sum().item())
        total = int(bank.size)
        per_variable[variable] = {
            "changed_count": changed,
            "unchanged_count": total - changed,
            "changed_rate": float(changed / total) if total else 0.0,
        }
    changed_any = int(bank.changed_any.to(torch.int64).sum().item())
    total = int(bank.size)
    return {
        "total_pairs": total,
        "changed_any_count": changed_any,
        "unchanged_any_count": total - changed_any,
        "changed_any_rate": float(changed_any / total) if total else 0.0,
        "per_variable": per_variable,
    }


def concat_pair_banks(
    banks: list[PairBank],
    *,
    split: str,
    seed: int,
    pair_policy: str,
    pair_policy_target: str,
    mixed_positive_fraction: float,
) -> PairBank:
    if not banks:
        raise ValueError("Expected at least one bank to concatenate")
    first = banks[0]
    for bank in banks[1:]:
        if bank.target_vars != first.target_vars:
            raise ValueError("Mismatched target_vars across banks")
        if bank.pair_policy_vars != first.pair_policy_vars:
            raise ValueError("Mismatched pair_policy_vars across banks")

    merged = PairBank(
        split=split,
        seed=seed,
        base_rows=torch.cat([bank.base_rows for bank in banks], dim=0),
        source_rows=torch.cat([bank.source_rows for bank in banks], dim=0),
        base_inputs=torch.cat([bank.base_inputs for bank in banks], dim=0),
        source_inputs=torch.cat([bank.source_inputs for bank in banks], dim=0),
        base_labels=torch.cat([bank.base_labels for bank in banks], dim=0),
        cf_labels_by_var={
            variable: torch.cat([bank.cf_labels_by_var[variable] for bank in banks], dim=0)
            for variable in first.target_vars
        },
        changed_by_var={
            variable: torch.cat([bank.changed_by_var[variable] for bank in banks], dim=0)
            for variable in first.pair_policy_vars
        },
        changed_any=torch.cat([bank.changed_any for bank in banks], dim=0),
        pair_policy=pair_policy,
        pair_policy_target=pair_policy_target,
        mixed_positive_fraction=float(mixed_positive_fraction),
        target_vars=first.target_vars,
        pair_policy_vars=first.pair_policy_vars,
        pair_pool_size=None,
        pair_stats={},
    )
    return PairBank(
        split=merged.split,
        seed=merged.seed,
        base_rows=merged.base_rows,
        source_rows=merged.source_rows,
        base_inputs=merged.base_inputs,
        source_inputs=merged.source_inputs,
        base_labels=merged.base_labels,
        cf_labels_by_var=merged.cf_labels_by_var,
        changed_by_var=merged.changed_by_var,
        changed_any=merged.changed_any,
        pair_policy=merged.pair_policy,
        pair_policy_target=merged.pair_policy_target,
        mixed_positive_fraction=merged.mixed_positive_fraction,
        target_vars=merged.target_vars,
        pair_policy_vars=merged.pair_policy_vars,
        pair_pool_size=merged.pair_pool_size,
        pair_stats=_pair_stats_from_bank(merged),
    )


def bank_metadata(bank_or_banks):
    if isinstance(bank_or_banks, dict):
        return {variable: bank.metadata() for variable, bank in bank_or_banks.items()}
    return bank_or_banks.metadata()


def build_shared_bank(
    problem,
    *,
    size: int,
    seed: int,
    split: str,
    target: str,
    positive_fraction: float,
    pair_pool_size: int = PAIR_POOL_SIZE,
):
    return build_pair_bank(
        problem,
        size,
        seed,
        split,
        target_vars=tuple(TARGET_VARS),
        pair_policy="mixed",
        pair_policy_target=target,
        mixed_positive_fraction=positive_fraction,
        pair_pool_size=pair_pool_size,
    )


def build_strategy_calibration_bank(problem, strategy_name: str):
    if strategy_name == "shared_wx_positive":
        return build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 301, split="calibration", target="WX", positive_fraction=1.0)
    if strategy_name == "shared_yz_positive":
        return build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 302, split="calibration", target="YZ", positive_fraction=1.0)
    if strategy_name == "shared_any_positive":
        return build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 303, split="calibration", target="any", positive_fraction=1.0)
    if strategy_name == "shared_any_mixed50":
        return build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 304, split="calibration", target="any", positive_fraction=0.5)
    if strategy_name == "shared_both_positive":
        return build_shared_bank(
            problem,
            size=CALIBRATION_PAIR_SIZE,
            seed=SEED + 305,
            split="calibration",
            target="both",
            positive_fraction=1.0,
            pair_pool_size=8192,
        )
    if strategy_name == "shared_balanced_wx_yz_only":
        wx_bank = build_shared_bank(
            problem,
            size=CALIBRATION_PAIR_SIZE // 2,
            seed=SEED + 306,
            split="calibration_wx_only",
            target="WX_only",
            positive_fraction=1.0,
            pair_pool_size=4096,
        )
        yz_bank = build_shared_bank(
            problem,
            size=CALIBRATION_PAIR_SIZE // 2,
            seed=SEED + 307,
            split="calibration_yz_only",
            target="YZ_only",
            positive_fraction=1.0,
            pair_pool_size=4096,
        )
        return concat_pair_banks(
            [wx_bank, yz_bank],
            split="calibration",
            seed=SEED + 308,
            pair_policy="mixed",
            pair_policy_target="balanced_wx_yz_only",
            mixed_positive_fraction=1.0,
        )
    if strategy_name == "separate_variable_positive":
        return {
            "WX": build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 309, split="calibration_wx", target="WX", positive_fraction=1.0),
            "YZ": build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 310, split="calibration_yz", target="YZ", positive_fraction=1.0),
        }
    if strategy_name == "separate_variable_only":
        return {
            "WX": build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 311, split="calibration_wx_only", target="WX_only", positive_fraction=1.0, pair_pool_size=4096),
            "YZ": build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=SEED + 312, split="calibration_yz_only", target="YZ_only", positive_fraction=1.0, pair_pool_size=4096),
        }
    raise ValueError(f"Unsupported strategy {strategy_name}")


def build_fixed_train_bank(problem):
    return build_pair_bank(
        problem,
        TRAIN_PAIR_SIZE,
        SEED + 201,
        "train",
        target_vars=tuple(TARGET_VARS),
        pair_policy="mixed",
        pair_policy_target="any",
        mixed_positive_fraction=0.5,
        pair_pool_size=PAIR_POOL_SIZE,
    )


def build_fixed_test_banks(problem):
    return {
        "WX": build_shared_bank(problem, size=TEST_PAIR_SIZE, seed=SEED + 401, split="test_wx", target="WX", positive_fraction=1.0),
        "YZ": build_shared_bank(problem, size=TEST_PAIR_SIZE, seed=SEED + 402, split="test_yz", target="YZ", positive_fraction=1.0),
    }


def summarize_results(records: list[dict[str, object]]) -> dict[str, object]:
    per_variable = {str(record["variable"]): float(record["exact_acc"]) for record in records}
    return {
        "per_variable_exact_acc": per_variable,
        "average_exact_acc": float(sum(per_variable.values()) / len(per_variable)) if per_variable else 0.0,
    }


def run_one_method(method: str, model, train_bank, calibration_bank, test_bank, device: torch.device):
    start = perf_counter()
    if method == "das":
        payload = run_das_pipeline(
            model=model,
            train_bank=train_bank,
            calibration_bank=calibration_bank,
            holdout_bank=test_bank,
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
                verbose=True,
            ),
        )
    elif method == "ot":
        payload = run_alignment_pipeline(
            model=model,
            fit_bank=train_bank,
            calibration_bank=calibration_bank,
            holdout_bank=test_bank,
            device=device,
            config=OTConfig(
                method="ot",
                batch_size=BATCH_SIZE,
                resolution=RESOLUTION,
                epsilon=OT_EPSILON,
                tau=OT_TAU,
                uot_reg_m=0.1,
                signature_mode=SIGNATURE_MODE,
                target_vars=tuple(TARGET_VARS),
                top_k_values=TOP_K_VALUES,
                lambda_values=LAMBDAS,
                selection_verbose=True,
            ),
        )
    elif method == "uot":
        payload = run_alignment_pipeline(
            model=model,
            fit_bank=train_bank,
            calibration_bank=calibration_bank,
            holdout_bank=test_bank,
            device=device,
            config=OTConfig(
                method="uot",
                batch_size=BATCH_SIZE,
                resolution=RESOLUTION,
                epsilon=OT_EPSILON,
                tau=UOT_TAU,
                uot_reg_m=UOT_REG_M,
                signature_mode=SIGNATURE_MODE,
                target_vars=tuple(TARGET_VARS),
                top_k_values=TOP_K_VALUES,
                lambda_values=LAMBDAS,
                selection_verbose=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported method {method}")

    runtime = perf_counter() - start
    summary = summarize_results(payload["results"])
    return {
        "runtime_seconds": float(runtime),
        "results": payload["results"],
        "summary": summary,
        "selected_hyperparameters": payload.get("selected_hyperparameters"),
        "search_records": payload.get("search_records"),
        "raw_payload": payload,
    }


def strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec("shared_wx_positive", "Shared positive calibration bank targeted to WX."),
        StrategySpec("shared_yz_positive", "Shared positive calibration bank targeted to YZ."),
        StrategySpec("shared_any_positive", "Shared positive calibration bank where at least one variable changes."),
        StrategySpec("shared_any_mixed50", "Shared mixed calibration bank with 50% positive pairs targeted to any change."),
        StrategySpec("shared_both_positive", "Shared positive calibration bank where both WX and YZ change."),
        StrategySpec("shared_balanced_wx_yz_only", "Single shared calibration bank built from half WX_only positives and half YZ_only positives."),
        StrategySpec("separate_variable_positive", "Per-variable calibration banks targeted separately to WX-positive and YZ-positive pairs."),
        StrategySpec("separate_variable_only", "Per-variable calibration banks targeted separately to WX_only and YZ_only pairs."),
    ]


def format_summary(payload: dict[str, object]) -> str:
    lines = [
        "HEQ Calibration Strategy Sweep",
        f"seed: {payload['seed']}",
        f"device: {payload['device']}",
        f"checkpoint: {payload['checkpoint_path']}",
        f"factual_validation_exact_acc: {float(payload['backbone']['factual_validation_metrics']['exact_acc']):.4f}",
        "",
        "Fixed train bank:",
        f"  {payload['train_bank']['pair_policy']} | target={payload['train_bank']['pair_policy_target']} | positive_fraction={float(payload['train_bank']['mixed_positive_fraction']):.2f}",
        "Fixed test banks:",
        f"  WX -> target={payload['test_banks']['WX']['pair_policy_target']}, positive_fraction={float(payload['test_banks']['WX']['mixed_positive_fraction']):.2f}",
        f"  YZ -> target={payload['test_banks']['YZ']['pair_policy_target']}, positive_fraction={float(payload['test_banks']['YZ']['mixed_positive_fraction']):.2f}",
        "",
    ]
    for strategy in payload["strategies"]:
        lines.append(f"[{strategy['name']}] {strategy['description']}")
        lines.append(f"  calibration_bank: {strategy['calibration_bank_description']}")
        for method in ("das", "ot", "uot"):
            result = strategy["methods"][method]
            lines.append(
                f"  {method.upper()}: avg={float(result['summary']['average_exact_acc']):.4f} "
                f"| WX={float(result['summary']['per_variable_exact_acc']['WX']):.4f} "
                f"| YZ={float(result['summary']['per_variable_exact_acc']['YZ']):.4f} "
                f"| runtime={float(result['runtime_seconds']):.2f}s"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    device = resolve_device(DEVICE)
    problem = load_equality_problem(num_entities=NUM_ENTITIES, embedding_dim=EMBEDDING_DIM)
    model, _, backbone_meta = ensure_backbone(problem, device)

    train_bank = build_fixed_train_bank(problem)
    test_banks = build_fixed_test_banks(problem)

    strategy_payloads = []
    for spec in strategy_specs():
        print("=" * 80)
        print(f"Calibration strategy: {spec.name}")
        calibration_bank = build_strategy_calibration_bank(problem, spec.name)
        calibration_description = (
            "per-variable banks" if isinstance(calibration_bank, dict)
            else f"shared bank | target={calibration_bank.pair_policy_target} | positive_fraction={float(calibration_bank.mixed_positive_fraction):.2f}"
        )
        method_results = {}
        for method in ("das", "ot", "uot"):
            print("-" * 80)
            print(f"Running {method.upper()} under strategy {spec.name}")
            method_results[method] = run_one_method(
                method,
                model=model,
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                test_bank=test_banks,
                device=device,
            )
        strategy_payloads.append(
            {
                "name": spec.name,
                "description": spec.description,
                "calibration_bank_description": calibration_description,
                "calibration_bank": bank_metadata(calibration_bank),
                "methods": method_results,
            }
        )

    payload = {
        "seed": SEED,
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "backbone": backbone_meta,
        "train_bank": train_bank.metadata(),
        "test_banks": bank_metadata(test_banks),
        "strategies": strategy_payloads,
        "fixed_method_hparams": {
            "ot": {
                "epsilon": OT_EPSILON,
                "tau": OT_TAU,
                "top_k_values": list(TOP_K_VALUES),
                "lambdas": [float(value) for value in LAMBDAS],
                "signature_mode": SIGNATURE_MODE,
            },
            "uot": {
                "epsilon": OT_EPSILON,
                "tau": UOT_TAU,
                "uot_reg_m": UOT_REG_M,
                "top_k_values": list(TOP_K_VALUES),
                "lambdas": [float(value) for value in LAMBDAS],
                "signature_mode": SIGNATURE_MODE,
            },
            "das": {
                "layers": "all",
                "subspace_dims": list(DAS_SUBSPACE_DIMS),
                "learning_rate": DAS_LEARNING_RATE,
                "max_epochs": DAS_MAX_EPOCHS,
                "min_epochs": DAS_MIN_EPOCHS,
                "plateau_patience": DAS_PLATEAU_PATIENCE,
                "plateau_rel_delta": DAS_PLATEAU_REL_DELTA,
            },
        },
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_PATH, payload)
    SUMMARY_PATH.write_text(format_summary(payload), encoding="utf-8")
    print(f"Saved sweep JSON to {OUTPUT_PATH}")
    print(f"Saved sweep summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
