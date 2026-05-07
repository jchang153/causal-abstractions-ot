import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from addition_experiment.backbone import load_backbone
from addition_experiment.das import DASConfig, evaluate_rotated_intervention, iter_search_specs, train_rotated_intervention
from addition_experiment.pair_bank import PairBankVariableDataset, build_structured_pair_bank
from addition_experiment.pyvene_utils import build_intervenable
from addition_experiment.runtime import resolve_device, write_json
from addition_experiment.scm import load_addition_problem
from pyvene import RotatedSpaceIntervention


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_addition_mib_c1_diagnostic"
CHECKPOINT_PATH = Path("models/addition_mlp_seed42.pt")
TARGET_VAR = "C1"


@dataclass(frozen=True)
class DiagnosticConfig:
    train_pair_size: int = 4000
    calibration_pair_size: int = 1500
    test_pair_size: int = 4000
    pair_pool_size: int | None = 2048
    batch_size: int = 128
    das_max_epochs: int = 1000
    das_min_epochs: int = 5
    das_plateau_patience: int = 2
    das_plateau_rel_delta: float = 5e-3
    das_learning_rate: float = 1e-3
    das_subspace_dims: tuple[int, ...] = (1, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192)
    das_layers: tuple[int, ...] | None = None


def _build_bank(problem, *, size: int, seed: int, split: str, positive_fraction: float):
    return build_structured_pair_bank(
        problem,
        size,
        seed,
        split,
        target_vars=(TARGET_VAR,),
        pair_policy="mixed",
        pair_policy_target=TARGET_VAR,
        mixed_positive_fraction=positive_fraction,
        pair_pool_size=config.pair_pool_size,
    )


def _selection_summary(records: list[dict[str, object]], score_name: str, score_fn):
    best = max(records, key=score_fn)
    return {
        "score_name": score_name,
        "selected_site": best["site_label"],
        "layer": best["layer"],
        "subspace_dim": best["subspace_dim"],
        "selection_score": float(score_fn(best)),
        "calibration_positive_exact_acc": float(best["calibration_positive_exact_acc"]),
        "calibration_invariant_exact_acc": float(best["calibration_invariant_exact_acc"]),
        "calibration_combined_exact_acc": float(best["calibration_combined_exact_acc"]),
        "test_positive_exact_acc": float(best["test_positive_exact_acc"]),
        "test_invariant_exact_acc": float(best["test_invariant_exact_acc"]),
        "test_combined_exact_acc": float(best["test_combined_exact_acc"]),
    }


def main() -> None:
    global config
    config = DiagnosticConfig()
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    model, model_config, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    train_bank = _build_bank(problem, size=config.train_pair_size, seed=SEED + 201, split="train_c1_mib", positive_fraction=0.5)
    calibration_positive = _build_bank(problem, size=config.calibration_pair_size, seed=SEED + 301, split="calibration_c1_positive", positive_fraction=1.0)
    calibration_invariant = _build_bank(problem, size=config.calibration_pair_size, seed=SEED + 302, split="calibration_c1_invariant", positive_fraction=0.0)
    test_positive = _build_bank(problem, size=config.test_pair_size, seed=SEED + 401, split="test_c1_positive", positive_fraction=1.0)
    test_invariant = _build_bank(problem, size=config.test_pair_size, seed=SEED + 402, split="test_c1_invariant", positive_fraction=0.0)

    das_config = DASConfig(
        batch_size=config.batch_size,
        max_epochs=config.das_max_epochs,
        min_epochs=config.das_min_epochs,
        learning_rate=config.das_learning_rate,
        subspace_dims=config.das_subspace_dims,
        search_layers=config.das_layers,
        target_vars=(TARGET_VAR,),
        plateau_patience=config.das_plateau_patience,
        plateau_rel_delta=config.das_plateau_rel_delta,
        verbose=True,
        progress_interval=25,
    )

    specs = iter_search_specs(model, das_config)
    train_dataset = PairBankVariableDataset(train_bank, TARGET_VAR)
    calib_positive_ds = PairBankVariableDataset(calibration_positive, TARGET_VAR)
    calib_invariant_ds = PairBankVariableDataset(calibration_invariant, TARGET_VAR)
    test_positive_ds = PairBankVariableDataset(test_positive, TARGET_VAR)
    test_invariant_ds = PairBankVariableDataset(test_invariant, TARGET_VAR)

    candidate_records = []
    t0 = perf_counter()
    print(f"DAS [{TARGET_VAR}] diagnostic | candidates={len(specs)}")
    for index, spec in enumerate(specs, start=1):
        intervention = RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[spec.layer]))
        intervenable = build_intervenable(
            model=model,
            layer=spec.layer,
            component=spec.component,
            intervention=intervention,
            device=device,
            unit=spec.unit,
            max_units=spec.max_units,
            freeze_model=True,
            freeze_intervention=False,
            use_fast=False,
        )
        loss_history = train_rotated_intervention(
            intervenable=intervenable,
            dataset=train_dataset,
            spec=spec,
            max_epochs=das_config.max_epochs,
            learning_rate=das_config.learning_rate,
            batch_size=das_config.batch_size,
            device=device,
            plateau_patience=das_config.plateau_patience,
            plateau_rel_delta=das_config.plateau_rel_delta,
            min_epochs=das_config.min_epochs,
        )
        calib_pos = evaluate_rotated_intervention(intervenable, calib_positive_ds, spec, das_config.batch_size, device)
        calib_inv = evaluate_rotated_intervention(intervenable, calib_invariant_ds, spec, das_config.batch_size, device)
        test_pos = evaluate_rotated_intervention(intervenable, test_positive_ds, spec, das_config.batch_size, device)
        test_inv = evaluate_rotated_intervention(intervenable, test_invariant_ds, spec, das_config.batch_size, device)
        record = {
            "site_label": spec.label,
            "layer": spec.layer,
            "subspace_dim": spec.subspace_dim,
            "train_epochs_ran": len(loss_history),
            "train_loss_history": loss_history,
            "calibration_positive_exact_acc": float(calib_pos["exact_acc"]),
            "calibration_invariant_exact_acc": float(calib_inv["exact_acc"]),
            "calibration_combined_exact_acc": 0.5 * (float(calib_pos["exact_acc"]) + float(calib_inv["exact_acc"])),
            "test_positive_exact_acc": float(test_pos["exact_acc"]),
            "test_invariant_exact_acc": float(test_inv["exact_acc"]),
            "test_combined_exact_acc": 0.5 * (float(test_pos["exact_acc"]) + float(test_inv["exact_acc"])),
        }
        candidate_records.append(record)
        print(
            f"{index}/{len(specs)} {spec.label} | "
            f"calib_pos={record['calibration_positive_exact_acc']:.4f} "
            f"calib_inv={record['calibration_invariant_exact_acc']:.4f} "
            f"test_pos={record['test_positive_exact_acc']:.4f} "
            f"test_inv={record['test_invariant_exact_acc']:.4f}"
        )

    runtime_sec = perf_counter() - t0
    weights = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    summaries = {
        "positive_only": _selection_summary(records=candidate_records, score_name="positive_only", score_fn=lambda r: float(r["calibration_positive_exact_acc"])),
        "invariant_only": _selection_summary(records=candidate_records, score_name="invariant_only", score_fn=lambda r: float(r["calibration_invariant_exact_acc"])),
    }
    for weight in weights:
        summaries[f"weighted_{weight:.2f}"] = _selection_summary(
            records=candidate_records,
            score_name=f"weighted_{weight:.2f}",
            score_fn=lambda r, w=weight: w * float(r["calibration_positive_exact_acc"]) + (1.0 - w) * float(r["calibration_invariant_exact_acc"]),
        )

    payload = {
        "seed": SEED,
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_config": model_config.to_dict(),
        "backbone_meta": backbone_meta,
        "config": asdict(config),
        "train_bank": train_bank.metadata(),
        "calibration_positive_bank": calibration_positive.metadata(),
        "calibration_invariant_bank": calibration_invariant.metadata(),
        "test_positive_bank": test_positive.metadata(),
        "test_invariant_bank": test_invariant.metadata(),
        "candidate_records": candidate_records,
        "selection_summaries": summaries,
        "runtime_sec": runtime_sec,
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RUN_DIR / "addition_mib_c1_diagnostic_results.json"
    write_json(json_path, payload)
    print(json.dumps({"json": str(json_path.resolve()), "runtime_sec": runtime_sec, "selection_summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
