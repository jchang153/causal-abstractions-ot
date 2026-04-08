import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from addition_experiment.backbone import load_backbone
from addition_experiment.das import (
    DASConfig,
    evaluate_rotated_intervention,
    iter_search_specs,
    train_rotated_intervention,
)
from addition_experiment.metrics import metrics_from_logits
from addition_experiment.pair_bank import PairBankVariableDataset, build_structured_pair_bank
from addition_experiment.pyvene_utils import build_intervenable
from addition_experiment.runtime import resolve_device, write_json
from addition_experiment.scm import load_addition_problem
from pyvene import RotatedSpaceIntervention
from torch.utils.data import DataLoader
from variable_width_mlp import logits_from_output


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_addition_mib_c1_eval"
CHECKPOINT_PATH = Path("models/addition_mlp_seed42.pt")
TARGET_VAR = "C1"


@dataclass(frozen=True)
class C1MIBConfig:
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


def _build_structured_c1_bank(problem, *, size: int, seed: int, split: str, positive_fraction: float):
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


def _evaluate_identity(model, bank, device: torch.device) -> dict[str, float]:
    dataset = PairBankVariableDataset(bank, TARGET_VAR)
    logits_all = []
    labels_all = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            outputs = model(inputs_embeds=inputs)
            logits_all.append(logits_from_output(outputs).cpu())
            labels_all.append(batch["labels"].to(torch.long).view(-1).cpu())
    return metrics_from_logits(torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0))


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def main() -> None:
    global config
    config = C1MIBConfig()
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    model, model_config, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    train_bank = _build_structured_c1_bank(
        problem,
        size=config.train_pair_size,
        seed=SEED + 201,
        split="train_c1_mib",
        positive_fraction=0.5,
    )
    calibration_positive = _build_structured_c1_bank(
        problem,
        size=config.calibration_pair_size,
        seed=SEED + 301,
        split="calibration_c1_positive",
        positive_fraction=1.0,
    )
    calibration_invariant = _build_structured_c1_bank(
        problem,
        size=config.calibration_pair_size,
        seed=SEED + 302,
        split="calibration_c1_invariant",
        positive_fraction=0.0,
    )
    test_positive = _build_structured_c1_bank(
        problem,
        size=config.test_pair_size,
        seed=SEED + 401,
        split="test_c1_positive",
        positive_fraction=1.0,
    )
    test_invariant = _build_structured_c1_bank(
        problem,
        size=config.test_pair_size,
        seed=SEED + 402,
        split="test_c1_invariant",
        positive_fraction=0.0,
    )

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

    best_record = None
    best_intervenable = None
    best_spec = None
    search_records = []
    t0 = perf_counter()
    print(
        f"DAS [{TARGET_VAR}] MIB-style | candidates={len(specs)} | "
        f"train_examples={len(train_dataset)} | "
        f"calib_positive={len(calib_positive_ds)} | calib_invariant={len(calib_invariant_ds)}"
    )
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
        calib_positive_metrics = evaluate_rotated_intervention(
            intervenable=intervenable,
            dataset=calib_positive_ds,
            spec=spec,
            batch_size=das_config.batch_size,
            device=device,
        )
        calib_invariant_metrics = evaluate_rotated_intervention(
            intervenable=intervenable,
            dataset=calib_invariant_ds,
            spec=spec,
            batch_size=das_config.batch_size,
            device=device,
        )
        calib_pos_exact = float(calib_positive_metrics["exact_acc"])
        calib_inv_exact = float(calib_invariant_metrics["exact_acc"])
        calib_avg_exact = 0.5 * (calib_pos_exact + calib_inv_exact)
        calib_harmonic_exact = _harmonic_mean(calib_pos_exact, calib_inv_exact)
        record = {
            "method": "das",
            "variable": TARGET_VAR,
            "site_label": spec.label,
            "layer": spec.layer,
            "subspace_dim": spec.subspace_dim,
            "train_epochs_ran": len(loss_history),
            "train_loss_history": loss_history,
            "calibration_positive_exact_acc": calib_positive_metrics["exact_acc"],
            "calibration_positive_mean_shared_digits": calib_positive_metrics["mean_shared_digits"],
            "calibration_invariant_exact_acc": calib_invariant_metrics["exact_acc"],
            "calibration_invariant_mean_shared_digits": calib_invariant_metrics["mean_shared_digits"],
            "calibration_avg_exact_acc": calib_avg_exact,
            "calibration_harmonic_exact_acc": calib_harmonic_exact,
        }
        search_records.append(record)
        is_better = best_record is None or (
            float(record["calibration_harmonic_exact_acc"]),
            float(record["calibration_avg_exact_acc"]),
            float(record["calibration_positive_exact_acc"]),
            float(record["calibration_invariant_exact_acc"]),
        ) > (
            float(best_record["calibration_harmonic_exact_acc"]),
            float(best_record["calibration_avg_exact_acc"]),
            float(best_record["calibration_positive_exact_acc"]),
            float(best_record["calibration_invariant_exact_acc"]),
        )
        status = "new best" if is_better else "candidate"
        print(
            f"DAS [{TARGET_VAR}] {status} {index}/{len(specs)} | site={spec.label} "
            f"| epochs={len(loss_history)} | train_loss={loss_history[-1]:.4f} "
            f"| calib_pos={calib_pos_exact:.4f} "
            f"| calib_inv={calib_inv_exact:.4f} "
            f"| calib_avg={calib_avg_exact:.4f} "
            f"| calib_h={calib_harmonic_exact:.4f}"
        )
        if is_better:
            best_record = record
            best_intervenable = intervenable
            best_spec = spec

    test_positive_metrics = evaluate_rotated_intervention(
        intervenable=best_intervenable,
        dataset=PairBankVariableDataset(test_positive, TARGET_VAR),
        spec=best_spec,
        batch_size=das_config.batch_size,
        device=device,
    )
    test_invariant_metrics = evaluate_rotated_intervention(
        intervenable=best_intervenable,
        dataset=PairBankVariableDataset(test_invariant, TARGET_VAR),
        spec=best_spec,
        batch_size=das_config.batch_size,
        device=device,
    )
    runtime_sec = perf_counter() - t0
    identity_positive = _evaluate_identity(model, test_positive, device)
    identity_invariant = _evaluate_identity(model, test_invariant, device)
    result = {
        **best_record,
        "test_positive_exact_acc": test_positive_metrics["exact_acc"],
        "test_positive_mean_shared_digits": test_positive_metrics["mean_shared_digits"],
        "test_invariant_exact_acc": test_invariant_metrics["exact_acc"],
        "test_invariant_mean_shared_digits": test_invariant_metrics["mean_shared_digits"],
        "test_avg_exact_acc": 0.5
        * (float(test_positive_metrics["exact_acc"]) + float(test_invariant_metrics["exact_acc"])),
        "test_harmonic_exact_acc": _harmonic_mean(
            float(test_positive_metrics["exact_acc"]),
            float(test_invariant_metrics["exact_acc"]),
        ),
    }

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
        "identity": {
            "positive": identity_positive,
            "invariant": identity_invariant,
            "avg_exact_acc": 0.5
            * (float(identity_positive["exact_acc"]) + float(identity_invariant["exact_acc"])),
        },
        "das": {
            "result": result,
            "runtime_sec": runtime_sec,
        },
        "search_records": search_records,
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RUN_DIR / "addition_mib_c1_eval_results.json"
    write_json(json_path, payload)
    (RUN_DIR / "addition_mib_c1_eval_summary.txt").write_text(
        "\n".join(
            [
                "Addition MIB-style C1 DAS Eval",
                f"checkpoint: {CHECKPOINT_PATH}",
                f"avg_test_exact: {float(result['test_avg_exact_acc']):.4f}",
                f"test_positive_exact: {float(result['test_positive_exact_acc']):.4f}",
                f"test_invariant_exact: {float(result['test_invariant_exact_acc']):.4f}",
                f"identity_avg_exact: {float(payload['identity']['avg_exact_acc']):.4f}",
                f"selected_site: {result['site_label']}",
                f"runtime_sec: {runtime_sec:.2f}",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": str(json_path.resolve()),
                "test_positive_exact_acc": result["test_positive_exact_acc"],
                "test_invariant_exact_acc": result["test_invariant_exact_acc"],
                "test_avg_exact_acc": result["test_avg_exact_acc"],
                "identity_avg_exact_acc": payload["identity"]["avg_exact_acc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
