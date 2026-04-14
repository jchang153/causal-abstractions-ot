import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import equality_run as equality_run_config
from equality_experiment.backbone import load_backbone
from equality_experiment.das import (
    DASConfig,
    _resolve_bank_for_variable,
    evaluate_rotated_intervention,
    train_rotated_intervention,
)
from equality_experiment.ot import OTConfig, run_alignment_pipeline
from equality_experiment.pair_bank import PairBankVariableDataset, build_pair_bank
from equality_experiment.pyvene_utils import DASSearchSpec, build_intervenable
from equality_experiment.scm import load_equality_problem
from pyvene import RotatedSpaceIntervention


RESULTS_DIR = Path("results/20260413_220248_equality")
SEED_RUN_PATH = RESULTS_DIR / "seed_1" / "equality_run_results.json"
OUTPUT_PATH = RESULTS_DIR / "method_intervention_heatmaps.png"


def build_train_bank(problem, seed: int):
    return build_pair_bank(
        problem,
        equality_run_config.TRAIN_PAIR_SIZE,
        int(seed) + 201,
        "train",
        target_vars=tuple(equality_run_config.TARGET_VARS),
        pair_policy=equality_run_config.TRAIN_PAIR_POLICY,
        pair_policy_target=equality_run_config.TRAIN_PAIR_POLICY_TARGET,
        mixed_positive_fraction=equality_run_config.TRAIN_MIXED_POSITIVE_FRACTION,
        pair_pool_size=equality_run_config.TRAIN_PAIR_POOL_SIZE,
    )


def das_weight_to_layer_row(weight: np.ndarray, subspace_dim: int) -> np.ndarray:
    """Convert a 16x16 rotation into one 16-wide layer row via subspace loadings."""
    active = weight[: int(subspace_dim), :]
    return np.linalg.norm(active, axis=0)


def recover_selected_das_result(
    *,
    model,
    variable: str,
    selected_record: dict[str, object],
    train_bank,
    calibration_bank,
    holdout_bank,
    device: torch.device,
) -> dict[str, object]:
    spec = DASSearchSpec(
        layer=int(selected_record["layer"]),
        subspace_dim=int(selected_record["subspace_dim"]),
        component=f"h[{int(selected_record['layer'])}].output",
    )
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
    train_dataset = PairBankVariableDataset(train_bank, variable)
    calibration_dataset = PairBankVariableDataset(_resolve_bank_for_variable(calibration_bank, variable), variable)
    holdout_dataset = PairBankVariableDataset(_resolve_bank_for_variable(holdout_bank, variable), variable)
    train_rotated_intervention(
        intervenable=intervenable,
        dataset=train_dataset,
        spec=spec,
        max_epochs=equality_run_config.DAS_MAX_EPOCHS,
        learning_rate=equality_run_config.DAS_LEARNING_RATE,
        batch_size=equality_run_config.BATCH_SIZE,
        device=device,
        plateau_patience=equality_run_config.DAS_PLATEAU_PATIENCE,
        plateau_rel_delta=equality_run_config.DAS_PLATEAU_REL_DELTA,
        min_epochs=equality_run_config.DAS_MIN_EPOCHS,
    )
    calibration_metrics = evaluate_rotated_intervention(
        intervenable=intervenable,
        dataset=calibration_dataset,
        spec=spec,
        batch_size=equality_run_config.BATCH_SIZE,
        device=device,
    )
    holdout_metrics = evaluate_rotated_intervention(
        intervenable=intervenable,
        dataset=holdout_dataset,
        spec=spec,
        batch_size=equality_run_config.BATCH_SIZE,
        device=device,
    )
    rotate_layer = next(iter(intervenable.interventions.values())).rotate_layer
    return {
        "variable": variable,
        "layer": int(spec.layer),
        "subspace_dim": int(spec.subspace_dim),
        "calibration_exact_acc": float(calibration_metrics["exact_acc"]),
        "exact_acc": float(holdout_metrics["exact_acc"]),
        "weight": rotate_layer.weight.detach().cpu().numpy(),
    }


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    seed_payload = json.loads(SEED_RUN_PATH.read_text())
    seed = int(seed_payload["seed"])

    equality_run_config.CALIBRATION_STRATEGY = str(seed_payload["calibration_strategy"])
    equality_run_config.TARGETED_EVAL = bool(seed_payload["targeted_eval"])

    device = torch.device("cpu")
    problem = load_equality_problem(
        run_checks=True,
        num_entities=equality_run_config.NUM_ENTITIES,
        embedding_dim=equality_run_config.EMBEDDING_DIM,
        seed=seed,
    )
    train_config = equality_run_config.build_train_config_for_seed(seed)
    model, _, _ = load_backbone(
        problem=problem,
        checkpoint_path=Path(seed_payload["checkpoint_path"]),
        device=device,
        train_config=train_config,
    )

    train_bank = build_train_bank(problem, seed)
    calibration_bank = equality_run_config._build_calibration_bank(problem, seed)
    test_bank = equality_run_config._build_test_bank(problem, seed)

    best_method_runs = seed_payload["best_method_runs"]

    ot_record = best_method_runs["ot"]
    ot_payload = run_alignment_pipeline(
        model=model,
        fit_bank=train_bank,
        calibration_bank=calibration_bank,
        holdout_bank=test_bank,
        device=device,
        config=OTConfig(
            method="ot",
            batch_size=equality_run_config.BATCH_SIZE,
            resolution=equality_run_config.RESOLUTION,
            alpha=equality_run_config.FGW_ALPHA,
            epsilon=float(ot_record["ot_epsilon"]),
            tau=float(ot_record.get("ot_tau", 1.0)),
            uot_beta_abstract=float(ot_record.get("uot_beta_abstract", equality_run_config.UOT_BETA_ABSTRACTS[0])),
            uot_beta_neural=float(ot_record.get("uot_beta_neural", equality_run_config.UOT_BETA_NEURALS[0])),
            solver_backend=equality_run_config.TRANSPORT_SOLVER_BACKEND,
            signature_mode=str(ot_record["signature_mode"]),
            target_vars=tuple(equality_run_config.TARGET_VARS),
            top_k_values=tuple(equality_run_config.OT_TOP_K_VALUES),
            lambda_values=tuple(equality_run_config.OT_LAMBDAS),
            selection_verbose=False,
        ),
    )

    uot_record = best_method_runs["uot"]
    uot_payload = run_alignment_pipeline(
        model=model,
        fit_bank=train_bank,
        calibration_bank=calibration_bank,
        holdout_bank=test_bank,
        device=device,
        config=OTConfig(
            method="uot",
            batch_size=equality_run_config.BATCH_SIZE,
            resolution=equality_run_config.RESOLUTION,
            alpha=equality_run_config.FGW_ALPHA,
            epsilon=float(uot_record["ot_epsilon"]),
            tau=float(uot_record.get("ot_tau", 1.0)),
            uot_beta_abstract=float(uot_record["uot_beta_abstract"]),
            uot_beta_neural=float(uot_record["uot_beta_neural"]),
            solver_backend=equality_run_config.TRANSPORT_SOLVER_BACKEND,
            signature_mode=str(uot_record["signature_mode"]),
            target_vars=tuple(equality_run_config.TARGET_VARS),
            top_k_values=tuple(equality_run_config.OT_TOP_K_VALUES),
            lambda_values=tuple(equality_run_config.OT_LAMBDAS),
            selection_verbose=False,
        ),
    )

    _ = DASConfig(
        batch_size=equality_run_config.BATCH_SIZE,
        max_epochs=equality_run_config.DAS_MAX_EPOCHS,
        min_epochs=equality_run_config.DAS_MIN_EPOCHS,
        plateau_patience=equality_run_config.DAS_PLATEAU_PATIENCE,
        plateau_rel_delta=equality_run_config.DAS_PLATEAU_REL_DELTA,
        learning_rate=equality_run_config.DAS_LEARNING_RATE,
        subspace_dims=tuple(equality_run_config.DAS_SUBSPACE_DIMS),
        search_layers=equality_run_config.DAS_LAYERS,
        target_vars=tuple(equality_run_config.TARGET_VARS),
        verbose=False,
    )
    das_selected_records = {
        str(record["variable"]): record
        for record in best_method_runs["das"]["comparison"]["method_selections"]["das"]["results"]
    }
    das_results = []
    for variable in equality_run_config.TARGET_VARS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        das_results.append(
            recover_selected_das_result(
                model=model,
                variable=variable,
                selected_record=das_selected_records[variable],
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device=device,
            )
        )

    method_to_panels: dict[str, dict[str, np.ndarray]] = {
        "OT": {},
        "UOT": {},
        "DAS": {},
    }
    ot_transport = np.asarray(ot_payload["transport"], dtype=float)
    uot_transport = np.asarray(uot_payload["transport"], dtype=float)
    for variable_index, variable in enumerate(equality_run_config.TARGET_VARS):
        method_to_panels["OT"][variable] = ot_transport[variable_index].reshape(3, 16)
        method_to_panels["UOT"][variable] = uot_transport[variable_index].reshape(3, 16)

    for result in das_results:
        variable = str(result["variable"])
        layer = int(result["layer"])
        subspace_dim = int(result["subspace_dim"])
        weight = np.asarray(result["weight"], dtype=float)
        matrix = np.zeros((3, 16), dtype=float)
        matrix[layer] = das_weight_to_layer_row(weight, subspace_dim)
        method_to_panels["DAS"][variable] = matrix

    all_values = np.concatenate(
        [
            method_to_panels[method][variable].ravel()
            for method in ("OT", "UOT", "DAS")
            for variable in equality_run_config.TARGET_VARS
        ]
    )
    vmax = float(np.max(all_values)) if all_values.size else 1.0
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.5), constrained_layout=True)
    methods = ("OT", "UOT", "DAS")
    variables = tuple(equality_run_config.TARGET_VARS)
    title_map = {"WX": r"$Z_{WX}$", "YZ": r"$Z_{YZ}$"}
    for row_index, method in enumerate(methods):
        for col_index, variable in enumerate(variables):
            ax = axes[row_index, col_index]
            image = ax.imshow(
                method_to_panels[method][variable],
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
            )
            if row_index == 0:
                ax.set_title(title_map[variable])
            if col_index == 0:
                ax.set_ylabel(method)
            ax.set_xticks(range(16))
            ax.set_xticklabels([str(i + 1) for i in range(16)], fontsize=8)
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"L{i}" for i in range(3)], fontsize=9)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.92)
    colorbar.set_label("Intervention Weight")
    fig.suptitle("Learned Intervention Heatmaps by Method and Variable", fontsize=14)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
