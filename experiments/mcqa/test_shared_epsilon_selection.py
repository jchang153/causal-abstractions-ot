from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.mcqa.mcqa_experiment.selection import select_shared_epsilon
from experiments.mcqa.mcqa_paper_runtime import (
    _matching_pca_stage_b_entries,
    _native_selected_width_epsilon_runtime,
    _pca_selected_config_epsilon_runtime,
)

MCQA_DIR = Path(__file__).resolve().parent
if str(MCQA_DIR) not in sys.path:
    sys.path.insert(0, str(MCQA_DIR))

from mcqa_delta_hierarchical_sweep import (
    _extract_dimension_das_rankings,
    _extract_layer_das_rankings,
    _extract_native_support_rankings,
    _extract_stage_b_best_configs,
)
from mcqa_plot_das_layer import _core_method_runtime_seconds
import mcqa_plot_layer as plot_layer


def test_shared_epsilon_averages_each_variables_best_config() -> None:
    candidates = [
        {"variable": "A", "epsilon": 0.1, "resolution": 1, "calibration_score": 0.9},
        {"variable": "A", "epsilon": 0.1, "resolution": 2, "calibration_score": 0.8},
        {"variable": "B", "epsilon": 0.1, "resolution": 1, "calibration_score": 0.3},
        {"variable": "B", "epsilon": 0.1, "resolution": 2, "calibration_score": 0.4},
        {"variable": "A", "epsilon": 1.0, "resolution": 1, "calibration_score": 0.7},
        {"variable": "A", "epsilon": 1.0, "resolution": 2, "calibration_score": 0.6},
        {"variable": "B", "epsilon": 1.0, "resolution": 1, "calibration_score": 0.7},
        {"variable": "B", "epsilon": 1.0, "resolution": 2, "calibration_score": 0.6},
    ]

    selected = select_shared_epsilon(
        candidates,
        variables=("A", "B"),
        epsilon_order=(0.1, 1.0),
    )

    assert selected["selected_epsilon"] == 1.0
    assert selected["epsilon_plans"][0]["mean_best_variable_calibration_score"] == 0.65
    assert selected["epsilon_plans"][1]["mean_best_variable_calibration_score"] == 0.7
    assert selected["selected_by_variable"]["A"]["resolution"] == 1
    assert selected["selected_by_variable"]["B"]["resolution"] == 1


def test_shared_epsilon_ties_follow_declared_grid_order() -> None:
    selected = select_shared_epsilon(
        [
            {"variable": "A", "epsilon": 0.1, "calibration_score": 0.5},
            {"variable": "B", "epsilon": 0.1, "calibration_score": 0.5},
            {"variable": "A", "epsilon": 1.0, "calibration_score": 0.5},
            {"variable": "B", "epsilon": 1.0, "calibration_score": 0.5},
        ],
        variables=("A", "B"),
        epsilon_order=(1.0, 0.1),
    )

    assert selected["selected_epsilon"] == 1.0


def test_native_runtime_charges_every_width_at_selected_epsilon_and_shares_setup(tmp_path: Path) -> None:
    wrapper_paths = []
    for width, runtimes in ((1, (2.0, 3.0)), (2, (4.0, 5.0))):
        child_paths = []
        for epsilon in (0.1, 1.0):
            child_path = tmp_path / f"w{width}_e{epsilon}.json"
            child_path.write_text(
                json.dumps(
                    {
                        "ot_epsilon": epsilon,
                        "method_payloads": {
                            "ot": [
                                {"target_var": "answer_pointer", "runtime_seconds": runtimes[0], "signature_prepare_runtime_seconds": 0.0},
                                {"target_var": "answer_token", "runtime_seconds": runtimes[1], "signature_prepare_runtime_seconds": 0.0},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            child_paths.append(str(child_path))
        wrapper_path = tmp_path / f"w{width}.json"
        wrapper_path.write_text(
            json.dumps(
                {
                    "layer": 7,
                    "native_resolution": width,
                    "signature_prepare_runtime_seconds": 1.0,
                    "ot_output_paths": child_paths,
                }
            ),
            encoding="utf-8",
        )
        wrapper_paths.append(wrapper_path)

    rankings = {
        variable: [
            {"variable": variable, "layer": 7, "native_resolution": width, "epsilon": 1.0, "payload_path": str(path)}
            for width, path in ((1, wrapper_paths[0]), (2, wrapper_paths[1]))
        ]
        for variable in ("answer_pointer", "answer_token")
    }
    entries = {variable: records[0] for variable, records in rankings.items()}
    downstream, _, _ = _native_selected_width_epsilon_runtime(
        rankings=rankings,
        entries_by_var=entries,
        restrict_to_selected_width=False,
        restrict_to_selected_epsilon=True,
    )

    # The selected epsilon is charged at both widths. The one-second signature
    # setup for each width is shared across the two jointly aligned variables.
    assert downstream["answer_pointer"] == 7.0
    assert downstream["answer_token"] == 9.0


def test_native_rankings_use_one_macro_selected_epsilon(tmp_path: Path) -> None:
    payload_paths = []
    scores = {
        "answer_pointer": {0.1: (0.9, 0.8), 1.0: (0.7, 0.8)},
        "answer_token": {0.1: (0.2, 0.3), 1.0: (0.8, 0.7)},
    }
    for variable in ("answer_pointer", "answer_token"):
        for width_index, width in enumerate((1, 2)):
            child_paths = []
            for epsilon in (0.1, 1.0):
                child = tmp_path / f"{variable}_w{width}_e{epsilon}.json"
                score = scores[variable][epsilon][width_index]
                child.write_text(
                    json.dumps(
                        {
                            "ot_epsilon": epsilon,
                            "method_payloads": {
                                "ot": [
                                    {
                                        "target_var": variable,
                                        "selected_hyperparameters": {"top_k": 1, "lambda": 1.0},
                                        "results": [
                                            {
                                                "variable": variable,
                                                "selection_score": score,
                                                "calibration_iia_acc": score,
                                                "iia_acc": score,
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                child_paths.append(str(child))
            wrapper = tmp_path / f"{variable}_w{width}.json"
            wrapper.write_text(
                json.dumps(
                    {
                        "kind": "mcqa_plot_native_support_layer",
                        "layer": 7,
                        "native_resolution": width,
                        "alignment_method": "ot",
                        "site_labels": ["site"],
                        "localization_runtime_seconds": 1.0,
                        "ot_output_paths": child_paths,
                    }
                ),
                encoding="utf-8",
            )
            payload_paths.append(wrapper)

    rankings = _extract_native_support_rankings(payload_paths=payload_paths)

    assert rankings["answer_pointer"][0]["epsilon"] == 1.0
    assert rankings["answer_pointer"][0]["native_resolution"] == 2
    assert rankings["answer_token"][0]["epsilon"] == 1.0
    assert rankings["answer_token"][0]["native_resolution"] == 1


def test_pca_runtime_charges_selected_epsilon_all_configs_and_fits_basis_once(tmp_path: Path) -> None:
    rankings: dict[str, list[dict[str, object]]] = {
        "answer_pointer": [],
        "answer_token": [],
    }
    core_seconds = {
        "answer_pointer": {8: 3.0, 16: 4.0},
        "answer_token": {8: 5.0, 16: 6.0},
    }
    for target_var in rankings:
        for num_bands in (8, 16):
            child_paths = []
            for epsilon in (0.1, 1.0):
                child_path = tmp_path / f"{target_var}_b{num_bands}_e{epsilon}.json"
                child_path.write_text(
                    json.dumps(
                        {
                            "ot_epsilon": epsilon,
                            "method_payloads": {
                                "ot": [
                                    {
                                        "target_var": target_var,
                                        "runtime_seconds": core_seconds[target_var][num_bands],
                                        "signature_prepare_runtime_seconds": 0.0,
                                    }
                                ]
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                child_paths.append(str(child_path))
            wrapper_path = tmp_path / f"{target_var}_b{num_bands}.json"
            wrapper_path.write_text(
                json.dumps(
                    {
                        "layer": 7,
                        "basis_source_mode": "all_variants",
                        "site_menu": "partition",
                        "num_bands": num_bands,
                        "pca_fit_runtime_seconds": 10.0,
                        "pca_site_build_runtime_seconds": 1.0,
                        "signature_prepare_runtime_seconds": 2.0,
                        "ot_output_paths": child_paths,
                    }
                ),
                encoding="utf-8",
            )
            rankings[target_var].append(
                {
                    "variable": target_var,
                    "layer": 7,
                    "basis_source_mode": "all_variants",
                    "site_menu": "partition",
                    "num_bands": num_bands,
                    "epsilon": 1.0,
                    "payload_path": str(wrapper_path),
                }
            )

    entries = {target_var: target_entries[0] for target_var, target_entries in rankings.items()}
    downstream, _, _ = _pca_selected_config_epsilon_runtime(
        rankings=rankings,
        entries_by_var=entries,
        restrict_to_selected_config=False,
        restrict_to_selected_epsilon=True,
    )

    # One PCA fit (10s), one 3s setup per band, and all selected-epsilon cores.
    assert downstream["answer_pointer"] == 15.0
    assert downstream["answer_token"] == 19.0


def test_pca_guided_runtime_restricts_stage_b_to_selected_layer(tmp_path: Path) -> None:
    rankings = {"answer_pointer": []}
    for layer, core_seconds in ((7, 3.0), (8, 30.0)):
        child_path = tmp_path / f"pca_layer{layer}.json"
        child_path.write_text(
            json.dumps(
                {
                    "ot_epsilon": 1.0,
                    "method_payloads": {
                        "ot": [
                            {
                                "target_var": "answer_pointer",
                                "runtime_seconds": core_seconds,
                                "signature_prepare_runtime_seconds": 0.0,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        wrapper_path = tmp_path / f"pca_wrapper_layer{layer}.json"
        wrapper_path.write_text(
            json.dumps(
                {
                    "layer": layer,
                    "basis_source_mode": "all_variants",
                    "site_menu": "partition",
                    "num_bands": 8,
                    "pca_fit_runtime_seconds": 1.0,
                    "ot_output_paths": [str(child_path)],
                }
            ),
            encoding="utf-8",
        )
        rankings["answer_pointer"].append(
            {
                "variable": "answer_pointer",
                "layer": layer,
                "basis_source_mode": "all_variants",
                "site_menu": "partition",
                "num_bands": 8,
                "epsilon": 1.0,
                "payload_path": str(wrapper_path),
            }
        )

    downstream, _, _ = _pca_selected_config_epsilon_runtime(
        rankings=rankings,
        entries_by_var={"answer_pointer": rankings["answer_pointer"][0]},
        restrict_to_selected_layer=True,
        restrict_to_selected_config=False,
        restrict_to_selected_epsilon=True,
    )

    assert downstream["answer_pointer"] == 4.0


def test_pca_rankings_use_one_macro_selected_epsilon(tmp_path: Path) -> None:
    payload_paths = []
    for variable, scores in {
        "answer_pointer": {0.1: 0.9, 1.0: 0.7},
        "answer_token": {0.1: 0.2, 1.0: 0.8},
    }.items():
        children = []
        for epsilon, score in scores.items():
            child = tmp_path / f"pca_{variable}_{epsilon}.json"
            child.write_text(
                json.dumps(
                    {
                        "ot_epsilon": epsilon,
                        "method_payloads": {
                            "ot": [
                                {
                                    "target_var": variable,
                                    "selected_hyperparameters": {"top_k": 1, "lambda": 1.0},
                                    "results": [
                                        {
                                            "variable": variable,
                                            "selection_score": score,
                                            "calibration_iia_acc": score,
                                            "iia_acc": score,
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            children.append(str(child))
        wrapper = tmp_path / f"pca_{variable}.json"
        wrapper.write_text(
            json.dumps(
                {
                    "layer": 7,
                    "token_position_id": "last_token",
                    "basis_source_mode": "all_variants",
                    "site_menu": "partition",
                    "num_bands": 8,
                    "runtime_seconds": 1.0,
                    "ot_output_paths": children,
                }
            ),
            encoding="utf-8",
        )
        payload_paths.append(wrapper)

    rankings = _extract_stage_b_best_configs(payload_paths=payload_paths)

    assert rankings["answer_pointer"][0]["epsilon"] == 1.0
    assert rankings["answer_token"][0]["epsilon"] == 1.0


def test_pca_guided_runtime_recovers_epsilon_from_stage_b_ranking() -> None:
    stage_b_entry = {
        "variable": "answer_pointer",
        "layer": 18,
        "basis_source_mode": "all_variants",
        "site_menu": "partition",
        "num_bands": 16,
        "epsilon": 0.5,
    }
    guided_entry = {
        "variable": "answer_pointer",
        "layer": 18,
        "basis_source_mode": "all_variants",
        "site_menu": "partition",
        "num_bands": 16,
    }

    matched = _matching_pca_stage_b_entries(
        pca_rankings={"answer_pointer": [stage_b_entry]},
        guided_entries_by_var={"answer_pointer": guided_entry},
    )

    assert matched["answer_pointer"] == stage_b_entry
    assert matched["answer_pointer"]["epsilon"] == 0.5


def test_das_rankings_use_inner_method_runtime_not_wrapper_setup(tmp_path: Path) -> None:
    payload_path = tmp_path / "das_layer_summary.json"
    method_payloads = [
        {
            "target_var": "answer_pointer",
            "runtime_seconds": 2.0,
            "results": [{"variable": "answer_pointer", "selection_score": 0.8, "iia_acc": 0.7}],
        },
        {
            "target_var": "answer_token",
            "runtime_seconds": 3.0,
            "results": [{"variable": "answer_token", "selection_score": 0.9, "iia_acc": 0.8}],
        },
    ]
    payload = {
        "kind": "mcqa_plot_das_layer",
        "layer": 7,
        "runtime_seconds": 999.0,
        "method_payloads": {"das": method_payloads},
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    rankings = _extract_layer_das_rankings(payload_paths=[payload_path])
    assert rankings["answer_pointer"][0]["runtime_seconds"] == 2.0
    assert rankings["answer_token"][0]["runtime_seconds"] == 3.0
    assert _core_method_runtime_seconds(payload, "das") == 5.0

    dimension_rankings = _extract_dimension_das_rankings(
        payload_records=[
            (
                payload_path,
                {"variable": "answer_pointer", "layer": 7, "native_resolution": 128},
            )
        ]
    )
    assert dimension_rankings["answer_pointer"][0]["runtime_seconds"] == 2.0


def test_stage_a_selected_plan_runtime_includes_final_holdout(monkeypatch) -> None:
    def fake_holdout(**kwargs):
        return {
            "runtime_seconds": 1.0,
            "results": [{"iia_acc": 0.75, "variable": kwargs["holdout_bank"]}],
            "ranking": [],
        }

    monkeypatch.setattr(plot_layer, "_evaluate_fixed_single_layer_holdout_only", fake_holdout)
    selected = {
        "runtime_with_signatures_seconds": 5.0,
        "per_var_records": {
            "answer_pointer": {"site_label": "site", "selection_score": 0.5},
            "answer_token": {"site_label": "site", "selection_score": 0.5},
        },
    }
    updated = plot_layer._evaluate_selected_stage_a_holdout(
        model=None,
        tokenizer=None,
        banks_by_split={
            "test": {"answer_pointer": "answer_pointer", "answer_token": "answer_token"}
        },
        sites=[type("Site", (), {"label": "site"})()],
        device="cpu",
        batch_size=1,
        selected_config=selected,
    )

    assert updated["holdout_eval_runtime_seconds"] == 2.0
    assert updated["runtime_with_signatures_seconds"] == 7.0
