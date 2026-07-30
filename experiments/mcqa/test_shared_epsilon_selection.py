from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.mcqa.mcqa_experiment.selection import select_shared_epsilon
from experiments.mcqa.mcqa_paper_runtime import _native_selected_width_epsilon_runtime

MCQA_DIR = Path(__file__).resolve().parent
if str(MCQA_DIR) not in sys.path:
    sys.path.insert(0, str(MCQA_DIR))

from mcqa_delta_hierarchical_sweep import _extract_native_support_rankings, _extract_stage_b_best_configs


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


def test_native_runtime_charges_every_width_at_selected_epsilon(tmp_path: Path) -> None:
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

    assert downstream["answer_pointer"] == 8.0
    assert downstream["answer_token"] == 10.0


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
                                                "calibration_exact_acc": score,
                                                "exact_acc": score,
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
                                            "calibration_exact_acc": score,
                                            "exact_acc": score,
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
