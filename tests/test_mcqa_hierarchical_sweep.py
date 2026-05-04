from __future__ import annotations

from pathlib import Path
import json

from mcqa_delta_hierarchical_sweep import (
    _extract_stage_a_rankings,
    _extract_stage_b_best_configs,
    _select_stage_b_layers,
    _select_stage_c_configs,
)
from mcqa_delta_hierarchical_parallel import (
    plan_stage_b_tasks,
    plan_stage_c_tasks,
)


def _write_layer_run(path: Path, *, layer: int, pointer_cal: float, pointer_exact: float, token_cal: float, token_exact: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "ot_epsilon": 0.5,
                        "candidate_sites": [f"L{layer}:last_token[0:2304]"],
                        "method_payloads": {
                            "ot": [
                                {
                                    "target_var": "answer_pointer",
                                    "results": [
                                        {
                                            "variable": "answer_pointer",
                                            "exact_acc": pointer_exact,
                                            "selection_score": pointer_cal,
                                            "site_label": "soft:k1,l2",
                                        }
                                    ],
                                },
                                {
                                    "target_var": "answer_token",
                                    "results": [
                                        {
                                            "variable": "answer_token",
                                            "exact_acc": token_exact,
                                            "selection_score": token_cal,
                                            "site_label": "soft:k1,l2",
                                        }
                                    ],
                                },
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_extract_stage_a_rankings_uses_layer_sweep_calibration_not_mass(tmp_path: Path) -> None:
    weak_layer = tmp_path / "layer_06" / "mcqa_run_results.json"
    strong_layer = tmp_path / "layer_17" / "mcqa_run_results.json"
    _write_layer_run(weak_layer, layer=6, pointer_cal=0.03, pointer_exact=0.0, token_cal=0.2, token_exact=0.25)
    _write_layer_run(strong_layer, layer=17, pointer_cal=0.84, pointer_exact=0.79, token_cal=0.9, token_exact=0.95)
    manifest_path = tmp_path / "layer_sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"layer": 6, "output_path": str(weak_layer)},
                    {"layer": 17, "output_path": str(strong_layer)},
                ]
            }
        ),
        encoding="utf-8",
    )

    rankings = _extract_stage_a_rankings(aggregate_path=manifest_path)

    assert [entry["layer"] for entry in rankings["answer_pointer"]] == [17, 6]
    assert rankings["answer_pointer"][0]["selection_score"] == 0.84
    assert rankings["answer_pointer"][0]["selection_basis"] == "full_layer_calibration"


def test_extract_stage_a_joint_run_selects_one_global_uot_hyperparameter(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "mcqa_run_results.json"

    def payload(target_var: str, row_index: int, cal: float, exact: float, masses: list[list[float]]) -> dict[str, object]:
        return {
            "target_var": target_var,
            "source_target_vars": ["answer_pointer", "answer_token"],
            "target_var_row_index": row_index,
            "target_normalized_transport": masses,
            "results": [
                {
                    "variable": target_var,
                    "exact_acc": exact,
                    "selection_score": cal,
                    "site_label": "soft:k1,l2",
                }
            ],
        }

    aggregate_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "ot_epsilon": 0.5,
                        "uot_beta_neural": 0.03,
                        "candidate_sites": ["L17:last_token[0:2304]", "L24:last_token[0:2304]"],
                        "method_payloads": {
                            "uot": [
                                payload("answer_pointer", 0, 0.9, 0.7, [[0.9, 0.1], [0.3, 0.7]]),
                                payload("answer_token", 1, 0.2, 0.8, [[0.9, 0.1], [0.3, 0.7]]),
                            ]
                        },
                    },
                    {
                        "ot_epsilon": 4.0,
                        "uot_beta_neural": 3.0,
                        "candidate_sites": ["L14:last_token[0:2304]", "L24:last_token[0:2304]"],
                        "method_payloads": {
                            "uot": [
                                payload("answer_pointer", 0, 0.7, 0.72, [[0.6, 0.4], [0.1, 0.9]]),
                                payload("answer_token", 1, 0.8, 0.88, [[0.6, 0.4], [0.1, 0.9]]),
                            ]
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rankings = _extract_stage_a_rankings(aggregate_path=aggregate_path, hparam_selection="joint")

    assert rankings["answer_pointer"][0]["layer"] == 14
    assert rankings["answer_token"][0]["layer"] == 24
    assert rankings["answer_pointer"][0]["epsilon"] == 4.0
    assert rankings["answer_token"][0]["epsilon"] == 4.0
    assert rankings["answer_pointer"][0]["uot_beta_neural"] == 3.0
    assert rankings["answer_token"][0]["uot_beta_neural"] == 3.0
    assert rankings["answer_pointer"][0]["joint_handle_calibration_score"] == 0.75
    assert rankings["answer_token"][0]["selection_basis"] == "joint_uot_global_hparam_target_mass"


def test_select_stage_b_layers_uses_union_of_top_layers_per_row() -> None:
    rankings = {
        "answer_pointer": [
            {"layer": 20, "exact_acc": 0.7, "selection_score": 0.6},
            {"layer": 18, "exact_acc": 0.6, "selection_score": 0.5},
        ],
        "answer_token": [
            {"layer": 20, "exact_acc": 0.9, "selection_score": 0.8},
            {"layer": 25, "exact_acc": 0.88, "selection_score": 0.8},
        ],
    }

    assert _select_stage_b_layers(
        rankings=rankings,
        top_layers_per_var=2,
        neighbor_radius=1,
        max_layers_per_var=5,
    ) == (18, 19, 20, 21, 25)


def test_extract_stage_b_best_configs_collapses_epsilons_per_config(tmp_path: Path) -> None:
    ot_eps_05 = tmp_path / "eps_05.json"
    ot_eps_1 = tmp_path / "eps_1.json"
    layer_payload_path = tmp_path / "layer_payload.json"

    ot_eps_05.write_text(
        json.dumps(
            {
                "ot_epsilon": 0.5,
                "method_payloads": {
                    "ot": [
                        {
                            "target_var": "answer_token",
                            "results": [
                                {
                                    "variable": "answer_token",
                                    "exact_acc": 0.9,
                                    "selection_score": 0.8,
                                    "site_label": "soft:k1,l2",
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    ot_eps_1.write_text(
        json.dumps(
            {
                "ot_epsilon": 1.0,
                "method_payloads": {
                    "ot": [
                        {
                            "target_var": "answer_token",
                            "results": [
                                {
                                    "variable": "answer_token",
                                    "exact_acc": 0.93,
                                    "selection_score": 0.81,
                                    "site_label": "soft:k1,l2",
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    layer_payload_path.write_text(
        json.dumps(
            {
                "layer": 20,
                "token_position_id": "last_token",
                "basis_source_mode": "pair_bank",
                "site_menu": "mixed",
                "num_bands": 8,
                "ot_output_paths": [str(ot_eps_05), str(ot_eps_1)],
            }
        ),
        encoding="utf-8",
    )

    rankings = _extract_stage_b_best_configs(payload_paths=[layer_payload_path])

    assert len(rankings["answer_token"]) == 1
    best = rankings["answer_token"][0]
    assert best["exact_acc"] == 0.93
    assert best["epsilon"] == 1.0
    assert best["basis_source_mode"] == "pair_bank"
    assert best["site_menu"] == "mixed"


def test_select_stage_c_configs_groups_by_token_basis_menu_and_bands() -> None:
    rankings = {
        "answer_pointer": [
            {
                "token_position_id": "last_token",
                "basis_source_mode": "all_variants",
                "site_menu": "partition",
                "num_bands": 8,
                "layer": 20,
                "exact_acc": 0.71,
                "selection_score": 0.66,
            }
        ],
        "answer_token": [
            {
                "token_position_id": "last_token",
                "basis_source_mode": "all_variants",
                "site_menu": "partition",
                "num_bands": 8,
                "layer": 25,
                "exact_acc": 0.83,
                "selection_score": 0.79,
            }
        ],
    }

    selected = _select_stage_c_configs(rankings=rankings, top_configs_per_var=1)

    assert selected == {("custom", "ot", "last_token", "all_variants", "partition", 8): (20, 25)}


def test_parallel_stage_b_planner_writes_disjoint_task_roots(tmp_path: Path) -> None:
    timestamp = "parallel_test"
    sweep_root = tmp_path / f"{timestamp}_mcqa_hierarchical_sweep"
    sweep_root.mkdir(parents=True)
    (sweep_root / "stage_a_last_token_layer_rankings.json").write_text(
        json.dumps(
            {
                "answer_pointer": [
                    {"layer": 18, "exact_acc": 0.8, "selection_score": 0.74},
                ],
                "answer_token": [
                    {"layer": 24, "exact_acc": 0.97, "selection_score": 0.98},
                ],
            }
        ),
        encoding="utf-8",
    )
    from mcqa_delta_hierarchical_sweep import _build_parser

    args = _build_parser().parse_args(
        [
            "--results-root",
            str(tmp_path),
            "--results-timestamp",
            timestamp,
            "--stage-b-top-layers-per-var",
            "1",
            "--stage-b-neighbor-radius",
            "0",
            "--stage-b-max-layers-per-var",
            "1",
            "--pca-site-menus",
            "partition,mixed",
            "--pca-basis-source-modes",
            "pair_bank,all_variants",
            "--pca-num-bands-values",
            "8,16",
        ]
    )

    plan_stage_b_tasks(args)

    native_tasks = json.loads((sweep_root / "stage_b_native_tasks.json").read_text())["tasks"]
    pca_tasks = json.loads((sweep_root / "stage_b_pca_tasks.json").read_text())["tasks"]

    assert len(native_tasks) == 16
    assert len(pca_tasks) == 12
    native_stage_roots = [Path(task["expected_outputs"][0]).parent for task in native_tasks]
    assert len(native_stage_roots) == len(set(native_stage_roots))
    assert all("_res" in task["stage_timestamp"] for task in native_tasks)
    stage_roots = [Path(task["expected_outputs"][0]).parent for task in pca_tasks]
    assert len(stage_roots) == len(set(stage_roots))
    assert all("_L" in task["stage_timestamp"] for task in pca_tasks)


def test_parallel_stage_c_planner_reuses_stage_b_task_root(tmp_path: Path) -> None:
    timestamp = "parallel_test"
    sweep_root = tmp_path / f"{timestamp}_mcqa_hierarchical_sweep"
    sweep_root.mkdir(parents=True)
    (sweep_root / "stage_a_last_token_layer_rankings.json").write_text(
        json.dumps(
            {
                "answer_pointer": [],
                "answer_token": [
                    {
                        "layer": 23,
                        "exact_acc": 0.96,
                        "selection_score": 0.94,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage_b_timestamp = f"{timestamp}_stageB_last_token_pair_bank_partition_8b_L23"
    (sweep_root / "stage_b_pca_rankings.json").write_text(
        json.dumps(
            {
                "answer_pointer": [],
                "answer_token": [
                    {
                        "token_position_id": "last_token",
                        "basis_source_mode": "pair_bank",
                        "site_menu": "partition",
                        "num_bands": 8,
                        "layer": 23,
                        "exact_acc": 0.96,
                        "selection_score": 0.94,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage_b_payload_path = (
        tmp_path
        / f"{stage_b_timestamp}_mcqa_ot_pca_focus"
        / "layer_23"
        / "mcqa_layer-23_pos-last_token_pca-menu-partition-bands-8-scheme-equal_basis-pair_bank_sig-family_label_delta_norm_ot_pca.json"
    )
    (sweep_root / "stage_b_pca_tasks.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "pca_last_token_pair_bank_partition_8b_L23",
                        "token_position_id": "last_token",
                        "basis_source_mode": "pair_bank",
                        "site_menu": "partition",
                        "num_bands": 8,
                        "layer": 23,
                        "stage_timestamp": stage_b_timestamp,
                        "payload_paths": [str(stage_b_payload_path)],
                        "expected_outputs": [str(stage_b_payload_path)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    from mcqa_delta_hierarchical_sweep import _build_parser

    args = _build_parser().parse_args(
        [
            "--results-root",
            str(tmp_path),
            "--results-timestamp",
            timestamp,
            "--stage-c-top-configs-per-var",
            "1",
        ]
    )

    plan_stage_c_tasks(args)

    stage_c_tasks = json.loads((sweep_root / "stage_c_pca_tasks.json").read_text())["tasks"]
    a_only_tasks = json.loads((sweep_root / "stage_c_a_only_tasks.json").read_text())["tasks"]
    assert len(stage_c_tasks) == 1
    assert len(a_only_tasks) == 1
    assert stage_c_tasks[0]["stage_timestamp"] == stage_b_timestamp
    assert any(path.endswith("_answer_token_das_guided.json") for path in stage_c_tasks[0]["expected_outputs"])
    assert "--das-restarts" in a_only_tasks[0]["command"]
    assert a_only_tasks[0]["command"][a_only_tasks[0]["command"].index("--das-restarts") + 1] == "2"
