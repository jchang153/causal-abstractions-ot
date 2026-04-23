from __future__ import annotations

from mcqa_delta_hierarchical_sweep import _extract_stage_b_best_configs, _select_stage_b_layers, _select_stage_c_configs
from pathlib import Path
import json


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

    assert _select_stage_b_layers(rankings=rankings, top_layers_per_var=2) == (18, 20, 25)


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

    assert selected == {("last_token", "all_variants", "partition", 8): (20, 25)}
