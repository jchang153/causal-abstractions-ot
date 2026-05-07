from __future__ import annotations

import argparse
from pathlib import Path

from mcqa_delta_broad_sweep import _build_parser, _normalize_args, build_broad_sweep_plan


def _parse_args(*extra: str) -> tuple[argparse.Namespace, dict[str, object]]:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--results-root",
            "results/delta",
            "--results-timestamp",
            "20260423_broad_test",
            *extra,
        ]
    )
    return args, _normalize_args(args)


def test_default_broad_sweep_plan_contains_expected_stage_matrix() -> None:
    args, normalized = _parse_args()
    plan = build_broad_sweep_plan(repo_root=Path("."), args=args, normalized=normalized)

    assert [stage.name for stage in plan] == [
        "vanilla_ot_full",
        "pca_ot_pair_bank_partition",
        "pca_ot_pair_bank_mixed",
        "pca_ot_all_variants_partition",
        "pca_ot_all_variants_mixed",
        "pca_guided_das_pair_bank_partition",
        "pca_guided_das_all_variants_partition",
        "regular_das_full",
    ]


def test_guided_stage_outputs_target_layer_and_row_jsons() -> None:
    args, normalized = _parse_args("--stages", "pca_guided_das")
    plan = build_broad_sweep_plan(repo_root=Path("."), args=args, normalized=normalized)

    assert len(plan) == 2
    first_stage = plan[0]
    assert first_stage.category == "pca_guided_das"
    assert len(first_stage.expected_outputs) == 4
    assert all(output.endswith("_das_guided.json") for output in first_stage.expected_outputs)
    assert any("answer_pointer" in output for output in first_stage.expected_outputs)
    assert any("answer_token" in output for output in first_stage.expected_outputs)


def test_stage_subset_respects_requested_pca_matrix() -> None:
    args, normalized = _parse_args(
        "--stages",
        "pca_ot",
        "--pca-basis-source-modes",
        "all_variants",
        "--pca-site-menus",
        "mixed",
    )
    plan = build_broad_sweep_plan(repo_root=Path("."), args=args, normalized=normalized)

    assert len(plan) == 1
    assert plan[0].name == "pca_ot_all_variants_mixed"
