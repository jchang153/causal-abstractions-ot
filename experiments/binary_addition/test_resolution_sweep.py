from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.binary_addition import run_progressive_plot as progressive
from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split


def _stage(*, calibration: float, test: float, wall: float, sites: list[str]) -> dict[str, object]:
    row = {
        "calibration": {
            "combined": calibration,
            "sensitivity": calibration,
            "invariance": calibration,
        },
        "test": {
            "combined": test,
            "sensitivity": test,
            "invariance": test,
        },
        "top_k": 1,
        "lambda": 1.0,
        "selected_sites": [{"site_key": sites[0], "weight": 1.0}],
        "row_mass": [1.0],
    }
    trial = {
        "alignment_method": "ot",
        "epsilon": 0.1,
        "calibration": {
            "mean_combined": calibration,
            "mean_sensitivity": calibration,
            "mean_invariance": calibration,
        },
        "test": {
            "per_row": {"C1": row},
            "mean_combined": test,
            "mean_sensitivity": test,
            "mean_invariance": test,
            "subset": {
                "mean_combined": test,
                "mean_sensitivity": test,
                "mean_invariance": test,
            },
        },
    }
    return {
        "runtime_seconds": wall / 2.0,
        "wall_runtime_seconds": wall,
        "setup_runtime_seconds": wall / 4.0,
        "sites": sites,
        "trials": [trial],
        "best_trial": trial,
    }


def test_resolution_sweep_selects_shared_epsilon_and_charges_its_full_resolution_sweep() -> None:
    calls: list[tuple[str, ...]] = []

    def fake_stage(**kwargs):
        sites = tuple(site.key() for site in kwargs["sites"])
        calls.append(sites)
        if sites == ("r1",):
            # Better test accuracy must not influence model selection.
            return _stage(calibration=0.6, test=0.95, wall=2.0, sites=list(sites))
        return _stage(calibration=0.8, test=0.4, wall=3.0, sites=list(sites))

    class FakeSite:
        def __init__(self, name: str):
            self.name = name

        def key(self) -> str:
            return self.name

    selected_test = {
        "combined": 0.4,
        "sensitivity": 0.4,
        "invariance": 0.4,
        "selected_sites": [],
        "top_k": 1,
        "lambda": 1.0,
    }
    with patch.object(progressive, "_run_alignment_stage", side_effect=fake_stage), patch.object(
        progressive,
        "evaluate_single_calibrated_transport",
        return_value=selected_test,
    ):
        result = progressive._run_alignment_resolution_sweep(
            stage_name="stage_b",
            alignment_method="ot",
            model=None,
            specs=(),
            row_keys=("C1",),
            banks={"test_positive_by_row": {"C1": ()}, "test_invariant_by_row": {"C1": ()}},
            sites_by_resolution={1: (FakeSite("r1"),), 2: (FakeSite("r2"),)},
            family_order=(),
            transport_cfg=None,
            selection_rule="combined",
            invariance_floor=0.0,
            device=None,
            run_cache=None,
            batch_size=1,
            normalize_signatures=True,
            fit_signature_mode="all",
            fit_stratify_mode="none",
            fit_family_profile="all",
            cost_metric="sq_l2",
            cosine_temperature=1.0,
            bruteforce_temperature=1.0,
            rotation_map=None,
        )

    assert calls == [("r1",), ("r2",)]
    assert result["selected_resolution_by_row"] == {"C1": 2}
    assert result["best_trial"]["test"]["subset"]["mean_combined"] == 0.4
    assert result["selected_epsilon"] == 0.1
    assert result["calibration_sweep_runtime_seconds"] == 2.5
    assert result["selected_epsilon_resolution_sweep_runtime_seconds"] == 2.5
    assert result["full_hyperparameter_sweep_wall_runtime_seconds"] == 5.0
    assert result["runtime_seconds"] >= 2.5
    assert result["resolution_results"]["1"]["sites"] == ["r1"]
    assert result["resolution_results"]["2"]["sites"] == ["r2"]


def test_paper_addition_banks_use_identical_full_fit_pairs_for_every_variable() -> None:
    all_examples = enumerate_all_examples(width=4)
    split = stratified_base_split(
        all_examples,
        fit_count=128,
        calib_count=64,
        test_count=64,
        seed=0,
    )
    specs = progressive._row_specs("all_endogenous", width=4)
    banks = progressive._build_banks(
        split,
        specs,
        width=4,
        seed=0,
        source_policy="structured_26_top3carry_c2x5_c3x7_no_random",
        all_examples=all_examples,
    )

    def pair_keys(row_key: str) -> tuple[tuple[int, int, int, int, str], ...]:
        records = progressive._fit_records_for_row(
            banks["fit_by_row"][row_key],
            row_key=row_key,
            fit_bank_mode="shared",
        )
        return tuple(
            (record.base.a, record.base.b, record.source.a, record.source.b, record.family)
            for record in records
        )

    reference = pair_keys("C1")
    assert len(reference) == 3328
    for spec in specs:
        assert pair_keys(spec.key) == reference
        assert (
            len(banks["calib_positive_by_row"][spec.key])
            + len(banks["calib_invariant_by_row"][spec.key])
            == 1664
        )
        assert (
            len(banks["test_positive_by_row"][spec.key])
            + len(banks["test_invariant_by_row"][spec.key])
            == 1664
        )
