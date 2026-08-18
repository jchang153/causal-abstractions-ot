from __future__ import annotations

import json

import torch

from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.interventions import build_run_cache, factual_run
from experiments.binary_addition.model import GRUAdder
from experiments.binary_addition.provenance import protocol_provenance
from experiments.binary_addition.run_joint_endogenous_resolution_sweep import (
    _build_structured_source_index,
    _structured_sources_for_base,
)
from experiments.binary_addition.summarize_plot4_pilot import summarize


LEGACY_POLICY = "structured_26_top3carry_c2x5_c3x7_no_random"
WIDTH_NEUTRAL_POLICY = "structured_top3carry_c2x5_c3x7_no_random"


def _source_keys(rows: tuple[tuple[str, object], ...]) -> list[tuple[str, int, int]]:
    return [(family, int(source.a), int(source.b)) for family, source in rows]


def test_dense_batched_run_cache_matches_single_example_forwards() -> None:
    torch.manual_seed(0)
    examples = enumerate_all_examples(width=3)
    model = GRUAdder(width=3, hidden_size=5)
    cache = build_run_cache(model, examples, device=torch.device("cpu"), batch_size=7)
    assert cache.hidden_table.shape == (64, 3, 5)
    for example in examples:
        expected = factual_run(model, example, device=torch.device("cpu"))
        actual = cache.get_run(example)
        assert torch.allclose(actual.hidden_states, expected.hidden_states, atol=1e-7, rtol=1e-6)
        assert torch.allclose(actual.output_logits, expected.output_logits, atol=1e-7, rtol=1e-6)
        assert cache.get_input(example).shape == (1, 3, 2)


def test_indexed_sources_exactly_match_width4_reference() -> None:
    examples = enumerate_all_examples(width=4)
    source_index = _build_structured_source_index(examples, width=4)
    for base in examples[::29]:
        reference = _structured_sources_for_base(
            base,
            width=4,
            all_examples=examples,
            seed=0,
            source_policy=LEGACY_POLICY,
        )
        indexed = _structured_sources_for_base(
            base,
            width=4,
            all_examples=examples,
            seed=0,
            source_policy=WIDTH_NEUTRAL_POLICY,
            source_index=source_index,
        )
        assert _source_keys(indexed) == _source_keys(reference)


def test_width8_protocol_has_46_sources_and_disjoint_splits() -> None:
    examples = enumerate_all_examples(width=8)
    source_index = _build_structured_source_index(examples, width=8)
    sources = _structured_sources_for_base(
        examples[12345],
        width=8,
        all_examples=examples,
        seed=0,
        source_policy=WIDTH_NEUTRAL_POLICY,
        source_index=source_index,
    )
    assert len(sources) == 46
    assert len({family for family, _source in sources}) == 46
    split = stratified_base_split(examples, fit_count=512, calib_count=256, test_count=256, seed=0)
    fit = {(item.a, item.b) for item in split.fit}
    calib = {(item.a, item.b) for item in split.calib}
    test = {(item.a, item.b) for item in split.test}
    assert not (fit & calib or fit & test or calib & test)
    assert (len(fit) * 46, len(calib) * 46, len(test) * 46) == (23552, 11776, 11776)


def test_protocol_and_four_method_summary_require_shared_data(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"local checkpoint")
    examples = enumerate_all_examples(width=3)
    split = stratified_base_split(examples, fit_count=32, calib_count=16, test_count=16, seed=0)
    rows = ("C1", "C2")
    protocol = protocol_provenance(split=split, checkpoint=checkpoint, rows=rows)
    single_root = tmp_path / "single_stage" / "h64" / "seed_0"
    progressive_root = tmp_path / "progressive_ot" / "h64" / "seed_0"
    single_root.mkdir(parents=True)
    progressive_root.mkdir(parents=True)
    (single_root / "single_stage_plot_seed_summary.json").write_text(
        json.dumps({"protocol": protocol, "method": {"mean_combined": 0.7, "runtime_seconds": 3.0}})
    )
    (single_root / "single_stage_plot_pca_seed_summary.json").write_text(
        json.dumps({"protocol": protocol, "method": {"mean_combined": 0.8, "runtime_seconds": 2.0}})
    )
    (progressive_root / "progressive_seed_summary.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "methods": {
                    "plot_in_timestep": {"mean_combined": 0.75, "runtime_seconds": 2.5},
                    "plot_pca_in_timestep": {"mean_combined": 0.85, "runtime_seconds": 1.5},
                },
            }
        )
    )
    result = summarize(tmp_path, hidden_size=64, seed=0)
    assert [row["method"] for row in result["methods"]] == [
        "PLOT (single-stage)",
        "PLOT-native (two-stage)",
        "PLOT-PCA (single-stage)",
        "PLOT-PCA (two-stage)",
    ]
    table = (tmp_path / "plot4_summary.md").read_text()
    assert "Average accuracy" in table and "Serial runtime" in table
    assert "C1" not in table
