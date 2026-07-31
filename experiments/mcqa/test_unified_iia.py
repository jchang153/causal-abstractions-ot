from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mcqa_experiment.checking import (
    checker_accuracy,
    normalize_answer_symbol,
    normalized_answer_checker,
    payload_uses_unified_iia,
    selection_metric_from_metrics,
)
from mcqa_experiment.metrics import (
    STRUCTURED_LABEL_DIM,
    das_metrics_from_logits,
    metrics_from_logits,
    prediction_details_from_logits,
)
from mcqa_experiment.ot import (
    OTConfig,
    _select_bruteforce_site,
    _select_hyperparameters,
    solve_bruteforce_coupling_transport,
)
from mcqa_experiment.reporting import summarize_method_records
from mcqa_experiment.signatures import signature_from_logits


class FakeTokenizer:
    def __init__(self) -> None:
        self.text_by_id = {
            **{index + 1: f" {chr(ord('A') + index)}" for index in range(26)},
            50: "AB",
            51: "?",
            52: " b",
            53: " d",
        }

    def decode(self, token_ids: list[int]) -> str:
        assert len(token_ids) == 1
        return self.text_by_id.get(int(token_ids[0]), "<other>")


def make_bank(
    expected: list[str],
    *,
    target_var: str = "answer_token",
    families: list[str] | None = None,
) -> SimpleNamespace:
    batch = len(expected)
    alphabet_ids = torch.arange(1, 27, dtype=torch.long).view(1, 26, 1).repeat(batch, 1, 1)
    symbol_ids = alphabet_ids[:, :4, :].clone()
    expected_indices = [ord(normalize_answer_symbol(value) or "A") - ord("A") for value in expected]
    answer_token_ids = torch.tensor([index + 1 for index in expected_indices], dtype=torch.long)
    return SimpleNamespace(
        split="synthetic",
        target_var=target_var,
        expected_answer_texts=expected,
        counterfactual_family_names=families
        or ["answerPosition", "randomLetter", "answerPosition_randomLetter"][:batch],
        alphabet_variant_token_ids=alphabet_ids,
        alphabet_token_ids=alphabet_ids[:, :, 0],
        symbol_variant_token_ids=symbol_ids,
        symbol_token_ids=symbol_ids[:, :, 0],
        labels=torch.zeros(batch, dtype=torch.long),
        answer_token_ids=answer_token_ids,
        base_inputs=[{"raw_input": f"base-{index}"} for index in range(batch)],
        source_inputs=[{"raw_input": f"source-{index}"} for index in range(batch)],
    )


def logits_with_top_ids(top_ids: list[int], *, vocab_size: int = 80) -> torch.Tensor:
    logits = torch.full((len(top_ids), vocab_size), -100.0)
    for row, token_id in enumerate(top_ids):
        logits[row, int(token_id)] = 20.0
    return logits


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("A", "A"),
        ("a", "A"),
        (" A", "A"),
        (" a", "A"),
        ("ａ", "A"),
        ("", None),
        ("   ", None),
        ("AB", None),
        ("?", None),
        ("answer A", None),
    ],
)
def test_strict_answer_normalizer(value: str, expected: str | None) -> None:
    assert normalize_answer_symbol(value) == expected


def test_verdict_checker_never_uses_substring_matching() -> None:
    assert normalized_answer_checker(" a", "A")
    assert not normalized_answer_checker("AB", "A")
    assert not normalized_answer_checker("A", "answer A")
    assert checker_accuracy([" a", "AB", "?"], ["A", "A", "B"]) == pytest.approx(1 / 3)


def test_full_vocab_top1_overrides_alphabet_restricted_winner() -> None:
    tokenizer = FakeTokenizer()
    bank = make_bank([" A", "b", " C"])
    logits = torch.full((3, 80), -100.0)
    logits[0, 1] = 10.0  # A wins only within the alphabet projection.
    logits[0, 50] = 11.0  # Global top-1 decodes to invalid "AB".
    logits[1, 2] = 10.0
    logits[1, 52] = 11.0  # Global top-1 is a lower-case/leading-space B variant.
    logits[2, 3] = 11.0

    metrics = metrics_from_logits(logits, bank, tokenizer=tokenizer)

    assert metrics["iia_acc"] == pytest.approx(2 / 3)
    assert metrics["exact_acc"] == metrics["iia_acc"]
    assert metrics["diagnostic_alphabet_restricted_acc"] == 1.0
    assert metrics["diagnostic_full_vocab_raw_token_acc"] == pytest.approx(1 / 3)
    assert metrics["iia_validity_counts"] == {
        "total": 3,
        "correct": 2,
        "valid_predictions": 2,
        "invalid_predictions": 1,
        "valid_expected": 3,
        "invalid_expected": 0,
    }
    assert metrics["family_iia_accs"] == {
        "answerPosition": 0.0,
        "randomLetter": 1.0,
        "answerPosition_randomLetter": 1.0,
    }
    details = prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    assert details["predicted_token_ids"] == [50, 52, 3]
    assert details["normalized_predicted_text"] == [None, "B", "C"]
    assert details["correct"] == [0, 1, 1]


@pytest.mark.parametrize(
    ("target_var", "expected", "token_id"),
    [("answer_pointer", " c", 3), ("answer_token", "D", 53)],
)
def test_ap_and_at_use_final_causal_interchange_symbol(
    target_var: str,
    expected: str,
    token_id: int,
) -> None:
    bank = make_bank([expected], target_var=target_var, families=["answerPosition"])
    metrics = das_metrics_from_logits(
        logits_with_top_ids([token_id]), bank, tokenizer=FakeTokenizer()
    )
    assert metrics["iia_acc"] == 1.0
    assert metrics["family_iia_accs"] == {"answerPosition": 1.0}


def test_ot_signatures_remain_26_symbol_projected() -> None:
    assert STRUCTURED_LABEL_DIM == 26
    bank = make_bank(
        ["A", "B", "C"],
        families=["answerPosition", "randomLetter", "answerPosition_randomLetter"],
    )
    base_logits = torch.zeros((3, 80))
    counterfactual_a = torch.zeros((3, 80))
    counterfactual_a[:, 1:27] = torch.arange(26, dtype=torch.float32)
    counterfactual_b = counterfactual_a.clone()
    counterfactual_b[:, 50] = 10_000.0  # A non-alphabet global winner cannot affect signatures.

    signature_a = signature_from_logits(
        counterfactual_logits=counterfactual_a,
        base_logits=base_logits,
        bank=bank,
        signature_mode="family_label_delta_norm",
    )
    signature_b = signature_from_logits(
        counterfactual_logits=counterfactual_b,
        base_logits=base_logits,
        bank=bank,
        signature_mode="family_label_delta_norm",
    )

    assert signature_a.shape == (3 * 26,)
    assert torch.equal(signature_a, signature_b)


def test_bruteforce_scores_candidates_directly_by_iia(monkeypatch: pytest.MonkeyPatch) -> None:
    sites = [SimpleNamespace(label="low", layer=0), SimpleNamespace(label="high", layer=1)]
    scores = {"low": 0.2, "high": 0.9}

    def fake_evaluate(*, site, **_kwargs):
        score = scores[site.label]
        return {
            "iia_acc": score,
            "exact_acc": 1.0 - score,
            "diagnostic_alphabet_restricted_acc": 1.0 - score,
        }, []

    monkeypatch.setattr("mcqa_experiment.ot._evaluate_single_site_intervention", fake_evaluate)
    config = OTConfig(
        method="bruteforce-coupling",
        source_target_vars=("answer_pointer",),
        lambda_values=(1.0,),
        calibration_metric="iia_acc",
        selection_verbose=False,
    )
    transport, metadata = solve_bruteforce_coupling_transport(
        model=None,
        fit_banks_by_var={"answer_pointer": SimpleNamespace(target_var="answer_pointer")},
        sites=sites,
        batch_size=1,
        device=torch.device("cpu"),
        tokenizer=FakeTokenizer(),
        config=config,
    )
    assert int(transport[0].argmax()) == 1
    assert metadata["score_type"] == "fit_bank_single_site_normalized_full_vocab_iia"
    assert metadata["row_iia_accs"] == {"answer_pointer": [0.2, 0.9]}

    selected, _ = _select_bruteforce_site(
        model=None,
        calibration_bank=SimpleNamespace(target_var="answer_pointer"),
        sites=sites,
        batch_size=1,
        device=torch.device("cpu"),
        tokenizer=FakeTokenizer(),
        config=config,
    )
    assert selected["site_label"] == "high"
    assert selected["iia_acc"] == 0.9


def test_transport_handle_selection_and_ties_use_iia(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_soft(*, top_k, **_kwargs):
        iia = {1: 0.25, 2: 0.75}[int(top_k)]
        return {
            "iia_acc": iia,
            "exact_acc": 1.0 - iia,
            "diagnostic_alphabet_restricted_acc": 1.0 - iia,
            "family_iia_accs": {},
        }, []

    monkeypatch.setattr("mcqa_experiment.ot._evaluate_soft_intervention", fake_soft)
    config = OTConfig(
        top_k_values=(1, 2),
        lambda_values=(1.0,),
        calibration_metric="iia_acc",
        selection_verbose=False,
    )
    selected, _ = _select_hyperparameters(
        model=None,
        calibration_bank=SimpleNamespace(target_var="answer_token"),
        sites=[SimpleNamespace(label="a"), SimpleNamespace(label="b")],
        selection_transport=torch.tensor([[0.6, 0.4]]).numpy(),
        renormalize_selected_transport=True,
        batch_size=1,
        device=torch.device("cpu"),
        tokenizer=FakeTokenizer(),
        config=config,
        source_target_vars=("answer_token",),
    )
    assert selected["top_k"] == 2
    assert selected["iia_acc"] == 0.75


def test_selection_and_reporting_ignore_legacy_diagnostics() -> None:
    name, score = selection_metric_from_metrics(
        {
            "iia_acc": 0.2,
            "exact_acc": 0.9,
            "checker_acc": 1.0,
            "diagnostic_alphabet_restricted_acc": 1.0,
        }
    )
    assert (name, score) == ("iia_acc", 0.2)
    with pytest.raises(KeyError):
        selection_metric_from_metrics({"exact_acc": 1.0})
    summary = summarize_method_records(
        [
            {"method": "ot", "iia_acc": 0.2, "exact_acc": 1.0},
            {"method": "ot", "iia_acc": 0.4, "exact_acc": 1.0},
        ]
    )
    assert summary == [{"method": "ot", "iia_acc": pytest.approx(0.3)}]


def test_legacy_cached_verdicts_are_rejected() -> None:
    assert not payload_uses_unified_iia({"results": [{"exact_acc": 1.0}]})
    assert payload_uses_unified_iia(
        {"results": [{"iia_acc": 0.5, "exact_acc": 0.5, "checker_acc": 0.5}]}
    )


def test_all_verdict_bearing_paths_name_iia_not_legacy_accuracy() -> None:
    root = Path(__file__).resolve().parent
    paths = [
        root / "mcqa_experiment" / name
        for name in (
            "ot.py",
            "das.py",
            "bdas.py",
            "support.py",
            "reporting.py",
            "compare_runner.py",
        )
    ] + [
        root / name
        for name in (
            "mcqa_plot_layer.py",
            "mcqa_plot_native_support.py",
            "mcqa_ot_pca_focus.py",
            "mcqa_delta_hierarchical_sweep.py",
            "mcqa_dbm_baselines.py",
        )
    ]
    forbidden = (
        "selection_exact_acc",
        "calibration_exact_acc",
        "holdout_exact_acc",
        "family_weighted_macro_exact_acc",
        "fit_bank_single_site_exact_acc",
        'get("exact_acc"',
        '["exact_acc"]',
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "iia_acc" in source, path
        for token in forbidden:
            assert token not in source, f"{path}: forbidden verdict token {token}"

    dbm_source = (root / "mcqa_experiment" / "dbm.py").read_text(encoding="utf-8")
    assert "das_metrics_from_logits" in dbm_source
