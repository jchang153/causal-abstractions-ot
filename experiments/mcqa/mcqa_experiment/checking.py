"""Decoded-text checkers and selection helpers for MCQA evaluation."""

from __future__ import annotations

from typing import Sequence
import unicodedata


IIA_METRIC_NAME = "normalized_full_vocab_top1_v1"
POOLED_CALIBRATION_METRIC = "iia_acc"
_RECORDED_CALIBRATION_KINDS = {
    "mcqa_plot_layer",
    "mcqa_plot_native_support",
    "mcqa_plot_native_support_layer",
    "mcqa_ot_pca_focus",
    "mcqa_ot_pca_focus_epsilon",
    "mcqa_plot_pca_support_layer",
}


def normalize_answer_symbol(value: object) -> str | None:
    """Normalize one decoded answer token to exactly one ASCII symbol A-Z.

    Compatibility characters are normalized with NFKC, surrounding whitespace
    is removed, and case variants are folded to uppercase. Empty strings,
    multi-symbol outputs, punctuation, and non-ASCII symbols are invalid.
    """

    text = unicodedata.normalize("NFKC", str(value)).strip().casefold().upper()
    if len(text) != 1 or not ("A" <= text <= "Z"):
        return None
    return text


def normalized_answer_checker(neural_output: object, causal_output: object) -> bool:
    """Return strict normalized agreement for two single-symbol outputs."""

    neural_symbol = normalize_answer_symbol(neural_output)
    causal_symbol = normalize_answer_symbol(causal_output)
    return neural_symbol is not None and causal_symbol is not None and neural_symbol == causal_symbol


def causalab_substring_checker(neural_output: object, causal_output: object) -> bool:
    """Legacy factual-filter checker; never use this for IIA or selection.

    The source dataset filter historically accepted bidirectional substrings.
    It remains isolated here so this metric-only change does not alter dataset
    membership. Verdict-bearing evaluation uses ``normalized_answer_checker``.
    """

    neural_text = str(neural_output)
    causal_text = str(causal_output)
    return causal_text in neural_text or neural_text in causal_text


def checker_accuracy(predicted_texts: Sequence[object], expected_texts: Sequence[object]) -> float:
    """Average strict normalized single-symbol agreement."""

    total = len(expected_texts)
    if total == 0:
        return 0.0
    correct = sum(
        int(normalized_answer_checker(predicted, expected))
        for predicted, expected in zip(predicted_texts, expected_texts)
    )
    return float(correct) / float(total)


def iia_acc_from_metrics(metrics: dict[str, object]) -> float:
    """Return the required normalized full-vocabulary IIA scalar."""

    if "iia_acc" not in metrics:
        raise KeyError("verdict-bearing MCQA payload is missing required iia_acc")
    return float(metrics["iia_acc"])


def selection_metric_from_metrics(metrics: dict[str, object]) -> tuple[str, float]:
    """Return the only scalar permitted for MCQA calibration selection."""

    return POOLED_CALIBRATION_METRIC, iia_acc_from_metrics(metrics)


def require_pooled_calibration_metric(calibration_metric: object) -> None:
    """Reject family-balanced or otherwise non-pooled MCQA objectives."""

    if str(calibration_metric) != POOLED_CALIBRATION_METRIC:
        raise ValueError(
            "MCQA calibration must use pooled iia_acc across examples; "
            f"got calibration_metric={calibration_metric!r}"
        )


def payload_uses_unified_iia(payload: object) -> bool:
    """Reject cached verdict payloads that predate normalized full-vocab IIA."""

    if isinstance(payload, dict):
        legacy_keys = {"exact_acc", "checker_acc", "family_exact_accs"}
        if legacy_keys.intersection(payload) and "iia_acc" not in payload:
            return False
        return all(payload_uses_unified_iia(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(payload_uses_unified_iia(value) for value in payload)
    return True


def payload_uses_pooled_iia_calibration(payload: object) -> bool:
    """Reject cached outputs that record a non-pooled calibration objective."""

    if isinstance(payload, dict):
        if str(payload.get("kind", "")) in _RECORDED_CALIBRATION_KINDS:
            if str(payload.get("calibration_metric", "")) != POOLED_CALIBRATION_METRIC:
                return False
        if "calibration_metric" in payload:
            if str(payload["calibration_metric"]) != POOLED_CALIBRATION_METRIC:
                return False
        return all(payload_uses_pooled_iia_calibration(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(payload_uses_pooled_iia_calibration(value) for value in payload)
    return True
