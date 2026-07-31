"""MCQA-specific label extraction and reporting metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .checking import IIA_METRIC_NAME, normalize_answer_symbol
from .data import (
    ALPHABET_LABELS,
    COUNTERFACTUAL_FAMILIES,
    MCQAPairBank,
    canonicalize_target_var,
)


STRUCTURED_SLOT_DIM = 4
STRUCTURED_LABEL_DIM = len(ALPHABET_LABELS)
STRUCTURED_FEATURE_DIM = STRUCTURED_SLOT_DIM + STRUCTURED_LABEL_DIM


def _label_indices_from_texts(texts: list[object]) -> torch.Tensor:
    labels = [normalize_answer_symbol(text) for text in texts]
    if any(label is None for label in labels):
        raise ValueError(f"Expected one ASCII answer symbol A-Z, got {texts!r}")
    return torch.tensor([ALPHABET_LABELS.index(str(label)) for label in labels], dtype=torch.long)


def _base_answer_indices(bank: MCQAPairBank) -> torch.Tensor:
    return _label_indices_from_texts([output["answer"] for output in bank.base_outputs])


def _interchange_answer_indices(bank: MCQAPairBank) -> torch.Tensor:
    return _label_indices_from_texts(list(bank.expected_answer_texts))


def _gather_variant_logits(logits: torch.Tensor, variant_token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, num_classes, num_variants = variant_token_ids.shape
    gathered = torch.gather(
        logits,
        dim=1,
        index=variant_token_ids.to(logits.device).reshape(batch_size, num_classes * num_variants),
    )
    gathered = gathered.reshape(batch_size, num_classes, num_variants)
    return gathered.max(dim=-1).values


def _gather_slot_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    return _gather_variant_logits(logits, bank.symbol_variant_token_ids)


def _gather_label_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    return _gather_variant_logits(logits, bank.alphabet_variant_token_ids)


def structured_output_features(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Return the combined slot-plus-label output feature vector per example."""
    return torch.cat((_gather_slot_logits(logits, bank), _gather_label_logits(logits, bank)), dim=1)


def aggregate_family_features(per_example_features: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Aggregate per-example features into mean feature blocks for each counterfactual family."""
    blocks = []
    feature_dim = int(per_example_features.shape[1])
    for family_name in COUNTERFACTUAL_FAMILIES:
        mask = torch.tensor(
            [str(current_family) == str(family_name) for current_family in bank.counterfactual_family_names],
            device=per_example_features.device,
            dtype=torch.bool,
        )
        if bool(mask.any()):
            block = per_example_features[mask].mean(dim=0)
        else:
            block = torch.zeros(feature_dim, dtype=per_example_features.dtype, device=per_example_features.device)
        blocks.append(block)
    return torch.cat(blocks, dim=0)


def normalize_family_feature_blocks(aggregated_features: torch.Tensor) -> torch.Tensor:
    """Center and L2-normalize each family's slot block and label block separately."""
    if aggregated_features.ndim != 1:
        raise ValueError("normalize_family_feature_blocks expects a 1D aggregated feature vector")
    expected_dim = len(COUNTERFACTUAL_FAMILIES) * STRUCTURED_FEATURE_DIM
    if int(aggregated_features.numel()) != expected_dim:
        raise ValueError(
            f"Expected aggregated feature dim {expected_dim}, got {int(aggregated_features.numel())}"
        )
    normalized_blocks = []
    offset = 0
    for _family_name in COUNTERFACTUAL_FAMILIES:
        family_block = aggregated_features[offset : offset + STRUCTURED_FEATURE_DIM]
        slot_block = family_block[:STRUCTURED_SLOT_DIM]
        label_block = family_block[STRUCTURED_SLOT_DIM:]
        slot_block = slot_block - slot_block.mean()
        label_block = label_block - label_block.mean()
        slot_norm = torch.linalg.vector_norm(slot_block, ord=2)
        label_norm = torch.linalg.vector_norm(label_block, ord=2)
        if float(slot_norm.item()) > 0.0:
            slot_block = slot_block / slot_norm
        if float(label_norm.item()) > 0.0:
            label_block = label_block / label_norm
        normalized_blocks.append(torch.cat((slot_block, label_block), dim=0))
        offset += STRUCTURED_FEATURE_DIM
    return torch.cat(normalized_blocks, dim=0)


def build_family_signature(
    per_example_features: torch.Tensor,
    bank: MCQAPairBank,
    *,
    normalize_blocks: bool = False,
) -> torch.Tensor:
    aggregated = aggregate_family_features(per_example_features, bank)
    return normalize_family_feature_blocks(aggregated) if normalize_blocks else aggregated


def normalize_family_label_feature_blocks(aggregated_features: torch.Tensor) -> torch.Tensor:
    """Center and L2-normalize each family's label-only block separately."""
    if aggregated_features.ndim != 1:
        raise ValueError("normalize_family_label_feature_blocks expects a 1D aggregated feature vector")
    expected_dim = len(COUNTERFACTUAL_FAMILIES) * STRUCTURED_LABEL_DIM
    if int(aggregated_features.numel()) != expected_dim:
        raise ValueError(
            f"Expected aggregated feature dim {expected_dim}, got {int(aggregated_features.numel())}"
        )
    normalized_blocks = []
    offset = 0
    for _family_name in COUNTERFACTUAL_FAMILIES:
        family_block = aggregated_features[offset : offset + STRUCTURED_LABEL_DIM]
        family_block = family_block - family_block.mean()
        label_norm = torch.linalg.vector_norm(family_block, ord=2)
        if float(label_norm.item()) > 0.0:
            family_block = family_block / label_norm
        normalized_blocks.append(family_block)
        offset += STRUCTURED_LABEL_DIM
    return torch.cat(normalized_blocks, dim=0)


def build_family_label_signature(
    per_example_label_features: torch.Tensor,
    bank: MCQAPairBank,
    *,
    normalize_blocks: bool = False,
) -> torch.Tensor:
    aggregated = aggregate_family_features(per_example_label_features, bank)
    return normalize_family_label_feature_blocks(aggregated) if normalize_blocks else aggregated


def gather_variable_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Project full-vocab logits into the task logits for the chosen target variable."""
    if not hasattr(bank, "target_var"):
        return logits
    canonical_target_var = canonicalize_target_var(bank.target_var)
    if canonical_target_var == "answer_pointer":
        # MCQA interventions are evaluated on the base prompt, so answer-index
        # logits must be read from the base prompt's label tokens (typically A-D),
        # not from the source prompt's randomized labels.
        return _gather_variant_logits(logits, bank.symbol_variant_token_ids)
    if canonical_target_var == "answer_token":
        return _gather_variant_logits(logits, bank.alphabet_variant_token_ids)
    raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")


def cross_entropy_for_bank(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Compute supervised cross-entropy for the bank's target variable."""
    target_logits = gather_variable_logits(logits, bank)
    return F.cross_entropy(target_logits.float(), bank.labels.to(target_logits.device))


def _family_iia_metrics(
    correct: torch.Tensor,
    prediction_valid: torch.Tensor,
    expected_valid: torch.Tensor,
    bank: MCQAPairBank,
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    """Return per-family normalized IIA and explicit validity counts."""

    if not hasattr(bank, "counterfactual_family_names"):
        return {}, {}
    family_accs: dict[str, float] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for family_name in COUNTERFACTUAL_FAMILIES:
        mask = torch.tensor(
            [str(current_family) == str(family_name) for current_family in bank.counterfactual_family_names],
            dtype=torch.bool,
        )
        if not bool(mask.any()):
            continue
        family_correct = correct[mask]
        family_prediction_valid = prediction_valid[mask]
        family_expected_valid = expected_valid[mask]
        total = int(mask.sum().item())
        correct_count = int(family_correct.sum().item())
        family_accs[str(family_name)] = float(correct_count / total)
        family_counts[str(family_name)] = {
            "total": total,
            "correct": correct_count,
            "valid_predictions": int(family_prediction_valid.sum().item()),
            "invalid_predictions": int((~family_prediction_valid).sum().item()),
            "valid_expected": int(family_expected_valid.sum().item()),
            "invalid_expected": int((~family_expected_valid).sum().item()),
        }
    return family_accs, family_counts


def cross_entropy_for_das(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Compute DAS training loss on full next-token logits."""
    return F.cross_entropy(logits, bank.answer_token_ids.to(logits.device))


def _decoded_full_vocab_top1(logits: torch.Tensor, tokenizer) -> tuple[torch.Tensor, list[str]]:
    if tokenizer is None:
        raise ValueError("tokenizer is required for normalized full-vocabulary MCQA IIA")
    if logits.ndim != 2:
        raise ValueError(f"Expected [batch, vocab] logits, got shape={tuple(logits.shape)}")
    token_ids = logits.argmax(dim=-1)
    decoded = [tokenizer.decode([int(token_id)]) for token_id in token_ids.detach().cpu().tolist()]
    return token_ids, decoded


def _normalized_iia_state(
    logits: torch.Tensor,
    bank: MCQAPairBank,
    tokenizer,
) -> dict[str, object]:
    predicted_token_ids, predicted_texts = _decoded_full_vocab_top1(logits, tokenizer)
    expected_texts = [str(text) for text in bank.expected_answer_texts]
    if len(expected_texts) != len(predicted_texts):
        raise ValueError(
            f"Expected {len(predicted_texts)} answer texts, got {len(expected_texts)}"
        )
    normalized_predictions = [normalize_answer_symbol(text) for text in predicted_texts]
    normalized_expected = [normalize_answer_symbol(text) for text in expected_texts]
    prediction_valid = torch.tensor(
        [symbol is not None for symbol in normalized_predictions], dtype=torch.bool
    )
    expected_valid = torch.tensor(
        [symbol is not None for symbol in normalized_expected], dtype=torch.bool
    )
    correct = torch.tensor(
        [
            predicted is not None and expected is not None and predicted == expected
            for predicted, expected in zip(normalized_predictions, normalized_expected)
        ],
        dtype=torch.bool,
    )
    return {
        "predicted_token_ids": predicted_token_ids.detach().cpu(),
        "predicted_texts": predicted_texts,
        "expected_texts": expected_texts,
        "normalized_predictions": normalized_predictions,
        "normalized_expected": normalized_expected,
        "prediction_valid": prediction_valid,
        "expected_valid": expected_valid,
        "correct": correct,
    }


def _diagnostic_alphabet_metrics(
    logits: torch.Tensor,
    bank: MCQAPairBank,
    normalized_expected: list[str | None],
) -> dict[str, object]:
    if not hasattr(bank, "alphabet_variant_token_ids"):
        return {}
    alphabet_logits = _gather_label_logits(logits, bank)
    alphabet_predictions = alphabet_logits.argmax(dim=-1).detach().cpu()
    expected_indices = torch.tensor(
        [ALPHABET_LABELS.index(symbol) if symbol in ALPHABET_LABELS else -1 for symbol in normalized_expected],
        dtype=torch.long,
    )
    alphabet_correct = (expected_indices >= 0) & (alphabet_predictions == expected_indices)
    total = int(alphabet_predictions.numel())
    return {
        "diagnostic_alphabet_restricted_acc": (
            float(alphabet_correct.float().mean().item()) if total else 0.0
        ),
        "diagnostic_alphabet_prediction_indices": alphabet_predictions.tolist(),
        "diagnostic_alphabet_prediction_texts": [
            ALPHABET_LABELS[int(index)] for index in alphabet_predictions.tolist()
        ],
    }


def full_vocab_iia_metrics(
    logits: torch.Tensor,
    bank: MCQAPairBank,
    tokenizer,
) -> dict[str, object]:
    """Evaluate normalized agreement of the model's actual full-vocab top-1 token."""

    state = _normalized_iia_state(logits, bank, tokenizer)
    correct = state["correct"]
    prediction_valid = state["prediction_valid"]
    expected_valid = state["expected_valid"]
    assert isinstance(correct, torch.Tensor)
    assert isinstance(prediction_valid, torch.Tensor)
    assert isinstance(expected_valid, torch.Tensor)
    total = int(correct.numel())
    correct_count = int(correct.sum().item())
    iia_acc = float(correct_count / total) if total else 0.0
    family_iia_accs, family_iia_counts = _family_iia_metrics(
        correct, prediction_valid, expected_valid, bank
    )
    metrics: dict[str, object] = {
        "metric_name": IIA_METRIC_NAME,
        "iia_acc": iia_acc,
        "family_iia_accs": family_iia_accs,
        "iia_validity_counts": {
            "total": total,
            "correct": correct_count,
            "valid_predictions": int(prediction_valid.sum().item()),
            "invalid_predictions": int((~prediction_valid).sum().item()),
            "valid_expected": int(expected_valid.sum().item()),
            "invalid_expected": int((~expected_valid).sum().item()),
        },
        "family_iia_counts": family_iia_counts,
        # Compatibility aliases: these are deliberately equal to normalized IIA.
        "exact_acc": iia_acc,
        "family_exact_accs": dict(family_iia_accs),
        "decoded_answer_acc": iia_acc,
        "checker_acc": iia_acc,
    }
    predicted_token_ids = state["predicted_token_ids"]
    assert isinstance(predicted_token_ids, torch.Tensor)
    if hasattr(bank, "answer_token_ids"):
        target_token_ids = bank.answer_token_ids.detach().cpu().to(torch.long)
        metrics["diagnostic_full_vocab_raw_token_acc"] = (
            float((predicted_token_ids == target_token_ids).float().mean().item())
            if total else 0.0
        )
    metrics.update(
        _diagnostic_alphabet_metrics(
            logits, bank, state["normalized_expected"]
        )
    )
    return metrics


def metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Compute normalized full-vocabulary IIA plus explicit diagnostics."""
    return full_vocab_iia_metrics(logits, bank, tokenizer)


def das_metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Compute the same normalized full-vocabulary IIA used by every MCQA method."""
    return full_vocab_iia_metrics(logits, bank, tokenizer)


def prediction_details_from_logits(
    logits: torch.Tensor,
    bank: MCQAPairBank,
    tokenizer=None,
) -> dict[str, object]:
    """Return per-example details for normalized full-vocabulary IIA."""

    state = _normalized_iia_state(logits, bank, tokenizer)
    predicted_token_ids = state["predicted_token_ids"]
    correct = state["correct"]
    prediction_valid = state["prediction_valid"]
    expected_valid = state["expected_valid"]
    assert isinstance(predicted_token_ids, torch.Tensor)
    assert isinstance(correct, torch.Tensor)
    assert isinstance(prediction_valid, torch.Tensor)
    assert isinstance(expected_valid, torch.Tensor)
    details: dict[str, object] = {
        "metric_name": IIA_METRIC_NAME,
        "predictions": predicted_token_ids.tolist(),
        "predicted_token_ids": predicted_token_ids.tolist(),
        "correct": correct.to(torch.int64).tolist(),
        "predicted_text": list(state["predicted_texts"]),
        "normalized_predicted_text": list(state["normalized_predictions"]),
        "expected_answer_texts": list(state["expected_texts"]),
        "normalized_expected_answer_texts": list(state["normalized_expected"]),
        "prediction_valid": prediction_valid.to(torch.int64).tolist(),
        "expected_valid": expected_valid.to(torch.int64).tolist(),
        "base_raw_inputs": [str(item.get("raw_input", "")) for item in getattr(bank, "base_inputs", [])],
        "source_raw_inputs": [str(item.get("raw_input", "")) for item in getattr(bank, "source_inputs", [])],
    }
    if hasattr(bank, "answer_token_ids"):
        target_token_ids = bank.answer_token_ids.detach().cpu().to(torch.long)
        details["labels"] = target_token_ids.tolist()
        details["target_token_ids"] = target_token_ids.tolist()
        details["diagnostic_raw_token_correct"] = (
            predicted_token_ids == target_token_ids
        ).to(torch.int64).tolist()
    if hasattr(bank, "alphabet_variant_token_ids"):
        diagnostic = _diagnostic_alphabet_metrics(
            logits, bank, state["normalized_expected"]
        )
        details.update(diagnostic)
    # Preserve projected task logits only as a named diagnostic.
    details["diagnostic_projected_target_logits"] = (
        gather_variable_logits(logits, bank).detach().cpu().tolist()
    )
    return details


def das_prediction_details_from_logits(
    logits: torch.Tensor,
    bank: MCQAPairBank,
    tokenizer=None,
) -> dict[str, object]:
    """Return the shared normalized full-vocabulary IIA details for DAS-like methods."""
    return prediction_details_from_logits(logits, bank, tokenizer=tokenizer)


def build_variable_signature(bank: MCQAPairBank, signature_mode: str) -> torch.Tensor:
    """Build the abstract-variable signature for one MCQA target variable."""
    canonicalize_target_var(bank.target_var)
    if signature_mode in {"whole_vocab_kl_t1", "whole_vocab_tv_t1"}:
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "label_logit_delta":
        base_answer_indices = _base_answer_indices(bank)
        target_label_indices = _interchange_answer_indices(bank)
        return (
            F.one_hot(target_label_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
        ).reshape(-1)
    if signature_mode == "family_label_delta_norm":
        base_answer_indices = _base_answer_indices(bank)
        target_label_indices = _interchange_answer_indices(bank)
        label_delta = (
            F.one_hot(target_label_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
        )
        return build_family_label_signature(
            label_delta,
            bank,
            normalize_blocks=True,
        )
    raise ValueError(f"Unsupported signature_mode={signature_mode}")
