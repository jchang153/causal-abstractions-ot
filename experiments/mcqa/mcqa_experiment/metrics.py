"""MCQA-specific label extraction and reporting metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .checking import checker_accuracy
from .data import (
    ALPHABET_LABELS,
    COUNTERFACTUAL_FAMILIES,
    MCQAPairBank,
    canonicalize_target_var,
)


STRUCTURED_SLOT_DIM = 4
STRUCTURED_LABEL_DIM = len(ALPHABET_LABELS)
STRUCTURED_FEATURE_DIM = STRUCTURED_SLOT_DIM + STRUCTURED_LABEL_DIM


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


def _family_exact_accs(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    bank: MCQAPairBank,
) -> dict[str, float]:
    if not hasattr(bank, "counterfactual_family_names"):
        return {}
    metrics: dict[str, float] = {}
    for family_name in COUNTERFACTUAL_FAMILIES:
        mask = torch.tensor(
            [str(current_family) == str(family_name) for current_family in bank.counterfactual_family_names],
            device=predictions.device,
            dtype=torch.bool,
        )
        if bool(mask.any()):
            metrics[str(family_name)] = float((predictions[mask] == labels[mask]).float().mean().item())
    return metrics


def cross_entropy_for_das(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Compute DAS training loss on full next-token logits."""
    return F.cross_entropy(logits, bank.answer_token_ids.to(logits.device))


def metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, float]:
    """Compute exact accuracy, family-wise accuracy, and optional decoded answer accuracy."""
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = (
        bank.labels.to(predictions.device)
        if hasattr(bank, "labels")
        else bank.answer_token_ids.to(predictions.device)
    )
    exact_acc = float((predictions == labels).float().mean().item())
    metrics: dict[str, object] = {
        "exact_acc": exact_acc,
        "family_exact_accs": _family_exact_accs(predictions, labels, bank),
    }
    if tokenizer is not None:
        canonical_target_var = canonicalize_target_var(bank.target_var) if hasattr(bank, "target_var") else "answer_token"
        if hasattr(bank, "symbol_token_ids") and hasattr(bank, "alphabet_token_ids"):
            token_bank = bank.symbol_token_ids if canonical_target_var == "answer_pointer" else bank.alphabet_token_ids
            token_predictions = torch.gather(
                token_bank.to(logits.device),
                dim=1,
                index=predictions.view(-1, 1),
            ).view(-1)
        else:
            token_predictions = predictions
        decoded_predictions = [tokenizer.decode([int(token_id)]) for token_id in token_predictions.detach().cpu().tolist()]
        decoded_acc = 0.0
        if decoded_predictions:
            if canonical_target_var == "answer_pointer" and hasattr(bank, "symbol_token_ids"):
                target_token_ids = torch.gather(
                    bank.symbol_token_ids.to(logits.device),
                    dim=1,
                    index=labels.view(-1, 1),
                ).view(-1)
            else:
                target_token_ids = bank.answer_token_ids.to(logits.device)
            decoded_targets = [tokenizer.decode([int(token_id)]) for token_id in target_token_ids.detach().cpu().tolist()]
            decoded_acc = sum(
                int(str(expected).strip() == str(decoded).strip())
                for expected, decoded in zip(decoded_targets, decoded_predictions)
            ) / len(decoded_predictions)
        metrics["decoded_answer_acc"] = float(decoded_acc)
        metrics["checker_acc"] = float(checker_accuracy(decoded_predictions, bank.expected_answer_texts))
    return metrics


def das_metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, float]:
    """Compute DAS metrics directly on full-vocab next-token predictions."""
    predictions = logits.argmax(dim=-1)
    labels = bank.answer_token_ids.to(predictions.device)
    exact_acc = float((predictions == labels).float().mean().item())
    metrics = {"exact_acc": exact_acc}
    if tokenizer is not None:
        decoded_predictions = [tokenizer.decode([int(token_id)]) for token_id in predictions.detach().cpu().tolist()]
        decoded_acc = 0.0
        if decoded_predictions:
            decoded_acc = sum(
                int(str(expected).strip() == str(decoded).strip())
                for expected, decoded in zip(bank.expected_answer_texts, decoded_predictions)
            ) / len(decoded_predictions)
        metrics["decoded_answer_acc"] = float(decoded_acc)
        metrics["checker_acc"] = float(checker_accuracy(decoded_predictions, bank.expected_answer_texts))
    return metrics


def prediction_details_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Return parse-friendly per-example prediction details for one bank."""
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = (
        bank.labels.to(predictions.device)
        if hasattr(bank, "labels")
        else bank.answer_token_ids.to(predictions.device)
    )
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "target_logits": target_logits.detach().cpu().tolist(),
        "base_raw_inputs": [str(item["raw_input"]) for item in bank.base_inputs],
        "source_raw_inputs": [str(item["raw_input"]) for item in bank.source_inputs],
        "expected_answer_texts": list(bank.expected_answer_texts),
    }
    canonical_target_var = canonicalize_target_var(bank.target_var) if hasattr(bank, "target_var") else "answer_token"
    if hasattr(bank, "symbol_token_ids") and hasattr(bank, "alphabet_token_ids"):
        token_bank = bank.symbol_token_ids if canonical_target_var == "answer_pointer" else bank.alphabet_token_ids
        predicted_token_ids = torch.gather(
            token_bank.to(logits.device),
            dim=1,
            index=predictions.view(-1, 1),
        ).view(-1)
    else:
        predicted_token_ids = predictions
    details["predicted_token_ids"] = predicted_token_ids.detach().cpu().tolist()
    if canonical_target_var == "answer_pointer" and hasattr(bank, "symbol_token_ids"):
        target_token_ids = bank.symbol_token_ids.gather(1, labels.view(-1, 1).cpu()).view(-1)
    else:
        target_token_ids = bank.answer_token_ids.cpu()
    details["target_token_ids"] = target_token_ids.detach().cpu().tolist()
    if tokenizer is not None:
        details["predicted_text"] = [
            tokenizer.decode([int(token_id)]) for token_id in predicted_token_ids.detach().cpu().tolist()
        ]
    return details


def das_prediction_details_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Return parse-friendly DAS prediction details from full-vocab logits."""
    predictions = logits.argmax(dim=-1)
    labels = bank.answer_token_ids.to(predictions.device)
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "base_raw_inputs": [str(item["raw_input"]) for item in bank.base_inputs],
        "source_raw_inputs": [str(item["raw_input"]) for item in bank.source_inputs],
        "expected_answer_texts": list(bank.expected_answer_texts),
        "target_token_ids": bank.answer_token_ids.detach().cpu().tolist(),
    }
    if tokenizer is not None:
        details["predicted_text"] = [
            tokenizer.decode([int(token_id)]) for token_id in predictions.detach().cpu().tolist()
        ]
    return details


def build_variable_signature(bank: MCQAPairBank, signature_mode: str) -> torch.Tensor:
    """Build the abstract-variable signature for one MCQA target variable."""
    canonical_target_var = canonicalize_target_var(bank.target_var)
    if signature_mode == "whole_vocab_kl_t1":
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "answer_logit_delta":
        if canonical_target_var == "answer_pointer":
            source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=4).to(torch.float32)
            base_pointer_indices = torch.tensor(
                [int(output["answer_pointer"]) for output in bank.base_outputs],
                dtype=torch.long,
            )
            base_onehot = F.one_hot(base_pointer_indices, num_classes=4).to(torch.float32)
            return (source_onehot - base_onehot).reshape(-1)
        if canonical_target_var == "answer_token":
            source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=26).to(torch.float32)
            base_answer_indices = torch.tensor(
                [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
                dtype=torch.long,
            )
            base_onehot = F.one_hot(base_answer_indices, num_classes=26).to(torch.float32)
            return (source_onehot - base_onehot).reshape(-1)
    if signature_mode in {"family_slot_label_delta", "family_slot_label_delta_norm"}:
        batch_size = bank.size
        slot_delta = torch.zeros((batch_size, STRUCTURED_SLOT_DIM), dtype=torch.float32)
        label_delta = torch.zeros((batch_size, STRUCTURED_LABEL_DIM), dtype=torch.float32)
        base_pointer_indices = torch.tensor(
            [int(output["answer_pointer"]) for output in bank.base_outputs],
            dtype=torch.long,
        )
        base_answer_indices = torch.tensor(
            [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
            dtype=torch.long,
        )
        if canonical_target_var == "answer_pointer":
            source_pointer_indices = torch.tensor(
                [int(output["answer_pointer"]) for output in bank.source_outputs],
                dtype=torch.long,
            )
            slot_delta = (
                F.one_hot(source_pointer_indices, num_classes=STRUCTURED_SLOT_DIM).to(torch.float32)
                - F.one_hot(base_pointer_indices, num_classes=STRUCTURED_SLOT_DIM).to(torch.float32)
            )
            base_symbol_at_source_pointer = torch.tensor(
                [
                    ALPHABET_LABELS.index(str(bank.base_inputs[index][f"symbol{int(source_pointer)}"]).strip())
                    for index, source_pointer in enumerate(source_pointer_indices.tolist())
                ],
                dtype=torch.long,
            )
            label_delta = (
                F.one_hot(base_symbol_at_source_pointer, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
                - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            )
        elif canonical_target_var == "answer_token":
            source_answer_indices = torch.tensor(
                [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.source_outputs],
                dtype=torch.long,
            )
            label_delta = (
                F.one_hot(source_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
                - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            )
        else:
            raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")
        per_example_features = torch.cat((slot_delta, label_delta), dim=1)
        return build_family_signature(
            per_example_features,
            bank,
            normalize_blocks=(signature_mode == "family_slot_label_delta_norm"),
        )
    if signature_mode in {"family_label_delta", "family_label_delta_norm", "family_label_logit_delta", "family_label_logit_delta_norm"}:
        base_answer_indices = torch.tensor(
            [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
            dtype=torch.long,
        )
        if canonical_target_var == "answer_pointer":
            source_pointer_indices = torch.tensor(
                [int(output["answer_pointer"]) for output in bank.source_outputs],
                dtype=torch.long,
            )
            target_label_indices = torch.tensor(
                [
                    ALPHABET_LABELS.index(str(bank.base_inputs[index][f"symbol{int(source_pointer)}"]).strip())
                    for index, source_pointer in enumerate(source_pointer_indices.tolist())
                ],
                dtype=torch.long,
            )
        elif canonical_target_var == "answer_token":
            target_label_indices = torch.tensor(
                [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.source_outputs],
                dtype=torch.long,
            )
        else:
            raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")
        label_delta = (
            F.one_hot(target_label_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
            - F.one_hot(base_answer_indices, num_classes=STRUCTURED_LABEL_DIM).to(torch.float32)
        )
        return build_family_label_signature(
            label_delta,
            bank,
            normalize_blocks=signature_mode in {"family_label_delta_norm", "family_label_logit_delta_norm"},
        )
    raise ValueError(f"Unsupported signature_mode={signature_mode}")
