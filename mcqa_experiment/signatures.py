"""MCQA residual-site effect signatures."""

from __future__ import annotations

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .data import MCQAPairBank
from .intervention import forward_factual_logits, run_soft_site_intervention
from .metrics import (
    _gather_label_logits,
    build_family_label_signature,
    build_family_signature,
    gather_variable_logits,
    structured_output_features,
)
from .pca import LayerPCABasis
from .sites import SiteLike


def _per_example_kl(counterfactual_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    counterfactual_log_probs = torch.log_softmax(counterfactual_logits, dim=-1)
    counterfactual_probs = counterfactual_log_probs.exp()
    return torch.sum(counterfactual_probs * (counterfactual_log_probs - base_log_probs), dim=-1)


def signature_from_logits(
    *,
    counterfactual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bank: MCQAPairBank,
    signature_mode: str,
) -> torch.Tensor:
    """Convert factual/counterfactual logits into one site-effect signature."""
    if signature_mode == "whole_vocab_kl_t1":
        return _per_example_kl(counterfactual_logits, base_logits).reshape(-1)
    if signature_mode == "answer_logit_delta":
        counterfactual_target_logits = gather_variable_logits(counterfactual_logits, bank)
        base_target_logits = gather_variable_logits(base_logits, bank)
        return (counterfactual_target_logits - base_target_logits).reshape(-1)
    if signature_mode in {"family_slot_label_delta", "family_slot_label_delta_norm"}:
        counterfactual_features = structured_output_features(counterfactual_logits, bank)
        base_features = structured_output_features(base_logits, bank)
        return build_family_signature(
            counterfactual_features - base_features,
            bank,
            normalize_blocks=(signature_mode == "family_slot_label_delta_norm"),
        )
    if signature_mode in {"family_label_delta", "family_label_delta_norm", "family_label_logit_delta", "family_label_logit_delta_norm"}:
        counterfactual_features = _gather_label_logits(counterfactual_logits, bank)
        base_features = _gather_label_logits(base_logits, bank)
        return build_family_label_signature(
            counterfactual_features - base_features,
            bank,
            normalize_blocks=signature_mode in {"family_label_delta_norm", "family_label_logit_delta_norm"},
        )
    raise ValueError(f"Unsupported signature_mode={signature_mode}")


def collect_base_logits(
    *,
    model,
    bank: MCQAPairBank,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect factual last-token logits for a bank."""
    outputs = []
    with torch.no_grad():
        for start in range(0, bank.size, batch_size):
            end = min(start + batch_size, bank.size)
            logits = forward_factual_logits(
                model=model,
                input_ids=bank.base_input_ids[start:end].to(device),
                attention_mask=bank.base_attention_mask[start:end].to(device),
            )
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def collect_site_signatures(
    *,
    model,
    bank: MCQAPairBank,
    sites: list[SiteLike],
    base_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
    signature_mode: str,
    show_progress: bool = False,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> torch.Tensor:
    """Measure each residual site's intervention effect signature."""
    signatures = []
    site_iterator = sites
    target_var = str(bank.target_var)
    if show_progress and tqdm is not None:
        site_iterator = tqdm(
            sites,
            desc=f"Site signatures ({target_var}, {signature_mode}, {len(sites)} sites)",
            leave=False,
        )
    with torch.no_grad():
        for site in site_iterator:
            site_logits_chunks = []
            for start in range(0, bank.size, batch_size):
                end = min(start + batch_size, bank.size)
                logits = run_soft_site_intervention(
                    model=model,
                    base_input_ids=bank.base_input_ids[start:end].to(device),
                    base_attention_mask=bank.base_attention_mask[start:end].to(device),
                    source_input_ids=bank.source_input_ids[start:end].to(device),
                    source_attention_mask=bank.source_attention_mask[start:end].to(device),
                    site_weights={site: 1.0},
                    strength=1.0,
                    base_position_by_id={
                        key: value[start:end] for key, value in bank.base_position_by_id.items()
                    },
                    source_position_by_id={
                        key: value[start:end] for key, value in bank.source_position_by_id.items()
                    },
                    pca_bases_by_id=pca_bases_by_id,
                )
                site_logits_chunks.append(logits.detach().cpu())
            site_logits = torch.cat(site_logits_chunks, dim=0)
            signatures.append(
                signature_from_logits(
                    counterfactual_logits=site_logits,
                    base_logits=base_logits,
                    bank=bank,
                    signature_mode=signature_mode,
                )
            )
    return torch.stack(signatures, dim=0)


def collect_multi_variable_site_signatures(
    *,
    model,
    banks_by_var: dict[str, MCQAPairBank],
    sites: list[SiteLike],
    base_logits_by_var: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    signature_mode: str,
    show_progress: bool = False,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> dict[str, torch.Tensor]:
    """Collect site signatures separately for each abstract variable."""
    signatures_by_var: dict[str, torch.Tensor] = {}
    total_targets = len(banks_by_var)
    for index, (target_var, bank) in enumerate(banks_by_var.items(), start=1):
        if show_progress:
            print(
                f"[OT prep] collecting site signatures target={target_var} "
                f"pass={index}/{total_targets} sites={len(sites)} split={bank.split}"
            )
        signatures_by_var[target_var] = collect_site_signatures(
            model=model,
            bank=bank,
            sites=sites,
            base_logits=base_logits_by_var[target_var],
            batch_size=batch_size,
            device=device,
            signature_mode=signature_mode,
            show_progress=show_progress,
            pca_bases_by_id=pca_bases_by_id,
        )
    return signatures_by_var
