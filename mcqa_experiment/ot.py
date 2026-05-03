"""Transport-based alignment and intervention for MCQA residual-stream sites."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from . import _env  # noqa: F401
from .data import COUNTERFACTUAL_FAMILIES, MCQAPairBank, canonicalize_target_var
from .intervention import run_soft_site_intervention
from .metrics import build_variable_signature, metrics_from_logits, prediction_details_from_logits
from .pca import LayerPCABasis
from .signatures import collect_base_logits, collect_multi_variable_site_signatures, collect_site_signatures
from .sites import SiteLike


def _synchronize_if_cuda(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _shared_site_signature_mode(signature_mode: str) -> bool:
    return signature_mode in {
        "whole_vocab_kl_t1",
        "family_slot_label_delta",
        "family_slot_label_delta_norm",
        "family_label_delta",
        "family_label_delta_norm",
        "family_label_logit_delta",
        "family_label_logit_delta_norm",
    }


def _squared_euclidean_cost(u_points: torch.Tensor, v_points: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean transport costs between two point clouds."""
    u = u_points.to(dtype=torch.float32)
    v = v_points.to(dtype=torch.float32)
    return torch.cdist(u, v, p=2).pow(2)


def _balanced_marginal_error(pi: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute the max absolute marginal violation for a balanced transport plan."""
    row_error = torch.max(torch.abs(pi.sum(dim=1) - a))
    col_error = torch.max(torch.abs(pi.sum(dim=0) - b))
    return float(torch.maximum(row_error, col_error).item())


def _transport_validation_stats(
    transport: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> dict[str, float]:
    """Summarize mass and marginal residuals for a candidate balanced transport."""
    transport = np.asarray(transport, dtype=float)
    row_residual = float(np.max(np.abs(transport.sum(axis=1) - p))) if transport.size else float("inf")
    col_residual = float(np.max(np.abs(transport.sum(axis=0) - q))) if transport.size else float("inf")
    total_mass = float(transport.sum())
    return {
        "matched_mass": total_mass,
        "max_row_residual": row_residual,
        "max_col_residual": col_residual,
    }


def _is_valid_balanced_transport(
    transport: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tol: float,
) -> bool:
    """Return True only for finite, nonnegative, mass-preserving balanced couplings."""
    transport = np.asarray(transport, dtype=float)
    if transport.shape != (p.shape[0], q.shape[0]):
        return False
    if not np.isfinite(transport).all():
        return False
    if np.any(transport < 0.0):
        return False
    stats = _transport_validation_stats(transport, p, q)
    residual_tol = max(float(tol), 1e-6)
    mass_tol = max(float(tol), 1e-6)
    return (
        abs(stats["matched_mass"] - 1.0) <= mass_tol
        and stats["max_row_residual"] <= residual_tol
        and stats["max_col_residual"] <= residual_tol
    )


def sinkhorn_uniform_ot(
    u_points: torch.Tensor,
    v_points: torch.Tensor,
    epsilon: float,
    n_iter: int,
    temperature: float = 1.0,
    tol: float = 1e-9,
) -> tuple[torch.Tensor, float]:
    """Entropic OT with uniform marginals and squared Euclidean cost."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    if tol < 0:
        raise ValueError("tol must be >= 0")

    u = u_points.to(dtype=torch.float32)
    v = v_points.to(dtype=torch.float32)
    m, n = u.size(0), v.size(0)

    a = torch.full((m,), 1.0 / m, dtype=torch.float32, device=u.device)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32, device=v.device)
    cost = _squared_euclidean_cost(u, v)
    kernel = torch.exp(-cost / (epsilon * temperature)).clamp_min(1e-30)

    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(n_iter):
        kr = kernel @ c
        r = a / kr.clamp_min(1e-30)
        kt = kernel.transpose(0, 1) @ r
        c = b / kt.clamp_min(1e-30)
        pi = r[:, None] * kernel * c[None, :]
        if _balanced_marginal_error(pi, a, b) <= float(tol):
            break

    pi = r[:, None] * kernel * c[None, :]
    ot_cost = float((pi * cost).sum().item())
    return pi, ot_cost


def sinkhorn_unbalanced_ot(
    u_points: torch.Tensor,
    v_points: torch.Tensor,
    epsilon: float,
    n_iter: int,
    temperature: float = 1.0,
    tau_neural: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One-sided entropic UOT with hard abstract marginal and soft neural marginal."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    if tau_neural <= 0:
        raise ValueError("tau_neural must be > 0")

    u = u_points.to(dtype=torch.float32)
    v = v_points.to(dtype=torch.float32)
    m, n = u.size(0), v.size(0)

    a = torch.full((m,), 1.0 / m, dtype=torch.float32, device=u.device)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32, device=v.device)
    cost = _squared_euclidean_cost(u, v)
    kernel = torch.exp(-cost / (epsilon * temperature)).clamp_min(1e-30)

    rho_b = float(tau_neural / (tau_neural + epsilon))

    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(n_iter):
        kr = kernel @ c
        r = a / kr.clamp_min(1e-30)
        kt = kernel.transpose(0, 1) @ r
        c = (b / kt.clamp_min(1e-30)).pow(rho_b)

    pi = r[:, None] * kernel * c[None, :]
    pi_row = pi.sum(dim=1)
    pi_col = pi.sum(dim=0)
    transport_cost = float((pi * cost).sum().item())
    kl_row = float(
        (
            pi_row * torch.log(pi_row.clamp_min(1e-30) / a.clamp_min(1e-30))
            - pi_row
            + a
        ).sum().item()
    )
    kl_col = float(
        (
            pi_col * torch.log(pi_col.clamp_min(1e-30) / b.clamp_min(1e-30))
            - pi_col
            + b
        ).sum().item()
    )
    total_obj = transport_cost + float(tau_neural) * kl_col
    return pi, {
        "transport_cost": transport_cost,
        "kl_abstract": kl_row,
        "kl_neural": kl_col,
        "estimated_cost": total_obj,
        "matched_mass": float(pi.sum().item()),
    }


@dataclass(frozen=True)
class OTConfig:
    """Hyperparameters for OT/UOT alignment and intervention runs."""

    method: str = "ot"
    batch_size: int = 16
    epsilon: float = 1.0
    uot_beta_neural: float = 1.0
    max_iter: int = 500
    tol: float = 1e-9
    signature_mode: str = "family_slot_label_delta"
    top_k_values: tuple[int, ...] | None = None
    lambda_values: tuple[float, ...] = (1.0,)
    selection_verbose: bool = True
    source_target_vars: tuple[str, ...] = ("answer_pointer", "answer_token")
    calibration_metric: str = "exact_acc"
    calibration_family_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    top_k_values_by_var: dict[str, tuple[int, ...]] | None = None
    lambda_values_by_var: dict[str, tuple[float, ...]] | None = None


def load_prepared_alignment_artifacts(
    cache_path: str | Path,
    *,
    expected_spec: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Load cached MCQA signature artifacts when the on-disk spec matches."""
    path = Path(cache_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    cached_spec = payload.get("cache_spec")
    if expected_spec is not None and cached_spec != expected_spec:
        return None
    base_logits_by_var = payload.get("base_logits_by_var")
    site_signatures_by_var = payload.get("site_signatures_by_var")
    if not isinstance(base_logits_by_var, dict) or not isinstance(site_signatures_by_var, dict):
        legacy_base_logits = payload.get("base_logits")
        legacy_site_signatures = payload.get("site_signatures")
        if isinstance(legacy_base_logits, torch.Tensor) and isinstance(legacy_site_signatures, torch.Tensor):
            target_var = "answer_token"
            if isinstance(cached_spec, dict):
                source_target_vars = cached_spec.get("source_target_vars")
                if isinstance(source_target_vars, list) and source_target_vars:
                    target_var = canonicalize_target_var(str(source_target_vars[0]))
            base_logits_by_var = {target_var: legacy_base_logits.detach().cpu()}
            site_signatures_by_var = {target_var: legacy_site_signatures.detach().cpu()}
        else:
            return None
    normalized_base_logits = {}
    normalized_site_signatures = {}
    for target_var, tensor in base_logits_by_var.items():
        if not isinstance(tensor, torch.Tensor):
            return None
        normalized_base_logits[canonicalize_target_var(str(target_var))] = tensor.detach().cpu()
    for target_var, tensor in site_signatures_by_var.items():
        if not isinstance(tensor, torch.Tensor):
            return None
        normalized_site_signatures[canonicalize_target_var(str(target_var))] = tensor.detach().cpu()
    return {
        "base_logits_by_var": normalized_base_logits,
        "site_signatures_by_var": normalized_site_signatures,
        "prepare_runtime_seconds": float(payload.get("prepare_runtime_seconds", 0.0)),
        "cache_spec": cached_spec,
        "cache_path": str(path),
        "loaded_from_disk": True,
        "artifact_cache_hit": True,
    }


def save_prepared_alignment_artifacts(
    cache_path: str | Path,
    *,
    prepared_artifacts: dict[str, object],
    cache_spec: dict[str, object],
) -> None:
    """Persist reusable MCQA signature artifacts for future epsilon sweeps and reruns."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_spec": cache_spec,
            "prepare_runtime_seconds": float(prepared_artifacts.get("prepare_runtime_seconds", 0.0)),
            "base_logits_by_var": {
                target_var: tensor.detach().cpu()
                for target_var, tensor in prepared_artifacts["base_logits_by_var"].items()
            },
            "site_signatures_by_var": {
                target_var: tensor.detach().cpu()
                for target_var, tensor in prepared_artifacts["site_signatures_by_var"].items()
            },
            "saved_with": "mcqa_signature_cache_v2",
            "saved_spec_json": json.dumps(cache_spec, sort_keys=True),
        },
        path,
    )


def prepare_alignment_artifacts(
    *,
    model,
    fit_banks_by_var: dict[str, MCQAPairBank],
    sites: list[SiteLike],
    device: torch.device | str,
    config: OTConfig,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> dict[str, object]:
    """Build reusable factual logits and neural site signatures for OT/UOT runs."""
    device = torch.device(device)
    start = perf_counter()
    target_vars = list(fit_banks_by_var.keys())
    if config.selection_verbose:
        print(
            f"[OT prep] start signature_mode={config.signature_mode} "
            f"candidate_sites={len(sites)} target_vars={target_vars}"
        )
    if _shared_site_signature_mode(config.signature_mode):
        reference_target_var = str(target_vars[0])
        reference_bank = fit_banks_by_var[reference_target_var]
        if config.selection_verbose:
            print(
                f"[OT prep] using shared site signatures across targets; "
                f"reference_target={reference_target_var} split={reference_bank.split} examples={reference_bank.size}"
            )
            print(
                "[OT prep] rationale: the neural site signature depends only on the shared training rows "
                f"for signature_mode={config.signature_mode}, while only the abstract variable signature is target-specific."
            )
            print(
                f"[OT prep] collecting shared base logits target={reference_target_var} "
                f"split={reference_bank.split} examples={reference_bank.size}"
            )
        shared_base_logits = collect_base_logits(
            model=model,
            bank=reference_bank,
            batch_size=config.batch_size,
            device=device,
        )
        if config.selection_verbose:
            print(
                f"[OT prep] collecting one shared site-signature pass "
                f"target={reference_target_var} sites={len(sites)}"
            )
        shared_site_signatures = collect_site_signatures(
            model=model,
            bank=reference_bank,
            sites=sites,
            base_logits=shared_base_logits,
            batch_size=config.batch_size,
            device=device,
            signature_mode=config.signature_mode,
            show_progress=config.selection_verbose,
            pca_bases_by_id=pca_bases_by_id,
        )
        base_logits_by_var = {
            str(target_var): shared_base_logits
            for target_var in target_vars
        }
        site_signatures_by_var = {
            str(target_var): shared_site_signatures
            for target_var in target_vars
        }
    else:
        if config.selection_verbose:
            print(
                "[OT prep] signature mode requires target-specific site signatures; "
                "collecting separate passes per target variable"
            )
        base_logits_by_var = {}
        for target_var, bank in fit_banks_by_var.items():
            if config.selection_verbose:
                print(
                    f"[OT prep] collecting base logits target={target_var} "
                    f"split={bank.split} examples={bank.size}"
                )
            base_logits_by_var[target_var] = collect_base_logits(
                model=model,
                bank=bank,
                batch_size=config.batch_size,
                device=device,
            )
        if config.selection_verbose:
            print("[OT prep] collecting target-specific site signatures")
        site_signatures_by_var = collect_multi_variable_site_signatures(
            model=model,
            banks_by_var=fit_banks_by_var,
            sites=sites,
            base_logits_by_var=base_logits_by_var,
            batch_size=config.batch_size,
            device=device,
            signature_mode=config.signature_mode,
            show_progress=config.selection_verbose,
            pca_bases_by_id=pca_bases_by_id,
        )
    if config.selection_verbose:
        print(
            f"[OT prep] complete target_vars={list(site_signatures_by_var.keys())} "
            f"runtime={float(perf_counter() - start):.2f}s"
        )
    return {
        "base_logits_by_var": base_logits_by_var,
        "site_signatures_by_var": site_signatures_by_var,
        "prepare_runtime_seconds": float(perf_counter() - start),
        "loaded_from_disk": False,
        "artifact_cache_hit": False,
    }


def build_rankings(
    transport: np.ndarray,
    sites: list[SiteLike],
    ranking_k: int,
    source_target_vars: tuple[str, ...],
) -> list[dict[str, object]]:
    site_scores = transport.max(axis=0)
    dominant_source = transport.argmax(axis=0)
    order = np.argsort(-site_scores, kind="stable")[: int(ranking_k)]
    return [
        {
            "site_index": int(site_index),
            "site_label": sites[int(site_index)].label,
            "layer": int(sites[int(site_index)].layer),
            "token_position_id": str(sites[int(site_index)].token_position_id),
            "dim_start": int(sites[int(site_index)].dim_start),
            "dim_end": int(sites[int(site_index)].dim_end),
            "transport_mass": float(site_scores[int(site_index)]),
            "dominant_source_index": int(dominant_source[int(site_index)]),
            "dominant_source_var": str(source_target_vars[int(dominant_source[int(site_index)])]),
        }
        for site_index in order
    ]


def normalize_transport_rows(transport: np.ndarray) -> np.ndarray:
    row_sums = transport.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return transport / safe_row_sums


def _select_transport_row_for_target(
    transport: np.ndarray,
    source_target_vars: tuple[str, ...],
    target_var: str,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    """Select the transport row corresponding to the evaluated abstract variable."""
    try:
        row_index = tuple(str(variable) for variable in source_target_vars).index(str(target_var))
    except ValueError as exc:
        raise ValueError(
            f"Target variable {target_var!r} is not present in source_target_vars={list(source_target_vars)}"
        ) from exc
    return transport[row_index : row_index + 1], (str(source_target_vars[row_index]),), int(row_index)


def _selection_transport_for_target(
    *,
    method: str,
    target_transport: np.ndarray,
    target_normalized_transport: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return the transport used for calibration/intervention and whether to renormalize after truncation."""
    if method == "uot":
        # Preserve unmatched-mass behavior: if neural mass backs off under UOT,
        # the intervention should see that reduced mass rather than a renormalized proxy.
        return target_transport, False
    return target_normalized_transport, True


def truncate_transport_rows(normalized_transport: np.ndarray, top_k: int, renormalize: bool = False) -> np.ndarray:
    truncated = np.zeros_like(normalized_transport)
    limit = max(1, min(int(top_k), normalized_transport.shape[1]))
    site_scores = normalized_transport.max(axis=0)
    dominant_source = normalized_transport.argmax(axis=0)
    order = np.argsort(-site_scores, kind="stable")[:limit]
    for site_index in order:
        row_index = int(dominant_source[site_index])
        truncated[row_index, site_index] = normalized_transport[row_index, site_index]
    if renormalize:
        row_sums = truncated.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        truncated = truncated / safe_row_sums
    return truncated


def _stack_cost_matrix(
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures_by_var: dict[str, torch.Tensor],
    source_target_vars: tuple[str, ...],
) -> torch.Tensor:
    rows = []
    for target_var in source_target_vars:
        variable_signature = variable_signatures_by_var[target_var].reshape(1, -1)
        site_signatures = site_signatures_by_var[target_var].reshape(site_signatures_by_var[target_var].shape[0], -1)
        rows.append(_squared_euclidean_cost(variable_signature, site_signatures))
    return torch.cat(rows, dim=0)


def sinkhorn_from_cost_matrix(
    cost: torch.Tensor,
    *,
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float,
    n_iter: int,
    tol: float = 1e-9,
) -> tuple[torch.Tensor, float]:
    kernel = torch.exp(-cost.to(torch.float32) / float(epsilon)).clamp_min(1e-30)
    r = torch.ones_like(p)
    c = torch.ones_like(q)
    for _ in range(int(n_iter)):
        r = p / (kernel @ c).clamp_min(1e-30)
        c = q / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)
        pi = r[:, None] * kernel * c[None, :]
        if _balanced_marginal_error(pi, p, q) <= float(tol):
            break
    pi = r[:, None] * kernel * c[None, :]
    return pi, float((pi * cost).sum().item())


def sinkhorn_unbalanced_from_cost_matrix(
    cost: torch.Tensor,
    *,
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float,
    n_iter: int,
    tau_neural: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    kernel = torch.exp(-cost.to(torch.float32) / float(epsilon)).clamp_min(1e-30)
    rho_b = float(tau_neural / (tau_neural + epsilon))
    r = torch.ones_like(p)
    c = torch.ones_like(q)
    for _ in range(int(n_iter)):
        r = p / (kernel @ c).clamp_min(1e-30)
        c = (q / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)).pow(rho_b)
    pi = r[:, None] * kernel * c[None, :]
    pi_row = pi.sum(dim=1)
    pi_col = pi.sum(dim=0)
    transport_cost = float((pi * cost).sum().item())
    kl_row = float((pi_row * torch.log(pi_row.clamp_min(1e-30) / p.clamp_min(1e-30)) - pi_row + p).sum().item())
    kl_col = float((pi_col * torch.log(pi_col.clamp_min(1e-30) / q.clamp_min(1e-30)) - pi_col + q).sum().item())
    total_obj = transport_cost + float(tau_neural) * kl_col
    return pi, {
        "transport_cost": transport_cost,
        "kl_abstract": kl_row,
        "kl_neural": kl_col,
        "estimated_cost": total_obj,
        "matched_mass": float(pi.sum().item()),
    }


def solve_ot_transport(
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures_by_var: dict[str, torch.Tensor],
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    cost_cross = _stack_cost_matrix(variable_signatures_by_var, site_signatures_by_var, config.source_target_vars)
    m, n = cost_cross.shape
    p = torch.full((m,), 1.0 / m, dtype=torch.float32, device=cost_cross.device)
    q = torch.full((n,), 1.0 / n, dtype=torch.float32, device=cost_cross.device)
    transport_tensor, transport_cost = sinkhorn_from_cost_matrix(
        cost_cross,
        p=p,
        q=q,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        tol=float(config.tol),
    )
    transport = transport_tensor.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    meta = {
        "method": "ot",
        "regularization_used": float(config.epsilon),
        "epsilon_config": float(config.epsilon),
        "transport_cost": float(transport_cost),
        **_transport_validation_stats(transport, p_np, q_np),
    }
    if _is_valid_balanced_transport(transport, p_np, q_np, float(config.tol)):
        return transport, meta
    meta.update({"failed": True, "failure_reason": "invalid_balanced_transport"})
    return transport, meta


def solve_uot_transport(
    variable_signatures_by_var: dict[str, torch.Tensor],
    site_signatures_by_var: dict[str, torch.Tensor],
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    cost_cross = _stack_cost_matrix(variable_signatures_by_var, site_signatures_by_var, config.source_target_vars)
    m, n = cost_cross.shape
    p = torch.full((m,), 1.0 / m, dtype=torch.float32, device=cost_cross.device)
    q = torch.full((n,), 1.0 / n, dtype=torch.float32, device=cost_cross.device)
    transport_tensor, info = sinkhorn_unbalanced_from_cost_matrix(
        cost_cross,
        p=p,
        q=q,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        tau_neural=float(config.uot_beta_neural),
    )
    transport = transport_tensor.detach().cpu().numpy()
    meta = {
        "method": "uot",
        "regularization_used": float(config.epsilon),
        "uot_beta_neural": float(config.uot_beta_neural),
        "epsilon_config": float(config.epsilon),
        **info,
    }
    if np.isfinite(transport).all() and float(np.sum(transport)) > 0.0:
        return transport, meta
    meta.update({"failed": True, "failure_reason": "invalid_unbalanced_transport"})
    return transport, meta


def _site_weights_from_transport(selected_transport: np.ndarray, sites: list[SiteLike]) -> dict[SiteLike, float]:
    column_mass = selected_transport.sum(axis=0)
    return {
        sites[index]: float(column_mass[index])
        for index in range(selected_transport.shape[1])
        if float(column_mass[index]) > 0.0
    }


def _site_ranking_record(site: SiteLike, *, site_index: int, target_var: str) -> dict[str, object]:
    return {
        "site_index": int(site_index),
        "site_label": site.label,
        "layer": int(site.layer),
        "token_position_id": str(site.token_position_id),
        "dim_start": int(site.dim_start),
        "dim_end": int(site.dim_end),
        "transport_mass": 1.0,
        "dominant_source_index": 0,
        "dominant_source_var": str(target_var),
    }


def _position_mass_from_transport(transport: np.ndarray, sites: list[SiteLike]) -> dict[str, float]:
    column_mass = transport.sum(axis=0)
    position_mass: dict[str, float] = {}
    for site, mass in zip(sites, column_mass.tolist()):
        if float(mass) <= 0.0:
            continue
        position_mass[str(site.token_position_id)] = position_mass.get(str(site.token_position_id), 0.0) + float(mass)
    return position_mass


def _resolve_calibration_grids(
    *,
    target_var: str,
    config: OTConfig,
    num_sites: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    top_k_values = None
    if config.top_k_values_by_var is not None:
        top_k_values = config.top_k_values_by_var.get(str(target_var))
    if top_k_values is None:
        top_k_values = tuple(range(1, num_sites + 1)) if config.top_k_values is None else tuple(config.top_k_values)
    lambda_values = None
    if config.lambda_values_by_var is not None:
        lambda_values = config.lambda_values_by_var.get(str(target_var))
    if lambda_values is None:
        lambda_values = tuple(config.lambda_values)
    return tuple(int(value) for value in top_k_values), tuple(float(value) for value in lambda_values)


def _calibration_score_from_result(result: dict[str, object], config: OTConfig) -> float:
    exact_acc = float(result.get("exact_acc", 0.0))
    if config.calibration_metric == "exact_acc":
        return exact_acc
    if config.calibration_metric == "family_weighted_macro_exact_acc":
        family_exact_accs = result.get("family_exact_accs", {})
        if not isinstance(family_exact_accs, dict):
            return exact_acc
        weighted_sum = 0.0
        total_weight = 0.0
        for family_name, weight in zip(COUNTERFACTUAL_FAMILIES, config.calibration_family_weights):
            if family_name not in family_exact_accs:
                continue
            weighted_sum += float(weight) * float(family_exact_accs[family_name])
            total_weight += float(weight)
        return exact_acc if total_weight <= 0.0 else float(weighted_sum / total_weight)
    raise ValueError(f"Unsupported calibration_metric={config.calibration_metric}")


def _evaluate_soft_intervention(
    *,
    model,
    bank: MCQAPairBank,
    sites: list[SiteLike],
    selected_transport: np.ndarray,
    top_k: int,
    strength: float,
    batch_size: int,
    device: torch.device,
    tokenizer,
    source_target_vars: tuple[str, ...],
    include_details: bool = False,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    site_weights = _site_weights_from_transport(selected_transport, sites)
    logits_chunks = []
    for start in range(0, bank.size, batch_size):
        end = min(start + batch_size, bank.size)
        logits = run_soft_site_intervention(
            model=model,
            base_input_ids=bank.base_input_ids[start:end].to(device),
            base_attention_mask=bank.base_attention_mask[start:end].to(device),
            source_input_ids=bank.source_input_ids[start:end].to(device),
            source_attention_mask=bank.source_attention_mask[start:end].to(device),
            site_weights=site_weights,
            strength=strength,
            base_position_by_id={
                key: value[start:end] for key, value in bank.base_position_by_id.items()
            },
            source_position_by_id={
                key: value[start:end] for key, value in bank.source_position_by_id.items()
            },
            pca_bases_by_id=pca_bases_by_id,
        )
        logits_chunks.append(logits.detach().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    ranking = build_rankings(selected_transport, sites, ranking_k=max(1, top_k), source_target_vars=source_target_vars)
    record = {
        "method": "soft_transport",
        "variable": bank.target_var,
        "split": bank.split,
        "site_label": f"soft:k{int(top_k)},l{float(strength):g}",
        "top_k": int(top_k),
        "lambda": float(strength),
        "top_site_label": ranking[0]["site_label"] if ranking else None,
        "selected_site_labels": [site.label for site in site_weights],
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if include_details:
        record["prediction_details"] = prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    return record, ranking


def _evaluate_single_site_intervention(
    *,
    model,
    bank: MCQAPairBank,
    site: SiteLike,
    site_index: int,
    strength: float,
    batch_size: int,
    device: torch.device,
    tokenizer,
    include_details: bool = False,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    logits_chunks = []
    for start in range(0, bank.size, batch_size):
        end = min(start + batch_size, bank.size)
        logits = run_soft_site_intervention(
            model=model,
            base_input_ids=bank.base_input_ids[start:end].to(device),
            base_attention_mask=bank.base_attention_mask[start:end].to(device),
            source_input_ids=bank.source_input_ids[start:end].to(device),
            source_attention_mask=bank.source_attention_mask[start:end].to(device),
            site_weights={site: 1.0},
            strength=strength,
            base_position_by_id={
                key: value[start:end] for key, value in bank.base_position_by_id.items()
            },
            source_position_by_id={
                key: value[start:end] for key, value in bank.source_position_by_id.items()
            },
            pca_bases_by_id=pca_bases_by_id,
        )
        logits_chunks.append(logits.detach().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    ranking = [_site_ranking_record(site, site_index=site_index, target_var=bank.target_var)]
    record = {
        "method": "bruteforce",
        "variable": bank.target_var,
        "split": bank.split,
        "site_label": site.label,
        "layer": int(site.layer),
        "token_position_id": str(site.token_position_id),
        "top_k": 1,
        "lambda": float(strength),
        "top_site_label": site.label,
        "selected_site_labels": [site.label],
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if include_details:
        record["prediction_details"] = prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    return record, ranking


def _select_hyperparameters(
    *,
    model,
    calibration_bank: MCQAPairBank,
    sites: list[SiteLike],
    selection_transport: np.ndarray,
    renormalize_selected_transport: bool,
    batch_size: int,
    device: torch.device,
    tokenizer,
    config: OTConfig,
    source_target_vars: tuple[str, ...],
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    top_k_values, lambda_values = _resolve_calibration_grids(
        target_var=calibration_bank.target_var,
        config=config,
        num_sites=selection_transport.shape[1],
    )
    best = None
    sweep_records: list[dict[str, object]] = []
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] calibration start variable={calibration_bank.target_var} "
            f"signature_mode={config.signature_mode} top_k_values={list(top_k_values)} "
            f"lambda_values={list(lambda_values)} metric={config.calibration_metric}"
        )
    calibration_candidates = [(int(top_k), float(strength)) for top_k in top_k_values for strength in lambda_values]
    candidate_iterator = calibration_candidates
    if config.selection_verbose and tqdm is not None:
        candidate_iterator = tqdm(
            calibration_candidates,
            desc=f"{config.method.upper()} calibration sweep ({calibration_bank.target_var})",
            leave=False,
        )
    for top_k, strength in candidate_iterator:
        truncated = truncate_transport_rows(
            selection_transport,
            top_k,
            renormalize=renormalize_selected_transport,
        )
        result, ranking = _evaluate_soft_intervention(
            model=model,
            bank=calibration_bank,
            sites=sites,
            selected_transport=truncated,
            top_k=top_k,
            strength=strength,
            batch_size=batch_size,
            device=device,
            tokenizer=tokenizer,
            source_target_vars=source_target_vars,
            include_details=False,
            pca_bases_by_id=pca_bases_by_id,
        )
        calibration_score = _calibration_score_from_result(result, config)
        candidate = {
            "top_k": top_k,
            "lambda": strength,
            "result": result,
            "ranking": ranking,
            "exact_acc": float(result["exact_acc"]),
            "calibration_score": float(calibration_score),
            "calibration_metric": str(config.calibration_metric),
        }
        sweep_records.append(candidate)
        if best is None or float(candidate["calibration_score"]) > float(best["calibration_score"]) or (
            float(candidate["calibration_score"]) == float(best["calibration_score"]) and float(candidate["exact_acc"]) > float(best["exact_acc"])
        ):
            best = candidate
            if config.selection_verbose:
                print(
                    f"[{config.method.upper()}] new best variable={calibration_bank.target_var} "
                    f"top_k={int(top_k)} lambda={float(strength):g} "
                    f"calibration_score={float(candidate['calibration_score']):.4f} "
                    f"calibration_exact_acc={float(candidate['exact_acc']):.4f}"
                )
    if best is None:
        raise RuntimeError(f"Failed to select OT/UOT hyperparameters for {calibration_bank.target_var}")
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] selected variable={calibration_bank.target_var} "
            f"top_k={int(best['top_k'])} lambda={float(best['lambda']):g} "
            f"calibration_score={float(best['calibration_score']):.4f} "
            f"calibration_exact_acc={float(best['exact_acc']):.4f}"
        )
    return best, sweep_records


def _select_bruteforce_site(
    *,
    model,
    calibration_bank: MCQAPairBank,
    sites: list[SiteLike],
    batch_size: int,
    device: torch.device,
    tokenizer,
    config: OTConfig,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best = None
    sweep_records: list[dict[str, object]] = []
    _, lambda_values = _resolve_calibration_grids(
        target_var=calibration_bank.target_var,
        config=config,
        num_sites=len(sites),
    )
    if config.selection_verbose:
        print(
            f"[BRUTEFORCE] calibration start variable={calibration_bank.target_var} "
            f"sites={len(sites)} lambda_values={list(lambda_values)} "
            f"metric={config.calibration_metric}"
        )
    calibration_candidates = [
        (int(site_index), site, float(strength))
        for site_index, site in enumerate(sites)
        for strength in lambda_values
    ]
    candidate_iterator = calibration_candidates
    if config.selection_verbose and tqdm is not None:
        candidate_iterator = tqdm(
            calibration_candidates,
            desc=f"BRUTEFORCE calibration sweep ({calibration_bank.target_var})",
            leave=False,
        )
    for site_index, site, strength in candidate_iterator:
        result, ranking = _evaluate_single_site_intervention(
            model=model,
            bank=calibration_bank,
            site=site,
            site_index=site_index,
            strength=strength,
            batch_size=batch_size,
            device=device,
            tokenizer=tokenizer,
            include_details=False,
            pca_bases_by_id=pca_bases_by_id,
        )
        calibration_score = _calibration_score_from_result(result, config)
        candidate = {
            "site_index": int(site_index),
            "site_label": site.label,
            "lambda": strength,
            "result": result,
            "ranking": ranking,
            "exact_acc": float(result["exact_acc"]),
            "calibration_score": float(calibration_score),
            "calibration_metric": str(config.calibration_metric),
        }
        sweep_records.append(candidate)
        if best is None or float(candidate["calibration_score"]) > float(best["calibration_score"]) or (
            float(candidate["calibration_score"]) == float(best["calibration_score"]) and float(candidate["exact_acc"]) > float(best["exact_acc"])
        ):
            best = candidate
            if config.selection_verbose:
                print(
                    f"[BRUTEFORCE] new best variable={calibration_bank.target_var} "
                    f"site={site.label} lambda={float(strength):g} "
                    f"calibration_score={float(candidate['calibration_score']):.4f} "
                    f"calibration_exact_acc={float(candidate['exact_acc']):.4f}"
                )
    if best is None:
        raise RuntimeError(f"Failed to select a brute-force site for {calibration_bank.target_var}")
    if config.selection_verbose:
        print(
            f"[BRUTEFORCE] selected variable={calibration_bank.target_var} "
            f"site={best['site_label']} lambda={float(best['lambda']):g} "
            f"calibration_score={float(best['calibration_score']):.4f} "
            f"calibration_exact_acc={float(best['exact_acc']):.4f}"
        )
    return best, sweep_records


def run_alignment_pipeline(
    *,
    model,
    fit_banks_by_var: dict[str, MCQAPairBank],
    calibration_bank: MCQAPairBank,
    holdout_bank: MCQAPairBank,
    sites: list[SiteLike],
    device: torch.device | str,
    tokenizer,
    config: OTConfig,
    prepared_artifacts: dict[str, object] | None = None,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> dict[str, object]:
    """Run OT or UOT for one MCQA target variable using a two-source abstraction."""
    device = torch.device(device)
    total_start = perf_counter()
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] start variable={holdout_bank.target_var} "
            f"signature_mode={config.signature_mode} candidate_sites={len(sites)} "
            f"epsilon={float(config.epsilon):g} sources={list(config.source_target_vars)}"
        )
    signature_prepare_wall_seconds = 0.0
    artifact_prepare_load_seconds = 0.0
    if prepared_artifacts is None:
        if config.selection_verbose:
            print(f"[{config.method.upper()}] no prepared artifacts supplied; building signatures inline")
        signature_start = perf_counter()
        prepared_artifacts = prepare_alignment_artifacts(
            model=model,
            fit_banks_by_var=fit_banks_by_var,
            sites=sites,
            device=device,
            config=config,
            pca_bases_by_id=pca_bases_by_id,
        )
        _synchronize_if_cuda(device)
        signature_prepare_wall_seconds = float(perf_counter() - signature_start)
    else:
        artifact_prepare_load_seconds = 0.0
        if config.selection_verbose:
            print(
                f"[{config.method.upper()}] using prepared artifacts "
                f"cache_hit={bool(prepared_artifacts.get('artifact_cache_hit', False))} "
                f"loaded_from_disk={bool(prepared_artifacts.get('loaded_from_disk', False))}"
            )
    artifact_cache_hit = bool(prepared_artifacts.get("loaded_from_disk", False))
    artifact_prepare_recorded_seconds = float(prepared_artifacts.get("prepare_runtime_seconds", 0.0))
    artifact_prepare_create_seconds = float(signature_prepare_wall_seconds)
    signature_prepare_runtime_seconds = float(signature_prepare_wall_seconds)
    site_signatures_by_var = prepared_artifacts["site_signatures_by_var"]
    variable_signature_start = perf_counter()
    if config.selection_verbose:
        print(f"[{config.method.upper()}] building abstract variable signatures for sources={list(config.source_target_vars)}")
    variable_signatures_by_var = {
        target_var: build_variable_signature(fit_banks_by_var[target_var], config.signature_mode)
        for target_var in config.source_target_vars
    }
    _synchronize_if_cuda(device)
    variable_signature_seconds = float(perf_counter() - variable_signature_start)
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] solving transport rows={len(config.source_target_vars)} "
            f"cols={len(sites)}"
        )
    transport_start = perf_counter()
    if config.method == "ot":
        transport, transport_meta = solve_ot_transport(variable_signatures_by_var, site_signatures_by_var, config)
    elif config.method == "uot":
        transport, transport_meta = solve_uot_transport(variable_signatures_by_var, site_signatures_by_var, config)
    else:
        raise ValueError(f"Unsupported MCQA transport method {config.method}")
    _synchronize_if_cuda(device)
    transport_solve_seconds = float(perf_counter() - transport_start)
    normalized_transport = normalize_transport_rows(transport)
    target_transport, target_source_target_vars, target_row_index = _select_transport_row_for_target(
        transport,
        config.source_target_vars,
        holdout_bank.target_var,
    )
    target_normalized_transport = normalized_transport[target_row_index : target_row_index + 1]
    selection_transport, renormalize_selected_transport = _selection_transport_for_target(
        method=config.method,
        target_transport=target_transport,
        target_normalized_transport=target_normalized_transport,
    )
    calibration_select_start = perf_counter()
    selected, calibration_sweep = _select_hyperparameters(
        model=model,
        calibration_bank=calibration_bank,
        sites=sites,
        selection_transport=selection_transport,
        renormalize_selected_transport=renormalize_selected_transport,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        config=config,
        source_target_vars=target_source_target_vars,
        pca_bases_by_id=pca_bases_by_id,
    )
    _synchronize_if_cuda(device)
    calibration_select_seconds = float(perf_counter() - calibration_select_start)
    top_k = int(selected["top_k"])
    strength = float(selected["lambda"])
    selected_transport = truncate_transport_rows(
        selection_transport,
        top_k,
        renormalize=renormalize_selected_transport,
    )
    selected_column_mask = selected_transport.sum(axis=0) > 0.0
    selected_raw_transport = np.where(selected_column_mask[None, :], target_transport, 0.0)
    selected_position_mass = _position_mass_from_transport(selected_transport, sites)
    selected_raw_position_mass = _position_mass_from_transport(selected_raw_transport, sites)
    selected_raw_captured_mass = float(selected_raw_transport.sum())
    selected_calibration_start = perf_counter()
    selected_calibration_result, selected_calibration_ranking = _evaluate_soft_intervention(
        model=model,
        bank=calibration_bank,
        sites=sites,
        selected_transport=selected_transport,
        top_k=top_k,
        strength=strength,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        source_target_vars=target_source_target_vars,
        include_details=True,
        pca_bases_by_id=pca_bases_by_id,
    )
    _synchronize_if_cuda(device)
    selected_calibration_eval_seconds = float(perf_counter() - selected_calibration_start)
    holdout_start = perf_counter()
    holdout_result, holdout_ranking = _evaluate_soft_intervention(
        model=model,
        bank=holdout_bank,
        sites=sites,
        selected_transport=selected_transport,
        top_k=top_k,
        strength=strength,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        source_target_vars=target_source_target_vars,
        include_details=True,
        pca_bases_by_id=pca_bases_by_id,
    )
    _synchronize_if_cuda(device)
    holdout_eval_seconds = float(perf_counter() - holdout_start)
    total_wall_seconds = float(perf_counter() - total_start)
    localization_runtime_seconds = float(
        artifact_prepare_create_seconds
        + artifact_prepare_load_seconds
        + variable_signature_seconds
        + transport_solve_seconds
        + calibration_select_seconds
        + selected_calibration_eval_seconds
        + holdout_eval_seconds
    )
    holdout_result["method"] = config.method
    holdout_result["selection_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["calibration_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["selection_score"] = float(selected["calibration_score"])
    holdout_result["calibration_metric"] = str(config.calibration_metric)
    holdout_result["signature_mode"] = str(config.signature_mode)
    holdout_result["selected_transport_nonzero"] = int((selected_transport.sum(axis=0) > 0.0).sum())
    holdout_result["selected_raw_captured_mass"] = float(selected_raw_captured_mass)
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] holdout variable={holdout_bank.target_var} "
            f"top_k={top_k} lambda={strength:g} exact_acc={float(holdout_result['exact_acc']):.4f}"
        )
    return {
        "target_var": holdout_bank.target_var,
        "source_target_vars": list(config.source_target_vars),
        "target_var_row_index": int(target_row_index),
        "signature_mode": config.signature_mode,
        "calibration_metric": config.calibration_metric,
        "calibration_family_weights": [float(weight) for weight in config.calibration_family_weights],
        "transport": transport.tolist(),
        "normalized_transport": normalized_transport.tolist(),
        "target_transport": target_transport.tolist(),
        "target_normalized_transport": target_normalized_transport.tolist(),
        "selection_transport": selection_transport.tolist(),
        "selection_transport_renormalized": bool(renormalize_selected_transport),
        "selected_transport": selected_transport.tolist(),
        "selected_position_mass": selected_position_mass,
        "selected_raw_position_mass": selected_raw_position_mass,
        "selected_raw_captured_mass": float(selected_raw_captured_mass),
        "transport_meta": transport_meta,
        "artifact_cache_hit": bool(artifact_cache_hit),
        "artifact_prepare_create_seconds": float(artifact_prepare_create_seconds),
        "artifact_prepare_load_seconds": float(artifact_prepare_load_seconds),
        "artifact_prepare_recorded_seconds": float(artifact_prepare_recorded_seconds),
        "signature_prepare_runtime_seconds": float(signature_prepare_runtime_seconds),
        "wall_runtime_seconds": float(total_wall_seconds),
        "runtime_seconds": float(total_wall_seconds),
        "localization_runtime_seconds": float(localization_runtime_seconds),
        "core_method_seconds": float(total_wall_seconds),
        "timing_seconds": {
            "t_artifact_prepare_create": float(artifact_prepare_create_seconds),
            "t_artifact_prepare_load": float(artifact_prepare_load_seconds),
            "t_signature_prepare": float(signature_prepare_runtime_seconds),
            "t_variable_signature_build": float(variable_signature_seconds),
            "t_transport_solve": float(transport_solve_seconds),
            "t_handle_calibration": float(calibration_select_seconds),
            "t_calibration_select": float(calibration_select_seconds),
            "t_selected_calibration_eval": float(selected_calibration_eval_seconds),
            "t_final_holdout_eval": float(holdout_eval_seconds),
            "t_localization_runtime": float(localization_runtime_seconds),
            "t_total_wall": float(total_wall_seconds),
        },
        "selected_hyperparameters": {
            "top_k": top_k,
            "lambda": strength,
            "signature_mode": config.signature_mode,
            "calibration_metric": config.calibration_metric,
        },
        "selected_calibration_result": selected_calibration_result,
        "selected_calibration_ranking": selected_calibration_ranking,
        "ranking": holdout_ranking,
        "calibration_sweep": calibration_sweep,
        "results": [holdout_result],
    }


def run_bruteforce_site_pipeline(
    *,
    model,
    calibration_bank: MCQAPairBank,
    holdout_bank: MCQAPairBank,
    sites: list[SiteLike],
    device: torch.device | str,
    tokenizer,
    config: OTConfig,
    pca_bases_by_id: dict[str, LayerPCABasis] | None = None,
) -> dict[str, object]:
    """Run a direct 3-site baseline by calibrating over single full-vector sites."""
    device = torch.device(device)
    if config.selection_verbose:
        print(
            f"[BRUTEFORCE] start variable={holdout_bank.target_var} "
            f"candidate_sites={len(sites)} lambda_values={list(config.lambda_values)}"
        )
    selected, calibration_sweep = _select_bruteforce_site(
        model=model,
        calibration_bank=calibration_bank,
        sites=sites,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        config=config,
        pca_bases_by_id=pca_bases_by_id,
    )
    selected_site = sites[int(selected["site_index"])]
    strength = float(selected["lambda"])
    selected_calibration_result, selected_calibration_ranking = _evaluate_single_site_intervention(
        model=model,
        bank=calibration_bank,
        site=selected_site,
        site_index=int(selected["site_index"]),
        strength=strength,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        include_details=True,
        pca_bases_by_id=pca_bases_by_id,
    )
    holdout_result, holdout_ranking = _evaluate_single_site_intervention(
        model=model,
        bank=holdout_bank,
        site=selected_site,
        site_index=int(selected["site_index"]),
        strength=strength,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        include_details=True,
        pca_bases_by_id=pca_bases_by_id,
    )
    holdout_result["method"] = "bruteforce"
    holdout_result["selection_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["calibration_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["selection_score"] = float(selected["calibration_score"])
    holdout_result["calibration_metric"] = str(config.calibration_metric)
    holdout_result["signature_mode"] = str(config.signature_mode)
    if config.selection_verbose:
        print(
            f"[BRUTEFORCE] holdout variable={holdout_bank.target_var} "
            f"site={selected_site.label} lambda={strength:g} exact_acc={float(holdout_result['exact_acc']):.4f}"
        )
    return {
        "target_var": holdout_bank.target_var,
        "signature_mode": config.signature_mode,
        "calibration_metric": config.calibration_metric,
        "calibration_family_weights": [float(weight) for weight in config.calibration_family_weights],
        "selected_hyperparameters": {
            "site_label": selected_site.label,
            "lambda": strength,
            "signature_mode": config.signature_mode,
            "calibration_metric": config.calibration_metric,
        },
        "selected_calibration_result": selected_calibration_result,
        "selected_calibration_ranking": selected_calibration_ranking,
        "ranking": holdout_ranking,
        "calibration_sweep": calibration_sweep,
        "results": [holdout_result],
    }
