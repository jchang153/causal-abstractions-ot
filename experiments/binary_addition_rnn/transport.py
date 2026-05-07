from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .data import CarryPairRecord, ExhaustiveBanks
from .features import abstract_effect_signature, aggregate_mean, neural_effect_signature_for_site
from .interventions import RunCache, intervene_with_site_handle_batch
from .model import GRUAdder
from .sites import Site


@dataclass(frozen=True)
class TransportConfig:
    epsilon_grid: tuple[float, ...] = (0.01, 0.03, 0.1)
    beta_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
    topk_grid: tuple[int, ...] = (1, 2)
    lambda_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    sinkhorn_iters: int = 200
    temperature: float = 1.0
    invariance_floor: float = 0.0
    selection_rule: str = "combined"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def cost_matrix_from_tables(
    abstract_table: torch.Tensor,
    neural_table: torch.Tensor,
) -> torch.Tensor:
    if abstract_table.ndim != 2 or neural_table.ndim != 2:
        raise ValueError("abstract_table and neural_table must be rank-2")
    if abstract_table.size(1) != neural_table.size(1):
        raise ValueError(
            f"feature dimensions must match, got {abstract_table.size(1)} and {neural_table.size(1)}"
        )
    diffs = abstract_table[:, None, :] - neural_table[None, :, :]
    return torch.sum(diffs * diffs, dim=2).to(torch.float32)


def fit_cost_matrix(
    model: GRUAdder,
    fit_by_carry: dict[int, Sequence[CarryPairRecord]],
    sites: Sequence[Site],
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    batch_size: int = 256,
) -> tuple[torch.Tensor, dict[str, object]]:
    width = len(fit_by_carry)
    abstract_rows = []
    pair_signature_cache: dict[tuple[int, int, int, int, str], torch.Tensor] = {}
    site_means = []
    shared_records = fit_by_carry[1]
    for site in sites:
        sigs = []
        uncached_indices = []
        uncached_records = []
        for idx, rec in enumerate(shared_records):
            key = (int(rec.base.a), int(rec.base.b), int(rec.source.a), int(rec.source.b), site.key())
            sig = pair_signature_cache.get(key)
            if sig is None:
                uncached_indices.append(idx)
                uncached_records.append(rec)
            else:
                sigs.append(sig)

        for start in range(0, len(uncached_records), int(batch_size)):
            batch_records = uncached_records[start : start + int(batch_size)]
            batch_bases = [rec.base for rec in batch_records]
            batch_sources = [rec.source for rec in batch_records]
            batch_logits = intervene_with_site_handle_batch(
                model,
                batch_bases,
                batch_sources,
                [(site, 1.0)],
                lambda_scale=1.0,
                device=device,
                run_cache=run_cache,
            )
            batch_probs = torch.sigmoid(batch_logits)
            if run_cache is None:
                base_probs = torch.stack(
                    [neural_effect_signature_for_site(model, rec.base, rec.source, site, device=device, run_cache=None) for rec in batch_records],
                    dim=0,
                )
                batch_sigs = base_probs
            else:
                factual = torch.stack([run_cache.get_run(rec.base).output_probs for rec in batch_records], dim=0)
                batch_sigs = batch_probs - factual
            for rec, sig in zip(batch_records, batch_sigs):
                key = (int(rec.base.a), int(rec.base.b), int(rec.source.a), int(rec.source.b), site.key())
                sig = sig.detach().cpu()
                pair_signature_cache[key] = sig
                sigs.append(sig)
        site_means.append(aggregate_mean(sigs))
    neural_table = torch.stack(site_means, dim=0)  # [site, feat]
    for carry_index in range(1, width + 1):
        records = fit_by_carry[carry_index]
        abs_sig = aggregate_mean(
            [abstract_effect_signature(rec.base, rec.carry_index, rec.forced_value) for rec in records]
        )
        abstract_rows.append(abs_sig)

    abstract_table = torch.stack(abstract_rows, dim=0)
    cost = cost_matrix_from_tables(abstract_table, neural_table)

    diagnostics = {
        "abstract_signatures": abstract_table.tolist(),
        "neural_signatures": neural_table.tolist(),
        "cost_matrix": cost.tolist(),
        "cached_pair_site_signatures": int(len(pair_signature_cache)),
    }
    return cost, diagnostics


def sinkhorn_uniform_ot(cost: torch.Tensor, epsilon: float, n_iter: int, temperature: float = 1.0) -> torch.Tensor:
    if epsilon <= 0 or n_iter <= 0 or temperature <= 0:
        raise ValueError("epsilon, n_iter, and temperature must be > 0")
    m, n = cost.shape
    a = torch.full((m,), 1.0 / m, dtype=torch.float32)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32)
    kernel = torch.exp(-cost.to(torch.float32) / (epsilon * temperature)).clamp_min(1e-30)
    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(int(n_iter)):
        r = a / (kernel @ c).clamp_min(1e-30)
        c = b / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)
    return r[:, None] * kernel * c[None, :]


def sinkhorn_one_sided_uot(
    cost: torch.Tensor,
    epsilon: float,
    beta_neural: float,
    n_iter: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    if epsilon <= 0 or beta_neural <= 0 or n_iter <= 0 or temperature <= 0:
        raise ValueError("epsilon, beta_neural, n_iter, and temperature must be > 0")
    m, n = cost.shape
    a = torch.full((m,), 1.0 / m, dtype=torch.float32)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32)
    kernel = torch.exp(-cost.to(torch.float32) / (epsilon * temperature)).clamp_min(1e-30)
    rho_a = 1.0
    rho_b = float(beta_neural / (beta_neural + epsilon))
    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(int(n_iter)):
        r = (a / (kernel @ c).clamp_min(1e-30)).pow(rho_a)
        c = (b / (kernel.transpose(0, 1) @ r).clamp_min(1e-30)).pow(rho_b)
    return r[:, None] * kernel * c[None, :]


def rowwise_argmin_matching(cost: torch.Tensor) -> torch.Tensor:
    if cost.ndim != 2:
        raise ValueError("cost must be rank-2")
    out = torch.zeros_like(cost, dtype=torch.float32)
    idx = torch.argmin(cost, dim=1)
    out[torch.arange(cost.size(0)), idx] = 1.0
    return out


def rowwise_entropic_matching(cost: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = -cost.to(torch.float32) / float(temperature)
    return torch.softmax(logits, dim=1)


def _row_normalize(pi: torch.Tensor) -> torch.Tensor:
    return pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-30)


def _truncate_row(row: torch.Tensor, top_k: int) -> list[tuple[int, float]]:
    vals, idx = torch.topk(row, k=min(int(top_k), row.numel()))
    vals = vals / vals.sum().clamp_min(1e-30)
    return [(int(i.item()), float(v.item())) for i, v in zip(idx, vals)]


def _calibration_key(
    *,
    combined: float,
    sensitivity: float,
    invariance: float,
    selection_rule: str,
    invariance_floor: float,
) -> tuple[float, float, float]:
    admissible = 1.0 if float(invariance) >= float(invariance_floor) else 0.0
    if selection_rule == "combined":
        return (admissible, float(combined), float(sensitivity))
    if selection_rule == "sensitivity_only":
        return (1.0, float(sensitivity), float(invariance))
    if selection_rule == "sensitivity_then_invariance":
        return (admissible, float(sensitivity), float(invariance))
    raise ValueError(f"unknown selection_rule: {selection_rule!r}")


def _exact_match_rate(
    model: GRUAdder,
    records: Sequence[CarryPairRecord],
    selected_sites: Sequence[tuple[Site, float]],
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache | None = None,
    batch_size: int = 256,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> float:
    if not records:
        return 0.0
    hits = 0
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start : start + int(batch_size)]
        batch_logits = intervene_with_site_handle_batch(
            model,
            [rec.base for rec in batch_records],
            [rec.source for rec in batch_records],
            selected_sites,
            lambda_scale=lambda_scale,
            device=device,
            run_cache=run_cache,
            rotation_map=rotation_map,
        )
        pred_bits = (torch.sigmoid(batch_logits) >= 0.5).to(torch.int64)
        tgt_bits = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch_records], dtype=torch.int64)
        hits += int((pred_bits == tgt_bits).all(dim=1).sum().item())
    return float(hits / len(records))


def calibrate_transport_rows(
    model: GRUAdder,
    row_coupling: torch.Tensor,
    sites: Sequence[Site],
    calib_positive_by_carry: dict[int, Sequence[CarryPairRecord]],
    calib_invariant_by_carry: dict[int, Sequence[CarryPairRecord]],
    config: TransportConfig,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> dict[int, dict[str, object]]:
    normalized = _row_normalize(row_coupling)
    chosen: dict[int, dict[str, object]] = {}
    for carry_index in range(1, normalized.size(0) + 1):
        row = normalized[carry_index - 1]
        best_key = None
        best_rec = None
        for top_k in config.topk_grid:
            truncated = _truncate_row(row, int(top_k))
            selected = [(sites[idx], weight) for idx, weight in truncated]
            for lambda_scale in config.lambda_grid:
                sens = _exact_match_rate(
                    model,
                    calib_positive_by_carry[carry_index],
                    selected,
                    lambda_scale=float(lambda_scale),
                    device=device,
                    run_cache=run_cache,
                    rotation_map=rotation_map,
                )
                inv = _exact_match_rate(
                    model,
                    calib_invariant_by_carry[carry_index],
                    selected,
                    lambda_scale=float(lambda_scale),
                    device=device,
                    run_cache=run_cache,
                    rotation_map=rotation_map,
                )
                combined = 0.5 * (sens + inv)
                key = _calibration_key(
                    combined=combined,
                    sensitivity=sens,
                    invariance=inv,
                    selection_rule=config.selection_rule,
                    invariance_floor=config.invariance_floor,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_rec = {
                        "top_k": int(top_k),
                        "lambda": float(lambda_scale),
                        "selected_sites": [
                            {"site_index": int(idx), "site_key": sites[idx].key(), "weight": float(weight)}
                            for idx, weight in truncated
                        ],
                        "calibration": {
                            "sensitivity": float(sens),
                            "invariance": float(inv),
                            "combined": float(combined),
                            "selection_rule": str(config.selection_rule),
                            "invariance_floor": float(config.invariance_floor),
                        },
                    }
        chosen[carry_index] = best_rec if best_rec is not None else {}
    return chosen


def evaluate_calibrated_transport(
    model: GRUAdder,
    calibrated: dict[int, dict[str, object]],
    sites: Sequence[Site],
    test_positive_by_carry: dict[int, Sequence[CarryPairRecord]],
    test_invariant_by_carry: dict[int, Sequence[CarryPairRecord]],
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> dict[str, object]:
    per_carry = {}
    sens_vals = []
    inv_vals = []
    for carry_index, rec in calibrated.items():
        selected = [
            (sites[int(site_rec["site_index"])], float(site_rec["weight"]))
            for site_rec in rec["selected_sites"]
        ]
        lambda_scale = float(rec["lambda"])
        sens = _exact_match_rate(
            model,
            test_positive_by_carry[carry_index],
            selected,
            lambda_scale=lambda_scale,
            device=device,
            run_cache=run_cache,
            rotation_map=rotation_map,
        )
        inv = _exact_match_rate(
            model,
            test_invariant_by_carry[carry_index],
            selected,
            lambda_scale=lambda_scale,
            device=device,
            run_cache=run_cache,
            rotation_map=rotation_map,
        )
        combined = 0.5 * (sens + inv)
        sens_vals.append(sens)
        inv_vals.append(inv)
        per_carry[str(carry_index)] = {
            "sensitivity": float(sens),
            "invariance": float(inv),
            "combined": float(combined),
            "selected_sites": rec["selected_sites"],
            "lambda": lambda_scale,
            "top_k": int(rec["top_k"]),
        }
    return {
        "per_carry": per_carry,
        "sensitivity_mean": float(sum(sens_vals) / max(1, len(sens_vals))),
        "invariance_mean": float(sum(inv_vals) / max(1, len(inv_vals))),
        "combined_mean": float((sum(sens_vals) + sum(inv_vals)) / max(1, 2 * len(sens_vals))),
    }


def run_transport_sweep(
    method: str,
    model: GRUAdder,
    banks: ExhaustiveBanks,
    sites: Sequence[Site],
    config: TransportConfig,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> dict[str, object]:
    cost, diagnostics = fit_cost_matrix(model, banks.fit_by_carry, sites, device=device, run_cache=run_cache)
    trials = []
    best_key = None
    best_trial = None

    if method == "ot":
        for eps in config.epsilon_grid:
            pi = sinkhorn_uniform_ot(cost, epsilon=float(eps), n_iter=config.sinkhorn_iters, temperature=config.temperature)
            calibrated = calibrate_transport_rows(
                model,
                pi,
                sites,
                banks.calib_positive_by_carry,
                banks.calib_invariant_by_carry,
                config,
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            test_eval = evaluate_calibrated_transport(
                model,
                calibrated,
                sites,
                banks.test_positive_by_carry,
                banks.test_invariant_by_carry,
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            key = (test_eval["combined_mean"], test_eval["sensitivity_mean"], test_eval["invariance_mean"])
            trial = {
                "config": {"method": method, "epsilon": float(eps)},
                "coupling": pi.tolist(),
                "calibrated": calibrated,
                "test": test_eval,
            }
            trials.append(trial)
            if best_key is None or key > best_key:
                best_key = key
                best_trial = trial
    elif method == "uot":
        for eps in config.epsilon_grid:
            for beta in config.beta_grid:
                pi = sinkhorn_one_sided_uot(
                    cost,
                    epsilon=float(eps),
                    beta_neural=float(beta),
                    n_iter=config.sinkhorn_iters,
                    temperature=config.temperature,
                )
                calibrated = calibrate_transport_rows(
                    model,
                    pi,
                    sites,
                    banks.calib_positive_by_carry,
                    banks.calib_invariant_by_carry,
                    config,
                    device=device,
                    run_cache=run_cache,
                    rotation_map=rotation_map,
                )
                test_eval = evaluate_calibrated_transport(
                    model,
                    calibrated,
                    sites,
                    banks.test_positive_by_carry,
                    banks.test_invariant_by_carry,
                    device=device,
                    run_cache=run_cache,
                    rotation_map=rotation_map,
                )
                key = (test_eval["combined_mean"], test_eval["sensitivity_mean"], test_eval["invariance_mean"])
                trial = {
                    "config": {"method": method, "epsilon": float(eps), "beta_neural": float(beta)},
                    "coupling": pi.tolist(),
                    "calibrated": calibrated,
                    "test": test_eval,
                }
                trials.append(trial)
                if best_key is None or key > best_key:
                    best_key = key
                    best_trial = trial
    else:
        raise ValueError(f"unknown transport method: {method!r}")

    return {
        "method": method,
        "transport_config": config.as_dict(),
        "fit_diagnostics": diagnostics,
        "trials": trials,
        "best_trial": best_trial,
    }


def calibrate_single_transport_row(
    model: GRUAdder,
    row: torch.Tensor,
    sites: Sequence[Site],
    positive_records: Sequence[CarryPairRecord],
    invariant_records: Sequence[CarryPairRecord],
    config: TransportConfig,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> dict[str, object]:
    candidates = enumerate_transport_row_candidates(
        model,
        row,
        sites,
        positive_records,
        invariant_records,
        config,
        device=device,
        run_cache=run_cache,
        rotation_map=rotation_map,
    )
    return select_transport_calibration_candidate(
        candidates,
        selection_rule=str(config.selection_rule),
        invariance_floor=float(config.invariance_floor),
    )


def enumerate_transport_row_candidates(
    model: GRUAdder,
    row: torch.Tensor,
    sites: Sequence[Site],
    positive_records: Sequence[CarryPairRecord],
    invariant_records: Sequence[CarryPairRecord],
    config: TransportConfig,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> list[dict[str, object]]:
    normalized = row / row.sum().clamp_min(1e-30)
    candidates: list[dict[str, object]] = []
    for top_k in config.topk_grid:
        truncated = _truncate_row(normalized, int(top_k))
        selected = [(sites[idx], weight) for idx, weight in truncated]
        for lambda_scale in config.lambda_grid:
            sens = _exact_match_rate(
                model,
                positive_records,
                selected,
                lambda_scale=float(lambda_scale),
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            inv = _exact_match_rate(
                model,
                invariant_records,
                selected,
                lambda_scale=float(lambda_scale),
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            combined = 0.5 * (sens + inv)
            candidates.append(
                {
                    "top_k": int(top_k),
                    "lambda": float(lambda_scale),
                    "selected_sites": [
                        {"site_index": int(idx), "site_key": sites[idx].key(), "weight": float(weight)}
                        for idx, weight in truncated
                    ],
                    "calibration": {
                        "sensitivity": float(sens),
                        "invariance": float(inv),
                        "combined": float(combined),
                        "count_positive": int(len(positive_records)),
                        "count_invariant": int(len(invariant_records)),
                    },
                }
            )
    return candidates


def select_transport_calibration_candidate(
    candidates: Sequence[dict[str, object]],
    *,
    selection_rule: str,
    invariance_floor: float,
) -> dict[str, object]:
    best_key = None
    best_rec = None
    for candidate in candidates:
        calibration = candidate["calibration"]
        key = _calibration_key(
            combined=float(calibration["combined"]),
            sensitivity=float(calibration["sensitivity"]),
            invariance=float(calibration["invariance"]),
            selection_rule=str(selection_rule),
            invariance_floor=float(invariance_floor),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_rec = {
                "top_k": int(candidate["top_k"]),
                "lambda": float(candidate["lambda"]),
                "selected_sites": list(candidate["selected_sites"]),
                "calibration": {
                    **dict(calibration),
                    "selection_rule": str(selection_rule),
                    "invariance_floor": float(invariance_floor),
                },
            }
    return best_rec if best_rec is not None else {}


def evaluate_single_calibrated_transport(
    model: GRUAdder,
    calibrated: dict[str, object],
    sites: Sequence[Site],
    positive_records: Sequence[CarryPairRecord],
    invariant_records: Sequence[CarryPairRecord],
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
    rotation_map: dict[int, torch.Tensor] | None = None,
) -> dict[str, object]:
    selected = [
        (sites[int(site_rec["site_index"])], float(site_rec["weight"]))
        for site_rec in calibrated["selected_sites"]
    ]
    lambda_scale = float(calibrated["lambda"])
    sens = _exact_match_rate(
        model,
        positive_records,
        selected,
        lambda_scale=lambda_scale,
        device=device,
        run_cache=run_cache,
        rotation_map=rotation_map,
    )
    inv = _exact_match_rate(
        model,
        invariant_records,
        selected,
        lambda_scale=lambda_scale,
        device=device,
        run_cache=run_cache,
        rotation_map=rotation_map,
    )
    return {
        "sensitivity": float(sens),
        "invariance": float(inv),
        "combined": float(0.5 * (sens + inv)),
        "count_positive": int(len(positive_records)),
        "count_invariant": int(len(invariant_records)),
        "selected_sites": calibrated["selected_sites"],
        "top_k": int(calibrated["top_k"]),
        "lambda": lambda_scale,
    }
