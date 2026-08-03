"""PLOT signatures, costs, one-sided UOT, and staged calibration for IOI."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .causal import LinearCausalModel, VARIABLES
from .data import IOIExample
from .interventions import (
    Head,
    all_gpt2_heads,
    evaluate_mse,
    head_label,
    intervention_logit_differences,
    parse_head_label,
)


@dataclass(frozen=True)
class SignatureBank:
    """Factual and one-head full-vector effects, aligned within each family."""

    heads: tuple[Head, ...]
    factual: dict[str, tuple[float, ...]]
    neural_effects: dict[str, np.ndarray]  # family -> [example, head]
    runtime_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "heads": [head_label(head) for head in self.heads],
            "factual": {family: list(values) for family, values in self.factual.items()},
            "neural_effects": {
                family: values.astype(np.float32).tolist()
                for family, values in self.neural_effects.items()
            },
            "runtime_seconds": float(self.runtime_seconds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SignatureBank":
        return cls(
            heads=tuple(parse_head_label(label) for label in value["heads"]),
            factual={
                family: tuple(float(item) for item in values)
                for family, values in value["factual"].items()
            },
            neural_effects={
                family: np.asarray(values, dtype=np.float64)
                for family, values in value["neural_effects"].items()
            },
            runtime_seconds=float(value.get("runtime_seconds", 0.0)),
        )


def collect_signature_bank(
    model,
    tokenizer,
    groups: Mapping[str, Sequence[IOIExample]],
    *,
    device: torch.device,
    batch_size: int,
    heads: Sequence[Head] | None = None,
) -> SignatureBank:
    """Collect all-token, full-vector intervention effects for every candidate head."""

    started = perf_counter()
    selected_heads = tuple(heads or all_gpt2_heads())
    factual: dict[str, tuple[float, ...]] = {}
    effects: dict[str, np.ndarray] = {}
    for family, rows in groups.items():
        base_values = intervention_logit_differences(
            model,
            tokenizer,
            rows,
            factual=True,
            device=device,
            batch_size=batch_size,
        )
        factual[family] = tuple(base_values)
        columns = []
        for head in selected_heads:
            patched = intervention_logit_differences(
                model,
                tokenizer,
                rows,
                heads=(head,),
                rotations=None,
                device=device,
                batch_size=batch_size,
            )
            columns.append(np.asarray(patched, dtype=np.float64) - np.asarray(base_values))
        effects[family] = (
            np.stack(columns, axis=1)
            if columns
            else np.empty((len(rows), 0), dtype=np.float64)
        )
    return SignatureBank(
        heads=selected_heads,
        factual=factual,
        neural_effects=effects,
        runtime_seconds=float(perf_counter() - started),
    )


def build_cost_matrix(
    signatures: SignatureBank, causal_model: LinearCausalModel
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the variable-by-head family-macro effect-MSE cost matrix."""

    costs = np.empty((len(VARIABLES), len(signatures.heads)), dtype=np.float64)
    by_family: dict[str, dict[str, list[float]]] = {}
    for variable_index, variable in enumerate(VARIABLES):
        family_costs = []
        by_family[variable] = {}
        for family, effects in signatures.neural_effects.items():
            target = float(causal_model.effect(variable, family))
            family_values = np.mean(np.square(effects - target), axis=0)
            family_costs.append(family_values)
            by_family[variable][family] = family_values.tolist()
        costs[variable_index] = np.mean(np.stack(family_costs, axis=0), axis=0)
    if costs.shape != (2, len(signatures.heads)) or not np.isfinite(costs).all():
        raise ValueError(f"Expected a finite 2x{len(signatures.heads)} cost matrix")
    return costs, {"family_costs": by_family, "cost_matrix": costs.tolist()}


def sinkhorn_one_sided_uot(
    cost: np.ndarray,
    *,
    epsilon: float,
    beta_neural: float,
    iterations: int = 500,
) -> np.ndarray:
    """One-sided UOT with exact abstract and relaxed neural marginals."""

    if epsilon <= 0 or beta_neural <= 0 or iterations <= 0:
        raise ValueError("epsilon, beta_neural, and iterations must be positive")
    values = torch.as_tensor(cost, dtype=torch.float64)
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("cost must be a finite rank-2 matrix")
    rows, columns = values.shape
    abstract_mass = torch.full((rows,), 1.0 / rows, dtype=torch.float64)
    neural_mass = torch.full((columns,), 1.0 / columns, dtype=torch.float64)
    kernel = torch.exp(-values / float(epsilon)).clamp_min(1e-300)
    left = torch.ones_like(abstract_mass)
    right = torch.ones_like(neural_mass)
    neural_power = float(beta_neural / (beta_neural + epsilon))
    for _ in range(int(iterations)):
        left = abstract_mass / (kernel @ right).clamp_min(1e-300)
        right = (neural_mass / (kernel.T @ left).clamp_min(1e-300)).pow(neural_power)
    coupling = left[:, None] * kernel * right[None, :]
    coupling = coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-300)
    result = coupling.cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("UOT produced a non-finite coupling")
    return result


def ranked_heads(coupling: np.ndarray, heads: Sequence[Head], variable: str) -> tuple[Head, ...]:
    row = VARIABLES.index(variable)
    # Lexicographic sort makes equal-mass selection reproducible across NumPy versions.
    order = sorted(range(len(heads)), key=lambda index: (-float(coupling[row, index]), index))
    return tuple(heads[index] for index in order)


def top_k_heads(
    coupling: np.ndarray, heads: Sequence[Head], variable: str, k: int
) -> tuple[Head, ...]:
    return ranked_heads(coupling, heads, variable)[: int(k)]


def choose_shared_uot(
    trial_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Choose a setting using each row's best full-vector calibration MSE."""

    if not trial_records:
        raise ValueError("No UOT trials supplied")

    def key(record: Mapping[str, object]):
        best = record["best_by_variable"]
        score = float(np.mean([float(best[variable]["macro_mse"]) for variable in VARIABLES]))
        return score, float(record["epsilon"]), float(record["beta_neural"])

    return min(trial_records, key=key)


def choose_k(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Select the lowest calibration MSE, breaking ties toward smaller K."""

    if not records:
        raise ValueError("No K candidates supplied")
    return min(records, key=lambda record: (float(record["macro_mse"]), int(record["k"])))


def calibrate_uot_grid(
    model,
    tokenizer,
    calibration_groups: Mapping[str, Sequence[IOIExample]],
    *,
    causal_model: LinearCausalModel,
    cost: np.ndarray,
    heads: Sequence[Head],
    epsilons: Sequence[float],
    beta_neurals: Sequence[float],
    k_values: Sequence[int],
    device: torch.device,
    batch_size: int,
    evaluator: Callable[..., Mapping[str, object]] = evaluate_mse,
) -> tuple[Mapping[str, object], list[dict[str, object]]]:
    """Stage 1: tune UOT using full-vector top-K calibration interventions."""

    trials: list[dict[str, object]] = []
    for epsilon in epsilons:
        for beta_neural in beta_neurals:
            coupling = sinkhorn_one_sided_uot(
                cost, epsilon=float(epsilon), beta_neural=float(beta_neural)
            )
            per_variable: dict[str, list[dict[str, object]]] = {}
            best_by_variable: dict[str, Mapping[str, object]] = {}
            for variable in VARIABLES:
                records = []
                for k in sorted(set(int(value) for value in k_values)):
                    selected = top_k_heads(coupling, heads, variable, k)
                    metrics = evaluator(
                        model,
                        tokenizer,
                        calibration_groups,
                        variable=variable,
                        causal_model=causal_model,
                        heads=selected,
                        rotations=None,
                        device=device,
                        batch_size=batch_size,
                    )
                    records.append(
                        {
                            "k": k,
                            "heads": [head_label(head) for head in selected],
                            "macro_mse": float(metrics["macro_mse"]),
                            "metrics": dict(metrics),
                        }
                    )
                per_variable[variable] = records
                best_by_variable[variable] = choose_k(records)
            trials.append(
                {
                    "epsilon": float(epsilon),
                    "beta_neural": float(beta_neural),
                    "coupling": coupling.tolist(),
                    "per_variable": per_variable,
                    "best_by_variable": best_by_variable,
                }
            )
    return choose_shared_uot(trials), trials
