"""High-level IOI causal model and MIB linear-regression fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import IOIExample, family_signals


VARIABLES = ("position", "token")
PUBLISHED_COEFFICIENTS = {
    "bias": 0.048,
    "position_coeff": 2.005,
    "token_coeff": 0.768,
}


@dataclass(frozen=True)
class LinearCausalModel:
    bias: float
    position_coeff: float
    token_coeff: float
    r2: float | None = None

    def predict_signals(self, position_signal: int, token_signal: int) -> float:
        return float(
            self.bias
            + self.position_coeff * int(position_signal)
            + self.token_coeff * int(token_signal)
        )

    def factual(self) -> float:
        return self.predict_signals(1, 1)

    def intervention_target(self, variable: str, family: str) -> float:
        position_signal, token_signal = family_signals(family)
        if variable == "position":
            return self.predict_signals(position_signal, 1)
        if variable == "token":
            return self.predict_signals(1, token_signal)
        raise ValueError(f"Unknown IOI variable: {variable}")

    def effect(self, variable: str, family: str) -> float:
        return self.intervention_target(variable, family) - self.factual()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LinearCausalModel":
        return cls(
            bias=float(value["bias"]),
            position_coeff=float(value["position_coeff"]),
            token_coeff=float(value["token_coeff"]),
            r2=None if value.get("r2") is None else float(value["r2"]),
        )


def regression_design(
    same_values: Sequence[float],
    patched_by_family: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build MIB's per-example OLS design with intercept added by the fitter."""

    rows: list[tuple[float, float]] = []
    outputs: list[float] = []
    counts = {"same": len(same_values)}
    for value in same_values:
        rows.append((1.0, 1.0))
        outputs.append(float(value))
    for family, values in patched_by_family.items():
        position_signal, token_signal = family_signals(family)
        counts[family] = len(values)
        for value in values:
            rows.append((float(position_signal), float(token_signal)))
            outputs.append(float(value))
    return np.asarray(rows, dtype=np.float64), np.asarray(outputs, dtype=np.float64), counts


def fit_linear_causal_model(
    same_values: Sequence[float],
    patched_by_family: Mapping[str, Sequence[float]],
) -> tuple[LinearCausalModel, dict[str, object]]:
    X, y, counts = regression_design(same_values, patched_by_family)
    if len(y) < 3:
        raise ValueError("At least three regression observations are required")
    design = np.concatenate([np.ones((len(X), 1), dtype=np.float64), X], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ coefficients
    residual_sum = float(np.square(y - predictions).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    model = LinearCausalModel(
        bias=float(coefficients[0]),
        position_coeff=float(coefficients[1]),
        token_coeff=float(coefficients[2]),
        r2=float(r2),
    )
    deltas = {
        key: float(getattr(model, key) - published)
        for key, published in PUBLISHED_COEFFICIENTS.items()
    }
    return model, {
        "coefficients": model.as_dict(),
        "published_coefficients": dict(PUBLISHED_COEFFICIENTS),
        "published_deltas": deltas,
        "counts": counts,
        "n_observations": int(len(y)),
        "residual_mse": float(np.square(y - predictions).mean()),
    }


def targets_for_examples(
    model: LinearCausalModel, variable: str, examples: Sequence[IOIExample]
) -> list[float]:
    return [model.intervention_target(variable, item.family) for item in examples]
