"""Shared calibration-only hyperparameter selection utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def select_shared_epsilon(
    candidates: Iterable[dict[str, object]],
    *,
    variables: Sequence[str],
    epsilon_order: Sequence[float] | None = None,
) -> dict[str, object]:
    """Average each variable's best calibration score at every epsilon."""

    records = list(candidates)
    variable_order = tuple(str(variable) for variable in variables)
    if not variable_order:
        raise ValueError("variables must be non-empty")
    observed_epsilons: list[float] = []
    for record in records:
        epsilon = float(record["epsilon"])
        if epsilon not in observed_epsilons:
            observed_epsilons.append(epsilon)
    ordered_epsilons = [float(value) for value in (epsilon_order or observed_epsilons)]
    plans: list[dict[str, object]] = []
    for epsilon in ordered_epsilons:
        selected_by_variable: dict[str, dict[str, object]] = {}
        for variable in variable_order:
            matching = [
                record
                for record in records
                if str(record["variable"]) == variable and float(record["epsilon"]) == epsilon
            ]
            if not matching:
                break
            best = matching[0]
            for candidate in matching[1:]:
                if float(candidate["calibration_score"]) > float(best["calibration_score"]):
                    best = candidate
            selected_by_variable[variable] = best
        if len(selected_by_variable) != len(variable_order):
            continue
        scores = {
            variable: float(selected_by_variable[variable]["calibration_score"])
            for variable in variable_order
        }
        plans.append(
            {
                "epsilon": float(epsilon),
                "mean_best_variable_calibration_score": float(sum(scores.values()) / len(scores)),
                "best_variable_calibration_scores": scores,
                "selected_by_variable": selected_by_variable,
            }
        )
    if not plans:
        raise RuntimeError("no epsilon has a calibration candidate for every abstract variable")
    selected = plans[0]
    for candidate in plans[1:]:
        if float(candidate["mean_best_variable_calibration_score"]) > float(
            selected["mean_best_variable_calibration_score"]
        ):
            selected = candidate
    return {
        "selection_rule": "macro_average_of_each_variable_best_calibration_score",
        "selected_epsilon": float(selected["epsilon"]),
        "selected_by_variable": selected["selected_by_variable"],
        "epsilon_plans": plans,
    }
