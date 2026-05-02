"""Support extraction utilities for OT-guided MCQA DAS runs."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .sites import (
    ResidualCompositeSite,
    ResidualSite,
    ResidualUnionSite,
    RotatedBandSite,
    RotatedCompositeSite,
    SiteLike,
    site_total_width,
)


def _position_mass_by_var(
    transport: np.ndarray,
    sites: list[ResidualSite],
    source_target_vars: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    position_mass_by_var = {
        str(target_var): {}
        for target_var in source_target_vars
    }
    for row_index, target_var in enumerate(source_target_vars):
        for site_index, site in enumerate(sites):
            position_mass_by_var[str(target_var)].setdefault(str(site.token_position_id), 0.0)
            position_mass_by_var[str(target_var)][str(site.token_position_id)] += float(transport[row_index, site_index])
    return position_mass_by_var


def _enumerate_ranked_unions(ranked_positions: tuple[str, ...]) -> list[tuple[str, ...]]:
    unions: list[tuple[str, ...]] = []
    for union_size in range(1, len(ranked_positions) + 1):
        unions.extend(tuple(position for position in subset) for subset in combinations(ranked_positions, union_size))
    return unions


def _block_mask_from_prefix(
    ranked_site_indices: tuple[int, ...],
    evidence_by_site: dict[int, float],
    *,
    coverage: float,
) -> tuple[int, ...]:
    total_evidence = sum(float(score) for score in evidence_by_site.values() if float(score) > 0.0)
    if total_evidence <= 0.0:
        return ranked_site_indices[:1]
    threshold = float(coverage) * float(total_evidence)
    selected: list[int] = []
    captured = 0.0
    for site_index in ranked_site_indices:
        if float(evidence_by_site.get(int(site_index), 0.0)) <= 0.0:
            continue
        selected.append(int(site_index))
        captured += float(evidence_by_site[int(site_index)])
        if captured >= threshold:
            break
    return tuple(selected or ranked_site_indices[:1])


def _merged_rotated_segments(segments: tuple[RotatedBandSite, ...]) -> tuple[RotatedBandSite, ...]:
    if not segments:
        return ()
    first_segment = segments[0]
    if not all(
        int(segment.layer) == int(first_segment.layer)
        and str(segment.token_position_id) == str(first_segment.token_position_id)
        and str(segment.basis_id) == str(first_segment.basis_id)
        for segment in segments
    ):
        raise ValueError("Rotated support masks must stay within one layer/token-position/basis")
    ordered = sorted(
        {
            (int(segment.component_start), int(segment.component_end))
            for segment in segments
        }
    )
    merged: list[tuple[int, int]] = []
    for component_start, component_end in ordered:
        if not merged or int(component_start) > int(merged[-1][1]):
            merged.append((int(component_start), int(component_end)))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (int(previous_start), max(int(previous_end), int(component_end)))
    return tuple(
        RotatedBandSite(
            layer=int(first_segment.layer),
            token_position_id=str(first_segment.token_position_id),
            basis_id=str(first_segment.basis_id),
            component_start=int(component_start),
            component_end=int(component_end),
        )
        for component_start, component_end in merged
    )


def _site_mask_total_dim(*, sites: list[SiteLike], site_mask: tuple[int, ...]) -> int:
    selected_sites = tuple(sites[site_index] for site_index in site_mask)
    if selected_sites and all(isinstance(site, RotatedBandSite) for site in selected_sites):
        merged_segments = _merged_rotated_segments(tuple(selected_sites))
        return sum(
            int(segment.component_end) - int(segment.component_start)
            for segment in merged_segments
        )
    return sum(
        int(site_total_width(sites[site_index], model_hidden_size=0))
        for site_index in site_mask
    )


def _mask_candidates_from_ordered_support(
    *,
    ranked_site_indices: tuple[int, ...],
    evidence_by_site: dict[int, float],
    sites: list[SiteLike],
    prefix_sizes: tuple[int, ...],
    coverage_specs: tuple[tuple[str, float], ...],
) -> list[dict[str, object]]:
    mask_candidates: list[dict[str, object]] = []
    seen_masks: set[tuple[int, ...]] = set()
    for count in prefix_sizes:
        if int(count) <= 0:
            continue
        site_mask = tuple(int(site_index) for site_index in ranked_site_indices[: min(int(count), len(ranked_site_indices))])
        if not site_mask or site_mask in seen_masks:
            continue
        seen_masks.add(site_mask)
        mask_candidates.append(
            {
                "name": f"Top{int(count)}",
                "site_indices": list(site_mask),
                "site_labels": [sites[site_index].label for site_index in site_mask],
                "site_total_dim": int(_site_mask_total_dim(sites=sites, site_mask=site_mask)),
            }
        )
    for name, coverage in coverage_specs:
        site_mask = _block_mask_from_prefix(
            ranked_site_indices,
            evidence_by_site,
            coverage=float(coverage),
        )
        if not site_mask or site_mask in seen_masks:
            continue
        seen_masks.add(site_mask)
        mask_candidates.append(
            {
                "name": str(name),
                "site_indices": list(site_mask),
                "site_labels": [sites[site_index].label for site_index in site_mask],
                "site_total_dim": int(_site_mask_total_dim(sites=sites, site_mask=site_mask)),
            }
        )
    return mask_candidates


def extract_ordered_site_support(
    *,
    ot_run_payloads: list[dict[str, object]],
    sites: list[SiteLike],
    score_slack: float = 0.05,
    prefix_sizes: tuple[int, ...] = (1, 2, 4),
    coverage_specs: tuple[tuple[str, float], ...] = (("S50", 0.50), ("S80", 0.80)),
) -> dict[str, dict[str, object]]:
    """Pool near-best OT runs into row-dominant support over an ordered site catalog."""
    grouped_payloads: dict[str, list[dict[str, object]]] = {}
    for compare_payload in ot_run_payloads:
        method_payloads = compare_payload.get("method_payloads", {})
        for payload in method_payloads.get("ot", []):
            grouped_payloads.setdefault(str(payload.get("target_var")), []).append(payload)

    support_by_var: dict[str, dict[str, object]] = {}
    for target_var, payloads in grouped_payloads.items():
        if not payloads:
            continue
        selection_scores = [
            float(payload.get("results", [{}])[0].get("selection_score", 0.0))
            for payload in payloads
        ]
        best_score = max(selection_scores) if selection_scores else 0.0
        score_floor = float(best_score) - float(score_slack)
        kept_payloads = [
            payload
            for payload in payloads
            if float(payload.get("results", [{}])[0].get("selection_score", 0.0)) >= score_floor
        ]
        if not kept_payloads:
            kept_payloads = [
                max(payloads, key=lambda payload: float(payload.get("results", [{}])[0].get("selection_score", 0.0)))
            ]

        site_indices = tuple(range(len(sites)))
        evidence_by_site = {site_index: 0.0 for site_index in site_indices}
        mean_target_mass_by_site = {site_index: 0.0 for site_index in site_indices}
        source_target_vars = tuple(str(target) for target in kept_payloads[0].get("source_target_vars", []))
        if not source_target_vars:
            source_target_vars = (str(target_var),)
        for payload in kept_payloads:
            transport = np.asarray(payload.get("transport", []), dtype=float)
            if transport.ndim != 2 or transport.shape[1] != len(sites):
                continue
            row_sums = transport.sum(axis=1, keepdims=True)
            safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
            row_mass = transport / safe_row_sums
            target_row_index = int(payload.get("target_var_row_index", source_target_vars.index(str(target_var))))
            for site_index in site_indices:
                target_mass = float(row_mass[target_row_index, site_index])
                competitor_mass = max(
                    [
                        float(row_mass[other_index, site_index])
                        for other_index in range(row_mass.shape[0])
                        if int(other_index) != int(target_row_index)
                    ]
                    or [0.0]
                )
                evidence_by_site[int(site_index)] += max(target_mass - competitor_mass, 0.0)
                mean_target_mass_by_site[int(site_index)] += target_mass
        if kept_payloads:
            mean_target_mass_by_site = {
                site_index: float(total_mass / len(kept_payloads))
                for site_index, total_mass in mean_target_mass_by_site.items()
            }
        ranked_site_indices = tuple(
            site_index
            for site_index, _score in sorted(
                evidence_by_site.items(),
                key=lambda item: (
                    float(item[1]),
                    float(mean_target_mass_by_site.get(int(item[0]), 0.0)),
                    -int(item[0]),
                ),
                reverse=True,
            )
        )
        if not ranked_site_indices:
            continue
        support_by_var[str(target_var)] = {
            "target_var": str(target_var),
            "best_selection_score": float(best_score),
            "score_slack": float(score_slack),
            "kept_trial_count": len(kept_payloads),
            "kept_trials": [
                {
                    "selection_score": float(payload.get("results", [{}])[0].get("selection_score", 0.0)),
                    "exact_acc": float(payload.get("results", [{}])[0].get("exact_acc", 0.0)),
                    "epsilon": float(payload.get("transport_meta", {}).get("epsilon_config", payload.get("ot_epsilon", 0.0))),
                }
                for payload in kept_payloads
            ],
            "site_evidence": {
                sites[site_index].label: float(evidence_by_site[site_index])
                for site_index in ranked_site_indices
            },
            "mean_target_site_mass": {
                sites[site_index].label: float(mean_target_mass_by_site[site_index])
                for site_index in ranked_site_indices
            },
            "ranked_site_indices": [int(site_index) for site_index in ranked_site_indices],
            "ranked_site_labels": [sites[site_index].label for site_index in ranked_site_indices],
            "mask_candidates": _mask_candidates_from_ordered_support(
                ranked_site_indices=ranked_site_indices,
                evidence_by_site=evidence_by_site,
                sites=sites,
                prefix_sizes=prefix_sizes,
                coverage_specs=coverage_specs,
            ),
        }
    return support_by_var


def extract_layer_position_support(
    *,
    ot_run_payloads: list[dict[str, object]],
    sites: list[ResidualSite],
    score_slack: float = 0.05,
) -> dict[str, dict[str, object]]:
    """Pool near-best OT full-vector runs into row-dominant layer-position support."""
    grouped_payloads: dict[str, list[dict[str, object]]] = {}
    for compare_payload in ot_run_payloads:
        method_payloads = compare_payload.get("method_payloads", {})
        for payload in method_payloads.get("ot", []):
            grouped_payloads.setdefault(str(payload.get("target_var")), []).append(payload)

    support_by_var: dict[str, dict[str, object]] = {}
    all_positions = tuple(dict.fromkeys(str(site.token_position_id) for site in sites))
    for target_var, payloads in grouped_payloads.items():
        if not payloads:
            continue
        selection_scores = [
            float(payload.get("results", [{}])[0].get("selection_score", 0.0))
            for payload in payloads
        ]
        best_score = max(selection_scores) if selection_scores else 0.0
        score_floor = float(best_score) - float(score_slack)
        kept_payloads = [
            payload
            for payload in payloads
            if float(payload.get("results", [{}])[0].get("selection_score", 0.0)) >= score_floor
        ]
        if not kept_payloads:
            kept_payloads = [max(payloads, key=lambda payload: float(payload.get("results", [{}])[0].get("selection_score", 0.0)))]

        evidence_by_position = {position: 0.0 for position in all_positions}
        mean_target_mass_by_position = {position: 0.0 for position in all_positions}
        total_weight = 0.0
        source_target_vars = tuple(str(target) for target in kept_payloads[0].get("source_target_vars", []))
        if not source_target_vars:
            source_target_vars = (str(target_var),)
        for payload in kept_payloads:
            transport = np.asarray(payload.get("normalized_transport", []), dtype=float)
            if transport.ndim != 2 or transport.shape[1] != len(sites):
                continue
            position_mass = _position_mass_by_var(transport, sites, source_target_vars)
            weight = float(payload.get("results", [{}])[0].get("selection_score", 0.0))
            if weight <= 0.0:
                weight = 1.0
            total_weight += weight
            for position in all_positions:
                target_mass = float(position_mass.get(str(target_var), {}).get(position, 0.0))
                competitor_mass = max(
                    [
                        float(position_mass.get(str(other_var), {}).get(position, 0.0))
                        for other_var in source_target_vars
                        if str(other_var) != str(target_var)
                    ]
                    or [0.0]
                )
                evidence_by_position[position] += weight * max(target_mass - competitor_mass, 0.0)
                mean_target_mass_by_position[position] += weight * target_mass
        if total_weight > 0.0:
            mean_target_mass_by_position = {
                position: float(target_mass / total_weight)
                for position, target_mass in mean_target_mass_by_position.items()
            }
        ranked_positions = tuple(
            position
            for position, _score in sorted(
                evidence_by_position.items(),
                key=lambda item: (float(item[1]), float(mean_target_mass_by_position.get(item[0], 0.0))),
                reverse=True,
            )
        )
        active_positions = tuple(position for position in ranked_positions if float(evidence_by_position[position]) > 0.0)
        if not active_positions and ranked_positions:
            active_positions = (ranked_positions[0],)
        support_by_var[str(target_var)] = {
            "target_var": str(target_var),
            "best_selection_score": float(best_score),
            "score_slack": float(score_slack),
            "kept_trial_count": len(kept_payloads),
            "kept_trials": [
                {
                    "selection_score": float(payload.get("results", [{}])[0].get("selection_score", 0.0)),
                    "exact_acc": float(payload.get("results", [{}])[0].get("exact_acc", 0.0)),
                    "epsilon": float(payload.get("transport_meta", {}).get("epsilon_config", payload.get("results", [{}])[0].get("epsilon", 0.0))),
                }
                for payload in kept_payloads
            ],
            "position_evidence": {position: float(score) for position, score in evidence_by_position.items()},
            "mean_target_position_mass": mean_target_mass_by_position,
            "ranked_positions": list(ranked_positions),
            "active_positions": list(active_positions),
            "candidate_unions": [list(union) for union in _enumerate_ranked_unions(active_positions)],
        }
    return support_by_var


def extract_block_mask_support(
    *,
    ot_run_payloads: list[dict[str, object]],
    sites: list[ResidualSite],
    score_slack: float = 0.05,
) -> dict[str, dict[str, object]]:
    """Pool near-best OT block runs into row-dominant within-layer block masks."""
    grouped_payloads: dict[str, list[dict[str, object]]] = {}
    for compare_payload in ot_run_payloads:
        method_payloads = compare_payload.get("method_payloads", {})
        for payload in method_payloads.get("ot", []):
            grouped_payloads.setdefault(str(payload.get("target_var")), []).append(payload)

    support_by_var: dict[str, dict[str, object]] = {}
    for target_var, payloads in grouped_payloads.items():
        if not payloads:
            continue
        selection_scores = [
            float(payload.get("results", [{}])[0].get("selection_score", 0.0))
            for payload in payloads
        ]
        best_score = max(selection_scores) if selection_scores else 0.0
        score_floor = float(best_score) - float(score_slack)
        kept_payloads = [
            payload
            for payload in payloads
            if float(payload.get("results", [{}])[0].get("selection_score", 0.0)) >= score_floor
        ]
        if not kept_payloads:
            kept_payloads = [max(payloads, key=lambda payload: float(payload.get("results", [{}])[0].get("selection_score", 0.0)))]

        site_indices = tuple(range(len(sites)))
        evidence_by_site = {site_index: 0.0 for site_index in site_indices}
        mean_target_mass_by_site = {site_index: 0.0 for site_index in site_indices}
        source_target_vars = tuple(str(target) for target in kept_payloads[0].get("source_target_vars", []))
        if not source_target_vars:
            source_target_vars = (str(target_var),)
        for payload in kept_payloads:
            transport = np.asarray(payload.get("transport", []), dtype=float)
            if transport.ndim != 2 or transport.shape[1] != len(sites):
                continue
            row_sums = transport.sum(axis=1, keepdims=True)
            safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
            row_mass = transport / safe_row_sums
            target_row_index = int(payload.get("target_var_row_index", source_target_vars.index(str(target_var))))
            for site_index in site_indices:
                target_mass = float(row_mass[target_row_index, site_index])
                competitor_mass = max(
                    [
                        float(row_mass[other_index, site_index])
                        for other_index in range(row_mass.shape[0])
                        if int(other_index) != int(target_row_index)
                    ]
                    or [0.0]
                )
                evidence_by_site[int(site_index)] += max(target_mass - competitor_mass, 0.0)
                mean_target_mass_by_site[int(site_index)] += target_mass
        if kept_payloads:
            mean_target_mass_by_site = {
                site_index: float(total_mass / len(kept_payloads))
                for site_index, total_mass in mean_target_mass_by_site.items()
            }
        ranked_site_indices = tuple(
            site_index
            for site_index, _score in sorted(
                evidence_by_site.items(),
                key=lambda item: (
                    float(item[1]),
                    float(mean_target_mass_by_site.get(int(item[0]), 0.0)),
                    -int(item[0]),
                ),
                reverse=True,
            )
        )
        if not ranked_site_indices:
            continue
        mask_candidates: list[dict[str, object]] = []
        seen_masks: set[tuple[int, ...]] = set()
        fixed_prefix_specs = (
            ("Top1", 1),
            ("Top2", 2),
            ("Top4", 4),
        )
        for name, count in fixed_prefix_specs:
            site_mask = tuple(int(site_index) for site_index in ranked_site_indices[: min(int(count), len(ranked_site_indices))])
            if not site_mask or site_mask in seen_masks:
                continue
            seen_masks.add(site_mask)
            mask_candidates.append(
                {
                    "name": str(name),
                    "site_indices": list(site_mask),
                    "site_labels": [sites[site_index].label for site_index in site_mask],
                    "site_total_dim": sum(int(sites[site_index].dim_end) - int(sites[site_index].dim_start) for site_index in site_mask),
                }
            )
        for name, coverage in (("S50", 0.50), ("S80", 0.80)):
            site_mask = _block_mask_from_prefix(
                ranked_site_indices,
                evidence_by_site,
                coverage=float(coverage),
            )
            if not site_mask or site_mask in seen_masks:
                continue
            seen_masks.add(site_mask)
            mask_candidates.append(
                {
                    "name": str(name),
                    "site_indices": list(site_mask),
                    "site_labels": [sites[site_index].label for site_index in site_mask],
                    "site_total_dim": sum(int(sites[site_index].dim_end) - int(sites[site_index].dim_start) for site_index in site_mask),
                }
            )
        support_by_var[str(target_var)] = {
            "target_var": str(target_var),
            "best_selection_score": float(best_score),
            "score_slack": float(score_slack),
            "kept_trial_count": len(kept_payloads),
            "kept_trials": [
                {
                    "selection_score": float(payload.get("results", [{}])[0].get("selection_score", 0.0)),
                    "exact_acc": float(payload.get("results", [{}])[0].get("exact_acc", 0.0)),
                    "epsilon": float(payload.get("transport_meta", {}).get("epsilon_config", payload.get("ot_epsilon", 0.0))),
                }
                for payload in kept_payloads
            ],
            "site_evidence": {
                sites[site_index].label: float(evidence_by_site[site_index])
                for site_index in ranked_site_indices
            },
            "mean_target_site_mass": {
                sites[site_index].label: float(mean_target_mass_by_site[site_index])
                for site_index in ranked_site_indices
            },
            "ranked_site_indices": [int(site_index) for site_index in ranked_site_indices],
            "ranked_site_labels": [sites[site_index].label for site_index in ranked_site_indices],
            "mask_candidates": mask_candidates,
        }
    return support_by_var


def build_union_sites_from_support(
    *,
    layer: int,
    hidden_size: int,
    support_summary: dict[str, object],
) -> list[ResidualUnionSite]:
    candidate_unions = support_summary.get("candidate_unions", [])
    sites: list[ResidualUnionSite] = []
    seen: set[tuple[str, ...]] = set()
    for union in candidate_unions:
        token_position_ids = tuple(str(token_position_id) for token_position_id in union)
        if not token_position_ids or token_position_ids in seen:
            continue
        seen.add(token_position_ids)
        sites.append(
            ResidualUnionSite(
                layer=int(layer),
                token_position_ids=token_position_ids,
                hidden_size=int(hidden_size),
            )
        )
    return sites


def build_mask_sites_from_support(
    *,
    support_summary: dict[str, object],
    sites: list[ResidualSite],
) -> list[ResidualCompositeSite]:
    mask_candidates = support_summary.get("mask_candidates", [])
    composite_sites: list[ResidualCompositeSite] = []
    seen_masks: set[tuple[int, ...]] = set()
    for candidate in mask_candidates:
        site_indices = tuple(int(site_index) for site_index in candidate.get("site_indices", []))
        if not site_indices or site_indices in seen_masks:
            continue
        seen_masks.add(site_indices)
        segments = tuple(sites[site_index] for site_index in site_indices)
        composite_sites.append(
            ResidualCompositeSite(
                layer=int(segments[0].layer),
                segments=segments,
            )
        )
    return composite_sites


def build_ordered_composite_sites_from_support(
    *,
    support_summary: dict[str, object],
    sites: list[SiteLike],
) -> list[SiteLike]:
    """Build composite sites from ordered support masks over atomic site catalogs."""
    mask_candidates = support_summary.get("mask_candidates", [])
    composite_sites: list[SiteLike] = []
    seen_masks: set[tuple[int, ...]] = set()
    for candidate in mask_candidates:
        site_indices = tuple(int(site_index) for site_index in candidate.get("site_indices", []))
        if not site_indices or site_indices in seen_masks:
            continue
        seen_masks.add(site_indices)
        segments = tuple(sites[site_index] for site_index in site_indices)
        first_segment = segments[0]
        if isinstance(first_segment, RotatedBandSite):
            if not all(isinstance(segment, RotatedBandSite) for segment in segments):
                raise ValueError("Rotated composite support masks must contain only RotatedBandSite segments")
            merged_segments = _merged_rotated_segments(tuple(segments))
            composite_sites.append(
                RotatedCompositeSite(
                    layer=int(first_segment.layer),
                    token_position_id=str(first_segment.token_position_id),
                    basis_id=str(first_segment.basis_id),
                    segments=tuple(merged_segments),
                )
            )
            continue
        if not all(isinstance(segment, ResidualSite) for segment in segments):
            raise ValueError("Residual composite support masks must contain only ResidualSite segments")
        composite_sites.append(
            ResidualCompositeSite(
                layer=int(first_segment.layer),
                segments=tuple(segments),
            )
        )
    return composite_sites


def build_atomic_candidate_sites_from_support(
    *,
    support_summary: dict[str, object],
    sites: list[SiteLike],
) -> list[SiteLike]:
    """Return the unique atomic sites referenced by ordered support masks."""
    mask_candidates = support_summary.get("mask_candidates", [])
    selected_indices: list[int] = []
    seen_indices: set[int] = set()
    for candidate in mask_candidates:
        for site_index in candidate.get("site_indices", []):
            resolved_index = int(site_index)
            if resolved_index in seen_indices:
                continue
            seen_indices.add(resolved_index)
            selected_indices.append(resolved_index)
    if not selected_indices:
        ranked_site_indices = [int(site_index) for site_index in support_summary.get("ranked_site_indices", [])]
        if ranked_site_indices:
            selected_indices = [ranked_site_indices[0]]
    return [
        sites[site_index]
        for site_index in selected_indices
        if 0 <= int(site_index) < len(sites)
    ]


def build_rotated_span_sites_from_support(
    *,
    support_summary: dict[str, object],
    sites: list[RotatedBandSite],
) -> list[RotatedCompositeSite]:
    composite_sites = build_ordered_composite_sites_from_support(
        support_summary=support_summary,
        sites=sites,
    )
    if not all(isinstance(site, RotatedCompositeSite) for site in composite_sites):
        raise ValueError("Expected only rotated composite sites from rotated support")
    return list(composite_sites)
