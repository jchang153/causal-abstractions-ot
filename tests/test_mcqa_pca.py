from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcqa_experiment.intervention import apply_rotated_das_site_update
from mcqa_experiment.pca import LayerPCABasis, apply_rotated_component_update, fit_pca_basis_from_states
from mcqa_experiment.sites import enumerate_rotated_band_sites, enumerate_rotated_top_prefix_sites, site_total_width
from mcqa_experiment.support import build_rotated_span_sites_from_support, extract_ordered_site_support


def test_fit_pca_basis_from_states_returns_orthonormal_components():
    states = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 3.0],
            [2.0, 1.0, 4.0],
        ],
        dtype=torch.float32,
    )

    basis = fit_pca_basis_from_states(
        states=states,
        layer=20,
        token_position_id="last_token",
        basis_id="test-basis",
    )

    gram = basis.components.transpose(0, 1) @ basis.components
    assert basis.hidden_size == 3
    assert 1 <= basis.rank <= 3
    assert basis.mean.shape == (3,)
    assert torch.allclose(gram, torch.eye(basis.rank), atol=1e-5, rtol=1e-4)


def test_enumerate_rotated_band_sites_covers_full_rank_without_overlap():
    sites = enumerate_rotated_band_sites(
        rank=10,
        num_bands=4,
        layer=25,
        token_position_id="last_token",
        basis_id="basis",
    )

    assert len(sites) == 4
    assert sites[0].component_start == 0
    assert sites[-1].component_end == 10
    covered = []
    for site in sites:
        covered.extend(range(site.component_start, site.component_end))
        assert site_total_width(site, model_hidden_size=0) == site.component_end - site.component_start
    assert covered == list(range(10))


def test_enumerate_rotated_head_bands_split_head_more_finely():
    sites = enumerate_rotated_band_sites(
        rank=20,
        num_bands=4,
        layer=25,
        token_position_id="last_token",
        basis_id="basis",
        schedule="head",
    )

    widths = [site.component_end - site.component_start for site in sites]
    assert sum(widths) == 20
    assert widths[0] < widths[-1]
    assert widths == sorted(widths)


def test_enumerate_rotated_top_prefix_sites_clips_and_dedupes():
    sites = enumerate_rotated_top_prefix_sites(
        rank=20,
        prefix_sizes=(8, 16, 16, 32),
        layer=25,
        token_position_id="last_token",
        basis_id="basis",
    )

    assert [site.component_start for site in sites] == [0, 0, 0]
    assert [site.component_end for site in sites] == [8, 16, 20]


def test_apply_rotated_component_update_identity_basis_matches_coordinate_swap():
    identity = torch.eye(4, dtype=torch.float32)
    basis = LayerPCABasis(
        basis_id="identity",
        layer=20,
        token_position_id="last_token",
        hidden_size=4,
        rank=4,
        mean=torch.zeros(4, dtype=torch.float32),
        components=identity,
        singular_values=torch.ones(4, dtype=torch.float32),
        explained_variance=torch.ones(4, dtype=torch.float32),
        num_fit_states=8,
    )
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    source = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)

    updated = apply_rotated_component_update(
        base_vectors=base,
        source_vectors=source,
        basis=basis,
        component_segments=[(1, 3, 1.0)],
        strength=1.0,
    )

    expected = torch.tensor([[1.0, 20.0, 30.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(updated, expected)


def test_apply_rotated_component_update_preserves_unselected_pca_coordinates():
    basis = LayerPCABasis(
        basis_id="identity",
        layer=20,
        token_position_id="last_token",
        hidden_size=3,
        rank=3,
        mean=torch.zeros(3, dtype=torch.float32),
        components=torch.eye(3, dtype=torch.float32),
        singular_values=torch.ones(3, dtype=torch.float32),
        explained_variance=torch.ones(3, dtype=torch.float32),
        num_fit_states=6,
    )
    base = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    source = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)

    updated = apply_rotated_component_update(
        base_vectors=base,
        source_vectors=source,
        basis=basis,
        component_segments=[(0, 1, 1.0)],
        strength=1.0,
    )
    delta = updated - base

    assert torch.allclose(delta[:, 1:], torch.zeros_like(delta[:, 1:]))


def test_extract_ordered_site_support_and_build_rotated_spans():
    sites = enumerate_rotated_band_sites(
        rank=8,
        num_bands=4,
        layer=20,
        token_position_id="last_token",
        basis_id="basis",
    )
    ot_run_payloads = [
        {
            "method_payloads": {
                "ot": [
                    {
                        "target_var": "answer_pointer",
                        "source_target_vars": ["answer_pointer", "answer_token"],
                        "target_var_row_index": 0,
                        "transport": [[0.40, 0.30, 0.30, 0.00], [0.10, 0.10, 0.10, 0.70]],
                        "results": [{"selection_score": 0.61, "exact_acc": 0.75}],
                    },
                    {
                        "target_var": "answer_token",
                        "source_target_vars": ["answer_pointer", "answer_token"],
                        "target_var_row_index": 1,
                        "transport": [[0.40, 0.30, 0.30, 0.00], [0.10, 0.10, 0.10, 0.70]],
                        "results": [{"selection_score": 0.90, "exact_acc": 0.92}],
                    },
                ]
            }
        }
    ]

    support = extract_ordered_site_support(
        ot_run_payloads=ot_run_payloads,
        sites=sites,
        score_slack=0.05,
        prefix_sizes=(1, 2, 4),
        coverage_specs=(("S50", 0.50), ("S80", 0.80)),
    )

    pointer_support = support["answer_pointer"]
    assert pointer_support["ranked_site_labels"][0] == sites[0].label
    assert [candidate["name"] for candidate in pointer_support["mask_candidates"]] == ["Top1", "Top2", "Top4", "S80"]

    span_sites = build_rotated_span_sites_from_support(
        support_summary=pointer_support,
        sites=sites,
    )
    assert span_sites[0].label == f"L20:last_token:pc[{sites[0].component_start}:{sites[0].component_end}]"
    assert site_total_width(span_sites[1], model_hidden_size=0) == (
        site_total_width(sites[0], model_hidden_size=0) + site_total_width(sites[1], model_hidden_size=0)
    )


def test_build_rotated_span_sites_merge_overlapping_prefix_and_partition_masks():
    partition_sites = enumerate_rotated_band_sites(
        rank=64,
        num_bands=4,
        layer=20,
        token_position_id="last_token",
        basis_id="basis",
    )
    prefix_sites = enumerate_rotated_top_prefix_sites(
        rank=64,
        prefix_sizes=(32,),
        layer=20,
        token_position_id="last_token",
        basis_id="basis",
    )
    sites = [*partition_sites, *prefix_sites]
    support_summary = {
        "mask_candidates": [
            {
                "name": "Top2",
                "site_indices": [0, 4],
            }
        ]
    }

    span_sites = build_rotated_span_sites_from_support(
        support_summary=support_summary,
        sites=sites,
    )

    assert len(span_sites) == 1
    assert span_sites[0].label == "L20:last_token:pc[0:32]"
    assert site_total_width(span_sites[0], model_hidden_size=0) == 32


class _CopySourceIntervention(nn.Module):
    def forward(self, base_vectors: torch.Tensor, source_vectors: torch.Tensor) -> torch.Tensor:
        return source_vectors


def test_apply_rotated_das_site_update_replaces_only_selected_components():
    basis = LayerPCABasis(
        basis_id="identity",
        layer=20,
        token_position_id="last_token",
        hidden_size=4,
        rank=4,
        mean=torch.zeros(4, dtype=torch.float32),
        components=torch.eye(4, dtype=torch.float32),
        singular_values=torch.ones(4, dtype=torch.float32),
        explained_variance=torch.ones(4, dtype=torch.float32),
        num_fit_states=8,
    )
    site = enumerate_rotated_top_prefix_sites(
        rank=4,
        prefix_sizes=(2,),
        layer=20,
        token_position_id="last_token",
        basis_id="identity",
    )[0]
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    source = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)

    updated = apply_rotated_das_site_update(
        base_vectors=base,
        source_vectors=source,
        basis=basis,
        site=site,
        intervention=_CopySourceIntervention(),
    )

    expected = torch.tensor([[10.0, 20.0, 3.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(updated, expected)
