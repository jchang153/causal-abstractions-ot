from __future__ import annotations

from pathlib import Path
import sys


MCQA_DIR = Path(__file__).resolve().parent
if str(MCQA_DIR) not in sys.path:
    sys.path.insert(0, str(MCQA_DIR))

from mcqa_delta_hierarchical_sweep import (
    DEFAULT_PCA_NUM_BANDS_VALUES as HIERARCHICAL_PCA_NUM_BANDS_VALUES,
    _dim_hint_subspace_dims,
)
from mcqa_ot_pca_focus import (
    DEFAULT_NUM_BANDS_VALUES as DIRECT_PCA_NUM_BANDS_VALUES,
    _dimension_hint_subspace_dims,
    _pca_effective_dims,
)


def test_pca_band_sweep_includes_coarse_and_fine_partitions() -> None:
    expected = (1, 2, 4, 8, 16, 32, 64)
    assert HIERARCHICAL_PCA_NUM_BANDS_VALUES == expected
    assert DIRECT_PCA_NUM_BANDS_VALUES == expected


def test_native_dimension_hint_uses_paper_scale_factors() -> None:
    assert _dim_hint_subspace_dims(128, max_dim=2304) == (64, 96, 128, 192, 256)
    assert _dim_hint_subspace_dims(16, max_dim=2304) == (8, 12, 16, 24, 32)


def test_pca_dimension_hint_allows_small_dims_and_clamps_to_rank() -> None:
    assert _dimension_hint_subspace_dims(11, max_width=189) == (6, 8, 11, 16, 22)
    assert _dimension_hint_subspace_dims(120, max_width=189) == (60, 90, 120, 180, 189)


def test_pca_effective_dimension_matches_paper_formula() -> None:
    support = {"answer_pointer": {"selected_trial": {"top_k": 4}}}
    assert _pca_effective_dims(support, rank=189, num_bands=8) == {"answer_pointer": 92}
