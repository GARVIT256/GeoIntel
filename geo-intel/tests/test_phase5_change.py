"""Fixture-only tests for Phase 5 change processing and association math."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from geointel.change.association import area_adjusted_estimate, summarize_association
from geointel.change.ndvi_baseline import ndvi_difference_baseline
from geointel.change.postprocess import majority_filter, postprocess_class_map, remove_small_patches


def test_majority_filter_tie_retains_center_class() -> None:
    fixture = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.uint8)
    assert majority_filter(fixture, 3)[1, 1] == 1


def test_majority_filter_keeps_nodata_center() -> None:
    fixture = np.array([[2, 2, 2], [2, 0, 2], [2, 2, 2]], dtype=np.uint8)
    assert majority_filter(fixture, 3)[1, 1] == 0


def test_mmu_removes_small_patch_without_relabeling() -> None:
    fixture = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.uint8)
    result, metadata = remove_small_patches(fixture, pixel_size_m=10, minimum_area_ha=0.02)
    assert result[1, 1] == 0
    assert metadata["removed_pixels"] == 1
    assert metadata["removed_area_ha"] == pytest.approx(0.01)


def test_postprocess_majority_then_mmu() -> None:
    fixture = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.uint8)
    result, report = postprocess_class_map(fixture, 3, 0.02, 10)
    assert result[1, 1] == 1
    assert report["mmu"]["removed_pixels"] == 0


def test_ndvi_decline_baseline_masks_invalid_pixels() -> None:
    t1 = np.array([[0.6, 0.2], [np.nan, 0.4]], dtype="float32")
    t2 = np.array([[0.3, 0.1], [0.2, 0.5]], dtype="float32")
    result = ndvi_difference_baseline(t1, t2, threshold=-0.15)
    assert result["decline_mask"].tolist() == [[1, 0], [255, 0]]
    assert result["valid_pixels"] == 3


def test_area_adjusted_estimate_uses_stratified_counts() -> None:
    result = area_adjusted_estimate(
        np.array([60.0, 40.0]), np.array([[8, 2], [1, 9]])
    )
    assert result["reference_class_area_ha"] == pytest.approx([52.0, 48.0])
    assert result["confidence_interval_95_ha"][0][0] >= 0


def test_area_adjusted_estimate_requires_each_stratum_sampled() -> None:
    with pytest.raises(ValueError, match="sampled rows"):
        area_adjusted_estimate(np.array([10.0]), np.array([[0, 0]]))


def test_association_statistics_use_fixture_grid_and_return_hotspots() -> None:
    neighbors = {
        0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
        3: [0, 4], 4: [1, 3, 5], 5: [2, 4],
    }
    weights = SimpleNamespace(neighbors=neighbors)
    result = summarize_association(
        np.array([0, 1, 0, 1, 4, 1]), np.array([0, 1, 0, 1, 4, 1]),
        weights, alpha=0.99, top_n=2, permutations=19, seed=17,
    )
    assert result["grid_cells"] == 6
    assert result["spearman_urban_gain_vs_vegetation_loss"]["rho"] == pytest.approx(1.0)
    assert len(result["top_n_hotspots"]) == 2
    assert result["morans_i_vegetation_loss"]["I"] is not None
