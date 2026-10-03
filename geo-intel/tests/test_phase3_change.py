"""
tests/test_phase3_change.py — Unit test suite for Phase 3 Land Cover Classification & Change Detection.

Tests use synthetic raster data with analytically verified expected values:
- Spectral index calculations (NDVI, NDBI, MNDWI, BSI) & bounds [-1.0, 1.0]
- Zero-denominator division safety (returns NaN, no division-by-zero runtime error)
- Rule-based decision-tree classifier (Baseline 1)
- Random Forest feature extraction & prediction pipeline (Baseline 2)
- LULC 4x4 transition matrix & spatial change mask derivation

Run with:
    pytest tests/test_phase3_change.py
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geointel.change.detection import compute_spatial_change_masks, compute_transition_matrix
from geointel.change.indices import (
    compute_all_indices,
    compute_bsi,
    compute_mndwi,
    compute_ndbi,
    compute_ndvi,
)
from geointel.change.rf_classifier import (
    extract_feature_stack,
    generate_pseudo_training_data,
    predict_rf_lulc,
    train_rf_classifier,
)
from geointel.change.rule_based import classify_rule_based


@pytest.fixture
def synthetic_composite() -> xr.DataArray:
    """
    Synthetic 6-band Sentinel-2 composite DataArray (band, y, x) shape (6, 10, 10).
    Grid locations:
    - (0..4, 0..4): Vegetation signatures (High NIR, low Red)
    - (0..4, 5..9): Water signatures (High Green, low SWIR)
    - (5..9, 0..4): Built-up signatures (High SWIR1, low NIR)
    - (5..9, 5..9): Bare Soil signatures (High SWIR1 + Red)
    """
    data = np.zeros((6, 10, 10), dtype=np.float32)

    # 1. Vegetation (top-left 5x5)
    data[0, 0:5, 0:5] = 500.0   # B02 Blue
    data[1, 0:5, 0:5] = 800.0   # B03 Green
    data[2, 0:5, 0:5] = 600.0   # B04 Red
    data[3, 0:5, 0:5] = 4000.0  # B08 NIR (High)
    data[4, 0:5, 0:5] = 1200.0  # B11 SWIR1
    data[5, 0:5, 0:5] = 1000.0  # B12 SWIR2

    # 2. Water (top-right 5x5)
    data[0, 0:5, 5:10] = 1500.0  # B02
    data[1, 0:5, 5:10] = 2000.0  # B03 Green (High)
    data[2, 0:5, 5:10] = 800.0   # B04
    data[3, 0:5, 5:10] = 300.0   # B08 NIR (Very low)
    data[4, 0:5, 5:10] = 100.0   # B11 SWIR1 (Very low)
    data[5, 0:5, 5:10] = 50.0    # B12

    # 3. Built-up (bottom-left 5x5)
    data[0, 5:10, 0:5] = 1200.0  # B02
    data[1, 5:10, 0:5] = 1400.0  # B03
    data[2, 5:10, 0:5] = 1600.0  # B04
    data[3, 5:10, 0:5] = 2000.0  # B08 NIR
    data[4, 5:10, 0:5] = 3200.0  # B11 SWIR1 (High > NIR)
    data[5, 5:10, 0:5] = 2800.0  # B12

    # 4. Bare Soil (bottom-right 5x5)
    data[0, 5:10, 5:10] = 1000.0  # B02
    data[1, 5:10, 5:10] = 1500.0  # B03
    data[2, 5:10, 5:10] = 2200.0  # B04 Red (High)
    data[3, 5:10, 5:10] = 2400.0  # B08 NIR
    data[4, 5:10, 5:10] = 3500.0  # B11 SWIR1 (High)
    data[5, 5:10, 5:10] = 3000.0  # B12

    return xr.DataArray(
        data,
        dims=["band", "y", "x"],
        coords={
            "band": ["B02", "B03", "B04", "B08", "B11", "B12"],
            "y": np.arange(10),
            "x": np.arange(10),
        },
        attrs={"crs": "EPSG:32644"},
    )


# ---------------------------------------------------------------------------
# Test Spectral Indices
# ---------------------------------------------------------------------------

class TestSpectralIndices:

    def test_ndvi_known_values(self) -> None:
        nir = np.array([4000.0, 1000.0], dtype=np.float32)
        red = np.array([1000.0, 1000.0], dtype=np.float32)
        ndvi = compute_ndvi(nir, red)
        # Expected: (4000-1000)/(4000+1000) = 3000/5000 = 0.6
        # Expected: (1000-1000)/(1000+1000) = 0.0
        np.testing.assert_allclose(ndvi, [0.6, 0.0], atol=1e-3)

    def test_zero_denominator_returns_nan(self) -> None:
        b1 = np.array([0.0, 500.0], dtype=np.float32)
        b2 = np.array([0.0, -500.0], dtype=np.float32)
        res = compute_ndvi(b1, b2)
        assert np.isnan(res[0]), "Zero denominator should yield NaN"
        assert np.isnan(res[1]), "Zero denominator should yield NaN"

    def test_indices_range_clipping(self) -> None:
        # Values that would produce out-of-bounds results
        b1 = np.array([10000.0], dtype=np.float32)
        b2 = np.array([-2000.0], dtype=np.float32)
        res = compute_ndvi(b1, b2)
        assert res[0] <= 1.0 and res[0] >= -1.0, f"Index out of bounds: {res[0]}"

    def test_compute_all_indices_shape(self, synthetic_composite: xr.DataArray) -> None:
        idx_stack = compute_all_indices(synthetic_composite)
        assert idx_stack.shape == (4, 10, 10)
        assert list(idx_stack.coords["index_band"].values) == ["NDVI", "NDBI", "MNDWI", "BSI"]


# ---------------------------------------------------------------------------
# Test Rule-based Classifier
# ---------------------------------------------------------------------------

class TestRuleBasedClassifier:

    def test_rule_based_classification_quadrants(self, synthetic_composite: xr.DataArray) -> None:
        lulc = classify_rule_based(synthetic_composite)
        arr = lulc.values

        # Check quadrant class assignments
        # Top-left (0..5, 0..5) should be Vegetation (2)
        assert (arr[0:5, 0:5] == 2).all(), "Top-left quadrant should be classified as Vegetation"

        # Top-right (0..5, 5..10) should be Water (5)
        assert (arr[0:5, 5:10] == 5).all(), "Top-right quadrant should be classified as Water"

        # Bottom-left (5..10, 0..5) should be Built-up (1)
        assert (arr[5:10, 0:5] == 1).all(), "Bottom-left quadrant should be classified as Built-up"

    def test_low_vegetation_is_separated_and_bare_rule_is_thresholded(self) -> None:
        from geointel.change.rule_based import classify_rule_based

        indices = xr.DataArray(
            np.array([
                [[0.40, 0.10]],
                [[-0.10, -0.10]],
                [[-0.10, -0.10]],
                [[-0.20, -0.20]],
            ], dtype=np.float32),
            dims=["index_band", "y", "x"],
            coords={"index_band": ["NDVI", "NDBI", "MNDWI", "BSI"]},
        )

        result = classify_rule_based(indices)
        np.testing.assert_array_equal(result.values, [[3, 0]])

        indices.loc[dict(index_band="BSI", x=1)] = 0.2
        result = classify_rule_based(indices)
        np.testing.assert_array_equal(result.values, [[3, 4]])


# ---------------------------------------------------------------------------
# Test Random Forest Classifier
# ---------------------------------------------------------------------------

class TestRFClassifier:

    def test_feature_stack_shape(self, synthetic_composite: xr.DataArray) -> None:
        X, (h, w) = extract_feature_stack(synthetic_composite)
        assert X.shape == (100, 13), f"Expected shape (100, 13), got {X.shape}"
        assert h == 10 and w == 10

    def test_aspect_encoding_wraps_at_north(self, synthetic_composite: xr.DataArray) -> None:
        aspect = np.full((10, 10), 359.0, dtype=np.float32)
        aspect[0, 1] = 1.0

        features, _ = extract_feature_stack(synthetic_composite, aspect=aspect)

        assert np.linalg.norm(features[0, 11:13] - features[1, 11:13]) < 0.04

    def test_pseudo_training_data_generation(self, synthetic_composite: xr.DataArray) -> None:
        X_tr, y_tr = generate_pseudo_training_data(synthetic_composite, n_samples_per_class=10)
        assert X_tr.shape[1] == 13
        assert len(X_tr) == len(y_tr)
        assert set(np.unique(y_tr)).issubset({1, 2, 3, 4, 5})

    def test_rf_train_and_predict(self, synthetic_composite: xr.DataArray) -> None:
        from geointel.change.rf_classifier import SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn native extensions are blocked in this interpreter")

        X_tr, y_tr = generate_pseudo_training_data(synthetic_composite, n_samples_per_class=15)
        model, metrics = train_rf_classifier(X_tr, y_tr, n_estimators=20)
        
        assert "oob_score" in metrics
        assert "cv_accuracy_mean" in metrics
        assert metrics["validation_status"].startswith("bootstrap only")
        assert metrics["cv_accuracy_mean"] >= 0.50

        rf_map = predict_rf_lulc(model, synthetic_composite)
        assert rf_map.shape == (10, 10)
        assert set(np.unique(rf_map.values)).issubset({1, 2, 3, 4, 5})


# ---------------------------------------------------------------------------
# Test Change Detection & Transition Matrix
# ---------------------------------------------------------------------------

class TestChangeDetection:

    def test_transition_matrix_analytical(self) -> None:
        # Create known 5x5 five-class LULC maps
        t1 = np.full((5, 5), 2, dtype=np.uint8)  # All tree/dense vegetation
        t2 = np.full((5, 5), 2, dtype=np.uint8)
        
        # 5 pixels change to Built-up (3) in T2
        t2[0, :] = 1

        result = compute_transition_matrix(t1, t2, pixel_size_m=10.0)
        matrix = np.array(result["matrix_pixels"])

        # Row 2 (tree class 2): 20 remain trees, 5 become built-up (class 1).
        assert matrix[1, 1] == 20, f"Expected 20 unchanged vegetation, got {matrix[1, 1]}"
        assert matrix[1, 0] == 5, f"Expected 5 vegetation -> built, got {matrix[1, 0]}"

        # Area conversion check (5 pixels * 100 m² = 500 m² = 0.0005 km²)
        assert result["transitions_km2"]["vegetation_to_builtup_km2"] == 0.0005

    def test_joint_nodata_mask_excludes_nan_and_zero(self) -> None:
        t1 = np.array([[2.0, np.nan, 0.0, 5.0]])
        t2 = np.array([[3.0, 2.0, 4.0, 5.0]])

        result = compute_transition_matrix(t1, t2)

        assert result["total_valid_area_km2"] == pytest.approx(0.0002)
        assert sum(map(sum, result["matrix_pixels"])) == 2

    def test_spatial_change_masks(self) -> None:
        t1 = np.array([[2, 2], [4, 1]], dtype=np.uint8)
        t2 = np.array([[2, 1], [1, 1]], dtype=np.uint8)

        masks = compute_spatial_change_masks(t1, t2)
        # urban_expansion: T1 non-built -> T2 built (1)
        expected_urban = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        urban_mask = masks["urban_expansion"].values if hasattr(masks["urban_expansion"], "values") else masks["urban_expansion"]
        np.testing.assert_array_equal(urban_mask, expected_urban)


class TestIndependentAccuracy:
    def test_spatial_holdout_keeps_blocks_disjoint(self) -> None:
        from geointel.change.accuracy import (
            evaluate_spatial_block_accuracy,
            spatial_block_split,
        )

        x = np.arange(10, dtype=np.float64) * 6000
        y = np.zeros(10, dtype=np.float64)
        train_idx, test_idx, groups = spatial_block_split(
            x, y, block_size_m=5000, test_fraction=0.3
        )
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])

        labels = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        predictions = labels.copy()
        predictions[0] = 2
        metrics = evaluate_spatial_block_accuracy(
            labels, predictions, x, y, block_size_m=5000, test_fraction=0.3
        )
        assert metrics["validation_status"] == "independent reference-label holdout"
        assert metrics["confusion_matrix"]
