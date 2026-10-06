"""
Phase 2 tests: data acquisition pipeline unit tests.

Tests use synthetic data (in-memory rasters, mock STAC items) so they pass
without internet access. All expected values are computed analytically.

Run with: pytest tests/test_phase2_data.py -v
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_scl_array() -> xr.DataArray:
    """
    5x5 SCL array with known class distribution.

    Layout (class, count):
      4 (vegetation) = 13 pixels → VALID
      8 (cloud)      =  6 pixels → MASKED
      9 (cloud high) =  6 pixels → MASKED

    Time steps: 2
    Shape: (2, 5, 5) = 50 pixels total
    """
    scl_t1 = np.array([
        [4, 4, 8, 9, 4],
        [4, 8, 4, 4, 9],
        [8, 4, 4, 4, 9],
        [4, 4, 8, 4, 4],
        [9, 4, 4, 8, 4],
    ], dtype=np.int16)

    scl_t2 = np.full((5, 5), 4, dtype=np.int16)  # all valid

    # Stack as (time=2, y=5, x=5)
    scl = np.stack([scl_t1, scl_t2], axis=0)

    return xr.DataArray(
        scl,
        dims=["time", "y", "x"],
        coords={
            "time": [0, 1],
            "y": np.arange(5),
            "x": np.arange(5),
        },
        attrs={"crs": "EPSG:32644"},
    )


@pytest.fixture
def spectral_stack(simple_scl_array: xr.DataArray) -> xr.DataArray:
    """
    Synthetic spectral stack (time=2, band=2, y=5, x=5) with KNOWN values.
    Band 0 (B04): all 1000.0 reflectance
    Band 1 (B08): all 3000.0 reflectance
    Expected NDVI at valid pixels = (3000-1000)/(3000+1000) = 0.5
    """
    data = np.ones((2, 2, 5, 5), dtype=np.float32)
    data[:, 0, :, :] = 1000.0   # B04 (Red)
    data[:, 1, :, :] = 3000.0   # B08 (NIR)

    return xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={
            "time": [0, 1],
            "band": ["B04", "B08"],
            "y": np.arange(5),
            "x": np.arange(5),
        },
        attrs={"crs": "EPSG:32644"},
    )


# ---------------------------------------------------------------------------
# cloud_mask.py tests
# ---------------------------------------------------------------------------

class TestBuildValidMask:
    """Tests for cloud_mask.build_valid_mask()"""

    def test_all_vegetation_is_all_valid(self) -> None:
        """SCL=4 everywhere → all pixels valid."""
        from geointel.data.cloud_mask import build_valid_mask

        scl = xr.DataArray(
            np.full((1, 4, 4), 4, dtype=np.int16),
            dims=["time", "y", "x"],
        )
        mask = build_valid_mask(scl, mask_values=[0, 1, 2, 3, 8, 9, 10, 11])
        assert mask.values.all(), "All vegetation pixels should be valid"

    def test_all_cloud_is_all_invalid(self) -> None:
        """SCL=9 (high cloud) everywhere → all pixels invalid."""
        from geointel.data.cloud_mask import build_valid_mask

        scl = xr.DataArray(
            np.full((1, 4, 4), 9, dtype=np.int16),
            dims=["time", "y", "x"],
        )
        mask = build_valid_mask(scl, mask_values=[0, 1, 2, 3, 8, 9, 10, 11])
        assert not mask.values.any(), "All cloud pixels should be invalid"

    def test_nan_scl_is_invalid(self) -> None:
        from geointel.data.cloud_mask import build_valid_mask

        scl = xr.DataArray(
            np.array([[[4.0, np.nan]]], dtype=np.float32),
            dims=["time", "y", "x"],
        )

        np.testing.assert_array_equal(build_valid_mask(scl).values, [[[True, False]]])

    def test_known_mix(self, simple_scl_array: xr.DataArray) -> None:
        """
        In time step 0: 13 valid (class 4) and 12 invalid (class 8 or 9).
        In time step 1: all 25 valid.
        """
        from geointel.data.cloud_mask import build_valid_mask

        mask = build_valid_mask(simple_scl_array, mask_values=[8, 9])
        # t0: count valid (16 pixels with class 4)
        assert int(mask.values[0].sum()) == 16, \
            f"Expected 16 valid in t0, got {int(mask.values[0].sum())}"
        # t1: all 25 valid
        assert int(mask.values[1].sum()) == 25, \
            f"Expected 25 valid in t1, got {int(mask.values[1].sum())}"

    def test_output_is_boolean(self) -> None:
        from geointel.data.cloud_mask import build_valid_mask

        scl = xr.DataArray(np.array([[[4, 8]]], dtype=np.int16), dims=["time", "y", "x"])
        mask = build_valid_mask(scl)
        assert mask.dtype == bool, "Mask must be boolean dtype"

    def test_raises_without_xy_dims(self) -> None:
        """Raise ValueError if 'x' or 'y' dimensions are missing."""
        from geointel.data.cloud_mask import build_valid_mask

        scl_bad = xr.DataArray(np.array([[4, 8]]), dims=["a", "b"])
        with pytest.raises(ValueError, match="must have 'x' and 'y' dimensions"):
            build_valid_mask(scl_bad)


class TestApplyValidMask:
    """Tests for cloud_mask.apply_valid_mask()"""

    def test_masked_pixels_become_nan(self) -> None:
        from geointel.data.cloud_mask import apply_valid_mask

        data = xr.DataArray(
            np.array([[[1.0, 2.0], [3.0, 4.0]]]),
            dims=["time", "y", "x"],
        )
        mask = xr.DataArray(
            np.array([[[True, False], [True, False]]]),
            dims=["time", "y", "x"],
        )
        result = apply_valid_mask(data, mask)
        assert np.isnan(result.values[0, 0, 1]), "Masked pixel must be NaN"
        assert np.isnan(result.values[0, 1, 1]), "Masked pixel must be NaN"
        assert result.values[0, 0, 0] == pytest.approx(1.0)

    def test_output_is_float32(self) -> None:
        from geointel.data.cloud_mask import apply_valid_mask

        data = xr.DataArray(np.ones((1, 2, 2), dtype=np.int16), dims=["time", "y", "x"])
        mask = xr.DataArray(np.ones((1, 2, 2), dtype=bool), dims=["time", "y", "x"])
        result = apply_valid_mask(data, mask)
        assert result.dtype == np.float32, "Output must be float32"


class TestMaskStack:
    def test_missing_spectral_band_reduces_valid_fraction(self) -> None:
        from geointel.data.cloud_mask import mask_stack

        stack = xr.DataArray(
            np.array([
                [[[100.0]], [[4.0]]],
                [[[np.nan]], [[4.0]]],
            ], dtype=np.float32),
            dims=["time", "band", "y", "x"],
            coords={"time": [0, 1], "band": ["B04", "SCL"]},
        )

        masked, valid_fraction = mask_stack(stack)

        assert valid_fraction.values[0, 0] == pytest.approx(0.5)
        assert np.isnan(masked.sel(band="B04").values[1, 0, 0])


class TestValidPixelFraction:
    """Tests for cloud_mask.valid_pixel_fraction()"""

    def test_all_valid_gives_fraction_one(self) -> None:
        from geointel.data.cloud_mask import valid_pixel_fraction

        mask = xr.DataArray(
            np.ones((3, 4, 4), dtype=bool),
            dims=["time", "y", "x"],
        )
        frac = valid_pixel_fraction(mask)
        assert frac.values.max() == pytest.approx(1.0)
        assert frac.values.min() == pytest.approx(1.0)

    def test_half_valid_gives_fraction_half(self) -> None:
        from geointel.data.cloud_mask import valid_pixel_fraction

        # 2 time steps: t0=all True, t1=all False → mean=0.5
        t0 = np.ones((4, 4), dtype=bool)
        t1 = np.zeros((4, 4), dtype=bool)
        mask = xr.DataArray(
            np.stack([t0, t1], axis=0),
            dims=["time", "y", "x"],
        )
        frac = valid_pixel_fraction(mask)
        np.testing.assert_allclose(
            frac.values,
            np.full((4, 4), 0.5, dtype=np.float32),
            rtol=1e-5,
            err_msg="Half-valid mask must give fraction 0.5 everywhere",
        )

    def test_output_shape_drops_time_dim(self) -> None:
        from geointel.data.cloud_mask import valid_pixel_fraction

        mask = xr.DataArray(np.ones((5, 6, 7), dtype=bool), dims=["time", "y", "x"])
        frac = valid_pixel_fraction(mask)
        assert frac.shape == (6, 7), f"Expected (6,7), got {frac.shape}"


class TestCheckCoverage:
    """Tests for cloud_mask.check_coverage()"""

    def test_passes_when_above_threshold(self) -> None:
        from geointel.data.cloud_mask import check_coverage

        frac = xr.DataArray(np.full((4, 4), 0.85, dtype=np.float32), dims=["y", "x"])
        assert check_coverage(frac, min_frac=0.70) is True

    def test_fails_when_below_threshold(self) -> None:
        from geointel.data.cloud_mask import check_coverage

        frac = xr.DataArray(np.full((4, 4), 0.50, dtype=np.float32), dims=["y", "x"])
        assert check_coverage(frac, min_frac=0.70) is False


# ---------------------------------------------------------------------------
# median composite test
# ---------------------------------------------------------------------------

class TestMedianComposite:
    """Tests for composite.compute_median_composite()"""

    def test_median_of_known_values(self) -> None:
        """
        Stack with known values: median must equal the middle value.
        t0=100, t1=200, t2=300 → median=200 for all pixels.
        """
        from geointel.data.composite import compute_median_composite

        data = np.stack([
            np.full((1, 3, 3), 100.0, dtype=np.float32),
            np.full((1, 3, 3), 200.0, dtype=np.float32),
            np.full((1, 3, 3), 300.0, dtype=np.float32),
        ], axis=0)   # (3, 1, 3, 3)

        stack = xr.DataArray(
            data,
            dims=["time", "band", "y", "x"],
            coords={"band": ["B04"]},
            attrs={"crs": "EPSG:32644"},
        )

        comp = compute_median_composite(stack)
        np.testing.assert_allclose(
            comp.values,
            np.full((1, 3, 3), 200.0, dtype=np.float32),
            rtol=1e-5,
            err_msg="Median of [100,200,300] must be 200",
        )

    def test_median_skips_nan(self) -> None:
        """
        Pixels with NaN at some time steps: median computed over valid values only.
        t0=NaN, t1=500, t2=700 → median of [500, 700] = 600.
        """
        from geointel.data.composite import compute_median_composite

        data = np.array([
            [[[np.nan]]],
            [[[500.0]]],
            [[[700.0]]],
        ], dtype=np.float32)   # (3, 1, 1, 1)

        stack = xr.DataArray(data, dims=["time", "band", "y", "x"],
                             coords={"band": ["B04"]})
        comp = compute_median_composite(stack)
        np.testing.assert_allclose(
            comp.values[0, 0, 0],
            600.0,
            rtol=1e-4,
            err_msg="Median of [nan, 500, 700] must be 600",
        )

    def test_all_nan_pixel_stays_nan(self) -> None:
        """Pixel that is always masked must remain NaN in composite."""
        from geointel.data.composite import compute_median_composite

        data = np.full((3, 1, 2, 2), np.nan, dtype=np.float32)
        data[:, :, 0, 0] = 500.0   # only (0,0) pixel has valid data

        stack = xr.DataArray(data, dims=["time", "band", "y", "x"],
                             coords={"band": ["B04"]})
        comp = compute_median_composite(stack)
        assert np.isnan(comp.values[0, 1, 1]), "All-NaN pixel must remain NaN"
        assert comp.values[0, 0, 0] == pytest.approx(500.0)


class TestL2AReflectanceNormalization:
    def test_pre_and_post_2022_values_are_harmonized(self) -> None:
        from geointel.data.composite import normalize_l2a_reflectance

        stack = xr.DataArray(
            np.array([
                [[[5000.0]], [[4.0]]],
                [[[5000.0]], [[4.0]]],
                [[[6000.0]], [[4.0]]],
                [[[6000.0]], [[4.0]]],
            ], dtype=np.float32),
            dims=["time", "band", "y", "x"],
            coords={"time": [0, 1, 2, 3], "band": ["B04", "SCL"]},
        )
        items = [
            SimpleNamespace(id="pre-baseline-4", properties={
                "s2:processing_baseline": "02.12", "datetime": "2019-03-16T05:26:49Z"
            }),
            SimpleNamespace(id="reprocessed-historical", properties={
                "s2:processing_baseline": "05.00", "datetime": "2019-03-16T05:26:49Z"
            }),
            SimpleNamespace(id="post-2022-offset", properties={
                "s2:processing_baseline": "05.11", "datetime": "2024-11-04T05:29:19Z"
            }),
            SimpleNamespace(id="aws-offset-already-applied", properties={
                "s2:processing_baseline": "05.11",
                "datetime": "2024-11-04T05:29:19Z",
                "earthsearch:boa_offset_applied": True,
            }),
        ]

        normalized = normalize_l2a_reflectance(stack, items)

        np.testing.assert_allclose(
            normalized.sel(band="B04").values[:, 0, 0], [0.5, 0.5, 0.5, 0.6]
        )
        np.testing.assert_array_equal(
            normalized.sel(band="SCL").values[:, 0, 0], [4.0, 4.0, 4.0, 4.0]
        )

    def test_asset_scale_and_offset_take_precedence(self) -> None:
        from geointel.data.composite import normalize_l2a_reflectance

        stack = xr.DataArray(
            np.array([[[[1920.0]]], [[[1920.0]]]], dtype=np.float32),
            dims=["time", "band", "y", "x"],
            coords={"time": [0, 1], "band": ["B04"]},
        )
        item_properties = [
            {"s2:processing_baseline": "05.00", "datetime": "2019-03-31T05:40:34Z"},
            {"s2:processing_baseline": "05.11", "datetime": "2025-03-31T05:40:44Z"},
        ]
        items = [
            SimpleNamespace(
                id=f"scene-{index}",
                properties=properties,
                assets={
                    "B04": SimpleNamespace(
                        extra_fields={
                            "raster:bands": [{"scale": 0.0001, "offset": -0.1}]
                        }
                    )
                },
            )
            for index, properties in enumerate(item_properties)
        ]

        normalized = normalize_l2a_reflectance(stack, items)

        np.testing.assert_allclose(
            normalized.sel(band="B04").values[:, 0, 0], [0.092, 0.092]
        )


# ---------------------------------------------------------------------------
# config loader test
# ---------------------------------------------------------------------------

class TestConfigLoader:
    """Basic config loader smoke tests."""

    def test_load_config_returns_dict(self, tmp_path: Path) -> None:
        from geointel.utils.config import load_config

        # Create a minimal config in a temp dir
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "config.yaml"
        cfg_file.write_text(
            "aoi:\n  bbox: [77.90, 30.20, 78.20, 30.50]\n"
            "epochs:\n  t1:\n    label: test\n    start: '2018-11-01'\n    end: '2019-03-31'\n"
            "  t2:\n    label: test2\n    start: '2024-11-01'\n    end: '2025-03-31'\n"
            "paths:\n  data_cache: data/cache\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert isinstance(cfg, dict)
        assert "aoi" in cfg
        assert "_config_hash" in cfg

    def test_config_hash_is_deterministic(self, tmp_path: Path) -> None:
        from geointel.utils.config import load_config

        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "config.yaml"
        cfg_file.write_text("aoi:\n  bbox: [1,2,3,4]\npaths:\n  data_cache: data/cache\n",
                            encoding="utf-8")

        h1 = load_config(cfg_file)["_config_hash"]
        h2 = load_config(cfg_file)["_config_hash"]
        assert h1 == h2, "Config hash must be deterministic for the same file"


class TestSceneDeduplication:
    def test_prefers_highest_baseline_for_same_acquisition(self) -> None:
        from geointel.data.stac_fetch import deduplicate_scene_items

        shared = {
            "datetime": "2019-03-16T05:26:49Z",
            "platform": "sentinel-2b",
            "s2:mgrs_tile": "44RKU",
        }
        items = [
            SimpleNamespace(id="PB02_product", properties={
                **shared, "s2:processing_baseline": "02.12"
            }),
            SimpleNamespace(id="PB05_product", properties={
                **shared, "s2:processing_baseline": "05.00"
            }),
            SimpleNamespace(id="different_tile", properties={
                **{**shared, "s2:mgrs_tile": "43RGP"}, "s2:processing_baseline": "02.12"
            }),
        ]

        result = deduplicate_scene_items(items)

        assert [item.id for item in result] == ["different_tile", "PB05_product"]

    def test_provider_scene_caches_are_isolated(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        import pystac

        from geointel.data.stac_fetch import cache_scenes, load_cached_items

        item = pystac.Item(
            id="S2B_44RKU_20181106",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2018, 11, 6, tzinfo=timezone.utc),
            properties={"platform": "sentinel-2b", "s2:processing_baseline": "02.12"},
        )
        cfg = {"paths": {"data_cache": tmp_path}}

        cache_scenes([item], cfg, "t1", provider="aws")

        assert load_cached_items(cfg, "t1", provider="pc") is None
        assert len(load_cached_items(cfg, "t1", provider="aws") or []) == 1

    def test_earth_search_tile_id_deduplicates_with_pc_properties(self) -> None:
        from geointel.data.stac_fetch import deduplicate_scene_items

        shared = {
            "datetime": "2019-03-16T05:26:49Z",
            "platform": "sentinel-2b",
        }
        items = [
            SimpleNamespace(id="pc-item", properties={
                **shared, "s2:mgrs_tile": "44RKU", "s2:processing_baseline": "02.12"
            }),
            SimpleNamespace(id="S2B_44RKU_20190316_0_L2A", properties={
                **shared, "s2:processing_baseline": "05.00"
            }),
        ]

        result = deduplicate_scene_items(items)

        assert [item.id for item in result] == ["S2B_44RKU_20190316_0_L2A"]

    def test_deduplicates_calendar_date_across_times_and_platforms_with_audit_log(self) -> None:
        from geointel.data.stac_fetch import deduplicate_scene_items

        items = [
            SimpleNamespace(id="S2C_44RKU_20181201", properties={
                "datetime": "2018-12-01T05:20:00Z", "platform": "sentinel-2c",
                "s2:mgrs_tile": "44RKU", "s2:processing_baseline": "05.00",
            }),
            SimpleNamespace(id="S2A_44RKU_20181201", properties={
                "datetime": "2018-12-01T05:21:00Z", "platform": "sentinel-2a",
                "s2:mgrs_tile": "44RKU", "s2:processing_baseline": "05.00",
            }),
            SimpleNamespace(id="old_product", properties={
                "datetime": "2018-12-01T05:22:00Z", "platform": "sentinel-2b",
                "s2:mgrs_tile": "44RKU", "s2:processing_baseline": "00.01",
            }),
        ]
        dropped: list[dict[str, object]] = []

        result = deduplicate_scene_items(items, dropped_items=dropped)

        assert [item.id for item in result] == ["S2A_44RKU_20181201"]
        assert {row["dropped_scene_id"] for row in dropped} == {"S2C_44RKU_20181201", "old_product"}


# ---------------------------------------------------------------------------
# DEM slope/aspect test
# ---------------------------------------------------------------------------

class TestSlopeAspect:
    """Tests for dem._derive_slope_aspect()"""

    def test_flat_terrain_gives_zero_slope(self) -> None:
        from geointel.data.dem import _derive_slope_aspect

        elevation = np.full((10, 10), 500.0, dtype=np.float32)
        slope, aspect = _derive_slope_aspect(elevation, resolution_m=30.0)
        np.testing.assert_allclose(
            slope[1:-1, 1:-1],   # interior pixels (edges may differ)
            0.0,
            atol=1e-4,
            err_msg="Flat terrain must give slope ≈ 0 degrees",
        )

    def test_slope_is_non_negative(self) -> None:
        from geointel.data.dem import _derive_slope_aspect

        np.random.seed(42)
        elevation = np.random.uniform(200, 800, (20, 20)).astype(np.float32)
        slope, aspect = _derive_slope_aspect(elevation, resolution_m=30.0)
        assert (slope >= 0).all(), "Slope must always be >= 0 degrees"

    def test_aspect_range_is_0_to_360(self) -> None:
        from geointel.data.dem import _derive_slope_aspect

        np.random.seed(42)
        elevation = np.random.uniform(200, 800, (20, 20)).astype(np.float32)
        slope, aspect = _derive_slope_aspect(elevation, resolution_m=30.0)
        assert (aspect >= 0).all() and (aspect < 360).all(), \
            f"Aspect must be in [0, 360); got min={aspect.min():.2f}, max={aspect.max():.2f}"
