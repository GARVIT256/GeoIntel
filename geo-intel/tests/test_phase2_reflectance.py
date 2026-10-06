"""Fixture tests for Earth Search offset handling and balanced month medians."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geointel.data.composite import compute_month_balanced_composite, reflectance_scale_offset


def _item(flag: bool | None, offset: float = -0.1) -> SimpleNamespace:
    asset = SimpleNamespace(extra_fields={"raster:bands": [{"scale": 0.0001, "offset": offset}]})
    return SimpleNamespace(
        id="fixture",
        properties={
            "s2:processing_baseline": "05.00",
            "datetime": "2024-12-09T05:00:00Z",
            **({} if flag is None else {"earthsearch:boa_offset_applied": flag}),
        },
        assets={"B04": asset},
    )


def test_earthsearch_boa_offset_is_applied_exactly_once_for_true_flag() -> None:
    scale, offset = reflectance_scale_offset(_item(True), "B04")
    # Earth Search already shifted DN 1500 -> 500 in the COG.
    assert 500 * scale + offset == pytest.approx(0.05)


def test_unapplied_boa_offset_uses_raster_band_offset_once() -> None:
    scale, offset = reflectance_scale_offset(_item(False), "B04")
    # Unshifted DN 1500 gets scale then -0.1, matching the baked-offset case.
    assert 1500 * scale + offset == pytest.approx(0.05)


def test_unapplied_boa_offset_has_baseline_fallback_when_band_offset_missing() -> None:
    scale, offset = reflectance_scale_offset(_item(False, offset=0.0), "B04")
    assert 1500 * scale + offset == pytest.approx(0.05)


def test_absent_earthsearch_flag_uses_stac_raster_offset_once() -> None:
    # Planetary Computer does not provide the Earth Search-specific property.
    # With no provider flag, use the asset's declared scale/offset as-is.
    scale, offset = reflectance_scale_offset(_item(None, offset=-0.1), "B04")
    assert 1500 * scale + offset == pytest.approx(0.05)


def test_absent_earthsearch_flag_uses_baseline_fallback_when_offset_is_zero() -> None:
    # Sentinel-2 PB4+ uses BOA_ADD_OFFSET=-1000 DN; PC need not provide the
    # Earth Search flag for the processing-baseline fallback to apply.
    scale, offset = reflectance_scale_offset(_item(None, offset=0.0), "B04")
    assert 1500 * scale + offset == pytest.approx(0.05)


def test_historical_pb05_without_flag_does_not_get_post_2022_fallback() -> None:
    item = _item(None, offset=0.0)
    item.properties["datetime"] = "2019-03-16T05:26:49Z"
    scale, offset = reflectance_scale_offset(item, "B04")
    assert 1500 * scale + offset == pytest.approx(0.15)


def test_true_flag_with_zero_raster_offset_does_not_apply_baseline_fallback() -> None:
    # An explicit true flag says the provider already applied the correction.
    scale, offset = reflectance_scale_offset(_item(True, offset=0.0), "B04")
    assert 1500 * scale + offset == pytest.approx(0.15)


def test_month_balanced_composite_equalizes_month_contribution() -> None:
    dates = pd.to_datetime(
        ["2024-11-01"] * 9 + ["2024-12-01", "2025-01-01"]
    )
    values = np.array([0.0] * 9 + [10.0, 20.0], dtype="float32")
    stack = xr.DataArray(
        values.reshape(11, 1, 1, 1),
        dims=["time", "band", "y", "x"],
        coords={"time": dates, "band": ["B04"]},
    )
    result = compute_month_balanced_composite(stack)
    assert float(result.item()) == pytest.approx(10.0)
