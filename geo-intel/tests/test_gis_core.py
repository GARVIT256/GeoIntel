"""Known-answer synthetic tests for GIS vector and raster utilities."""
from __future__ import annotations

import importlib

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from geointel.gis.core import (
    COMPUTE_CRS,
    assert_crs,
    buffer_m,
    clip_vector,
    geometry_area_m2,
    geometry_length_m,
    make_fishnet,
    overlay,
    raster_grid_aggregation,
    reproject_clip_raster,
    reproject_vector,
    zonal_stats,
)


def test_assert_crs_accepts_expected_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=COMPUTE_CRS)
    assert_crs(frame, COMPUTE_CRS)


def test_assert_crs_rejects_mismatch() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="Expected CRS"):
        assert_crs(frame, COMPUTE_CRS)


def test_assert_crs_rejects_missing_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    with pytest.raises(ValueError, match="no CRS"):
        assert_crs(frame, COMPUTE_CRS)


def test_reproject_vector_changes_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(78, 30)], crs="EPSG:4326")
    projected = reproject_vector(frame)
    assert str(projected.crs).upper().endswith("32644")
    assert projected.geometry.iloc[0].x > 0


def test_clip_vector_returns_only_intersection() -> None:
    data = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[box(0, 0, 2, 2), box(5, 5, 6, 6)], crs=COMPUTE_CRS)
    boundary = gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs=COMPUTE_CRS)
    result = clip_vector(data, boundary)
    assert len(result) == 1
    assert result.geometry.iloc[0].area == pytest.approx(1)


def test_clip_vector_reprojects_boundary_to_input_crs() -> None:
    boundary = gpd.GeoDataFrame(
        geometry=[box(77.999, 29.999, 78.001, 30.001)], crs="EPSG:4326"
    )
    boundary_metric = boundary.to_crs(COMPUTE_CRS)
    data = gpd.GeoDataFrame(
        geometry=[boundary_metric.geometry.iloc[0].buffer(10)], crs=COMPUTE_CRS
    )
    result = clip_vector(data, boundary)
    assert result.crs == data.crs
    assert len(result) == 1
    assert result.geometry.iloc[0].area == pytest.approx(boundary_metric.geometry.iloc[0].area)


def test_area_known_square_metres() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 20)], crs=COMPUTE_CRS)
    assert geometry_area_m2(frame).tolist() == [200]


def test_area_requires_metric_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError):
        geometry_area_m2(frame)


def test_length_known_metres() -> None:
    frame = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (3, 4)])], crs=COMPUTE_CRS)
    assert geometry_length_m(frame).tolist() == [5]


def test_buffer_expands_bounds_by_distance() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=COMPUTE_CRS)
    result = buffer_m(frame, 2)
    assert result.total_bounds.tolist() == pytest.approx([-2, -2, 12, 12])


def test_buffer_rejects_negative_distance() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=COMPUTE_CRS)
    with pytest.raises(ValueError, match="non-negative"):
        buffer_m(frame, -1)


def test_overlay_intersection_area() -> None:
    left = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 2, 2)], crs=COMPUTE_CRS)
    right = gpd.GeoDataFrame({"b": [2]}, geometry=[box(1, 1, 3, 3)], crs=COMPUTE_CRS)
    result = overlay(left, right)
    assert len(result) == 1
    assert result.geometry.iloc[0].area == pytest.approx(1)


def test_overlay_reprojects_inputs() -> None:
    left = gpd.GeoDataFrame(geometry=[box(77.99, 29.99, 78.01, 30.01)], crs="EPSG:4326")
    right = reproject_vector(left)
    result = overlay(left, right)
    assert_crs(result, COMPUTE_CRS)
    assert len(result) == 1


def test_fishnet_one_square_has_expected_area() -> None:
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=COMPUTE_CRS)
    grid = make_fishnet(boundary, cell_size_m=100)
    assert len(grid) == 1
    assert geometry_area_m2(grid).tolist() == [10000]


def test_fishnet_four_cells_for_two_by_two_extent() -> None:
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 200, 200)], crs=COMPUTE_CRS)
    grid = make_fishnet(boundary, cell_size_m=100)
    assert len(grid) == 4
    assert set(grid["row_idx"]) == {0, 1}
    assert set(grid["col_idx"]) == {0, 1}


def test_fishnet_clips_edge_cells_to_irregular_boundary() -> None:
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 150, 150)], crs=COMPUTE_CRS)
    grid = make_fishnet(boundary, cell_size_m=100)
    assert len(grid) == 4
    assert sum(geometry_area_m2(grid)) == pytest.approx(22500)


def test_fishnet_rejects_nonpositive_size() -> None:
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=COMPUTE_CRS)
    with pytest.raises(ValueError, match="positive"):
        make_fishnet(boundary, 0)


def _rasterio_or_skip():
    try:
        return importlib.import_module("rasterio")
    except Exception as exc:  # DLL/Application Control errors also skip on Windows.
        pytest.skip(f"rasterio unavailable in this runtime: {exc}")


@pytest.mark.raster
def test_zonal_stats_known_values(tmp_path) -> None:
    rio = _rasterio_or_skip()
    from rasterio.transform import from_origin

    path = tmp_path / "values.tif"
    with rio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="int16", crs=COMPUTE_CRS, transform=from_origin(0, 20, 10, 10), nodata=-1) as dst:
        dst.write(np.array([[1, 2], [3, 4]], dtype=np.int16), 1)
    polygons = gpd.GeoDataFrame(geometry=[box(0, 0, 20, 20)], crs=COMPUTE_CRS)
    result = zonal_stats(polygons, str(path))
    assert result.loc[0, "count"] == 4
    assert result.loc[0, "mean"] == pytest.approx(2.5)


@pytest.mark.raster
def test_raster_grid_aggregation_known_sum(tmp_path) -> None:
    rio = _rasterio_or_skip()
    from rasterio.transform import from_origin

    path = tmp_path / "values.tif"
    with rio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8", crs=COMPUTE_CRS, transform=from_origin(0, 20, 10, 10)) as dst:
        dst.write(np.array([[1, 2], [3, 4]], dtype=np.uint8), 1)
    grid = gpd.GeoDataFrame(geometry=[box(0, 0, 20, 20)], crs=COMPUTE_CRS)
    assert raster_grid_aggregation(str(path), grid, "sum").loc[0, "sum"] == 10


@pytest.mark.raster
def test_reproject_clip_raster_uses_expected_crs_and_shape(tmp_path) -> None:
    rio = _rasterio_or_skip()
    from rasterio.transform import from_origin

    src_path, dst_path = tmp_path / "src.tif", tmp_path / "dst.tif"
    with rio.open(src_path, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8", crs=COMPUTE_CRS, transform=from_origin(0, 40, 10, 10)) as dst:
        dst.write(np.ones((4, 4), dtype=np.uint8), 1)
    boundary = gpd.GeoDataFrame(geometry=[box(10, 10, 30, 30)], crs=COMPUTE_CRS)
    reproject_clip_raster(str(src_path), boundary, str(dst_path), resolution_m=10)
    with rio.open(dst_path) as result:
        assert result.crs.to_string() == COMPUTE_CRS
        assert (result.width, result.height) == (2, 2)
