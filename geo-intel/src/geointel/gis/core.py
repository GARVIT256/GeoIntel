"""CRS-explicit vector and raster operations for GEO-INTEL."""
from __future__ import annotations

from typing import Any, Iterable

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import box

COMPUTE_CRS = "EPSG:32644"


def assert_crs(data: Any, expected: str | CRS) -> None:
    """Raise when a GeoDataFrame/GeoSeries CRS does not match expected."""
    actual = getattr(data, "crs", None)
    if actual is None:
        raise ValueError("Input geometry has no CRS")
    if CRS.from_user_input(actual) != CRS.from_user_input(expected):
        raise ValueError(f"Expected CRS {expected}, got {actual}")


def reproject_vector(data: gpd.GeoDataFrame, target_crs: str = COMPUTE_CRS) -> gpd.GeoDataFrame:
    """Reproject a vector layer after checking its source CRS."""
    if data.crs is None:
        raise ValueError("Input vector layer has no CRS")
    result = data.to_crs(target_crs)
    assert_crs(result, target_crs)
    return result


def clip_vector(
    data: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Clip features to a boundary after matching both layers' CRS."""
    if data.crs is None or boundary.crs is None:
        raise ValueError("Both vector inputs must have a CRS")
    boundary_aligned = boundary.to_crs(data.crs)
    result = gpd.clip(data, boundary_aligned)
    assert_crs(result, data.crs)
    return result


def geometry_area_m2(data: gpd.GeoDataFrame) -> np.ndarray:
    """Return feature areas in square metres in the project's metric CRS."""
    assert_crs(data, COMPUTE_CRS)
    return data.geometry.area.to_numpy(dtype="float64")


def geometry_length_m(data: gpd.GeoDataFrame) -> np.ndarray:
    """Return feature lengths in metres in the project's metric CRS."""
    assert_crs(data, COMPUTE_CRS)
    return data.geometry.length.to_numpy(dtype="float64")


def buffer_m(data: gpd.GeoDataFrame, distance_m: float) -> gpd.GeoDataFrame:
    """Buffer geometries by a metric distance in EPSG:32644."""
    assert_crs(data, COMPUTE_CRS)
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")
    result = data.copy()
    result.geometry = data.geometry.buffer(distance_m)
    return result


def overlay(
    left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, how: str = "intersection"
) -> gpd.GeoDataFrame:
    """Perform a GeoPandas overlay after aligning both layers to EPSG:32644."""
    left_metric = reproject_vector(left, COMPUTE_CRS)
    right_metric = reproject_vector(right, COMPUTE_CRS)
    result = gpd.overlay(left_metric, right_metric, how=how, keep_geom_type=False)
    assert_crs(result, COMPUTE_CRS)
    return result


def make_fishnet(
    boundary: gpd.GeoDataFrame, cell_size_m: float = 1000.0
) -> gpd.GeoDataFrame:
    """Create clipped square fishnet polygons and stable row/column indices."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    metric = reproject_vector(boundary, COMPUTE_CRS)
    geom = metric.geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds
    features, rows, cols = [], [], []
    for row, y in enumerate(np.arange(miny, maxy, cell_size_m)):
        for col, x in enumerate(np.arange(minx, maxx, cell_size_m)):
            cell = box(x, y, min(x + cell_size_m, maxx), min(y + cell_size_m, maxy))
            part = cell.intersection(geom)
            if not part.is_empty and part.area > 0:
                features.append(part)
                rows.append(row)
                cols.append(col)
    result = gpd.GeoDataFrame({"row_idx": rows, "col_idx": cols}, geometry=features, crs=COMPUTE_CRS)
    assert_crs(result, COMPUTE_CRS)
    return result


def zonal_stats(
    polygons: gpd.GeoDataFrame,
    raster_path: str,
    stats: Iterable[str] = ("count", "mean", "min", "max", "sum"),
) -> gpd.GeoDataFrame:
    """Compute deterministic single-band zonal statistics using rasterio masks."""
    import rasterio
    from rasterio.mask import mask

    if polygons.crs is None:
        raise ValueError("Polygon layer has no CRS")
    requested = set(stats)
    supported = {"count", "mean", "min", "max", "sum"}
    unknown = requested - supported
    if unknown:
        raise ValueError(f"Unsupported zonal statistics: {sorted(unknown)}")
    rows = []
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS")
        aligned = polygons.to_crs(src.crs)
        for geom in aligned.geometry:
            arr, _ = mask(src, [geom], crop=True, filled=False, indexes=1)
            values = arr.compressed().astype("float64")
            values = values[np.isfinite(values)]
            row: dict[str, float | int] = {}
            if "count" in requested:
                row["count"] = int(values.size)
            for key, fn in (("mean", np.mean), ("min", np.min), ("max", np.max), ("sum", np.sum)):
                if key in requested:
                    row[key] = float(fn(values)) if values.size else float("nan")
            rows.append(row)
    result = polygons.copy()
    for key in sorted(requested):
        result[key] = [row[key] for row in rows]
    assert_crs(result, polygons.crs)
    return result


def raster_grid_aggregation(
    raster_path: str,
    grid: gpd.GeoDataFrame,
    value_name: str = "mean",
) -> gpd.GeoDataFrame:
    """Aggregate raster values by fishnet polygon (mean, sum, or valid count)."""
    if value_name not in {"mean", "sum", "count"}:
        raise ValueError("value_name must be mean, sum, or count")
    return zonal_stats(grid, raster_path, stats=(value_name,))


def reproject_clip_raster(
    source_path: str,
    boundary: gpd.GeoDataFrame,
    output_path: str,
    target_crs: str = COMPUTE_CRS,
    resolution_m: float = 10.0,
    categorical: bool = False,
) -> str:
    """Reproject then clip a raster to a vector boundary, preserving nodata."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject

    if boundary.crs is None:
        raise ValueError("Boundary has no CRS")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError("Source raster has no CRS")
        target = CRS.from_user_input(target_crs)
        transform, width, height = calculate_default_transform(
            src.crs, target, src.width, src.height, *src.bounds, resolution=resolution_m
        )
        dst = np.full((src.count, height, width), src.nodata or 0, dtype=src.dtypes[0])
        for index in range(1, src.count + 1):
            reproject(
                rasterio.band(src, index), dst[index - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=target,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
            )
        profile = src.profile.copy()
        profile.update(crs=target, transform=transform, width=width, height=height, compress="deflate")
    boundary_target = boundary.to_crs(target)
    with rasterio.io.MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(dst)
            clipped, clipped_transform = mask(
                tmp, list(boundary_target.geometry), crop=True, filled=True,
                nodata=profile.get("nodata", 0),
            )
            profile.update(width=clipped.shape[2], height=clipped.shape[1], transform=clipped_transform)
    with rasterio.open(output_path, "w", **profile) as out:
        out.write(clipped)
    return output_path
