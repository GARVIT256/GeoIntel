"""Compare Earth Search baseline 00.01 and 05.00 reflectance in one small window.

This script performs raster reads only when explicitly invoked in a hosted
GDAL/Rasterio environment. Its output is diagnostic and is not run as part of
repository preparation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry

from geointel.data.composite import AWS_ASSET_KEYS, reflectance_scale_offset
from geointel.data.stac_fetch import _mgrs_tile


TARGET_CRS = "EPSG:32644"
BANDS = ("B04", "B08", "B11")


def _date(item: Any) -> str:
    return str(item.properties.get("datetime") or item.properties.get("start_datetime"))[:10]


def _baseline(item: Any) -> float:
    return float(item.properties.get("s2:processing_baseline", -1))


def _aoi_geometry(config_path: Path) -> Any:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features = config.get("features", [])
    if features:
        from shapely.ops import unary_union

        return unary_union([shape(feature["geometry"]) for feature in features])
    if config.get("type") == "Feature":
        return shape(config["geometry"])
    return shape(config)


def _load_item_dicts(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Include deduplicated-out originals needed for the baseline comparison."""
    result: dict[str, dict[str, Any]] = {}
    for rows in metadata.get("epochs", {}).values():
        for row in rows:
            result[row["id"]] = row
    dropped_by_epoch = metadata.get("deduplication", {}).get("dropped_items", {})
    for rows in dropped_by_epoch.values():
        for row in rows:
            item = row.get("dropped_item")
            if item:
                result[item["id"]] = item
    return list(result.values())


def _choose_pair(raw_items: list[dict[str, Any]], tile: str | None) -> tuple[Any, Any, str]:
    import pystac

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for raw in raw_items:
        item = pystac.Item.from_dict(raw)
        item_tile = _mgrs_tile(item)
        if tile and item_tile != tile:
            continue
        if _date(item) == "2018-12-01":
            grouped[(str(item_tile), _date(item))].append(item)
    for (item_tile, _), items in sorted(grouped.items()):
        old = next((item for item in items if math.isclose(_baseline(item), 0.01, abs_tol=1e-6)), None)
        new = next((item for item in items if math.isclose(_baseline(item), 5.0, abs_tol=1e-6)), None)
        if old is not None and new is not None:
            return old, new, item_tile
    raise RuntimeError(
        "metadata.json does not contain both processing baselines 00.01 and 05.00 "
        f"for 2018-12-01 on a shared tile{f' {tile}' if tile else ''}. "
        "Ensure the deduplication log preserves the dropped item STAC JSON."
    )


def _asset_key(item: Any, band: str) -> str:
    if band in item.assets:
        return band
    key = AWS_ASSET_KEYS[band]
    if key not in item.assets:
        raise KeyError(f"Item {item.id} lacks {band}/{key} asset")
    return key


def _projected_aoi_window(aoi: Any, item: Any, window_size: int) -> tuple[Window, rasterio.Affine]:
    intersection = aoi.intersection(shape(item.geometry))
    if intersection.is_empty:
        raise ValueError(f"AOI does not intersect requested tile {item.id}")
    project = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True).transform
    projected = transform_geometry(project, intersection)
    center_x, center_y = projected.centroid.x, projected.centroid.y
    resolution = 10.0
    left = math.floor((center_x - window_size * resolution / 2) / resolution) * resolution
    top = math.ceil((center_y + window_size * resolution / 2) / resolution) * resolution
    return Window(0, 0, window_size, window_size), rasterio.Affine(
        resolution, 0, left, 0, -resolution, top
    )


def _read_pipeline_reflectance(
    item: Any,
    band: str,
    window_size: int,
    destination_transform: rasterio.Affine,
) -> tuple[np.ndarray, int, tuple[float, float]]:
    href = item.assets[_asset_key(item, band)].href
    destination = np.full((window_size, window_size), np.nan, dtype="float32")
    left = destination_transform.c
    top = destination_transform.f
    right = left + destination_transform.a * window_size
    bottom = top + destination_transform.e * window_size
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(href) as src:
            projected_bounds = transform_bounds(
                TARGET_CRS, src.crs, left, bottom, right, top, densify_pts=21
            )
            read_window = from_bounds(*projected_bounds, transform=src.transform)
            col0 = max(0, math.floor(read_window.col_off) - 3)
            row0 = max(0, math.floor(read_window.row_off) - 3)
            col1 = min(src.width, math.ceil(read_window.col_off + read_window.width) + 3)
            row1 = min(src.height, math.ceil(read_window.row_off + read_window.height) + 3)
            window = Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))
            raw = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            reproject(
                source=raw,
                destination=destination,
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=destination_transform,
                dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    scale, offset = reflectance_scale_offset(item, band)
    valid_count = int(np.isfinite(destination).sum())
    destination = destination * scale + offset
    return destination, valid_count, (scale, offset)


def run(metadata_path: Path, config_path: Path, tile: str | None, window_size: int) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    item_dicts = _load_item_dicts(metadata)
    old, new, chosen_tile = _choose_pair(item_dicts, tile)
    aoi = _aoi_geometry(config_path)
    _, target_transform = _projected_aoi_window(aoi, old, window_size)
    print(f"Provider: {metadata.get('provider', 'unknown')}")
    print(f"Tile/date: {chosen_tile} / 2018-12-01")
    print(f"Baseline 00.01 item: {old.id}")
    print(f"Baseline 05.00 item: {new.id}")
    print(f"Target: {TARGET_CRS}, {window_size}x{window_size} pixels at 10 m")
    print("Reflectance transform: raw DN * raster scale + effective offset")
    print("Difference sign: baseline 05.00 mean minus baseline 00.01 mean")
    print("Band | PB00.01 mean | PB05.00 mean | difference | valid px each | transforms")
    for band in BANDS:
        old_values, old_count, old_transform = _read_pipeline_reflectance(
            old, band, window_size, target_transform
        )
        new_values, new_count, new_transform = _read_pipeline_reflectance(
            new, band, window_size, target_transform
        )
        jointly_valid = np.isfinite(old_values) & np.isfinite(new_values)
        old_mean = float(np.mean(old_values[jointly_valid])) if jointly_valid.any() else math.nan
        new_mean = float(np.mean(new_values[jointly_valid])) if jointly_valid.any() else math.nan
        print(
            f"{band} | {old_mean:.8f} | {new_mean:.8f} | {new_mean-old_mean:.8f} | "
            f"{int(jointly_valid.sum())} | {old_transform}; {new_transform}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Hosted run metadata.json")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--tile", help="Optional MGRS tile; otherwise first matching tile in metadata")
    parser.add_argument("--window-size", type=int, default=128, help="Target 10 m pixels per side")
    args = parser.parse_args()
    if args.window_size < 16:
        parser.error("--window-size must be at least 16 pixels")
    run(args.metadata, args.config, args.tile, args.window_size)


if __name__ == "__main__":
    main()
