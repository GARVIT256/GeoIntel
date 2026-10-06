"""Hosted diagnostic of reflectance transforms across retained T1/T2 scenes.

Requires the same two fixed 1 km target windows for every scene. Supply bounds
in EPSG:32644 after visually confirming one is dark/dense and one bright. This
script performs remote raster reads; it is intended for Colab/Kaggle only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from geointel.data.composite import AWS_ASSET_KEYS, reflectance_scale_offset

BANDS = ("B04", "B08", "B11")
WINDOW_METRES = 1000.0
PIXEL_METRES = 10.0


def _center(text: str) -> tuple[float, float]:
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("center must be longitude,latitude in WGS84") from exc
    if len(values) != 2:
        raise argparse.ArgumentTypeError("center must be longitude,latitude in WGS84")
    lon, lat = values
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise argparse.ArgumentTypeError("center must be valid WGS84 longitude,latitude")
    return values  # type: ignore[return-value]


def _date(item: Any) -> str:
    return str(item.properties.get("datetime") or item.properties.get("start_datetime") or "")[:10]


def _flag(item: Any) -> str:
    value = item.properties.get("earthsearch:boa_offset_applied")
    return "true" if value is True else "false" if value is False else "absent"


def _offset_maps(item: Any) -> tuple[dict[str, float], dict[str, float]]:
    raster_offsets: dict[str, float] = {}
    effective_offsets: dict[str, float] = {}
    for band in BANDS:
        key = band if band in item.assets else AWS_ASSET_KEYS[band]
        asset = item.assets[key]
        bands = asset.extra_fields.get("raster:bands", [])
        meta = bands[0] if bands else {}
        raster_offsets[band] = float(meta.get("offset", 0.0))
        _, effective_offsets[band] = reflectance_scale_offset(item, band)
    return raster_offsets, effective_offsets


def _read_reflectance(item: Any, band: str, transform: Affine) -> np.ndarray:
    """Read one 1 km target and apply exactly the pipeline's item transform."""
    key = band if band in item.assets else AWS_ASSET_KEYS[band]
    if key not in item.assets:
        raise KeyError(f"Item {item.id} lacks {band}/{AWS_ASSET_KEYS[band]} asset")
    href = item.assets[key].href
    size = round(WINDOW_METRES / PIXEL_METRES)
    destination = np.full((size, size), np.nan, dtype="float32")
    left, top = transform.c, transform.f
    right = left + transform.a * size
    bottom = top + transform.e * size
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(href) as src:
            bounds = transform_bounds("EPSG:32644", src.crs, left, bottom, right, top, densify_pts=21)
            fractional = from_bounds(*bounds, transform=src.transform)
            col0 = max(0, math.floor(fractional.col_off) - 3)
            row0 = max(0, math.floor(fractional.row_off) - 3)
            col1 = min(src.width, math.ceil(fractional.col_off + fractional.width) + 3)
            row1 = min(src.height, math.ceil(fractional.row_off + fractional.height) + 3)
            window = Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))
            raw = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            reproject(source=raw, destination=destination,
                      src_transform=src.window_transform(window), src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=transform, dst_crs="EPSG:32644",
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
    scale, offset = reflectance_scale_offset(item, band)
    return destination * scale + offset


def _item_records(metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []
    for epoch in ("t1", "t2"):
        rows = metadata.get("epochs", {}).get(epoch, [])
        if not rows:
            raise ValueError(f"metadata.json has no deduplicated items for {epoch}")
        records.extend((epoch, pystac.Item.from_dict(raw)) for raw in rows)
    return records


def _tile(item: Any) -> str:
    props = item.properties
    return str(props.get("s2:mgrs_tile") or props.get("grid:code") or item.id)


def _target_grid(center_lonlat: tuple[float, float]) -> tuple[int, Affine, tuple[float, float, float, float]]:
    lon, lat = center_lonlat
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True).transform(lon, lat)
    xmin, ymin = x - WINDOW_METRES / 2, y - WINDOW_METRES / 2
    xmax, ymax = x + WINDOW_METRES / 2, y + WINDOW_METRES / 2
    size = round(WINDOW_METRES / PIXEL_METRES)
    return size, Affine(PIXEL_METRES, 0, xmin, 0, -PIXEL_METRES, ymax), (xmin, ymin, xmax, ymax)


def run(metadata_path: Path, dark_center: tuple[float, float], bright_center: tuple[float, float]) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = _item_records(metadata)
    print(f"Provider: {metadata.get('provider', 'unknown')}")
    print(f"Python: {__import__('sys').version.split()[0]} | GDAL: {rasterio.__gdal_version__} | rasterio: {rasterio.__version__}")
    grids = {"dark_dense": _target_grid(dark_center), "bright": _target_grid(bright_center)}
    for target, center in (("dark_dense", dark_center), ("bright", bright_center)):
        bounds = grids[target][2]
        print(f"{target}_center_WGS84_lonlat: {center[0]},{center[1]}")
        print(f"{target}_bounds_EPSG32644: {','.join(f'{v:.3f}' for v in bounds)}")
    print("Windows: fixed 1000 m x 1000 m bounds; 10 m output grid")
    print("Review the chosen windows visually: dark/dense and bright targets are user-selected, not classified here.")
    print("Reflectance: raw DN * scale + effective offset; medians use finite reprojected pixels.")
    headers = ("epoch", "date", "baseline", "boa_offset_applied", "scene_id", "target",
               "raster_offset", "effective_offset", "median_B04", "median_B08", "median_B11",
               "epoch_delta_over_0.03")
    print(",".join(headers))

    target_grids = {key: (value[0], value[1]) for key, value in grids.items()}
    target_bounds = tuple(target_grids)

    observations: list[dict[str, Any]] = []
    for epoch, item in records:
        try:
            baseline = str(item.properties.get("s2:processing_baseline", "unknown"))
        except AttributeError:
            baseline = "unknown"
        raster_offsets, effective_offsets = _offset_maps(item)
        for target, (_, transform) in target_grids.items():
            medians: dict[str, float] = {}
            for band in BANDS:
                values = _read_reflectance(item, band, transform)
                finite = values[np.isfinite(values)]
                medians[band] = float(np.median(finite)) if finite.size else math.nan
            observations.append({"epoch": epoch, "date": _date(item), "baseline": baseline,
                                 "flag": _flag(item), "scene": item.id, "target": target,
                                 "raster_offsets": raster_offsets,
                                 "effective_offsets": effective_offsets, "medians": medians})

    epoch_medians: dict[tuple[str, str, str], float] = {}
    for epoch in ("t1", "t2"):
        for target in target_bounds:
            for band in BANDS:
                vals = [r["medians"][band] for r in observations
                        if r["epoch"] == epoch and r["target"] == target and math.isfinite(r["medians"][band])]
                epoch_medians[(epoch, target, band)] = float(np.median(vals)) if vals else math.nan

    def offset_json(row: dict[str, float]) -> str:
        return json.dumps({band: round(value, 6) for band, value in row.items()}, separators=(",", ":"))

    for row in observations:
        deltas = {band: abs(row["medians"][band] - epoch_medians[(row["epoch"], row["target"], band)])
                  for band in BANDS if math.isfinite(row["medians"][band]) and
                  math.isfinite(epoch_medians[(row["epoch"], row["target"], band)])}
        flagged = ";".join(f"{b}:{d:.4f}" for b, d in deltas.items() if d > 0.03) or ""
        fields = [row["epoch"], row["date"], row["baseline"], row["flag"], row["scene"], row["target"],
                  offset_json(row["raster_offsets"]), offset_json(row["effective_offsets"]),
                  *(f"{row['medians'][b]:.6f}" if math.isfinite(row["medians"][b]) else "NA" for b in BANDS),
                  flagged]
        print(",".join(fields))

    print("\nFlag-group step summary (group median; false minus true; about-0.1 if absolute step is 0.07–0.13):")
    for epoch in ("t1", "t2"):
        for target in target_bounds:
            for band in BANDS:
                groups = {flag: [r["medians"][band] for r in observations
                                 if r["epoch"] == epoch and r["target"] == target and r["flag"] == flag
                                 and math.isfinite(r["medians"][band])]
                          for flag in ("true", "false", "absent")}
                summaries = {flag: float(np.median(vals)) if vals else math.nan for flag, vals in groups.items()}
                step = summaries["false"] - summaries["true"] if groups["false"] and groups["true"] else math.nan
                about = "YES" if math.isfinite(step) and 0.07 <= abs(step) <= 0.13 else "NO/INSUFFICIENT GROUPS"
                print(f"{epoch},{target},{band},true_n={len(groups['true'])},false_n={len(groups['false'])},"
                      f"absent_n={len(groups['absent'])},true_median={summaries['true']:.6f},"
                      f"false_median={summaries['false']:.6f},false_minus_true={step:.6f},about_0.1={about}")

    print("\nSame-day item pairs grouped by tile + calendar date (each comparison uses the same target window):")
    groups: dict[tuple[str, str], list[tuple[str, Any]]] = {}
    for epoch, item in records:
        groups.setdefault((_tile(item), _date(item)), []).append((epoch, item))
    print("epoch,tile,date,item_a,item_b,flags_differ,target,band,flag_a,flag_b,raster_offset_a,raster_offset_b,median_a,median_b,difference_b_minus_a")
    for (tile, date), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda pair: pair[1].id)
        # Emit all pair combinations; metadata epoch membership is retained in the epoch field.
        for i, (epoch_a, item_a) in enumerate(members):
            for epoch_b, item_b in members[i + 1:]:
                flags_differ = _flag(item_a) != _flag(item_b)
                ra, _ = _offset_maps(item_a)
                rb, _ = _offset_maps(item_b)
                for target, (_, transform) in target_grids.items():
                    for band in BANDS:
                        va = _read_reflectance(item_a, band, transform)
                        vb = _read_reflectance(item_b, band, transform)
                        fa, fb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                        ma = float(np.median(fa)) if fa.size else math.nan
                        mb = float(np.median(fb)) if fb.size else math.nan
                        difference = mb - ma if math.isfinite(ma) and math.isfinite(mb) else math.nan
                        marker = " *** FLAGS DIFFER ***" if flags_differ else ""
                        print(f"{epoch_a}/{epoch_b},{tile},{date},{item_a.id},{item_b.id},{flags_differ},"
                              f"{target},{band},{_flag(item_a)},{_flag(item_b)},{ra[band]:.6f},{rb[band]:.6f},"
                              f"{ma:.6f},{mb:.6f},{difference:.6f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="metadata.json containing retained T1/T2 items")
    parser.add_argument("--dark-center", type=_center, required=True,
                        help="visually confirmed dark/dense target center lon,lat in WGS84")
    parser.add_argument("--bright-center", type=_center, required=True,
                        help="visually confirmed bright target center lon,lat in WGS84")
    args = parser.parse_args()
    run(args.metadata, args.dark_center, args.bright_center)


if __name__ == "__main__":
    main()
