"""Hosted Python 3.11 execution and output-derived report for GEO-INTEL."""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from osgeo import gdal

from geointel.data.composite import build_composite_for_epoch, run_smoke_test
from geointel.data.stac_fetch import fetch_or_load
from geointel.utils.config import load_config
from geointel.utils.manifest import RunManifest


def _write_ndvi_difference(t1_path: Path, t2_path: Path, output_path: Path) -> Path:
    """Stream T2-minus-T1 NDVI from the compact B04/B08 composite COGs."""
    from rasterio.shutil import copy as raster_copy

    with rasterio.open(t1_path) as t1, rasterio.open(t2_path) as t2:
        if (t1.width, t1.height, t1.transform, t1.crs) != (
            t2.width, t2.height, t2.transform, t2.crs
        ):
            raise ValueError("T1 and T2 compact composites must use the same grid")

        def band_index(source: rasterio.DatasetReader, band: str) -> int:
            descriptions = list(source.descriptions)
            if descriptions and all(descriptions) and band in descriptions:
                return descriptions.index(band) + 1
            if source.count == 2:
                return {"B04": 1, "B08": 2}[band]
            raise ValueError(f"Could not identify {band} in {source.name}")

        red1, nir1 = band_index(t1, "B04"), band_index(t1, "B08")
        red2, nir2 = band_index(t2, "B04"), band_index(t2, "B08")
        profile = t1.profile.copy()
        profile.update(
            driver="GTiff", count=1, dtype="float32", nodata=float("nan"),
            compress="deflate", predictor=3, tiled=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path = output_path.with_name(f"{output_path.stem}.staged.tif")

        def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
            denominator = nir + red
            valid = np.isfinite(red) & np.isfinite(nir) & (np.abs(denominator) > 1e-8)
            result = np.full(red.shape, np.nan, dtype="float32")
            np.divide(nir - red, denominator, out=result, where=valid)
            return result

        with rasterio.open(staged_path, "w", **profile) as destination:
            for _, window in t1.block_windows(1):
                red_t1 = t1.read(red1, window=window, masked=True).astype("float32").filled(np.nan)
                nir_t1 = t1.read(nir1, window=window, masked=True).astype("float32").filled(np.nan)
                red_t2 = t2.read(red2, window=window, masked=True).astype("float32").filled(np.nan)
                nir_t2 = t2.read(nir2, window=window, masked=True).astype("float32").filled(np.nan)
                difference = ndvi(red_t2, nir_t2) - ndvi(red_t1, nir_t1)
                destination.write(difference.astype("float32"), 1, window=window)
            destination.set_band_description(1, "NDVI T2 minus T1")

    with rasterio.open(staged_path) as source:
        raster_copy(
            source, output_path, driver="COG", compress="DEFLATE", blocksize=256,
            overview_resampling="AVERAGE",
        )
    staged_path.unlink(missing_ok=True)
    return output_path


def run(
    repo: Path,
    output_dir: Path,
    provider: str = "aws",
    fast_ndvi: bool = False,
    chunk_size: int | None = None,
    workers: int | None = None,
) -> None:
    if sys.version_info[:2] not in {(3, 11), (3, 13)}:
        raise RuntimeError(f"Python 3.11 or 3.13 is supported; found {sys.version}")
    repo, output_dir = repo.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)

    print(f"Python: {sys.version}")
    print(f"GDAL: {gdal.VersionInfo('--version')}")
    print(f"rasterio: {rasterio.__version__}")
    print(f"Provider: {provider}")

    cfg = load_config("config/config.yaml")
    output_bands = ["B04", "B08"] if fast_ndvi else (
        list(cfg["sentinel2"]["bands_10m"])
        + list(cfg["sentinel2"]["bands_20m"])
    )
    chunk_size = chunk_size or (512 if fast_ndvi else 256)
    workers = workers or (4 if fast_ndvi else 2)
    if chunk_size < 128 or workers < 1:
        raise ValueError("chunk_size must be >=128 and workers must be >=1")
    print(f"Composite bands: {', '.join(output_bands)} plus SCL for masking")
    print(f"Spatial chunk size: {chunk_size}px | Dask workers: {workers}")
    for epoch in ("t1", "t2"):
        cached = cfg["paths"]["data_cache"] / "composites" / epoch
        if cached.exists():
            shutil.rmtree(cached)

    discovered_scenes = {}
    dedup_drops: dict[str, list[dict[str, object]]] = {"t1": [], "t2": []}
    for epoch in ("t1", "t2"):
        discovered_scenes[epoch] = fetch_or_load(
            cfg, epoch, provider=provider, force_refresh=True,
            dropped_items=dedup_drops[epoch],
        )
        if not discovered_scenes[epoch]:
            raise RuntimeError(f"No scenes returned by {provider} for {epoch}")

    expected_counts = {"t1": 36, "t2": 46}
    scene_counts = {
        epoch: {
            "deduplicated_count": len(discovered_scenes[epoch]),
            "expected_count": expected_counts[epoch],
            "matches_expected": len(discovered_scenes[epoch]) == expected_counts[epoch],
        }
        for epoch in ("t1", "t2")
    }
    for epoch in ("t1", "t2"):
        print(
            f"{epoch.upper()} deduplicated scene count: {scene_counts[epoch]['deduplicated_count']} "
            f"(expected {expected_counts[epoch]}; match={scene_counts[epoch]['matches_expected']})"
        )

    month_balanced = bool(cfg.get("composite", {}).get("month_balanced", False))
    months_by_epoch = {
        epoch: {int(str(item.properties["datetime"])[5:7]) for item in discovered_scenes[epoch]}
        for epoch in ("t1", "t2")
    }
    matched_months = sorted(months_by_epoch["t1"] & months_by_epoch["t2"]) if month_balanced else None
    if month_balanced and not matched_months:
        raise RuntimeError("Month-balanced mode found no common calendar months")

    scenes = {}
    month_drops: dict[str, list[dict[str, str]]] = {"t1": [], "t2": []}
    for epoch in ("t1", "t2"):
        scenes[epoch] = discovered_scenes[epoch]
        if matched_months is not None:
            scenes[epoch] = []
            for item in discovered_scenes[epoch]:
                month = int(str(item.properties["datetime"])[5:7])
                if month in matched_months:
                    scenes[epoch].append(item)
                else:
                    month_drops[epoch].append({
                        "scene_id": item.id,
                        "date": str(item.properties["datetime"])[:10],
                        "reason": "calendar month is not present in both epochs",
                    })
        if not scenes[epoch]:
            raise RuntimeError(f"No {epoch} scenes remain after matched-month filtering")
        print(f"{epoch.upper()} scene IDs:")
        print(json.dumps([item.id for item in scenes[epoch]], indent=2))

    monthly_distribution = {
        epoch: dict(sorted(Counter(str(item.properties["datetime"])[:7] for item in scenes[epoch]).items()))
        for epoch in ("t1", "t2")
    }
    print("Monthly scene distribution:", json.dumps(monthly_distribution, indent=2))

    metadata = {
        "provider": provider,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "gdal": gdal.VersionInfo("--version"),
            "rasterio": rasterio.__version__,
        },
        "stac_endpoint": (
            "https://earth-search.aws.element84.com/v1"
            if provider == "aws"
            else "https://planetarycomputer.microsoft.com/api/stac/v1"
        ),
        "collection": "sentinel-2-l2a",
        "scene_counts": scene_counts,
        "monthly_scene_counts": monthly_distribution,
        "deduplication": {
            "key": "(MGRS tile, calendar date)",
            "selection": "highest s2:processing_baseline; ties prefer S2A, then S2B, then S2C, then scene ID",
            "dropped_items": dedup_drops,
        },
        "cloud_filtering": {
            "tile_level_eo_cloud_cover_cutoff": cfg["sentinel2"].get("max_cloud_cover"),
            "strategy": "mask invalid SCL pixels within the AOI; do not reject a scene by tile-wide cloud percentage",
            "jan_feb_2019_scenes_dropped_by_cloud_cutoff": [],
        },
        "compositing": {
            "month_balanced": month_balanced,
            "bands": output_bands,
            "spatial_chunk_size": chunk_size,
            "workers": workers,
            "matched_calendar_months": matched_months,
            "month_filtered_items": month_drops,
            "method": (
                "median within each matched calendar month, then equal-month median"
                if month_balanced else "median across all retained observations"
            ),
        },
        "epochs": {epoch: [item.to_dict() for item in scenes[epoch]] for epoch in scenes},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    smoke = run_smoke_test(cfg, provider=provider)
    print("5x5 km smoke test:", json.dumps(smoke, indent=2, default=str))
    manifest = RunManifest(cfg)
    manifest._data["provider"] = provider
    manifest._data["runtime"] = metadata["runtime"]
    results = {
        epoch: build_composite_for_epoch(
            cfg, epoch, manifest=manifest, force_refresh=True, provider=provider,
            items=scenes[epoch], matched_months=matched_months,
            bands=output_bands, chunk_size=chunk_size, workers=workers,
        )
        for epoch in ("t1", "t2")
    }
    manifest.save(output_dir / "manifests")
    manifest_data = manifest.to_dict()
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest_data, indent=2, default=str), encoding="utf-8"
    )

    cog_dir = output_dir / "cogs"
    cog_dir.mkdir(exist_ok=True)
    for source in (smoke["composite_path"], smoke["valid_fraction_path"]):
        source = Path(source)
        shutil.copy2(source, cog_dir / source.name)
    for epoch, result in results.items():
        for key in ("composite_path", "valid_frac_path"):
            source = Path(result[key])
            shutil.copy2(source, cog_dir / f"{epoch}_{source.name}")

    ndvi_difference_path = None
    if fast_ndvi:
        ndvi_difference_path = _write_ndvi_difference(
            Path(results["t1"]["composite_path"]),
            Path(results["t2"]["composite_path"]),
            cog_dir / "ndvi_difference_t2_minus_t1.tif",
        )
        metadata["outputs"] = {"ndvi_difference_t2_minus_t1": str(ndvi_difference_path)}
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"NDVI difference COG: {ndvi_difference_path}")

    band_names = output_bands
    if fast_ndvi:
        print("Skipping RGB previews: fast mode contains only B04 and B08.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        for row, (title, bands) in enumerate(
            (("RGB B04/B03/B02", ("B04", "B03", "B02")),
             ("False colour B08/B04/B03", ("B08", "B04", "B03")))
        ):
            for col, epoch in enumerate(("t1", "t2")):
                with rasterio.open(results[epoch]["composite_path"]) as src:
                    descriptions = list(src.descriptions)
                    names = descriptions if all(descriptions) else band_names
                    data = src.read([names.index(b) + 1 for b in bands], masked=True).filled(np.nan)
                rgb = np.zeros_like(data, dtype="float32")
                for index, band in enumerate(data):
                    finite = np.isfinite(band)
                    if finite.any():
                        low, high = np.nanpercentile(band[finite], (2, 98))
                        if high > low:
                            rgb[index] = np.clip((band - low) / (high - low), 0, 1)
                axes[row, col].imshow(np.moveaxis(rgb, 0, -1))
                axes[row, col].set_title(f"{title} — {epoch.upper()}")
                axes[row, col].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "t1_t2_previews.png", dpi=160, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    report = [
        "# Data report", "",
        f"Run ID: `{manifest_data['run_id']}`  ",
        f"Python: `{sys.version.split()[0]}`; GDAL: `{gdal.VersionInfo('--version')}`; rasterio: `{rasterio.__version__}`  ",
        f"STAC provider: `{provider}`", "",
        "## Scene IDs", "",
    ]
    report.extend([
        "## Composite configuration", "",
        f"Mode: **{'fast NDVI' if fast_ndvi else 'full-band'}**; output bands: `{', '.join(output_bands)}`; SCL used for masking. Chunk size: `{chunk_size}` pixels; Dask workers: `{workers}`.",
        "",
    ])
    if ndvi_difference_path:
        report.extend([
            "## NDVI change output", "",
            "T2 minus T1 NDVI difference COG: `cogs/ndvi_difference_t2_minus_t1.tif`.",
            "",
        ])
    for epoch in ("t1", "t2"):
        report.extend([f"### {epoch.upper()}", ""])
        report.extend(f"- `{item.id}`" for item in scenes[epoch])
        report.append("")
    report.extend([
        "## Scene selection and temporal balance", "",
        "Deduplication key: `(MGRS tile, calendar date)`. Highest processing baseline is retained; same-baseline ties prefer S2A, then S2B, then S2C, then scene ID.",
        "",
        f"Month-balanced composites: **{'enabled' if month_balanced else 'disabled'}**. "
        + (f"Shared calendar months used: {', '.join(map(str, matched_months or []))}. Each month is composited separately then receives equal weight in the final median."
           if month_balanced else "All retained observations contribute to a single median."),
        "",
        "| Epoch | Deduplicated scenes | Expected | Count check |",
        "|---|---:|---:|---|",
    ])
    for epoch in ("t1", "t2"):
        count_info = scene_counts[epoch]
        report.append(
            f"| {epoch.upper()} | {count_info['deduplicated_count']} | {count_info['expected_count']} | "
            f"{'PASS' if count_info['matches_expected'] else 'MISMATCH'} |"
        )
    report.extend([
        "", "### Monthly distribution of composite input scenes", "",
        "| Epoch | Year-month | Scene count |", "|---|---|---:|",
    ])
    for epoch in ("t1", "t2"):
        for month, count in monthly_distribution[epoch].items():
            report.append(f"| {epoch.upper()} | {month} | {count} |")
    report.extend([
        "", "### January/February 2019 scene drops", "",
        "Tile-level cloud filtering is disabled; no scene is dropped on whole-tile cloud percentage. Cloud/shadow pixels are masked from SCL over the AOI. January/February scene IDs and reasons removed by deduplication are listed below; month-balance exclusions are also shown where applicable.",
        "",
    ])
    jan_feb_drops = [
        record for record in dedup_drops["t1"]
        if str(record.get("calendar_date", "")).startswith(("2019-01", "2019-02"))
    ] + [
        record for record in month_drops["t1"]
        if record.get("date", "").startswith(("2019-01", "2019-02"))
    ]
    if jan_feb_drops:
        report.extend(
            f"- `{record.get('dropped_scene_id', record.get('scene_id'))}` — {record.get('reason')}"
            for record in jan_feb_drops
        )
    else:
        report.append("- None. The tile-level cloud cutoff is disabled and no January/February 2019 T1 duplicates or unmatched-month items were dropped in this run.")
    report.extend([
        "", "## Reflectance offset comparison", "",
        "Same-tile 2018-12-01 baseline 00.01 vs 05.00 window check: **NOT YET RUN**. Run `scripts/check_offset.py --metadata <output>/metadata.json`; its per-band means and differences are not available until then.",
        "", "## Reprojection and tile seams", "",
        "Source inspection confirms stackstac loads to EPSG:32644 at 10 m, with bilinear resampling for reflectance and nearest-neighbour for SCL. Output seam inspection and boundary QA: **NOT YET RUN** on Colab.",
    ])
    report.extend([
        "## Valid-pixel coverage", "",
        "| Epoch | Mean valid-pixel coverage (%) |", "|---|---:|",
    ])
    coverage = {}
    stat_rows = []
    for epoch in ("t1", "t2"):
        with rasterio.open(results[epoch]["valid_frac_path"]) as src:
            coverage[epoch] = src.read(1, masked=True).filled(np.nan).astype("float32")
        mean_pct = float(np.nanmean(coverage[epoch]) * 100)
        report.append(f"| {epoch.upper()} | {mean_pct:.4f} |")
        with rasterio.open(results[epoch]["composite_path"]) as src:
            names = list(src.descriptions)
            if not all(names):
                names = band_names
            for index, name in enumerate(names, 1):
                values = src.read(index, masked=True).compressed().astype("float64")
                stats = (
                    (0, math.nan, math.nan, math.nan, math.nan)
                    if values.size == 0
                    else (
                        values.size, float(values.min()), float(values.mean()),
                        float(np.median(values)), float(values.max()),
                    )
                )
                stat_rows.append((epoch.upper(), name, *stats))
    report.extend([
        "", "## Per-band composite statistics", "",
        "Computed from valid pixels read from this run's COGs; units are the stored values.", "",
        "| Epoch | Band | Valid pixels | Min | Mean | Median | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for epoch, band, count, minimum, mean, median, maximum in stat_rows:
        report.append(
            f"| {epoch} | {band} | {count} | {minimum:.7g} | {mean:.7g} | "
            f"{median:.7g} | {maximum:.7g} |"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, epoch in zip(axes, ("t1", "t2")):
        image = ax.imshow(coverage[epoch] * 100, cmap="viridis", vmin=0, vmax=100)
        ax.set_title(f"{epoch.upper()} valid observations (%)")
        ax.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.75, label="Valid observations (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "valid_pixel_map.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    report.extend([
        "", "## Valid-pixel map", "",
        "![T1 and T2 valid-observation coverage](valid_pixel_map.png)", "",
        "Generated from the T1 and T2 valid-pixel fraction COGs written during this run.",
    ])
    text = "\n".join(report) + "\n"
    report_path = repo / "docs" / "data_report.md"
    report_path.write_text(text, encoding="utf-8")
    shutil.copy2(output_dir / "valid_pixel_map.png", report_path.parent / "valid_pixel_map.png")
    shutil.copy2(report_path, output_dir / "data_report.md")
    print(text)
    print(f"Run products: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("aws", "pc"), default="aws")
    parser.add_argument("--fast-ndvi", action="store_true",
                        help="Build B04/B08 composites with SCL masking and a T2-T1 NDVI COG")
    parser.add_argument("--chunk-size", type=int,
                        help="Spatial chunk pixels (default 512 fast, 256 full)")
    parser.add_argument("--workers", type=int,
                        help="Dask workers per output chunk (default 4 fast, 2 full)")
    args = parser.parse_args()
    run(args.repo, args.output, args.provider, args.fast_ndvi, args.chunk_size, args.workers)
