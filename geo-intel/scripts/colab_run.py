"""Hosted Python 3.11 execution and output-derived report for GEO-INTEL."""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from osgeo import gdal

from geointel.data.composite import build_composite_for_epoch, run_smoke_test
from geointel.data.stac_fetch import fetch_or_load
from geointel.utils.config import load_config
from geointel.utils.manifest import RunManifest


def run(repo: Path, output_dir: Path, provider: str = "aws") -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 is required; found {sys.version}")
    repo, output_dir = repo.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)

    print(f"Python: {sys.version}")
    print(f"GDAL: {gdal.VersionInfo('--version')}")
    print(f"rasterio: {rasterio.__version__}")
    print(f"Provider: {provider}")

    cfg = load_config("config/config.yaml")
    for epoch in ("t1", "t2"):
        cached = cfg["paths"]["data_cache"] / "composites" / epoch
        if cached.exists():
            shutil.rmtree(cached)

    scenes = {}
    for epoch in ("t1", "t2"):
        scenes[epoch] = fetch_or_load(cfg, epoch, provider=provider, force_refresh=True)
        if not scenes[epoch]:
            raise RuntimeError(f"No scenes returned by {provider} for {epoch}")
        print(f"{epoch.upper()} scene IDs:")
        print(json.dumps([item.id for item in scenes[epoch]], indent=2))

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
            cfg, epoch, manifest=manifest, force_refresh=True, provider=provider
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

    band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
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
    for epoch in ("t1", "t2"):
        report.extend([f"### {epoch.upper()}", ""])
        report.extend(f"- `{item.id}`" for item in scenes[epoch])
        report.append("")
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
    args = parser.parse_args()
    run(args.repo, args.output, args.provider)
