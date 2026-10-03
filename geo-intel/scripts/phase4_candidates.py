"""Classify hosted composite COGs with the baseline rules and make label candidates.

Writes rule-map rasters, a stratified candidate GeoPackage, and an empty label CSV
template. It does not create reference labels or accuracy results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import xarray as xr

from geointel.change.rule_based import classify_rule_based
from geointel.utils.config import load_config
from scripts.sample_labels import create_candidate_files

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def make_rule_map(composite_path: Path, output_path: Path) -> Path:
    with rasterio.open(composite_path) as source:
        values = source.read().astype("float32")
        descriptions = list(source.descriptions)
        bands = descriptions if len(descriptions) == len(BANDS) and all(descriptions) else BANDS
        image = xr.DataArray(
            values,
            dims=("band", "y", "x"),
            coords={"band": bands, "y": np.arange(source.height), "x": np.arange(source.width)},
            attrs={"crs": source.crs.to_string() if source.crs else None},
        )
        classified = classify_rule_based(image).values.astype("uint8")
        profile = source.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(classified, 1)
        destination.set_band_description(1, "rule_based_lulc_class")
        destination.update_tags(
            classification="rule-based baseline; candidate strata only",
            classes="1=built-up,2=tree,3=cropland/grass,4=bare,5=water,0=unclassified",
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--t1-composite", type=Path, required=True)
    parser.add_argument("--t2-composite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    maps_dir = args.output_dir / "rule_maps"
    t1_map = make_rule_map(args.t1_composite, maps_dir / "lulc_t1_rule.tif")
    t2_map = make_rule_map(args.t2_composite, maps_dir / "lulc_t2_rule.tif")
    create_candidate_files(
        t1_map, t2_map, args.output_dir / "labels",
        samples_per_class=int(cfg["classifier"]["target_labels_per_class"]),
        seed=int(cfg["reproducibility"]["numpy_seed"]),
    )
    print("Reference labels created: 0. The candidate classes are map-derived strata only.")


if __name__ == "__main__":
    main()
