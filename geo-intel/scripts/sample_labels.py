"""Create stratified QGIS-ready candidate points and an empty label CSV schema.

This command creates no reference labels. Candidate class values are map-derived
strata only and must not be treated as ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Point

from geointel.change.sampling import build_candidate_records
from geointel.utils.config import load_config

LABEL_COLUMNS = ["id", "lon", "lat", "candidate_epoch", "class_t1", "class_t2", "labeller", "confidence", "notes"]


def create_candidate_files(
    t1_map: Path,
    t2_map: Path,
    output_dir: Path,
    samples_per_class: int = 150,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write candidate point GeoPackage and empty, manually completed CSV header."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(t1_map) as t1, rasterio.open(t2_map) as t2:
        if t1.crs is None or t2.crs is None:
            raise ValueError("Both classified rasters must declare a CRS")
        if (t1.crs != t2.crs or t1.transform != t2.transform
                or t1.width != t2.width or t1.height != t2.height):
            raise ValueError("T1 and T2 class rasters must use the same CRS and pixel grid")
        map_t1, map_t2 = t1.read(1), t2.read(1)
        candidates = build_candidate_records(map_t1, map_t2, samples_per_class, seed)
        to_wgs84 = Transformer.from_crs(t1.crs, "EPSG:4326", always_xy=True)
        rows = []
        for index, candidate in enumerate(candidates, start=1):
            x, y = rasterio.transform.xy(t1.transform, candidate["row"], candidate["col"], offset="center")
            lon, lat = to_wgs84.transform(x, y)
            rows.append({
                "id": f"candidate-{index:07d}",
                "lon": lon,
                "lat": lat,
                "class_t1": None,
                "class_t2": None,
                "labeller": None,
                "confidence": None,
                "notes": None,
                "candidate_epoch": candidate["candidate_epoch"],
                "candidate_class": candidate["candidate_class"],
                "pred_t1": candidate["pred_t1"],
                "pred_t2": candidate["pred_t2"],
                "geometry": Point(x, y),
            })
        columns = [
            "id", "lon", "lat", "class_t1", "class_t2", "labeller", "confidence",
            "notes", "candidate_epoch", "candidate_class", "pred_t1", "pred_t2", "geometry",
        ]
        candidates_gdf = gpd.GeoDataFrame(
            pd.DataFrame(rows, columns=columns), geometry="geometry", crs=t1.crs
        )
        if candidates_gdf.empty:
            raise ValueError("No pixels in classes 1..5; no candidate points can be generated")
        for column in ("class_t1", "class_t2"):
            candidates_gdf[column] = pd.array(candidates_gdf[column], dtype="Int64")
        candidates_gdf["confidence"] = pd.array(candidates_gdf["confidence"], dtype="Float64")
        for column in ("labeller", "notes"):
            candidates_gdf[column] = pd.array(candidates_gdf[column], dtype="string")

    gpkg_path = output_dir / "candidate_points.gpkg"
    candidates_gdf.to_file(gpkg_path, layer="candidate_points", driver="GPKG")
    csv_path = output_dir / "labels_template.csv"
    pd.DataFrame(columns=LABEL_COLUMNS).to_csv(csv_path, index=False)
    print(f"Candidate points: {gpkg_path} (map-derived strata, not reference labels)")
    print(f"Empty label schema: {csv_path}; no labels were created")
    print("Candidates by epoch and class:")
    if not candidates_gdf.empty:
        print(candidates_gdf.groupby(["candidate_epoch", "candidate_class"]).size().to_string())
    return gpkg_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--t1-map", type=Path, required=True, help="T1 five-class prediction GeoTIFF")
    parser.add_argument("--t2-map", type=Path, required=True, help="T2 five-class prediction GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=Path("data/labels/candidates"))
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    samples = (
        args.samples_per_class
        if args.samples_per_class is not None
        else int(cfg["classifier"]["target_labels_per_class"])
    )
    seed = args.seed if args.seed is not None else int(cfg["reproducibility"]["numpy_seed"])
    create_candidate_files(args.t1_map, args.t2_map, args.output_dir, samples, seed)


if __name__ == "__main__":
    main()
