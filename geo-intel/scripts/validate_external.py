"""Compare mapped patterns with GHSL, Hansen GFC, and ESA WorldCover rasters.

This is external consistency checking, not a substitute for reference labels or
accuracy assessment. Inputs should be manually downloaded and documented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def read_on_grid(
    path: Path,
    reference: rasterio.io.DatasetReader,
    method: Resampling = Resampling.nearest,
    *,
    zero_is_valid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        destination = np.zeros((reference.height, reference.width), dtype=src.dtypes[0])
        valid_source = np.ones((src.height, src.width), dtype=bool) if zero_is_valid else (src.read_masks(1) > 0)
        valid_destination = np.zeros((reference.height, reference.width), dtype="uint8")
        reproject(
            source=rasterio.band(src, 1), destination=destination,
            src_transform=src.transform, src_crs=src.crs, src_nodata=None if zero_is_valid else src.nodata,
            dst_transform=reference.transform, dst_crs=reference.crs, dst_nodata=0,
            resampling=method,
        )
        reproject(
            source=valid_source.astype("uint8"), destination=valid_destination,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=reference.transform, dst_crs=reference.crs,
            resampling=Resampling.nearest,
        )
        valid = valid_destination > 0
        return destination, valid


def overlap_stats(mapped: np.ndarray, external: np.ndarray, valid: np.ndarray) -> dict[str, float | int | None]:
    a, b = mapped[valid].astype(bool), external[valid].astype(bool)
    tp = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return {
        "common_valid_pixels": int(valid.sum()),
        "mapped_positive_pixels": int(a.sum()),
        "external_positive_pixels": int(b.sum()),
        "intersection_over_union": float(tp / union) if union else None,
        "mapped_positive_fraction_overlapping_external": float(tp / a.sum()) if a.sum() else None,
        "external_positive_fraction_overlapping_mapped": float(tp / b.sum()) if b.sum() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urban-expansion", type=Path, required=True)
    parser.add_argument("--vegetation-loss", type=Path, required=True)
    parser.add_argument("--lulc-t1", type=Path, required=True)
    parser.add_argument("--lulc-t2", type=Path, required=True)
    parser.add_argument("--ghsl-t1", type=Path)
    parser.add_argument("--ghsl-t2", type=Path)
    parser.add_argument("--worldcover-2021", type=Path)
    parser.add_argument("--hansen-lossyear", type=Path)
    parser.add_argument("--hansen-datamask", type=Path, help="recommended: use this to distinguish valid no-loss pixels")
    parser.add_argument("--hansen-treecover2000", type=Path)
    parser.add_argument("--hansen-treecover-threshold", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hansen-first-year", type=int, default=2020)
    parser.add_argument("--hansen-last-year", type=int, default=2025)
    args = parser.parse_args()

    results: dict[str, object] = {
        "status": "EXTERNAL CONSISTENCY CHECK; NOT REFERENCE ACCURACY",
        "comparisons": {},
        "notes": [
            "Different sensors, class definitions, spatial supports, and dates limit interpretation.",
            "Positive overlap is descriptive and does not establish correctness or causation.",
        ],
    }
    with rasterio.open(args.urban_expansion) as urban:
        urban_values = urban.read(1)
        urban_valid = (urban_values != urban.nodata) if urban.nodata is not None else np.ones(urban_values.shape, bool)
        for epoch, path in (("t1", args.ghsl_t1), ("t2", args.ghsl_t2)):
            if path:
                ghsl, valid = read_on_grid(path, urban, Resampling.bilinear)
                # GHSL built surface is positive where built-up surface is present.
                lulc_path = args.lulc_t1 if epoch == "t1" else args.lulc_t2
                lulc, lulc_valid = read_on_grid(lulc_path, urban)
                results["comparisons"][f"ghsl_{epoch}_built_presence"] = overlap_stats(
                    lulc == 1, ghsl > 0, lulc_valid & valid
                )
                results["comparisons"][f"ghsl_{epoch}_mapped_vs_reference_overlap_pixels"] = int(
                    np.count_nonzero((lulc == 1) & (ghsl > 0) & lulc_valid & valid)
                )
        if args.ghsl_t1 and args.ghsl_t2:
            g1, v1 = read_on_grid(args.ghsl_t1, urban, Resampling.bilinear)
            g2, v2 = read_on_grid(args.ghsl_t2, urban, Resampling.bilinear)
            ghsl_gain = (g1 == 0) & (g2 > 0)
            results["comparisons"]["ghsl_built_presence_gain_vs_mapped_urban_expansion"] = overlap_stats(
                urban_values == 1, ghsl_gain, urban_valid & v1 & v2
            )
        if args.hansen_lossyear:
            hansen, valid = read_on_grid(args.hansen_lossyear, urban, zero_is_valid=True)
            if args.hansen_datamask:
                data_mask, mask_valid = read_on_grid(args.hansen_datamask, urban)
                valid &= mask_valid & (data_mask == 1)
            first_code, last_code = args.hansen_first_year - 2000, args.hansen_last_year - 2000
            loss = (hansen >= first_code) & (hansen <= last_code)
            if args.hansen_treecover2000:
                treecover, tree_valid = read_on_grid(args.hansen_treecover2000, urban)
                valid &= tree_valid
                loss &= treecover >= args.hansen_treecover_threshold
            veg, veg_valid = read_on_grid(args.vegetation_loss, urban)
            results["comparisons"]["hansen_forest_loss_year_range"] = overlap_stats(
                veg == 1, loss, urban_valid & valid & veg_valid
            ) | {"year_range": [args.hansen_first_year, args.hansen_last_year]}
            if args.hansen_treecover2000:
                results["comparisons"]["hansen_forest_loss_year_range"]["treecover2000_threshold_percent"] = args.hansen_treecover_threshold
            if not args.hansen_datamask:
                results["notes"].append("Hansen datamask was not supplied; zero-coded lossyear validity may be ambiguous.")
    if args.worldcover_2021:
        with rasterio.open(args.lulc_t2) as t2:
            predicted, pred_valid = read_on_grid(args.lulc_t2, t2)
            wc, wc_valid = read_on_grid(args.worldcover_2021, t2)
            common = pred_valid & wc_valid
            # ESA WorldCover Map class 50 is built-up; Map class 10 is tree cover.
            results["comparisons"]["worldcover_2021_built_vs_t2"] = overlap_stats(
                predicted == 1, wc == 50, common
            )
            results["comparisons"]["worldcover_2021_tree_vs_t2"] = overlap_stats(
                predicted == 2, wc == 10, common
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
