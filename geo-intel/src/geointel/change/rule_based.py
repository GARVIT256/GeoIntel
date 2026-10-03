"""
change/rule_based.py — Decision tree rule-based LULC classifier (Baseline 1).

Applies deterministic spectral rules to classify Sentinel-2 pixels into five
land-use/land-cover (LULC) classes:

Class Codes
-----------
1: Built-up            (NDBI > NDVI and NDBI > 0.0)
2: Tree/dense veg      (NDVI > 0.50)
3: Cropland/grass      (0.25 < NDVI <= 0.50)
4: Bare/sparse         (BSI > 0.0 and NDVI <= 0.25)
5: Water               (MNDWI > 0.0)
0: Unclassified / NaN

Decision hierarchy (non-overlapping precedence):
1. Water (MNDWI > 0.0) evaluated FIRST.
2. Built-up (NDBI > NDVI and NDBI > 0.0).
3. Tree/dense vegetation (NDVI > 0.50).
4. Cropland/grass (0.25 < NDVI <= 0.50).
5. Bare/sparse (BSI > 0.0); other valid land remains unclassified.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from geointel.change.indices import compute_all_indices
from geointel.utils.logging import get_logger

logger = get_logger(__name__)

# LULC Class Mapping
LULC_CLASSES = {
    0: "Unclassified",
    1: "Built-up",
    2: "Tree",
    3: "Cropland/grass",
    4: "Bare",
    5: "Water",
}

CLASS_COLOR_MAP = {
    0: "#000000",  # Black / No-data
    1: "#d62728",  # Built-up
    2: "#2ca02c",  # Green
    3: "#b2df8a",  # Low vegetation
    4: "#fdbf6f",  # Bare/sparse
    5: "#1f77b4",  # Water
}


def classify_rule_based(
    composite_or_indices: xr.DataArray,
) -> xr.DataArray:
    """
    Classify pixels into LULC categories using decision rules.

    Parameters
    ----------
    composite_or_indices : xr.DataArray
        Can be either a 6-band composite (band=["B02", ...]) or a 4-band index stack
        (index_band=["NDVI", "NDBI", "MNDWI", "BSI"]).

    Returns
    -------
    xr.DataArray
        Single-band uint8 raster (y, x) with pixel values 0..5.
    """
    if "band" in composite_or_indices.dims:
        indices = compute_all_indices(composite_or_indices)
    elif "index_band" in composite_or_indices.dims:
        indices = composite_or_indices
    else:
        raise ValueError("Input DataArray must have 'band' or 'index_band' dimension")

    ndvi = indices.sel(index_band="NDVI").values
    ndbi = indices.sel(index_band="NDBI").values
    mndwi = indices.sel(index_band="MNDWI").values
    bsi = indices.sel(index_band="BSI").values

    # Output array initialized to 0 (Unclassified / No-data)
    lulc = np.zeros(ndvi.shape, dtype=np.uint8)

    # Valid pixel mask (where indices are not NaN)
    valid_mask = ~(np.isnan(ndvi) | np.isnan(ndbi) | np.isnan(mndwi) | np.isnan(bsi))

    # Rule 1: Water (MNDWI > 0.0)
    water_mask = valid_mask & (mndwi > 0.0)
    lulc[water_mask] = 5

    # Non-water mask
    land_mask = valid_mask & (~water_mask)

    # Rule 2: Built-up (NDBI > NDVI and NDBI > 0.0)
    built_mask = land_mask & (ndbi > ndvi) & (ndbi > 0.0)
    lulc[built_mask] = 1

    # Remaining land mask after higher-priority water and built-up rules.
    rem_land_mask = land_mask & (~built_mask)

    # Rules 3-4 separate dense/tree cover from lower vegetation.
    tree_mask = rem_land_mask & (ndvi > 0.50)
    lulc[tree_mask] = 2
    crop_mask = rem_land_mask & (~tree_mask) & (ndvi > 0.25)
    lulc[crop_mask] = 3

    # Bare/sparse is lower priority than vegetation and requires both cues.
    # Low-NDVI pixels with BSI <= 0 remain unclassified.
    soil_mask = rem_land_mask & (~tree_mask) & (~crop_mask) & (bsi > 0.0)
    lulc[soil_mask] = 4

    out_coords = {k: v for k, v in indices.coords.items() if k not in ["band", "index_band"]}
    out_da = xr.DataArray(
        lulc,
        dims=[d for d in indices.dims if d not in ["band", "index_band"]],
        coords=out_coords,
        attrs={
            "description": "Rule-based five-class LULC map",
            "classes": LULC_CLASSES,
            "crs": indices.attrs.get("crs", "EPSG:32644"),
        },
    )

    class_counts = {LULC_CLASSES[k]: int((lulc == k).sum()) for k in LULC_CLASSES}
    logger.info("Rule-based classification completed. Class counts: %s", class_counts)
    return out_da
