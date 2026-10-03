"""
change/detection.py — LULC change detection and transition matrix computation for GEO-INTEL.

Computes:
1. Full 5x5 LULC transition matrix (T1 → T2) in pixel count and square kilometres (km²).
2. Net change, gain, loss, and percentage change per class.
3. Specific transition masks:
   - Urban expansion mask: Non-Built-up in T1 → Built-up in T2
   - Vegetation loss mask: Vegetation in T1 → Non-Vegetation in T2
   - Water transition mask: Water gain/loss

Spatial Unit Guarantee
----------------------
Resolution = 10 m → Pixel area = 100 m² = 0.0001 km².
All area calculations explicitly assert positive pixel scale factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from geointel.change.rule_based import LULC_CLASSES
from geointel.utils.logging import get_logger

logger = get_logger(__name__)


def compute_transition_matrix(
    lulc_t1: xr.DataArray | np.ndarray,
    lulc_t2: xr.DataArray | np.ndarray,
    pixel_size_m: float = 10.0,
) -> dict[str, Any]:
    """
    Compute 5x5 transition matrix and net change statistics between T1 and T2.

    Parameters
    ----------
    lulc_t1 : DataArray or ndarray
        LULC map at time 1 (integer array with values 1..5, 0 or NaN for invalid).
    lulc_t2 : DataArray or ndarray
        LULC map at time 2 (integer array with values 1..5, 0 or NaN for invalid).
    pixel_size_m : float
        Pixel spatial resolution in metres. Default 10.0 m.

    Returns
    -------
    dict containing:
        'matrix_pixels': 5x5 ndarray (rows=T1, cols=T2 for classes 1..5)
        'matrix_km2': 5x5 ndarray in km²
        'class_stats': dict of per-class area in T1, T2, net_change_km2, pct_change
        'transitions_km2': dict of notable class transition areas (e.g. Veg -> Built)
    """
    arr1 = lulc_t1.values if isinstance(lulc_t1, xr.DataArray) else lulc_t1
    arr2 = lulc_t2.values if isinstance(lulc_t2, xr.DataArray) else lulc_t2

    assert arr1.shape == arr2.shape, f"LULC map shape mismatch: {arr1.shape} vs {arr2.shape}"

    # Pixel area in square kilometres
    pixel_area_km2 = (pixel_size_m ** 2) / 1_000_000.0  # 100 m² = 0.0001 km²

    valid_mask = (
        np.isfinite(arr1) & np.isfinite(arr2)
        & (arr1 >= 1) & (arr1 <= 5)
        & (arr2 >= 1) & (arr2 <= 5)
    )

    matrix_pixels = np.zeros((5, 5), dtype=np.int64)

    for i in range(1, 6):
        for j in range(1, 6):
            mask = valid_mask & (arr1 == i) & (arr2 == j)
            matrix_pixels[i - 1, j - 1] = int(mask.sum())

    matrix_km2 = matrix_pixels * pixel_area_km2

    # Per-class summary statistics
    class_stats = {}
    for c_id in range(1, 6):
        c_name = LULC_CLASSES[c_id]
        t1_pixels = int((valid_mask & (arr1 == c_id)).sum())
        t2_pixels = int((valid_mask & (arr2 == c_id)).sum())
        
        t1_km2 = float(t1_pixels * pixel_area_km2)
        t2_km2 = float(t2_pixels * pixel_area_km2)
        net_change_km2 = t2_km2 - t1_km2
        pct_change = (net_change_km2 / t1_km2 * 100.0) if t1_km2 > 0 else 0.0

        class_stats[c_name] = {
            "class_id": c_id,
            "t1_km2": round(t1_km2, 4),
            "t2_km2": round(t2_km2, 4),
            "net_change_km2": round(net_change_km2, 4),
            "pct_change": round(pct_change, 2),
        }

    # Key environmental transitions
    transitions_km2 = {
        "vegetation_to_builtup_km2": round(float(matrix_km2[1, 0] + matrix_km2[2, 0]), 4),
        "soil_to_builtup_km2": round(float(matrix_km2[3, 0]), 4),
        "vegetation_loss_total_km2": round(float(
            matrix_km2[1, [0, 3, 4]].sum() + matrix_km2[2, [0, 3, 4]].sum()
        ), 4),
        "urban_expansion_total_km2": round(float(matrix_km2[1:, 0].sum()), 4),
    }

    logger.info(
        "Computed transition matrix. Urban Expansion: %.4f km²",
        transitions_km2["urban_expansion_total_km2"],
    )

    return {
        "matrix_pixels": matrix_pixels.tolist(),
        "matrix_km2": np.round(matrix_km2, 4).tolist(),
        "class_stats": class_stats,
        "transitions_km2": transitions_km2,
        "total_valid_area_km2": round(float(valid_mask.sum() * pixel_area_km2), 4),
    }


def compute_spatial_change_masks(
    lulc_t1: xr.DataArray | np.ndarray,
    lulc_t2: xr.DataArray | np.ndarray,
) -> dict[str, xr.DataArray | np.ndarray]:
    """
    Derive spatial binary masks for specific change trajectory classes.

    Returns
    -------
    dict with keys:
        'urban_expansion': 1 where non-built → Built-up (1), 0 elsewhere
        'vegetation_loss': 1 where tree/cropland (2,3) → non-vegetation, 0 elsewhere
        'unchanged': 1 where T1 == T2, 0 elsewhere
    """
    arr1 = lulc_t1.values if isinstance(lulc_t1, xr.DataArray) else lulc_t1
    arr2 = lulc_t2.values if isinstance(lulc_t2, xr.DataArray) else lulc_t2

    valid = (
        np.isfinite(arr1) & np.isfinite(arr2)
        & (arr1 >= 1) & (arr1 <= 5)
        & (arr2 >= 1) & (arr2 <= 5)
    )

    urban_exp = valid & (arr1 != 1) & (arr2 == 1)
    t1_vegetation = (arr1 == 2) | (arr1 == 3)
    t2_vegetation = (arr2 == 2) | (arr2 == 3)
    veg_loss = valid & t1_vegetation & (~t2_vegetation)
    unchanged = valid & (arr1 == arr2)

    if isinstance(lulc_t1, xr.DataArray):
        coords = lulc_t1.coords
        dims = lulc_t1.dims
        return {
            "urban_expansion": xr.DataArray(urban_exp.astype(np.uint8), dims=dims, coords=coords),
            "vegetation_loss": xr.DataArray(veg_loss.astype(np.uint8), dims=dims, coords=coords),
            "unchanged": xr.DataArray(unchanged.astype(np.uint8), dims=dims, coords=coords),
        }

    return {
        "urban_expansion": urban_exp.astype(np.uint8),
        "vegetation_loss": veg_loss.astype(np.uint8),
        "unchanged": unchanged.astype(np.uint8),
    }
