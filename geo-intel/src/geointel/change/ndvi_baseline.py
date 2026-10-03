"""NDVI difference baseline for comparing two reflectance composites."""
from __future__ import annotations

from typing import Any

import numpy as np


def ndvi_difference_baseline(
    ndvi_t1: Any,
    ndvi_t2: Any,
    threshold: float = -0.15,
    nodata: int = 255,
) -> dict[str, Any]:
    """Return T2-T1 NDVI and a decline mask; this is a baseline, not attribution.

    Inputs are NDVI arrays derived from the corresponding reflectance composites.
    """
    ndvi1 = np.asarray(ndvi_t1, dtype="float32")
    ndvi2 = np.asarray(ndvi_t2, dtype="float32")
    if ndvi1.shape != ndvi2.shape:
        raise ValueError(f"NDVI shape mismatch: {ndvi1.shape} vs {ndvi2.shape}")
    valid = np.isfinite(ndvi1) & np.isfinite(ndvi2)
    diff = np.full(ndvi1.shape, np.nan, dtype="float32")
    diff[valid] = ndvi2[valid] - ndvi1[valid]
    decline = np.full(ndvi1.shape, nodata, dtype="uint8")
    decline[valid] = (diff[valid] < threshold).astype("uint8")
    return {
        "ndvi_t1": ndvi1,
        "ndvi_t2": ndvi2,
        "difference_t2_minus_t1": diff,
        "decline_mask": decline,
        "threshold": float(threshold),
        "valid_pixels": int(valid.sum()),
        "decline_pixels": int(np.count_nonzero(valid & (diff < threshold))),
    }
