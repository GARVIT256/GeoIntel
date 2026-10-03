"""Deterministic majority filtering and minimum mapping unit cleanup."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


def majority_filter(classes: np.ndarray, window_size: int = 3, nodata: int = 0) -> np.ndarray:
    """Replace each valid cell by its neighborhood mode; keep nodata centers.

    Ties retain the center class when it is tied for the mode, otherwise the
    smallest class ID wins. Neighborhoods beyond the raster edge are nodata.
    """
    values = np.asarray(classes)
    if values.ndim != 2:
        raise ValueError("classes must be a two-dimensional array")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    result = values.copy()
    radius = window_size // 2
    padded = np.pad(values, radius, mode="constant", constant_values=nodata)
    for row, col in zip(*np.nonzero(values != nodata), strict=True):
        window = padded[row:row + window_size, col:col + window_size]
        ids, counts = np.unique(window[window != nodata], return_counts=True)
        if ids.size:
            modes = ids[counts == counts.max()]
            result[row, col] = values[row, col] if values[row, col] in modes else modes[0]
    return result


def remove_small_patches(
    classes: np.ndarray,
    pixel_size_m: float = 10.0,
    minimum_area_ha: float = 0.5,
    nodata: int = 0,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Set same-class components below the area threshold to nodata.

    Components use 4- or 8-connectivity. Removed pixels are not assigned to a
    neighboring class, avoiding implicit relabeling as reference truth.
    """
    values = np.asarray(classes)
    if values.ndim != 2:
        raise ValueError("classes must be a two-dimensional array")
    if pixel_size_m <= 0 or minimum_area_ha < 0:
        raise ValueError("pixel_size_m must be positive and minimum_area_ha nonnegative")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    minimum_pixels = math.ceil(minimum_area_ha * 10000 / pixel_size_m**2)
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    result = values.copy()
    removed: dict[str, int] = {}
    for class_id in np.unique(values):
        if class_id == nodata:
            continue
        labels, count = ndimage.label(values == class_id, structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        small = np.flatnonzero((sizes < minimum_pixels) & (np.arange(sizes.size) > 0))
        if small.size:
            mask = np.isin(labels, small)
            result[mask] = nodata
            removed[str(int(class_id))] = int(mask.sum())
    total = sum(removed.values())
    return result, {
        "minimum_area_ha": float(minimum_area_ha),
        "minimum_pixels": int(minimum_pixels),
        "pixel_size_m": float(pixel_size_m),
        "connectivity": connectivity,
        "removed_pixels_by_class": removed,
        "removed_pixels": total,
        "removed_area_ha": float(total * pixel_size_m**2 / 10000),
    }


def postprocess_class_map(
    classes: np.ndarray,
    majority_window: int = 3,
    minimum_area_ha: float = 0.5,
    pixel_size_m: float = 10.0,
    nodata: int = 0,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply majority filtering first, then remove sub-MMU components."""
    majority = majority_filter(classes, majority_window, nodata)
    cleaned, mmu = remove_small_patches(
        majority, pixel_size_m, minimum_area_ha, nodata, connectivity
    )
    return cleaned, {"majority_window": majority_window, "mmu": mmu}
