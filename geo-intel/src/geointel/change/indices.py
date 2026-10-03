"""
change/indices.py — Spectral index calculation for Sentinel-2 composites in GEO-INTEL.

Computes standard remote-sensing spectral indices:
- NDVI (Normalized Difference Vegetation Index): (B08 - B04) / (B08 + B04)
- NDBI (Normalized Difference Built-up Index):  (B11 - B08) / (B11 + B08)
- MNDWI (Modified Normalized Difference Water Index): (B03 - B11) / (B03 + B11)
- BSI (Bare Soil Index): ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))

Band mappings (Sentinel-2 L2A):
  B02: Blue     (490 nm, 10 m)
  B03: Green    (560 nm, 10 m)
  B04: Red      (665 nm, 10 m)
  B08: NIR      (842 nm, 10 m)
  B11: SWIR-1  (1610 nm, 20 m resampled to 10 m)
  B12: SWIR-2  (2190 nm, 20 m resampled to 10 m)

Numerical safety
----------------
- Uses np.where and np.errstate to handle division by zero.
- Replaces zero-denominator pixels with NaN.
- Clips output values to valid theoretical index bounds [-1.0, +1.0].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from geointel.utils.logging import get_logger

logger = get_logger(__name__)


def compute_normalized_difference(
    b1: np.ndarray | xr.DataArray,
    b2: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Generic normalized difference formula: (b1 - b2) / (b1 + b2).

    Parameters
    ----------
    b1 : ndarray or DataArray
        First spectral band.
    b2 : ndarray or DataArray
        Second spectral band.

    Returns
    -------
    ndarray or DataArray
        Index values in [-1.0, +1.0], NaN at zero denominator or masked inputs.
    """
    denom = b1 + b2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0.0, np.nan, (b1 - b2) / denom)
    
    # Clip to theoretical bounds [-1.0, +1.0]
    result = np.clip(result, -1.0, 1.0)
    
    if isinstance(b1, xr.DataArray):
        return xr.DataArray(
            result.astype("float32"),
            dims=b1.dims,
            coords=b1.coords,
            attrs=b1.attrs,
        )
    return result.astype("float32")


def compute_ndvi(
    b08_nir: np.ndarray | xr.DataArray,
    b04_red: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Target range: [-1.0, 1.0]. Dense vegetation > 0.5, sparse 0.2-0.5, water < 0.0.
    """
    return compute_normalized_difference(b08_nir, b04_red)


def compute_ndbi(
    b11_swir1: np.ndarray | xr.DataArray,
    b08_nir: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Target range: [-1.0, 1.0]. Built-up/impervious surfaces > 0.0, vegetation < 0.0.
    """
    return compute_normalized_difference(b11_swir1, b08_nir)


def compute_mndwi(
    b03_green: np.ndarray | xr.DataArray,
    b11_swir1: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    Target range: [-1.0, 1.0]. Water bodies > 0.0, land < 0.0.
    """
    return compute_normalized_difference(b03_green, b11_swir1)


def compute_bsi(
    b11_swir1: np.ndarray | xr.DataArray,
    b04_red: np.ndarray | xr.DataArray,
    b08_nir: np.ndarray | xr.DataArray,
    b02_blue: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    Target range: [-1.0, 1.0]. Bare soil / exposed earth > 0.0, vegetation/water < 0.0.
    """
    num = (b11_swir1 + b04_red) - (b08_nir + b02_blue)
    denom = (b11_swir1 + b04_red) + (b08_nir + b02_blue)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0.0, np.nan, num / denom)
    
    result = np.clip(result, -1.0, 1.0)
    
    if isinstance(b11_swir1, xr.DataArray):
        return xr.DataArray(
            result.astype("float32"),
            dims=b11_swir1.dims,
            coords=b11_swir1.coords,
            attrs=b11_swir1.attrs,
        )
    return result.astype("float32")


def compute_all_indices(
    composite: xr.DataArray,
) -> xr.DataArray:
    """
    Compute all four spectral indices for a 6-band Sentinel-2 composite DataArray.

    Parameters
    ----------
    composite : xr.DataArray
        Array of shape (band, y, x) containing bands ["B02", "B03", "B04", "B08", "B11", "B12"].

    Returns
    -------
    xr.DataArray
        Array of shape (index_band, y, x) containing ["NDVI", "NDBI", "MNDWI", "BSI"].
    """
    b02 = composite.sel(band="B02")
    b03 = composite.sel(band="B03")
    b04 = composite.sel(band="B04")
    b08 = composite.sel(band="B08")
    b11 = composite.sel(band="B11")

    ndvi = compute_ndvi(b08, b04)
    ndbi = compute_ndbi(b11, b08)
    mndwi = compute_mndwi(b03, b11)
    bsi = compute_bsi(b11, b04, b08, b02)

    indices_stack = xr.concat(
        [ndvi, ndbi, mndwi, bsi],
        dim="index_band",
    )
    indices_stack.coords["index_band"] = ["NDVI", "NDBI", "MNDWI", "BSI"]
    logger.info("Computed spectral indices stack shape: %s", indices_stack.shape)
    return indices_stack
