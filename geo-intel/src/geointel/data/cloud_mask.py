"""
data/cloud_mask.py — SCL-based cloud/shadow masking for Sentinel-2 L2A.

The Sentinel-2 Scene Classification Layer (SCL) assigns a class to every
10 m pixel. We mask out cloud, shadow, snow, and saturated pixels before
compositing.

CRS: all operations are performed on the input array's native CRS/grid
(EPSG:32644 after stackstac loads and reprojects). No reprojection here.

Inputs / Outputs
----------------
All functions take and return xarray.DataArray objects with the dimensions
(time, y, x) or (y, x), float32, in EPSG:32644.

SCL class legend (Sentinel-2 L2A, Sen2Cor):
    0  No data
    1  Saturated / defective
    2  Dark area (cast shadow candidate)
    3  Cloud shadow
    4  Vegetation
    5  Bare soils
    6  Water
    7  Unclassified
    8  Cloud (medium probability)
    9  Cloud (high probability)
    10 Thin cirrus
    11 Snow / Ice

Default mask values (config: sentinel2.scl_mask_values):
    [0, 1, 2, 3, 8, 9, 10, 11]  → these pixels are INVALID

Why these classes?
    - 0,1: sensor artefacts
    - 2: dark-area pixels are flagged conservatively; in Himalayan terrain
          cast shadows are prevalent and contaminate spectral values
    - 3: cloud shadow — critical for urban/vegetation confusion
    - 8,9,10: clouds at various probability thresholds
    - 11: snow/ice — rare in Nov–Mar but present on ridge tops

Note: class 7 (unclassified) is NOT masked by default. In Dehradun fringe,
unclassified pixels are often construction dust or bare soil; masking them
would remove valid data. Monitor in the valid-pixel map.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from geointel.utils.logging import get_logger

logger = get_logger(__name__)

# Default SCL values treated as invalid (cloud/shadow/snow/saturated)
DEFAULT_MASK_VALUES: list[int] = [0, 1, 2, 3, 8, 9, 10, 11]


# ---------------------------------------------------------------------------
# Core masking functions
# ---------------------------------------------------------------------------

def build_valid_mask(
    scl: xr.DataArray,
    mask_values: list[int] | None = None,
) -> xr.DataArray:
    """
    Build a boolean valid-pixel mask from the SCL band.

    Parameters
    ----------
    scl : xr.DataArray
        SCL band array, shape (time, y, x) or (y, x).
        Integer dtype (uint8 or int16 expected).
        CRS: EPSG:32644 (after stackstac load).
    mask_values : list[int], optional
        SCL class integers to mark as INVALID (masked).
        Defaults to DEFAULT_MASK_VALUES.

    Returns
    -------
    xr.DataArray
        Boolean array, same shape as ``scl``.
        True  = VALID pixel (keep)
        False = INVALID pixel (cloud/shadow/snow/nodata)

    Raises
    ------
    ValueError
        If ``scl`` contains no spatial dimensions named 'x' and 'y'.
    """
    if mask_values is None:
        mask_values = DEFAULT_MASK_VALUES

    if "x" not in scl.dims or "y" not in scl.dims:
        raise ValueError(
            f"SCL DataArray must have 'x' and 'y' dimensions; got {scl.dims}"
        )

    valid = np.isfinite(scl)
    for val in mask_values:
        valid &= scl != val

    valid.name = "valid_mask"
    valid.attrs = {
        "description": "True = valid (not cloud/shadow/snow/nodata)",
        "masked_scl_values": mask_values,
        "crs": scl.attrs.get("crs", "EPSG:32644"),
    }
    logger.debug("Valid-pixel mask built lazily for shape=%s", valid.shape)
    return valid


def apply_valid_mask(
    data: xr.DataArray,
    valid_mask: xr.DataArray,
    nodata_value: float = float("nan"),
) -> xr.DataArray:
    """
    Apply a valid-pixel mask to a DataArray, setting invalid pixels to nodata.

    Parameters
    ----------
    data : xr.DataArray
        Array to mask, any shape that broadcasts with ``valid_mask``.
        Will be cast to float32 if not already.
    valid_mask : xr.DataArray
        Boolean array (True = keep). Must be broadcastable to ``data``.
    nodata_value : float
        Value written to invalid pixels. Default NaN (compatible with
        xarray median(skipna=True)).

    Returns
    -------
    xr.DataArray
        float32 array with invalid pixels set to ``nodata_value``.

    CRS Assumption
    --------------
    Spatial reference of ``data`` and ``valid_mask`` must match. This is
    guaranteed when both come from the same stackstac stack.
    """
    data_f32 = data.astype("float32")
    masked = data_f32.where(valid_mask, other=nodata_value)
    masked.attrs.update(data.attrs)
    masked.attrs["nodata"] = nodata_value
    return masked


def valid_pixel_fraction(
    valid_mask: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Compute per-pixel fraction of valid observations across the time dimension.

    Parameters
    ----------
    valid_mask : xr.DataArray
        Boolean mask (time, y, x). True = valid.
    time_dim : str
        Name of the time dimension. Default 'time'.

    Returns
    -------
    xr.DataArray
        Float array (y, x), values in [0, 1].
        0.0 = never valid; 1.0 = always valid across all scenes.

    Usage
    -----
    Use this to warn if any region has fewer than
    ``config.sentinel2.min_valid_pixel_fraction`` valid observations.
    """
    frac = valid_mask.mean(dim=time_dim).astype("float32")
    frac.attrs = {
        "description": "Fraction of time steps with valid (non-cloud) data",
        "range": "[0, 1]",
        "crs": valid_mask.attrs.get("crs", "EPSG:32644"),
    }
    return frac


def check_coverage(
    valid_frac: xr.DataArray,
    min_frac: float = 0.70,
    epoch_key: str = "unknown",
) -> bool:
    """
    Warn if mean valid-pixel fraction is below ``min_frac``.

    Parameters
    ----------
    valid_frac : xr.DataArray
        Per-pixel fraction of valid observations (from valid_pixel_fraction()).
    min_frac : float
        Minimum acceptable mean fraction. From config.sentinel2.min_valid_pixel_fraction.
    epoch_key : str
        Used in log messages.

    Returns
    -------
    bool
        True if coverage is acceptable (mean >= min_frac).
        False if below threshold (warning logged).
    """
    mean_frac = float(valid_frac.values.mean())
    min_local = float(valid_frac.values.min())

    logger.info(
        "Valid-pixel coverage [epoch=%s]: mean=%.1f%%  min=%.1f%%  threshold=%.1f%%",
        epoch_key,
        mean_frac * 100,
        min_local * 100,
        min_frac * 100,
    )

    if mean_frac < min_frac:
        logger.warning(
            "⚠  Low valid-pixel coverage for epoch=%s: %.1f%% < %.1f%% threshold. "
            "Consider raising max_cloud_cover in config.yaml or widening the date window.",
            epoch_key,
            mean_frac * 100,
            min_frac * 100,
        )
        return False

    if min_local < 0.30:
        logger.warning(
            "⚠  Some pixels in epoch=%s have < 30%% valid observations. "
            "Check the valid-pixel map for spatial gaps.",
            epoch_key,
        )

    return True


def mask_stack(
    stack: xr.DataArray,
    scl_band_name: str = "SCL",
    mask_values: list[int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convenience function: extract SCL from a full band stack, build valid mask,
    apply to all non-SCL bands, and return.

    Parameters
    ----------
    stack : xr.DataArray
        Full stackstac output with dimensions (time, band, y, x).
        Must contain a band named ``scl_band_name``.
    scl_band_name : str
        Name of the SCL band in the 'band' dimension. Default 'SCL'.
        Note: stackstac may lowercase band names; adjust if needed.
    mask_values : list[int], optional
        SCL classes to mask. Default: DEFAULT_MASK_VALUES.

    Returns
    -------
    (masked_stack, valid_frac)
        masked_stack : xr.DataArray (time, band, y, x) — float32, NaN where invalid
        valid_frac   : xr.DataArray (y, x)              — fraction valid across time

    CRS Assumption
    --------------
    Input stack CRS must be EPSG:32644. stackstac ensures this when loaded
    with epsg=32644. This function does NOT reproject.

    Notes
    -----
    - SCL is dropped from the returned masked_stack (it's categorical, not
      a spectral band for compositing).
    - stackstac may name the SCL band as 'SCL', 'scl', or use the asset key.
      Adjust scl_band_name if the band dimension label differs.
    """
    if mask_values is None:
        mask_values = DEFAULT_MASK_VALUES

    # ── Find SCL in the band dimension ─────────────────────────────────────
    band_dim = stack.coords.get("band")
    if band_dim is None:
        raise ValueError("Stack must have a 'band' coordinate dimension.")

    band_names = list(band_dim.values)
    # Try exact match first, then case-insensitive
    if scl_band_name not in band_names:
        lower_map = {b.lower(): b for b in band_names}
        scl_band_name = lower_map.get(scl_band_name.lower(), scl_band_name)
        if scl_band_name not in band_names:
            raise ValueError(
                f"SCL band '{scl_band_name}' not found in stack bands: {band_names}. "
                "Check the band name used by your STAC provider."
            )

    scl = stack.sel(band=scl_band_name)          # (time, y, x)
    spectral = stack.drop_sel(band=scl_band_name) # (time, band, y, x) without SCL

    valid_mask = build_valid_mask(scl, mask_values)           # (time, y, x)
    valid_mask = valid_mask & np.isfinite(spectral).all(dim="band")
    # Broadcast valid_mask over band dim for spectral stack
    masked_spectral = apply_valid_mask(
        spectral,
        valid_mask.expand_dims({"band": spectral.coords["band"]}, axis=1),
    )

    valid_frac = valid_pixel_fraction(valid_mask)  # (y, x)

    logger.info(
        "Masked stack: %d bands, %d time steps",
        len(spectral.coords["band"]),
        len(stack.coords["time"]),
    )

    return masked_spectral, valid_frac
