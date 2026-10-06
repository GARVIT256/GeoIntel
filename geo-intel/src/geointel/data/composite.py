"""
data/composite.py — Sentinel-2 median compositing and COG output for GEO-INTEL.

Loads a set of STAC items, applies SCL cloud masking, computes a pixel-wise
median composite, and writes the result as a Cloud-Optimized GeoTIFF (COG).

    bbox_4326 = bounds_latlon or cfg["aoi"]["bbox"]
--------------
- Uses stackstac for loading (UNVERIFIED: API for stackstac 0.5.x; check docs
  shot. stackstac handles resampling internally via `resolution=10`.
- Median compositing: robust to remaining outliers after SCL masking (e.g.
  thin cloud residuals). skipna=True skips masked pixels.
- Output: one COG per epoch per "band group" (all 6 spectral bands + a
  valid-pixel-fraction layer as a separate COG).

CRS: All raster data is loaded and saved in EPSG:32644 (UTM 44N). The AOI bbox
passed to stackstac is in EPSG:4326 for the spatial query, but the output grid
is EPSG:32644.

UNVERIFIED items (verify against library docs before running):
  - stackstac.stack() parameter names: epsg, resolution, bounds_latlon, assets
  - stackstac may require dask; if not using dask, pass chunks=None (but then
    the entire time-stack loads into RAM — risky for > 100 scenes).
  - rioxarray write_nodata / to_raster COG options

CLI
---
    python -m geointel.data.composite --config config/config.yaml --epoch t1
    python -m geointel.data.composite --config config/config.yaml --epoch t2
                item.assets[asset_key].href,
Outputs
-------
    import stackstac  # type: ignore[import]
    data/cache/composites/t1/composite.tif          # 6 spectral bands, float32
    data/cache/composites/t1/valid_pixel_frac.tif   # valid fraction, float32

    data/cache/composites/t2/composite.tif
    data/cache/composites/t2/valid_pixel_frac.tif
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import xarray as xr

from geointel.data.cloud_mask import mask_stack
from geointel.data.stac_fetch import fetch_or_load, search_scenes
from geointel.utils.config import load_config
from geointel.utils.logging import get_logger
from geointel.utils.manifest import RunManifest

logger = get_logger(__name__)

# Band order in the output composite (determines band index 1..6 in GeoTIFF)
BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12"]

# STAC asset keys for SCL (may vary by provider)
SCL_ASSET_KEY = "SCL"
AWS_ASSET_KEYS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}


# ---------------------------------------------------------------------------
# Stack loading
# ---------------------------------------------------------------------------

def load_stack(
    items: list[Any],
    cfg: dict[str, Any],
    epoch_key: str,
    provider: str = "pc",
    bands: list[str] | None = None,
    bounds_latlon: list[float] | None = None,
    chunksize: tuple[int, int, int, int] = (1, 1, 256, 256),
) -> xr.DataArray:
    """
    Load Sentinel-2 scenes into a 4-D xarray stack (time, band, y, x).

    Parameters
    ----------
    items : list of pystac.Item
        Signed STAC items from stac_fetch.fetch_or_load().
    cfg : dict
        Loaded config.
    epoch_key : str
        'T1' or 't1' etc. Used for logging.

    Returns
    -------
    xr.DataArray
        Dimensions: (time, band, y, x)
        CRS: EPSG:32644  — stackstac reprojects to this CRS.
        Resolution: 10 m (20 m bands bilinearly resampled).
        Bands: BAND_ORDER + SCL (as the last band).
        dtype: float32 (spectral bands) / uint8 (SCL)

    Notes
    -----
    UNVERIFIED: stackstac 0.5.x API. Key parameters:
      - ``epsg``: target CRS EPSG code (integer)
      - ``resolution``: output resolution in CRS units (metres for UTM)
      - ``bounds_latlon``: spatial extent in EPSG:4326 [W, S, E, N]
      - ``assets``: list of asset keys to load
      - ``resampling``: rasterio.enums.Resampling per asset (dict)
      - ``xy_coords``: 'center' to get pixel-centre coordinates
    If this call fails, check stackstac docs at:
    https://stackstac.readthedocs.io/en/stable/api/main/stackstac.stack.html
    """
    bbox_4326 = bounds_latlon or cfg["aoi"]["bbox"]
    compute_epsg: int = int(
        cfg["crs"]["compute"].replace("EPSG:", "").replace("epsg:", "")
    )
    resolution_m: int = cfg["resolution"]["reference_m"]  # 10
    ordered_items = sorted(
        items,
        key=lambda item: item.properties.get("datetime", ""),
    )
    if provider == "pc":
        import planetary_computer

        for item in ordered_items:
            planetary_computer.sign_inplace(item)
    elif provider != "aws":
        raise ValueError("provider must be 'pc' or 'aws'")

    spectral_bands = bands or (
        list(cfg["sentinel2"]["bands_10m"])
        + list(cfg["sentinel2"]["bands_20m"])
    )
    unknown_bands = set(spectral_bands) - set(BAND_ORDER)
    if unknown_bands:
        raise ValueError(f"Unsupported Sentinel-2 bands: {sorted(unknown_bands)}")
    source_asset_keys = {
        band: band if band in ordered_items[0].assets else AWS_ASSET_KEYS[band]
        for band in spectral_bands
    }
    scl_asset_key = (
        SCL_ASSET_KEY if SCL_ASSET_KEY in ordered_items[0].assets else AWS_ASSET_KEYS[SCL_ASSET_KEY]
    )
    missing_assets = [
        key for key in [*source_asset_keys.values(), scl_asset_key]
        if any(key not in item.assets for item in ordered_items)
    ]
    if missing_assets:
        raise ValueError(f"Scenes are missing required assets: {sorted(set(missing_assets))}")

    gdal_options = {
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_CACHEMAX": 256,
        "GDAL_NUM_THREADS": "1",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "5000000",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    if provider == "aws":
        gdal_options["AWS_NO_SIGN_REQUEST"] = "YES"

    logger.info(
        "Loading stack [epoch=%s]: %d items, %d assets, %d m grid, EPSG:%d",
        epoch_key, len(items), len(spectral_bands) + 1, resolution_m, compute_epsg,
    )
    hrefs = {
        **source_asset_keys,
        SCL_ASSET_KEY: scl_asset_key,
    }
    for item in ordered_items[:3]:
        for canonical_band, asset_key in hrefs.items():
            logger.info(
                "Raster asset href provider=%s scene=%s band=%s href=%s",
                provider,
                item.id,
                canonical_band,
                item.assets[asset_key].href,
            )

    import stackstac  # type: ignore[import]
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError

    stack_options = {
        "epsg": compute_epsg,
        "resolution": resolution_m,
        "bounds_latlon": bbox_4326,
        "xy_coords": "center",
        "dtype": "float32",
        "fill_value": np.float32(np.nan),
        "rescale": False,
        "sortby_date": False,
        "chunksize": chunksize,
        "gdal_env": stackstac.LayeredEnv(always=gdal_options),
        "errors_as_nodata": (
            RasterioIOError("HTTP response code: 404"),
            RasterioIOError("Read failed.*"),
        ),
    }

    spectral_stack = stackstac.stack(
        ordered_items,
        assets=list(source_asset_keys.values()),
        resampling=Resampling.bilinear,
        **stack_options,
    )
    scl_stack = stackstac.stack(
        ordered_items,
        assets=[scl_asset_key],
        resampling=Resampling.nearest,
        **stack_options,
    )
    stack = xr.concat(
        [spectral_stack.reset_coords(drop=True), scl_stack.reset_coords(drop=True)],
        dim="band",
    )
    canonical_names = {asset: band for band, asset in source_asset_keys.items()}
    canonical_names[scl_asset_key] = SCL_ASSET_KEY
    stack = stack.assign_coords(
        band=[canonical_names[str(asset)] for asset in stack.coords["band"].values]
    )

    stack = normalize_l2a_reflectance(stack, ordered_items)

    logger.info(
        "Stack shape: %s  dtype=%s  bands=%s",
        stack.shape, stack.dtype,
        list(stack.coords["band"].values),
    )
    return stack


def normalize_l2a_reflectance(
    stack: xr.DataArray,
    items: list[Any],
) -> xr.DataArray:
    """Apply scale and the BOA offset once, honoring Earth Search's COG flag."""
    if "time" not in stack.dims or "band" not in stack.dims:
        raise ValueError("Stack must have 'time' and 'band' dimensions")
    if stack.sizes["time"] != len(items):
        raise ValueError("Stack time dimension must match the supplied STAC items")

    band_values = list(stack.coords["band"].values)
    normalized_bands = []
    for band in band_values:
        band_data = stack.sel(band=band)
        if str(band).upper() != SCL_ASSET_KEY:
            scene_arrays = []
            for time_index, item in enumerate(items):
                scene = band_data.isel(time=time_index, drop=True).astype("float32")
                assets = getattr(item, "assets", {})
                if str(band) in assets:
                    asset = assets[str(band)]
                elif AWS_ASSET_KEYS.get(str(band)) in assets:
                    asset = assets[AWS_ASSET_KEYS[str(band)]]
                else:
                    asset = None
                scale, offset = reflectance_scale_offset(item, str(band))
                scene = scene * scale + offset
                scene_arrays.append(
                    scene.expand_dims(time=[stack.coords["time"].values[time_index]])
                )
            band_data = xr.concat(scene_arrays, dim="time").astype("float32")
        normalized_bands.append(band_data.expand_dims(band=[band]))

    normalized = xr.concat(normalized_bands, dim="band")
    normalized.attrs.update(stack.attrs)
    normalized.attrs["reflectance_scale"] = 0.0001
    normalized.attrs["boa_offset_application"] = (
        "asset raster:bands offset once; PB4+ post-2022 fallback when offset is absent/zero; "
        "suppress -0.1 for flagged Earth Search Sentinel-2 COGs observed already harmonized"
    )
    return normalized


def reflectance_scale_offset(item: Any, band: str) -> tuple[float, float]:
    """Return the DN scale/offset transform applied to one item's reflectance band.

    Earth Search Sentinel-2 items can retain ``raster:bands.offset=-0.1`` even
    when their COG pixels are already harmonized. Project diagnostics include
    such an example with the flag false, so for items carrying the
    Earth-Search-specific property the declared -0.1 is suppressed. Generic
    assets without that property continue to use declared STAC offsets.
    """
    assets = getattr(item, "assets", {})
    asset = assets.get(band) or assets.get(AWS_ASSET_KEYS.get(band, ""))
    raster_bands = asset.extra_fields.get("raster:bands", []) if asset is not None else []
    metadata = raster_bands[0] if raster_bands else {}
    scale_value = metadata.get("scale")
    offset_value = metadata.get("offset")
    scale = float(0.0001 if scale_value is None else scale_value)
    offset = float(0.0 if offset_value is None else offset_value)
    boa_applied = item.properties.get("earthsearch:boa_offset_applied")
    try:
        baseline = float(item.properties.get("s2:processing_baseline", 0.0))
    except (TypeError, ValueError):
        baseline = 0.0
    is_earth_search_item = "earthsearch:boa_offset_applied" in item.properties
    # Earth Search Sentinel-2 COGs have exhibited mismatches between this flag,
    # the declared -0.1 raster offset, and sampled pixels. For flagged Earth
    # Search items the tested COG pixels are already BOA-harmonized, including
    # examples carrying a false flag; suppress that declared -0.1 to avoid a
    # second correction. This narrow exception does not change generic STAC.
    if is_earth_search_item and np.isclose(offset, -0.1):
        offset = 0.0
    else:
        acquired = str(
            item.properties.get("datetime") or item.properties.get("start_datetime") or ""
        )[:10]
        if (
            not (is_earth_search_item and boa_applied is True)
            and baseline >= 4.0
            and acquired >= "2022-01-25"
            and np.isclose(offset, 0.0)
        ):
            # PB4 introduced the -1000 DN L2A BOA_ADD_OFFSET on acquisitions
            # from 2022-01-25. Historical archive products later reprocessed
            # with PB05 are already radiometrically harmonized; do not infer a
            # second offset for those pre-2022 acquisition dates. Some catalog
            # providers omit raster:bands.offset and the Earth Search-specific
            # flag, so the acquisition date + baseline provide the fallback.
            # The explicit true flag still prevents duplicate application.
            offset = -0.1
    return scale, offset


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def compute_median_composite(
    masked_stack: xr.DataArray,
) -> xr.DataArray:
    """
    Compute pixel-wise median across the time dimension.

    Parameters
    ----------
    masked_stack : xr.DataArray
        Float32 stack (time, band, y, x) with NaN at cloud/shadow pixels.
        CRS: EPSG:32644.

    Returns
    -------
    xr.DataArray
        Float32 array (band, y, x). Each pixel is the median of all valid
        (non-NaN) observations at that location across the season.
        Pixels with ZERO valid observations remain NaN.

    Why median?
    -----------
    Median is robust to outliers (residual thin cloud, haze, wind-blown dust)
    in a way that mean is not. For a typical dry-season window of 50–100 Sentinel-2
    passes over Dehradun, median converges to a stable spectral signature.
    """
    logger.info("Computing median composite (skipna=True)...")
    composite = masked_stack.median(dim="time", skipna=True).astype("float32")
    composite.attrs.update(masked_stack.attrs)
    composite.attrs["composite_method"] = "median"
    composite.attrs["crs"] = "EPSG:32644"
    logger.info("Composite shape: %s", composite.shape)
    return composite


def compute_month_balanced_composite(masked_stack: xr.DataArray) -> xr.DataArray:
    """Median each calendar month, then give every represented month equal weight.

    Callers must first restrict both epochs to the same set of calendar months.
    This prevents a month with more acquisitions from contributing more layers
    to the final epoch composite.
    """
    if "time" not in masked_stack.dims or "time" not in masked_stack.coords:
        raise ValueError("Month-balanced composites require a datetime 'time' coordinate")
    monthly = masked_stack.groupby("time.month").median(dim="time", skipna=True)
    return monthly.median(dim="month", skipna=True)


# ---------------------------------------------------------------------------
# COG output
# ---------------------------------------------------------------------------

def _ensure_cog_options() -> dict[str, Any]:
    """
    Return rasterio profile options for COG output.
    These options instruct rasterio to write a Cloud-Optimized GeoTIFF.
    """
    return {
        "driver": "GTiff",
        "compress": "deflate",
        "predictor": 2,        # horizontal differencing (good for continuous data)
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "interleave": "band",
        "BIGTIFF": "IF_SAFER",
    }


def save_as_cog(
    da: xr.DataArray,
    output_path: Path,
    nodata: float = float("nan"),
    crs: str = "EPSG:32644",
) -> None:
    """
    Save an xr.DataArray as a Cloud-Optimized GeoTIFF using rioxarray.

    Parameters
    ----------
    da : xr.DataArray
        Array with spatial dimensions (y, x) or (band, y, x).
        CRS must be EPSG:32644 (set via rio.write_crs).
    output_path : Path
        Destination .tif file. Parent directory is created if needed.
    nodata : float
        Nodata value written to the GeoTIFF metadata. Default NaN.
    crs : str
        CRS string. Default 'EPSG:32644'. Written to the GeoTIFF.

        The data are staged as a tiled GeoTIFF and then converted with GDAL's COG
        driver so outputs have a valid cloud-optimized layout.
    """
    import rioxarray  # noqa: F401 — registers .rio accessor

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set spatial reference
    da_out = da.rio.write_crs(crs, inplace=False)
    da_out = da_out.rio.write_nodata(nodata, inplace=False)

    import rasterio
    from rasterio.shutil import copy as raster_copy

    staged_path = output_path.with_name(f"{output_path.stem}.staged.tif")
    da_out.rio.to_raster(
        str(staged_path),
        driver="GTiff",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(staged_path) as source:
        raster_copy(
            source,
            output_path,
            driver="COG",
            compress="DEFLATE",
            blocksize=256,
            overview_resampling="AVERAGE",
        )
    staged_path.unlink(missing_ok=True)
    logger.info("COG written: %s  (shape=%s)", output_path, da_out.shape)


def write_chunked_composites(
    composite: xr.DataArray,
    valid_frac: xr.DataArray,
    composite_path: Path,
    valid_frac_path: Path,
    partial_dir: Path,
    chunk_size: int = 256,
    max_attempts: int = 3,
    num_workers: int = 2,
) -> dict[str, Any]:
    """Retry, persist, and mosaic spatial chunks without holding the full raster in RAM."""
    import dask
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.shutil import copy as raster_copy
    from rasterio.windows import Window

    height = composite.sizes["y"]
    width = composite.sizes["x"]
    band_count = composite.sizes["band"]
    crs = composite.rio.crs or "EPSG:32644"
    transform = composite.rio.transform(recalc=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    working_composite = composite_path.with_name(f"{composite_path.stem}.working.tif")
    working_fraction = valid_frac_path.with_name(f"{valid_frac_path.stem}.working.tif")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": band_count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float("nan"),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    fraction_profile = {**profile, "count": 1}
    sums = np.zeros(band_count, dtype=np.float64)
    counts = np.zeros(band_count, dtype=np.int64)
    minima = np.full(band_count, np.inf, dtype=np.float64)
    maxima = np.full(band_count, -np.inf, dtype=np.float64)
    fraction_sum = 0.0
    fraction_count = 0
    fraction_min = 1.0
    fraction_max = 0.0
    failed_chunks: list[str] = []

    with rasterio.open(working_composite, "w", **profile) as composite_dst, rasterio.open(
        working_fraction, "w", **fraction_profile
    ) as fraction_dst:
        for row in range(0, height, chunk_size):
            for col in range(0, width, chunk_size):
                row_stop = min(row + chunk_size, height)
                col_stop = min(col + chunk_size, width)
                window = Window(col, row, col_stop - col, row_stop - row)
                composite_chunk = composite.isel(y=slice(row, row_stop), x=slice(col, col_stop))
                fraction_chunk = valid_frac.isel(y=slice(row, row_stop), x=slice(col, col_stop))
                chunk_key = f"r{row:05d}_c{col:05d}"

                values = None
                fraction_values = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        computed, computed_fraction = dask.compute(
                            composite_chunk,
                            fraction_chunk,
                            scheduler="threads",
                            num_workers=num_workers,
                        )
                        values = np.asarray(computed.values, dtype=np.float32)
                        fraction_values = np.asarray(computed_fraction.values, dtype=np.float32)
                        break
                    except (RasterioIOError, RuntimeError) as exc:
                        logger.warning(
                            "Chunk read failed epoch=%s chunk=%s attempt=%d/%d: %s",
                            composite_path.parent.name,
                            chunk_key,
                            attempt,
                            max_attempts,
                            exc,
                        )

                if values is None or fraction_values is None:
                    failed_chunks.append(chunk_key)
                    values = np.full(
                        (band_count, row_stop - row, col_stop - col),
                        np.nan,
                        dtype=np.float32,
                    )
                    fraction_values = np.zeros(
                        (row_stop - row, col_stop - col),
                        dtype=np.float32,
                    )

                part_composite = xr.DataArray(
                    values,
                    dims=["band", "y", "x"],
                    coords={
                        "band": composite.coords["band"],
                        "y": composite.coords["y"].isel(y=slice(row, row_stop)),
                        "x": composite.coords["x"].isel(x=slice(col, col_stop)),
                    },
                    attrs=composite.attrs,
                )
                part_fraction = xr.DataArray(
                    fraction_values,
                    dims=["y", "x"],
                    coords={
                        "y": valid_frac.coords["y"].isel(y=slice(row, row_stop)),
                        "x": valid_frac.coords["x"].isel(x=slice(col, col_stop)),
                    },
                    attrs=valid_frac.attrs,
                )
                save_as_cog(part_composite, partial_dir / f"{chunk_key}_composite.tif")
                save_as_cog(part_fraction, partial_dir / f"{chunk_key}_valid_frac.tif")
                composite_dst.write(values, window=window)
                fraction_dst.write(fraction_values, 1, window=window)

                for band_index in range(band_count):
                    finite_values = values[band_index][np.isfinite(values[band_index])]
                    if finite_values.size:
                        sums[band_index] += float(finite_values.sum(dtype=np.float64))
                        counts[band_index] += finite_values.size
                        minima[band_index] = min(minima[band_index], float(finite_values.min()))
                        maxima[band_index] = max(maxima[band_index], float(finite_values.max()))
                fraction_sum += float(fraction_values.sum(dtype=np.float64))
                fraction_count += fraction_values.size
                fraction_min = min(fraction_min, float(fraction_values.min()))
                fraction_max = max(fraction_max, float(fraction_values.max()))

    for staged_path, output_path in (
        (working_composite, composite_path),
        (working_fraction, valid_frac_path),
    ):
        with rasterio.open(staged_path) as source:
            raster_copy(
                source,
                output_path,
                driver="COG",
                compress="DEFLATE",
                blocksize=256,
                overview_resampling="AVERAGE",
            )
        staged_path.unlink(missing_ok=True)

    return {
        "band_stats": {
            str(band): {
                "min": float(minima[index]) if counts[index] else None,
                "mean": float(sums[index] / counts[index]) if counts[index] else None,
                "max": float(maxima[index]) if counts[index] else None,
                "valid_pixels": int(counts[index]),
            }
            for index, band in enumerate(composite.coords["band"].values)
        },
        "valid_fraction_mean": fraction_sum / fraction_count if fraction_count else 0.0,
        "valid_fraction_min": fraction_min,
        "valid_fraction_max": fraction_max,
        "chunk_count": int(np.ceil(height / chunk_size) * np.ceil(width / chunk_size)),
        "failed_chunks": failed_chunks,
        "partial_cogs_dir": str(partial_dir),
    }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def build_composite_for_epoch(
    cfg: dict[str, Any],
    epoch_key: str,
    manifest: RunManifest | None = None,
    force_refresh: bool = False,
    provider: str = "pc",
    items: list[Any] | None = None,
    matched_months: list[int] | None = None,
    bands: list[str] | None = None,
    chunk_size: int = 256,
    workers: int = 2,
) -> dict[str, Any]:
    """
    Full pipeline for one epoch: fetch → mask → composite → save COG.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    epoch_key : str
        't1' or 't2'.
    manifest : RunManifest, optional
        If provided, records scene IDs, timings, and valid-pixel stats.
    force_refresh : bool
        If True, recompute even if COG exists in cache.

    Returns
    -------
    dict with keys:
        'composite_path'   — Path to 6-band COG
        'valid_frac_path'  — Path to valid-pixel fraction COG

    CRS guarantee
    -------------
    All output files are in EPSG:32644 at 10 m resolution.
    """
    epoch_key = epoch_key.lower()
    cache_dir: Path = cfg["paths"]["data_cache"] / "composites" / epoch_key
    composite_path = cache_dir / "composite.tif"
    valid_frac_path = cache_dir / "valid_pixel_frac.tif"

    # ── Cache hit ──────────────────────────────────────────────────────────
    if composite_path.exists() and valid_frac_path.exists() and not force_refresh:
        logger.info(
            "Composite cache hit for epoch=%s: %s", epoch_key, composite_path
        )
        return {
            "composite_path": composite_path,
            "valid_frac_path": valid_frac_path,
        }

    # ── Fetch scenes ───────────────────────────────────────────────────────
    if manifest:
        manifest.start_timer(f"fetch_{epoch_key}")

    if items is None:
        items = fetch_or_load(
            cfg,
            epoch_key,
            provider=provider,
            force_refresh=force_refresh,
        )
    if cfg.get("composite", {}).get("month_balanced", False):
        if not matched_months:
            raise ValueError("Month-balanced mode requires non-empty matched_months from both epochs")
        items = [
            item for item in items
            if datetime.fromisoformat(
                str(item.properties.get("datetime") or item.properties.get("start_datetime"))
                .replace("Z", "+00:00")
            ).month in matched_months
        ]
        if not items:
            raise ValueError(f"No scenes remain for {epoch_key} in matched months {matched_months}")

    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for epoch={epoch_key}. "
            "Run stac_fetch.py or check network / config."
        )

    if manifest:
        manifest.stop_timer(f"fetch_{epoch_key}")
        manifest.add_scene_ids(epoch_key, [i.id for i in items])
        manifest.add_scene_dates(
            epoch_key,
            [str(i.properties.get("datetime")) for i in items],
        )

    # ── Load stack ─────────────────────────────────────────────────────────
    if manifest:
        manifest.start_timer(f"load_stack_{epoch_key}")

    logger.info("Loading raster stack for epoch=%s (%d scenes)...", epoch_key, len(items))
    selected_bands = bands or (
        list(cfg["sentinel2"]["bands_10m"])
        + list(cfg["sentinel2"]["bands_20m"])
    )
    stack = load_stack(
        items,
        cfg,
        epoch_key,
        provider=provider,
        bands=selected_bands,
        chunksize=(1, 1, chunk_size, chunk_size),
    )

    if manifest:
        manifest.stop_timer(f"load_stack_{epoch_key}")

    # ── Cloud masking ──────────────────────────────────────────────────────
    if manifest:
        manifest.start_timer(f"cloud_mask_{epoch_key}")

    scl_band = cfg["sentinel2"]["scl_band"]
    mask_values: list[int] = cfg["sentinel2"]["scl_mask_values"]
    masked_stack, valid_frac = mask_stack(stack, scl_band_name=scl_band, mask_values=mask_values)

    if manifest:
        manifest.stop_timer(f"cloud_mask_{epoch_key}")

    # ── Select spectral bands only (in BAND_ORDER) ─────────────────────────
    available_bands = list(masked_stack.coords["band"].values)
    ordered = [b for b in selected_bands if b in available_bands]
    if len(ordered) < len(selected_bands):
        missing = set(selected_bands) - set(ordered)
        logger.warning("Missing bands from stack: %s", missing)
    spectral = masked_stack.sel(band=ordered)

    # ── Composite ──────────────────────────────────────────────────────────
    if manifest:
        manifest.start_timer(f"composite_{epoch_key}")

    if cfg.get("composite", {}).get("month_balanced", False):
        composite = compute_month_balanced_composite(spectral)
        if manifest:
            manifest._data.setdefault("compositing", {})[epoch_key] = {
                "strategy": "monthly median followed by equal-month median",
                "matched_calendar_months": matched_months,
            }
    else:
        composite = compute_median_composite(spectral)

    if manifest:
        manifest.stop_timer(f"composite_{epoch_key}")

    min_frac: float = cfg["sentinel2"]["min_valid_pixel_fraction"]
    partial_dir = cache_dir / "partials" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    logger.info(
        "Writing chunked COGs [epoch=%s]: chunk=%d px, workers=%d, retries=3",
        epoch_key,
        chunk_size,
        workers,
    )
    stats = write_chunked_composites(
        composite,
        valid_frac,
        composite_path,
        valid_frac_path,
        partial_dir,
        chunk_size=chunk_size,
        max_attempts=3,
        num_workers=workers,
    )
    coverage_ok = stats["valid_fraction_mean"] >= min_frac
    logger.info(
        "Valid-pixel coverage [epoch=%s]: mean=%.1f%% min=%.1f%% max=%.1f%% threshold=%.1f%%",
        epoch_key,
        stats["valid_fraction_mean"] * 100,
        stats["valid_fraction_min"] * 100,
        stats["valid_fraction_max"] * 100,
        min_frac * 100,
    )
    if not coverage_ok:
        logger.warning("Low valid-pixel coverage for epoch=%s", epoch_key)

    if manifest:
        manifest.add_valid_pixel_stat(epoch_key, {
            "mean_frac": stats["valid_fraction_mean"],
            "min_frac": stats["valid_fraction_min"],
            "max_frac": stats["valid_fraction_max"],
            "n_scenes": len(items),
            "coverage_ok": coverage_ok,
            "failed_chunks": len(stats["failed_chunks"]),
        })
        manifest.set_result(f"composite_stats_{epoch_key}", stats["band_stats"])

    logger.info(
        "Epoch %s composite complete → %s; failed chunks=%d",
        epoch_key,
        composite_path,
        len(stats["failed_chunks"]),
    )
    return {
        "composite_path": composite_path,
        "valid_frac_path": valid_frac_path,
        "stats": stats,
    }


def run_smoke_test(cfg: dict[str, Any], provider: str = "aws") -> dict[str, Any]:
    """Run a real three-scene, one-band smoke test over a 5 x 5 km window."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as transform_geometry

    smoke_cfg = deepcopy(cfg)
    smoke_cfg["epochs"]["t1"] = {
        "label": "2018-11 smoke test",
        "start": "2018-11-01",
        "end": "2018-11-30",
    }
    center_lon, center_lat = 78.04099, 30.320742
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    center_x, center_y = to_utm.transform(center_lon, center_lat)
    smoke_window_utm = box(
        center_x - 2500,
        center_y - 2500,
        center_x + 2500,
        center_y + 2500,
    )
    to_wgs84 = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    smoke_bbox = list(transform_geometry(to_wgs84.transform, smoke_window_utm).bounds)
    smoke_cfg["aoi"]["bbox"] = smoke_bbox

    items = search_scenes(smoke_cfg, "t1", provider=provider)
    if len(items) < 3:
        raise RuntimeError(f"Smoke test requires 3 scenes; provider returned {len(items)}")
    selected_items = items[:3]
    logger.info(
        "Smoke test scenes (%s): %s",
        provider,
        [item.id for item in selected_items],
    )

    stack = load_stack(
        selected_items,
        smoke_cfg,
        "t1-smoke",
        provider=provider,
        bands=["B04"],
        bounds_latlon=smoke_bbox,
        chunksize=(1, 1, 128, 128),
    )
    masked_stack, valid_frac = mask_stack(
        stack,
        scl_band_name=smoke_cfg["sentinel2"]["scl_band"],
        mask_values=smoke_cfg["sentinel2"]["scl_mask_values"],
    )
    composite = masked_stack.sel(band="B04").median(dim="time", skipna=True)

    import dask

    composite, valid_frac = dask.compute(
        composite,
        valid_frac,
        scheduler="threads",
        num_workers=2,
    )
    values = composite.values
    valid_values = values[np.isfinite(values)]
    if valid_values.size == 0:
        raise RuntimeError("Smoke test produced no valid B04 composite pixels")

    output_dir = cfg["paths"]["data_cache"] / "smoke_test"
    composite_path = output_dir / f"{provider}_t1_nov2018_b04_5km.tif"
    valid_frac_path = output_dir / f"{provider}_t1_nov2018_valid_frac_5km.tif"
    save_as_cog(composite, composite_path)
    save_as_cog(valid_frac, valid_frac_path)

    result = {
        "provider": provider,
        "scene_ids": [item.id for item in selected_items],
        "bands": ["B04"],
        "window_km": [5, 5],
        "shape_yx": list(composite.shape),
        "valid_observation_percent_mean": float(valid_frac.mean().item() * 100),
        "composite_min_reflectance": float(valid_values.min()),
        "composite_mean_reflectance": float(valid_values.mean()),
        "composite_max_reflectance": float(valid_values.max()),
        "composite_path": str(composite_path),
        "valid_fraction_path": str(valid_frac_path),
    }
    logger.info("Smoke test result: %s", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default="config/config.yaml", show_default=True)
@click.option("--epoch", "epoch_key", default="both",
              type=click.Choice(["t1", "t2", "both"]), show_default=True)
@click.option("--provider", type=click.Choice(["pc", "aws"]), default="pc", show_default=True)
@click.option("--force-refresh", is_flag=True, default=False)
@click.option("--smoke-test", is_flag=True, default=False, help="Run 3-scene, 5 km, B04 smoke test")
def main(
    config_path: str,
    epoch_key: str,
    provider: str,
    force_refresh: bool,
    smoke_test: bool,
) -> None:
    """Build Sentinel-2 median composites and save as COGs."""
    cfg = load_config(config_path)
    if smoke_test:
        run_smoke_test(cfg, provider=provider)
        return
    manifest = RunManifest(cfg)

    epochs = ["t1", "t2"] if epoch_key == "both" else [epoch_key]
    results = {}

    for ek in epochs:
        paths = build_composite_for_epoch(
            cfg,
            ek,
            manifest=manifest,
            force_refresh=force_refresh,
            provider=provider,
        )
        results[ek] = paths
        logger.info("  composite  → %s", paths["composite_path"])
        logger.info("  valid_frac → %s", paths["valid_frac_path"])
        logger.info("  stats      → %s", paths["stats"])

    saved = manifest.save()
    logger.info("Manifest: %s", saved)


if __name__ == "__main__":
    main()
