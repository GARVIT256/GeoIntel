"""
data/dem.py — Copernicus DEM 30 m download and preprocessing for GEO-INTEL.

Downloads the Copernicus Digital Elevation Model (GLO-30, 30 m resolution)
via the Microsoft Planetary Computer STAC API, clips to the AOI, derives
slope and aspect, and caches all products as COGs in EPSG:32644.

Why DEM?
--------
- Hill shadow is a known source of confusion (dark slopes misclassified as
  water or bare soil). DEM-derived slope and aspect can be added as RF
  features in Phase 4 to reduce this confusion.
- The DEM is for ANALYSIS ONLY in Phase 4+. In Phase 2 it is only downloaded
  and cached.

CRS handling
------------
- Copernicus DEM tiles are in EPSG:4326 (geographic coordinates).
- We reproject to EPSG:32644 at 30 m resolution (the DEM's native resolution;
  NOT resampled to 10 m — that would imply false precision).
- The 30 m DEM is stored separately from the 10 m Sentinel-2 composite.

UNVERIFIED items
----------------
- Planetary Computer collection name for Copernicus DEM:
  'cop-dem-glo-30' (verify at https://planetarycomputer.microsoft.com/catalog)
- STAC item structure and asset key for the elevation band.
- rasterio merge() for multi-tile DEMs.

CLI
---
    python -m geointel.data.dem --config config/config.yaml

Outputs
-------
    data/cache/dem/copernicus_dem30_32644.tif   — elevation (m), EPSG:32644, 30 m
    data/cache/dem/slope_32644.tif              — slope (degrees), EPSG:32644
    data/cache/dem/aspect_32644.tif             — aspect (degrees), EPSG:32644
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from geointel.utils.config import load_config
from geointel.utils.logging import get_logger

logger = get_logger(__name__)

PC_STAC_URL        = "https://planetarycomputer.microsoft.com/api/stac/v1"
COP_DEM_COLLECTION = "cop-dem-glo-30"   # UNVERIFIED: verify at PC catalogue
COMPUTE_CRS        = "EPSG:32644"
DISPLAY_CRS        = "EPSG:4326"
DEM_RESOLUTION_M   = 30                 # native DEM resolution; do NOT resample to 10 m


# ---------------------------------------------------------------------------
# DEM download
# ---------------------------------------------------------------------------

def _search_dem_items(bbox_4326: list[float]) -> list[Any]:
    """
    Search Planetary Computer for Copernicus DEM tiles covering the AOI.

    UNVERIFIED: collection name and query params.
    """
    import pystac_client       # type: ignore[import]
    import planetary_computer  # type: ignore[import]

    client = pystac_client.Client.open(PC_STAC_URL)
    search = client.search(
        collections=[COP_DEM_COLLECTION],
        bbox=bbox_4326,
        max_items=20,
    )
    items = list(search.items())
    logger.info("Found %d CopDEM tiles covering the AOI", len(items))

    signed = []
    for item in items:
        planetary_computer.sign_inplace(item)
        signed.append(item)
    return signed


def _load_dem_tile(item: Any, asset_key: str = "data") -> Any:
    """
    Load a DEM tile from a STAC item using rasterio.

    Parameters
    ----------
    item : pystac.Item
        Signed STAC item.
    asset_key : str
        Asset key for the elevation band. UNVERIFIED: may be 'data' or 'elevation'.

    Returns
    -------
    tuple (data_array, transform, crs)

    UNVERIFIED: exact asset key name for CopDEM. Check item.assets.keys().
    """
    import rasterio

    href = item.assets.get(asset_key)
    if href is None:
        # Try alternate keys
        for alt in ("elevation", "data", list(item.assets.keys())[0]):
            if alt in item.assets:
                href = item.assets[alt]
                break

    if href is None:
        raise ValueError(
            f"DEM asset not found in item {item.id}. "
            f"Available assets: {list(item.assets.keys())}. "
            "Update asset_key in dem.py."
        )

    url = href.href
    logger.debug("Loading DEM tile: %s", url)

    with rasterio.open(url) as src:
        data = src.read(1).astype("float32")
        data[data == src.nodata] = float("nan") if src.nodata else data
        return data, src.transform, str(src.crs)


def _derive_slope_aspect(
    elevation: np.ndarray,
    resolution_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees) from an elevation grid.

    Parameters
    ----------
    elevation : np.ndarray  (y, x), float32, in EPSG:32644 (metric)
    resolution_m : float
        Grid spacing in metres. Used for gradient computation.

    Returns
    -------
    (slope_deg, aspect_deg)
        slope_deg  : 0 (flat) to 90 (vertical cliff)
        aspect_deg : 0/360 (North) clockwise

    Notes
    -----
    Uses numpy gradient for finite differences. This is a standard first-order
    approximation. For production accuracy, use GDAL's gdaldem, but this is
    sufficient for the RF feature in Phase 4.
    """
    dz_dy, dz_dx = np.gradient(elevation, resolution_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype("float32")

    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) % 360).astype("float32")

    return slope_deg, aspect_deg


def resample_terrain_features(
    dem_path: Path,
    reference_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Derive 30 m terrain features then bilinearly align them to a reference grid.

    Aspect is converted to sine/cosine on its native 30 m grid before either
    component is bilinearly resampled. This avoids interpolating through the
    0/360 degree discontinuity. Elevation and slope are bilinear as well.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as dem, rasterio.open(reference_path) as ref:
        if dem.crs is None or ref.crs is None:
            raise ValueError("DEM and reference raster must both declare a CRS")
        if dem.crs != ref.crs:
            raise ValueError(f"DEM CRS {dem.crs} does not match reference CRS {ref.crs}")
        if abs(dem.transform.a) != DEM_RESOLUTION_M or abs(dem.transform.e) != DEM_RESOLUTION_M:
            raise ValueError(f"Expected a {DEM_RESOLUTION_M} m native DEM grid")
        elevation = dem.read(1).astype("float32")
        if dem.nodata is not None:
            elevation[elevation == dem.nodata] = np.nan
        slope, aspect = _derive_slope_aspect(elevation, DEM_RESOLUTION_M)
        aspect_rad = np.deg2rad(aspect)
        native = {
            "elevation_10m": elevation,
            "slope_10m": slope,
            "aspect_sin_10m": np.sin(aspect_rad).astype("float32"),
            "aspect_cos_10m": np.cos(aspect_rad).astype("float32"),
        }
        outputs: dict[str, Path] = {}
        for name, source in native.items():
            destination = np.full((ref.height, ref.width), np.nan, dtype="float32")
            reproject(
                source=source,
                destination=destination,
                src_transform=dem.transform,
                src_crs=dem.crs,
                src_nodata=np.nan,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            path = output_dir / f"{name}.tif"
            profile = ref.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan, compress="deflate")
            with rasterio.open(path, "w", **profile) as out:
                out.write(destination, 1)
                out.set_band_description(1, name)
            outputs[name] = path
    return outputs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_or_load_dem(
    cfg: dict[str, Any],
    force_refresh: bool = False,
) -> dict[str, Path]:
    """
    Cache-first DEM download, merge, reproject, and derive slope/aspect.

    Returns
    -------
    dict with keys:
        'elevation_path' — elevation COG in EPSG:32644
        'slope_path'     — slope COG in EPSG:32644
        'aspect_path'    — aspect COG in EPSG:32644

    CRS guarantee
    -------------
    All output rasters are in EPSG:32644 at 30 m resolution.
    The DEM is NOT resampled to 10 m to avoid implying false precision.
    """
    import rasterio
    import rasterio.merge
    import rasterio.warp
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    cache_dir: Path = cfg["paths"]["data_cache"] / "dem"
    cache_dir.mkdir(parents=True, exist_ok=True)

    elev_path   = cache_dir / "copernicus_dem30_32644.tif"
    slope_path  = cache_dir / "slope_32644.tif"
    aspect_path = cache_dir / "aspect_32644.tif"

    # ── Cache hit ──────────────────────────────────────────────────────────
    if elev_path.exists() and slope_path.exists() and not force_refresh:
        logger.info("DEM cache hit: %s", elev_path)
        return {
            "elevation_path": elev_path,
            "slope_path": slope_path,
            "aspect_path": aspect_path,
        }

    # ── STAC search and download ───────────────────────────────────────────
    bbox_4326: list[float] = cfg["aoi"]["bbox"]
    items = _search_dem_items(bbox_4326)

    if not items:
        raise RuntimeError(
            "No Copernicus DEM tiles found for AOI. "
            "Check network connection and PC STAC URL."
        )

    # ── Load and merge tiles with rasterio ────────────────────────────────
    logger.info("Loading %d DEM tile(s)...", len(items))
    src_files = []
    for item in items:
        href = item.assets.get("data") or item.assets.get("elevation") \
               or list(item.assets.values())[0]
        src_files.append(rasterio.open(href.href))

    if len(src_files) == 1:
        merged = src_files[0].read(1).astype("float32")
        merged_transform = src_files[0].transform
        src_crs = src_files[0].crs
    else:
        merged, merged_transform = rasterio.merge.merge(src_files)
        merged = merged[0].astype("float32")
        src_crs = src_files[0].crs

    for f in src_files:
        f.close()

    logger.info("Merged DEM shape: %s  CRS: %s", merged.shape, src_crs)

    # ── Reproject to EPSG:32644 at 30 m ───────────────────────────────────
    dst_crs = CRS.from_epsg(32644)
    # Compute output transform and shape at 30 m resolution
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs,
        merged.shape[1], merged.shape[0],
        left=bbox_4326[0], bottom=bbox_4326[1],
        right=bbox_4326[2], top=bbox_4326[3],
        resolution=DEM_RESOLUTION_M,
    )

    dst_array = np.full((dst_height, dst_width), np.nan, dtype="float32")
    rasterio.warp.reproject(
        source=merged,
        destination=dst_array,
        src_transform=merged_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=rasterio.enums.Resampling.bilinear,
    )

    logger.info(
        "DEM reprojected to EPSG:32644: shape=%s  resolution=%d m",
        dst_array.shape, DEM_RESOLUTION_M,
    )

    # Verify output CRS
    assert dst_crs.to_epsg() == 32644, f"DEM output CRS must be 32644, got {dst_crs}"

    def _write_cog(arr: np.ndarray, path: Path, description: str) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype="float32",
            crs=dst_crs,
            transform=dst_transform,
            nodata=float("nan"),
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        ) as dst:
            dst.write(arr, 1)
            dst.update_tags(1, description=description)
        logger.info("%s → %s", description, path)

    # ── Save elevation ────────────────────────────────────────────────────
    _write_cog(dst_array, elev_path, "Elevation (m) — Copernicus DEM GLO-30, EPSG:32644")

    # ── Derive and save slope / aspect ───────────────────────────────────
    slope_deg, aspect_deg = _derive_slope_aspect(dst_array, DEM_RESOLUTION_M)
    _write_cog(slope_deg,  slope_path,  "Slope (degrees), EPSG:32644")
    _write_cog(aspect_deg, aspect_path, "Aspect (degrees, 0=N), EPSG:32644")

    return {
        "elevation_path": elev_path,
        "slope_path":     slope_path,
        "aspect_path":    aspect_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default="config/config.yaml", show_default=True)
@click.option("--force-refresh", is_flag=True, default=False)
def main(config_path: str, force_refresh: bool) -> None:
    """Download Copernicus DEM 30m and derive slope/aspect COGs."""
    cfg = load_config(config_path)
    result = fetch_or_load_dem(cfg, force_refresh=force_refresh)
    for key, path in result.items():
        logger.info("%-20s → %s", key, path)


if __name__ == "__main__":
    main()
