"""
data/boundary.py — Administrative boundary download and cache for GEO-INTEL.

Downloads the Dehradun district boundary from the Overpass API (OpenStreetMap)
and caches it as a GeoPackage. Falls back to a rectangular AOI polygon if
the Overpass query fails or is unavailable offline.

CRS handling
------------
- Downloaded geometry is in EPSG:4326 (OSM coordinates are always WGS 84).
- Saved as EPSG:4326 for display and EPSG:32644 for compute.
- Both CRS versions are written to the cache.

UNVERIFIED: Overpass API endpoint stability and the OSM admin_level for
Dehradun district. In India, OSM typically uses:
  admin_level=4 → State (Uttarakhand)
  admin_level=5 → Division
  admin_level=6 → District (Dehradun District)
  admin_level=8 → City/Town
Verify by checking https://overpass-turbo.eu with the query below.

CLI
---
    python -m geointel.data.boundary --config config/config.yaml

Outputs
-------
    data/cache/boundary/dehradun_district_4326.gpkg  — WGS 84
    data/cache/boundary/dehradun_district_32644.gpkg — UTM 44N (compute CRS)
    data/cache/boundary/aoi_rectangle_4326.gpkg      — fallback: AOI bbox
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import httpx
from shapely.geometry import shape

from geointel.utils.config import load_config
from geointel.utils.logging import get_logger

logger = get_logger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass query: Dehradun district boundary (admin_level=6)
# UNVERIFIED: confirm admin_level and relation ID at overpass-turbo.eu
OVERPASS_QUERY_DEHRADUN = """
[out:json][timeout:60];
relation["name:en"="Dehradun district"]["admin_level"="6"];
(._;>;);
out geom;
"""

COMPUTE_CRS = "EPSG:32644"
DISPLAY_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _overpass_to_gdf(response_json: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Parse an Overpass API GeoJSON-style response into a GeoDataFrame.

    UNVERIFIED: exact structure of Overpass 'out geom' format.
    This parsing assumes features are in 'elements' with type 'relation'
    and geometry in 'members'. A safer alternative is to use osmnx:
        import osmnx as ox
        gdf = ox.geocode_to_gdf("Dehradun District, Uttarakhand, India")
    But osmnx is not in the approved stack. Using direct parse instead.
    """
    from shapely.geometry import mapping
    import shapely.ops as ops

    elements = response_json.get("elements", [])
    logger.debug("Overpass returned %d elements", len(elements))

    # Collect way geometries
    ways = [el for el in elements if el.get("type") == "way" and "geometry" in el]
    if not ways:
        logger.warning("No way geometries in Overpass response; will use AOI bbox.")
        return gpd.GeoDataFrame()

    from shapely.geometry import LineString, MultiPolygon
    import shapely.ops

    lines = []
    for way in ways:
        coords = [(pt["lon"], pt["lat"]) for pt in way["geometry"]]
        if len(coords) >= 2:
            lines.append(LineString(coords))

    if not lines:
        return gpd.GeoDataFrame()

    # Polygonise all way lines
    merged = shapely.ops.unary_union(lines)
    try:
        poly = shapely.ops.polygonize(merged)
        geoms = list(poly)
    except Exception as exc:
        logger.warning("Polygonise failed (%s); using convex hull.", exc)
        geoms = [merged.convex_hull]

    if not geoms:
        logger.warning("Could not polygonise boundary; will use AOI bbox.")
        return gpd.GeoDataFrame()

    union_geom = shapely.ops.unary_union(geoms)
    gdf = gpd.GeoDataFrame(
        [{"name": "Dehradun District", "source": "OpenStreetMap/Overpass"}],
        geometry=[union_geom],
        crs=DISPLAY_CRS,
    )
    return gdf


def download_boundary(timeout_s: int = 60) -> gpd.GeoDataFrame | None:
    """
    Query the Overpass API for the Dehradun district boundary.

    Returns
    -------
    GeoDataFrame in EPSG:4326, or None if download fails.
    """
    logger.info("Querying Overpass API for Dehradun district boundary...")
    try:
        resp = httpx.post(
            OVERPASS_URL,
            data={"data": OVERPASS_QUERY_DEHRADUN},
            timeout=timeout_s,
            headers={"User-Agent": "geo-intel-research/0.1"},
        )
        resp.raise_for_status()
        data = resp.json()
        gdf = _overpass_to_gdf(data)
        if gdf.empty:
            return None
        logger.info("Boundary downloaded: %d geometry(ies)", len(gdf))
        return gdf
    except Exception as exc:
        logger.warning("Overpass download failed: %s. Will use AOI bbox fallback.", exc)
        return None


def aoi_bbox_gdf(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame with the rectangular AOI bounding box.
    Used as a fallback when the Overpass boundary download fails.

    CRS: EPSG:4326
    """
    from shapely.geometry import box

    bbox: list[float] = cfg["aoi"]["bbox"]   # [W, S, E, N]
    geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
    gdf = gpd.GeoDataFrame(
        [{"name": cfg["aoi"]["name"], "source": "config_bbox_fallback"}],
        geometry=[geom],
        crs=DISPLAY_CRS,
    )
    logger.info("Using AOI bounding box as boundary fallback: %s", bbox)
    return gdf


# ---------------------------------------------------------------------------
# Cache and main function
# ---------------------------------------------------------------------------

def fetch_or_load_boundary(
    cfg: dict[str, Any],
    force_refresh: bool = False,
) -> dict[str, Path]:
    """
    Cache-first boundary download.

    Returns
    -------
    dict with keys:
        'boundary_4326'  — Path to GPKG in EPSG:4326
        'boundary_32644' — Path to GPKG in EPSG:32644
        'source'         — 'overpass' or 'aoi_bbox_fallback'

    CRS guarantee
    -------------
    - boundary_4326  → EPSG:4326
    - boundary_32644 → EPSG:32644  (all area/distance computations use this)
    """
    cache_dir: Path = cfg["paths"]["data_cache"] / "boundary"
    path_4326   = cache_dir / "dehradun_district_4326.gpkg"
    path_32644  = cache_dir / "dehradun_district_32644.gpkg"
    aoi_path    = cache_dir / "aoi_rectangle_4326.gpkg"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Always save the AOI rectangle (cheap, useful for spatial filter)
    if not aoi_path.exists() or force_refresh:
        aoi_gdf = aoi_bbox_gdf(cfg)
        aoi_gdf.to_file(aoi_path, driver="GPKG")
        logger.info("AOI bbox GPKG saved: %s", aoi_path)

    # Cache hit check
    if path_4326.exists() and path_32644.exists() and not force_refresh:
        logger.info("Boundary cache hit: %s", path_4326)
        return {"boundary_4326": path_4326, "boundary_32644": path_32644, "source": "cache"}

    # Try Overpass download
    gdf = download_boundary()
    source = "overpass"

    if gdf is None or gdf.empty:
        logger.warning("Using AOI bbox as district boundary (Overpass unavailable).")
        gdf = aoi_bbox_gdf(cfg)
        source = "aoi_bbox_fallback"
        path_4326  = cache_dir / "aoi_rectangle_4326.gpkg"
        path_32644 = cache_dir / "aoi_rectangle_32644.gpkg"

    # Save EPSG:4326
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326, \
        f"Boundary CRS must be EPSG:4326 at this point, got {gdf.crs}"
    gdf.to_file(path_4326, driver="GPKG")
    logger.info("Boundary (4326) → %s", path_4326)

    # Reproject to EPSG:32644
    gdf_32644 = gdf.to_crs(COMPUTE_CRS)
    assert gdf_32644.crs is not None and gdf_32644.crs.to_epsg() == 32644, \
        f"After reprojection, CRS must be EPSG:32644, got {gdf_32644.crs}"
    gdf_32644.to_file(path_32644, driver="GPKG")
    logger.info("Boundary (32644) → %s", path_32644)

    return {"boundary_4326": path_4326, "boundary_32644": path_32644, "source": source}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default="config/config.yaml", show_default=True)
@click.option("--force-refresh", is_flag=True, default=False)
def main(config_path: str, force_refresh: bool) -> None:
    """Download and cache the Dehradun district boundary."""
    cfg = load_config(config_path)
    result = fetch_or_load_boundary(cfg, force_refresh=force_refresh)
    logger.info("Boundary source: %s", result["source"])
    logger.info("4326 path:   %s", result["boundary_4326"])
    logger.info("32644 path:  %s", result["boundary_32644"])


if __name__ == "__main__":
    main()
