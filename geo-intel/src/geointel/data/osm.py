"""
data/osm.py — OpenStreetMap data download for GEO-INTEL.

Downloads roads, rivers, and building footprints for the AOI from the
Overpass API and caches as GeoPackages. These layers are for later phases
(Phase 3+) and for contextual map display. They are NOT used to compute
change statistics.

IMPORTANT: OSM data reflects what has been mapped, not ground truth.
OSM completeness varies across the Dehradun fringe. Do NOT interpret
the presence or absence of OSM features as evidence of construction or
road building. This limitation is documented in docs/limitations.md.

CRS handling
------------
- OSM coordinates are always EPSG:4326.
- We save EPSG:4326 for display and EPSG:32644 for spatial joins in PostGIS.

UNVERIFIED: Overpass API query syntax for India / Dehradun completeness.
  Verify queries at https://overpass-turbo.eu before the demo.

CLI
---
    python -m geointel.data.osm --config config/config.yaml

Outputs
-------
    data/cache/osm/roads_4326.gpkg
    data/cache/osm/rivers_4326.gpkg
    data/cache/osm/buildings_4326.gpkg
    data/cache/osm/roads_32644.gpkg
    data/cache/osm/rivers_32644.gpkg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import httpx
from shapely.geometry import LineString, Point, Polygon, mapping

from geointel.utils.config import load_config
from geointel.utils.logging import get_logger

logger = get_logger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DISPLAY_CRS  = "EPSG:4326"
COMPUTE_CRS  = "EPSG:32644"


# ---------------------------------------------------------------------------
# Overpass queries (bounding box version for reliability)
# ---------------------------------------------------------------------------

def _bbox_str(bbox: list[float]) -> str:
    """Convert [W, S, E, N] bbox to Overpass bbox string 'S,W,N,E'."""
    w, s, e, n = bbox
    return f"{s},{w},{n},{e}"


def _roads_query(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:90];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]({bb});
);
out geom;
"""


def _rivers_query(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:60];
(
  way["waterway"~"^(river|stream|canal)$"]({bb});
);
out geom;
"""


def _buildings_query(bbox: list[float]) -> str:
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:120];
(
  way["building"]({bb});
);
out geom;
"""


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_ways_to_gdf(response_json: dict[str, Any], layer_name: str) -> gpd.GeoDataFrame:
    """Parse Overpass 'out geom' way elements into a GeoDataFrame."""
    elements = response_json.get("elements", [])
    rows = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        geom = LineString(coords) if layer_name != "buildings" else _to_polygon(coords)
        tags = el.get("tags", {})
        rows.append({
            "osm_id": el["id"],
            "layer": layer_name,
            "name": tags.get("name", ""),
            "type": tags.get("highway") or tags.get("waterway") or tags.get("building", ""),
            "geometry": geom,
        })

    if not rows:
        logger.warning("No %s features returned by Overpass.", layer_name)
        return gpd.GeoDataFrame(columns=["osm_id", "layer", "name", "type", "geometry"],
                                 geometry="geometry", crs=DISPLAY_CRS)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=DISPLAY_CRS)
    logger.info("Parsed %d %s features from Overpass.", len(gdf), layer_name)
    return gdf


def _to_polygon(coords: list[tuple[float, float]]) -> Polygon | LineString:
    """Try to make a Polygon; fall back to LineString if < 3 unique points."""
    if len(coords) >= 4 and coords[0] == coords[-1]:
        try:
            return Polygon(coords)
        except Exception:
            pass
    return LineString(coords)


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def _download_layer(
    query: str,
    layer_name: str,
    timeout_s: int = 120,
) -> gpd.GeoDataFrame | None:
    """Run an Overpass query and return a GeoDataFrame, or None on failure."""
    logger.info("Downloading OSM layer: %s ...", layer_name)
    try:
        resp = httpx.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=timeout_s,
            headers={"User-Agent": "geo-intel-research/0.1"},
        )
        resp.raise_for_status()
        data = resp.json()
        return _parse_ways_to_gdf(data, layer_name)
    except Exception as exc:
        logger.warning("OSM %s download failed: %s", layer_name, exc)
        return None


def fetch_or_load_osm(
    cfg: dict[str, Any],
    force_refresh: bool = False,
) -> dict[str, Path | None]:
    """
    Cache-first OSM download for roads, rivers, buildings.

    Returns
    -------
    dict mapping layer name → Path (or None if download failed).

    CRS guarantee
    -------------
    All *_4326.gpkg files are EPSG:4326.
    All *_32644.gpkg files are EPSG:32644.
    """
    cache_dir: Path = cfg["paths"]["data_cache"] / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bbox: list[float] = cfg["aoi"]["bbox"]

    layers = {
        "roads":     (_roads_query(bbox),     "roads"),
        "rivers":    (_rivers_query(bbox),    "rivers"),
        "buildings": (_buildings_query(bbox), "buildings"),
    }

    result: dict[str, Path | None] = {}

    for key, (query, layer_name) in layers.items():
        out_4326   = cache_dir / f"{key}_4326.gpkg"
        out_32644  = cache_dir / f"{key}_32644.gpkg"

        if out_4326.exists() and not force_refresh:
            logger.info("OSM cache hit: %s", out_4326)
            result[f"{key}_4326"]  = out_4326
            result[f"{key}_32644"] = out_32644 if out_32644.exists() else None
            continue

        gdf = _download_layer(query, layer_name)
        if gdf is None or gdf.empty:
            result[f"{key}_4326"]  = None
            result[f"{key}_32644"] = None
            continue

        # Explicit CRS check before saving
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326, \
            f"OSM GDF CRS must be 4326, got {gdf.crs}"

        gdf.to_file(out_4326, driver="GPKG")
        logger.info("OSM %s (4326) → %s", key, out_4326)

        if key != "buildings":   # buildings in 4326 only; skip 32644 for size
            gdf_32644 = gdf.to_crs(COMPUTE_CRS)
            assert gdf_32644.crs.to_epsg() == 32644
            gdf_32644.to_file(out_32644, driver="GPKG")
            logger.info("OSM %s (32644) → %s", key, out_32644)
            result[f"{key}_32644"] = out_32644
        else:
            result[f"{key}_32644"] = None

        result[f"{key}_4326"] = out_4326

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default="config/config.yaml", show_default=True)
@click.option("--force-refresh", is_flag=True, default=False)
def main(config_path: str, force_refresh: bool) -> None:
    """Download and cache OSM roads, rivers, and buildings for the AOI."""
    cfg = load_config(config_path)
    result = fetch_or_load_osm(cfg, force_refresh=force_refresh)
    for key, path in result.items():
        logger.info("%s → %s", key, path or "FAILED/SKIPPED")


if __name__ == "__main__":
    main()
