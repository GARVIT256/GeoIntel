"""
GEO-INTEL data module.

Provides deterministic data retrieval and preprocessing functions:
- Sentinel-2 STAC search, SCL cloud masking, and median compositing
- Admin boundary downloading and clipping
- OpenStreetMap vector data fetching
- Copernicus DEM GLO-30 acquisition, clipping, and slope/aspect derivation
"""

from geointel.data.boundary import (
    aoi_bbox_gdf,
    download_boundary,
    fetch_or_load_boundary,
)
get_dehradun_boundary = fetch_or_load_boundary
from geointel.data.composite import (
    build_composite_for_epoch,
    compute_median_composite,
)
build_sentinel2_composite = build_composite_for_epoch
from geointel.data.dem import fetch_or_load_dem
get_copernicus_dem = fetch_or_load_dem
from geointel.data.osm import fetch_or_load_osm
get_osm_layers = fetch_or_load_osm

__all__ = [
    "search_sentinel2_items",
    "build_sentinel2_composite",
    "get_dehradun_boundary",
    "get_osm_layers",
    "get_copernicus_dem",
]
