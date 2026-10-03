"""GIS operations with explicit coordinate reference system checks."""

from geointel.gis.core import (
    COMPUTE_CRS,
    assert_crs,
    buffer_m,
    clip_vector,
    geometry_area_m2,
    geometry_length_m,
    make_fishnet,
    overlay,
    raster_grid_aggregation,
    reproject_clip_raster,
    reproject_vector,
    zonal_stats,
)

__all__ = [
    "COMPUTE_CRS", "assert_crs", "buffer_m", "clip_vector",
    "geometry_area_m2", "geometry_length_m", "make_fishnet", "overlay",
    "raster_grid_aggregation", "reproject_clip_raster", "reproject_vector",
    "zonal_stats",
]
