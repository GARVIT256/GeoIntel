# Phase 3 execution report

**Status:** Accepted as code-complete, not verified. No real classification, raster, or change result is claimed. The 30 km AOI proposal has since been approved and adopted; hosted composites must be rerun for it.

## Files built or changed

- `src/geointel/gis/core.py`, `src/geointel/gis/__init__.py`: CRS-checked reproject/clip, metric area/length, buffer, overlay, zonal statistics, fishnet, raster-to-grid aggregation, and raster reproject/clip functions.
- `src/geointel/change/rule_based.py`, `src/geointel/change/rf_classifier.py`: restored 5-class names and strict priority; RF accepts pre-encoded aspect sine/cosine features.
- `src/geointel/change/detection.py`: joint valid mask and excluded pixel count, area, percent, and footprint area fields.
- `src/geointel/data/dem.py`: derive terrain on the 30 m DEM grid, encode aspect there, then bilinearly align elevation, slope, sine, and cosine to the reference grid.
- `config/config.yaml`, `config/aoi_dehradun_30km_approved.geojson`: approved 5-class configuration and approved AOI. The former `config/aoi_dehradun.geojson` is retained as historical geometry.
- `docs/definitions.md`, `docs/methods.md`, `docs/postgis_queries.md`, `docs/phase3_report.md`: class, resampling, exclusion, AOI, SQL-query, and phase documentation.
- `tests/test_phase3_change.py`, `tests/test_gis_core.py`, `pyproject.toml`: rule/matrix checks, GIS synthetic checks, `raster` marker.

## Exact test command and result

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase3_change.py tests/test_gis_core.py tests/test_phase2_data.py -q
```

Actual output summary: **60 passed, 3 skipped**, Python 3.14.2, pytest 8.3.2. The three skipped tests are marked `@pytest.mark.raster`; Rasterio could not be imported in this Windows runtime. The run emitted pytest-asyncio deprecation warnings.

## Measured geometry and unrun results

- Existing AOI from `config/aoi_dehradun.geojson`, projected to EPSG:32644: bounds (km) `[204.870, 3331.134, 231.965, 3376.903]`; extent **27.096 × 45.770 km**; polygon area **1,170.378 km²**. These are geometry calculations, not raster outputs.
- Approved AOI: 30 × 30 km projected square (900 km²), north edge south of the Mussoorie ridge. Adopted in `config/config.yaml` on user approval. Rerun the hosted Colab composites on this footprint.
- Real composite reflectance-offset comparison: **NOT YET RUN**.
- Real tile-seam review: **NOT YET RUN**.
- Real valid-pixel coverage and transition excluded area: **NOT YET RUN**.
- Real scene counts and classification/change outputs: **NOT YET RUN**.
- Rasterio/GDAL-dependent test behavior: **NOT VERIFIED in this Windows environment**; run the marked tests in Colab/Kaggle.
- Docker Compose/PostGIS integration script: `scripts/postgis_check.py`; status **NOT YET RUN**. Run it manually with `python scripts/postgis_check.py`; it starts the service and compares all three synthetic query outputs with documented expectations.
- Hosted notebook now runs the full pytest suite without filtering the `raster` marker and saves `full_pytest_output.txt` under the selected run output directory.

## Limitations and manual actions

- AOI approval is complete. The active config points to the approved square; regenerate all hosted composites and downstream products for this footprint.
- Execute the hosted Colab/Kaggle run to produce composites. Inspect T1/T2 reflectance harmonization, seams, coverage against the configured threshold, and scene counts before closing Phase 2.
- PostGIS schema already exists at `src/geointel/db/schema.sql`; query examples and synthetic expected outputs are in `docs/postgis_queries.md`. Database execution was not run as part of this phase check.
- Collect reference labels before reporting independent classification accuracy. Current pseudo-label RF evaluation remains bootstrap-only.

## Suggested slide content

1. **Methods:** five-class precedence and index thresholds; explain that BSI <= 0 with NDVI <= 0.25 is unclassified unless a higher-priority rule matches.
2. **Spatial workflow:** Sentinel-2 10 m grid, Copernicus DEM native 30 m terrain derivation, aspect sine/cosine encoding before bilinear resampling, and the joint T1/T2 validity mask.
3. **Study area and status:** show the approved 30 km square and label real raster findings NOT YET RUN until the hosted run completes.

## Report text

Phase 3 adds CRS-checked vector/raster GIS utilities, synthetic known-answer tests, and PostGIS query examples. The classifier uses the approved five classes with water > built-up > tree > cropland/grass > bare > unclassified precedence. Transition accounting uses a joint valid-pixel mask and reports excluded footprint area. Terrain aspect is encoded at the DEM's native 30 m resolution before bilinear resampling to the 10 m reference grid. Synthetic test summary: 60 passed and 3 raster-marked tests skipped because Rasterio is unavailable in the local Windows runtime. No real composite or change result is reported. The 30 km AOI is approved and active; rerun hosted composites before real-data reporting.
