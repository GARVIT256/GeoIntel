# GEO-INTEL limitations and evidence boundaries

## Data and spatial analysis

- The approved AOI composites have not been rerun in hosted Colab/Kaggle. Real
  scene counts, coverage, reflectance offsets, tile seams, classifications,
  transition statistics, associations, and hotspot results remain **NOT YET
  RUN**.
- T1 (2018–19) and T2 (2024–25) composites require visual QA for harmonization,
  cloud-mask artifacts, seams, and valid-pixel coverage before interpretation.
- Terrain derivatives, classification, postprocessing, and change analysis are
  implemented, but local fixture tests do not establish real-raster behavior.
- A joint valid-pixel mask defines the transition-matrix comparison footprint;
  excluded area must be reported from actual hosted outputs.
- Spatial association and NDVI difference describe measured patterns. They do
  not establish a causal mechanism.

## Reference and external data

- No manual reference labels exist. Classification accuracy, inter-rater
  agreement, and area-adjusted estimates are unavailable until labels are
  independently interpreted under the sampling protocol.
- GHSL E2020/E2025 do not match the T1/T2 epochs; E2025 is a projection. Use
  these products only as approximate consistency checks.
- Hansen and WorldCover tiles still need to be acquired and checked against the
  approved AOI bounds before external consistency checks. They are not ground
  truth for the project epochs.
- Candidate points, if generated later, are sampling aids only; predicted map
  classes must not be copied into reference-label fields.

## Agent and demo

- The 10-query mock benchmark checks cached-plan plumbing only. Its scores are
  not accuracy or model quality. The 40-query live-planner benchmark is **NOT
  YET RUN / NOT YET MEASURED**.
- The provenance checker checks exact numeric tokens and declared units against
  supplied result JSON. It does not verify the underlying data, derived
  arithmetic, or whether a value is statistically appropriate.
- The causal-language linter is phrase-based; it can miss unlisted wording and
  flag some harmless or negated uses.
- The Streamlit app is fixture-only and does not consume real analysis outputs.

## Verification boundary

Passing local tests verifies only the cases those tests exercise. Raster-marked
tests require a functioning GDAL/Rasterio environment; Docker/PostGIS, hosted
Colab/Kaggle execution, PDF/reference verification, and browser review require
their own manual runs. Current states and actions are tracked in
[`STATUS.md`](STATUS.md).
