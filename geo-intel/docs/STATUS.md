# GEO-INTEL phase status

Status reflects repository state and the local full-suite run recorded below.
“Verified” is limited to the cited code/fixture checks; it does not imply
verification on real imagery or hosted infrastructure.

## Phase summary

| Phase | Verified | NOT YET RUN / not verified | Remaining manual steps |
|---|---|---|---|
| 1 — Problem definition and literature | Project scope, definitions, configuration, and literature inventory are present. | Literature entries have not all been checked against source PDFs. | Retrieve/read the cited PDFs, validate title/authors/year/DOI and relevant claims, then update `data/corpus/papers_index.csv` and `docs/gap_table.md`. |
| 2 — Data acquisition and preprocessing | Pipeline code and synthetic cases are included in the local suite; approved AOI is recorded in config. | Approved-AOI Colab smoke test and full T1/T2 composites; real scenes, COGs, valid-pixel coverage, per-band statistics, seams, and reflectance-offset review. | Run `notebooks/colab_run.ipynb` in Python 3.11 Colab/Kaggle against the approved AOI; inspect smoke-test output and T1/T2 QA; persist COGs, metadata JSON, and manifest to Drive; populate `data_report.md` only from that run. |
| 3 — GIS and database | GIS/change fixture tests are included in the local suite; PostGIS expected-query comparison script exists. | Docker Compose/PostGIS query execution and comparison. | Start the compose PostGIS service and run `python scripts/postgis_check.py`; inspect its output and replace the script’s **NOT YET RUN** marker only after execution. |
| 4 — Labelling and accuracy | Candidate sampling and accuracy workflow code are present; local fixture tests pass as part of the full suite. | Candidate export for real composites, manual reference labels, inter-rater agreement, independent accuracy, and area-adjusted estimates. | After the approved-AOI hosted composites exist, generate candidate points; two annotators independently label the same IDs in QGIS; then run agreement and spatial holdout accuracy workflows. No labels currently exist. |
| 5 — Change detection and association | Fixture tests cover postprocessing, NDVI baseline, joint-mask transition logic, association methods, and external-check code paths. | Real Phase 4–5 rasters, association outputs, external checks, and area-adjusted estimates. | Run `scripts/run_phase45_colab.py` on the approved-AOI COGs after Phase 2; review seams/coverage/alignment and inspect outputs before interpretation. Obtain reference tiles listed below and run `scripts/validate_external.py`. |
| 6 — Agent, RAG, and validator | Mock plumbing, adversarial provenance/linter checks, and benchmark data separation pass locally. The mock’s cached-plan scores are plumbing checks, not accuracy. | Live LLM benchmark and real-data answers: **NOT YET MEASURED / NOT YET RUN**. | Configure `GEOINTEL_LLM_PROVIDER` and provider credentials, then explicitly run `python scripts/phase6_live_benchmark.py --execute-live`; review failures and keep gold plans out of prompts. |
| 7 — Streamlit demo | Fixture-only interface and DEMO DATA banners are implemented. | Streamlit launch/browser review and screenshots are not verified; real outputs are not loaded. | Run `python -m streamlit run app.py`, complete the checklist in `phase7_report.md`, and use the five-minute walkthrough in `demo_script.md`. |

## Full local test suite

Command: `.venv\\Scripts\\python.exe -m pytest -q --tb=short`

Latest full-suite run in this work session: **96 collected; 92 passed, 4
skipped, 0 failed** (Python 3.14.2, pytest 8.3.2). The skips are raster-marked
tests; this Windows run does not verify hosted GDAL/Rasterio execution. Pytest
reported 4,802 pytest-asyncio deprecation warnings. No hosted data pipeline was
run.

## Manual data and evidence checklist

- **Colab/Kaggle:** Run the notebook from top to bottom after mounting Drive.
  Confirm approved-AOI smoke-test success before the full T1/T2 composite run.
  Preserve the generated COGs, metadata JSON, manifest, and preview/valid-pixel
  outputs; review reflectance offset, seams, scene IDs/counts, and each epoch’s
  valid-pixel coverage against the configured threshold.
- **Reference labels:** None exist. Create them only through independent human
  interpretation after real composites are reviewed. Candidate map classes are
  not labels or ground truth.
- **PostGIS:** Start Docker Compose and run `scripts/postgis_check.py`. Its state
  remains **NOT YET RUN** until that command is run and its output is checked.
- **PDFs:** Verify the candidate literature entries from their source papers;
  record actual bibliographic metadata and claim support. Inventory and search
  notes are not proof that a paper was read or verified.
- **Reference tiles for Phase 5 consistency checks:**
  - GHSL built-up surface archives: `GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.zip`
    and `GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0.zip`; extract the
    GeoTIFFs. These epochs do not match T1/T2; E2025 is a projection.
  - Hansen GFC GeoTIFFs: `Hansen_GFC-2025-v1.13_lossyear_40N_070E.tif`,
    `Hansen_GFC-2025-v1.13_datamask_40N_070E.tif`, and
    `Hansen_GFC-2025-v1.13_treecover2000_40N_070E.tif` (tile bounds 30–40°N,
    70–80°E).
  - ESA WorldCover 2021 v200 map COGs:
    `ESA_WorldCover_10m_2021_v200_N30E075_Map.tif` and
    `ESA_WorldCover_10m_2021_v200_N30E078_Map.tif` (bounds 30–33°N,
    75–78°E and 78–81°E respectively).
  - These are approximate consistency references, not epoch-matched ground
    truth. External checks remain **NOT YET RUN** until acquisition and execution.
