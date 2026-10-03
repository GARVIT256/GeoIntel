# Phase 4 report: candidate sampling and accuracy

**Status:** Started. Candidate-point tooling, label schema, and metrics are implemented; no reference labels have been created and no real accuracy result exists.

## Files built

- `src/geointel/change/sampling.py`: seeded stratified pixel candidate sampler over class IDs 1–5.
- `scripts/sample_labels.py`: writes a QGIS-ready candidate GeoPackage and empty canonical CSV header. Candidate classes are map-derived strata, never ground truth.
- `scripts/phase4_candidates.py`: works from the T1/T2 composite COGs produced in the hosted run, creates baseline rule maps, then writes candidate points and the empty CSV template.
- `src/geointel/change/accuracy.py`: accepts canonical WGS84 label records, reprojects coordinates to the prediction raster CRS, reports spatial-block confusion matrix/OA/kappa/per-class precision-recall-F1, warns on low sample support, and computes two-labeller Cohen's kappa.
- `docs/labeling.md`: documents schema, QGIS workflow, and commands. `tests/test_phase4_labels.py` covers synthetic sampling, agreement, and a raster-marked coordinate evaluation.
- `notebooks/colab_run.ipynb`: after the AOI-specific composites finish, its optional Phase 4 cell creates rule-map candidates from those hosted COGs; it does not create reference labels.

## Commands

Create candidate points after T1/T2 class maps exist:

```bash
python scripts/sample_labels.py --config config/config.yaml \
  --t1-map data/processed/lulc_t1.tif --t2-map data/processed/lulc_t2.tif \
  --output-dir data/labels/candidates
```

Inter-rater agreement after two annotators complete rows for shared IDs:

```bash
python -m geointel.change.accuracy --labels data/labels/labels.csv --epoch t1 --interrater
```

Independent block holdout after labels and the target epoch map exist:

```bash
python -m geointel.change.accuracy --labels data/labels/labels.csv \
  --prediction-raster data/processed/lulc_t1.tif --epoch t1 --labeller annotator_A
```

## Real data and tests

- Reference labels created: **none**.
- Independent accuracy, confusion matrix, and inter-rater kappa: **NOT YET RUN**; require manual labels.
- Candidate points from approved-AOI composites: **NOT YET RUN**; first rerun Colab composites for `config/aoi_dehradun_30km_approved.geojson`.
- Local test command: `.\\.venv\\Scripts\\python.exe -m pytest -q --tb=short`
- Actual local result: **68 passed, 4 skipped** (72 collected) under Python 3.14.2/pytest 8.3.2. The four `@pytest.mark.raster` tests skipped because the Windows Application Control policy blocked Rasterio's native extension. This does not substitute for the hosted notebook run, where raster tests are attempted.

## Manual work and limitations

Run the hosted composite workflow with the approved AOI. Review candidate locations in QGIS alongside T1/T2 composite rasters. Interpret each epoch independently, enter class IDs, confidence, and notes, and duplicate each point ID for the second annotator. Use the same ID for both annotators so kappa pairs the same locations. Candidate map classes must not be used as reference values. Independent evaluation requires at least two spatial blocks; low total or per-class support is reported with warnings.

## Suggested slide content

1. Candidate selection: seeded stratified sample from each epoch's baseline map, with map classes shown only as sampling strata.
2. Label protocol: QGIS point GeoPackage, matching IDs for two annotators, class definitions, confidence, and notes.
3. Validation: 5 km spatially held-out confusion matrix, OA, per-class precision/recall/F1, kappa, and explicit sample-count warnings; show **NOT YET RUN** until labels are supplied.

## Report text

Phase 4 has begun with reproducible candidate sampling and an empty labeling schema. The hosted notebook can derive rule-map strata from the approved-AOI T1/T2 COGs and export QGIS-ready points. No field/reference labels were created. Accuracy and inter-rater agreement remain unavailable until manual labeling is complete; candidate map classes are not ground truth.
