# Reference labeling and accuracy workflow

No reference labels have been created. Candidate map classes are only strata for sampling and are not ground truth.

## Label record schema

`data/labels/labels.csv` is completed manually and contains one row per point per labeller:

| Column | Meaning |
|---|---|
| `id` | Candidate point identifier; retain the same ID across annotators |
| `lon`, `lat` | WGS84 point coordinates |
| `candidate_epoch` | Candidate sampling stratum epoch (`t1` or `t2`); copy from the candidate point layer |
| `class_t1`, `class_t2` | Independently interpreted reference class IDs, 1–5; blank until assessed |
| `labeller` | Annotator name or stable code |
| `confidence` | Annotator confidence, use a consistent scale |
| `notes` | Interpretation notes and evidence |

The class IDs are 1 built-up, 2 tree, 3 cropland/grass, 4 bare, 5 water. The candidate GeoPackage also stores predicted classes and sampling strata. Do not copy those map predictions into the reference class columns without independent interpretation.

Keep `candidate_epoch` when exporting labels. Accuracy evaluation filters the
candidate sample to the epoch-specific map strata. The optional area-adjusted
estimate also requires this field to retain the epoch's mapped-class probability
sample design.

## Candidate points

After T1 and T2 five-class maps exist, run from the repository root:

```bash
python scripts/sample_labels.py \
  --config config/config.yaml \
  --t1-map data/processed/lulc_t1.tif \
  --t2-map data/processed/lulc_t2.tif \
  --output-dir data/labels/candidates
```

The script uses the configured per-class target and random seed unless overridden. It writes a QGIS-ready point layer `candidate_points.gpkg` and an empty `labels_template.csv` header. Load the GeoPackage with the corresponding composite rasters in QGIS to inspect the imagery chips around candidate locations, interpret both epochs, and enter reference classes. Duplicate each point row for the second labeller while retaining its `id`. The script does not write any class labels.

## Agreement and accuracy

Inter-rater agreement for T1 or T2:

```bash
python -m geointel.change.accuracy --labels data/labels/labels.csv --epoch t1 --interrater
```

Independent classification accuracy uses a class map and a chosen labeller. Coordinates are transformed from WGS84 to the prediction raster CRS; the holdout split keeps 5 km spatial blocks disjoint.

```bash
python -m geointel.change.accuracy \
  --labels data/labels/labels.csv \
  --prediction-raster data/processed/lulc_t1.tif \
  --epoch t1 --labeller annotator_A
```

The report includes the confusion matrix, overall accuracy, Cohen's kappa, per-class precision/recall/F1, and warnings when overall or held-out per-class support is below 30. Use `--epoch t2` and the T2 map for the second epoch. No metrics are available until labels are supplied. Pseudo-label RF metrics remain bootstrap-only and are not independent accuracy.
