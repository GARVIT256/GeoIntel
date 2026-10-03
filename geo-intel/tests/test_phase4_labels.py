"""Synthetic Phase 4 candidate-sampling and agreement checks; no field labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import importlib

from geointel.change.accuracy import _label_count_warnings, evaluate_interrater_kappa
from geointel.change.sampling import build_candidate_records, sample_stratified_pixels


def test_stratified_sampler_returns_requested_known_class_counts() -> None:
    classes = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.uint8), (20, 1))
    samples = sample_stratified_pixels(classes, samples_per_class=3, seed=7)
    assert len(samples) == 15
    assert {class_id: sum(s["candidate_class"] == class_id for s in samples) for class_id in range(1, 6)} == {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 3
    }


def test_stratified_sampler_is_deterministic() -> None:
    classes = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.uint8), (20, 1))
    assert sample_stratified_pixels(classes, 4, seed=17) == sample_stratified_pixels(classes, 4, seed=17)


def test_stratified_sampler_caps_to_available_pixels() -> None:
    classes = np.array([[1, 1, 3], [5, 0, 0]], dtype=np.uint8)
    samples = sample_stratified_pixels(classes, samples_per_class=10)
    assert len(samples) == 4
    assert {s["candidate_class"] for s in samples} == {1, 3, 5}


def test_stratified_sampler_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sample_stratified_pixels(np.ones(4), 2)
    with pytest.raises(ValueError, match="positive"):
        sample_stratified_pixels(np.ones((2, 2)), 0)


def test_candidate_records_keep_both_epoch_predictions() -> None:
    t1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    t2 = np.array([[5, 2], [3, 4]], dtype=np.uint8)
    records = build_candidate_records(t1, t2, samples_per_class=1, seed=1)
    assert {r["candidate_epoch"] for r in records} == {"t1", "t2"}
    assert all("pred_t1" in r and "pred_t2" in r for r in records)


def test_interrater_kappa_uses_shared_ids_and_two_annotators(tmp_path) -> None:
    frame = pd.DataFrame([
        {"id": "p1", "lon": 78.0, "lat": 30.0, "class_t1": 1, "class_t2": 2, "labeller": "A", "confidence": 1, "notes": ""},
        {"id": "p1", "lon": 78.0, "lat": 30.0, "class_t1": 1, "class_t2": 2, "labeller": "B", "confidence": 1, "notes": ""},
        {"id": "p2", "lon": 78.1, "lat": 30.1, "class_t1": 3, "class_t2": 4, "labeller": "A", "confidence": 2, "notes": ""},
        {"id": "p2", "lon": 78.1, "lat": 30.1, "class_t1": 3, "class_t2": 4, "labeller": "B", "confidence": 2, "notes": ""},
    ])
    path = tmp_path / "synthetic_labels.csv"
    frame.to_csv(path, index=False)
    result = evaluate_interrater_kappa(path, "t1")
    assert result["n_paired_points"] == 2
    assert result["cohen_kappa"] == pytest.approx(1.0)
    assert "not classification accuracy" in result["status"]


def test_interrater_kappa_rejects_unpaired_schema(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("id,class_t2,labeller\np1,1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two labellers"):
        evaluate_interrater_kappa(path, "t2")


@pytest.mark.raster
def test_canonical_lon_lat_csv_evaluates_against_raster(tmp_path) -> None:
    try:
        rio = importlib.import_module("rasterio")
    except Exception as exc:
        pytest.skip(f"rasterio unavailable in this runtime: {exc}")
    from rasterio.transform import from_origin, xy
    from rasterio.warp import transform

    from geointel.change.accuracy import evaluate_labels_csv

    raster_path = tmp_path / "synthetic_lulc.tif"
    raster_transform = from_origin(200000, 3350000, 10, 10)
    prediction = np.ones((20, 502), dtype=np.uint8)
    prediction[:, 500:] = 2
    with rio.open(
        raster_path, "w", driver="GTiff", height=20, width=502, count=1,
        dtype="uint8", crs="EPSG:32644", transform=raster_transform,
    ) as dst:
        dst.write(prediction, 1)

    pixel_xy = [xy(raster_transform, 5, col, offset="center") for col in (1, 2, 500, 501)]
    lon, lat = transform(
        "EPSG:32644", "EPSG:4326", [point[0] for point in pixel_xy], [point[1] for point in pixel_xy]
    )
    label_path = tmp_path / "synthetic_reference.csv"
    pd.DataFrame({
        "id": ["p1", "p2", "p3", "p4"], "lon": lon, "lat": lat,
        "class_t1": [1, 1, 1, 1], "class_t2": [1, 1, 2, 2],
        "labeller": ["A"] * 4, "confidence": [3] * 4, "notes": [""] * 4,
    }).to_csv(label_path, index=False)

    result = evaluate_labels_csv(
        label_path, raster_path, block_size_m=5000, test_fraction=0.5, epoch="t2"
    )
    assert result["n_reference_samples"] == 4
    assert result["validation_status"] == "independent reference-label holdout"


def test_label_count_warning_flags_low_class_support() -> None:
    warnings = _label_count_warnings(40, {1: 35, 2: 5})
    assert len(warnings) == 1
    assert "Class 2" in warnings[0]
