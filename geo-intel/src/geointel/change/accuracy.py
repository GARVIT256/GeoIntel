"""Independent LULC accuracy assessment with spatially blocked holdout labels.

Expected reference CSV columns: x, y, class_id. Coordinates must be in the
prediction raster CRS; class IDs follow the configured five-class scheme.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd


def spatial_block_split(
    x: np.ndarray,
    y: np.ndarray,
    block_size_m: float = 5000.0,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices by projected-coordinate blocks, with no block leakage."""
    from sklearn.model_selection import GroupShuffleSplit

    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have equal length with at least two samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Sample coordinates must be finite")

    block_x = np.floor(np.asarray(x, dtype=np.float64) / block_size_m).astype(np.int64)
    block_y = np.floor(np.asarray(y, dtype=np.float64) / block_size_m).astype(np.int64)
    groups = np.array([f"{bx}:{by}" for bx, by in zip(block_x, block_y)])
    if np.unique(groups).size < 2:
        raise ValueError("At least two spatial blocks are required for holdout validation")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx, groups


def evaluate_spatial_block_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    block_size_m: float = 5000.0,
    test_fraction: float = 0.2,
    random_state: int = 42,
    class_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Report independent holdout metrics for reference labels in held-out blocks."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if not np.isin(y_true, class_ids).all():
        raise ValueError(f"Reference class IDs must be in {class_ids}")
    if not np.isin(y_pred, class_ids).all():
        raise ValueError(f"Predicted class IDs must be in {class_ids}; remove nodata samples")

    _, test_idx, groups = spatial_block_split(
        x, y, block_size_m, test_fraction, random_state
    )
    truth_holdout = y_true[test_idx]
    pred_holdout = y_pred[test_idx]
    report = classification_report(
        truth_holdout,
        pred_holdout,
        labels=list(class_ids),
        output_dict=True,
        zero_division=0,
    )
    return {
        "split": "spatial_block_holdout",
        "block_size_m": float(block_size_m),
        "n_reference_samples": int(len(y_true)),
        "n_test_samples": int(len(test_idx)),
        "n_spatial_blocks": int(np.unique(groups).size),
        "n_test_blocks": int(np.unique(groups[test_idx]).size),
        "overall_accuracy": float(accuracy_score(truth_holdout, pred_holdout)),
        "cohen_kappa": float(
            cohen_kappa_score(truth_holdout, pred_holdout, labels=list(class_ids))
        ),
        "confusion_matrix_labels": list(class_ids),
        "confusion_matrix": confusion_matrix(
            truth_holdout, pred_holdout, labels=list(class_ids)
        ).tolist(),
        "per_class": {
            str(class_id): report[str(class_id)]
            for class_id in class_ids
        },
        "reference_test_count_by_class": {
            str(class_id): int(np.count_nonzero(truth_holdout == class_id))
            for class_id in class_ids
        },
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "validation_status": "independent reference-label holdout",
        "warnings": _label_count_warnings(
            len(y_true),
            {class_id: int(np.count_nonzero(truth_holdout == class_id)) for class_id in class_ids},
        ),
    }


def _label_count_warnings(
    n_reference: int, test_counts: dict[int, int], minimum_per_class: int = 30
) -> list[str]:
    """Describe insufficient total and per-class reference sample support."""
    warnings = []
    if n_reference < minimum_per_class:
        warnings.append(f"Only {n_reference} reference samples; fewer than {minimum_per_class} overall.")
    for class_id, count in test_counts.items():
        if count < minimum_per_class:
            warnings.append(
                f"Class {class_id} has {count} held-out reference samples; fewer than {minimum_per_class}."
            )
    return warnings


def evaluate_interrater_kappa(
    labels_path: Path,
    epoch: str,
    labellers: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Calculate Cohen's kappa between two labellers on shared point IDs."""
    from sklearn.metrics import cohen_kappa_score

    if epoch not in {"t1", "t2"}:
        raise ValueError("epoch must be 't1' or 't2'")
    labels = pd.read_csv(labels_path)
    class_column = f"class_{epoch}"
    required = {"id", class_column, "labeller"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label CSV is missing required columns: {sorted(missing)}")
    labels = labels.dropna(subset=["id", class_column, "labeller"]).copy()
    labels[class_column] = pd.to_numeric(labels[class_column], errors="raise")
    if not np.isin(labels[class_column].to_numpy(), [1, 2, 3, 4, 5]).all():
        raise ValueError("Reference class IDs must be in 1..5")
    labels[class_column] = labels[class_column].astype(int)
    if labellers is None:
        names = sorted(labels["labeller"].astype(str).unique())
        if len(names) != 2:
            raise ValueError(f"Expected exactly two labellers with labels; found {len(names)}")
        labellers = (names[0], names[1])
    if labellers[0] == labellers[1]:
        raise ValueError("Two distinct labeller names are required")
    selected = labels[labels["labeller"].astype(str).isin(labellers)]
    if selected.duplicated(["id", "labeller"]).any():
        raise ValueError("Each point ID may have only one label per labeller")
    paired = selected.pivot(index="id", columns="labeller", values=class_column)
    if any(name not in paired.columns for name in labellers):
        raise ValueError("No labels found for one or both selected labellers")
    paired = paired.loc[:, list(labellers)].dropna()
    if paired.empty:
        raise ValueError("No shared labeled point IDs for the selected labellers")
    return {
        "epoch": epoch,
        "labellers": list(labellers),
        "n_paired_points": int(len(paired)),
        "cohen_kappa": float(
            cohen_kappa_score(paired.iloc[:, 0], paired.iloc[:, 1], labels=[1, 2, 3, 4, 5])
        ),
        "status": "inter-rater agreement only; not classification accuracy",
    }


def evaluate_labels_csv(
    labels_path: Path,
    prediction_raster: Path,
    block_size_m: float = 5000.0,
    test_fraction: float = 0.2,
    random_state: int = 42,
    epoch: str = "t2",
    labeller: str | None = None,
) -> dict[str, Any]:
    """Sample a prediction raster at supplied reference points and score holdout blocks."""
    import rasterio

    if epoch not in {"t1", "t2"}:
        raise ValueError("epoch must be 't1' or 't2'")
    labels = pd.read_csv(labels_path)
    canonical = {"lon", "lat", f"class_{epoch}", "labeller"}.issubset(labels.columns)
    if canonical:
        if "candidate_epoch" in labels.columns:
            labels = labels[labels["candidate_epoch"].astype(str) == epoch].copy()
        labels = labels.dropna(subset=["lon", "lat", f"class_{epoch}", "labeller"]).copy()
        if labeller is not None:
            labels = labels[labels["labeller"].astype(str) == labeller]
        elif labels["labeller"].nunique(dropna=True) > 1:
            raise ValueError("Choose a labeller when the CSV contains multiple annotators")
        if labels.empty:
            raise ValueError("No completed reference labels for the selected epoch/labeller")
        if "id" in labels and labels.duplicated(["id", "labeller"]).any():
            raise ValueError("Each point ID may have only one row per labeller")
        reference_values = pd.to_numeric(labels[f"class_{epoch}"], errors="raise")
    else:
        required_columns = {"x", "y", "class_id"}
        missing = required_columns - set(labels.columns)
        if missing:
            raise ValueError(
                "Expected canonical (id, lon, lat, class_t1, class_t2, labeller, confidence, notes) "
                f"or legacy (x, y, class_id) columns; missing: {sorted(missing)}"
            )
        reference_values = pd.to_numeric(labels["class_id"], errors="raise")
    if labels.empty:
        raise ValueError("Labels CSV contains no reference labels")

    with rasterio.open(prediction_raster) as raster:
        if raster.crs is None:
            raise ValueError("Prediction raster must declare a CRS")
        if canonical:
            from rasterio.warp import transform

            xs, ys = transform(
                "EPSG:4326", raster.crs,
                labels["lon"].to_numpy(dtype=np.float64).tolist(),
                labels["lat"].to_numpy(dtype=np.float64).tolist(),
            )
            coordinates = np.column_stack((xs, ys)).astype(np.float64)
        else:
            coordinates = labels[["x", "y"]].to_numpy(dtype=np.float64)
        sampled = list(raster.sample(coordinates, masked=True))
        class_predictions = np.array(
            [int(sample[0]) if not np.ma.is_masked(sample[0]) else 0 for sample in sampled],
            dtype=np.int64,
        )

    reference = reference_values.to_numpy(dtype=np.int64)
    valid = np.isin(reference, [1, 2, 3, 4, 5]) & np.isin(class_predictions, [1, 2, 3, 4, 5])
    if valid.sum() < 2:
        raise ValueError("Fewer than two valid reference/prediction samples; no metrics computed")

    result = evaluate_spatial_block_accuracy(
        reference[valid],
        class_predictions[valid],
        coordinates[valid, 0],
        coordinates[valid, 1],
        block_size_m=block_size_m,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    result["labels_path"] = str(labels_path)
    result["prediction_raster"] = str(prediction_raster)
    result["coordinate_crs"] = raster.crs.to_string()
    result["epoch"] = epoch
    if canonical and labels["labeller"].nunique(dropna=True) == 1:
        result["labeller"] = str(labels["labeller"].iloc[0])
    result["n_excluded_nodata_or_invalid"] = int((~valid).sum())
    return result


@click.command()
@click.option("--labels", "labels_path", type=click.Path(path_type=Path),
              default=Path("data/labels/labels.csv"), show_default=True)
@click.option("--prediction-raster", type=click.Path(path_type=Path),
              default=Path("data/processed/lulc_t2.tif"), show_default=True)
@click.option("--block-size-m", type=click.FloatRange(min=0, min_open=True), default=5000.0)
@click.option(
    "--test-fraction",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    default=0.2,
)
@click.option("--random-state", type=int, default=42)
@click.option("--epoch", type=click.Choice(["t1", "t2"]), default="t2", show_default=True)
@click.option("--labeller", type=str, default=None, help="Required when CSV has multiple annotators")
@click.option("--interrater", is_flag=True, help="Compute Cohen's kappa between two labellers instead of accuracy")
def main(
    labels_path: Path,
    prediction_raster: Path,
    block_size_m: float,
    test_fraction: float,
    random_state: int,
    epoch: str,
    labeller: str | None,
    interrater: bool,
) -> None:
    """Evaluate supplied reference labels; never substitutes pseudo-labels."""
    if not labels_path.exists():
        raise click.ClickException(
            f"Reference labels not found: {labels_path}. No accuracy metrics were produced."
        )
    try:
        if interrater:
            metrics = evaluate_interrater_kappa(labels_path, epoch)
        else:
            metrics = evaluate_labels_csv(
                labels_path,
                prediction_raster,
                block_size_m,
                test_fraction,
                random_state,
                epoch,
                labeller,
            )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
