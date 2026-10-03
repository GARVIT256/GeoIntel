"""
change/rf_classifier.py — Supervised Random Forest LULC classifier (Baseline 2).

Trains a scikit-learn Random Forest model using 13 features per pixel:
- 6 spectral bands: B02, B03, B04, B08, B11, B12
- 4 spectral indices: NDVI, NDBI, MNDWI, BSI
- 3 terrain features: slope, sin(aspect), cos(aspect)

Rule-derived metrics are circular and are for bootstrap diagnostics only.

Model Persistence
-----------------
Model artifacts saved to data/cache/models/rf_lulc.joblib
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from geointel.change.indices import compute_all_indices
from geointel.change.rule_based import LULC_CLASSES, classify_rule_based
from geointel.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except Exception as exc:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None  # type: ignore
    logger.warning("sklearn/scipy import unavailable; RF training is disabled (%s).", exc)

FEATURE_NAMES = [
    "B02", "B03", "B04", "B08", "B11", "B12",
    "NDVI", "NDBI", "MNDWI", "BSI",
    "slope", "aspect_sin", "aspect_cos",
]


def extract_feature_stack(
    composite: xr.DataArray,
    slope: xr.DataArray | np.ndarray | None = None,
    aspect: xr.DataArray | np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Construct a (N_pixels, 13) feature matrix from composite and terrain data.

    Parameters
    ----------
    composite : xr.DataArray
        Shape (6, H, W) float32 composite.
    slope : DataArray or ndarray, optional
        Shape (H, W) float32 slope in degrees. Default zeros.
    aspect : DataArray or ndarray, optional
        Shape (H, W) float32 aspect in degrees, encoded as sine and cosine.

    Returns
    -------
    tuple of (X_matrix, (height, width))
        X_matrix shape: (H*W, 13)
    """
    h, w = composite.shape[1], composite.shape[2]

    # Compute 4 spectral indices
    indices = compute_all_indices(composite)  # shape (4, H, W)

    # Convert spectral bands to numpy (6, H, W)
    bands_np = composite.values.astype("float32")
    indices_np = indices.values.astype("float32")

    if slope is None:
        slope_np = np.zeros((1, h, w), dtype="float32")
    else:
        slope_val = slope.values if isinstance(slope, xr.DataArray) else slope
        slope_np = slope_val.reshape(1, h, w).astype("float32")

    if aspect is None:
        aspect_features = np.zeros((2, h, w), dtype="float32")
    else:
        aspect_val = aspect.values if isinstance(aspect, xr.DataArray) else aspect
        aspect_rad = np.deg2rad(aspect_val.reshape(h, w).astype("float32"))
        aspect_features = np.stack([np.sin(aspect_rad), np.cos(aspect_rad)]).astype("float32")

    # Circular encoding keeps aspects near 0° and 360° close in feature space.
    feat_stack = np.concatenate([bands_np, indices_np, slope_np, aspect_features], axis=0)

    X = feat_stack.reshape(13, -1).T
    return X, (h, w)


def generate_pseudo_training_data(
    composite: xr.DataArray,
    slope: xr.DataArray | np.ndarray | None = None,
    aspect: xr.DataArray | np.ndarray | None = None,
    n_samples_per_class: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract rule-based pseudo-labels for bootstrap training only. Agreement with
    this baseline is circular and is not independent accuracy.

    Returns
    -------
    X_train : ndarray of shape (N_samples, 13)
    y_train : ndarray of shape (N_samples,)
    """
    rule_map = classify_rule_based(composite).values.ravel()
    X, _ = extract_feature_stack(composite, slope, aspect)

    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for class_id in [1, 2, 3, 4, 5]:
        idx = np.where(rule_map == class_id)[0]
        # Filter out NaN feature rows
        valid_idx = idx[~np.isnan(X[idx]).any(axis=1)]
        
        if len(valid_idx) == 0:
            logger.warning("No valid pseudo-training samples for class %d", class_id)
            continue
            
        n_sample = min(n_samples_per_class, len(valid_idx))
        sampled = rng.choice(valid_idx, size=n_sample, replace=False)
        
        X_list.append(X[sampled])
        y_list.append(np.full(n_sample, class_id, dtype=np.uint8))

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    logger.info("Generated %d pseudo-training samples across %d classes", len(y_train), len(X_list))
    return X_train, y_train


def train_rf_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """
    Train Random Forest classifier with stratified 5-fold cross-validation metrics.

    Returns
    -------
    tuple of (trained_rf_model, metrics_dict)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required to train the Random Forest")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        oob_score=True,
        class_weight="balanced",
        n_jobs=-1,
    )
    
    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    acc_scores, kappa_scores = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train[train_idx], y_train[train_idx])
        y_pred = clf.predict(X_train[val_idx])
        
        acc_scores.append(accuracy_score(y_train[val_idx], y_pred))
        kappa_scores.append(cohen_kappa_score(y_train[val_idx], y_pred))

    # Fit final model on full dataset
    rf.fit(X_train, y_train)

    metrics = {
        "oob_score": float(rf.oob_score_),
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
        "cv_kappa_mean": float(np.mean(kappa_scores)),
        "cv_kappa_std": float(np.std(kappa_scores)),
        "feature_importances": dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist())),
        "label_source": "rule_based_pseudo_labels",
        "validation_status": "bootstrap only; circular, not independent accuracy",
    }

    logger.info("RF Model Trained. OOB Score: %.4f, CV Acc: %.4f (+/- %.4f)",
                metrics["oob_score"], metrics["cv_accuracy_mean"], metrics["cv_accuracy_std"])
    return rf, metrics


def predict_rf_lulc(
    model: RandomForestClassifier,
    composite: xr.DataArray,
    slope: xr.DataArray | np.ndarray | None = None,
    aspect: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray:
    """
    Apply a trained Random Forest model to classify a full composite into a five-class LULC raster.
    """
    X, (h, w) = extract_feature_stack(composite, slope, aspect)
    
    # Handle NaNs in test feature matrix
    valid_mask = ~np.isnan(X).any(axis=1)
    y_pred = np.zeros(h * w, dtype=np.uint8)

    if valid_mask.any():
        y_pred[valid_mask] = model.predict(X[valid_mask])

    lulc_map = y_pred.reshape(h, w)

    return xr.DataArray(
        lulc_map,
        dims=["y", "x"],
        coords={"y": composite.coords["y"], "x": composite.coords["x"]},
        attrs={
            "description": "Random Forest five-class LULC map",
            "classes": LULC_CLASSES,
            "crs": composite.attrs.get("crs", "EPSG:32644"),
        },
    )
