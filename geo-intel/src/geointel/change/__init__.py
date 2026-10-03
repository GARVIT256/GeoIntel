"""
GEO-INTEL change module.

Provides LULC classification and change detection algorithms:
- Spectral index calculations (NDVI, NDBI, MNDWI, BSI)
- Rule-based decision tree classification (Baseline 1)
- Supervised Random Forest classification (Baseline 2)
- LULC transition matrix & spatial change detection
"""

from geointel.change.detection import (
    compute_spatial_change_masks,
    compute_transition_matrix,
)
from geointel.change.indices import (
    compute_all_indices,
    compute_bsi,
    compute_mndwi,
    compute_ndbi,
    compute_ndvi,
)
from geointel.change.rf_classifier import (
    extract_feature_stack,
    generate_pseudo_training_data,
    predict_rf_lulc,
    train_rf_classifier,
)
from geointel.change.rule_based import (
    LULC_CLASSES,
    classify_rule_based,
)

__all__ = [
    "compute_ndvi",
    "compute_ndbi",
    "compute_mndwi",
    "compute_bsi",
    "compute_all_indices",
    "classify_rule_based",
    "LULC_CLASSES",
    "extract_feature_stack",
    "generate_pseudo_training_data",
    "train_rf_classifier",
    "predict_rf_lulc",
    "compute_transition_matrix",
    "compute_spatial_change_masks",
]
