"""Grid aggregation and spatial association for change masks."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm, spearmanr


def summarize_association(
    urban_gain_ha: np.ndarray,
    vegetation_loss_ha: np.ndarray,
    weights: Any,
    *,
    alpha: float = 0.05,
    top_n: int = 20,
    permutations: int = 999,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate Spearman, global Moran's I, local Gi* and top positive hotspots.

    `weights` is a PySAL spatial weights object whose order matches the arrays.
    Queen contiguity and binary weights are the project convention.
    """
    urban = np.asarray(urban_gain_ha, dtype="float64").ravel()
    vegetation = np.asarray(vegetation_loss_ha, dtype="float64").ravel()
    if urban.shape != vegetation.shape or urban.size != len(weights.neighbors):
        raise ValueError("values and spatial weights must have equal lengths")
    valid = np.isfinite(urban) & np.isfinite(vegetation)
    if not valid.all():
        raise ValueError("grid values must be finite; use zero for cells with no change")
    if top_n < 0 or permutations < 0 or not 0 < alpha < 1:
        raise ValueError("invalid alpha, top_n, or permutations")
    rho = spearmanr(urban, vegetation) if urban.size >= 2 else None
    spearman = {
        "rho": float(rho.statistic) if rho is not None and np.isfinite(rho.statistic) else None,
        "p_value": float(rho.pvalue) if rho is not None and np.isfinite(rho.pvalue) else None,
    }
    # Build a binary queen matrix explicitly so weights-library defaults do not
    # silently change the statistic. Gi* includes the focal cell; Moran excludes it.
    n = urban.size
    queen = np.zeros((n, n), dtype="float64")
    for i, neighbors in weights.neighbors.items():
        for j in neighbors:
            queen[int(i), int(j)] = 1.0
    queen = np.maximum(queen, queen.T)
    np.fill_diagonal(queen, 0.0)
    moran_matrix = queen.copy()
    s0 = float(moran_matrix.sum())
    centered = vegetation - vegetation.mean()
    denominator = float(centered @ centered)
    moran_i = (n / s0) * float(centered @ moran_matrix @ centered) / denominator if s0 and denominator else None
    moran_p = None
    if moran_i is not None and permutations:
        rng = np.random.default_rng(seed)
        null = np.empty(permutations, dtype="float64")
        for k in range(permutations):
            shuffled = rng.permutation(vegetation)
            z = shuffled - shuffled.mean()
            d = float(z @ z)
            null[k] = (n / s0) * float(z @ moran_matrix @ z) / d if d else np.nan
        moran_p = float((1 + np.count_nonzero(np.abs(null) >= abs(moran_i))) / (permutations + 1))

    gi_weights = queen + np.eye(n, dtype="float64")
    sum_w = gi_weights.sum(axis=1)
    sum_w2 = np.square(gi_weights).sum(axis=1)
    sample_variance = float(np.square(centered).sum() / n)
    gi_numerator = gi_weights @ vegetation - vegetation.mean() * sum_w
    gi_variance = sample_variance * (n * sum_w2 - np.square(sum_w)) / (n - 1) if n > 1 else np.zeros(n)
    z_local = np.divide(gi_numerator, np.sqrt(np.maximum(gi_variance, 0)), out=np.full(n, np.nan), where=gi_variance > 0)
    # Conventional one-sided normal tail probability for positive Gi* hotspots.
    p_local = norm.sf(z_local)
    candidates = np.flatnonzero(np.isfinite(z_local) & (z_local > 0) & (p_local <= alpha))
    ranked = candidates[np.argsort(z_local[candidates])[::-1]][:top_n]
    return {
        "grid_cells": int(urban.size),
        "alpha": float(alpha),
        "permutations": int(permutations),
        "seed": int(seed),
        "neighbor_definition": "Queen contiguity, binary weights, Gi* star=True",
        "gi_star_p_method": "one-sided analytical normal approximation; positive hotspots",
        "spearman_urban_gain_vs_vegetation_loss": spearman,
        "morans_i_vegetation_loss": {
            "I": float(moran_i) if moran_i is not None else None,
            "p_sim": moran_p,
        },
        "gi_star": {
            "z_scores": z_local.tolist(),
            "permutation_p_values": p_local.tolist(),
            "significant_positive": (np.isfinite(p_local) & (p_local <= alpha) & (z_local > 0)).tolist(),
        },
        "top_n_hotspots": [
            {
                "grid_position": int(i),
                "urban_gain_ha": float(urban[i]),
                "vegetation_loss_ha": float(vegetation[i]),
                "gi_star_z": float(z_local[i]),
                "permutation_p": float(p_local[i]) if np.isfinite(p_local[i]) else None,
                "significant_at_alpha": bool(np.isfinite(p_local[i]) and p_local[i] <= alpha),
            }
            for i in ranked
        ],
    }


def aggregate_binary_masks_to_grid(
    urban_mask_path: str,
    vegetation_mask_path: str,
    grid: Any,
    pixel_area_m2: float = 100.0,
) -> Any:
    """Sum valid positive mask pixels per grid polygon and report hectares."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    from rasterio.windows import transform as window_transform

    if grid.crs is None:
        raise ValueError("grid must have a CRS")
    if pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be positive")
    rows: list[dict[str, float]] = []
    with rasterio.open(urban_mask_path) as urban, rasterio.open(vegetation_mask_path) as vegetation:
        if (urban.crs, urban.transform, urban.width, urban.height) != (
            vegetation.crs, vegetation.transform, vegetation.width, vegetation.height
        ):
            raise ValueError("urban and vegetation masks must share an identical raster grid")
        aligned = grid.to_crs(urban.crs)
        for geom in aligned.geometry:
            window = from_bounds(*geom.bounds, transform=urban.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, urban.width, urban.height))
            if window.width <= 0 or window.height <= 0:
                rows.append({"urban_gain_ha": 0.0, "vegetation_loss_ha": 0.0})
                continue
            shape = (int(window.height), int(window.width))
            transform = window_transform(window, urban.transform)
            inside = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)
            a = urban.read(1, window=window)
            b = vegetation.read(1, window=window)
            rows.append({
                "urban_gain_ha": float(np.count_nonzero(inside & (a == 1)) * pixel_area_m2 / 10000),
                "vegetation_loss_ha": float(np.count_nonzero(inside & (b == 1)) * pixel_area_m2 / 10000),
            })
    result = grid.copy()
    for key in ("urban_gain_ha", "vegetation_loss_ha"):
        result[key] = [row[key] for row in rows]
    return result


def area_adjusted_estimate(
    mapped_area_ha: np.ndarray,
    stratified_reference_counts: np.ndarray,
    *,
    confidence_z: float = 1.96,
) -> dict[str, Any]:
    """Estimate reference class areas and normal 95% CIs from stratified labels.

    Rows are mapped-class strata and columns are reference classes. Counts must
    come from a probability sample with the sample design documented.
    """
    areas = np.asarray(mapped_area_ha, dtype="float64")
    counts = np.asarray(stratified_reference_counts, dtype="float64")
    if counts.ndim != 2 or areas.ndim != 1 or counts.shape[0] != areas.size:
        raise ValueError("counts rows must match mapped class strata")
    if np.any(areas < 0) or np.any(counts < 0) or np.any(counts.sum(axis=1) == 0):
        raise ValueError("areas/counts must be nonnegative with sampled rows in each stratum")
    total = float(areas.sum())
    proportions = counts / counts.sum(axis=1, keepdims=True)
    weights = areas / total if total else np.zeros_like(areas)
    class_proportions = (weights[:, None] * proportions).sum(axis=0)
    variance = np.zeros(counts.shape[1], dtype="float64")
    for row, n in enumerate(counts.sum(axis=1)):
        if n > 1:
            variance += weights[row] ** 2 * proportions[row] * (1 - proportions[row]) / (n - 1)
    est = class_proportions * total
    margin = confidence_z * np.sqrt(variance) * total
    return {
        "method": "stratified mapped-class area adjustment; normal interval",
        "total_mapped_area_ha": total,
        "reference_class_area_ha": est.tolist(),
        "standard_error_ha": (np.sqrt(variance) * total).tolist(),
        "confidence_interval_95_ha": np.column_stack((np.maximum(0, est - margin), np.minimum(total, est + margin))).tolist(),
        "requires_probability_sample": True,
    }


def area_adjusted_estimate_from_labels(
    prediction_path: str,
    labels_path: str,
    epoch: str,
    labeller: str,
    pixel_area_m2: float = 100.0,
) -> dict[str, Any]:
    """Build a mapped-stratum/reference-class table from completed label CSV rows."""
    import pandas as pd
    import rasterio
    from pyproj import Transformer

    if epoch not in {"t1", "t2"}:
        raise ValueError("epoch must be t1 or t2")
    labels = pd.read_csv(labels_path)
    required = {"lon", "lat", f"class_{epoch}", "labeller", "candidate_epoch"}
    if not required.issubset(labels.columns):
        raise ValueError(f"label file needs columns {sorted(required)}")
    labels = labels.loc[
        (labels["labeller"].astype(str) == str(labeller))
        & (labels["candidate_epoch"].astype(str) == epoch)
    ].dropna(
        subset=["lon", "lat", f"class_{epoch}"]
    )
    if labels.empty:
        raise ValueError(f"no completed labels for {epoch} by {labeller}")
    if "id" in labels and labels["id"].duplicated().any():
        raise ValueError("area adjustment requires one row per candidate ID and labeller")
    with rasterio.open(prediction_path) as source:
        if source.crs is None:
            raise ValueError("prediction map has no CRS")
        transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
        xs, ys = transformer.transform(labels["lon"].to_numpy(), labels["lat"].to_numpy())
        mapped = np.array([sample[0] for sample in source.sample(zip(xs, ys, strict=True))])
        ref = pd.to_numeric(labels[f"class_{epoch}"], errors="raise").to_numpy(dtype=int)
        if not np.isin(ref, [1, 2, 3, 4, 5]).all():
            raise ValueError("reference classes must be IDs 1..5")
        accepted = np.isin(mapped, [1, 2, 3, 4, 5])
        mapped, ref = mapped[accepted].astype(int), ref[accepted]
        if mapped.size == 0:
            raise ValueError("no labels fall on valid mapped classes")
        area_by_class = np.zeros(5, dtype="float64")
        pixel_counts = np.bincount(source.read(1).ravel().astype(int), minlength=6)
        area_by_class[:] = pixel_counts[1:6] * pixel_area_m2 / 10000
    counts = np.zeros((5, 5), dtype="int64")
    np.add.at(counts, (mapped - 1, ref - 1), 1)
    estimate = area_adjusted_estimate(area_by_class, counts)
    estimate.update({"epoch": epoch, "labeller": labeller, "label_count_used": int(mapped.size)})
    return estimate
