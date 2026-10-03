"""Stratified candidate-point sampling; outputs candidates, never reference labels."""
from __future__ import annotations

from typing import Any

import numpy as np


def sample_stratified_pixels(
    class_map: np.ndarray,
    samples_per_class: int = 150,
    seed: int = 42,
    class_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> list[dict[str, int]]:
    """Select deterministic pixel coordinates without replacement per mapped class."""
    values = np.asarray(class_map)
    if values.ndim != 2:
        raise ValueError("class_map must be a two-dimensional array")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    rng = np.random.default_rng(seed)
    result: list[dict[str, int]] = []
    for class_id in class_ids:
        rows, cols = np.where(values == class_id)
        selected = rng.choice(len(rows), size=min(samples_per_class, len(rows)), replace=False)
        result.extend(
            {"row": int(rows[index]), "col": int(cols[index]), "candidate_class": int(class_id)}
            for index in selected
        )
    return result


def build_candidate_records(
    map_t1: np.ndarray,
    map_t2: np.ndarray,
    samples_per_class: int = 150,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample each epoch's predicted classes and attach predictions at both dates."""
    if map_t1.shape != map_t2.shape:
        raise ValueError("T1 and T2 class maps must share the same grid shape")
    candidates: list[dict[str, Any]] = []
    for epoch_index, (epoch, class_map) in enumerate((("t1", map_t1), ("t2", map_t2))):
        sampled = sample_stratified_pixels(
            class_map, samples_per_class=samples_per_class,
            seed=seed + epoch_index,
        )
        for item in sampled:
            row, col = item["row"], item["col"]
            candidates.append({
                "candidate_epoch": epoch,
                "candidate_class": item["candidate_class"],
                "row": row,
                "col": col,
                "pred_t1": int(map_t1[row, col]),
                "pred_t2": int(map_t2[row, col]),
            })
    return candidates
