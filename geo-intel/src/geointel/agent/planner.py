"""Deterministic mock planner for fixture benchmarks; no LLM calls are made."""
from __future__ import annotations

import re
from typing import Any


METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("spearman_rho", ("spearman", "rank correlation")),
    ("morans_i", ("moran",)),
    ("hotspot_count", ("hotspot", "hot spot")),
    ("tree_to_built_ha", ("tree to built", "tree-to-built", "tree-to-built-up")),
    ("urban_expansion_ha", ("urban expansion", "urban gain", "built-up gain")),
    ("vegetation_loss_ha", ("vegetation loss", "veg loss")),
    ("ndvi_decline_ha", ("ndvi decline", "ndvi difference")),
    ("valid_pixel_t1_pct", ("valid-pixel coverage", "valid pixel coverage")),
)


class MockPlanner:
    """Small, deterministic query parser used to test the plan/validate boundary."""

    SUPPORTED_TOOLS = ["retrieve_method", "read_result"]

    def plan(self, query: str) -> dict[str, Any]:
        text = query.casefold()
        if re.search(r"\b(cause\w*|lead to|leads to|led to|result\w* in|forecast\w*|predict\w*|will flood|flood risk|responsible for)\b", text):
            return {
                "status": "refuse", "intent": None, "metric_id": None,
                "epoch": None, "aoi": None, "tools": [],
                "refusal_reason": "Causal claims and flood forecasts are outside the supported fixture workflow.",
            }
        metric_id = next(
            (metric for metric, aliases in METRIC_ALIASES if any(alias in text for alias in aliases)),
            None,
        )
        if metric_id is None:
            return {
                "status": "refuse", "intent": None, "metric_id": None,
                "epoch": None, "aoi": None, "tools": [],
                "refusal_reason": "No supported GEO-INTEL metric was identified.",
            }
        epoch = (
            "t1" if re.search(r"\b(t1|2018[- ]?19)\b", text)
            else "t2" if re.search(r"\b(t2|2024[- ]?25)\b", text)
            else None
        )
        if epoch == "t2" and metric_id == "valid_pixel_t1_pct":
            metric_id = "valid_pixel_t2_pct"
        return {
            "status": "ready", "intent": "query_result", "metric_id": metric_id,
            "epoch": epoch, "aoi": "Dehradun" if "dehradun" in text else None,
            "tools": self.SUPPORTED_TOOLS.copy(), "refusal_reason": None,
        }
