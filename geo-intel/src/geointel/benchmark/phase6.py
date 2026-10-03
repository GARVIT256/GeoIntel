"""Ten-query fixture benchmark for mock planning, RAG, and output validation."""
from __future__ import annotations

from typing import Any

from geointel.agent.planner import MockPlanner
from geointel.agent.service import answer_with_plan
from geointel.agent.validator import validate_plan
from geointel.rag.retriever import Chunk, FixtureRetriever


QUERIES: tuple[dict[str, Any], ...] = (
    {"query": "How much urban expansion in Dehradun?", "metric_id": "urban_expansion_ha"},
    {"query": "Report vegetation loss in Dehradun.", "metric_id": "vegetation_loss_ha"},
    {"query": "What is the Spearman rho between urban gain and vegetation loss?", "metric_id": "spearman_rho"},
    {"query": "Report Moran's I for vegetation loss.", "metric_id": "morans_i"},
    {"query": "How many significant Gi star hotspots were found?", "metric_id": "hotspot_count"},
    {"query": "What is the NDVI decline area in T2?", "metric_id": "ndvi_decline_ha", "epoch": "t2"},
    {"query": "What is valid-pixel coverage for T1?", "metric_id": "valid_pixel_t1_pct", "epoch": "t1"},
    {"query": "Area converted from tree to built-up?", "metric_id": "tree_to_built_ha"},
    {"query": "What caused vegetation loss in Dehradun?", "status": "refuse"},
    {"query": "Forecast flood risk in Mussoorie.", "status": "refuse"},
)

# Synthetic values exist only to exercise numeric provenance. They are not
# GEO-INTEL measurements and must never be described as real-data results.
FIXTURE_RESULTS = {
    "urban_expansion_ha": 12.5,
    "vegetation_loss_ha": 20.0,
    "spearman_rho": 0.42,
    "morans_i": 0.31,
    "hotspot_count": 4,
    "ndvi_decline_ha": 7.3,
    "valid_pixel_t1_pct": 86.2,
    "valid_pixel_t2_pct": 82.4,
    "tree_to_built_ha": 5.8,
}

FIXTURE_CHUNKS = (
    Chunk("P5_CHANGE", "fixture:phase5", "Urban expansion is T1 non-built-up to T2 built-up. Vegetation loss is T1 tree or cropland/grass to T2 non-vegetation. Areas are hectares."),
    Chunk("P5_ASSOC", "fixture:association", "Spearman rho reports monotonic co-occurrence. Moran I measures spatial autocorrelation. Getis Ord Gi star identifies local hotspots on the 1 km grid."),
    Chunk("P5_NDVI", "fixture:ndvi", "NDVI decline is a thresholded T2 minus T1 spectral baseline and does not establish causation."),
    Chunk("P2_COVERAGE", "fixture:coverage", "Valid-pixel coverage is reported separately for T1 and T2 as percent of valid observations."),
    Chunk("P5_TREE_BUILT", "fixture:transitions", "Tree to built-up area is a class transition between the two mapped epochs."),
    Chunk("SCOPE", "fixture:scope", "Causal attribution and flood forecasting are unsupported. Report measured spatial co-occurrence only."),
)


def run_phase6_benchmark() -> dict[str, Any]:
    planner = MockPlanner()
    retriever = FixtureRetriever(FIXTURE_CHUNKS)
    rows: list[dict[str, Any]] = []
    parameter_correct = refusals_expected = refusals_correct = valid_plans = 0
    cited_supported = supported_count = provenance_pass = causal_pass = 0
    for item in QUERIES:
        result = answer_with_plan(item["query"], planner, retriever, FIXTURE_RESULTS)
        expected_refusal = item.get("status") == "refuse"
        plan = result.get("plan", {})
        valid = validate_plan(plan)["valid"]
        valid_plans += int(valid)
        correct = (
            plan.get("status") == "refuse" if expected_refusal
            else plan.get("status") == "ready" and plan.get("metric_id") == item["metric_id"]
            and ("epoch" not in item or plan.get("epoch") == item["epoch"])
        )
        parameter_correct += int(correct)
        refusals_expected += int(expected_refusal)
        refusals_correct += int(expected_refusal and result["status"] == "refused")
        if not expected_refusal:
            supported_count += 1
            cited_supported += int(bool(result.get("citations")))
        provenance_pass += int(result.get("number_provenance", {}).get("passed", False))
        causal_pass += int(result.get("causal_lint", {}).get("passed", False))
        rows.append({
            "query": item["query"], "expected_metric": item.get("metric_id"),
            "expected_refusal": expected_refusal, "planned_metric": plan.get("metric_id"),
            "status": result["status"], "plan_valid": valid,
            "has_citation": bool(result.get("citations")),
            "number_provenance_pass": result.get("number_provenance", {}).get("passed", False),
            "causal_lint_pass": result.get("causal_lint", {}).get("passed", False),
        })
    total = len(QUERIES)
    return {
        "benchmark": "phase6_mock_planner_rag_validator_fixture_v1",
        "data_status": "SYNTHETIC FIXTURES ONLY; NOT REAL GEO-INTEL RESULTS",
        "query_count": total,
        "metrics": {
            "parameter_extraction_accuracy": parameter_correct / total,
            "correct_refusal_rate": refusals_correct / refusals_expected,
            "plan_validity_rate": valid_plans / total,
            "citation_rate_supported_queries": cited_supported / supported_count,
            "number_provenance_pass_rate": provenance_pass / total,
            "causal_linter_pass_rate": causal_pass / total,
        },
        "queries": rows,
    }
