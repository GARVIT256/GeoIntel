"""Validated fixture answer flow: mock plan, retrieve, read results, lint."""
from __future__ import annotations

from typing import Any

from geointel.agent.validator import check_number_provenance, lint_causal_language, validate_plan
from geointel.rag.retriever import FixtureRetriever


def _lookup_metric(results: Any, metric_id: str) -> Any:
    if isinstance(results, dict):
        if metric_id in results:
            return results[metric_id]
        for value in results.values():
            found = _lookup_metric(value, metric_id)
            if found is not None:
                return found
    return None


def answer_with_plan(query: str, planner: Any, retriever: FixtureRetriever, results: Any) -> dict[str, Any]:
    plan = planner.plan(query)
    validation = validate_plan(plan)
    if not validation["valid"]:
        return {"status": "invalid_plan", "validation": validation, "plan": plan}
    if plan["status"] == "refuse":
        narrative = f"Request declined: {plan['refusal_reason']}"
        return {
            "status": "refused", "plan": plan, "narrative": narrative,
            "citations": [], "causal_lint": lint_causal_language(narrative),
            "number_provenance": check_number_provenance(narrative, results),
        }
    chunks = retriever.retrieve(query, top_k=3)
    metric_id = plan["metric_id"]
    value = _lookup_metric(results, metric_id)
    if value is None:
        narrative = f"No recorded result is available for {metric_id}."
    else:
        narrative = f"Recorded {metric_id}: {value}."
    citations = [chunk["chunk_id"] for chunk in chunks]
    if citations:
        narrative += " Evidence: " + ", ".join(f"[{chunk_id}]" for chunk_id in citations) + "."
    return {
        "status": "answered" if value is not None else "result_missing",
        "plan": plan,
        "validation": validation,
        "narrative": narrative,
        "citations": citations,
        "retrieval": chunks,
        "causal_lint": lint_causal_language(narrative),
        "number_provenance": check_number_provenance(narrative, results),
    }
