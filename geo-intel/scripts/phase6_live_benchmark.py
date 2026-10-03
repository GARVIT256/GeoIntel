"""Explicit opt-in benchmark of a live LLM planner against separate gold plans."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from geointel.agent.live_planner import LivePlanner
from geointel.agent.validator import validate_plan


ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "benchmarks" / "phase6" / "queries.json"
GOLD_FILE = ROOT / "benchmarks" / "phase6" / "gold_plans.json"
FIELDS = ("status", "intent", "metric_id", "epoch", "aoi", "tools")


def run(*, execute_live: bool = False) -> dict[str, Any]:
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    if len(queries) != 40 or {q["id"] for q in queries} != set(gold):
        raise ValueError("Expected 40 uniquely keyed queries with one separate gold plan each.")
    provider = os.getenv("GEOINTEL_LLM_PROVIDER", "not configured")
    if not execute_live:
        return {
            "status": "NOT YET RUN",
            "measurement": "NOT YET MEASURED",
            "provider": provider,
            "temperature": 0,
            "query_count": len(queries),
            "message": "Re-run with --execute-live and provider credentials to make API calls.",
        }
    planner = LivePlanner(provider)
    rows: list[dict[str, Any]] = []
    for item in queries:
        try:
            plan = planner.plan(item["query"])
            expected = gold[item["id"]]
            rows.append({
                "id": item["id"],
                "plan_valid": validate_plan(plan)["valid"],
                "field_matches": {field: plan.get(field) == expected.get(field) for field in FIELDS},
                "refusal_shape_matches": (
                    (plan.get("status") == "refuse")
                    == (expected.get("status") == "refuse")
                ),
                "error": None,
            })
        except Exception as exc:  # preserve per-query failures in benchmark output
            rows.append({"id": item["id"], "plan_valid": False, "field_matches": {},
                         "refusal_shape_matches": False, "error": f"{type(exc).__name__}: {exc}"})
    denominator = len(rows)
    scored = sum(
        bool(row["field_matches"]) and all(row["field_matches"].values())
        for row in rows
    )
    field_rates = {
        field: sum(bool(row.get("field_matches", {}).get(field)) for row in rows) / denominator
        for field in FIELDS
    }
    return {
        "status": "RUN",
        "measurement": "LIVE PLANNER SCORED AGAINST SEPARATE GOLD PLANS; NOT AN ACCURACY CLAIM",
        "provider": planner.provider,
        "temperature": 0,
        "query_count": denominator,
        "plan_field_match_rate": scored / denominator,
        "per_field_match_rate": field_rates,
        "valid_plan_rate": sum(bool(row["plan_valid"]) for row in rows) / denominator,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-live", action="store_true", help="Explicitly call the configured LLM provider")
    parser.add_argument("--output", type=Path, help="Optional path for JSON output")
    args = parser.parse_args()
    result = run(execute_live=args.execute_live)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
