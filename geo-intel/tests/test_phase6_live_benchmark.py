"""Tests for benchmark data separation and the no-network default."""
from __future__ import annotations

import json
from pathlib import Path

from geointel.agent.live_planner import build_planner_prompt
from scripts.phase6_live_benchmark import run


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_has_40_queries_and_separate_gold_plans() -> None:
    queries = json.loads((ROOT / "benchmarks/phase6/queries.json").read_text(encoding="utf-8"))
    gold = json.loads((ROOT / "benchmarks/phase6/gold_plans.json").read_text(encoding="utf-8"))
    assert len(queries) == 40
    assert len({item["id"] for item in queries}) == 40
    assert {item["id"] for item in queries} == set(gold)


def test_planner_prompt_contains_query_but_no_gold_plan() -> None:
    prompt = build_planner_prompt("What is urban expansion?")
    rendered = json.dumps(prompt)
    assert "What is urban expansion?" in rendered
    assert "Q01" not in rendered
    assert "refusal_reason" in rendered  # schema instructions only
    assert "urban_expansion_ha\"" not in rendered  # no gold plan object/value


def test_live_benchmark_default_does_not_call_provider() -> None:
    result = run(execute_live=False)
    assert result["status"] == "NOT YET RUN"
    assert result["measurement"] == "NOT YET MEASURED"
    assert result["query_count"] == 40
