"""Offline fixture tests for Phase 6 planning, RAG, and safety validators."""
from __future__ import annotations

from geointel.agent.planner import MockPlanner
from geointel.agent.service import answer_with_plan
from geointel.agent.validator import check_number_provenance, lint_causal_language, validate_plan
from geointel.benchmark.phase6 import run_phase6_benchmark
from geointel.rag.retriever import Chunk, FixtureRetriever


def test_mock_planner_extracts_metric_precedence_and_epoch() -> None:
    plan = MockPlanner().plan("What is Spearman rho for T2 urban gain and vegetation loss?")
    assert plan["metric_id"] == "spearman_rho"
    assert plan["epoch"] == "t2"
    assert validate_plan(plan)["valid"]


def test_mock_planner_refuses_causal_and_out_of_scope_queries() -> None:
    planner = MockPlanner()
    assert planner.plan("What caused urban expansion?")["status"] == "refuse"
    assert planner.plan("Forecast flood risk")["status"] == "refuse"


def test_plan_validator_rejects_arbitrary_tools() -> None:
    plan = MockPlanner().plan("urban expansion area")
    plan["tools"] = ["run_python"]
    assert not validate_plan(plan)["valid"]


def test_number_provenance_flags_values_absent_from_results() -> None:
    passed = check_number_provenance("Measured value is 12.5.", {"metric": 12.5})
    assert passed["passed"]
    assert passed["provenance_paths"]["12.5"] == ["$.metric"]
    check = check_number_provenance("Measured value is 12.6.", {"metric": 12.5})
    assert not check["passed"]
    assert check["unprovenanced_numbers"] == ["12.6"]


def test_causal_linter_flags_blacklisted_phrases_case_insensitively() -> None:
    assert not lint_causal_language("Urban expansion CAUSED vegetation loss.")["passed"]
    assert lint_causal_language("Urban gain co-occurs with vegetation loss.")["passed"]


def test_rag_answer_cites_retrieved_fixture_chunk_and_checks_number() -> None:
    retriever = FixtureRetriever([Chunk("METHOD_A", "fixture", "Urban expansion area is measured in hectares.")])
    result = answer_with_plan("urban expansion area", MockPlanner(), retriever, {"urban_expansion_ha": 12.5})
    assert result["status"] == "answered"
    assert result["citations"] == ["METHOD_A"]
    assert result["number_provenance"]["passed"]
    assert result["causal_lint"]["passed"]


def test_ten_query_fixture_benchmark_passes_all_declared_checks() -> None:
    result = run_phase6_benchmark()
    assert result["query_count"] == 10
    assert result["data_status"].startswith("SYNTHETIC FIXTURES ONLY")
    assert result["metrics"]["parameter_extraction_accuracy"] == 1.0
    assert result["metrics"]["correct_refusal_rate"] == 1.0
    assert result["metrics"]["plan_validity_rate"] == 1.0
    assert result["metrics"]["citation_rate_supported_queries"] == 1.0
    assert result["metrics"]["number_provenance_pass_rate"] == 1.0
    assert result["metrics"]["causal_linter_pass_rate"] == 1.0
