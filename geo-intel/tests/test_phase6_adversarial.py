"""Adversarial number-provenance and causal-language regression cases."""
from __future__ import annotations

from geointel.agent.validator import check_number_provenance, lint_causal_language


def test_number_provenance_rejects_rounding_not_present_in_results() -> None:
    result = check_number_provenance("Area was 12.35 ha.", {"area_ha": 12.345})
    assert not result["passed"]
    assert result["failures"] == [{"token": "12.35", "reason": "numeric_value_not_in_results"}]


def test_number_provenance_rejects_hectares_mislabeled_as_square_kilometres() -> None:
    result = check_number_provenance("Area was 1.2 km².", {"area_ha": 1.2})
    assert not result["passed"]
    assert result["failures"] == [{"token": "1.2", "reason": "unit_mismatch"}]


def test_number_provenance_rejects_percent_that_is_only_a_fraction() -> None:
    result = check_number_provenance("Coverage was 85%.", {"coverage_fraction": 0.85})
    assert not result["passed"]


def test_number_provenance_accepts_declared_percent_and_rejects_unsupported_year() -> None:
    assert check_number_provenance("Coverage was 85%.", {"coverage_pct": 85})["passed"]
    wrong_year = check_number_provenance("The run year was 2024.", {"run_year": 2025})
    assert not wrong_year["passed"]
    assert wrong_year["failures"][0]["reason"] == "numeric_value_not_in_results"
    assert check_number_provenance("The run year was 2025.", {"run_year": 2025})["passed"]


def test_causal_linter_rejects_adversarial_phrases() -> None:
    for phrase in ("due to", "because of", "responsible for"):
        result = lint_causal_language(f"Urban loss was {phrase} development.")
        assert not result["passed"]
        assert phrase in result["matches"]


def test_causal_linter_allows_spatial_cooccurrence_statement() -> None:
    assert lint_causal_language("Urban gain co-occurs with vegetation loss.")["passed"]
