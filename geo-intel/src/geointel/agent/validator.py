"""Plan, output-number, and causal-language validators."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


ALLOWED_METRICS = {
    "urban_expansion_ha", "vegetation_loss_ha", "spearman_rho", "morans_i",
    "hotspot_count", "ndvi_decline_ha", "valid_pixel_t1_pct", "valid_pixel_t2_pct",
    "tree_to_built_ha",
}
ALLOWED_TOOLS = {"retrieve_method", "read_result"}
DEFAULT_CAUSAL_TERMS = (
    "cause", "caused", "causes", "causing", "led to", "lead to", "leads to",
    "resulted in", "results in", "will flood", "flood risk", "responsible for",
    "drove", "drives", "triggered", "triggers", "due to", "because of",
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def validate_plan(plan: Any) -> dict[str, Any]:
    """Validate the narrow JSON plan contract; never executes arbitrary tools."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return {"valid": False, "errors": ["plan must be a JSON object"]}
    required = {"status", "intent", "metric_id", "epoch", "aoi", "tools", "refusal_reason"}
    missing = required - plan.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if plan.get("status") == "ready":
        if plan.get("intent") != "query_result":
            errors.append("ready plans must use query_result intent")
        if plan.get("metric_id") not in ALLOWED_METRICS:
            errors.append("ready plan metric_id is not allowlisted")
        if plan.get("epoch") not in {None, "t1", "t2"}:
            errors.append("epoch must be null, t1, or t2")
        if plan.get("aoi") not in {None, "Dehradun"}:
            errors.append("AOI is not allowlisted")
        tools = plan.get("tools")
        if not isinstance(tools, list) or set(tools) != ALLOWED_TOOLS:
            errors.append("ready plan must request exactly the allowlisted retrieval/result tools")
        if plan.get("refusal_reason") is not None:
            errors.append("ready plan must not contain a refusal reason")
    elif plan.get("status") == "refuse":
        if not isinstance(plan.get("refusal_reason"), str) or not plan["refusal_reason"].strip():
            errors.append("refused plan needs a reason")
        if plan.get("tools") != [] or plan.get("metric_id") is not None:
            errors.append("refused plan must not request tools or metrics")
    else:
        errors.append("status must be ready or refuse")
    return {"valid": not errors, "errors": errors}


def _number_key(value: str | int | float | Decimal) -> str | None:
    text = str(value).replace(",", "").removesuffix("%")
    try:
        return format(Decimal(text).normalize(), "f")
    except (InvalidOperation, ValueError):
        return None


def _unit_for_path(path: str) -> str | None:
    key = path.rsplit(".", 1)[-1].casefold()
    if key.endswith("_ha") or key.endswith("_hectares"):
        return "ha"
    if key.endswith("_km2") or key.endswith("_km²"):
        return "km2"
    if key.endswith(("_pct", "_percent")):
        return "percent"
    if key.endswith("_fraction"):
        return "fraction"
    if key.endswith("_year") or key == "year":
        return "year"
    return None


def _result_number_sources(value: Any, path: str = "$") -> dict[str, list[dict[str, str | None]]]:
    numbers: dict[str, list[dict[str, str | None]]] = {}
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float, Decimal)):
        key = _number_key(value)
        if key is not None:
            numbers.setdefault(key, []).append({"path": path, "unit": _unit_for_path(path)})
    elif isinstance(value, dict):
        for name, child in value.items():
            nested = _result_number_sources(child, f"{path}.{name}")
            for key, locations in nested.items():
                numbers.setdefault(key, []).extend(locations)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            nested = _result_number_sources(child, f"{path}[{index}]")
            for key, locations in nested.items():
                numbers.setdefault(key, []).extend(locations)
    return numbers


def check_number_provenance(narrative: str, results: Any) -> dict[str, Any]:
    """Require exact numeric and explicit-unit matches in generated text."""
    sources = _result_number_sources(results)
    matches = list(NUMBER_PATTERN.finditer(narrative))
    found = [match.group(0) for match in matches]
    provenance: dict[str, list[str]] = {}
    failures: list[dict[str, str]] = []
    for match, token in zip(matches, found):
        srcs = sources.get(_number_key(token) or "", [])
        provenance[token] = [str(source["path"]) for source in srcs]
        following = narrative[match.end():]
        unit_match = re.match(r"\s*(%|percent\b|ha\b|hectares?\b|km²|km2\b|km\^2\b)", following, re.I)
        stated_unit = "percent" if token.endswith("%") else None
        if unit_match:
            raw = unit_match.group(1).casefold()
            stated_unit = "percent" if raw in {"%", "percent"} else "ha" if raw.startswith(("ha", "hectare")) else "km2"
        reason = None
        if not srcs:
            reason = "numeric_value_not_in_results"
        elif stated_unit and all(source["unit"] not in {None, stated_unit} for source in srcs):
            reason = "unit_mismatch"
        elif stated_unit and any(source["unit"] is None for source in srcs):
            reason = "unit_not_declared_in_results"
        if reason:
            failures.append({"token": token, "reason": reason})
    unprovenanced = [entry["token"] for entry in failures]
    return {
        "passed": not failures,
        "numbers_found": found,
        "unprovenanced_numbers": unprovenanced,
        "failures": failures,
        "provenance_paths": provenance,
    }


def lint_causal_language(narrative: str, terms: tuple[str, ...] = DEFAULT_CAUSAL_TERMS) -> dict[str, Any]:
    """Flag blacklisted causal phrases in generated narrative text."""
    matches = [term for term in terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", narrative, re.I)]
    return {"passed": not matches, "matches": matches}
