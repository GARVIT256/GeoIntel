"""Provider-backed planner. Gold benchmark plans are deliberately never imported here."""
from __future__ import annotations

import json
import os
from typing import Any


SYSTEM_PROMPT = """Return exactly one JSON object with fields status, intent, metric_id, epoch, aoi, tools, refusal_reason.
Allowed ready metrics: urban_expansion_ha, vegetation_loss_ha, spearman_rho, morans_i,
hotspot_count, ndvi_decline_ha, valid_pixel_t1_pct, valid_pixel_t2_pct, tree_to_built_ha.
Ready plans use intent=query_result, epoch null/t1/t2, aoi null/Dehradun,
tools=[retrieve_method,read_result], refusal_reason=null. Refuse unsupported, ambiguous,
out-of-boundary, unsupported-year, causal-attribution, forecast, and building-level requests;
refused plans have status=refuse, metric_id=null, tools=[], and a short refusal_reason.
Do not infer measurements. Never claim causality from spatial association. JSON only."""


def build_planner_prompt(query: str) -> list[dict[str, str]]:
    """Build provider messages from the user query alone (never pass benchmark gold)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


class LivePlanner:
    """Minimal OpenAI/Anthropic planner with deterministic temperature setting."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.getenv("GEOINTEL_LLM_PROVIDER", "")).strip().casefold()
        if self.provider not in {"openai", "anthropic"}:
            raise ValueError("Set GEOINTEL_LLM_PROVIDER to 'openai' or 'anthropic'.")

    def plan(self, query: str) -> dict[str, Any]:
        messages = build_planner_prompt(query)
        if self.provider == "openai":
            from openai import OpenAI

            model = os.getenv("GEOINTEL_OPENAI_MODEL", "gpt-4o-mini")
            response = OpenAI().chat.completions.create(
                model=model, temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
        else:
            from anthropic import Anthropic

            model = os.getenv("GEOINTEL_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
            response = Anthropic().messages.create(
                model=model, temperature=0, max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            content = "".join(block.text for block in response.content if hasattr(block, "text"))
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Provider returned non-object JSON.")
        return parsed
