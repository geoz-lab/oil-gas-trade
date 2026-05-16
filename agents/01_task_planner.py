"""
Agent 1 — Task Planner
Decomposes a raw user query into a structured ForecastTask.
Justification: explicit decomposition prevents downstream agents from
misinterpreting scope, commodity, region, or time-horizon.
"""
from __future__ import annotations
import json

from core.llm import chat_json
from core.models import ForecastTask

SYSTEM_PROMPT = """You are the Task Planner for an oil & gas supply-chain forecasting system.

Your role: decompose a natural-language user query into a precise, structured forecast specification.

Extract:
- commodity: the primary commodity (e.g. "diesel", "brent_crude", "WTI", "heating_oil", "LNG")
- region: geographic scope (e.g. "Gulf Coast", "US", "global", "PADD 3")
- horizon_weeks: integer number of weeks to forecast (default 8 if not specified)
- forecast_variable: what is being forecast ("inventory_level", "price", "demand", "supply")
- sub_tasks: list of 3-6 specific analytical tasks needed to answer the query
- context_notes: any important constraints, caveats, or special considerations from the query

Be precise. If the query is ambiguous, make the most operationally useful interpretation.
"""

USER_TEMPLATE = """User query: {query}

Return a JSON object with keys:
- commodity (string)
- region (string)
- horizon_weeks (integer)
- forecast_variable (string)
- sub_tasks (list of strings)
- context_notes (string)
"""


def run(query: str) -> ForecastTask:
    print("\n[1/10] Task Planner — decomposing query...")
    result = chat_json(
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(query=query),
        max_tokens=1024,
    )
    return ForecastTask(
        raw_query=query,
        commodity=result.get("commodity", "diesel"),
        region=result.get("region", "Gulf Coast"),
        horizon_weeks=int(result.get("horizon_weeks", 8)),
        forecast_variable=result.get("forecast_variable", "inventory_level"),
        sub_tasks=result.get("sub_tasks", []),
        context_notes=result.get("context_notes", ""),
    )
