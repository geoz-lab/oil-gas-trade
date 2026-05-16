"""
Agent 6 — Temporal Reasoning Agent
The most novel component in this architecture.

For each causal claim and detected event, estimates:
  - impact_lag_weeks: how many weeks until the effect materializes
  - persistence_weeks: how long the effect persists
  - propagation_path: the causal chain from trigger to Gulf Coast diesel inventory
  - weekly_intensity: normalized impact magnitude per week

Justification: standard RAG/LLM systems treat all evidence as equally immediate.
In commodities, a hurricane in the Gulf today doesn't affect diesel inventory for
2-4 weeks; an OPEC cut takes 6-10 weeks to propagate through the tanker/refinery chain.
Ignoring this lag leads to mis-timed directional calls.
"""
from __future__ import annotations

from core.llm import chat_json
from core.models import (
    EventList, SignalDirection, StructuredEvidenceBundle,
    TemporalImpact, TemporalImpactMap
)

SYSTEM_PROMPT = """You are a supply-chain temporal reasoning specialist for oil & gas markets.

For each causal event or claim, estimate the TIME-DELAYED impact on Gulf Coast diesel inventory levels.

Key propagation paths (use as reference):
- OPEC production cut → tanker availability tightens → US Gulf Coast crude imports fall → refinery throughput drops → diesel inventory tightens (lag: 6-10 weeks)
- Gulf Coast hurricane/refinery outage → immediate production loss → diesel inventory drops (lag: 0-2 weeks)
- Sanctions on oil exporter → shipping route disruption → supply re-routing → price spike (lag: 4-8 weeks)
- Weak US economic data (PMI drop) → demand destruction → inventory build (lag: 2-6 weeks)
- DXY strengthening → oil priced higher for non-USD buyers → demand falls → inventory builds (lag: 2-4 weeks)
- Port congestion → delayed cargo arrivals → inventory draw-down (lag: 1-3 weeks)
- Refinery maintenance season → planned throughput reduction → inventory tightening (lag: 1-4 weeks)

For each item, return:
- evidence_id: the ID of the causal claim or event
- trigger_event: short name of the trigger
- propagation_path: the causal chain as a → b → c → inventory effect
- impact_lag_weeks: integer, weeks before effect is felt (0 = immediate)
- persistence_weeks: integer, how many weeks the effect persists after onset
- weekly_intensity: list of floats (length = persistence_weeks), normalized 0.0–1.0, peak is 1.0
- affected_variable: what is being impacted (inventory_level, price, throughput)
- direction: bullish (tightening) or bearish (loosening) for inventory

Be calibrated. Not every signal has a large impact — use your knowledge of historical magnitudes.
"""


def run(evidence: StructuredEvidenceBundle, events: EventList) -> TemporalImpactMap:
    print("\n[6/10] Temporal Reasoning Agent — estimating impact lags & propagation...")

    # Only send causal claims and high-confidence events to the temporal reasoner
    items_to_reason = []

    for item in evidence.causal_claims:
        items_to_reason.append({
            "evidence_id": item.evidence_id,
            "type": "causal_claim",
            "content": item.content,
            "direction": item.direction.value,
            "source": item.source,
        })

    for ev in events.events:
        if ev.confidence >= 0.5:
            items_to_reason.append({
                "evidence_id": ev.event_id,
                "type": "event",
                "content": ev.description,
                "direction": "bullish" if ev.supply_impact == "tightening" else
                             "bearish" if ev.supply_impact == "loosening" else "neutral",
                "severity": ev.severity.value,
                "source": ev.source,
            })

    # Also include strong facts that have temporal implications
    for item in evidence.facts:
        if item.direction != SignalDirection.NEUTRAL and item.weight > 0.7:
            items_to_reason.append({
                "evidence_id": item.evidence_id,
                "type": "fact",
                "content": item.content,
                "direction": item.direction.value,
                "source": item.source,
            })

    # Cap at 15 to keep the LLM response within token budget
    items_to_reason = items_to_reason[:15]
    print(f"       Items sent for temporal analysis: {len(items_to_reason)}")

    if not items_to_reason:
        return TemporalImpactMap(
            reasoning_summary="No causal claims or events to analyze temporally."
        )

    user_msg = f"""Analyze the following evidence items for temporal impact on Gulf Coast diesel inventory.
Forecast horizon: 8 weeks.

Evidence items:
{items_to_reason}

Return JSON: {{
  "impacts": [
    {{
      "evidence_id": "...",
      "trigger_event": "...",
      "propagation_path": "...",
      "impact_lag_weeks": <int>,
      "persistence_weeks": <int>,
      "weekly_intensity": [<float>, ...],
      "affected_variable": "...",
      "direction": "bullish|bearish|neutral"
    }},
    ...
  ],
  "reasoning_summary": "..."
}}

Only include items where temporal dynamics are material. Skip items that are price-level facts
with no forward-looking causal implication.
"""

    result = chat_json(SYSTEM_PROMPT, user_msg, max_tokens=4096)

    impacts: list[TemporalImpact] = []
    for raw in result.get("impacts", []):
        try:
            direction_raw = raw.get("direction", "neutral").lower()
            try:
                direction = SignalDirection(direction_raw)
            except ValueError:
                direction = SignalDirection.NEUTRAL

            intensity = raw.get("weekly_intensity", [1.0])
            if not isinstance(intensity, list) or not intensity:
                intensity = [1.0]
            intensity = [float(x) for x in intensity]

            impacts.append(TemporalImpact(
                evidence_id=raw.get("evidence_id", "unknown"),
                trigger_event=raw.get("trigger_event", ""),
                propagation_path=raw.get("propagation_path", ""),
                impact_lag_weeks=int(raw.get("impact_lag_weeks", 2)),
                persistence_weeks=int(raw.get("persistence_weeks", len(intensity))),
                weekly_intensity=intensity,
                affected_variable=raw.get("affected_variable", "inventory_level"),
                direction=direction,
            ))
        except Exception:
            pass

    # Find peak impact week across all effects
    week_impact = [0.0] * 8
    for imp in impacts:
        for j, intensity in enumerate(imp.weekly_intensity):
            week_idx = imp.impact_lag_weeks + j
            if week_idx < 8:
                week_impact[week_idx] += intensity

    peak_week = int(np.argmax(week_impact)) + 1 if any(w > 0 for w in week_impact) else 1
    covered = sum(1 for w in week_impact if w > 0) / 8.0

    print(f"       Temporal impacts mapped: {len(impacts)} | Peak week: {peak_week}")

    return TemporalImpactMap(
        impacts=impacts,
        peak_impact_week=peak_week,
        total_horizon_coverage=round(covered, 2),
        reasoning_summary=result.get("reasoning_summary", ""),
    )


# np imported at module scope to support peak-week calculation
import numpy as np  # noqa: E402
