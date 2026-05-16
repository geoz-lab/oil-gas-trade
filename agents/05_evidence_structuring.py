"""
Agent 5 — Evidence Structuring Agent
Classifies all signals and events into an epistemic taxonomy:
  FACT        — verifiable, sourced, dated data
  CAUSAL_CLAIM — argued connection between events (may have delay)
  SPECULATION  — opinion, analyst view, possible scenario
  FORECAST     — numeric prediction from an external source

This is a key anti-hallucination component. Only FACT-class evidence
gets full weight in the debate layer. SPECULATION is quarantined.
"""
from __future__ import annotations
import uuid

from core.llm import chat_json
from core.models import (
    EvidenceClass, EvidenceItem, EventList,
    SignalBundle, SignalDirection, StructuredEvidenceBundle
)

SYSTEM_PROMPT = """You are an evidence classification analyst for an oil & gas forecasting system.

Your task: classify each piece of evidence into exactly one of four categories:

- FACT: verifiable, sourced, dated — e.g. "Brent crude closed at $82.4 on 2025-05-01"
- CAUSAL_CLAIM: an argued connection between two facts — e.g. "OPEC cut will tighten supply"
- SPECULATION: opinion, scenario, or prediction without strong empirical grounding
- FORECAST: a numeric prediction attributed to a named external source or model

Rules:
1. Do not upgrade SPECULATION to FACT — err on the side of downgrading.
2. Statistical price data and EIA inventory data are always FACT.
3. Analyst price targets are FORECAST, not FACT.
4. "Could", "may", "might", "could lead to" → SPECULATION.
5. "X caused Y" → CAUSAL_CLAIM.

For each item output:
- evidence_id (string)
- classification: fact | causal_claim | speculation | forecast
- source (string)
- content: the core claim in one sentence
- direction: bullish | bearish | neutral | unknown
- weight: 0.0–1.0 (fact=0.85–1.0, causal_claim=0.55–0.75, speculation=0.2–0.45, forecast=0.5–0.7)
- supporting_data: any numeric evidence (can be empty string)
- contradicts: list of evidence_ids this contradicts (empty if none)
"""


def _collect_all_evidence_text(bundle: SignalBundle, events: EventList) -> list[dict]:
    items = []

    # Market & macro signals
    for sig in bundle.all_signals():
        items.append({
            "id": sig.signal_id,
            "source": sig.source,
            "text": sig.raw_text,
            "value": sig.value,
            "direction": sig.direction.value,
        })

    # Detected events
    for ev in events.events:
        items.append({
            "id": ev.event_id,
            "source": ev.source,
            "text": f"[Event] {ev.description} | Region: {ev.affected_region} | Impact: {ev.supply_impact}",
            "direction": "bullish" if ev.supply_impact == "tightening" else
                         "bearish" if ev.supply_impact == "loosening" else "neutral",
        })

    return items


def run(bundle: SignalBundle, events: EventList) -> StructuredEvidenceBundle:
    print("\n[5/10] Evidence Structuring Agent — classifying evidence...")

    raw_items = _collect_all_evidence_text(bundle, events)
    print(f"       Evidence items to classify: {len(raw_items)}")

    # Batch in chunks of 15 to stay within token budget
    chunk_size = 15
    all_classified: list[EvidenceItem] = []

    for i in range(0, len(raw_items), chunk_size):
        chunk = raw_items[i : i + chunk_size]
        user_msg = f"""Classify each of the following evidence items.

Evidence items:
{chunk}

Return JSON: {{"items": [{{evidence_id, classification, source, content, direction, weight, supporting_data, contradicts}}, ...]}}
"""
        result = chat_json(SYSTEM_PROMPT, user_msg, max_tokens=2048)
        for item in result.get("items", []):
            try:
                cls_raw = item.get("classification", "speculation").lower().replace(" ", "_")
                try:
                    cls = EvidenceClass(cls_raw)
                except ValueError:
                    cls = EvidenceClass.SPECULATION

                dir_raw = item.get("direction", "unknown").lower()
                try:
                    direction = SignalDirection(dir_raw)
                except ValueError:
                    direction = SignalDirection.UNKNOWN

                all_classified.append(EvidenceItem(
                    evidence_id=item.get("evidence_id", str(uuid.uuid4())[:8]),
                    classification=cls,
                    source=item.get("source", "unknown"),
                    content=item.get("content", ""),
                    direction=direction,
                    weight=float(item.get("weight", 0.5)),
                    supporting_data=item.get("supporting_data", ""),
                    contradicts=item.get("contradicts", []),
                ))
            except Exception:
                pass

    facts = [e for e in all_classified if e.classification == EvidenceClass.FACT]
    causal = [e for e in all_classified if e.classification == EvidenceClass.CAUSAL_CLAIM]
    specs = [e for e in all_classified if e.classification == EvidenceClass.SPECULATION]
    forecasts = [e for e in all_classified if e.classification == EvidenceClass.FORECAST]

    # Identify contradictions
    contradiction_count = sum(1 for e in all_classified if e.contradicts)

    # Determine dominant narrative
    bull_weight = sum(e.weight for e in all_classified if e.direction == SignalDirection.BULLISH)
    bear_weight = sum(e.weight for e in all_classified if e.direction == SignalDirection.BEARISH)
    if bull_weight > bear_weight * 1.3:
        narrative = "bullish supply tightening"
    elif bear_weight > bull_weight * 1.3:
        narrative = "bearish demand weakness"
    else:
        narrative = "mixed / contested"

    print(f"       Facts={len(facts)} Causal={len(causal)} Spec={len(specs)} Forecast={len(forecasts)}")
    print(f"       Dominant narrative: {narrative}")

    return StructuredEvidenceBundle(
        facts=facts,
        causal_claims=causal,
        speculations=specs,
        forecasts=forecasts,
        contradiction_count=contradiction_count,
        dominant_narrative=narrative,
    )
