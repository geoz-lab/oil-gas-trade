"""
Agent 4 — Event Detection Agent
Detects supply/demand disruption events from two sources:
  (a) Statistical anomaly detection on price series (rolling Z-score)
  (b) LLM-based extraction of named events from textual signals

Justification: statistical anomalies ground the LLM in real market movements;
LLM extraction gives semantic context (OPEC cut vs. refinery fire vs. hurricane).
"""
from __future__ import annotations
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from core.llm import chat_json
from core.models import (
    DetectedEvent, EventList, EventSeverity, SignalBundle
)

SYSTEM_PROMPT = """You are an energy market event detection specialist.

Given a set of market signals and news headlines, identify concrete supply or demand disruption events.

For each event, extract:
- event_type: one of [refinery_outage, pipeline_disruption, weather_event, sanction,
                      OPEC_decision, port_congestion, demand_shock, geopolitical_event,
                      shipping_disruption, regulatory_change]
- description: 1-2 sentence factual description
- detected_date: ISO date (YYYY-MM-DD) — today if not specified
- affected_region: geographic region most impacted
- severity: one of [low, medium, high, critical]
- supply_impact: one of [tightening, loosening, neutral]
- source: which signal source mentioned this
- confidence: 0.0 to 1.0

Focus on events that would materially affect Gulf Coast diesel inventory or crude supply chains.
Return ONLY events with real evidence — do not fabricate events.
"""


def _detect_statistical_anomalies(bundle: SignalBundle) -> list[dict]:
    """Rolling Z-score on Brent and HO=F to flag unusual price moves."""
    anomalies = []
    tickers = {"Brent": "BZ=F", "Diesel (HO)": "HO=F"}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
            if df.empty:
                continue
            close = df["Close"].squeeze().dropna()
            roll_mean = close.rolling(20).mean()
            roll_std = close.rolling(20).std()
            zscore = (close - roll_mean) / roll_std
            # Flag last 10 days for anomalies
            recent_z = zscore.iloc[-10:]
            for dt, z in recent_z.items():
                if abs(z) > 2.0:
                    anomalies.append({
                        "series": name,
                        "date": str(dt.date()),
                        "zscore": round(float(z), 2),
                        "price": round(float(close.loc[dt]), 4),
                        "direction": "spike" if z > 0 else "crash",
                    })
        except Exception:
            pass
    return anomalies


def _build_text_context(bundle: SignalBundle) -> str:
    parts = []
    for sig in bundle.textual_signals[:8]:
        parts.append(f"[{sig.source}] {sig.raw_text[:300]}")
    for sig in bundle.market_signals:
        parts.append(f"[Market] {sig.raw_text}")
    return "\n".join(parts)


def run(bundle: SignalBundle) -> EventList:
    print("\n[4/10] Event Detection Agent — scanning for disruption events...")

    anomalies = _detect_statistical_anomalies(bundle)
    print(f"       Statistical anomalies detected: {len(anomalies)}")

    text_context = _build_text_context(bundle)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    user_msg = f"""Today's date: {today}

Market and news signals:
{text_context}

Statistical price anomalies detected:
{anomalies}

Extract all material supply/demand disruption events.
Return JSON: {{"events": [...], "detection_summary": "..."}}
Each event object: event_type, description, detected_date, affected_region,
severity, supply_impact, source, confidence.
If no events found, return {{"events": [], "detection_summary": "No significant disruptions detected."}}
"""

    result = chat_json(SYSTEM_PROMPT, user_msg, max_tokens=2048)

    raw_events = result.get("events", [])
    events = []
    for e in raw_events:
        try:
            events.append(DetectedEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type=e.get("event_type", "unknown"),
                description=e.get("description", ""),
                detected_date=e.get("detected_date", today),
                affected_region=e.get("affected_region", "unknown"),
                severity=EventSeverity(e.get("severity", "low")),
                supply_impact=e.get("supply_impact", "neutral"),
                source=e.get("source", "LLM extraction"),
                confidence=float(e.get("confidence", 0.5)),
            ))
        except Exception:
            pass

    print(f"       Events identified: {len(events)}")

    return EventList(
        events=events,
        statistical_anomalies=anomalies,
        detection_summary=result.get("detection_summary", ""),
    )
