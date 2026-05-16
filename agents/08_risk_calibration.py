"""
Agent 8 — Confidence & Risk Calibration Agent
Transforms debate outputs into calibrated uncertainty estimates.

Scoring logic:
  - Base directional confidence from agreement_score
  - Skeptic flags widen uncertainty
  - Fact-heavy evidence bundle narrows uncertainty
  - High contradiction count widens uncertainty
  - Historical analog match boosts confidence

Outputs:
  - directional_confidence: probability the direction call is correct
  - upside_risk_pct / downside_risk_pct: tail risk magnitudes
  - uncertainty_regime: low / moderate / high / extreme
  - interval_multiplier: CI width adjustment factor for forecast adjustment
  - direction_bias_pct: signed % adjustment to baseline point forecasts
"""
from __future__ import annotations

from core.llm import chat_json
from core.models import (
    BaselineForecast, CalibrationResult, DebateSummary,
    StructuredEvidenceBundle, UncertaintyRegime
)

SYSTEM_PROMPT = """You are a risk calibration analyst for an oil & gas forecasting desk.

Given a multi-agent debate summary and evidence quality assessment, produce calibrated
uncertainty estimates for an 8-week Gulf Coast diesel inventory/price forecast.

Calibration rules:
1. agreement_score > 0.75 → higher directional_confidence
2. Skeptic raised ≥ 3 flags → reduce directional_confidence by 10-20 ppts
3. Evidence bundle mostly FACTs → narrow intervals (multiplier < 1.0)
4. High contradiction count → widen intervals (multiplier > 1.3)
5. Historical analog match → adjust direction_bias toward analog outcome
6. Contested/disagreed direction → wider intervals regardless of other factors

Uncertainty regime thresholds:
  - directional_confidence ≥ 0.75 → low uncertainty
  - 0.55–0.75 → moderate
  - 0.40–0.55 → high
  - < 0.40 → extreme

direction_bias_pct:
  - Bullish consensus: positive (e.g. +2% to +8%)
  - Bearish consensus: negative (e.g. -2% to -8%)
  - Contested: near zero
  - Scale with agreement_score and confidence

upside_risk_pct and downside_risk_pct: tail risk magnitude (not direction bias),
  typically 5-25% for near-term commodity forecasts.
"""


def run(
    baseline: BaselineForecast,
    debate: DebateSummary,
    evidence: StructuredEvidenceBundle,
) -> CalibrationResult:
    print("\n[8/10] Confidence & Risk Calibration Agent...")

    # Build quantitative scoring inputs
    n_facts = len(evidence.facts)
    n_specs = len(evidence.speculations)
    n_all = max(len(evidence.all_items()), 1)
    fact_ratio = n_facts / n_all
    n_skeptic_flags = len(debate.skeptic_flags)
    n_contradictions = evidence.contradiction_count

    user_msg = f"""Debate summary:
- Consensus direction: {debate.consensus_direction}
- Agreement score: {debate.agreement_score:.2f}
- Skeptic flags raised: {n_skeptic_flags}
- Strongest bull argument: {debate.strongest_bull_argument[:200]}
- Strongest bear argument: {debate.strongest_bear_argument[:200]}
- Historical analog: {debate.historical_analog[:200]}
- Key disagreements: {debate.key_disagreements[:3]}

Evidence quality:
- Total evidence items: {n_all}
- Facts: {n_facts} ({fact_ratio:.0%} of total)
- Speculations: {n_specs}
- Contradiction count: {n_contradictions}

Baseline statistical forecast:
- Trend direction: {baseline.trend_direction}
- Trend magnitude: {baseline.trend_magnitude_pct:+.1f}% over {baseline.task.horizon_weeks} weeks
- Best model: {baseline.best_model}

Return JSON:
{{
  "directional_confidence": 0.0-1.0,
  "upside_risk_pct": <float>,
  "downside_risk_pct": <float>,
  "uncertainty_regime": "low|moderate|high|extreme",
  "interval_multiplier": <float>,
  "direction_bias_pct": <float>,
  "calibration_notes": "...",
  "key_risk_factors": ["...", "...", "..."]
}}
"""

    result = chat_json(SYSTEM_PROMPT, user_msg, max_tokens=1024)

    try:
        regime = UncertaintyRegime(result.get("uncertainty_regime", "moderate"))
    except ValueError:
        regime = UncertaintyRegime.MODERATE

    cal = CalibrationResult(
        directional_confidence=float(result.get("directional_confidence", 0.55)),
        upside_risk_pct=float(result.get("upside_risk_pct", 10.0)),
        downside_risk_pct=float(result.get("downside_risk_pct", 10.0)),
        uncertainty_regime=regime,
        interval_multiplier=float(result.get("interval_multiplier", 1.0)),
        direction_bias_pct=float(result.get("direction_bias_pct", 0.0)),
        calibration_notes=result.get("calibration_notes", ""),
        key_risk_factors=result.get("key_risk_factors", []),
    )

    print(f"       Directional confidence: {cal.directional_confidence:.0%}")
    print(f"       Uncertainty regime: {cal.uncertainty_regime.value}")
    print(f"       Direction bias: {cal.direction_bias_pct:+.1f}% | CI multiplier: {cal.interval_multiplier:.2f}x")

    return cal
