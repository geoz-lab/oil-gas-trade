"""
Agent 9 — Forecast Adjustment Engine
Translates qualitative calibration into numeric forecast modifications.

Steps:
  1. Apply direction_bias_pct to all baseline point forecasts
  2. Scale confidence intervals by interval_multiplier
  3. Generate bull / base / bear scenario series with probabilities
  4. Produce AdjustedForecast with all three scenarios

Justification: this is the bridge between the LLM reasoning layer and the
operational output — it ensures all qualitative adjustments are expressed
as concrete numbers that can feed into procurement/inventory decisions.
"""
from __future__ import annotations
from datetime import datetime

from core.models import (
    AdjustedForecast, BaselineForecast, CalibrationResult,
    DebateSummary, ScenarioForecast, WeeklyForecastPoint
)

# Scenario probability distributions based on uncertainty regime
SCENARIO_PROBS = {
    "low":      {"bull": 0.25, "base": 0.60, "bear": 0.15},
    "moderate": {"bull": 0.25, "base": 0.50, "bear": 0.25},
    "high":     {"bull": 0.30, "base": 0.40, "bear": 0.30},
    "extreme":  {"bull": 0.33, "base": 0.34, "bear": 0.33},
}

# Direction overrides — if consensus is clear, tilt probabilities
DIRECTION_PROB_SHIFT = 0.10  # shift this much probability toward consensus


def _adjust_probs(probs: dict, consensus: str) -> dict:
    p = dict(probs)
    if consensus == "up":
        p["bull"] = min(1.0, p["bull"] + DIRECTION_PROB_SHIFT)
        p["bear"] = max(0.0, p["bear"] - DIRECTION_PROB_SHIFT)
    elif consensus == "down":
        p["bear"] = min(1.0, p["bear"] + DIRECTION_PROB_SHIFT)
        p["bull"] = max(0.0, p["bull"] - DIRECTION_PROB_SHIFT)
    # Normalize
    total = sum(p.values())
    return {k: v / total for k, v in p.items()}


def run(
    baseline: BaselineForecast,
    calibration: CalibrationResult,
    debate: DebateSummary,
) -> AdjustedForecast:
    print("\n[9/10] Forecast Adjustment Engine — applying calibrated adjustments...")

    bias = calibration.direction_bias_pct / 100.0
    multiplier = calibration.interval_multiplier

    # Build adjusted weekly points
    adjusted_weekly: list[WeeklyForecastPoint] = []
    for pt in baseline.weekly_forecasts:
        # Ramp bias linearly — near-term adjustment smaller, grows with time
        ramp = pt.week_offset / baseline.task.horizon_weeks
        week_bias = bias * (0.3 + 0.7 * ramp)   # starts at 30% of bias, grows to 100%

        adj_point = pt.point * (1 + week_bias)
        ci_width_80 = (pt.ci_80_high - pt.ci_80_low) * multiplier / 2
        ci_width_95 = (pt.ci_95_high - pt.ci_95_low) * multiplier / 2

        adjusted_weekly.append(WeeklyForecastPoint(
            week_offset=pt.week_offset,
            forecast_date=pt.forecast_date,
            point=round(adj_point, 4),
            ci_80_low=round(adj_point - ci_width_80, 4),
            ci_80_high=round(adj_point + ci_width_80, 4),
            ci_95_low=round(adj_point - ci_width_95, 4),
            ci_95_high=round(adj_point + ci_width_95, 4),
        ))

    # Build scenarios
    regime = calibration.uncertainty_regime.value
    base_probs = SCENARIO_PROBS.get(regime, SCENARIO_PROBS["moderate"])
    probs = _adjust_probs(base_probs, debate.consensus_direction)

    upside_scale = 1 + calibration.upside_risk_pct / 100.0
    downside_scale = 1 - calibration.downside_risk_pct / 100.0

    bull_values = [round(pt.point * upside_scale, 4) for pt in adjusted_weekly]
    base_values = [round(pt.point, 4) for pt in adjusted_weekly]
    bear_values = [round(pt.point * downside_scale, 4) for pt in adjusted_weekly]

    scenarios = [
        ScenarioForecast(
            scenario="bull",
            probability=round(probs["bull"], 3),
            weekly_values=bull_values,
            narrative=(
                f"Supply tightening materializes: {debate.strongest_bull_argument[:120]}"
            ),
        ),
        ScenarioForecast(
            scenario="base",
            probability=round(probs["base"], 3),
            weekly_values=base_values,
            narrative=f"Baseline conditions with calibrated adjustments. {debate.synthesis_narrative[:120]}",
        ),
        ScenarioForecast(
            scenario="bear",
            probability=round(probs["bear"], 3),
            weekly_values=bear_values,
            narrative=(
                f"Demand weakness / oversupply: {debate.strongest_bear_argument[:120]}"
            ),
        ),
    ]

    # Overall direction call
    adj_last = adjusted_weekly[-1].point if adjusted_weekly else baseline.last_observed_value
    total_pct = (adj_last - baseline.last_observed_value) / baseline.last_observed_value * 100

    if total_pct > 2.0:
        direction_call = "up"
    elif total_pct < -2.0:
        direction_call = "down"
    else:
        direction_call = "flat"

    rationale = (
        f"Baseline {baseline.best_model} trend: {baseline.trend_direction} "
        f"({baseline.trend_magnitude_pct:+.1f}%). "
        f"Debate consensus: {debate.consensus_direction} "
        f"(agreement {debate.agreement_score:.0%}). "
        f"Applied direction bias: {calibration.direction_bias_pct:+.1f}%, "
        f"CI multiplier: {multiplier:.2f}x "
        f"(uncertainty regime: {regime})."
    )

    print(f"       Adjusted direction call: {direction_call} ({total_pct:+.1f}%)")
    print(f"       Scenarios: bull={probs['bull']:.0%} base={probs['base']:.0%} bear={probs['bear']:.0%}")

    return AdjustedForecast(
        baseline=baseline,
        calibration=calibration,
        adjusted_weekly=adjusted_weekly,
        scenarios=scenarios,
        adjustment_rationale=rationale,
        direction_call=direction_call,
        magnitude_call_pct=round(total_pct, 2),
    )
