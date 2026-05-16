"""
Agent 10 — Report & Explanation Generator
Produces the final structured output:
  - Executive summary (improved, analyst-grade prompt)
  - Evidence table
  - Temporal impact timeline
  - Debate agent positions
  - Forecast chart: historical + LightGBM quantile bands + signal driver annotations
  - Forecast table (baseline vs adjusted, CI bands)
  - Uncertainty and conflict analysis

Output: Markdown report + JSON + PNG chart, all saved to reports/
"""
from __future__ import annotations
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yfinance as yf

from core.config import REPORTS_DIR
from core.llm import chat
from core.models import (
    AdjustedForecast, DebateSummary, EvidenceTableRow,
    EventList, FinalReport, ForecastTask, RobotDecision,
    StructuredEvidenceBundle, TemporalImpactMap,
)

def _escape_dollars(text: str) -> str:
    return re.sub(r'(?<!\\)\$', r'\\$', text)


SYSTEM_PROMPT = """You are a senior commodity analyst at a top-tier energy trading desk.

Write an executive summary for an internal forecast report. Your audience is:
portfolio managers and supply-chain directors who need to make procurement decisions.

Requirements:
1. Open with a one-sentence verdict: direction + magnitude + horizon + confidence level.
2. Second sentence: what the statistical model says vs. what the debate adjusted it to, and why they differ.
3. Third sentence: the single strongest bull argument with its temporal lag.
4. Fourth sentence: the single strongest bear counterargument and the skeptic's most important flag.
5. Fifth sentence: the most relevant historical analogue and what it implies for the base case.
6. Final sentence: the most actionable caveat — what single data point or event would flip the directional call.

Rules:
- Use precise numbers (prices, percentages, weeks).
- No hedging phrases like "could potentially" or "may possibly".
- No markdown headers inside the summary — plain flowing prose only.
- Max 250 words.
"""


# ── Chart generation ───────────────────────────────────────────────────────

def _fetch_history_for_chart(ticker: str, weeks: int = 26) -> pd.Series:
    end   = datetime.today()
    start = end - timedelta(weeks=weeks + 2)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].squeeze().dropna().resample("W-FRI").last().dropna().iloc[-weeks:]


TICKER_MAP = {
    "brent_crude": "BZ=F",
    "brent":       "BZ=F",
    "wti":         "CL=F",
    "WTI":         "CL=F",
    "diesel":      "HO=F",
    "heating_oil": "HO=F",
    "gasoline":    "RB=F",
    "natural_gas": "NG=F",
    "LNG":         "LNG",
}


def _generate_forecast_chart(
    task: ForecastTask,
    adjusted: AdjustedForecast,
    evidence: StructuredEvidenceBundle,
    events: Optional[EventList],
    temporal: TemporalImpactMap,
    out_dir: Path,
) -> Path:
    """
    Generates a two-panel figure:
      Top panel:  Historical price (26w) + LightGBM quantile forecast bands +
                  key event/signal driver annotations
      Bottom panel: Bar chart of top evidence signal drivers by weight & direction
    """
    ticker = TICKER_MAP.get(task.commodity.lower(), "BZ=F")
    unit   = adjusted.baseline.unit

    # ── Fetch history ──────────────────────────────────────────────────────
    hist = _fetch_history_for_chart(ticker, weeks=26)

    # ── Forecast series ────────────────────────────────────────────────────
    fc_dates  = pd.to_datetime([pt.forecast_date for pt in adjusted.adjusted_weekly])
    fc_median = np.array([pt.point         for pt in adjusted.adjusted_weekly])
    fc_lo80   = np.array([pt.ci_80_low     for pt in adjusted.adjusted_weekly])
    fc_hi80   = np.array([pt.ci_80_high    for pt in adjusted.adjusted_weekly])
    fc_lo95   = np.array([pt.ci_95_low     for pt in adjusted.adjusted_weekly])
    fc_hi95   = np.array([pt.ci_95_high    for pt in adjusted.adjusted_weekly])

    # ── Scenario tails ─────────────────────────────────────────────────────
    sc_map = {sc.scenario: sc for sc in adjusted.scenarios}

    # ── Signal driver data ─────────────────────────────────────────────────
    # Top N evidence items by weight, coloured by direction
    all_ev = evidence.all_items()
    all_ev_sorted = sorted(all_ev, key=lambda e: -e.weight)[:12]

    driver_labels = [e.source[:22] + "…" if len(e.source) > 22 else e.source
                     for e in all_ev_sorted]
    driver_weights = [e.weight for e in all_ev_sorted]
    driver_colors  = [
        "#2ecc71" if e.direction.value == "bullish"
        else "#e74c3c" if e.direction.value == "bearish"
        else "#95a5a6"
        for e in all_ev_sorted
    ]

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")

    ax_price = fig.add_axes([0.06, 0.40, 0.88, 0.54])  # top panel
    ax_drivers = fig.add_axes([0.06, 0.05, 0.88, 0.28])  # bottom panel

    _style_ax(ax_price)
    _style_ax(ax_drivers)

    # ── TOP PANEL: price chart ─────────────────────────────────────────────

    # Historical
    if not hist.empty:
        ax_price.plot(hist.index, hist.values, color="#e0e0e0", linewidth=1.8,
                      label="Historical close", zorder=3)
        last_hist_val = float(hist.iloc[-1])
        last_hist_date = hist.index[-1]
        # Connect history to forecast with a small dot
        ax_price.scatter([last_hist_date], [last_hist_val],
                         color="#ffffff", s=50, zorder=5)

    # 95% CI band
    ax_price.fill_between(fc_dates, fc_lo95, fc_hi95,
                          color="#3498db", alpha=0.12, label="95% CI")
    # 80% CI band
    ax_price.fill_between(fc_dates, fc_lo80, fc_hi80,
                          color="#3498db", alpha=0.22, label="80% CI")

    # Scenario tails (dashed)
    if "bull" in sc_map and sc_map["bull"].weekly_values:
        bull_vals = sc_map["bull"].weekly_values
        ax_price.plot(fc_dates[:len(bull_vals)], bull_vals,
                      "--", color="#2ecc71", linewidth=1.2, alpha=0.8,
                      label=f"Bull ({sc_map['bull'].probability:.0%})")
    if "bear" in sc_map and sc_map["bear"].weekly_values:
        bear_vals = sc_map["bear"].weekly_values
        ax_price.plot(fc_dates[:len(bear_vals)], bear_vals,
                      "--", color="#e74c3c", linewidth=1.2, alpha=0.8,
                      label=f"Bear ({sc_map['bear'].probability:.0%})")

    # Median forecast
    ax_price.plot(fc_dates, fc_median, color="#3498db", linewidth=2.2,
                  marker="o", markersize=4, label="Adjusted median", zorder=4)

    # Vertical divider at forecast start
    if not hist.empty:
        ax_price.axvline(x=hist.index[-1], color="#555", linewidth=1,
                         linestyle=":", alpha=0.7)
        ax_price.text(hist.index[-1], ax_price.get_ylim()[0],
                      " Forecast →", color="#888", fontsize=7.5, va="bottom")

    # Annotate key temporal signal drivers on the forecast
    annotated = 0
    for imp in sorted(temporal.impacts, key=lambda x: -max(x.weekly_intensity if x.weekly_intensity else [0])):
        if annotated >= 4:
            break
        onset_week = imp.impact_lag_weeks
        if 0 < onset_week <= len(fc_dates):
            ann_date = fc_dates[onset_week - 1]
            ann_val  = float(fc_median[onset_week - 1])
            color    = "#2ecc71" if imp.direction.value == "bullish" else "#e74c3c"
            label    = imp.trigger_event[:28]
            ax_price.annotate(
                label,
                xy=(ann_date, ann_val),
                xytext=(ann_date, ann_val + (ann_val * 0.04 * (1 if imp.direction.value == "bullish" else -1))),
                color=color, fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )
            annotated += 1

    direction_color = "#2ecc71" if adjusted.direction_call == "up" else \
                      "#e74c3c" if adjusted.direction_call == "down" else "#f39c12"
    ax_price.set_title(
        f"{task.commodity.upper()} Price Forecast — {task.horizon_weeks}-Week Horizon  "
        f"| Direction: {adjusted.direction_call.upper()} ({adjusted.magnitude_call_pct:+.1f}%)  "
        f"| Confidence: {adjusted.calibration.directional_confidence:.0%}  "
        f"| Regime: {adjusted.calibration.uncertainty_regime.value.upper()}",
        color="#e0e0e0", fontsize=10, fontweight="bold", pad=8,
    )
    ax_price.set_ylabel(unit, color="#aaa", fontsize=9)
    ax_price.tick_params(colors="#888", labelsize=8)
    ax_price.xaxis.set_tick_params(rotation=20)
    ax_price.legend(loc="upper left", fontsize=7.5, framealpha=0.2,
                    labelcolor="#e0e0e0", facecolor="#1a1a2e")

    # ── BOTTOM PANEL: signal drivers ───────────────────────────────────────

    y_pos = np.arange(len(driver_labels))
    bars  = ax_drivers.barh(y_pos, driver_weights, color=driver_colors,
                             height=0.6, alpha=0.85)
    ax_drivers.set_yticks(y_pos)
    ax_drivers.set_yticklabels(driver_labels, fontsize=7.5, color="#ccc")
    ax_drivers.set_xlabel("Evidence Weight  (fact=0.85–1.0 / causal=0.55–0.75 / spec=0.2–0.45)",
                          color="#888", fontsize=7.5)
    ax_drivers.set_title("Key Signal Drivers by Weight & Direction",
                         color="#e0e0e0", fontsize=9, fontweight="bold")
    ax_drivers.set_xlim(0, 1.05)
    ax_drivers.tick_params(colors="#888", labelsize=8)

    # Add value labels on bars
    for bar, weight in zip(bars, driver_weights):
        ax_drivers.text(weight + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{weight:.2f}", va="center", fontsize=7, color="#aaa")

    # Legend for driver panel
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Bullish signal"),
        mpatches.Patch(color="#e74c3c", label="Bearish signal"),
        mpatches.Patch(color="#95a5a6", label="Neutral signal"),
    ]
    ax_drivers.legend(handles=legend_patches, loc="lower right",
                      fontsize=7.5, framealpha=0.2,
                      labelcolor="#e0e0e0", facecolor="#1a1a2e")

    # ── Save ───────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    chart_path = out_dir / f"forecast_chart_{ts}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return chart_path


def _style_ax(ax):
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#888")
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    ax.grid(True, color="#2a2a3e", linewidth=0.5, alpha=0.7)


# ── Executive summary ──────────────────────────────────────────────────────

def _build_executive_summary(
    task: ForecastTask,
    adjusted: AdjustedForecast,
    debate: DebateSummary,
    evidence: StructuredEvidenceBundle,
    temporal: TemporalImpactMap,
) -> str:
    top_temporal = sorted(temporal.impacts,
                          key=lambda x: -max(x.weekly_intensity or [0]))[:2]
    temporal_txt = " | ".join(
        f"{t.trigger_event} (onset W{t.impact_lag_weeks}, {t.direction.value})"
        for t in top_temporal
    )

    context = f"""Forecast specifications:
- Commodity: {task.commodity} | Region: {task.region} | Horizon: {task.horizon_weeks} weeks
- Forecast variable: {task.forecast_variable}

Statistical baseline (LightGBM quantile):
- Last observed: {adjusted.baseline.last_observed_value:.3f} {adjusted.baseline.unit} ({adjusted.baseline.last_observed_date})
- Baseline 8w trend: {adjusted.baseline.trend_direction} ({adjusted.baseline.trend_magnitude_pct:+.1f}%)
- Walk-forward directional accuracy: {adjusted.baseline.model_stats[0].params.get('directional_accuracy_wf', 'n/a')}

Multi-agent debate result:
- Consensus direction: {debate.consensus_direction} (agreement {debate.agreement_score:.0%})
- Directional confidence: {adjusted.calibration.directional_confidence:.0%}
- Uncertainty regime: {adjusted.calibration.uncertainty_regime.value}
- Direction bias applied: {adjusted.calibration.direction_bias_pct:+.1f}%
- CI width multiplier: {adjusted.calibration.interval_multiplier:.2f}x

Adjusted forecast: {adjusted.direction_call.upper()} {adjusted.magnitude_call_pct:+.1f}% over {task.horizon_weeks}w
  Week-8 median: {adjusted.adjusted_weekly[-1].point:.3f} | 80% CI: [{adjusted.adjusted_weekly[-1].ci_80_low:.3f}, {adjusted.adjusted_weekly[-1].ci_80_high:.3f}]

Agent positions:
{chr(10).join(f"  - {p.agent_name}: {p.stance.value} {p.price_direction_call} ({p.price_magnitude_estimate_pct:+.1f}%)" for p in debate.positions)}

Strongest bull argument: {debate.strongest_bull_argument}
Strongest bear argument: {debate.strongest_bear_argument}
Skeptic flags: {"; ".join(debate.skeptic_flags[:2])}
Historical analogue: {debate.historical_analog}

Top temporal signal drivers: {temporal_txt}

Evidence quality:
- Facts: {len(evidence.facts)} | Causal claims: {len(evidence.causal_claims)} | Speculations: {len(evidence.speculations)}
- Dominant narrative: {evidence.dominant_narrative}
- Contradictions: {evidence.contradiction_count}

Key risks: {adjusted.calibration.key_risk_factors[:3]}
"""
    return chat(SYSTEM_PROMPT, f"Write the executive summary.\n\n{context}", max_tokens=500)


# ── Evidence table ─────────────────────────────────────────────────────────

def _build_evidence_table(evidence: StructuredEvidenceBundle) -> list[EvidenceTableRow]:
    rows = []
    for item in sorted(evidence.all_items(), key=lambda e: -e.weight)[:20]:
        rows.append(EvidenceTableRow(
            source=item.source[:40],
            classification=item.classification.value,
            direction=item.direction.value,
            weight=item.weight,
            summary=item.content[:100],
        ))
    return rows


# ── Forecast table (markdown) ──────────────────────────────────────────────

def _forecast_table_md(adjusted: AdjustedForecast) -> str:
    unit = adjusted.baseline.unit
    lines = [
        f"| Week | Date | Baseline ({unit}) | Adjusted ({unit}) | 80% CI Low | 80% CI High | 95% CI Low | 95% CI High |",
        "|------|------|-------------------|-------------------|------------|-------------|------------|-------------|",
    ]
    base_pts = {pt.week_offset: pt for pt in adjusted.baseline.weekly_forecasts}
    for pt in adjusted.adjusted_weekly:
        base = base_pts.get(pt.week_offset)
        base_str = f"{base.point:.4f}" if base else "—"
        lines.append(
            f"| {pt.week_offset} | {pt.forecast_date} | {base_str} | "
            f"**{pt.point:.4f}** | {pt.ci_80_low:.4f} | {pt.ci_80_high:.4f} | "
            f"{pt.ci_95_low:.4f} | {pt.ci_95_high:.4f} |"
        )
    return "\n".join(lines)


# ── Temporal timeline ──────────────────────────────────────────────────────

def _temporal_timeline(temporal: TemporalImpactMap) -> list[dict]:
    return [
        {
            "trigger":        imp.trigger_event,
            "onset_week":     imp.impact_lag_weeks,
            "duration_weeks": imp.persistence_weeks,
            "direction":      imp.direction.value,
            "path":           imp.propagation_path[:140],
        }
        for imp in sorted(temporal.impacts, key=lambda x: x.impact_lag_weeks)
    ]


# ── Markdown renderer ──────────────────────────────────────────────────────

def _robot_section_md(robot: Optional[RobotDecision]) -> list[str]:
    """Render the RL robot decision block for the markdown report."""
    if robot is None:
        return ["*(RL robot not available this run)*", ""]

    dir_icon  = {"up": "🟢", "down": "🔴", "hold": "🟡"}.get(robot.direction, "⚪")
    comb_icon = {"up": "🟢", "down": "🔴", "hold": "🟡"}.get(robot.combined_signal, "⚪")

    return [
        f"| Source | Direction | Confidence | Notes |",
        f"|--------|-----------|------------|-------|",
        f"| LightGBM Baseline | — | — | Statistical price scale anchor |",
        f"| Multi-Agent Debate | — | — | 4-agent adversarial consensus |",
        f"| {dir_icon} RL Robot ({robot.model_name}) | **{robot.direction.upper()}** | {robot.confidence:.0%} | REINFORCE, {robot.training_windows} windows, {robot.training_epochs} epochs |",
        f"| {comb_icon} **Combined Signal** | **{robot.combined_signal.upper()}** | — | Weighted vote (baseline 25% + debate 45% + robot 30%) |",
        "",
        "**Robot action probabilities:**",
        f"- Down: {robot.down_prob:.1%}",
        f"- Hold: {robot.hold_prob:.1%}",
        f"- Up:   {robot.up_prob:.1%}",
        "",
        f"**Training:** {robot.training_windows} historical windows | "
        f"{robot.training_epochs} REINFORCE epochs | "
        f"Best directional accuracy: {robot.directional_accuracy_train:.1%}",
        "",
        f"**Combined signal reasoning:** {robot.combined_reasoning}",
        "",
    ]


def _write_markdown(report: FinalReport, chart_path: Optional[Path]) -> Path:
    adj   = report.adjusted_forecast
    debate = report.debate_summary
    task  = report.task

    chart_rel = chart_path.name if chart_path else None

    lines = [
        "# Oil & Gas Supply-Chain Forecast Report",
        f"**Generated:** {report.generated_at}",
        f"**Query:** {task.raw_query}",
        f"**Commodity:** {task.commodity}  |  **Region:** {task.region}  |  **Horizon:** {task.horizon_weeks} weeks",
        "",
        "---",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "---",
        "## Baseline Statistical Forecast",
        "",
        report.baseline_summary,
        "",
        "---",
        "## Forecast Chart",
        "",
    ]
    if chart_rel:
        lines += [f"![Forecast Chart]({chart_rel})", ""]
    else:
        lines += ["*(chart unavailable)*", ""]

    lines += [
        "---",
        "## 8-Week Adjusted Forecast Table",
        "",
        report.forecast_table_md,
        "",
        "### Scenario Analysis",
        "",
        f"| Scenario | Probability | Week-8 ({adj.baseline.unit}) | Narrative |",
        "|----------|-------------|------------------------------|-----------|",
    ]
    for sc in adj.scenarios:
        last = sc.weekly_values[-1] if sc.weekly_values else 0.0
        lines.append(
            f"| **{sc.scenario.upper()}** | {sc.probability:.0%} | {last:.4f} | {sc.narrative[:90]} |"
        )

    lines += [
        "",
        "---",
        "## Evidence Table",
        "",
        "| Source | Class | Direction | Weight | Summary |",
        "|--------|-------|-----------|--------|---------|",
    ]
    for row in report.evidence_table:
        lines.append(
            f"| {row.source} | {row.classification} | {row.direction} | "
            f"{row.weight:.2f} | {row.summary} |"
        )

    lines += [
        "",
        "---",
        "## Temporal Impact Timeline",
        "",
        "| Trigger | Onset Week | Duration | Direction | Propagation Path |",
        "|---------|-----------|----------|-----------|-----------------|",
    ]
    for t in report.temporal_timeline:
        lines.append(
            f"| {t['trigger']} | W{t['onset_week']} | {t['duration_weeks']}w | "
            f"{t['direction']} | {t['path']} |"
        )

    lines += [
        "",
        "---",
        "## Multi-Agent Debate",
        "",
        f"**Consensus:** {debate.consensus_direction}  |  "
        f"**Agreement:** {debate.agreement_score:.0%}  |  "
        f"**Historical analog:** {debate.historical_analog[:120]}",
        "",
        "### Agent Positions",
        "",
    ]
    for pos in sorted(debate.positions, key=lambda p: p.agent_name):
        stance_md = {
            "strongly_bullish": "🟢 STRONGLY BULLISH",
            "bullish": "🟢 BULLISH",
            "neutral": "🟡 NEUTRAL",
            "bearish": "🔴 BEARISH",
            "strongly_bearish": "🔴 STRONGLY BEARISH",
        }.get(pos.stance.value, pos.stance.value.upper())
        lines += [
            f"#### {pos.agent_name}",
            f"**Stance:** {stance_md}  |  **Conf:** {pos.confidence:.0%}  |  "
            f"**Call:** {pos.price_direction_call} ({pos.price_magnitude_estimate_pct:+.1f}%)",
            "",
            "**Arguments:**",
            *[f"- {arg}" for arg in pos.key_arguments],
            "",
            "**Acknowledges:**",
            *[f"- {ca}" for ca in pos.counterarguments_acknowledged[:2]],
            "",
        ]

    lines += [
        "### Debate Synthesis",
        "",
        debate.synthesis_narrative,
        "",
        f"**Strongest bull:** {debate.strongest_bull_argument}",
        "",
        f"**Strongest bear:** {debate.strongest_bear_argument}",
        "",
        "**Skeptic flags:**",
        *[f"- {f}" for f in debate.skeptic_flags],
        "",
        "---",
        "## RL Trading Robot Signal",
        "",
        *_robot_section_md(report.robot_decision),
        "---",
        "## Uncertainty & Confidence Analysis",
        "",
        report.uncertainty_analysis,
        "",
        "---",
        "## Conflict Discussion",
        "",
        report.conflict_discussion,
        "",
        "---",
        "## Key Risks",
        "",
        *[f"- {r}" for r in report.key_risks],
        "",
        "---",
        f"*System: Oil & Gas Multi-Agent Supply-Chain Forecasting System*  ",
        f"*Baseline: LightGBM Quantile Regression  |  Debate: 4-agent adversarial  |  Powered by claude-sonnet-4-6*",
    ]

    md_text  = _escape_dollars("\n".join(lines))
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"forecast_{ts}.md"
    out_path.write_text(md_text)
    return out_path


def _write_json(report: FinalReport) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"forecast_{ts}.json"
    out_path.write_text(report.model_dump_json(indent=2))
    return out_path


# ── Main entry point ───────────────────────────────────────────────────────

def run(
    task: ForecastTask,
    adjusted: AdjustedForecast,
    debate: DebateSummary,
    evidence: StructuredEvidenceBundle,
    temporal: TemporalImpactMap,
    events: Optional[EventList] = None,
    robot_decision: Optional[RobotDecision] = None,
) -> FinalReport:
    print("\n[10/10] Report & Explanation Generator — assembling report + chart...")

    now = datetime.now(timezone.utc).isoformat()

    exec_summary  = _build_executive_summary(task, adjusted, debate, evidence, temporal)
    evidence_table = _build_evidence_table(evidence)
    timeline      = _temporal_timeline(temporal)
    forecast_md   = _forecast_table_md(adjusted)

    _ms    = adjusted.baseline.model_stats[0]
    _rmse  = f"{_ms.rmse_insample:.4f}" if _ms.rmse_insample is not None else "n/a"
    _dacc  = _ms.params.get("directional_accuracy_wf", "n/a")

    # Build per-window stats rows for model_stats[1:] (2Y / 1Y / 6M windows)
    _window_rows = ""
    for wms in adjusted.baseline.model_stats[1:]:
        _w_rmse = f"{wms.rmse_insample:.4f}" if wms.rmse_insample is not None else "n/a"
        _w_dacc = wms.params.get("directional_accuracy_wf", "n/a")
        _w_wt   = wms.params.get("ensemble_weight", "n/a")
        _window_rows += (
            f"| {wms.model_name} | {_w_rmse} | {_w_dacc} | {_w_wt} |\n"
        )

    baseline_summary = (
        f"LightGBM quantile ensemble trained on {adjusted.baseline.series_name}. "
        f"Last observed: **{adjusted.baseline.last_observed_value:.4f} {adjusted.baseline.unit}** "
        f"({adjusted.baseline.last_observed_date}). "
        f"8-week trend: **{adjusted.baseline.trend_direction}** "
        f"({adjusted.baseline.trend_magnitude_pct:+.1f}%). "
        f"Ensemble walk-forward RMSE: {_rmse} | Directional accuracy: {_dacc}.\n\n"
        f"**Per-window model performance:**\n\n"
        f"| Window | RMSE | Dir. Acc. | Ensemble Weight |\n"
        f"|--------|------|-----------|----------------|\n"
        f"{_window_rows}"
    )

    uncertainty_analysis = (
        f"Uncertainty regime: **{adjusted.calibration.uncertainty_regime.value}**. "
        f"Directional confidence: **{adjusted.calibration.directional_confidence:.0%}**. "
        f"Upside tail risk: {adjusted.calibration.upside_risk_pct:.1f}%  |  "
        f"Downside tail risk: {adjusted.calibration.downside_risk_pct:.1f}%. "
        f"CI width multiplier: {adjusted.calibration.interval_multiplier:.2f}× the LightGBM baseline band. "
        f"{adjusted.calibration.calibration_notes}"
    )

    conflict_discussion = (
        "The four debate agents disagreed on the following: "
        + "; ".join(debate.key_disagreements[:3])
        + ". Skeptic challenges: "
        + "; ".join(debate.skeptic_flags[:3])
        + "."
    )

    key_risks = list(adjusted.calibration.key_risk_factors[:5])
    if debate.skeptic_flags:
        key_risks.append(f"Epistemic: {debate.skeptic_flags[0]}")

    report = FinalReport(
        generated_at=now,
        task=task,
        executive_summary=exec_summary,
        baseline_summary=baseline_summary,
        adjusted_forecast=adjusted,
        evidence_table=evidence_table,
        temporal_timeline=timeline,
        debate_summary=debate,
        uncertainty_analysis=uncertainty_analysis,
        conflict_discussion=conflict_discussion,
        key_risks=key_risks,
        forecast_table_md=forecast_md,
        robot_decision=robot_decision,
    )

    # Generate chart
    chart_path: Optional[Path] = None
    try:
        chart_path = _generate_forecast_chart(
            task, adjusted, evidence, events, temporal, REPORTS_DIR
        )
        print(f"       Chart:    {chart_path}")
    except Exception as exc:
        print(f"       Warning: chart failed — {exc}")

    md_path   = _write_markdown(report, chart_path)
    json_path = _write_json(report)
    print(f"       Markdown: {md_path}")
    print(f"       JSON:     {json_path}")

    return report
