"""
Oil & Gas Multi-Agent Supply-Chain Forecasting System
Orchestrates all 10 agents in sequence.

Usage:
    conda activate oil-gas-trade
    python main.py "Forecast diesel inventory demand for Gulf Coast refineries over the next 8 weeks"
    python main.py --query "Forecast WTI crude price for next 4 weeks" --horizon 4
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

console = Console()


def _print_banner() -> None:
    console.print(Panel(
        "[bold cyan]Oil & Gas Multi-Agent Supply-Chain Forecasting System[/bold cyan]\n"
        "[dim]Statistical baseline + multi-agent debate + temporal reasoning[/dim]",
        expand=False,
    ))


def _print_forecast_table(report) -> None:
    table = Table(title="8-Week Adjusted Forecast", show_header=True, header_style="bold magenta")
    table.add_column("Week", justify="center")
    table.add_column("Date", justify="center")
    table.add_column(f"Baseline ({report.adjusted_forecast.baseline.unit})", justify="right")
    table.add_column(f"Adjusted ({report.adjusted_forecast.baseline.unit})", justify="right")
    table.add_column("80% CI Low", justify="right")
    table.add_column("80% CI High", justify="right")

    baseline_pts = {pt.week_offset: pt for pt in report.adjusted_forecast.baseline.weekly_forecasts}
    for pt in report.adjusted_forecast.adjusted_weekly:
        base = baseline_pts.get(pt.week_offset)
        base_str = f"{base.point:.4f}" if base else "—"
        table.add_row(
            str(pt.week_offset),
            pt.forecast_date,
            base_str,
            f"[bold]{pt.point:.4f}[/bold]",
            f"[dim]{pt.ci_80_low:.4f}[/dim]",
            f"[dim]{pt.ci_80_high:.4f}[/dim]",
        )
    console.print(table)


def _print_debate_summary(debate) -> None:
    console.print("\n[bold]Multi-Agent Debate Summary[/bold]")
    for pos in debate.positions:
        stance_color = {
            "strongly_bullish": "bright_green",
            "bullish": "green",
            "neutral": "yellow",
            "bearish": "red",
            "strongly_bearish": "bright_red",
        }.get(pos.stance.value, "white")
        console.print(
            f"  [{stance_color}]{pos.agent_name}[/{stance_color}] "
            f"→ {pos.price_direction_call} ({pos.price_magnitude_estimate_pct:+.1f}%) "
            f"conf={pos.confidence:.0%}"
        )
    console.print(f"  [bold]Consensus:[/bold] {debate.consensus_direction} "
                  f"(agreement {debate.agreement_score:.0%})")


def run(query: str, horizon: int | None = None) -> None:
    t_start = time.time()
    _print_banner()

    # Import agents
    from agents import (
        task_planner, baseline_forecast, signal_retrieval, signal_judge,
        event_detection, evidence_structuring, temporal_reasoning,
        debate_system, risk_calibration, forecast_adjustment, report_generator,
    )

    console.print(f"\n[dim]Query:[/dim] {query}\n")

    # ── [1] Task Planner ──
    task = task_planner.run(query)
    if horizon:
        task.horizon_weeks = horizon
    console.print(f"  → Commodity: [cyan]{task.commodity}[/cyan] | "
                  f"Region: [cyan]{task.region}[/cyan] | "
                  f"Horizon: [cyan]{task.horizon_weeks}w[/cyan] | "
                  f"Variable: [cyan]{task.forecast_variable}[/cyan]")

    # ── [2] Baseline Forecast ──
    baseline = baseline_forecast.run(task)
    console.print(f"  → Baseline ({baseline.best_model}): "
                  f"trend=[bold]{baseline.trend_direction}[/bold] "
                  f"({baseline.trend_magnitude_pct:+.1f}%) | "
                  f"last={baseline.last_observed_value:.4f} {baseline.unit}")

    # ── [3] Signal Retrieval ──
    raw_signals = signal_retrieval.run(task)
    total_raw = len(raw_signals.all_signals())
    console.print(f"  → Signals collected: {total_raw} "
                  f"(market={len(raw_signals.market_signals)} "
                  f"macro={len(raw_signals.macro_signals)} "
                  f"text={len(raw_signals.textual_signals)}) "
                  f"| keywords={len(raw_signals.expanded_keywords)}")

    # ── [3b] Signal Judge ──
    judgment = signal_judge.run(raw_signals)
    signals  = judgment.filtered_bundle
    console.print(f"  → Judge: kept={len(signals.all_signals())} "
                  f"removed={judgment.removed_count} "
                  f"flagged={judgment.flagged_count}")

    # ── [4] Event Detection ──
    events = event_detection.run(signals)
    console.print(f"  → Events detected: {len(events.events)} | "
                  f"Anomalies: {len(events.statistical_anomalies)}")

    # ── [5] Evidence Structuring ──
    evidence = evidence_structuring.run(signals, events)
    console.print(f"  → Evidence: facts={len(evidence.facts)} "
                  f"causal={len(evidence.causal_claims)} "
                  f"spec={len(evidence.speculations)} | "
                  f"narrative=[italic]{evidence.dominant_narrative}[/italic]")

    # ── [6] Temporal Reasoning ──
    temporal = temporal_reasoning.run(evidence, events)
    console.print(f"  → Temporal impacts: {len(temporal.impacts)} | "
                  f"peak_week={temporal.peak_impact_week}")

    # ── [7] Debate ──
    debate = debate_system.run(baseline, evidence, temporal)
    _print_debate_summary(debate)

    # ── [8] Risk Calibration ──
    calibration = risk_calibration.run(baseline, debate, evidence)
    console.print(f"\n  → Confidence: [bold]{calibration.directional_confidence:.0%}[/bold] | "
                  f"Regime: [bold]{calibration.uncertainty_regime.value}[/bold] | "
                  f"Bias: {calibration.direction_bias_pct:+.1f}%")

    # ── [9] Forecast Adjustment ──
    adjusted = forecast_adjustment.run(baseline, calibration, debate)

    # ── [RL] Trading Robot ──
    robot_decision = None
    try:
        from robot.agent import run as robot_run
        robot_decision = robot_run(
            task=task, baseline=baseline, judgment=judgment,
            debate=debate, evidence=evidence, temporal=temporal,
            adjusted=adjusted,
        )
        dir_color = {"up": "green", "down": "red", "hold": "yellow"}.get(
            robot_decision.direction, "white"
        )
        comb_color = {"up": "green", "down": "red", "hold": "yellow"}.get(
            robot_decision.combined_signal, "white"
        )
        console.print(
            f"\n  → Robot ({robot_decision.model_name.split('/')[0]}): "
            f"[{dir_color}]{robot_decision.direction.upper()}[/{dir_color}] "
            f"(conf={robot_decision.confidence:.0%}) | "
            f"Combined: [{comb_color}]{robot_decision.combined_signal.upper()}[/{comb_color}]"
        )
    except Exception as exc:
        console.print(f"\n  [dim]Robot skipped — {exc}[/dim]")

    # ── [10] Report ──
    report = report_generator.run(
        task, adjusted, debate, evidence, temporal,
        events=events, robot_decision=robot_decision,
    )

    # Print final table
    console.print()
    _print_forecast_table(report)

    elapsed = time.time() - t_start
    console.print(f"\n[dim]Total runtime: {elapsed:.1f}s[/dim]")
    console.print("\n[bold green]Executive Summary[/bold green]")
    console.print(report.executive_summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oil & Gas Multi-Agent Supply-Chain Forecasting System"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Forecast Brent crude oil price over the next 8 weeks",
        help="Natural language forecast query",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override forecast horizon (weeks)",
    )
    args = parser.parse_args()
    run(args.query, args.horizon)


if __name__ == "__main__":
    main()
