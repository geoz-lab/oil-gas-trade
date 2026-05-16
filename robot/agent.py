"""
Robot agent entry point.

Workflow:
  1. Check if a valid checkpoint exists for this ticker/horizon.
  2. If yes: load checkpoint.  If no (or force_retrain): run REINFORCE training.
  3. Serialize current pipeline state as text.
  4. Run greedy inference to get direction + probabilities.
  5. Compute combined signal: weighted vote of baseline + debate + robot.
  6. Return RobotDecision.
"""
from __future__ import annotations
from datetime import datetime, timezone

import torch

from core.models import (
    AdjustedForecast, BaselineForecast, DebateSummary,
    ForecastTask, JudgmentResult, RobotDecision,
    StructuredEvidenceBundle, TemporalImpactMap,
)
from robot.train import CHECKPOINT_PATH, load_checkpoint, train

TICKER_MAP = {
    "brent": "BZ=F", "brent_crude": "BZ=F",
    "wti": "CL=F", "diesel": "HO=F", "heating_oil": "HO=F",
    "gasoline": "RB=F", "natural_gas": "NG=F",
}
ACTIONS = {0: "down", 1: "hold", 2: "up"}
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Weights for combined signal (must sum to 1.0)
W_BASELINE = 0.25
W_DEBATE   = 0.45
W_ROBOT    = 0.30


# ── State serialisation ────────────────────────────────────────────────────

def _build_state_text(
    task:     ForecastTask,
    baseline: BaselineForecast,
    judgment: JudgmentResult,
    debate:   DebateSummary,
    evidence: StructuredEvidenceBundle,
    temporal: TemporalImpactMap,
    adjusted: AdjustedForecast,
) -> str:
    last_pt  = baseline.weekly_forecasts[-1] if baseline.weekly_forecasts else None
    adj_last = adjusted.adjusted_weekly[-1]   if adjusted.adjusted_weekly  else None
    ms0      = baseline.model_stats[0] if baseline.model_stats else None

    lines = [
        f"COMMODITY FORECAST STATE",
        f"Commodity: {task.commodity} | Region: {task.region} | Horizon: {task.horizon_weeks}w",
        f"Last observed: {baseline.last_observed_value:.3f} {baseline.unit} ({baseline.last_observed_date})",
        "",
        f"STATISTICAL BASELINE (LightGBM Ensemble)",
        f"Trend: {baseline.trend_direction} ({baseline.trend_magnitude_pct:+.1f}% over {task.horizon_weeks}w)",
        f"Model: {baseline.best_model}",
    ]
    if ms0:
        dir_acc = ms0.params.get("directional_accuracy_wf", "n/a")
        lines.append(f"Walk-forward directional accuracy: {dir_acc}")
    if last_pt:
        lines.append(
            f"Week-{task.horizon_weeks} baseline: {last_pt.point:.3f} "
            f"[80% CI: {last_pt.ci_80_low:.3f}–{last_pt.ci_80_high:.3f}]"
        )

    lines += [
        "",
        f"SIGNAL QUALITY (after Constitutional AI judge)",
        f"Signals kept: {len(judgment.filtered_bundle.all_signals())} | "
        f"Removed: {judgment.removed_count} | Flagged: {judgment.flagged_count}",
        f"Keywords: {', '.join(judgment.filtered_bundle.expanded_keywords[:6])}",
        "",
        f"EVIDENCE STRUCTURE",
        f"Facts: {len(evidence.facts)} | Causal claims: {len(evidence.causal_claims)} | "
        f"Speculations: {len(evidence.speculations)}",
        f"Dominant narrative: {evidence.dominant_narrative}",
        f"Contradictions: {evidence.contradiction_count}",
        "",
        f"MULTI-AGENT DEBATE",
        f"Consensus: {debate.consensus_direction} (agreement={debate.agreement_score:.0%})",
        f"Bull: {debate.strongest_bull_argument[:130]}",
        f"Bear: {debate.strongest_bear_argument[:130]}",
        f"Skeptic flags: {'; '.join(debate.skeptic_flags[:2])}",
        f"Historical analog: {debate.historical_analog[:100]}",
        "",
        f"TEMPORAL DYNAMICS",
        f"Causal chains: {len(temporal.impacts)} | Peak impact week: {temporal.peak_impact_week}",
        f"Horizon coverage: {temporal.total_horizon_coverage:.0%}",
        "",
        f"ADJUSTED FORECAST",
        f"Direction: {adjusted.direction_call} ({adjusted.magnitude_call_pct:+.1f}%)",
        f"Confidence: {adjusted.calibration.directional_confidence:.0%} | "
        f"Regime: {adjusted.calibration.uncertainty_regime.value}",
    ]
    if adj_last:
        lines.append(
            f"Week-{task.horizon_weeks} adjusted: {adj_last.point:.3f} "
            f"[80% CI: {adj_last.ci_80_low:.3f}–{adj_last.ci_80_high:.3f}]"
        )

    return "\n".join(lines)


# ── Combined signal ────────────────────────────────────────────────────────

def _direction_score(direction: str) -> float:
    """Convert direction string to signed score."""
    return {"up": 1.0, "down": -1.0, "hold": 0.0, "flat": 0.0, "contested": 0.0}.get(
        direction.lower(), 0.0
    )


def _combined_signal(
    baseline_direction: str,
    debate_direction:   str,
    robot_direction:    str,
    robot_confidence:   float,
    debate_agreement:   float,
) -> tuple[str, str]:
    """
    Weighted vote across three sources. Robot and debate weights scale with
    their respective confidence/agreement scores for adaptive blending.
    """
    # Scale robot weight by its confidence, debate weight by agreement
    w_r = W_ROBOT   * (0.5 + 0.5 * robot_confidence)
    w_d = W_DEBATE  * (0.5 + 0.5 * debate_agreement)
    w_b = W_BASELINE
    total = w_b + w_d + w_r

    score = (
        w_b * _direction_score(baseline_direction) +
        w_d * _direction_score(debate_direction)   +
        w_r * _direction_score(robot_direction)
    ) / total

    if score > 0.20:
        combined = "up"
    elif score < -0.20:
        combined = "down"
    else:
        combined = "hold"

    reasoning = (
        f"Weighted vote (baseline×{w_b/total:.0%} + debate×{w_d/total:.0%} + "
        f"robot×{w_r/total:.0%}): score={score:+.2f} → {combined.upper()}. "
        f"Baseline says {baseline_direction}, debate says {debate_direction} "
        f"(agreement={debate_agreement:.0%}), robot says {robot_direction} "
        f"(conf={robot_confidence:.0%})."
    )
    return combined, reasoning


# ── Main entry point ───────────────────────────────────────────────────────

def run(
    task:         ForecastTask,
    baseline:     BaselineForecast,
    judgment:     JudgmentResult,
    debate:       DebateSummary,
    evidence:     StructuredEvidenceBundle,
    temporal:     TemporalImpactMap,
    adjusted:     AdjustedForecast,
    force_retrain: bool = False,
    n_epochs:     int   = 4,
) -> RobotDecision:
    print("\n[RL] Trading Robot — Qwen2.5 backbone + REINFORCE policy head")

    ticker  = TICKER_MAP.get(task.commodity.lower(), "BZ=F")
    device  = "mps" if _mps_available() else "cpu"

    # ── Load or train ──────────────────────────────────────────────────────
    need_train = force_retrain or not CHECKPOINT_PATH.exists()
    if not need_train:
        # Retrain if ticker or horizon changed
        import torch as _t
        meta = _t.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if meta.get("ticker") != ticker or meta.get("horizon_weeks") != task.horizon_weeks:
            print(f"     Ticker/horizon changed — retraining...")
            need_train = True

    if need_train:
        print(f"     Training on {ticker} ({task.horizon_weeks}w horizon, {n_epochs} epochs)...")
        policy = train(
            ticker=ticker, horizon_weeks=task.horizon_weeks,
            n_epochs=n_epochs, model_name=MODEL_NAME, device=device,
        )
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        policy.eval()
    else:
        print(f"     Loading cached checkpoint: {CHECKPOINT_PATH.name}")
        policy, ckpt = load_checkpoint(model_name=MODEL_NAME, device=device)

    # ── Inference ──────────────────────────────────────────────────────────
    state_text = _build_state_text(task, baseline, judgment, debate, evidence, temporal, adjusted)

    with torch.no_grad():
        action_idx, probs = policy.greedy_action(state_text)

    probs_np   = probs.cpu().numpy().flatten()
    direction  = ACTIONS[action_idx]
    confidence = float(probs_np[action_idx])

    print(
        f"     Robot → {direction.upper()} | confidence={confidence:.1%} | "
        f"probs=[down={probs_np[0]:.2f} hold={probs_np[1]:.2f} up={probs_np[2]:.2f}]"
    )

    # ── Combined signal ────────────────────────────────────────────────────
    combined, reasoning = _combined_signal(
        baseline_direction = baseline.trend_direction,
        debate_direction   = debate.consensus_direction,
        robot_direction    = direction,
        robot_confidence   = confidence,
        debate_agreement   = debate.agreement_score,
    )
    print(f"     Combined → {combined.upper()} | {reasoning[:90]}...")

    return RobotDecision(
        direction           = direction,
        confidence          = confidence,
        down_prob           = float(probs_np[0]),
        hold_prob           = float(probs_np[1]),
        up_prob             = float(probs_np[2]),
        combined_signal     = combined,
        combined_reasoning  = reasoning,
        state_summary       = state_text[:500],
        model_name          = MODEL_NAME,
        training_windows    = int(ckpt.get("n_windows", 0)),
        training_epochs     = int(ckpt.get("n_epochs", 0)),
        directional_accuracy_train = float(ckpt.get("best_dir_acc", 0.0)),
        generated_at        = datetime.now(timezone.utc).isoformat(),
    )


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        return False
