"""Shared Pydantic data models — the typed spine connecting all 10 agents."""
from __future__ import annotations
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── [1] Task Planner ───────────────────────────────────────────────────────

class ForecastTask(BaseModel):
    raw_query: str
    commodity: str                          # e.g. "diesel", "brent", "WTI"
    region: str                             # e.g. "Gulf Coast"
    horizon_weeks: int = 8
    forecast_variable: str                  # e.g. "inventory_level", "price"
    sub_tasks: list[str] = Field(default_factory=list)
    context_notes: str = ""


# ── [2] Baseline Forecast ──────────────────────────────────────────────────

class WeeklyForecastPoint(BaseModel):
    week_offset: int                        # 1 = next week
    forecast_date: str                      # ISO date string
    point: float
    ci_80_low: float
    ci_80_high: float
    ci_95_low: float
    ci_95_high: float

class ModelStats(BaseModel):
    model_name: str
    aic: Optional[float] = None
    rmse_insample: Optional[float] = None
    params: dict[str, Any] = Field(default_factory=dict)

class BaselineForecast(BaseModel):
    task: ForecastTask
    series_name: str
    last_observed_date: str
    last_observed_value: float
    unit: str                               # e.g. "million barrels", "USD/bbl"
    best_model: str
    model_stats: list[ModelStats] = Field(default_factory=list)
    weekly_forecasts: list[WeeklyForecastPoint] = Field(default_factory=list)
    trend_direction: str                    # "up", "down", "flat"
    trend_magnitude_pct: float             # % change over horizon


# ── [3] Signal Retrieval ───────────────────────────────────────────────────

class SignalType(str, Enum):
    MARKET = "market"
    MACRO = "macro"
    OPERATIONAL = "operational"
    TEXTUAL = "textual"

class SignalDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class Signal(BaseModel):
    signal_id: str
    signal_type: SignalType
    source: str
    name: str
    value: Optional[float] = None
    unit: str = ""
    timestamp: str                          # ISO datetime
    direction: SignalDirection = SignalDirection.UNKNOWN
    raw_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    # Filled by Agent 3b — Signal Reliability Judge
    reliability_score: float = Field(ge=0.0, le=1.0, default=1.0)
    harm_flags: list[str] = Field(default_factory=list)
    judgment: str = "reliable"             # "reliable" | "questionable" | "misleading"

class SignalBundle(BaseModel):
    retrieved_at: str
    market_signals: list[Signal] = Field(default_factory=list)
    macro_signals: list[Signal] = Field(default_factory=list)
    operational_signals: list[Signal] = Field(default_factory=list)
    textual_signals: list[Signal] = Field(default_factory=list)
    expanded_keywords: list[str] = Field(default_factory=list)

    def all_signals(self) -> list[Signal]:
        return (self.market_signals + self.macro_signals
                + self.operational_signals + self.textual_signals)


# ── [3b] Signal Reliability Judge ─────────────────────────────────────────

class SignalJudgment(BaseModel):
    signal_id: str
    judgment: str                           # "reliable" | "questionable" | "misleading"
    reliability_score: float = Field(ge=0.0, le=1.0, default=0.75)
    harm_flags: list[str] = Field(default_factory=list)
    reason: str = ""

class JudgmentResult(BaseModel):
    judgments: list[SignalJudgment] = Field(default_factory=list)
    filtered_bundle: SignalBundle
    removed_count: int = 0
    flagged_count: int = 0
    judge_summary: str = ""


# ── [4] Event Detection ────────────────────────────────────────────────────

class EventSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectedEvent(BaseModel):
    event_id: str
    event_type: str                         # e.g. "refinery_outage", "sanction", "weather"
    description: str
    detected_date: str
    affected_region: str
    severity: EventSeverity
    supply_impact: str                      # "tightening", "loosening", "neutral"
    source: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

class EventList(BaseModel):
    events: list[DetectedEvent] = Field(default_factory=list)
    statistical_anomalies: list[dict[str, Any]] = Field(default_factory=list)
    detection_summary: str = ""


# ── [5] Evidence Structuring ───────────────────────────────────────────────

class EvidenceClass(str, Enum):
    FACT = "fact"
    CAUSAL_CLAIM = "causal_claim"
    SPECULATION = "speculation"
    FORECAST = "forecast"

class EvidenceItem(BaseModel):
    evidence_id: str
    classification: EvidenceClass
    source: str
    content: str
    direction: SignalDirection
    weight: float = Field(ge=0.0, le=1.0)  # higher for FACT, lower for SPECULATION
    supporting_data: str = ""
    contradicts: list[str] = Field(default_factory=list)  # evidence_ids

class StructuredEvidenceBundle(BaseModel):
    facts: list[EvidenceItem] = Field(default_factory=list)
    causal_claims: list[EvidenceItem] = Field(default_factory=list)
    speculations: list[EvidenceItem] = Field(default_factory=list)
    forecasts: list[EvidenceItem] = Field(default_factory=list)
    contradiction_count: int = 0
    dominant_narrative: str = ""

    def all_items(self) -> list[EvidenceItem]:
        return self.facts + self.causal_claims + self.speculations + self.forecasts


# ── [6] Temporal Reasoning ─────────────────────────────────────────────────

class TemporalImpact(BaseModel):
    evidence_id: str
    trigger_event: str
    propagation_path: str                   # e.g. "OPEC cut → tanker rates → refinery input cost"
    impact_lag_weeks: int
    persistence_weeks: int
    weekly_intensity: list[float]           # intensity[0] = week 1, len = persistence_weeks
    affected_variable: str
    direction: SignalDirection

class TemporalImpactMap(BaseModel):
    impacts: list[TemporalImpact] = Field(default_factory=list)
    peak_impact_week: int = 1
    total_horizon_coverage: float = 0.0    # fraction of 8-week horizon with >0 impact
    reasoning_summary: str = ""


# ── [7] Debate System ──────────────────────────────────────────────────────

class DebateStance(str, Enum):
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"

class DebatePosition(BaseModel):
    agent_name: str
    agent_role: str
    stance: DebateStance
    confidence: float = Field(ge=0.0, le=1.0)
    key_arguments: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    counterarguments_acknowledged: list[str] = Field(default_factory=list)
    price_direction_call: str               # "up", "down", "flat"
    price_magnitude_estimate_pct: float    # % move expected

class DebateSummary(BaseModel):
    positions: list[DebatePosition] = Field(default_factory=list)
    consensus_direction: str               # "up", "down", "flat", "contested"
    agreement_score: float = Field(ge=0.0, le=1.0)   # 1 = full agreement
    key_disagreements: list[str] = Field(default_factory=list)
    strongest_bull_argument: str = ""
    strongest_bear_argument: str = ""
    skeptic_flags: list[str] = Field(default_factory=list)
    historical_analog: str = ""
    synthesis_narrative: str = ""


# ── [8] Risk Calibration ───────────────────────────────────────────────────

class UncertaintyRegime(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

class CalibrationResult(BaseModel):
    directional_confidence: float = Field(ge=0.0, le=1.0)
    upside_risk_pct: float
    downside_risk_pct: float
    uncertainty_regime: UncertaintyRegime
    interval_multiplier: float = 1.0       # widens/narrows CI from baseline
    direction_bias_pct: float              # signed % to add to baseline point forecasts
    calibration_notes: str = ""
    key_risk_factors: list[str] = Field(default_factory=list)


# ── [9] Adjusted Forecast ──────────────────────────────────────────────────

class ScenarioForecast(BaseModel):
    scenario: str                          # "bull", "base", "bear"
    probability: float
    weekly_values: list[float]             # 8-week series
    narrative: str

class AdjustedForecast(BaseModel):
    baseline: BaselineForecast
    calibration: CalibrationResult
    adjusted_weekly: list[WeeklyForecastPoint]
    scenarios: list[ScenarioForecast] = Field(default_factory=list)
    adjustment_rationale: str = ""
    direction_call: str                    # "up", "down", "flat"
    magnitude_call_pct: float


# ── [RL] Trading Robot ────────────────────────────────────────────────────

class RobotDecision(BaseModel):
    direction: str                          # "up", "down", "hold"
    confidence: float = Field(ge=0.0, le=1.0)
    down_prob: float
    hold_prob: float
    up_prob: float
    combined_signal: str = ""              # weighted vote across all three sources
    combined_reasoning: str = ""
    state_summary: str = ""               # truncated text fed to Qwen
    model_name: str = ""
    training_windows: int = 0
    training_epochs: int = 0
    directional_accuracy_train: float = 0.0
    generated_at: str = ""


# ── [10] Final Report ──────────────────────────────────────────────────────

class EvidenceTableRow(BaseModel):
    source: str
    classification: str
    direction: str
    weight: float
    summary: str

class FinalReport(BaseModel):
    generated_at: str
    task: ForecastTask
    executive_summary: str
    baseline_summary: str
    adjusted_forecast: AdjustedForecast
    evidence_table: list[EvidenceTableRow] = Field(default_factory=list)
    temporal_timeline: list[dict[str, Any]] = Field(default_factory=list)
    debate_summary: DebateSummary
    uncertainty_analysis: str
    conflict_discussion: str
    key_risks: list[str] = Field(default_factory=list)
    forecast_table_md: str = ""            # pre-rendered markdown table
    robot_decision: Optional[RobotDecision] = None
