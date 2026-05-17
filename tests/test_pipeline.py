"""
Unit tests for all pipeline agents using mock data — no API calls required.
Run with: pytest tests/test_pipeline.py -v
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from core.models import (
    ForecastTask, BaselineForecast, WeeklyForecastPoint, ModelStats,
    Signal, SignalBundle, SignalType, SignalDirection,
    DetectedEvent, EventList, EventSeverity,
    EvidenceItem, EvidenceClass, StructuredEvidenceBundle,
    TemporalImpact, TemporalImpactMap,
    DebatePosition, DebateStance, DebateSummary,
    CalibrationResult, UncertaintyRegime,
    AdjustedForecast, ScenarioForecast,
)

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_task() -> ForecastTask:
    return ForecastTask(
        raw_query="Forecast diesel inventory for Gulf Coast over 8 weeks",
        commodity="diesel",
        region="Gulf Coast",
        horizon_weeks=8,
        forecast_variable="inventory_level",
        sub_tasks=["fetch price data", "analyze OPEC signals", "estimate demand"],
    )


@pytest.fixture
def sample_baseline(sample_task) -> BaselineForecast:
    points = [
        WeeklyForecastPoint(
            week_offset=i,
            forecast_date=f"2025-05-{i+1:02d}",
            point=2.50 + i * 0.01,
            ci_80_low=2.40 + i * 0.01,
            ci_80_high=2.60 + i * 0.01,
            ci_95_low=2.35 + i * 0.01,
            ci_95_high=2.65 + i * 0.01,
        )
        for i in range(1, 9)
    ]
    return BaselineForecast(
        task=sample_task,
        series_name="diesel (HO=F)",
        last_observed_date="2025-04-30",
        last_observed_value=2.50,
        unit="USD/gal",
        best_model="ARIMA",
        model_stats=[ModelStats(model_name="ARIMA", aic=-120.5)],
        weekly_forecasts=points,
        trend_direction="up",
        trend_magnitude_pct=3.2,
    )


@pytest.fixture
def sample_signals() -> SignalBundle:
    return SignalBundle(
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        market_signals=[
            Signal(
                signal_id="mkt_brent",
                signal_type=SignalType.MARKET,
                source="yfinance",
                name="Brent Crude",
                value=82.5,
                unit="USD/bbl",
                timestamp="2025-04-30",
                direction=SignalDirection.BULLISH,
                raw_text="Brent: 82.50 USD/bbl",
                confidence=0.9,
            )
        ],
        macro_signals=[
            Signal(
                signal_id="fred_pmi",
                signal_type=SignalType.MACRO,
                source="FRED",
                name="ISM PMI",
                value=49.2,
                unit="index",
                timestamp="2025-04-30",
                direction=SignalDirection.BEARISH,
                raw_text="ISM PMI: 49.2 (contraction territory)",
                confidence=0.85,
            )
        ],
        textual_signals=[
            Signal(
                signal_id="news_0",
                signal_type=SignalType.TEXTUAL,
                source="Reuters",
                name="OPEC+ holds production cuts",
                timestamp="2025-04-30",
                direction=SignalDirection.BULLISH,
                raw_text="OPEC+ agreed to maintain current production cuts through Q3 2025.",
                confidence=0.7,
            )
        ],
    )


@pytest.fixture
def sample_events() -> EventList:
    return EventList(
        events=[
            DetectedEvent(
                event_id="ev1",
                event_type="OPEC_decision",
                description="OPEC+ maintains production cuts",
                detected_date="2025-04-30",
                affected_region="Global",
                severity=EventSeverity.MEDIUM,
                supply_impact="tightening",
                source="Reuters",
                confidence=0.8,
            )
        ],
        statistical_anomalies=[],
        detection_summary="One OPEC decision detected.",
    )


@pytest.fixture
def sample_evidence() -> StructuredEvidenceBundle:
    return StructuredEvidenceBundle(
        facts=[
            EvidenceItem(
                evidence_id="e1",
                classification=EvidenceClass.FACT,
                source="yfinance",
                content="Brent crude at $82.50/bbl on 2025-04-30",
                direction=SignalDirection.BULLISH,
                weight=0.9,
            )
        ],
        causal_claims=[
            EvidenceItem(
                evidence_id="e2",
                classification=EvidenceClass.CAUSAL_CLAIM,
                source="Reuters",
                content="OPEC+ cuts will tighten Gulf Coast crude supply",
                direction=SignalDirection.BULLISH,
                weight=0.65,
            )
        ],
        speculations=[
            EvidenceItem(
                evidence_id="e3",
                classification=EvidenceClass.SPECULATION,
                source="Analyst",
                content="US may enter recession in H2 2025",
                direction=SignalDirection.BEARISH,
                weight=0.3,
            )
        ],
        forecasts=[],
        contradiction_count=0,
        dominant_narrative="bullish supply tightening",
    )


@pytest.fixture
def sample_temporal() -> TemporalImpactMap:
    return TemporalImpactMap(
        impacts=[
            TemporalImpact(
                evidence_id="e2",
                trigger_event="OPEC+ production cut",
                propagation_path="OPEC cut → tanker rates → Gulf Coast imports → inventory tightening",
                impact_lag_weeks=6,
                persistence_weeks=4,
                weekly_intensity=[0.5, 1.0, 0.8, 0.5],
                affected_variable="inventory_level",
                direction=SignalDirection.BULLISH,
            )
        ],
        peak_impact_week=7,
        total_horizon_coverage=0.5,
        reasoning_summary="Primary impact delayed 6 weeks due to tanker transit time.",
    )


@pytest.fixture
def sample_debate() -> DebateSummary:
    return DebateSummary(
        positions=[
            DebatePosition(
                agent_name="Bullish Supply Risk Agent",
                agent_role="Supply tightening",
                stance=DebateStance.BULLISH,
                confidence=0.70,
                key_arguments=["OPEC cuts reduce supply", "refinery margins elevated"],
                evidence_refs=["e1", "e2"],
                counterarguments_acknowledged=["PMI below 50 signals weak demand"],
                price_direction_call="up",
                price_magnitude_estimate_pct=4.5,
            ),
            DebatePosition(
                agent_name="Bearish Demand Agent",
                agent_role="Demand weakness",
                stance=DebateStance.BEARISH,
                confidence=0.60,
                key_arguments=["PMI contraction", "EV adoption rising"],
                evidence_refs=["fred_pmi"],
                counterarguments_acknowledged=["Supply cuts are real and significant"],
                price_direction_call="down",
                price_magnitude_estimate_pct=-2.0,
            ),
            DebatePosition(
                agent_name="Skeptic Agent",
                agent_role="Evidence challenge",
                stance=DebateStance.NEUTRAL,
                confidence=0.45,
                key_arguments=["OPEC compliance historically poor", "demand data is lagged"],
                evidence_refs=[],
                counterarguments_acknowledged=[],
                price_direction_call="flat",
                price_magnitude_estimate_pct=0.0,
            ),
            DebatePosition(
                agent_name="Historical Analog Agent",
                agent_role="Historical analogue",
                stance=DebateStance.BULLISH,
                confidence=0.55,
                key_arguments=["2022 Russia-Ukraine analog: +30% in 8w after cut announcement"],
                evidence_refs=[],
                counterarguments_acknowledged=["Macro context differs significantly"],
                price_direction_call="up",
                price_magnitude_estimate_pct=5.0,
            ),
        ],
        consensus_direction="up",
        agreement_score=0.55,
        key_disagreements=["Demand outlook uncertain", "OPEC compliance unknown"],
        strongest_bull_argument="OPEC+ cuts reducing available crude for Gulf Coast refineries",
        strongest_bear_argument="PMI below 50 historically predicts diesel demand contraction",
        skeptic_flags=["OPEC compliance historically averages 70%", "demand data lagged 4-6 weeks"],
        historical_analog="2022 Russia-Ukraine: OPEC compensatory cuts led to 8-week supply tightening",
        synthesis_narrative="Supply tightening from OPEC cuts faces demand headwinds from weak PMI.",
    )


@pytest.fixture
def sample_calibration() -> CalibrationResult:
    return CalibrationResult(
        directional_confidence=0.62,
        upside_risk_pct=12.0,
        downside_risk_pct=8.0,
        uncertainty_regime=UncertaintyRegime.MODERATE,
        interval_multiplier=1.15,
        direction_bias_pct=2.5,
        calibration_notes="Moderate uncertainty driven by demand/supply conflict.",
        key_risk_factors=["OPEC compliance", "US recession risk", "hurricane season"],
    )


# ── Tests ──────────────────────────────────────────────────────────────────

class TestModels:
    def test_signal_bundle_all_signals(self, sample_signals):
        all_sigs = sample_signals.all_signals()
        assert len(all_sigs) == 3  # market + macro + textual
        for s in all_sigs:
            assert isinstance(s, Signal)

    def test_evidence_bundle_all_items(self, sample_evidence):
        all_items = sample_evidence.all_items()
        assert len(all_items) == 3  # 1 fact + 1 causal + 1 spec
        classes = {i.classification for i in all_items}
        assert EvidenceClass.FACT in classes
        assert EvidenceClass.CAUSAL_CLAIM in classes

    def test_baseline_forecast_fields(self, sample_baseline):
        assert sample_baseline.trend_direction == "up"
        assert len(sample_baseline.weekly_forecasts) == 8
        assert sample_baseline.weekly_forecasts[0].week_offset == 1
        assert sample_baseline.weekly_forecasts[-1].week_offset == 8

    def test_debate_summary_positions(self, sample_debate):
        assert len(sample_debate.positions) == 4
        names = [p.agent_name for p in sample_debate.positions]
        assert "Bullish Supply Risk Agent" in names
        assert "Skeptic Agent" in names


class TestForecastAdjustment:
    def test_adjustment_direction(self, sample_baseline, sample_calibration, sample_debate):
        from agents import forecast_adjustment
        result = forecast_adjustment.run(sample_baseline, sample_calibration, sample_debate)
        assert result.direction_call in ("up", "down", "flat")
        assert len(result.adjusted_weekly) == 8
        assert len(result.scenarios) == 3

    def test_adjustment_applies_bias(self, sample_baseline, sample_calibration, sample_debate):
        from agents import forecast_adjustment
        result = forecast_adjustment.run(sample_baseline, sample_calibration, sample_debate)
        # With positive bias, adjusted week-8 > baseline week-8
        adj_last = result.adjusted_weekly[-1].point
        base_last = sample_baseline.weekly_forecasts[-1].point
        assert adj_last > base_last  # 2.5% bias should make it larger

    def test_scenarios_sum_to_one(self, sample_baseline, sample_calibration, sample_debate):
        from agents import forecast_adjustment
        result = forecast_adjustment.run(sample_baseline, sample_calibration, sample_debate)
        total_prob = sum(sc.probability for sc in result.scenarios)
        assert abs(total_prob - 1.0) < 0.01

    def test_ci_multiplier_effect(self, sample_baseline, sample_calibration, sample_debate):
        from agents import forecast_adjustment
        result = forecast_adjustment.run(sample_baseline, sample_calibration, sample_debate)
        adj_pt = result.adjusted_weekly[0]
        base_pt = sample_baseline.weekly_forecasts[0]
        base_ci_width = base_pt.ci_80_high - base_pt.ci_80_low
        adj_ci_width = adj_pt.ci_80_high - adj_pt.ci_80_low
        # CI should be wider with multiplier=1.15
        assert adj_ci_width > base_ci_width * 1.0


class TestRiskCalibration:
    def test_uncertainty_regime_type(self, sample_baseline, sample_debate, sample_evidence):
        from agents import risk_calibration
        with patch("agents.risk_calibration.chat_json") as mock_chat:
            mock_chat.return_value = {
                "directional_confidence": 0.62,
                "upside_risk_pct": 12.0,
                "downside_risk_pct": 8.0,
                "uncertainty_regime": "moderate",
                "interval_multiplier": 1.15,
                "direction_bias_pct": 2.5,
                "calibration_notes": "test",
                "key_risk_factors": ["risk A", "risk B"],
            }
            result = risk_calibration.run(sample_baseline, sample_debate, sample_evidence)
            assert result.uncertainty_regime == UncertaintyRegime.MODERATE
            assert 0.0 <= result.directional_confidence <= 1.0
            assert result.interval_multiplier > 0


class TestTaskPlanner:
    def test_task_planner_returns_task(self):
        from agents import task_planner
        with patch("agents.task_planner.chat_json") as mock_chat:
            mock_chat.return_value = {
                "commodity": "diesel",
                "region": "Gulf Coast",
                "horizon_weeks": 8,
                "forecast_variable": "inventory_level",
                "sub_tasks": ["fetch data", "analyze signals"],
                "context_notes": "test",
            }
            result = task_planner.run("test query")
            assert isinstance(result, ForecastTask)
            assert result.commodity == "diesel"
            assert result.horizon_weeks == 8


class TestReportGenerator:
    def test_report_generation(
        self, sample_task, sample_baseline, sample_calibration,
        sample_debate, sample_evidence, sample_temporal, tmp_path, monkeypatch
    ):
        from agents import forecast_adjustment, report_generator
        import core.config as cfg
        monkeypatch.setattr(cfg, "REPORTS_DIR", tmp_path)

        adjusted = forecast_adjustment.run(sample_baseline, sample_calibration, sample_debate)

        with patch("agents.report_generator.chat") as mock_chat, \
             patch("agents.report_generator.REPORTS_DIR", tmp_path), \
             patch("agents.report_generator._generate_forecast_chart", return_value=None):
            mock_chat.return_value = "This is the executive summary."
            report = report_generator.run(
                sample_task, adjusted, sample_debate, sample_evidence, sample_temporal
            )

        assert report.executive_summary == "This is the executive summary."
        assert len(report.evidence_table) > 0
        assert "Week" in report.forecast_table_md

        # Verify files were written
        md_files = list(tmp_path.glob("*.md"))
        json_files = list(tmp_path.glob("*.json"))
        assert len(md_files) == 1
        assert len(json_files) == 1



# ── Live pipeline demo ─────────────────────────────────────────────────────

@pytest.mark.live
class TestLivePipeline:
    """Full end-to-end integration test / demo.

    Runs the complete 10-agent pipeline against live APIs and verifies that
    all output artefacts are produced and structurally sound.

    Keys required: ANTHROPIC_API_KEY, NEWSAPI_KEY
    Optional:      EIA_API_KEY, ALPHA_VANTAGE_KEY

    Run with:
        pytest tests/test_pipeline.py -m live -v -s
    """

    def test_brent_8week_full_run(self, tmp_path, monkeypatch):
        """Complete pipeline: Brent crude, 8-week horizon."""
        import core.config as cfg
        monkeypatch.setattr(cfg, "REPORTS_DIR", tmp_path)

        from main import run
        run("Forecast Brent crude oil price over the next 8 weeks")

        md_files   = list(tmp_path.glob("*.md"))
        json_files = list(tmp_path.glob("*.json"))
        png_files  = list(tmp_path.glob("*.png"))

        assert md_files,   "No markdown report produced"
        assert json_files, "No JSON report produced"
        assert png_files,  "No chart produced"

        report_text = md_files[0].read_text()
        for section in ("Executive Summary", "Forecast", "Evidence", "Debate"):
            assert section in report_text, f"Report missing section: {section}"

    def test_diesel_4week_full_run(self, tmp_path, monkeypatch):
        """Complete pipeline: diesel (heating oil), 4-week horizon."""
        import core.config as cfg
        monkeypatch.setattr(cfg, "REPORTS_DIR", tmp_path)

        from main import run
        run("Forecast diesel inventory for Gulf Coast over 4 weeks", horizon=4)

        assert list(tmp_path.glob("*.md")),   "No markdown report"
        assert list(tmp_path.glob("*.json")), "No JSON report"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
