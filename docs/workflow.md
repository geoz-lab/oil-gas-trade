# Oil & Gas Multi-Agent Supply-Chain Forecasting System
## System Architecture & Workflow

---

## Overview

This system augments statistical price baselines with a multi-agent reasoning pipeline to produce directionally-calibrated, uncertainty-aware forecasts for crude oil and petroleum products. The core hypothesis is that structured adversarial debate between specialized agents outperforms single-model forecasting on directional accuracy and decision quality under uncertainty.

**The baseline answers: *how large are the price moves?***
**The signal pipeline answers: *which direction and when?***

---

## Pipeline Architecture

```
User Query
    │
    ▼
[01] Task Planner ────────────────────────────► ForecastTask
    │                                           {commodity, region, horizon}
    ▼
[02] Baseline Forecast (LightGBM Ensemble) ──► BaselineForecast
    │  2Y + 1Y + 6M windows, 5 quantiles        {price scale, CI bands}
    │  Provides: price level anchor
    ▼
[03] Signal Retrieval ────────────────────────► SignalBundle (raw)
    │  ┌─ LLM keyword expansion                 {market + macro + textual}
    │  ├─ yfinance (Brent, WTI, spreads)
    │  ├─ EIA API v2 (stocks, inventories)
    │  ├─ Alpha Vantage (CPI, GDP, unemployment)
    │  ├─ NewsAPI + 6-month recency filter
    │  └─ EIA weekly text + OPEC press room
    │     Saves: data/cache/YYYYMMDD_signal.json
    ▼
[3b] Signal Judge (Constitutional AI) ────────► JudgmentResult
    │  Screens each signal for:                 {filtered SignalBundle}
    │  • Accuracy (verifiable vs. rumour)
    │  • Recency (< 6 months)
    │  • Source credibility
    │  • Proportionality (claim vs. evidence)
    │  • Independence (no coordinated narrative)
    │  REMOVES misleading signals
    │  HALVES confidence of questionable signals
    ▼
[04] Event Detection ─────────────────────────► EventList
    │  Z-score anomaly detection on prices       {disruptions, sanctions,
    │  LLM scans headlines for supply events      weather, OPEC decisions}
    ▼
[05] Evidence Structuring ────────────────────► StructuredEvidenceBundle
    │  Classifies each piece into:               {facts, causal_claims,
    │  • FACT (verifiable, dated)                 speculations, forecasts}
    │  • CAUSAL_CLAIM (argued connection)
    │  • SPECULATION (analyst opinion)
    │  • FORECAST (external numeric prediction)
    ▼
[06] Temporal Reasoning ──────────────────────► TemporalImpactMap
    │  For each causal claim, estimates:         {lag, persistence,
    │  • impact_lag_weeks                         propagation path}
    │  • persistence_weeks
    │  • propagation_path
    │  e.g. OPEC cut → tanker rates → refinery input → inventory draw
    ▼
[07] Multi-Agent Debate (parallel) ───────────► DebateSummary
    │  ┌─ Bullish Supply Risk Agent              {consensus direction,
    │  ├─ Bearish Demand Agent                    agreement score,
    │  ├─ Skeptic Agent                           agent positions}
    │  └─ Historical Analog Agent
    │  → 5th synthesis LLM call
    │  Provides: directional call + confidence
    ▼
[08] Risk Calibration ────────────────────────► CalibrationResult
    │  Agreement across agents → confidence      {directional_confidence,
    │  Skeptic objections → wider intervals       uncertainty_regime,
    │  Maps: debate → numeric adjustments         CI multiplier, bias%}
    ▼
[09] Forecast Adjustment ─────────────────────► AdjustedForecast
    │  Applies calibration to baseline:          {adjusted weekly series,
    │  • direction_bias_pct added to median       3 scenarios, CI bands}
    │  • CI width × interval_multiplier
    │  • Bull / base / bear scenario series
    ▼
[RL] Trading Robot ───────────────────────────► RobotDecision
    │  Qwen2.5-0.5B (frozen text encoder)        {direction: down/hold/up,
    │  + trainable 3-layer MLP policy head         confidence, down/hold/up
    │  REINFORCE + EMA baseline subtraction        probs, combined_signal}
    │  Trains on 3Y weekly price windows
    │  Combined signal: baseline×25% + debate×45% + robot×30%
    │  Checkpoint: robot/checkpoints/trading_policy.pt
    ▼
[10] Report Generator ────────────────────────► FinalReport
     LLM executive summary (analyst-grade)      {markdown + JSON + PNG}
     Evidence table, temporal timeline
     Debate positions, RL robot signal table
     Forecast chart with signal annotations
```

---

## Key Design Decisions

### 1. Baseline = Price Scale Anchor
The LightGBM quantile ensemble (2Y + 1Y + 6M windows) establishes *how large* the price move could be and provides statistically calibrated confidence intervals. It is intentionally run **before** any LLM reasoning to prevent anchoring bias.

### 2. Signals = Directional Driver
The signal pipeline (steps 3–7) determines *which direction* prices move and *when* the impact materializes. The baseline's point forecast is adjusted by `direction_bias_pct` derived from the adversarial debate.

### 3. Constitutional AI Signal Judge (Step 3b)
Inspired by RLHF / Constitutional AI principles, the judge evaluates every signal against five quality criteria before it can influence the forecast:

| Criterion | What it catches |
|-----------|----------------|
| Accuracy | Unverified rumors, no primary source |
| Recency | Articles older than 6 months |
| Credibility | Self-interested sources (traders, governments) |
| Proportionality | Exaggerated price impact claims |
| Independence | Coordinated market-moving narratives |

Signals are rated **reliable** / **questionable** / **misleading**. Misleading signals are removed; questionable ones have their confidence halved.

### 4. LLM Keyword Expansion
Rather than hardcoded queries, the system uses the LLM to generate 12 commodity-specific search phrases covering supply/demand drivers, geopolitical risks, infrastructure, and macro factors. This enables the same codebase to adapt to any commodity (Brent, WTI, diesel, natural gas, LNG).

### 5. Adversarial Debate Design
Four agents with opposing mandates run in parallel via `ThreadPoolExecutor`:
- **Bullish Supply Risk**: focus on shortages, disruptions, supply cuts
- **Bearish Demand**: focus on weak demand, oversupply, recession risk
- **Skeptic**: challenge evidence quality, flag stale data
- **Historical Analog**: surface prior analogous episodes (2008, 2014, 2020, 2022)

This structure prevents the LLM from defaulting to the most salient narrative in the news.

### 6. Temporal Reasoning
Most forecasting systems ignore *when* a causal chain materializes. Step 6 explicitly estimates propagation paths and lag times (e.g., "OPEC cut → tanker rates rise (W2) → Gulf Coast refinery input cost up (W4) → inventory draw (W6)") and annotates these onset weeks on the forecast chart.

---

## Data Flow Summary

```
Raw query string
    → ForecastTask (structured intent)
    → BaselineForecast (statistical anchor, price scale)
    → SignalBundle (raw evidence, 30–60 signals)
    → JudgmentResult (filtered, 20–50 signals)
    → EventList (detected disruptions)
    → StructuredEvidenceBundle (classified facts/claims/speculation)
    → TemporalImpactMap (lag-adjusted causal chains)
    → DebateSummary (4-agent directional consensus)
    → CalibrationResult (numeric uncertainty parameters)
    → AdjustedForecast (signal-adjusted 8-week series + scenarios)
    → RobotDecision (RL trading signal + combined weighted vote)
    → FinalReport (markdown + JSON + chart)
```

### 7. RL Trading Robot (between [9] and [10])
A reinforcement-learning trading signal that operates independently of the LLM debate chain, providing a third perspective grounded in historical price patterns.

**Architecture**

| Layer | Detail |
|-------|--------|
| Encoder | Qwen2.5-0.5B (frozen — 494M params, 896-dim hidden) |
| Pooling | Attention-mask-weighted mean pool of last hidden states |
| Policy head | `Linear(896→256) → LayerNorm → GELU → Dropout(0.15) → Linear(256→64) → GELU → Linear(64→3)` |
| Parameters | 247K trainable (head only) |
| Actions | 0 = down, 1 = hold, 2 = up |

**Training (REINFORCE)**

- 3 years of weekly price data from yfinance, sliding windows (step=2)
- Reward: +1.0 correct direction, -1.0 wrong, +0.3 hold-on-flat, -0.2 hold-on-trend (flat threshold = ±1.5%)
- Variance reduction: exponential moving-average baseline subtraction (α = 0.05)
- Optimizer: Adam, lr = 3e-4, gradient norm clip 1.0
- Saves best checkpoint by directional accuracy: `robot/checkpoints/trading_policy.pt`
- On subsequent runs, if ticker or horizon changed, retraining triggers automatically

**State serialization**

The current pipeline state is serialized as a multi-section text block fed to the Qwen encoder:
- Commodity, region, horizon; last observed price
- Baseline model (trend %, walk-forward directional accuracy, CI bands)
- Signal quality counts (kept / removed / flagged)
- Evidence structure (facts / causal claims / speculations)
- Debate consensus (direction, agreement, strongest bull/bear arguments)
- Temporal dynamics (causal chain count, peak impact week)
- Adjusted forecast (direction call, magnitude %, confidence, regime)

**Combined signal weighting**

| Source | Base weight | Scaling |
|--------|------------|---------|
| Statistical baseline | 25% | Fixed |
| Multi-agent debate | 45% | × (0.5 + 0.5 × agreement_score) |
| RL robot | 30% | × (0.5 + 0.5 × robot_confidence) |

Threshold: score > +0.20 → UP, score < −0.20 → DOWN, otherwise HOLD.

The combined signal appears in the final report's `## RL Trading Robot Signal` section alongside individual action probabilities and training metadata.

---

## API Keys & Data Sources

| Source | Purpose | Key Variable |
|--------|---------|-------------|
| Anthropic Claude | LLM calls (all reasoning steps) | `ANTHROPIC_API_KEY` |
| yfinance | Brent, WTI, HO, RB, NG futures | (free) |
| EIA API v2 | US crude/distillate/gasoline stocks | `EIA_API_KEY` |
| Alpha Vantage | CPI, real GDP, unemployment | `ALPHA_VANTAGE_KEY` |
| NewsAPI | Energy headlines with keyword search | `NEWSAPI_KEY` |
| EIA / OPEC | Web scrape (no key required) | — |

---

## Output Files

| File | Location | Description |
|------|---------|-------------|
| `YYYYMMDD_signal.json` | `data/cache/` | All raw signals + expanded keywords |
| `forecast_YYYYMMDD_HHMMSS.md` | `reports/` | Full markdown report (includes RL robot signal section) |
| `forecast_YYYYMMDD_HHMMSS.json` | `reports/` | Machine-readable report |
| `forecast_chart_YYYYMMDD_HHMMSS.png` | `reports/` | Two-panel forecast chart |
| `trading_policy.pt` | `robot/checkpoints/` | REINFORCE policy head checkpoint (ticker + horizon keyed) |

---

## Running the System

```bash
conda activate oil-gas-trade
python main.py "Forecast Brent crude oil price over the next 8 weeks"
python main.py "Forecast diesel inventory for Gulf Coast over 4 weeks" --horizon 4
```

Run unit tests (no API keys needed):
```bash
pytest tests/test_pipeline.py -v
```
