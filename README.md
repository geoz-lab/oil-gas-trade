# Oil & Gas Multi-Agent Supply-Chain Forecasting System

A 10-agent pipeline that combines a statistical price baseline with adversarial LLM debate and a reinforcement-learning trading robot to produce directionally-calibrated, uncertainty-aware forecasts for crude oil and petroleum products.

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
[03a] Signal Retrieval ───────────────────────► SignalBundle (raw)
    │  ┌─ LLM keyword expansion                 {market + macro + textual}
    │  ├─ yfinance (Brent, WTI, spreads)
    │  ├─ EIA API v2 (stocks, inventories)
    │  ├─ Alpha Vantage (CPI, GDP, unemployment)
    │  ├─ NewsAPI + 6-month recency filter
    │  └─ EIA weekly text + OPEC press room
    │     Saves: data/cache/YYYYMMDD_signal.json
    ▼
[03b] Signal Judge (Constitutional AI) ───────► JudgmentResult
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
    │  + trainable 3-layer MLP policy head         confidence, combined_signal}
    │  REINFORCE + EMA baseline subtraction
    │  Trains on 3Y weekly price windows
    │  Combined: baseline×25% + debate×45% + robot×30%
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
The LightGBM quantile ensemble (2Y + 1Y + 6M windows) establishes *how large* the price move could be and provides statistically calibrated confidence intervals. It runs **before** any LLM reasoning to prevent anchoring bias.

### 2. Constitutional AI Signal Judge
Inspired by RLHF / Constitutional AI, the judge evaluates every signal against five quality criteria before it can influence the forecast:

| Criterion | What it catches |
|-----------|----------------|
| Accuracy | Unverified rumors, no primary source |
| Recency | Articles older than 6 months |
| Credibility | Self-interested sources (traders, governments) |
| Proportionality | Exaggerated price impact claims |
| Independence | Coordinated market-moving narratives |

Signals are rated **reliable** / **questionable** / **misleading**. Misleading signals are removed; questionable ones have their confidence halved.

### 3. Adversarial Debate Design
Four agents with opposing mandates run in parallel via `ThreadPoolExecutor`:
- **Bullish Supply Risk** — shortages, disruptions, supply cuts
- **Bearish Demand** — weak demand, oversupply, recession risk
- **Skeptic** — challenges evidence quality, flags stale data
- **Historical Analog** — surfaces analogous prior episodes (2008, 2014, 2020, 2022)

A fifth synthesis call produces the consensus direction and agreement score. This structure prevents the LLM from defaulting to the most salient narrative in the news.

### 4. Temporal Reasoning
Step 6 explicitly estimates propagation paths and lag times for each causal claim — e.g., *"OPEC cut → tanker rates rise (W2) → Gulf Coast refinery input cost up (W4) → inventory draw (W6)"* — and annotates onset weeks on the forecast chart.

### 5. RL Trading Robot
An independent REINFORCE policy trained on 3 years of historical price data, using the full pipeline state as text input to a frozen Qwen2.5-0.5B encoder.

| Layer | Detail |
|-------|--------|
| Encoder | Qwen2.5-0.5B (frozen, 494M params, 896-dim) |
| Policy head | `Linear(896→256) → LayerNorm → GELU → Dropout(0.15) → Linear(256→64) → GELU → Linear(64→3)` |
| Trainable params | 247K (head only) |
| Actions | down / hold / up |
| Algorithm | REINFORCE + EMA baseline subtraction (α=0.05) |

Combined signal weighting:

| Source | Base weight | Scaling |
|--------|------------|---------|
| Statistical baseline | 25% | Fixed |
| Multi-agent debate | 45% | × (0.5 + 0.5 × agreement_score) |
| RL robot | 30% | × (0.5 + 0.5 × robot_confidence) |

Score > +0.20 → **UP** · Score < −0.20 → **DOWN** · Otherwise → **HOLD**

---

## Quickstart

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate oil-gas-trade

# 2. Set your API keys
cp .env.example .env
# edit .env and add your keys

# 3. Run a forecast
python main.py "Forecast Brent crude oil price over the next 8 weeks"
python main.py "Forecast diesel inventory for Gulf Coast over 4 weeks" --horizon 4

# 4. Unit tests (no API keys required)
pytest tests/test_pipeline.py -v
```

---

## API Keys & Data Sources

| Source | Purpose | Variable |
|--------|---------|----------|
| Anthropic Claude | All LLM calls (planning, judge, debate, report) | `ANTHROPIC_API_KEY` |
| yfinance | Brent, WTI, HO, RB, NG futures | free |
| EIA API v2 | US crude/distillate/gasoline stocks | `EIA_API_KEY` |
| Alpha Vantage | CPI, real GDP, unemployment | `ALPHA_VANTAGE_KEY` |
| NewsAPI | Energy headlines with keyword search | `NEWSAPI_KEY` |
| EIA / OPEC | Web scrape | free |

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `YYYYMMDD_signal.json` | `data/cache/` | All raw signals + expanded keywords |
| `forecast_YYYYMMDD_HHMMSS.md` | `reports/` | Full markdown report |
| `forecast_YYYYMMDD_HHMMSS.json` | `reports/` | Machine-readable structured output |
| `forecast_chart_YYYYMMDD_HHMMSS.png` | `reports/` | Two-panel forecast chart |
| `trading_policy.pt` | `robot/checkpoints/` | REINFORCE checkpoint (ticker + horizon keyed) |

---

## Project Structure

```
oil+gas_trade/
├── main.py                        # Orchestrator — runs all 11 steps
├── environment.yml                # Conda environment
├── core/
│   ├── config.py                  # API keys, model settings
│   ├── llm.py                     # Anthropic client with prompt caching
│   └── models.py                  # Pydantic data models (shared)
├── agents/
│   ├── 01_task_planner.py
│   ├── 02_baseline_forecast.py
│   ├── 03a_signal_retrieval.py
│   ├── 03b_signal_judge.py        # Constitutional AI filter
│   ├── 04_event_detection.py
│   ├── 05_evidence_structuring.py
│   ├── 06_temporal_reasoning.py
│   ├── 07_debate_system.py
│   ├── 08_risk_calibration.py
│   ├── 09_forecast_adjustment.py
│   └── 10_report_generator.py
├── robot/
│   ├── environment.py             # TradingEnvironment (yfinance data)
│   ├── model.py                   # Qwen encoder + PolicyHead
│   ├── train.py                   # REINFORCE training loop
│   └── agent.py                   # Inference + combined signal
├── docs/
│   ├── workflow.md
│   └── workflow.html
└── tests/
    └── test_pipeline.py
```
