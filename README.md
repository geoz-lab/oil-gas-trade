# Oil & Gas Multi-Agent Supply-Chain Forecasting System

A production-grade pipeline that combines a **statistical price baseline** with **adversarial LLM debate**, **Constitutional AI signal filtering**, **temporal supply-chain reasoning**, and a **reinforcement-learning trading robot** to produce directionally-calibrated, uncertainty-aware forecasts for crude oil and petroleum products.

**The baseline answers: *how large are the price moves?***
**The signal pipeline answers: *which direction and when?***

---

## The Core Problem

Statistical price models (ARIMA, LightGBM) are excellent at anchoring price *scale* and *confidence intervals*, but they systematically miss major directional turning points — OPEC decisions, geopolitical shocks, refinery outages — because these events have no price-history signature. Our backtest confirms this: LightGBM alone scored **0/4 directional hits** across four distinct 2023–2024 oil market regimes (see [Backtest Results](#backtest-results)).

The signal pipeline (agents 03a–09) exists to close this gap: multi-source evidence → quality filtering → structured reasoning → adversarial debate → calibrated directional adjustment.

---

## Pipeline Architecture

```
User Query
    │
    ▼
[01] Task Planner ────────────────────────────► ForecastTask
    │  LLM extracts: commodity, region,          {commodity, region, horizon,
    │  horizon, forecast_variable, sub_tasks       forecast_variable, sub_tasks}
    ▼
[02] Baseline Forecast (LightGBM Ensemble) ──► BaselineForecast
    │  Three historical windows: 2Y / 1Y / 6M    {price scale, CI bands q05–q95}
    │  Five quantiles: q05, q10, q50, q90, q95   Anchors magnitude — not direction
    │  Walk-forward validation → window weights
    ▼
[03a] Signal Retrieval ───────────────────────► SignalBundle (raw, 30–60 signals)
    │  ┌─ LLM keyword expansion (12 phrases)      {market + macro + textual}
    │  ├─ yfinance (Brent, WTI, spreads)
    │  ├─ EIA API v2 (stocks, inventories)
    │  ├─ Alpha Vantage (CPI, GDP, unemployment)
    │  ├─ NewsAPI + 6-month recency filter
    │  └─ EIA weekly text + OPEC press room
    │     Saves: data/cache/YYYYMMDD_signal.json
    ▼
[03b] Signal Judge (Constitutional AI) ───────► JudgmentResult
    │  Five quality criteria per signal:          {filtered SignalBundle, 20–50 signals}
    │  • Accuracy  • Recency  • Credibility
    │  • Proportionality  • Independence
    │  REMOVES misleading · HALVES questionable confidence
    ▼
[04] Event Detection ─────────────────────────► EventList
    │  Z-score anomaly detection (threshold: 2.0) {disruptions, sanctions,
    │  20-week rolling window on Brent & HO=F      weather, OPEC decisions}
    │  LLM scans headlines for supply events
    ▼
[05] Evidence Structuring ────────────────────► StructuredEvidenceBundle
    │  Classifies each signal into:               {facts (w=0.85–1.0),
    │  • FACT — verifiable, dated, sourced          causal_claims (0.55–0.75),
    │  • CAUSAL_CLAIM — argued connection           speculations (0.2–0.45),
    │  • SPECULATION — opinion, ungrounded          forecasts (0.5–0.7)}
    │  • FORECAST — external numeric prediction
    ▼
[06] Temporal Reasoning ──────────────────────► TemporalImpactMap
    │  Per causal claim / event, estimates:       {lag, persistence,
    │  • impact_lag_weeks (0–10)                   propagation path,
    │  • persistence_weeks                          weekly_intensity}
    │  • propagation_path (step-by-step chain)
    │  e.g. OPEC cut → tanker rates (W2) → refinery cost (W4) → inventory draw (W6)
    ▼
[07] Multi-Agent Debate (parallel) ───────────► DebateSummary
    │  ┌─ Bullish Supply Risk Agent               {consensus direction,
    │  ├─ Bearish Demand Agent                     agreement score 0–1,
    │  ├─ Skeptic Agent                            agent positions,
    │  └─ Historical Analog Agent                  strongest arguments}
    │  → 5th synthesis LLM call
    ▼
[08] Risk Calibration ────────────────────────► CalibrationResult
    │  Agreement score → directional_confidence   {directional_confidence,
    │  Skeptic flags → wider CI multiplier         uncertainty_regime,
    │  Maps debate → numeric adjustments           CI_multiplier, bias%}
    ▼
[09] Forecast Adjustment ─────────────────────► AdjustedForecast
    │  Applies calibration to baseline:           {adjusted weekly series,
    │  • direction_bias_pct ramped linearly         3 scenarios, CI bands}
    │    (30% of max at W1, 100% at W8)
    │  • CI width × interval_multiplier
    │  • Bull / base / bear scenario series
    ▼
[RL] Trading Robot ───────────────────────────► RobotDecision
    │  Qwen2.5-0.5B (frozen, 494M params)         {direction: down/hold/up,
    │  + trainable 3-layer MLP policy head (247K)   confidence, action probs,
    │  REINFORCE + EMA baseline (α=0.05)            combined_signal}
    │  Trains on 3Y weekly price sliding windows
    │  Combined: baseline×25% + debate×45% + robot×30%
    ▼
[10] Report Generator ────────────────────────► FinalReport
     LLM executive summary (6-sentence template) {markdown + JSON + PNG}
     Evidence table · temporal timeline
     Debate positions · RL robot signal table
     Forecast chart: 26w historical + 8w quantile bands + signal annotations
```

---

## Why 10 Agents?

| # | Agent | Role | LLM? |
|---|-------|------|------|
| 01 | Task Planner | Parse free-text query → structured ForecastTask | ✓ |
| 02 | Baseline Forecast | LightGBM quantile ensemble, price scale anchor | — |
| 03a | Signal Retrieval | Multi-source evidence collection (yfinance, EIA, NewsAPI…) | ✓ keyword expansion |
| 03b | Signal Judge | Constitutional AI quality filter, harm flag assignment | ✓ |
| 04 | Event Detection | Z-score anomalies + LLM headline extraction | ✓ |
| 05 | Evidence Structuring | Classify signals into FACT / CAUSAL_CLAIM / SPECULATION / FORECAST | ✓ |
| 06 | Temporal Reasoning | Estimate supply-chain propagation lags and onset weeks | ✓ |
| 07 | Debate System | Four adversarial agents (parallel) + synthesis moderator | ✓ ×5 |
| 08 | Risk Calibration | Convert debate outcome → numeric uncertainty parameters | ✓ |
| 09 | Forecast Adjustment | Apply calibration to baseline (bias ramp, CI scaling, scenarios) | — |
| RL | Trading Robot | Independent REINFORCE policy on historical prices | — |
| 10 | Report Generator | Analyst-grade markdown + JSON + two-panel chart | ✓ |

**Total LLM calls per run:** ~10 (serial) + 4 (parallel debate agents) = ~14 Claude Sonnet calls.
System prompts are cached with `cache_control=ephemeral` to reduce token cost on repeated runs.

---

## Key Design Decisions

### 1. Baseline = Price Scale Anchor Only
The LightGBM ensemble runs **before** any LLM reasoning to prevent anchoring bias. It answers *"how volatile is this market and what are the plausible price ranges?"* — not direction. Three historical windows (2Y / 1Y / 6M) are ensembled using walk-forward directional accuracy as weights.

Window weighting: `dir_acc ≥ 0.60 → w=1.0 | 0.50–0.60 → w=0.6 | < 0.50 → w=0.2`

### 2. Constitutional AI Signal Judge
Inspired by RLHF / Constitutional AI, every retrieved signal is evaluated against five quality criteria before it can influence the forecast:

| Criterion | What it catches |
|-----------|----------------|
| Accuracy | Unverified rumors, no primary source |
| Recency | Articles older than 6 months |
| Credibility | Self-interested sources (traders, governments) |
| Proportionality | Exaggerated price impact claims |
| Independence | Coordinated market-moving narratives |

Judgment tiers: **reliable** (full weight) → **questionable** (50% confidence discount) → **misleading** (removed). Processed in batches of 20 signals per LLM call.

### 3. Evidence Epistemic Taxonomy
Agent 05 classifies every signal into four tiers with explicit weight ranges:
- **FACT** (0.85–1.0) — verifiable, dated, primary-sourced. Feeds debate at full weight.
- **CAUSAL_CLAIM** (0.55–0.75) — argued connection, potential lag.
- **SPECULATION** (0.2–0.45) — analyst opinion, "could / may / might". Kept but downweighted.
- **FORECAST** (0.5–0.7) — external numeric predictions from named sources.

### 4. Temporal Reasoning — Explicit Supply-Chain Lags
Most forecasting systems treat causal effects as immediate. Agent 06 explicitly estimates propagation paths and onset weeks for each causal claim, grounded in commodity-specific reference chains:

| Event | Propagation | Lag |
|-------|-------------|-----|
| OPEC cut | tanker tightening → US imports ↓ → refinery throughput ↓ → inventory draw | 6–10w |
| Gulf hurricane | direct production loss → immediate inventory drop | 0–2w |
| Sanctions | shipping re-routing → supply re-route → price spike | 4–8w |
| PMI drop | demand destruction → inventory build | 2–6w |
| Port congestion | delayed cargo arrivals → inventory draw-down | 1–3w |

### 5. Adversarial Debate Design
Four agents with opposing mandates run in parallel via `ThreadPoolExecutor` (4 workers). Each sees the same evidence but argues from its assigned perspective:
- **Bullish Supply Risk** — shortages, OPEC cuts, sanctions, refinery outages
- **Bearish Demand** — recession risk, EV adoption, oversupply, PMI contraction
- **Skeptic** — challenges evidence quality, OPEC compliance gaps, stale data
- **Historical Analog** — surfaces prior analogous episodes (2008, 2014, 2020, 2022)

A 5th synthesis call produces consensus direction and agreement score. This structure prevents the LLM from defaulting to the most salient narrative in the news.

### 6. Bias Ramp in Forecast Adjustment
Direction bias is applied non-linearly: 30% of max at week 1, growing to 100% by week 8. Near-term prices are anchored closer to the statistical baseline; the full directional view matures over the horizon.

Direction call thresholds: `total_pct > 2.0% → "up" | < -2.0% → "down" | else → "flat"`

### 7. RL Trading Robot — Independent Signal
An independent REINFORCE policy provides a third perspective grounded in historical price patterns rather than LLM reasoning. The combined signal weights:

| Source | Base weight | Scaling |
|--------|------------|---------|
| Statistical baseline | 25% | Fixed |
| Multi-agent debate | 45% | × (0.5 + 0.5 × agreement_score) |
| RL robot | 30% | × (0.5 + 0.5 × robot_confidence) |

Score > +0.20 → **UP** · Score < −0.20 → **DOWN** · Otherwise → **HOLD**

### 8. Caching Strategy
| Source | TTL | Rationale |
|--------|-----|-----------|
| yfinance market signals | 1h | Price data changes intraday |
| EIA API v2 | 12h | Weekly releases, stable |
| Alpha Vantage macro | 24h | Monthly/quarterly data |
| NewsAPI | 3h | News freshness matters for signal quality |
| LLM keyword expansion | 24h | Commodity-specific, stable across a day |

---

## Backtest Results

Historical validation of the LightGBM baseline (Agent 02) against realized weekly Brent/WTI prices. No API keys required — yfinance only.

| Case | Cutoff | Last Price | Baseline Dir. | Actual Dir. | DirHit | CI-80 | MAE |
|------|--------|-----------|--------------|------------|--------|-------|-----|
| Brent 2023-Q4 decline | 2023-10-06 | $84.58 | up | down | ✗ | 25% | $5.20 |
| Brent 2024-Q1 rally | 2024-01-05 | $78.76 | down | up | ✗ | 50% | $3.79 |
| Brent 2024-Q2 peak | 2024-04-12 | $90.45 | flat | down | ✗ | 38% | $6.94 |
| WTI 2024-Q3 slump | 2024-07-05 | $83.16 | flat | down | ✗ | 25% | $6.06 |
| **Mean** | | | | | **0%** | **34%** | **$5.50** |

**Key takeaway:** The statistical baseline systematically forecasts near-flat/continuation and misses all four major turning points — because supply shocks, OPEC decisions, and geopolitical events carry no price-history signature. This is the precise gap agents 03a–09 are designed to fill. See `tests/test_backtest.py` for reproducible results.

Run the backtest yourself:
```bash
conda activate oil-gas-trade
python -m pytest tests/test_backtest.py -v -s
```

---

## Quickstart

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate oil-gas-trade

# 2. Set API keys
cp .env.example .env
# Edit .env — ANTHROPIC_API_KEY and NEWSAPI_KEY are required;
# EIA_API_KEY and ALPHA_VANTAGE_KEY improve signal quality

# 3. Run a forecast
python main.py "Forecast Brent crude oil price over the next 8 weeks"
python main.py "Forecast diesel inventory for Gulf Coast over 4 weeks" --horizon 4

# 4. Unit tests (no API keys required, ~5s)
python -m pytest tests/test_pipeline.py -m "not live" -v

# 5. Historical backtest (yfinance only, no API keys, ~90s)
python -m pytest tests/test_backtest.py -v -s

# 6. Full pipeline integration tests (all API keys, ~5–10 min)
python -m pytest tests/test_pipeline.py tests/test_backtest.py -m live -v -s
```

---

## API Keys & Data Sources

| Source | Purpose | Variable | Required? |
|--------|---------|----------|-----------|
| Anthropic Claude | All LLM calls (planning, judge, debate, report) | `ANTHROPIC_API_KEY` | **Yes** |
| yfinance | Brent, WTI, HO, RB, NG futures | — | Free |
| EIA API v2 | US crude/distillate/gasoline stocks | `EIA_API_KEY` | Recommended |
| Alpha Vantage | CPI, real GDP, unemployment | `ALPHA_VANTAGE_KEY` | Recommended |
| NewsAPI | Energy headlines with keyword search | `NEWSAPI_KEY` | **Yes** |
| EIA / OPEC scrapers | Weekly petroleum text, OPEC press room | — | Free |

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `YYYYMMDD_signal.json` | `data/cache/` | All raw signals + LLM-expanded keywords |
| `forecast_YYYYMMDD_HHMMSS.md` | `reports/` | Full markdown report (all sections) |
| `forecast_YYYYMMDD_HHMMSS.json` | `reports/` | Machine-readable Pydantic model dump |
| `forecast_chart_YYYYMMDD_HHMMSS.png` | `reports/` | Two-panel chart: historical + 8w forecast + evidence |
| `trading_policy.pt` | `robot/checkpoints/` | REINFORCE policy head (keyed to ticker + horizon) |

The two-panel chart:
- **Top panel:** 26w historical price + quantile bands (95%, 80%) + scenario lines + temporal signal onset annotations (up to 4 events)
- **Bottom panel:** Top 12 evidence items by weight, color-coded by direction (green=bullish, red=bearish, gray=neutral)

---

## Project Structure

```
oil+gas_trade/
├── main.py                          # Orchestrator — runs all 11 steps in sequence
├── environment.yml                  # Conda environment (Python 3.11)
├── pytest.ini                       # Registers the 'live' pytest mark
├── core/
│   ├── config.py                    # API keys, paths, model settings
│   ├── llm.py                       # Anthropic client with prompt caching
│   └── models.py                    # Pydantic models — typed spine connecting all agents
├── agents/
│   ├── 01_task_planner.py
│   ├── 02_baseline_forecast.py      # LightGBM multi-window quantile ensemble
│   ├── 03a_signal_retrieval.py      # Multi-source evidence collection
│   ├── 03b_signal_judge.py          # Constitutional AI quality filter
│   ├── 04_event_detection.py        # Z-score anomaly + LLM event extraction
│   ├── 05_evidence_structuring.py   # FACT / CAUSAL_CLAIM / SPECULATION / FORECAST
│   ├── 06_temporal_reasoning.py     # Supply-chain propagation lag estimation
│   ├── 07_debate_system.py          # 4-agent parallel debate + synthesis
│   ├── 08_risk_calibration.py       # Debate → numeric uncertainty parameters
│   ├── 09_forecast_adjustment.py    # Apply calibration to baseline
│   └── 10_report_generator.py       # Markdown + JSON + PNG chart
├── robot/
│   ├── environment.py               # TradingEnvironment (yfinance, reward function)
│   ├── model.py                     # Qwen2.5-0.5B encoder + PolicyHead (247K trainable)
│   ├── train.py                     # REINFORCE training loop with EMA baseline
│   └── agent.py                     # Inference + combined weighted signal
├── docs/
│   ├── workflow.md                  # Full technical reference (per-agent detail)
│   └── workflow.html                # Rendered interactive documentation
└── tests/
    ├── test_pipeline.py             # Unit tests (no API keys) + live demo tests
    └── test_backtest.py             # Baseline & full-pipeline historical validation
```

---

## RL Trading Robot — Architecture Detail

| Layer | Detail |
|-------|--------|
| Encoder | Qwen2.5-0.5B (frozen, 494M params, 896-dim hidden) |
| Pooling | Attention-mask weighted mean of last hidden states |
| Policy head | `Linear(896→256) → LayerNorm → GELU → Dropout(0.15) → Linear(256→64) → GELU → Linear(64→3)` |
| Trainable params | 247K (head only — Qwen weights are frozen) |
| Actions | 0 = down, 1 = hold, 2 = up |
| Algorithm | REINFORCE + EMA baseline subtraction (α=0.05) |
| Training data | 3Y of weekly prices, sliding windows (step=2) |
| Reward | +1.0 correct direction · −1.0 wrong · +0.3 hold-on-flat · −0.2 hold-on-trend |
| Flat zone | ±1.5% future return threshold |
| Optimizer | Adam, lr=3e-4, gradient norm clip 1.0 |
| Checkpoint | `robot/checkpoints/trading_policy.pt` (auto-retrain if ticker or horizon changes) |

**State text fed to Qwen at inference:** commodity + region + horizon, last observed price, baseline trend metrics, signal quality counts, evidence structure summary, debate consensus, temporal dynamics, adjusted forecast direction.
