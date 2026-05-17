# Oil & Gas Multi-Agent Forecasting — Technical Reference
## System Architecture & Workflow

---

## Overview

This system augments a statistical price baseline with a multi-agent reasoning pipeline to produce directionally-calibrated, uncertainty-aware forecasts for crude oil and petroleum products. The core hypothesis is that structured adversarial debate between specialized agents, grounded in quality-filtered multi-source evidence and explicit supply-chain temporal reasoning, outperforms single-model or single-LLM forecasting on directional accuracy under geopolitical and supply shock conditions.

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
    │  REMOVES misleading · HALVES questionable confidence
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
    │  • direction_bias_pct ramped linearly        3 scenarios, CI bands}
    │    (30% at W1, 100% at W8)
    │  • CI width × interval_multiplier
    │  • Bull / base / bear scenario series
    ▼
[RL] Trading Robot ───────────────────────────► RobotDecision
    │  Qwen2.5-0.5B (frozen text encoder)        {direction: down/hold/up,
    │  + trainable 3-layer MLP policy head         confidence, combined_signal}
    │  REINFORCE + EMA baseline (α=0.05)
    │  Combined: baseline×25% + debate×45% + robot×30%
    ▼
[10] Report Generator ────────────────────────► FinalReport
     LLM executive summary (6-sentence template) {markdown + JSON + PNG}
     Evidence table · temporal timeline
     Debate positions · RL robot signal table
     Forecast chart: 26w historical + 8w quantile bands + signal annotations
```

---

## Data Flow Summary

```
Raw query string
    → ForecastTask              (commodity, region, horizon, forecast_variable)
    → BaselineForecast          (statistical anchor: price scale, CI bands)
    → SignalBundle              (30–60 raw signals from 6+ sources)
    → JudgmentResult            (filtered bundle: 20–50 quality signals)
    → EventList                 (detected disruptions with severity + supply_impact)
    → StructuredEvidenceBundle  (classified facts / causal_claims / speculations / forecasts)
    → TemporalImpactMap         (lag-adjusted causal chains with onset weeks + intensity)
    → DebateSummary             (4-agent adversarial consensus + agreement score)
    → CalibrationResult         (directional_confidence, CI_multiplier, direction_bias_pct)
    → AdjustedForecast          (signal-adjusted 8-week series + 3 scenario paths)
    → RobotDecision             (RL signal: direction, confidence, combined vote)
    → FinalReport               (markdown + JSON + two-panel PNG chart)
```

---

## Agent Detail

### [01] Task Planner

**File:** `agents/01_task_planner.py`  
**Purpose:** Decompose a free-text query into a structured `ForecastTask` so all downstream agents share unambiguous intent.

**Output fields:**
| Field | Type | Example |
|-------|------|---------|
| `commodity` | str | "brent_crude", "diesel", "WTI", "natural_gas" |
| `region` | str | "Gulf Coast", "US", "Global" |
| `horizon_weeks` | int | 8 |
| `forecast_variable` | str | "price", "inventory_level", "demand" |
| `sub_tasks` | list[str] | 3–6 analytical steps |
| `context_notes` | str | Constraints, caveats |

**Design:** One LLM call. Prevents downstream agents from misinterpreting scope or commodity (e.g., "diesel" could be inventory or price).

---

### [02] Baseline Forecast — LightGBM Quantile Ensemble

**File:** `agents/02_baseline_forecast.py`  
**Purpose:** Produce a statistically calibrated price scale anchor *before* any LLM reasoning to prevent anchoring bias.

**Three historical windows:**

| Window | Days | ~Weekly obs | Max lag features |
|--------|------|-------------|-----------------|
| 2Y | 730 | ~104 | [1,2,4,8,12,26] |
| 1Y | 365 | ~52 | [1,2,4,8,12] |
| 6M | 182 | ~27 | [1,2,4] |

Adaptive lag rule: `max_lag = min(26, n_weeks // 4)` — prevents NaN dropouts in short windows.

**Feature engineering per window:**
- Lag features (as above)
- `roll_mean_4`, `roll_std_4`, `mom_4` — always included
- `roll_mean_12`, `roll_std_12` — if n_weeks ≥ 24
- `mom_12` — if max_lag ≥ 12
- `week`, `month` — seasonal

**Quantile models:** q05, q10, q50, q90, q95 (each a separate LightGBM regressor)

**LightGBM hyperparameters:**
```python
n_estimators=300, learning_rate=0.03, num_leaves=15,
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1
```

**Walk-forward validation:** 80/20 split → directional accuracy → window weight:
- `dir_acc ≥ 0.60` → weight 1.0
- `dir_acc 0.50–0.60` → weight 0.6
- `dir_acc < 0.50` → weight 0.2

**Recursive multi-step forecast:** Appends q50 prediction to rolling feature window at each step; enforces quantile monotonicity (q05 ≤ q10 ≤ q50 ≤ q90 ≤ q95).

**Trend classification:** `pct_change > 1.5%` → "up" | `< -1.5%` → "down" | else → "flat"

**Ticker map:**
```
brent_crude / brent → BZ=F
WTI / wti / crude → CL=F
diesel / heating_oil → HO=F (USD/gal)
gasoline → RB=F (USD/gal)
natural_gas → NG=F (USD/MMBtu)
LNG → LNG
```

---

### [03a] Signal Retrieval

**File:** `agents/03a_signal_retrieval.py`  
**Purpose:** Collect 30–60 signals from heterogeneous sources with differing epistemological status to reduce source bias and narrative collapse.

**Signal sources and confidence:**

| Source | Type | Confidence | TTL |
|--------|------|-----------|-----|
| yfinance (Brent, WTI, HO, RB, NG, spreads) | MARKET | 0.90 | 1h |
| EIA API v2 (crude/distillate/gasoline stocks, PADD3) | OPERATIONAL | 0.87–0.93 | 12h |
| Alpha Vantage (CPI, real GDP, unemployment) | MACRO | 0.78–0.82 | 24h |
| NewsAPI (6 queries: 4 base + 2 LLM keywords) | TEXTUAL | 0.55 | 3h |
| EIA "This Week in Petroleum" (scraped) | TEXTUAL | 0.82 | 12h |
| OPEC press room (scraped) | TEXTUAL | 0.75 | 12h |

**LLM keyword expansion:** One LLM call generates 12 commodity-specific search phrases covering supply/demand drivers, geopolitical risks, infrastructure, macro factors, and seasonal patterns. Cached 24h.

**Base NewsAPI queries (hardcoded):**
1. `"Brent crude oil price"`
2. `"OPEC production cut supply"`
3. `"oil refinery outage sanctions"`
4. `"crude oil inventory EIA weekly"`

**Direction classification:** `pct_chg > 1.0%` → BULLISH | `< -1.0%` → BEARISH | else → NEUTRAL

**News recency filter:** Articles older than 6 months are discarded before caching.

**Output:** `SignalBundle` with `market_signals`, `macro_signals`, `operational_signals`, `textual_signals`, `expanded_keywords`. Saved to `data/cache/YYYYMMDD_signal.json`.

---

### [03b] Signal Judge — Constitutional AI

**File:** `agents/03b_signal_judge.py`  
**Purpose:** Prevent misleading signals (exaggerated claims, coordinated narratives, stale articles) from biasing the adversarial debate.

**Five evaluation criteria:**
1. **Accuracy** — Is the claim factually verifiable from a primary source?
2. **Recency** — Is the information still actionable (< 6 months)?
3. **Credibility** — Is the source independent vs. self-interested (trader, government)?
4. **Proportionality** — Does the claimed price impact match the evidence magnitude?
5. **Independence** — Is this a coordinated narrative / geopolitical propaganda?

**Harm flags (exact strings used in code):**
`exaggerated` | `unverified` | `outdated` | `single_source` | `market_bias` | `geopolitical_propaganda` | `circular_reference`

**Judgment tiers:**
| Verdict | Action |
|---------|--------|
| `reliable` | Pass at full confidence weight |
| `questionable` | Halve `confidence` field, pass with `harm_flags` set |
| `misleading` | Remove from bundle entirely |

**Batching:** 20 signals per LLM call to manage token budget.

---

### [04] Event Detection

**File:** `agents/04_event_detection.py`  
**Purpose:** Ground the pipeline in objective market movements, not just text narratives.

**Statistical anomaly detection:**
- 20-week rolling mean and standard deviation on Brent (BZ=F) and Heating Oil (HO=F)
- Z-score threshold: **> 2.0** on last 10 trading days → flag as anomaly
- Records: series, date, Z-score, price, direction (spike/crash)

**LLM event extraction:** Scans top 8 textual signals + market anomaly summary for supply events. Output is grounded in provided text — not free hallucination.

**Detected event types:** `refinery_outage`, `pipeline_disruption`, `weather_event`, `sanction`, `OPEC_decision`, `port_congestion`, `demand_shock`, `geopolitical_event`, `shipping_disruption`, `regulatory_change`

**Severity levels:** LOW | MEDIUM | HIGH | CRITICAL  
**Supply impact:** `tightening` | `loosening` | `neutral`

---

### [05] Evidence Structuring

**File:** `agents/05_evidence_structuring.py`  
**Purpose:** Impose an epistemic taxonomy on all signals to prevent unverified narratives from dominating the debate.

**Four evidence classes with weight ranges:**

| Class | Weight | What qualifies |
|-------|--------|----------------|
| FACT | 0.85–1.0 | Verifiable, dated, primary-sourced (price observations, EIA data) |
| CAUSAL_CLAIM | 0.55–0.75 | Argued connection between events; may have lag |
| SPECULATION | 0.2–0.45 | "Could / may / might" language; no primary source |
| FORECAST | 0.5–0.7 | External numeric prediction from a named source |

**Downgrade triggers → SPECULATION:**
- Hedging language: "could lead to", "may", "might", "possible scenario"
- No verifiable primary source
- Price target without attribution

**Narrative determination:**
- Bullish weight sum > Bearish × 1.3 → "bullish supply tightening"
- Bearish weight sum > Bullish × 1.3 → "bearish demand weakness"
- Otherwise → "mixed / contested"

**Contradiction tracking:** Pairs of evidence items with opposing directions on the same variable are flagged as contradictions.

---

### [06] Temporal Reasoning

**File:** `agents/06_temporal_reasoning.py`  
**Purpose:** Estimate *when* price effects materialize — the most novel component. Most forecasting systems treat causal effects as immediate.

**Reference propagation paths (in LLM system prompt):**
| Trigger | Propagation chain | Lag |
|---------|------------------|-----|
| OPEC production cut | tanker tightening → US imports ↓ → refinery throughput ↓ → inventory draw | 6–10w |
| Gulf hurricane / refinery outage | direct production loss → inventory drop | 0–2w |
| Sanctions | shipping re-routing → supply re-route → price spike | 4–8w |
| US PMI drop | demand destruction → inventory build | 2–6w |
| DXY strengthening | non-USD buyers pay more → demand falls → inventory builds | 2–4w |
| Port congestion | delayed cargo arrivals → inventory draw-down | 1–3w |
| Refinery maintenance | planned throughput reduction → inventory tightening | 1–4w |

**Per-impact output (`TemporalImpact`):**
```python
trigger_event, propagation_path,
impact_lag_weeks (int, 0–10),
persistence_weeks (int),
weekly_intensity (list[float], 0.0–1.0, peak=1.0),
affected_variable, direction
```

**Selection criteria for analysis (capped at 15 items):**
- All `CAUSAL_CLAIM` evidence items
- Events with confidence ≥ 0.5
- Facts with direction ≠ NEUTRAL and weight > 0.7

**Onset week annotations** from this agent appear on the forecast chart (top 4 events, by intensity).

---

### [07] Multi-Agent Debate

**File:** `agents/07_debate_system.py`  
**Purpose:** Surface counterevidence and prevent the LLM from defaulting to the most salient news narrative.

**Four agents (parallel, ThreadPoolExecutor with 4 workers):**

| Agent | Mandate |
|-------|---------|
| Bullish Supply Risk | Make strongest case for supply tightening. Focus: OPEC cuts, sanctions, refinery outages, geopolitical risk, shipping constraints. |
| Bearish Demand | Make strongest case for demand weakness. Focus: PMI contraction, EV adoption, recession risk, oversupply, inventory builds. |
| Skeptic | Challenge ALL reasoning. Flag: stale data, single sources, poor OPEC compliance, market-already-pricing-in. No bullish/bearish stance. |
| Historical Analog | Surface most relevant analogous episodes (2008 spike/crash, 2014 market-share war, 2020 COVID, 2021–22 Russia-Ukraine, 2011 Libya, 2005 hurricanes). |

Each agent must acknowledge 1–2 counterarguments (prevents strawman debates).

**5th synthesis call (moderator):**
- Consensus direction: `up` | `down` | `flat` | `contested`
- Agreement score: 0–1 (1 = all agents aligned)
- Strongest bull and bear arguments
- Skeptic's most important flags
- Historical analogue and its implications

**System prompts are cached** (`cache_control=ephemeral`) to reduce token cost on repeated runs.

---

### [08] Risk Calibration

**File:** `agents/08_risk_calibration.py`  
**Purpose:** Convert qualitative debate outcome into calibrated numeric uncertainty parameters for the adjustment engine.

**Key calibration rules:**
- `agreement_score > 0.75` → higher `directional_confidence`
- Skeptic flags ≥ 3 → reduce `directional_confidence` by 10–20 percentage points
- Evidence mostly FACT → narrower intervals (`interval_multiplier < 1.0`)
- High contradiction count → wider intervals (`interval_multiplier > 1.3`)
- Historical analog match → adjust `direction_bias_pct` toward analog outcome
- Contested direction → wider intervals regardless of confidence

**Uncertainty regime thresholds:**
| `directional_confidence` | Regime |
|--------------------------|--------|
| ≥ 0.75 | LOW |
| 0.55–0.75 | MODERATE |
| 0.40–0.55 | HIGH |
| < 0.40 | EXTREME |

**Output:**
```python
CalibrationResult(
    directional_confidence: float (0–1)
    upside_risk_pct: float          # Tail magnitude — not directional bias
    downside_risk_pct: float        # Typical range: 5–25%
    uncertainty_regime: Enum        # LOW / MODERATE / HIGH / EXTREME
    interval_multiplier: float      # Applied to CI width (1.0 = unchanged)
    direction_bias_pct: float       # Signed % adjustment to baseline median
    calibration_notes: str
    key_risk_factors: list[str]     # Top 3–5 risks
)
```

**Scenario scenario probability distributions by regime:**
| Regime | Bull | Base | Bear |
|--------|------|------|------|
| LOW | 25% | 60% | 15% |
| MODERATE | 25% | 50% | 25% |
| HIGH | 30% | 40% | 30% |
| EXTREME | 33% | 34% | 33% |

---

### [09] Forecast Adjustment

**File:** `agents/09_forecast_adjustment.py`  
**Purpose:** Bridge LLM qualitative reasoning to concrete numeric forecast modifications.

**Bias ramp:** Applied linearly from 30% of max at week 1 to 100% at week 8.
```
ramp_factor(week) = 0.30 + 0.70 × (week / horizon)
adjusted_point(week) = baseline_point(week) × (1 + bias_pct × ramp_factor / 100)
```

**CI scaling:** `adj_ci = baseline_ci_width × interval_multiplier` (symmetric around adjusted median)

**Direction call thresholds:** `total_pct > 2.0%` → "up" | `< -2.0%` → "down" | else → "flat"

**Scenario generation:**
- Probability distribution from regime table above, then ±10% shifted toward consensus direction
- Bull scenario: `adjusted_point × (1 + upside_risk_pct / 100)`
- Base scenario: `adjusted_point` (bias applied)
- Bear scenario: `adjusted_point × (1 - downside_risk_pct / 100)`

---

### [RL] Trading Robot

**Files:** `robot/environment.py`, `robot/model.py`, `robot/train.py`, `robot/agent.py`  
**Purpose:** Independent pattern-based directional signal to complement LLM reasoning.

**Architecture:**
```
state_text → Qwen2.5-0.5B tokenizer → transformer → last hidden states
           → attention-mask weighted mean pool → 896-dim embedding
           → Linear(896→256) → LayerNorm → GELU
           → Dropout(0.15) → Linear(256→64) → GELU
           → Linear(64→3) → softmax → [P(down), P(hold), P(up)]
```

Only the 247K-param policy head trains. Qwen is frozen (no backprop through the encoder).

**State text encoding (for Qwen at inference):**
- Commodity, region, horizon; last observed price
- Baseline model metrics (trend direction, directional accuracy, CI width)
- Signal quality counts (kept / removed / flagged by judge)
- Evidence structure summary (fact/causal/speculation counts)
- Debate consensus (direction, agreement score, top bull/bear argument)
- Temporal dynamics (causal chain count, peak impact week)
- Adjusted forecast (direction call, magnitude %, confidence, regime)

**REINFORCE training:**
- Data: 3Y weekly prices from yfinance, sliding windows with step=2
- Reward: +1.0 correct direction · −1.0 wrong · +0.3 hold-on-flat · −0.2 hold-on-trend
- Flat threshold: ±1.5% future return
- Variance reduction: EMA baseline subtraction (α=0.05)
- Optimizer: Adam, lr=3e-4, gradient norm clip 1.0
- Save best checkpoint by directional accuracy

**Combined signal weighting (in `robot/agent.py`):**
```python
W_BASELINE = 0.25   # Fixed — price level anchor
W_DEBATE   = 0.45   # Scaled by agreement_score: × (0.5 + 0.5 × agreement)
W_ROBOT    = 0.30   # Scaled by robot confidence: × (0.5 + 0.5 × confidence)

# Signed scores: up=+1, hold=0, down=−1
combined_score = W_BASELINE×base_score + W_DEBATE×debate_score + W_ROBOT×robot_score

# Thresholds
if combined_score > +0.20: combined_signal = "up"
elif combined_score < −0.20: combined_signal = "down"
else: combined_signal = "hold"
```

**Checkpoint:** Saved to `robot/checkpoints/trading_policy.pt`. Auto-retrains if ticker or horizon changes from last checkpoint.

**Device:** Apple Silicon (MPS) if available, else CPU. Qwen encoder lazy-loaded on first call (~1 GB download).

---

### [10] Report Generator

**File:** `agents/10_report_generator.py`  
**Purpose:** Produce human-readable and machine-readable output with full evidence traceability.

**Executive summary template (6-sentence structure):**
1. One-sentence verdict: direction + magnitude + horizon + confidence level
2. Baseline vs. adjusted forecast and why they differ
3. Strongest bull argument with temporal lag (from Agent 06)
4. Strongest bear argument + Skeptic's most important flag
5. Most relevant historical analogue and its implication
6. Most actionable caveat (what would flip the call)

**Markdown report sections:**
1. Executive Summary
2. Baseline Statistical Forecast (RMSE, directional accuracy per window)
3. Forecast Chart (embedded PNG)
4. 8-Week Adjusted Forecast Table (baseline vs. adjusted, CI bands)
5. Scenario Analysis (bull/base/bear with probabilities and narratives)
6. Evidence Table (top 20 by weight: classification, direction, source, content)
7. Temporal Impact Timeline (trigger, onset week, duration, propagation path)
8. Multi-Agent Debate (consensus, agreement score, per-agent positions + arguments)
9. RL Trading Robot Signal (direction, confidence, action probabilities, combined vote)
10. Uncertainty & Confidence Analysis
11. Conflict Discussion (key disagreements + skeptic flags)
12. Key Risks (top 5 + epistemic risks)

**Two-panel chart:**
- **Top panel:** 26-week historical close + LightGBM quantile bands (95% shaded, 80% shaded) + scenario tails (dashed bull/bear) + median forecast (blue) + temporal signal onset annotations (up to 4 events labeled with onset week)
- **Bottom panel:** Top 12 evidence items by weight; bars colored green (bullish), red (bearish), gray (neutral). Source labels on y-axis.
- Dark theme: background `#0f1117`, forecast `#3498db`, historical `#e0e0e0`

**Output artifacts:**
- `forecast_YYYYMMDD_HHMMSS.md` — full markdown report
- `forecast_YYYYMMDD_HHMMSS.json` — Pydantic model dump (machine-readable)
- `forecast_chart_YYYYMMDD_HHMMSS.png` — two-panel visualization

---

## LLM Call Inventory

| Step | Agent | Role | Cache? |
|------|-------|------|--------|
| 01 | Task Planner | Intent extraction → ForecastTask JSON | — |
| 03a | Signal Retrieval | Keyword expansion (12 phrases) | 24h file cache |
| 03b | Signal Judge | Quality scoring (batches of 20) | — |
| 04 | Event Detection | Named event extraction from headlines | — |
| 05 | Evidence Structuring | FACT/CLAIM/SPECULATION/FORECAST classification | — |
| 06 | Temporal Reasoning | Propagation lag estimation per causal claim | — |
| 07 | Debate — Bullish | Bullish supply risk argument | system cache |
| 07 | Debate — Bearish | Bearish demand argument | system cache |
| 07 | Debate — Skeptic | Evidence challenge | system cache |
| 07 | Debate — Analog | Historical analogue identification | system cache |
| 07 | Debate — Synthesis | Moderator consensus + agreement score | — |
| 08 | Risk Calibration | Debate → numeric uncertainty parameters | — |
| 10 | Report Generator | Analyst-grade executive summary | — |

**Total: ~13–14 LLM calls per run** (4 debate agents parallel; all others serial).

---

## Hardcoded Constants Reference

| Agent | Parameter | Value | Purpose |
|-------|-----------|-------|---------|
| 02 | Window days | 730, 365, 182 | 2Y/1Y/6M historical regimes |
| 02 | Max lag rule | n_weeks // 4 | Prevent NaN dropouts |
| 02 | LightGBM n_estimators | 300 | Ensemble depth |
| 02 | LightGBM learning_rate | 0.03 | Conservative for quantile reg |
| 02 | LightGBM num_leaves | 15 | Low complexity, prevents overfit |
| 02 | Trend threshold | ±1.5% | up/down/flat classification |
| 02 | Dir. acc weights | 0.2 / 0.6 / 1.0 | Below/near/above coin flip |
| 03a | Signal direction threshold | ±1.0% | BULLISH / BEARISH cutoff |
| 03a | News recency cutoff | 6 months | Stale article filter |
| 03a | LLM keyword count | 12 | Query expansion breadth |
| 03b | Batch size | 20 signals | Token budget management |
| 03b | Questionable discount | 50% | Confidence half-reduction |
| 04 | Z-score threshold | 2.0 | Anomaly detection sensitivity |
| 04 | Rolling window | 20 weeks | For mean/std computation |
| 04 | Context signals | 8 | Top textual signals for LLM |
| 06 | Max causal items | 15 | Temporal reasoning budget |
| 06 | Lag range | 0–10 weeks | Plausible supply-chain delays |
| 07 | Thread workers | 4 | Parallel debate calls |
| 08 | Regime thresholds | 0.75 / 0.55 / 0.40 | LOW/MODERATE/HIGH/EXTREME |
| 08 | Tail risk range | 5–25% | Scenario magnitude |
| 09 | Bias ramp start | 30% | Conservative near-term adjustment |
| 09 | Bias ramp end | 100% | Full directional view at W8 |
| 09 | Direction threshold | ±2.0% | up/flat/down call |
| 09 | Scenario shift | ±10% | Toward consensus direction |
| 10 | Chart lookback | 26 weeks | Historical price display |
| 10 | Evidence table rows | 20 | Top by weight |
| 10 | Temporal annotations | 4 | Chart readability |
| RL | Flat threshold | ±1.5% | Hold vs. directional reward |
| RL | EMA alpha | 0.05 | Variance reduction smoothness |
| RL | Training epochs | 4 | Default training duration |
| RL | Learning rate | 3e-4 | Adam for 247K-param head |
| RL | Grad clip norm | 1.0 | Training stability |
| RL | W_BASELINE | 25% | Combined signal weight |
| RL | W_DEBATE | 45% | Combined signal weight |
| RL | W_ROBOT | 30% | Combined signal weight |
| RL | Combined threshold | ±0.20 | UP/HOLD/DOWN decision |

---

## Backtest Results

Historical validation of Agent 02 (LightGBM baseline, no signals) across four distinct market regimes. Run with `pytest tests/test_backtest.py -v -s` (yfinance only, no API keys).

| Case | Cutoff | Last Price | Baseline Dir. | Actual Dir. | DirHit | Weekly DirAcc | CI-80 | MAE |
|------|--------|-----------|--------------|------------|--------|--------------|-------|-----|
| Brent 2023-Q4 decline | 2023-10-06 | $84.58 | up | down | ✗ | 50% | 25% | $5.20 |
| Brent 2024-Q1 rally | 2024-01-05 | $78.76 | down | up | ✗ | 50% | 50% | $3.79 |
| Brent 2024-Q2 peak | 2024-04-12 | $90.45 | flat | down | ✗ | 25% | 38% | $6.94 |
| WTI 2024-Q3 slump | 2024-07-05 | $83.16 | flat | down | ✗ | 50% | 25% | $6.06 |
| **Mean** | | | | | **0%** | **44%** | **34%** | **$5.50** |

**Interpretation:** The baseline systematically forecasts near-flat/continuation and misses all major turning points. This is expected — supply shocks, OPEC decisions, and geopolitical events carry no price-history signature. The baseline's role is to anchor price *scale* and CI *width*, not to call direction. The full pipeline (agents 03a–09) addresses this through multi-source signal reasoning and adversarial debate.

---

## Running the System

```bash
conda activate oil-gas-trade

# Standard forecast
python main.py "Forecast Brent crude oil price over the next 8 weeks"
python main.py "Forecast diesel inventory for Gulf Coast" --horizon 4

# Unit tests (no API keys, ~5s)
python -m pytest tests/test_pipeline.py -m "not live" -v

# Historical baseline backtest (yfinance only, ~90s)
python -m pytest tests/test_backtest.py -v -s

# Full pipeline integration tests (requires all API keys, ~5–10 min)
python -m pytest tests/test_pipeline.py tests/test_backtest.py -m live -v -s
```

---

## API Keys & Data Sources

| Source | Purpose | Environment Variable |
|--------|---------|---------------------|
| Anthropic Claude | All LLM calls (all reasoning steps) | `ANTHROPIC_API_KEY` |
| yfinance | Brent, WTI, HO, RB, NG futures | Free — no key |
| EIA API v2 | US petroleum inventory stocks (weekly) | `EIA_API_KEY` |
| Alpha Vantage | Macro indicators: CPI, real GDP, unemployment | `ALPHA_VANTAGE_KEY` |
| NewsAPI | Energy headlines with keyword search, recency filter | `NEWSAPI_KEY` |
| EIA / OPEC (scrape) | Weekly petroleum text, OPEC press room | Free — no key |

---

## Architectural Philosophy

### Why separate the baseline from the signal pipeline?

The central design principle is **separation of price scale from price direction**. Statistical models (LightGBM, ARIMA) are reliable at estimating *how volatile* a market is and *what range of prices* is plausible given historical price dynamics. They are structurally blind to supply shocks, geopolitical events, and demand regime changes because these do not manifest in price history until they are already priced in.

Running the baseline *before* any LLM reasoning prevents a known cognitive bias called **anchoring**: if the LLM sees an extreme price target from the statistical model, it will anchor toward that number even when fundamentals suggest otherwise. By separating the two and combining them only at the end (Agent 09), each component can do what it does best.

### Why four debate agents instead of one LLM call?

A single LLM call asked to "assess the oil market direction" exhibits **narrative dominance** — it tends to summarize the most recent, loudest, or most dramatic news and produce a confident directional call even when evidence is ambiguous. Four agents with opposing mandates (bull, bear, skeptic, historical analog) are structurally forced to surface counterevidence and acknowledge uncertainty. The key design insight:

- **The Skeptic agent cannot take a directional position.** Its job is only to challenge the reasoning of the other three. This is unusual in multi-agent design — typically all agents produce a final recommendation. Here, one agent exists purely to create epistemic friction.
- **The Historical Analog agent has no market data access** — only the evidence bundle and known historical episodes. This prevents it from reverse-engineering a position from current prices and forces genuine pattern matching.

### Why Constitutional AI for signal filtering?

Energy market news is systematically contaminated:
- Traders and governments have financial and political incentives to publish false or exaggerated supply disruption claims.
- Geopolitical propaganda exploits energy price sensitivity.
- Single-source claims about OPEC compliance spread virally before they can be verified.
- Circular references (source A quotes source B which quotes source A) create the illusion of corroboration.

Without filtering, these signals would bias the evidence bundle and therefore the debate. The Constitutional AI approach evaluates each signal against explicit, auditable criteria (not a single holistic "is this credible?" prompt), producing a harm flag record alongside each verdict. This makes the filtering transparent and debuggable.

### Why explicit temporal reasoning instead of just passing causal claims to the debate?

In commodities, timing is everything. An OPEC cut announced today may not materially affect US inventory levels for 6–10 weeks because: (1) tankers already at sea deliver their cargo; (2) refineries run down existing feedstock before adjusting throughput; (3) pipeline contracts buffer immediate changes. A debate agent told "OPEC cut production" without lag context will incorrectly weigh it as an *immediate* supply effect.

By explicitly estimating propagation paths and onset weeks (Agent 06), the system can annotate the forecast chart with which weeks each causal chain is expected to materialize, and the debate agents can reason about whether the effect falls within or beyond the 8-week horizon.

### Why REINFORCE for the trading robot instead of a supervised model?

The RL formulation captures a key property of commodity trading: the reward depends on *future* prices, which are unknown at decision time. A supervised model trained on past price movements would have access to the labels (future prices) during training — producing apparent accuracy from look-ahead bias. REINFORCE samples actions and observes delayed rewards, correctly modeling the decision-under-uncertainty structure.

The frozen Qwen2.5-0.5B encoder is used because: (1) oil market dynamics are described naturally in language (supply reports, news, policy announcements) and a language model can encode these representations more richly than hand-crafted features; (2) freezing the encoder keeps the training computationally tractable (only 247K parameters are updated); (3) the Qwen family has strong multilingual commodity domain knowledge from pretraining.

---

## Key Architectural Decisions

| Decision | Alternative Considered | Chosen Approach | Rationale |
|----------|----------------------|-----------------|-----------|
| LightGBM over ARIMA | ARIMA, Prophet, LSTM | LightGBM quantile ensemble | Handles nonlinear features, direct quantile output, no normality assumption, fast training |
| Three historical windows | Single 2Y window | 2Y + 1Y + 6M weighted | Captures regime changes; short window catches structural breaks; weighting by walk-forward accuracy auto-adapts |
| Recursive multi-step forecast | Direct multi-step | Recursive (append q50) | Maintains feature consistency; quantile bands propagate uncertainty appropriately |
| Constitutional AI criteria | Single "credibility score" | Five explicit criteria with harm flags | Auditable per-criterion failures; each criterion catches different manipulation types |
| Four debate agents | Two (bull/bear) | Bull + Bear + Skeptic + Historical Analog | Skeptic prevents false consensus; Historical Analog provides out-of-distribution anchoring |
| Parallel debate (ThreadPoolExecutor) | Sequential calls | 4 parallel + 1 synthesis | ~4× faster; agents cannot "read" each other's positions and thus argue more independently |
| Bias ramp (30%→100%) | Constant bias | Linear ramp from 30% at W1 to 100% at W8 | Near-term prices are always better-anchored by current data than by qualitative reasoning |
| Combined signal (25/45/30%) | Equal weights | Debate-dominant, baseline anchor, robot supplemental | Debate is the directional driver; baseline prevents wild extrapolation; robot adds historical pattern check |
| Lazy Qwen load | Pre-load at startup | Load on first robot call | Robot failure is graceful; system works even without GPU/Qwen dependencies |
| Prompt caching (cache_control) | No caching | `cache_control=ephemeral` on all system prompts | Reduces token cost significantly on repeated runs with same prompt structure |

---

## Core Assumptions

### Market & Data Assumptions

1. **Weekly frequency is sufficient.** The system targets 4–8 week supply-chain forecasting horizons where weekly price data captures the relevant dynamics. Intraday or daily noise is intentionally filtered.

2. **Price data from yfinance is representative.** Brent (BZ=F) and WTI (CL=F) futures prices from yfinance are treated as ground truth for training and baseline modeling. Settlement prices may differ slightly from spot.

3. **EIA weekly inventory data is timely.** The EIA releases weekly petroleum status reports on Wednesdays. The system treats the most recently available EIA data as current.

4. **Historical price patterns have some predictive signal.** The baseline model assumes that recent price dynamics (lags, rolling means, momentum) carry *some* signal about near-term price levels. The backtest confirms this is weak for directional calls (44% weekly accuracy) but adequate for scale anchoring.

5. **Supply-chain propagation lags are stable across regimes.** The reference propagation paths (e.g., OPEC cut → inventory draw: 6–10 weeks) are calibrated to historical commodity market experience and are assumed stable absent extraordinary structural changes (e.g., new pipeline capacity, major VLCC fleet changes).

### LLM Assumptions

6. **Claude Sonnet 4.6 has adequate oil market domain knowledge.** The debate agents, temporal reasoning, and evidence structuring rely on the LLM's implicit knowledge of commodity market dynamics, geopolitical relationships, and historical supply-chain behavior. This knowledge is bounded by the training cutoff date.

7. **LLM hallucination is constrained by evidence grounding.** All LLM calls that produce factual claims (event detection, evidence structuring, temporal reasoning) are given explicit evidence bundles to ground their outputs. The system does not ask LLMs to recall facts from training data beyond background domain knowledge.

8. **Constitutional AI criteria are stable across commodity types.** The five signal quality criteria (Accuracy, Recency, Credibility, Proportionality, Independence) are designed to be commodity-agnostic and should generalize from crude oil to diesel, natural gas, and LNG.

9. **Four debate agents produce approximately independent views.** The debate agents are run in parallel and cannot observe each other's outputs. They share the same evidence bundle but have different mandate prompts. Independence is structural (parallel execution) rather than guaranteed (same LLM backbone).

### Signal Retrieval Assumptions

10. **NewsAPI provides a representative sample of market-relevant headlines.** NewsAPI's free/basic tier may miss paywalled content (FT, Bloomberg, WSJ). Signal confidence for textual signals is intentionally set low (0.55) to reflect this.

11. **LLM-expanded keywords cover the relevant search space.** Twelve LLM-generated search phrases are assumed to cover commodity-specific supply/demand drivers adequately. This may underperform for niche sub-commodities (e.g., specific PADD regions, specific crude grades).

12. **Six-month news recency cutoff captures actionable signals.** Articles older than 6 months are assumed to be either already priced in or not actionable for an 8-week forecast. Structural changes (e.g., permanent pipeline capacity changes) that occurred more than 6 months ago may be missed.

---

## Known Limitations and Failure Modes

### When the system is expected to underperform

| Scenario | Expected Failure Mode | Mitigation |
|----------|----------------------|------------|
| **Fast-moving supply shock** (e.g., hurricane, immediate sanctions) | Signals arrive after cutoff; baseline dominates | Z-score anomaly detection triggers on price spikes; reduce confidence intervals |
| **LLM training cutoff** | LLM has no knowledge of recent events | Grounded evidence bundle (Agent 04–06) limits reliance on LLM recall |
| **OPEC non-compliance** | Model treats announced cuts as implemented | Skeptic agent specifically flags "OPEC compliance historically averages 70%" |
| **Coordinated market-moving narrative** | Signal judge may pass coordinated signals if they appear independently sourced | Independence criterion (criterion 5) specifically targets this; circular reference harm flag |
| **Low-liquidity commodities** (e.g., specific PADD distillates) | yfinance / EIA data coverage may be thin | Signal confidence reduced; evidence count will be low; widen uncertainty regime |
| **Extended market contango/backwardation** | LightGBM features do not include term structure | No mitigation currently; known gap |
| **Model changes in LLM** | Debate consensus format may change with new Claude versions | Pydantic output validation will surface parse failures early |
| **NewsAPI rate limit** | NewsAPI free tier: 100 requests/day | Caching (3h TTL) reduces repeat calls; graceful degradation (empty textual bundle) |
| **Qwen download failure** | Robot is skipped (try/except in main.py) | Combined signal falls back to baseline + debate only |

### Structural limitations

- **No options / implied volatility data:** The system does not incorporate options market signals (VIX-oil, put/call ratios, implied vol surfaces) which carry forward-looking market sentiment.
- **No positioning data:** CFTC COT reports (commitment of traders) are a leading indicator of sentiment but are not currently retrieved.
- **No currency effects beyond macro signals:** DXY is mentioned in temporal reasoning reference paths but is not directly fetched as a market signal.
- **Single-commodity forecasting:** The pipeline forecasts one commodity at a time. Cross-commodity spread dynamics (e.g., Brent–WTI spread, crack spreads) are included as market signals but are not modeled as a system.
- **No inventory-price feedback loop:** The model does not explicitly model the feedback between price levels and inventory restocking/drawdown behavior.

---

## Pydantic Data Model Reference

All inter-agent data is typed via Pydantic v2 (`core/models.py`). Key classes:

```
ForecastTask               ← Agent 01 output
BaselineForecast           ← Agent 02 output
  WeeklyForecastPoint        (8 instances for 8-week horizon)
  ModelStats                 (per window: RMSE, directional accuracy, weight)
Signal                     ← Agent 03a signal unit
SignalBundle               ← Agent 03a output (lists: market, macro, operational, textual)
JudgmentResult             ← Agent 03b output
  filtered_bundle: SignalBundle
  removed_count, flagged_count
DetectedEvent              ← Agent 04 event unit
EventList                  ← Agent 04 output
EvidenceItem               ← Agent 05 evidence unit (classification, weight, direction)
StructuredEvidenceBundle   ← Agent 05 output
TemporalImpact             ← Agent 06 impact unit (lag, persistence, intensity)
TemporalImpactMap          ← Agent 06 output
DebatePosition             ← Agent 07 per-agent output (stance, arguments, confidence)
DebateSummary              ← Agent 07 output (consensus, agreement_score, synthesis)
CalibrationResult          ← Agent 08 output (confidence, multiplier, bias%)
AdjustedForecast           ← Agent 09 output (adjusted_weekly, 3 scenarios)
  ScenarioForecast           (bull / base / bear with probability)
RobotDecision              ← RL robot output (direction, confidence, combined_signal)
FinalReport                ← Agent 10 output (all sections, paths to artefacts)
  EvidenceTableRow           (flattened for markdown/JSON export)
```

All enums: `SignalType`, `SignalDirection`, `EventSeverity`, `EvidenceClass`, `DebateStance`, `UncertaintyRegime`
