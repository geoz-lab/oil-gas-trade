"""
Agent 7 — Multi-Agent Debate System
The centerpiece of this architecture.

Four independent agents with adversarial roles argue from the same evidence:
  A. Bullish Supply Risk Agent — shortages, rising costs, disruptions
  B. Bearish Demand Agent     — weak demand, oversupply, inventory build
  C. Skeptic Agent            — challenges weak evidence, stale data
  D. Historical Analog Agent  — surfaces prior macro analogues

A fifth synthesis call produces the DebateSummary.

Justification: single-agent systems collapse uncertainty, overfit recent news,
and hallucinate causal certainty. Adversarial debate forces the system to surface
counterevidence that a single-agent would suppress.
"""
from __future__ import annotations
import concurrent.futures

from core.llm import chat_json
from core.models import (
    BaselineForecast, DebatePosition, DebateStance, DebateSummary,
    StructuredEvidenceBundle, TemporalImpactMap
)

# ── Shared context builder ─────────────────────────────────────────────────

def _build_evidence_brief(evidence: StructuredEvidenceBundle, temporal: TemporalImpactMap) -> str:
    lines = []
    lines.append("=== ESTABLISHED FACTS ===")
    for e in evidence.facts[:8]:
        lines.append(f"  [{e.direction.value.upper()}] {e.content} (w={e.weight:.2f})")

    lines.append("\n=== CAUSAL CLAIMS ===")
    for e in evidence.causal_claims[:6]:
        lines.append(f"  [{e.direction.value.upper()}] {e.content} (w={e.weight:.2f})")

    lines.append("\n=== TEMPORAL IMPACTS ===")
    for t in temporal.impacts[:6]:
        lines.append(f"  {t.trigger_event}: lag={t.impact_lag_weeks}w persist={t.persistence_weeks}w | {t.propagation_path[:100]}")

    lines.append("\n=== SPECULATIONS (lower weight) ===")
    for e in evidence.speculations[:4]:
        lines.append(f"  [{e.direction.value.upper()}] {e.content} (w={e.weight:.2f})")

    return "\n".join(lines)


def _build_baseline_brief(baseline: BaselineForecast) -> str:
    pts = baseline.weekly_forecasts
    last = pts[-1] if pts else None
    return (
        f"Baseline ({baseline.best_model}): "
        f"last observed={baseline.last_observed_value:.3f} {baseline.unit} "
        f"trend={baseline.trend_direction} ({baseline.trend_magnitude_pct:+.1f}% over {baseline.task.horizon_weeks}w) "
        f"week-8 point={f'{last.point:.3f}' if last else 'N/A'}"
    )


DEBATE_SCHEMA = """{
  "stance": "strongly_bullish|bullish|neutral|bearish|strongly_bearish",
  "confidence": 0.0-1.0,
  "key_arguments": ["...", "...", "..."],
  "evidence_refs": ["..."],
  "counterarguments_acknowledged": ["..."],
  "price_direction_call": "up|down|flat",
  "price_magnitude_estimate_pct": <float>
}"""

# ── Agent A: Bullish Supply Risk ───────────────────────────────────────────

BULLISH_SYSTEM = """You are the Bullish Supply Risk Agent in an oil & gas forecast debate.

Your mandate: make the strongest possible case for supply TIGHTENING and upward price/inventory pressure.
Focus on: production disruptions, OPEC cuts, sanctions, refinery outages, infrastructure failures,
shipping disruptions, weather risks, geopolitical risk premium.

You MUST engage with the evidence provided — do not fabricate events.
Acknowledge the strongest counterarguments (1-2) to show intellectual honesty.
"""

# ── Agent B: Bearish Demand ────────────────────────────────────────────────

BEARISH_SYSTEM = """You are the Bearish Demand Agent in an oil & gas forecast debate.

Your mandate: make the strongest possible case for demand WEAKNESS and downward price/inventory pressure.
Focus on: economic slowdown signals (PMI, CPI, unemployment), EV adoption, demand destruction,
oversupply from non-OPEC producers, inventory builds, refinery overcapacity.

You MUST engage with the evidence provided — do not fabricate trends.
Acknowledge the strongest counterarguments (1-2) to show intellectual honesty.
"""

# ── Agent C: Skeptic ───────────────────────────────────────────────────────

SKEPTIC_SYSTEM = """You are the Skeptic Agent in an oil & gas forecast debate.

Your mandate: challenge ALL other agents' reasoning. Question:
- Is the evidence recent and reliable?
- Are causal claims actually supported by the data?
- Are historical analogues truly comparable?
- What data is missing that would change the conclusion?
- Is the market already pricing in these risks?

Do not take a bullish or bearish stance. Your job is to identify what we do NOT know
and where the analysis is weakest. This protects against overconfidence.
"""

# ── Agent D: Historical Analog ─────────────────────────────────────────────

ANALOG_SYSTEM = """You are the Historical Analog Agent in an oil & gas forecast debate.

Your mandate: identify the most relevant historical macro episodes that are analogous
to current conditions, and estimate what happened to Gulf Coast diesel inventories / prices.

Key analogues to consider (use your training knowledge):
- 2008 oil price spike and crash
- 2014-2016 oil price collapse (OPEC market share war)
- 2020 COVID demand collapse and OPEC+ cut
- 2021-2022 post-COVID recovery and Russia-Ukraine supply shock
- 2011 Libya disruption
- 2005 Gulf Coast hurricanes (Katrina/Rita)

Select the 1-2 most relevant analogues. Describe what happened and what it implies for
the 8-week outlook. State clearly if current conditions don't match any analogue well.
"""


def _run_debate_agent(
    agent_name: str,
    agent_role: str,
    system_prompt: str,
    evidence_brief: str,
    baseline_brief: str,
) -> DebatePosition:
    user_msg = f"""BASELINE STATISTICAL FORECAST:
{baseline_brief}

EVIDENCE BRIEF:
{evidence_brief}

Your role: {agent_role}

Based on this evidence, make your strongest case.
Return JSON matching this schema:
{DEBATE_SCHEMA}
"""
    result = chat_json(system_prompt, user_msg, max_tokens=2048, cache_system=True)

    stance_raw = result.get("stance", "neutral")
    try:
        stance = DebateStance(stance_raw)
    except ValueError:
        stance = DebateStance.NEUTRAL

    return DebatePosition(
        agent_name=agent_name,
        agent_role=agent_role,
        stance=stance,
        confidence=float(result.get("confidence", 0.5)),
        key_arguments=result.get("key_arguments", []),
        evidence_refs=result.get("evidence_refs", []),
        counterarguments_acknowledged=result.get("counterarguments_acknowledged", []),
        price_direction_call=result.get("price_direction_call", "flat"),
        price_magnitude_estimate_pct=float(result.get("price_magnitude_estimate_pct", 0.0)),
    )


SYNTHESIS_SYSTEM = """You are the Debate Moderator synthesizing four agents' positions.

Your task: produce an objective synthesis that:
1. Identifies where agents agree (consensus direction)
2. Quantifies the degree of agreement (0.0 = total disagreement, 1.0 = complete agreement)
3. Highlights the key unresolved disagreements
4. Flags the skeptic's most important challenges
5. Identifies the most historically relevant analogue
6. Produces a 3-5 sentence synthesis narrative for downstream use

Do not take sides — represent the evidence faithfully.
"""


def _synthesize(positions: list[DebatePosition], evidence_brief: str) -> DebateSummary:
    position_texts = []
    for p in positions:
        position_texts.append(
            f"{p.agent_name} ({p.stance.value}, conf={p.confidence:.0%}): "
            f"call={p.price_direction_call} ({p.price_magnitude_estimate_pct:+.1f}%) | "
            f"args={p.key_arguments[:2]}"
        )

    user_msg = f"""The four debate agents have spoken:

{chr(10).join(position_texts)}

Evidence context:
{evidence_brief[:600]}

Return JSON:
{{
  "consensus_direction": "up|down|flat|contested",
  "agreement_score": 0.0-1.0,
  "key_disagreements": ["...", "..."],
  "strongest_bull_argument": "...",
  "strongest_bear_argument": "...",
  "skeptic_flags": ["...", "..."],
  "historical_analog": "...",
  "synthesis_narrative": "..."
}}
"""
    result = chat_json(SYNTHESIS_SYSTEM, user_msg, max_tokens=1500)

    return DebateSummary(
        positions=positions,
        consensus_direction=result.get("consensus_direction", "contested"),
        agreement_score=float(result.get("agreement_score", 0.5)),
        key_disagreements=result.get("key_disagreements", []),
        strongest_bull_argument=result.get("strongest_bull_argument", ""),
        strongest_bear_argument=result.get("strongest_bear_argument", ""),
        skeptic_flags=result.get("skeptic_flags", []),
        historical_analog=result.get("historical_analog", ""),
        synthesis_narrative=result.get("synthesis_narrative", ""),
    )


def run(
    baseline: BaselineForecast,
    evidence: StructuredEvidenceBundle,
    temporal: TemporalImpactMap,
) -> DebateSummary:
    print("\n[7/10] Multi-Agent Debate System — launching 4 debate agents in parallel...")

    evidence_brief = _build_evidence_brief(evidence, temporal)
    baseline_brief = _build_baseline_brief(baseline)

    agents = [
        ("Bullish Supply Risk Agent", "Make the case for supply tightening / price upside.", BULLISH_SYSTEM),
        ("Bearish Demand Agent", "Make the case for demand weakness / price downside.", BEARISH_SYSTEM),
        ("Skeptic Agent", "Challenge the evidence quality and causal claims.", SKEPTIC_SYSTEM),
        ("Historical Analog Agent", "Surface the most relevant historical market analogues.", ANALOG_SYSTEM),
    ]

    # Run the four agents in parallel for efficiency
    positions: list[DebatePosition] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _run_debate_agent,
                name, role, system, evidence_brief, baseline_brief
            ): name
            for name, role, system in agents
        }
        for future in concurrent.futures.as_completed(futures):
            agent_name = futures[future]
            try:
                pos = future.result()
                positions.append(pos)
                print(f"       [{agent_name}] stance={pos.stance.value} conf={pos.confidence:.0%} call={pos.price_direction_call}")
            except Exception as e:
                print(f"       Warning: {agent_name} failed — {e}")

    # Sort by agent name for reproducibility
    positions.sort(key=lambda p: p.agent_name)

    print("       Synthesizing debate positions...")
    summary = _synthesize(positions, evidence_brief)
    print(f"       Consensus: {summary.consensus_direction} | Agreement: {summary.agreement_score:.0%}")

    return summary
