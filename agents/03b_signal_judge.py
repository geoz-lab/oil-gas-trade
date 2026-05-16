"""
Agent 3b — Signal Reliability Judge
Screens all retrieved signals for harmful, exaggerated, or misleading content
before they enter the evidence structuring and debate pipeline.

Inspired by Constitutional AI / RLHF principles:
  Each signal is evaluated against five information quality criteria.
  Signals rated "misleading" are removed from the bundle.
  Signals rated "questionable" are kept but confidence is halved.

Criteria:
  1. ACCURACY      — verifiable claim vs. rumour/speculation
  2. RECENCY       — actionable within 6 months
  3. CREDIBILITY   — independent primary source vs. self-interested actor
  4. PROPORTIONALITY — impact claim proportional to evidence
  5. INDEPENDENCE  — no coordinated market-moving agenda
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

from core.llm import chat_json
from core.models import JudgmentResult, Signal, SignalBundle, SignalJudgment

_SIX_MONTHS_AGO = datetime.now(timezone.utc) - timedelta(days=180)

JUDGE_SYSTEM = """You are a senior financial intelligence analyst and information integrity auditor.

Evaluate commodity market signals for quality and potential to mislead price forecasts.

Apply five Constitutional AI-inspired criteria for each signal:
1. ACCURACY       — Is the claim factually verifiable from primary sources, or rumour/speculation?
2. RECENCY        — Is information actionable? (>6 months old = outdated)
3. CREDIBILITY    — Independent source (Reuters, EIA, official body) vs. party with financial incentive?
4. PROPORTIONALITY — Is the claimed price impact proportional to the supporting evidence?
5. INDEPENDENCE   — Could this be coordinated narrative designed to move prices?

Harm flags (use exactly these strings):
  "exaggerated"          — price impact claim far exceeds what evidence supports
  "unverified"           — no verifiable primary source
  "outdated"             — information older than 6 months
  "single_source"        — no independent corroboration
  "market_bias"          — appears designed to influence price sentiment
  "geopolitical_propaganda" — politically motivated energy market narrative
  "circular_reference"   — sources cite each other without primary data

Judgment tiers:
  "reliable"      — high quality, use at full weight
  "questionable"  — concerns present, use at 50% confidence
  "misleading"    — discard; would systematically bias the forecast

Energy market news is frequently exaggerated or politically motivated. Be strict.
"""


def _build_prompt(signals: list[Signal], today: datetime) -> str:
    cutoff = today - timedelta(days=180)
    parts = []
    for i, sig in enumerate(signals, 1):
        age_flag = ""
        try:
            ts = datetime.fromisoformat(sig.timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                age_flag = f" [STALE: {sig.timestamp[:10]}]"
        except Exception:
            pass

        parts.append(
            f"#{i} id={sig.signal_id}\n"
            f"  Source: {sig.source} | Type: {sig.signal_type.value}\n"
            f"  Name: {sig.name[:80]}{age_flag}\n"
            f"  Content: {sig.raw_text[:280]}\n"
            f"  Confidence: {sig.confidence:.2f}"
        )

    return (
        f"Today: {today.strftime('%Y-%m-%d')}\n\n"
        + "\n\n".join(parts)
        + '\n\nReturn JSON:\n'
        + '{\n'
        + '  "judgments": [\n'
        + '    {"signal_id": "...", "judgment": "reliable|questionable|misleading",\n'
        + '     "reliability_score": 0.0-1.0, "harm_flags": [], "reason": "max 80 chars"}\n'
        + '  ],\n'
        + '  "judge_summary": "2-3 sentence overall quality assessment"\n'
        + '}'
    )


def run(bundle: SignalBundle) -> JudgmentResult:
    print("\n[3b] Signal Judge — screening for exaggeration / misleading content...")

    today = datetime.now(timezone.utc)
    all_signals = bundle.all_signals()

    if not all_signals:
        return JudgmentResult(
            judgments=[],
            filtered_bundle=bundle,
            removed_count=0,
            flagged_count=0,
            judge_summary="No signals to evaluate.",
        )

    # Judge in batches of 20
    BATCH = 20
    all_judgments: list[SignalJudgment] = []
    summary_parts: list[str] = []

    for start in range(0, len(all_signals), BATCH):
        batch = all_signals[start: start + BATCH]
        try:
            resp = chat_json(JUDGE_SYSTEM, _build_prompt(batch, today), max_tokens=1800)
            for j in resp.get("judgments", []):
                all_judgments.append(SignalJudgment(
                    signal_id=str(j.get("signal_id", "")),
                    judgment=str(j.get("judgment", "reliable")),
                    reliability_score=float(j.get("reliability_score", 0.75)),
                    harm_flags=[str(f) for f in j.get("harm_flags", [])],
                    reason=str(j.get("reason", ""))[:120],
                ))
            if s := resp.get("judge_summary"):
                summary_parts.append(s)
        except Exception as exc:
            print(f"       Judge batch failed — {exc}; treating all as reliable")
            for sig in batch:
                all_judgments.append(SignalJudgment(
                    signal_id=sig.signal_id,
                    judgment="reliable",
                    reliability_score=0.75,
                    harm_flags=[],
                    reason="judge unavailable",
                ))

    judgment_map = {j.signal_id: j for j in all_judgments}
    removed: list[str] = []
    flagged: list[str] = []

    def _apply(sig: Signal, bucket: list[Signal]) -> None:
        j = judgment_map.get(sig.signal_id)
        if j is None:
            bucket.append(sig)
            return
        sig.reliability_score = j.reliability_score
        sig.harm_flags = j.harm_flags
        sig.judgment = j.judgment
        if j.judgment == "misleading":
            removed.append(sig.signal_id)
            print(f"       REMOVED  [{sig.source[:20]}] {sig.name[:55]} — {j.reason}")
        else:
            if j.judgment == "questionable":
                sig.confidence = round(sig.confidence * 0.5, 3)
                flagged.append(sig.signal_id)
                print(f"       FLAGGED  [{sig.source[:20]}] {sig.name[:55]} — {j.reason}")
            bucket.append(sig)

    km, kmac, kop, kt = [], [], [], []
    for sig in bundle.market_signals:      _apply(sig, km)
    for sig in bundle.macro_signals:       _apply(sig, kmac)
    for sig in bundle.operational_signals: _apply(sig, kop)
    for sig in bundle.textual_signals:     _apply(sig, kt)

    filtered = SignalBundle(
        retrieved_at=bundle.retrieved_at,
        market_signals=km,
        macro_signals=kmac,
        operational_signals=kop,
        textual_signals=kt,
        expanded_keywords=bundle.expanded_keywords,
    )

    total_in  = len(all_signals)
    total_out = len(filtered.all_signals())
    print(f"       {total_in} signals in → {total_out} kept "
          f"| Removed: {len(removed)} | Flagged: {len(flagged)}")

    return JudgmentResult(
        judgments=all_judgments,
        filtered_bundle=filtered,
        removed_count=len(removed),
        flagged_count=len(flagged),
        judge_summary=" ".join(summary_parts),
    )
