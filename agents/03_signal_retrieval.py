"""
Agent 3 — Signal Retrieval Layer
Collects evidence from heterogeneous sources to reduce source bias and
narrative collapse. Intentionally pulls from conflicting signal types.

Sources:
  Market       — yfinance (Brent, WTI, spreads, freight proxies)
  Macro        — EIA API v2 (US crude/distillate/gasoline stocks)
                 Alpha Vantage (CPI, real GDP, unemployment, commodity indices)
  Textual      — EIA "This Week in Petroleum" web scrape
                 NewsAPI with LLM-expanded keywords + 6-month recency filter
                 OPEC press release page scrape

Design: each signal carries source, type, direction, and confidence so the
Evidence Structuring agent (layer 5) can weight them appropriately.
"""
from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import yfinance as yf
from bs4 import BeautifulSoup

from core.config import ALPHA_VANTAGE_KEY, CACHE_DIR, EIA_API_KEY, NEWSAPI_KEY
from core.llm import chat_json
from core.models import ForecastTask, Signal, SignalBundle, SignalDirection, SignalType

_NOW = datetime.now(timezone.utc).isoformat()
_SIX_MONTHS_AGO = datetime.now(timezone.utc) - timedelta(days=180)

# ── Cache helpers ──────────────────────────────────────────────────────────

def _cache_key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _load_cache(key: str, ttl_hours: float = 6.0) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists() and time.time() - p.stat().st_mtime < ttl_hours * 3600:
        return json.loads(p.read_text())
    return None


def _save_cache(key: str, data: dict) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data, default=str))


def _sid(prefix: str, name: str) -> str:
    return f"{prefix}_{name.replace(' ', '_').lower()[:30]}"


def _direction(pct: float, threshold: float = 1.0) -> SignalDirection:
    if pct > threshold:
        return SignalDirection.BULLISH
    if pct < -threshold:
        return SignalDirection.BEARISH
    return SignalDirection.NEUTRAL


def _pct_chg(series) -> float:
    s = series.dropna()
    return float((s.iloc[-1] - s.iloc[-5]) / s.iloc[-5] * 100) if len(s) >= 5 else 0.0


def _is_recent(timestamp_str: str) -> bool:
    """Return True if timestamp is within the last 6 months."""
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts >= _SIX_MONTHS_AGO
    except Exception:
        return True  # keep if unparseable


# ══════════════════════════════════════════════════════════════════════════
# KEYWORD EXPANSION  —  LLM generates related search terms
# ══════════════════════════════════════════════════════════════════════════

def _expand_keywords(task: ForecastTask) -> list[str]:
    """Use LLM to generate news search keywords for the commodity/region/variable."""
    cache_key = _cache_key(f"kw_{task.commodity}_{task.region}_{task.forecast_variable}")
    cached = _load_cache(cache_key, ttl_hours=24.0)
    if cached and "keywords" in cached:
        return cached["keywords"]

    try:
        result = chat_json(
            "You are a commodity research analyst specializing in energy markets.",
            f"""Generate 12 specific news search phrases for:
Commodity: {task.commodity}
Region: {task.region}
Forecast variable: {task.forecast_variable}

Provide diverse angles: supply/demand drivers, OPEC/producers decisions,
geopolitical risks, shipping/logistics, refinery operations, macro indicators,
seasonal demand, inventory builds/draws, sanctions, weather disruptions.

Each phrase: 2-6 words, specific enough to retrieve relevant news.

Return JSON: {{"keywords": ["phrase1", "phrase2", ...]}}""",
            max_tokens=300,
        )
        keywords = result.get("keywords", [])[:12]
    except Exception as exc:
        print(f"       Keyword expansion failed — {exc}")
        keywords = []

    _save_cache(cache_key, {"keywords": keywords})
    return keywords


# ══════════════════════════════════════════════════════════════════════════
# MARKET SIGNALS  —  yfinance
# ══════════════════════════════════════════════════════════════════════════

MARKET_TICKERS = {
    "Brent Crude (BZ=F)":        ("BZ=F",  "USD/bbl"),
    "WTI Crude (CL=F)":          ("CL=F",  "USD/bbl"),
    "Heating Oil/Diesel (HO=F)": ("HO=F",  "USD/gal"),
    "RBOB Gasoline (RB=F)":      ("RB=F",  "USD/gal"),
    "Natural Gas (NG=F)":        ("NG=F",  "USD/MMBtu"),
}


def _fetch_market_signals() -> list[Signal]:
    key = _cache_key("market_signals_v2")
    cached = _load_cache(key, ttl_hours=1.0)
    if cached:
        return [Signal(**s) for s in cached]

    signals: list[Signal] = []
    end   = datetime.today()
    start = end - timedelta(days=45)

    for name, (ticker, unit) in MARKET_TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False,
            )
            if df.empty:
                continue
            close = df["Close"].squeeze().dropna()
            price = float(close.iloc[-1])
            pct   = _pct_chg(close)
            signals.append(Signal(
                signal_id=_sid("mkt", name),
                signal_type=SignalType.MARKET,
                source="yfinance",
                name=name,
                value=round(price, 4),
                unit=unit,
                timestamp=_NOW,
                direction=_direction(pct),
                raw_text=f"{name}: {price:.3f} {unit} | 5-day chg: {pct:+.1f}%",
                confidence=0.90,
            ))
        except Exception as exc:
            print(f"       Warning: {name} — {exc}")

    # Brent–WTI spread
    try:
        brent = float(yf.download("BZ=F", period="5d", auto_adjust=True, progress=False)
                      ["Close"].squeeze().dropna().iloc[-1])
        wti   = float(yf.download("CL=F", period="5d", auto_adjust=True, progress=False)
                      ["Close"].squeeze().dropna().iloc[-1])
        spread = brent - wti
        signals.append(Signal(
            signal_id="mkt_brent_wti_spread",
            signal_type=SignalType.MARKET,
            source="yfinance",
            name="Brent-WTI Spread",
            value=round(spread, 3),
            unit="USD/bbl",
            timestamp=_NOW,
            direction=SignalDirection.BULLISH if spread > 3 else SignalDirection.NEUTRAL,
            raw_text=f"Brent-WTI spread: {spread:.2f} USD/bbl (Brent={brent:.2f}, WTI={wti:.2f})",
            confidence=0.90,
        ))
    except Exception:
        pass

    _save_cache(key, [s.model_dump() for s in signals])
    return signals


# ══════════════════════════════════════════════════════════════════════════
# MACRO SIGNALS  —  EIA API v2  (petroleum) + Alpha Vantage (macro)
# ══════════════════════════════════════════════════════════════════════════

EIA_V2_BASE = "https://api.eia.gov/v2"

EIA_V2_SERIES: list[tuple] = [
    (
        "petroleum/pri/spt/data",
        {"series": ["RBRTE"]},
        "Brent Spot Price (EIA)", "USD/bbl", "market", 0.93,
    ),
    (
        "petroleum/pri/spt/data",
        {"series": ["RWTC"]},
        "WTI Spot Price (EIA)", "USD/bbl", "market", 0.93,
    ),
    (
        "petroleum/stoc/wstk/data",
        {"duoarea": ["NUS"], "product": ["EPC0"], "process": ["SAX"]},
        "US Crude Oil Stocks excl. SPR (MBBL)", "MBBL", "operational", 0.90,
    ),
    (
        "petroleum/stoc/wstk/data",
        {"duoarea": ["NUS"], "product": ["EPD0"], "process": ["SAX"]},
        "US Distillate Stocks (MBBL)", "MBBL", "operational", 0.90,
    ),
    (
        "petroleum/stoc/wstk/data",
        {"duoarea": ["NUS"], "product": ["EPM0"], "process": ["SAX"]},
        "US Motor Gasoline Stocks (MBBL)", "MBBL", "operational", 0.87,
    ),
    (
        "petroleum/stoc/wstk/data",
        {"duoarea": ["R30"], "product": ["EPC0"], "process": ["SAX"]},
        "PADD 3 (Gulf Coast) Crude Stocks (MBBL)", "MBBL", "operational", 0.90,
    ),
]


def _eia_v2_fetch(route: str, facets: dict[str, list[str]], length: int = 8) -> list[dict]:
    params: list[tuple[str, str]] = [
        ("api_key",              EIA_API_KEY),
        ("frequency",            "weekly"),
        ("data[0]",              "value"),
        ("sort[0][column]",      "period"),
        ("sort[0][direction]",   "desc"),
        ("length",               str(length)),
    ]
    for facet_name, values in facets.items():
        for v in values:
            params.append((f"facets[{facet_name}][]", v))

    url = f"{EIA_V2_BASE}/{route}/"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["response"]["data"]


def _fetch_eia_v2_signal(
    route: str, facets: dict, name: str, unit: str,
    sig_type: str, confidence: float,
) -> Optional[Signal]:
    if not EIA_API_KEY:
        return None

    cache_key = _cache_key(f"eia_v2_{route}_{list(facets.values())}")
    cached = _load_cache(cache_key, ttl_hours=12.0)
    if cached:
        return Signal(**cached)

    try:
        rows = _eia_v2_fetch(route, facets, length=8)
        if not rows:
            return None

        latest_val  = float(rows[0]["value"])
        prev_val    = float(rows[1]["value"]) if len(rows) > 1 else latest_val
        prev4_val   = float(rows[4]["value"]) if len(rows) > 4 else latest_val
        latest_date = rows[0]["period"]
        desc        = rows[0].get("series-description", name)

        pct_1w = (latest_val - prev_val)  / abs(prev_val)  * 100 if prev_val  else 0.0
        pct_4w = (latest_val - prev4_val) / abs(prev4_val) * 100 if prev4_val else 0.0

        sig = Signal(
            signal_id=_sid("eia", name),
            signal_type=SignalType(sig_type),
            source="EIA API v2",
            name=name,
            value=round(latest_val, 3),
            unit=unit,
            timestamp=latest_date,
            direction=_direction(pct_4w),
            raw_text=(
                f"{desc}: {latest_val:,.1f} {unit} ({latest_date}) "
                f"| 1w: {pct_1w:+.1f}% | 4w: {pct_4w:+.1f}%"
            ),
            confidence=confidence,
        )
        _save_cache(cache_key, sig.model_dump())
        return sig
    except Exception as exc:
        print(f"       EIA v2 [{name}] failed — {exc}")
        return None


def _fetch_eia_signals() -> list[Signal]:
    signals: list[Signal] = []
    for route, facets, name, unit, sig_type, conf in EIA_V2_SERIES:
        sig = _fetch_eia_v2_signal(route, facets, name, unit, sig_type, conf)
        if sig:
            signals.append(sig)
    return signals


# ── Alpha Vantage macro signals ────────────────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"

# (function, extra_params, name, unit, sig_type, confidence)
AV_SERIES = [
    ("CPI",        {},                          "US CPI (Monthly)",        "index", "macro", 0.82),
    ("REAL_GDP",   {"interval": "quarterly"},   "US Real GDP (Quarterly)", "USD B", "macro", 0.80),
    ("UNEMPLOYMENT",{},                         "US Unemployment Rate",    "%",     "macro", 0.78),
]


def _fetch_av_signal(
    function: str, extra_params: dict, name: str, unit: str,
    sig_type: str, confidence: float,
) -> Optional[Signal]:
    if not ALPHA_VANTAGE_KEY:
        return None

    cache_key = _cache_key(f"av_{function}")
    cached = _load_cache(cache_key, ttl_hours=24.0)
    if cached:
        return Signal(**cached)

    try:
        params = {"function": function, "apikey": ALPHA_VANTAGE_KEY, **extra_params}
        r = requests.get(AV_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Alpha Vantage returns data under various keys
        for key in ("data", "annualReports", "quarterlyReports"):
            if key in data and data[key]:
                rows = data[key]
                break
        else:
            return None

        latest = rows[0]
        prev   = rows[1] if len(rows) > 1 else latest

        # Value field varies by endpoint
        val_key = next(
            (k for k in ("value", "realGDP", "value") if k in latest), None
        )
        if val_key is None:
            return None

        value = float(latest[val_key])
        prev_v = float(prev.get(val_key, value))
        pct = (value - prev_v) / abs(prev_v) * 100 if prev_v else 0.0
        date_str = latest.get("date", latest.get("fiscalDateEnding", _NOW[:10]))

        sig = Signal(
            signal_id=_sid("av", name),
            signal_type=SignalType(sig_type),
            source="Alpha Vantage",
            name=name,
            value=round(value, 4),
            unit=unit,
            timestamp=date_str,
            direction=_direction(pct),
            raw_text=f"{name}: {value} {unit} ({date_str}) | chg: {pct:+.1f}%",
            confidence=confidence,
        )
        _save_cache(cache_key, sig.model_dump())
        return sig
    except Exception as exc:
        print(f"       Alpha Vantage [{name}] failed — {exc}")
        return None


def _fetch_av_signals() -> list[Signal]:
    signals: list[Signal] = []
    for function, extra, name, unit, sig_type, conf in AV_SERIES:
        sig = _fetch_av_signal(function, extra, name, unit, sig_type, conf)
        if sig:
            signals.append(sig)
    return signals


def _fetch_macro_signals() -> list[Signal]:
    signals: list[Signal] = []
    signals.extend(_fetch_eia_signals())
    signals.extend(_fetch_av_signals())
    return signals


# ══════════════════════════════════════════════════════════════════════════
# OPERATIONAL SIGNALS  —  EIA This Week in Petroleum + OPEC press room
# ══════════════════════════════════════════════════════════════════════════

EIA_TWIP_URL  = "https://www.eia.gov/petroleum/supply/weekly/"
OPEC_NEWS_URL = "https://www.opec.org/opec_web/en/press_room/4316.htm"


def _scrape_eia_weekly_text() -> Optional[Signal]:
    cache_key = _cache_key("eia_twip_text_v2")
    cached = _load_cache(cache_key, ttl_hours=6.0)
    if cached:
        return Signal(**cached)

    try:
        r = requests.get(EIA_TWIP_URL, timeout=15, headers={"User-Agent": "research-bot/1.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        highlights: list[str] = []
        for tag in soup.find_all(["p", "li"])[:20]:
            text = tag.get_text(strip=True)
            if len(text) > 60 and any(
                kw in text.lower() for kw in
                ["crude", "petroleum", "inventory", "stock", "refinery", "barrel", "diesel", "gasoline"]
            ):
                highlights.append(text)
            if len(highlights) >= 5:
                break

        body = " | ".join(highlights)[:900] if highlights else "EIA weekly text unavailable."
        sig = Signal(
            signal_id="eia_twip_weekly",
            signal_type=SignalType.TEXTUAL,
            source="EIA This Week in Petroleum",
            name="EIA Weekly Petroleum Report Highlights",
            timestamp=_NOW,
            direction=SignalDirection.UNKNOWN,
            raw_text=body,
            confidence=0.82,
        )
        _save_cache(cache_key, sig.model_dump())
        return sig
    except Exception as exc:
        print(f"       EIA TWIP scrape failed — {exc}")
        return None


def _scrape_opec_news() -> Optional[Signal]:
    cache_key = _cache_key("opec_news_v2")
    cached = _load_cache(cache_key, ttl_hours=12.0)
    if cached:
        return Signal(**cached)

    try:
        r = requests.get(OPEC_NEWS_URL, timeout=15, headers={"User-Agent": "research-bot/1.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = []
        for tag in soup.find_all(["h2", "h3", "a"])[:15]:
            t = tag.get_text(strip=True)
            if len(t) > 20:
                headlines.append(t)
        body = " | ".join(headlines[:6])[:600] if headlines else "OPEC news unavailable."

        direction = SignalDirection.UNKNOWN
        body_lower = body.lower()
        if any(w in body_lower for w in ["cut", "reduce", "output", "compliance"]):
            direction = SignalDirection.BULLISH
        elif any(w in body_lower for w in ["increase", "boost", "ramp"]):
            direction = SignalDirection.BEARISH

        sig = Signal(
            signal_id="opec_press_news",
            signal_type=SignalType.TEXTUAL,
            source="OPEC Press Room",
            name="OPEC Press Room Headlines",
            timestamp=_NOW,
            direction=direction,
            raw_text=body,
            confidence=0.75,
        )
        _save_cache(cache_key, sig.model_dump())
        return sig
    except Exception as exc:
        print(f"       OPEC scrape failed — {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# TEXTUAL SIGNALS  —  NewsAPI with expanded keywords + 6-month recency filter
# ══════════════════════════════════════════════════════════════════════════

BASE_NEWSAPI_QUERIES = [
    "Brent crude oil price",
    "OPEC production cut supply",
    "oil refinery outage sanctions",
    "crude oil inventory EIA weekly",
]

BULL_KEYWORDS = [
    "surge", "rise", "spike", "jump", "cut", "shortage", "disruption",
    "sanction", "outage", "hurricane", "tension", "conflict", "tighten",
]
BEAR_KEYWORDS = [
    "fall", "drop", "plunge", "crash", "surplus", "glut", "weak",
    "recession", "slowdown", "oversupply", "build", "demand destruction",
]


def _classify_headline_direction(text: str) -> SignalDirection:
    t = text.lower()
    bull = sum(1 for w in BULL_KEYWORDS if w in t)
    bear = sum(1 for w in BEAR_KEYWORDS if w in t)
    if bull > bear:
        return SignalDirection.BULLISH
    if bear > bull:
        return SignalDirection.BEARISH
    return SignalDirection.UNKNOWN


def _fetch_newsapi_signals(extra_queries: list[str] | None = None) -> list[Signal]:
    if not NEWSAPI_KEY:
        return []

    all_queries = list(BASE_NEWSAPI_QUERIES)
    if extra_queries:
        # merge, deduplicate, cap at 6 queries
        seen = set(q.lower() for q in all_queries)
        for q in extra_queries:
            if q.lower() not in seen:
                all_queries.append(q)
                seen.add(q.lower())
    all_queries = all_queries[:6]

    cache_key = _cache_key(f"newsapi_v3_{'_'.join(all_queries)}")
    cached = _load_cache(cache_key, ttl_hours=3.0)
    if cached:
        return [Signal(**s) for s in cached]

    signals: list[Signal] = []
    raw_articles: list[dict] = []
    seen_titles: set[str] = set()
    skipped_old = 0

    for query in all_queries:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        query,
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": 8,
                    "apiKey":   NEWSAPI_KEY,
                },
                timeout=12,
            )
            r.raise_for_status()
            for art in r.json().get("articles", []):
                title = (art.get("title") or "").strip()
                if not title or title in seen_titles or len(title) < 15:
                    continue
                seen_titles.add(title)

                # 6-month recency filter
                published_at = art.get("publishedAt", "")
                if published_at and not _is_recent(published_at):
                    skipped_old += 1
                    continue

                raw_articles.append(art)
                desc      = (art.get("description") or "")[:250]
                full_text = f"{title} | {desc}"
                source    = art.get("source", {}).get("name", "NewsAPI")
                signals.append(Signal(
                    signal_id=_sid("news", title),
                    signal_type=SignalType.TEXTUAL,
                    source=source,
                    name=title[:90],
                    timestamp=published_at or _NOW,
                    direction=_classify_headline_direction(full_text),
                    raw_text=full_text,
                    confidence=0.55,
                ))
        except Exception as exc:
            print(f"       NewsAPI query '{query}' failed — {exc}")

    if skipped_old:
        print(f"       Recency filter: dropped {skipped_old} articles older than 6 months")

    _save_cache(cache_key, [s.model_dump() for s in signals])
    return signals, raw_articles  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════
# Consolidated signal cache  —  YYYYMMDD_signal.json
# ══════════════════════════════════════════════════════════════════════════

def _write_signal_cache(
    task: ForecastTask,
    keywords: list[str],
    bundle: SignalBundle,
    raw_articles: list[dict],
) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = CACHE_DIR / f"{date_str}_signal.json"
    payload = {
        "date": date_str,
        "topic": f"{task.commodity} | {task.region} | {task.forecast_variable}",
        "expanded_keywords": keywords,
        "total_signals": len(bundle.all_signals()),
        "signals_by_type": {
            "market":      len(bundle.market_signals),
            "macro":       len(bundle.macro_signals),
            "operational": len(bundle.operational_signals),
            "textual":     len(bundle.textual_signals),
        },
        "raw_news_articles": raw_articles,
        "signals": [s.model_dump() for s in bundle.all_signals()],
    }
    path.write_text(json.dumps(payload, default=str, indent=2))
    print(f"       Signal cache: {path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════

def run(task: ForecastTask) -> SignalBundle:
    print(f"\n[3/10] Signal Retrieval — market / EIA / Alpha Vantage / NewsAPI / OPEC...")

    # Step 1 — LLM keyword expansion
    print("       Expanding search keywords via LLM...")
    keywords = _expand_keywords(task)
    if keywords:
        print(f"       Keywords ({len(keywords)}): {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")

    # Step 2 — Fetch all signal types
    market = _fetch_market_signals()
    print(f"       Market signals (yfinance): {len(market)}")

    macro = _fetch_macro_signals()
    eia_count = sum(1 for s in macro if "EIA" in s.source)
    av_count  = sum(1 for s in macro if "Alpha Vantage" in s.source)
    print(f"       Macro signals: EIA={eia_count}  AlphaVantage={av_count}")

    operational: list[Signal] = []
    eia_text = _scrape_eia_weekly_text()
    if eia_text:
        operational.append(eia_text)
    opec_sig = _scrape_opec_news()
    if opec_sig:
        operational.append(opec_sig)

    # Step 3 — News with expanded keywords + recency filter
    news_result = _fetch_newsapi_signals(extra_queries=keywords[:4])
    if isinstance(news_result, tuple):
        textual_news, raw_articles = news_result
    else:
        textual_news, raw_articles = news_result, []

    all_textual = operational + textual_news
    print(f"       Textual signals: NewsAPI={len(textual_news)}  EIA/OPEC={len(operational)}")

    bundle = SignalBundle(
        retrieved_at=_NOW,
        market_signals=market,
        macro_signals=macro,
        operational_signals=[],
        textual_signals=all_textual,
        expanded_keywords=keywords,
    )

    # Step 4 — Write consolidated YYYYMMDD_signal.json
    _write_signal_cache(task, keywords, bundle, raw_articles)

    return bundle
