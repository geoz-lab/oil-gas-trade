"""
RL Trading Environment.
Wraps historical price data into a gym-style interface for REINFORCE training.

State  : text description of technical market conditions at time t
Action : 0 = down, 1 = hold, 2 = up
Reward : +1.0 correct direction | -1.0 wrong | +0.3 hold on flat market | -0.2 hold on trending
"""
from __future__ import annotations
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


TICKER_MAP = {
    "brent": "BZ=F", "brent_crude": "BZ=F",
    "wti": "CL=F", "diesel": "HO=F", "heating_oil": "HO=F",
    "gasoline": "RB=F", "natural_gas": "NG=F",
}

FLAT_THRESHOLD_PCT = 1.5   # movements < this are treated as "flat"


class TradingEnvironment:
    """Simulates sequential trading decisions on historical weekly price data."""

    def __init__(self, ticker: str = "BZ=F", horizon_weeks: int = 8, history_years: int = 3):
        self.ticker         = ticker
        self.horizon_weeks  = horizon_weeks
        self.history_years  = history_years
        self.prices: pd.Series | None = None

    def load(self) -> int:
        end   = datetime.today()
        start = end - timedelta(days=self.history_years * 365)
        df = yf.download(
            self.ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
        )
        self.prices = (
            df["Close"].squeeze().dropna()
            .resample("W-FRI").last().dropna()
        )
        return len(self.prices)

    # ── State encoding ─────────────────────────────────────────────────────

    def state_text(self, idx: int) -> str:
        """Return a natural-language description of market state at index idx."""
        w = self.prices
        lb = 26  # lookback
        start = max(0, idx - lb)
        window = w.iloc[start:idx]
        if len(window) < 5:
            return "Insufficient historical data available."

        price   = float(window.iloc[-1])
        ret_4w  = _pct(window, 4)
        ret_12w = _pct(window, 12) if len(window) >= 12 else 0.0
        ret_26w = _pct(window, 26) if len(window) >= 26 else 0.0
        vol_4w  = float(window.pct_change().rolling(4).std().iloc[-1] * 100) if len(window) >= 5 else 0.0
        ma4     = float(window.rolling(4).mean().iloc[-1])
        ma12    = float(window.rolling(12).mean().iloc[-1]) if len(window) >= 12 else price
        rsi     = _rsi(window)
        trend   = "bullish" if ret_4w > 2 else "bearish" if ret_4w < -2 else "neutral"
        momentum = "accelerating" if ret_4w > ret_12w else "decelerating"

        date_str = str(w.index[idx - 1].date()) if idx <= len(w) else "unknown"

        return (
            f"[{date_str}] {self.ticker} price: {price:.2f} USD | "
            f"4w return: {ret_4w:+.1f}% | 12w return: {ret_12w:+.1f}% | 26w return: {ret_26w:+.1f}% | "
            f"trend: {trend} | momentum: {momentum} | "
            f"volatility(4w): {vol_4w:.1f}% | RSI(14): {rsi:.1f} | "
            f"vs 4w MA: {(price/ma4 - 1)*100:+.1f}% | vs 12w MA: {(price/ma12 - 1)*100:+.1f}%"
        )

    # ── Reward ─────────────────────────────────────────────────────────────

    def reward(self, idx: int, action: int) -> float:
        future_idx = min(idx + self.horizon_weeks, len(self.prices) - 1)
        p_now    = float(self.prices.iloc[idx - 1])
        p_future = float(self.prices.iloc[future_idx - 1])
        pct      = (p_future - p_now) / p_now * 100

        is_up   = pct >  FLAT_THRESHOLD_PCT
        is_down = pct < -FLAT_THRESHOLD_PCT
        is_flat = not (is_up or is_down)

        if is_up:
            return {0: -1.0, 1: -0.2, 2: 1.0}[action]
        if is_down:
            return {0: 1.0, 1: -0.2, 2: -1.0}[action]
        return {0: -0.3, 1: 0.3, 2: -0.3}[action]  # flat market

    def future_return_pct(self, idx: int) -> float:
        future_idx = min(idx + self.horizon_weeks, len(self.prices) - 1)
        p_now    = float(self.prices.iloc[idx - 1])
        p_future = float(self.prices.iloc[future_idx - 1])
        return (p_future - p_now) / p_now * 100

    # ── Episode indices ────────────────────────────────────────────────────

    def episode_indices(self, step: int = 2) -> list[int]:
        """Return sliding-window episode start indices (skip every `step` weeks)."""
        lb  = 26
        end = len(self.prices) - self.horizon_weeks - 1
        return list(range(lb, end, step))


# ── Helpers ────────────────────────────────────────────────────────────────

def _pct(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    return float((series.iloc[-1] - series.iloc[-n - 1]) / abs(series.iloc[-n - 1]) * 100)


def _rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta  = series.diff().dropna()
    gain   = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss   = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - 100 / (1 + rs))
