"""
Agent 2 — Baseline Forecast Engine  (LightGBM Multi-Window Quantile Ensemble)

Trains one set of LightGBM quantile models per historical lookback window,
then ensembles their forecasts weighted by walk-forward directional accuracy.

Windows:
  2Y  (~104 weekly obs) — captures long macro cycles
  1Y  (~52 weekly obs)  — captures medium-term regime
  6M  (~26 weekly obs)  — captures recent regime / structural breaks

Adaptive feature engineering:
  Each window uses the longest lag features that leave enough training rows.
  Short windows drop deep lags (lag_26, mom_12) to avoid NaN-dominated datasets.

Quantiles: q05 / q10 / q50 / q90 / q95
Ensemble weighting: directional accuracy from walk-forward validation.
  Windows with insufficient data or dir-acc < 0.5 receive reduced weight.
"""
from __future__ import annotations
import warnings
from datetime import datetime, timedelta
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

from core.models import (
    BaselineForecast, ForecastTask, ModelStats, WeeklyForecastPoint,
)

warnings.filterwarnings("ignore")

# ── Tickers ────────────────────────────────────────────────────────────────

TICKER_MAP = {
    "brent_crude": "BZ=F",
    "brent":       "BZ=F",
    "WTI":         "CL=F",
    "wti":         "CL=F",
    "crude":       "CL=F",
    "diesel":      "HO=F",
    "heating_oil": "HO=F",
    "LNG":         "LNG",
    "gasoline":    "RB=F",
    "natural_gas": "NG=F",
}

UNIT_MAP = {
    "BZ=F": "USD/bbl",
    "CL=F": "USD/bbl",
    "HO=F": "USD/gal",
    "LNG":  "USD",
    "RB=F": "USD/gal",
    "NG=F": "USD/MMBtu",
}

# ── Historical windows ─────────────────────────────────────────────────────

HISTORY_WINDOWS: list[tuple[str, int]] = [
    ("2Y", 730),   # ~104 weekly obs — long macro cycles
    ("1Y", 365),   # ~52 weekly obs  — medium-term regime
    ("6M", 182),   # ~26 weekly obs  — recent regime / structural breaks
]

# ── LightGBM base config ───────────────────────────────────────────────────

QUANTILES = {
    "q05": 0.05,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q95": 0.95,
}

LGB_PARAMS_BASE = {
    "objective":      "quantile",
    "n_estimators":   300,
    "learning_rate":  0.03,
    "num_leaves":     15,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":      0.1,
    "reg_lambda":     0.1,
    "random_state":   42,
    "verbose":        -1,
}

# ── Data fetching ──────────────────────────────────────────────────────────

def _fetch_weekly(ticker: str, days: int) -> pd.Series:
    end   = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df["Close"].squeeze().dropna().resample("W-FRI").last().dropna()

# ── Adaptive feature engineering ───────────────────────────────────────────

def _window_config(n_weeks: int) -> tuple[list[int], list[str]]:
    """
    Return (lags, feature_cols) adapted to the available number of weeks.
    Rule: max usable lag = n_weeks // 4 (ensures enough post-dropout rows).
    """
    max_lag = min(26, max(1, n_weeks // 4))
    lags    = [l for l in [1, 2, 4, 8, 12, 26] if l <= max_lag]

    cols: list[str] = [f"lag_{l}" for l in lags]
    cols += ["roll_mean_4", "roll_std_4", "mom_4"]
    if n_weeks >= 24:          # enough history for 12-week rolling stats
        cols += ["roll_mean_12", "roll_std_12"]
    if max_lag >= 12:
        cols += ["mom_12"]
    cols += ["week", "month"]
    return lags, cols


def _build_features(series: pd.Series, lags: list[int], has_roll12: bool) -> pd.DataFrame:
    df = pd.DataFrame({"price": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["price"].shift(lag)
    df["roll_mean_4"] = df["price"].rolling(4).mean()
    df["roll_std_4"]  = df["price"].rolling(4).std()
    df["mom_4"]       = (df["price"] - df["price"].shift(4)) / df["price"].shift(4)
    if has_roll12:
        df["roll_mean_12"] = df["price"].rolling(12).mean()
        df["roll_std_12"]  = df["price"].rolling(12).std()
    if max(lags, default=0) >= 12:
        df["mom_12"] = (df["price"] - df["price"].shift(12)) / df["price"].shift(12)
    df["week"]  = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    return df.dropna()

# ── Training ───────────────────────────────────────────────────────────────

def _train(
    series: pd.Series,
    lags: list[int],
    feature_cols: list[str],
    has_roll12: bool,
    n_train: int,
) -> dict[str, lgb.LGBMRegressor]:
    feat_df = _build_features(series, lags, has_roll12)
    if feat_df.empty:
        return {}
    X = feat_df[feature_cols].values
    y = feat_df["price"].values
    min_cs = max(3, n_train // 8)        # scale leaf size to window

    models: dict[str, lgb.LGBMRegressor] = {}
    for name, alpha in QUANTILES.items():
        m = lgb.LGBMRegressor(**LGB_PARAMS_BASE, alpha=alpha, min_child_samples=min_cs)
        m.fit(X, y)
        models[name] = m
    return models

# ── Walk-forward validation ────────────────────────────────────────────────

def _walk_forward_eval(
    series: pd.Series,
    lags: list[int],
    feature_cols: list[str],
    has_roll12: bool,
) -> dict[str, float]:
    n     = len(series)
    split = int(n * 0.80)
    feat_check = _build_features(series.iloc[:split], lags, has_roll12)
    if split < 15 or feat_check.shape[0] < 10:
        return {"rmse": float("nan"), "directional_accuracy": float("nan")}

    train_series = series.iloc[:split]
    test_series  = series.iloc[split:]
    models       = _train(train_series, lags, feature_cols, has_roll12, len(feat_check))
    if not models:
        return {"rmse": float("nan"), "directional_accuracy": float("nan")}
    med = models["q50"]

    preds, actuals = [], []
    rolling = train_series.copy()

    for actual_val in test_series:
        fd = _build_features(rolling, lags, has_roll12)
        if fd.empty:
            break
        pred = float(med.predict(fd[feature_cols].iloc[[-1]].values)[0])
        preds.append(pred)
        actuals.append(float(actual_val))
        new_idx = rolling.index[-1] + timedelta(weeks=1)
        rolling = pd.concat([rolling, pd.Series([actual_val], index=[new_idx])])

    if not preds:
        return {"rmse": float("nan"), "directional_accuracy": float("nan")}

    pa, aa = np.array(preds), np.array(actuals)
    rmse   = float(np.sqrt(np.mean((pa - aa) ** 2)))
    ad, pd_ = np.sign(np.diff(aa)), np.sign(np.diff(pa))
    dir_acc = float(np.mean(ad == pd_)) if len(ad) > 0 else float("nan")

    return {"rmse": round(rmse, 4), "directional_accuracy": round(dir_acc, 4)}

# ── Recursive multi-step forecast ─────────────────────────────────────────

def _recursive_forecast(
    models: dict[str, lgb.LGBMRegressor],
    series: pd.Series,
    horizon: int,
    last_date: pd.Timestamp,
    lags: list[int],
    feature_cols: list[str],
    has_roll12: bool,
) -> list[dict[str, float]]:
    """Return list of raw quantile dicts (not yet WeeklyForecastPoint)."""
    rolling = series.copy()
    raw: list[dict[str, float]] = []

    for _ in range(horizon):
        fd = _build_features(rolling, lags, has_roll12)
        if fd.empty:
            break
        X = fd[feature_cols].iloc[[-1]].values
        q = {name: float(m.predict(X)[0]) for name, m in models.items()}

        # Enforce monotonicity
        q05 = min(q["q05"], q["q10"], q["q50"])
        q10 = min(q["q10"], q["q50"])
        q50 = q["q50"]
        q90 = max(q["q90"], q50)
        q95 = max(q["q95"], q90)

        raw.append({"q05": q05, "q10": q10, "q50": q50, "q90": q90, "q95": q95})
        new_idx = rolling.index[-1] + timedelta(weeks=1)
        rolling = pd.concat([rolling, pd.Series([q50], index=[new_idx])])

    return raw

# ── Window weight ──────────────────────────────────────────────────────────

def _window_weight(dir_acc: float) -> float:
    if np.isnan(dir_acc):
        return 0.25          # can't evaluate — give small contribution
    if dir_acc >= 0.60:
        return 1.0
    if dir_acc >= 0.50:
        return 0.6
    return 0.2               # below coin flip — heavily discounted

# ── Main entry point ───────────────────────────────────────────────────────

def run(task: ForecastTask) -> BaselineForecast:
    print(f"\n[2/10] Baseline Forecast Engine (LightGBM Ensemble) — {task.commodity}...")

    ticker      = TICKER_MAP.get(task.commodity.lower(), "BZ=F")
    unit        = UNIT_MAP.get(ticker, "USD/bbl")
    series_name = f"{task.commodity} ({ticker})"

    # Fetch the longest window once; slice for shorter windows
    max_days = max(d for _, d in HISTORY_WINDOWS)
    full_series = _fetch_weekly(ticker, max_days)
    last_date   = full_series.index[-1]
    last_value  = float(full_series.iloc[-1])

    print(f"       Last observed: {last_date.date()} = {last_value:.3f} {unit}  "
          f"(full series: {len(full_series)} weekly obs)")

    all_raw_fc: list[list[dict[str, float]]] = []
    all_weights: list[float] = []
    model_stats_list: list[ModelStats] = []

    for window_name, days in HISTORY_WINDOWS:
        cutoff = last_date - timedelta(days=days)
        series  = full_series[full_series.index >= cutoff]
        n_weeks = len(series)

        lags, feature_cols = _window_config(n_weeks)
        has_roll12 = n_weeks >= 24

        feat_check = _build_features(series, lags, has_roll12)
        n_train = len(feat_check)

        print(f"       [{window_name}] {n_weeks}w obs | "
              f"lags={lags} | roll12={has_roll12} | ~{n_train} train rows", end="")

        if n_train < 8:
            print(" → SKIPPED (too few rows)")
            continue

        eval_stats = _walk_forward_eval(series, lags, feature_cols, has_roll12)
        dir_acc    = eval_stats["directional_accuracy"]
        rmse       = eval_stats["rmse"]
        weight     = _window_weight(dir_acc)

        da_str = f"{dir_acc:.1%}" if not np.isnan(dir_acc) else "n/a"
        rm_str = f"{rmse:.3f}"   if not np.isnan(rmse)    else "n/a"
        print(f" | RMSE={rm_str}  DirAcc={da_str}  w={weight:.2f}")

        models  = _train(series, lags, feature_cols, has_roll12, n_train)
        raw_fc  = _recursive_forecast(
            models, series, task.horizon_weeks, last_date,
            lags, feature_cols, has_roll12,
        )

        if len(raw_fc) == task.horizon_weeks:
            all_raw_fc.append(raw_fc)
            all_weights.append(weight)

        rmse_val = float(rmse) if not np.isnan(rmse) else None
        dacc_val = float(dir_acc) if not np.isnan(dir_acc) else None
        model_stats_list.append(ModelStats(
            model_name=f"LightGBM-{window_name}",
            aic=None,
            rmse_insample=rmse_val,
            params={
                "window":                window_name,
                "n_obs":                 n_weeks,
                "n_train_rows":          n_train,
                "lags":                  lags,
                "quantiles":             list(QUANTILES.values()),
                "directional_accuracy_wf": dacc_val,
                "ensemble_weight":       weight,
            },
        ))

    if not all_raw_fc:
        raise RuntimeError("All windows failed — cannot produce forecast.")

    # ── Weighted quantile ensemble ─────────────────────────────────────────
    total_w = sum(all_weights)
    norm_w  = [w / total_w for w in all_weights]

    weekly_points: list[WeeklyForecastPoint] = []
    for step in range(task.horizon_weeks):
        q05 = sum(norm_w[i] * all_raw_fc[i][step]["q05"] for i in range(len(all_raw_fc)))
        q10 = sum(norm_w[i] * all_raw_fc[i][step]["q10"] for i in range(len(all_raw_fc)))
        q50 = sum(norm_w[i] * all_raw_fc[i][step]["q50"] for i in range(len(all_raw_fc)))
        q90 = sum(norm_w[i] * all_raw_fc[i][step]["q90"] for i in range(len(all_raw_fc)))
        q95 = sum(norm_w[i] * all_raw_fc[i][step]["q95"] for i in range(len(all_raw_fc)))

        fd = last_date + timedelta(weeks=step + 1)
        weekly_points.append(WeeklyForecastPoint(
            week_offset=step + 1,
            forecast_date=fd.strftime("%Y-%m-%d"),
            point=round(q50, 4),
            ci_80_low=round(q10, 4),
            ci_80_high=round(q90, 4),
            ci_95_low=round(q05, 4),
            ci_95_high=round(q95, 4),
        ))

    end_val    = weekly_points[-1].point
    pct_change = (end_val - last_value) / last_value * 100
    direction  = "up" if pct_change > 1.5 else "down" if pct_change < -1.5 else "flat"

    windows_used = "+".join(ms.params["window"] for ms in model_stats_list)
    print(f"       Ensemble ({windows_used}): {direction} "
          f"({pct_change:+.2f}% over {task.horizon_weeks}w)  "
          f"W{task.horizon_weeks} median={end_val:.3f} {unit}")

    # Weighted-average directional accuracy as the ensemble headline metric
    valid_accs = [
        (ms.params["directional_accuracy_wf"], ms.params["ensemble_weight"])
        for ms in model_stats_list
        if ms.params["directional_accuracy_wf"] is not None
    ]
    if valid_accs:
        ens_dacc = sum(a * w for a, w in valid_accs) / sum(w for _, w in valid_accs)
    else:
        ens_dacc = None

    # Prepend ensemble summary as first entry for report generator
    ensemble_stat = ModelStats(
        model_name=f"LightGBM-Ensemble ({windows_used})",
        aic=None,
        rmse_insample=None,
        params={
            "windows_used":            windows_used,
            "directional_accuracy_wf": ens_dacc,
            "n_windows":               len(model_stats_list),
        },
    )

    return BaselineForecast(
        task=task,
        series_name=series_name,
        last_observed_date=last_date.strftime("%Y-%m-%d"),
        last_observed_value=round(last_value, 4),
        unit=unit,
        best_model=f"LightGBM-Ensemble ({windows_used})",
        model_stats=[ensemble_stat] + model_stats_list,
        weekly_forecasts=weekly_points,
        trend_direction=direction,
        trend_magnitude_pct=round(pct_change, 2),
    )
