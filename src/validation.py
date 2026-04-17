from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: Dict[str, float]


def build_forward_labels(price_df: pd.DataFrame,
                         horizon: int = 20,
                         up_target: float = 0.08,
                         down_target: float = -0.08,
                         price_col: str = "close") -> pd.DataFrame:
    df = price_df.copy().sort_values(["ticker", "date"])
    out = []
    for t, g in df.groupby("ticker", sort=False):
        g = g.copy()
        px = pd.to_numeric(g[price_col], errors="coerce")
        fut = px.shift(-horizon) / px - 1.0
        g["fwd_return"] = fut
        g["label_long_success"] = (g["fwd_return"] >= up_target).astype(int)
        g["label_down_break"] = (g["fwd_return"] <= down_target).astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _precision_at_top(y_true: np.ndarray, y_score: np.ndarray, top_n: int) -> float:
    if len(y_true) == 0:
        return np.nan
    idx = np.argsort(-y_score)[:max(1, min(top_n, len(y_true)))]
    return float(np.mean(y_true[idx]))


def _expectancy(y_true: np.ndarray, fwd_returns: np.ndarray, y_score: np.ndarray, top_n: int) -> float:
    if len(fwd_returns) == 0:
        return np.nan
    idx = np.argsort(-y_score)[:max(1, min(top_n, len(fwd_returns)))]
    return float(np.nanmean(fwd_returns[idx]))


def run_walk_forward_validation(scan_with_labels: pd.DataFrame,
                                score_col: str = "long_rank_score",
                                label_col: str = "label_long_success",
                                return_col: str = "fwd_return",
                                date_col: str = "date",
                                train_days: int = 252 * 2,
                                test_days: int = 63,
                                top_n: int = 20) -> ValidationResult:
    df = scan_with_labels.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    if df.empty:
        return ValidationResult(pd.DataFrame(), pd.DataFrame(), {})

    dates = pd.Series(sorted(df[date_col].dropna().unique()))
    fold_rows: List[Dict] = []
    pred_rows: List[pd.DataFrame] = []

    start_idx = train_days
    fold = 0
    while start_idx + test_days <= len(dates):
        fold += 1
        train_start = dates.iloc[start_idx - train_days]
        train_end = dates.iloc[start_idx - 1]
        test_end = dates.iloc[start_idx + test_days - 1]

        train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        test_mask = (df[date_col] > train_end) & (df[date_col] <= test_end)
        test_df = df.loc[test_mask].copy()
        if test_df.empty:
            start_idx += test_days
            continue

        y_true = pd.to_numeric(test_df[label_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_score = pd.to_numeric(test_df[score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        fwd_ret = pd.to_numeric(test_df[return_col], errors="coerce").to_numpy(dtype=float)

        auc = _safe_auc(y_true, y_score)
        p_top = _precision_at_top(y_true, y_score, top_n=top_n)
        exp_top = _expectancy(y_true, fwd_ret, y_score, top_n=top_n)

        fold_rows.append({
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_end": test_end,
            "test_rows": int(len(test_df)),
            "auc": auc,
            f"precision_at_{top_n}": p_top,
            f"expectancy_at_{top_n}": exp_top,
        })

        pred_rows.append(test_df[[date_col, "ticker", score_col, label_col, return_col]].assign(fold=fold))
        start_idx += test_days

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = {}
    if not fold_metrics.empty:
        metric_cols = [c for c in fold_metrics.columns if c in {"auc", f"precision_at_{top_n}", f"expectancy_at_{top_n}"}]
        for c in metric_cols:
            summary[c] = float(pd.to_numeric(fold_metrics[c], errors="coerce").mean())
    return ValidationResult(fold_metrics=fold_metrics, predictions=predictions, summary=summary)


def build_price_side_validation_panel(price_df: pd.DataFrame,
                                      min_history: int = 80,
                                      horizon: int = 20) -> pd.DataFrame:
    """Build a lightweight historical panel for walk-forward validation.

    This is intentionally price-side only and should be treated as a scaffold,
    not a full broker/intraday truth validation. It lets the app audit whether
    the ranking logic is directionally useful before richer data is available.
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    df = price_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    rows = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.copy().sort_values("date").reset_index(drop=True)
        if len(g) < min_history + horizon + 5:
            continue
        close = pd.to_numeric(g["close"], errors="coerce")
        high = pd.to_numeric(g["high"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        volume = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        true_range = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = true_range.rolling(14).mean()
        high60 = high.rolling(60).max()
        low20 = low.rolling(20).min()
        ret20 = close.pct_change(20)
        realized = close.pct_change().rolling(20).std() * np.sqrt(252)
        vol_burst = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
        for i in range(min_history, len(g) - horizon):
            last_close = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
            if not np.isfinite(last_close) or last_close <= 0:
                continue
            e20 = float(ema20.iloc[i]) if pd.notna(ema20.iloc[i]) else np.nan
            e50 = float(ema50.iloc[i]) if pd.notna(ema50.iloc[i]) else np.nan
            e200 = float(ema200.iloc[i]) if pd.notna(ema200.iloc[i]) else np.nan
            breakout_distance = (last_close / max(float(high60.iloc[i]) if pd.notna(high60.iloc[i]) else last_close, 1e-9)) - 1.0
            range20 = float((high.iloc[max(0, i-19): i+1].max() - low.iloc[max(0, i-19): i+1].min()) / max(last_close, 1e-9))
            base_maturity = min((i + 1) / 60.0, 1.0)
            breakout_integrity = 100 * np.clip(
                0.40 * (1.0 if breakout_distance >= -0.01 else max(0.0, 1 + breakout_distance * 10))
                + 0.30 * (1 - min(range20 / 0.25, 1))
                + 0.30 * base_maturity,
                0.0,
                1.0,
            )
            if last_close > e20 > e50 > e200:
                trend_quality = 100.0
                phase = "MARKUP"
            elif last_close > e20 > e50:
                trend_quality = 66.7
                phase = "ACCUMULATION"
            elif last_close > e50:
                trend_quality = 33.3
                phase = "PULLBACK_HEALTHY"
            else:
                trend_quality = 0.0 if (np.isfinite(e20) and np.isfinite(e50) and np.isfinite(e200) and last_close < e20 < e50 < e200) else 25.0
                phase = "MARKDOWN" if trend_quality == 0.0 else "NEUTRAL"
            upper_wick = float((high.iloc[i] - max(close.iloc[i], g.loc[i, 'open'])) / max(high.iloc[i] - low.iloc[i], 1e-9)) if pd.notna(high.iloc[i]) and pd.notna(low.iloc[i]) else 0.0
            false_breakout_risk = 100 * np.clip(0.55 * upper_wick + 0.45 * (0 if breakout_distance >= 0 else min(abs(breakout_distance) * 12, 1)), 0.0, 1.0)
            dry_score = 100 * np.clip(0.6 * (1 - min((float(realized.iloc[i]) if pd.notna(realized.iloc[i]) else 0.3) / 1.2, 1)) + 0.4 * min((float(vol_burst.iloc[i]) if pd.notna(vol_burst.iloc[i]) else 1.0) / 2.0, 1), 0.0, 1.0)
            wet_score = 100 - dry_score
            rs20 = float(ret20.iloc[i]) if pd.notna(ret20.iloc[i]) else 0.0
            conf = 70.0 if i >= min_history + 20 else 55.0
            long_rank = (
                0.22 * trend_quality + 0.22 * breakout_integrity + 0.16 * dry_score + 0.12 * conf
                + 0.16 * np.clip(50 + rs20 * 250.0, 0, 100) - 0.12 * false_breakout_risk
            ) / 0.76
            rows.append({
                "date": pd.to_datetime(g.loc[i, "date"]),
                "ticker": ticker,
                "close": last_close,
                "phase": phase,
                "trend_quality": trend_quality,
                "breakout_integrity": breakout_integrity,
                "false_breakout_risk": false_breakout_risk,
                "dry_score": dry_score,
                "wet_score": wet_score,
                "relative_strength_20d": rs20,
                "long_rank_score": float(np.clip(long_rank, 0.0, 100.0)),
                "support_20d": float(low20.iloc[i]) if pd.notna(low20.iloc[i]) else np.nan,
                "resistance_60d": float(high60.iloc[i]) if pd.notna(high60.iloc[i]) else np.nan,
            })
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel = build_forward_labels(panel, horizon=horizon, up_target=0.08, down_target=-0.08, price_col="close")
    return panel
