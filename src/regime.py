from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_regime(price_df: pd.DataFrame) -> pd.DataFrame:
    """Return 1-row market context derived from universe breadth and momentum."""
    if price_df is None or price_df.empty:
        return pd.DataFrame([{
            "market_regime": "UNKNOWN",
            "execution_mode": "SELECTIVE",
            "market_breadth_pct": np.nan,
            "market_median_ret20": np.nan,
            "market_median_ret5": np.nan,
            "market_bias_score": np.nan,
            "market_vol_state": np.nan,
        }])

    rows = []
    for _, g in price_df.sort_values(["ticker", "date"]).groupby("ticker"):
        if len(g) < 40:
            continue
        c = pd.to_numeric(g["close"], errors="coerce")
        if c.isna().all():
            continue
        ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        ret20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) >= 21 and pd.notna(c.iloc[-21]) else np.nan
        ret5 = c.iloc[-1] / c.iloc[-6] - 1 if len(c) >= 6 and pd.notna(c.iloc[-6]) else np.nan
        tr = pd.concat([
            (pd.to_numeric(g["high"], errors="coerce") - pd.to_numeric(g["low"], errors="coerce")).abs(),
            (pd.to_numeric(g["high"], errors="coerce") - c.shift(1)).abs(),
            (pd.to_numeric(g["low"], errors="coerce") - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        rows.append({
            "above20": float(c.iloc[-1] > ema20) if pd.notna(ema20) else np.nan,
            "above50": float(c.iloc[-1] > ema50) if pd.notna(ema50) else np.nan,
            "ret20": ret20,
            "ret5": ret5,
            "atr_pct": atr14 / max(float(c.iloc[-1]), 1e-9) if pd.notna(atr14) else np.nan,
        })

    if not rows:
        return pd.DataFrame([{
            "market_regime": "UNKNOWN",
            "execution_mode": "SELECTIVE",
            "market_breadth_pct": np.nan,
            "market_median_ret20": np.nan,
            "market_median_ret5": np.nan,
            "market_bias_score": np.nan,
            "market_vol_state": np.nan,
        }])

    breadth_df = pd.DataFrame(rows)
    breadth20 = 100 * breadth_df["above20"].mean()
    breadth50 = 100 * breadth_df["above50"].mean()
    med20 = float(breadth_df["ret20"].median()) if breadth_df["ret20"].notna().any() else np.nan
    med5 = float(breadth_df["ret5"].median()) if breadth_df["ret5"].notna().any() else np.nan
    vol_state = float(breadth_df["atr_pct"].median()) if breadth_df["atr_pct"].notna().any() else np.nan

    bias_score = 0.45 * breadth20 + 0.25 * breadth50 + 30 * np.clip((0 if np.isnan(med20) else med20) / 0.12, -1, 1)
    bias_score += 20 * np.clip((0 if np.isnan(med5) else med5) / 0.05, -1, 1)
    bias_score = float(np.clip(bias_score, 0, 100))

    regime = "CHOPPY"
    mode = "SELECTIVE"
    if breadth20 >= 62 and (np.isnan(med20) or med20 >= 0.03):
        regime = "RISK_ON"
        mode = "AGGRESSIVE"
    elif breadth20 <= 38 and (np.isnan(med20) or med20 <= -0.03):
        regime = "RISK_OFF"
        mode = "DEFENSIVE"
    elif breadth20 >= 52 and (np.isnan(med5) or med5 >= 0):
        regime = "UPTREND_SELECTIVE"
        mode = "SELECTIVE"
    elif breadth20 <= 45 and (np.isnan(med5) or med5 < 0):
        regime = "DOWNTREND_SELECTIVE"
        mode = "DEFENSIVE"

    return pd.DataFrame([{
        "market_regime": regime,
        "execution_mode": mode,
        "market_breadth_pct": round(float(breadth20), 1),
        "market_median_ret20": round(float(med20), 4) if not np.isnan(med20) else np.nan,
        "market_median_ret5": round(float(med5), 4) if not np.isnan(med5) else np.nan,
        "market_bias_score": round(float(bias_score), 1),
        "market_vol_state": round(float(vol_state), 4) if not np.isnan(vol_state) else np.nan,
    }])
