
from __future__ import annotations
import math
import pandas as pd
import numpy as np

def _safe_last(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")

def _ret(s: pd.Series, n: int) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < n + 1:
        return float("nan")
    b = float(s.iloc[-(n+1)])
    return float(s.iloc[-1] / b - 1.0) if b else float("nan")

def _trend_quality(close: pd.Series) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < 50:
        return 0.0
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    px = close.iloc[-1]
    score = 0
    score += 50 if px > ma20 else 0
    score += 50 if px > ma50 else 0
    return float(score)

def _phase(close: pd.Series) -> str:
    r20 = _ret(close, 20)
    r60 = _ret(close, 60)
    if math.isfinite(r20) and math.isfinite(r60):
        if r20 > 0.08 and r60 > 0.12:
            return "MARKUP"
        if r20 > 0 and r60 <= 0.05:
            return "EARLY_MARKUP"
        if r20 < -0.08 and r60 < -0.12:
            return "MARKDOWN"
        if abs(r20) < 0.03 and abs(r60) < 0.05:
            return "BASE"
    return "NEUTRAL"

def compute_eod_scanner(price_df: pd.DataFrame, tickers_df: pd.DataFrame, min_avg_value: float = 0.0) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    out = []
    for ticker, g in price_df.groupby("ticker"):
        g = g.sort_values("date").copy()
        close = g["close"]
        vol = pd.to_numeric(g.get("volume"), errors="coerce").fillna(0.0)
        avg_value = float((close * vol).tail(20).mean()) if len(g) >= 20 else float((close * vol).mean())
        if avg_value < min_avg_value:
            verdict = "ILLIQUID"
        else:
            verdict = "NEUTRAL"
        tq = _trend_quality(close)
        phase = _phase(close)
        breakout = max(0.0, min(100.0, 50 + 300 * (_ret(close, 20) if math.isfinite(_ret(close, 20)) else 0)))
        false_br = max(0.0, min(100.0, 50 - 200 * (_ret(close, 5) if math.isfinite(_ret(close, 5)) else 0)))
        dry = max(0.0, min(100.0, 100 - 100 * vol.tail(20).rank(pct=True).iloc[-1])) if len(vol) >= 20 else 50.0
        wet = 100 - dry
        long_rank = 0.35 * tq + 0.25 * breakout + 0.20 * dry - 0.20 * false_br
        risk_rank = 0.40 * wet + 0.30 * false_br + 0.30 * max(0.0, 100 - tq)
        if verdict != "ILLIQUID":
            if long_rank >= 70 and phase in {"MARKUP","EARLY_MARKUP","BASE"}:
                verdict = "READY_LONG"
            elif long_rank >= 55:
                verdict = "WATCH"
            elif risk_rank >= 68 and phase == "MARKDOWN":
                verdict = "TRIM"
            elif risk_rank >= 62:
                verdict = "AVOID"
            else:
                verdict = "NEUTRAL"
        out.append({
            "ticker": ticker,
            "close": round(_safe_last(close), 2),
            "phase": phase,
            "trend_quality": round(tq, 1),
            "breakout_integrity": round(breakout, 1),
            "false_breakout_risk": round(false_br, 1),
            "dry_score": round(dry, 1),
            "wet_score": round(wet, 1),
            "avg_traded_value_20d": round(avg_value, 0),
            "long_rank_score": round(long_rank, 1),
            "risk_rank_score": round(risk_rank, 1),
            "ret_20d": _ret(close, 20),
            "ret_60d": _ret(close, 60),
            "verdict": verdict,
        })
    out_df = pd.DataFrame(out).merge(tickers_df, on="ticker", how="left")
    out_df["sector"] = out_df["sector"].fillna("Unknown")
    return out_df.sort_values(["long_rank_score", "risk_rank_score"], ascending=[False, True]).reset_index(drop=True)
