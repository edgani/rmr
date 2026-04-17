from __future__ import annotations

import numpy as np
import pandas as pd

from .broker import compute_broker_context
from .confidence import compute_confidence
from .done_detail import compute_done_bursts
from .drywet import compute_broker_aware_drywet
from .explain import build_invalidator, build_risk_note, build_trigger, build_why_not_yet, build_why_now
from .orderbook import compute_orderbook_context


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_price_side_features(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    out_rows = []
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.copy().sort_values("date")
        if len(g) < 40:
            continue
        close = pd.to_numeric(g["close"], errors="coerce")
        high = pd.to_numeric(g["high"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        vol = pd.to_numeric(g["volume"], errors="coerce").fillna(0)
        if close.isna().all():
            continue

        g["ema20"] = close.ewm(span=20, adjust=False).mean()
        g["ema50"] = close.ewm(span=50, adjust=False).mean()
        g["ema200"] = close.ewm(span=200, adjust=False).mean()
        tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        ret = close.pct_change()
        vol_avg20 = vol.rolling(20).mean()
        high_20 = high.rolling(20).max()
        high_60 = high.rolling(60).max()
        low_20 = low.rolling(20).min()

        last = g.iloc[-1]
        last_close = float(last["close"])
        ema20, ema50, ema200 = float(last["ema20"]), float(last["ema50"]), float(last["ema200"])
        atr = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan
        h60 = float(high_60.iloc[-1]) if pd.notna(high_60.iloc[-1]) else np.nan
        l20 = float(low_20.iloc[-1]) if pd.notna(low_20.iloc[-1]) else np.nan
        liquidity_mn = float((close.iloc[-20:] * vol.iloc[-20:]).mean() / 1_000_000.0)

        if last_close > ema20 > ema50 > ema200:
            trend_quality = 100.0
        elif last_close > ema20 > ema50:
            trend_quality = 66.7
        elif last_close > ema50:
            trend_quality = 33.3
        elif last_close < ema20 < ema50 < ema200:
            trend_quality = 0.0
        else:
            trend_quality = 25.0

        range_20 = float((high.iloc[-20:].max() - low.iloc[-20:].min()) / max(last_close, 1e-9))
        base_maturity = _clip01(min(len(g), 60) / 60)
        breakout_distance = 0.0 if np.isnan(h60) else (last_close / max(h60, 1e-9)) - 1.0
        breakout_integrity = 100 * _clip01(
            0.40 * (1.0 if breakout_distance >= -0.01 else max(0.0, 1 + breakout_distance * 10))
            + 0.30 * (1 - min(range_20 / 0.25, 1))
            + 0.30 * base_maturity
        )

        upper_wick = float((last["high"] - max(last["close"], last["open"])) / max(last["high"] - last["low"], 1e-9))
        false_breakout_risk = 100 * _clip01(0.55 * upper_wick + 0.45 * (0 if breakout_distance >= 0 else min(abs(breakout_distance) * 12, 1)))
        vol_burst = float(vol.iloc[-5:].mean() / max(vol_avg20.iloc[-1], 1e-9)) if pd.notna(vol_avg20.iloc[-1]) else 1.0
        realized_vol = float(ret.iloc[-20:].std() * np.sqrt(252)) if len(ret.iloc[-20:]) >= 5 else np.nan
        dry_score = 100 * _clip01(0.6 * (1 - min((realized_vol or 0.3) / 1.2, 1)) + 0.4 * min(vol_burst / 2.0, 1))
        wet_score = 100 - dry_score

        phase = "NEUTRAL"
        if last_close > ema20 > ema50 and breakout_integrity >= 55:
            phase = "MARKUP"
        elif last_close > ema50 and last_close < ema20:
            phase = "PULLBACK_HEALTHY"
        elif last_close > ema50 and range_20 < 0.14:
            phase = "ACCUMULATION"
        elif last_close < ema20 < ema50:
            phase = "MARKDOWN"

        out_rows.append({
            "date": pd.to_datetime(last["date"]).date(),
            "ticker": ticker,
            "close": round(last_close, 4),
            "phase": phase,
            "trend_quality": round(trend_quality, 1),
            "breakout_integrity": round(breakout_integrity, 1),
            "false_breakout_risk": round(false_breakout_risk, 1),
            "dry_score": round(dry_score, 1),
            "wet_score": round(wet_score, 1),
            "liquidity_mn": round(liquidity_mn, 1),
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema200": round(ema200, 4),
            "support_20d": round(l20, 4) if pd.notna(l20) else np.nan,
            "resistance_60d": round(h60, 4) if pd.notna(h60) else np.nan,
            "atr14": round(float(atr), 4) if pd.notna(atr) else np.nan,
            "records_used": len(g),
        })
    return pd.DataFrame(out_rows)



def compute_ticker_features(
    price_df: pd.DataFrame,
    broker_df: pd.DataFrame | None = None,
    done_df: pd.DataFrame | None = None,
    orderbook_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = compute_price_side_features(price_df)
    if out.empty:
        return out

    broker_ctx = compute_broker_context(broker_df if broker_df is not None else pd.DataFrame())
    burst_ctx = compute_done_bursts(done_df if done_df is not None else pd.DataFrame())
    book_ctx = compute_orderbook_context(orderbook_df if orderbook_df is not None else pd.DataFrame())
    drywet_ctx = compute_broker_aware_drywet(out, broker_ctx)

    if not broker_ctx.empty:
        out = out.merge(broker_ctx, on="ticker", how="left")
    else:
        out["broker_alignment_score"] = np.nan
        out["broker_mode"] = "NO_BROKER_UPLOAD"
        out["dominant_accumulator"] = "-"
        out["dominant_distributor"] = "-"
        out["institutional_support"] = np.nan
        out["institutional_resistance"] = np.nan
        out["overhang_score"] = np.nan
        out["broker_concentration_score"] = np.nan
        out["broker_data_days"] = 0
    if not burst_ctx.empty:
        out = out.merge(burst_ctx, on="ticker", how="left")
    else:
        out["gulungan_up_score"] = np.nan
        out["gulungan_down_score"] = np.nan
        out["effort_result_up"] = np.nan
        out["effort_result_down"] = np.nan
        out["latest_event_label"] = "NO_INTRADAY_SIGNAL"
        out["burst_bias"] = "NEUTRAL"
        out["last_burst_ts"] = pd.NaT
    if not book_ctx.empty:
        out = out.merge(book_ctx, on="ticker", how="left")
    else:
        out["bid_stack_quality"] = np.nan
        out["offer_stack_quality"] = np.nan
        out["absorption_after_up_score"] = np.nan
        out["absorption_after_down_score"] = np.nan
        out["tension_score"] = np.nan
        out["fragility_score"] = np.nan
    out = out.merge(drywet_ctx, on="ticker", how="left")

    out = _to_num(out, [
        "trend_quality", "breakout_integrity", "false_breakout_risk", "dry_score", "wet_score",
        "dry_score_final", "wet_score_final", "liquidity_mn", "broker_alignment_score",
        "institutional_support", "institutional_resistance", "overhang_score",
        "gulungan_up_score", "gulungan_down_score", "effort_result_up", "effort_result_down",
        "bid_stack_quality", "offer_stack_quality", "absorption_after_up_score", "absorption_after_down_score",
        "tension_score", "fragility_score", "support_20d", "resistance_60d"
    ])

    out["dry_score_final"] = out["dry_score_final"].fillna(out["dry_score"])
    out["wet_score_final"] = out["wet_score_final"].fillna(out["wet_score"])
    out["broker_alignment_score"] = out["broker_alignment_score"].fillna(0.0)
    out["overhang_score"] = out["overhang_score"].fillna(0.0)
    out["gulungan_up_score"] = out["gulungan_up_score"].fillna(0.0)
    out["gulungan_down_score"] = out["gulungan_down_score"].fillna(0.0)
    out["effort_result_up"] = out["effort_result_up"].fillna(0.0)
    out["effort_result_down"] = out["effort_result_down"].fillna(0.0)
    out["absorption_after_up_score"] = out["absorption_after_up_score"].fillna(0.0)
    out["absorption_after_down_score"] = out["absorption_after_down_score"].fillna(0.0)

    verdicts = []
    for _, row in out.iterrows():
        phase = row["phase"]
        tq = row["trend_quality"]
        bi = row["breakout_integrity"]
        fb = row["false_breakout_risk"]
        dry = row["dry_score_final"]
        wet = row["wet_score_final"]
        liq = row["liquidity_mn"]
        broker = row["broker_alignment_score"]
        overhang = row["overhang_score"]
        burst_bias = str(row.get("burst_bias", "NEUTRAL"))
        up = row["gulungan_up_score"]
        down = row["gulungan_down_score"]
        absorb_up = row["absorption_after_up_score"]
        absorb_down = row["absorption_after_down_score"]

        verdict = "NEUTRAL"
        if liq < 5:
            verdict = "ILLIQUID"
        elif tq >= 66 and bi >= 55 and fb <= 35 and dry >= 52 and broker >= 55 and not (up >= 60 and absorb_up >= 60):
            verdict = "READY_LONG"
        elif phase in {"MARKUP", "ACCUMULATION", "PULLBACK_HEALTHY"} and bi >= 42 and fb <= 45:
            verdict = "WATCH"
        elif down >= 60 and absorb_down >= 60 and phase in {"MARKDOWN", "PULLBACK_HEALTHY"}:
            verdict = "WATCH_REBOUND"
        elif phase == "MARKDOWN" and (wet >= 60 or overhang >= 55 or burst_bias == "BEARISH"):
            verdict = "TRIM"
        elif phase == "MARKDOWN" and (fb >= 45 or broker < 40):
            verdict = "AVOID"
        elif phase == "PULLBACK_HEALTHY" and dry >= 55 and broker >= 50:
            verdict = "WATCH_REBOUND"
        verdicts.append(verdict)
    out["verdict"] = verdicts

    conf = compute_confidence(out)
    out = out.merge(conf, on="ticker", how="left")

    out["why_now"] = out.apply(build_why_now, axis=1)
    out["why_not_yet"] = out.apply(build_why_not_yet, axis=1)
    out["trigger"] = out.apply(build_trigger, axis=1)
    out["invalidator"] = out.apply(build_invalidator, axis=1)
    out["dominant_risk"] = out.apply(build_risk_note, axis=1)

    round_cols = [
        "close", "trend_quality", "breakout_integrity", "false_breakout_risk", "dry_score", "wet_score",
        "dry_score_final", "wet_score_final", "liquidity_mn", "broker_alignment_score", "institutional_support",
        "institutional_resistance", "overhang_score", "gulungan_up_score", "gulungan_down_score",
        "effort_result_up", "effort_result_down", "bid_stack_quality", "offer_stack_quality",
        "absorption_after_up_score", "absorption_after_down_score", "tension_score", "fragility_score",
        "score_confidence", "data_completeness_score", "module_agreement_score", "support_20d", "resistance_60d"
    ]
    out = _to_num(out, round_cols)
    for c in round_cols:
        if c in out.columns:
            out[c] = out[c].round(2)

    sort_order = {"READY_LONG": 0, "WATCH": 1, "WATCH_REBOUND": 2, "NEUTRAL": 3, "TRIM": 4, "AVOID": 5, "ILLIQUID": 6}
    out["_sort"] = out["verdict"].map(sort_order).fillna(99)
    return out.sort_values(["_sort", "score_confidence", "trend_quality", "breakout_integrity", "liquidity_mn"], ascending=[True, False, False, False, False]).drop(columns=["_sort"]).reset_index(drop=True)
