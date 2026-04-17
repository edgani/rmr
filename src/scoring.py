from __future__ import annotations

import numpy as np
import pandas as pd

from .broker import compute_broker_context
from .confidence import compute_confidence
from .done_detail import compute_done_bursts
from .drywet import compute_broker_aware_drywet
from .explain import build_invalidator, build_risk_note, build_trigger, build_why_not_yet, build_why_now
from .orderbook import compute_orderbook_context
from .regime import compute_market_regime
from .thresholds import get_long_thresholds, get_risk_thresholds, liquidity_bucket


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
    ret20_all = []
    temp = {}
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

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        ret = close.pct_change()
        vol_avg20 = vol.rolling(20).mean()
        high_60 = high.rolling(60).max()
        low_20 = low.rolling(20).min()

        last_close = float(close.iloc[-1])
        ema20v, ema50v, ema200v = float(ema20.iloc[-1]), float(ema50.iloc[-1]), float(ema200.iloc[-1])
        atr = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan
        h60 = float(high_60.iloc[-1]) if pd.notna(high_60.iloc[-1]) else np.nan
        l20 = float(low_20.iloc[-1]) if pd.notna(low_20.iloc[-1]) else np.nan
        liquidity_mn = float((close.iloc[-20:] * vol.iloc[-20:]).mean() / 1_000_000.0)
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 and pd.notna(close.iloc[-21]) else np.nan
        ret5 = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 and pd.notna(close.iloc[-6]) else np.nan
        ret20_all.append(ret20)

        if last_close > ema20v > ema50v > ema200v:
            trend_quality = 100.0
        elif last_close > ema20v > ema50v:
            trend_quality = 66.7
        elif last_close > ema50v:
            trend_quality = 33.3
        elif last_close < ema20v < ema50v < ema200v:
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

        last = g.iloc[-1]
        upper_wick = float((last["high"] - max(last["close"], last["open"])) / max(last["high"] - last["low"], 1e-9))
        false_breakout_risk = 100 * _clip01(0.55 * upper_wick + 0.45 * (0 if breakout_distance >= 0 else min(abs(breakout_distance) * 12, 1)))
        vol_burst = float(vol.iloc[-5:].mean() / max(vol_avg20.iloc[-1], 1e-9)) if pd.notna(vol_avg20.iloc[-1]) else 1.0
        realized_vol = float(ret.iloc[-20:].std() * np.sqrt(252)) if len(ret.iloc[-20:]) >= 5 else np.nan
        dry_score = 100 * _clip01(0.6 * (1 - min((realized_vol or 0.3) / 1.2, 1)) + 0.4 * min(vol_burst / 2.0, 1))
        wet_score = 100 - dry_score

        phase = "NEUTRAL"
        if last_close > ema20v > ema50v and breakout_integrity >= 55:
            phase = "MARKUP"
        elif last_close > ema50v and last_close < ema20v:
            phase = "PULLBACK_HEALTHY"
        elif last_close > ema50v and range_20 < 0.14:
            phase = "ACCUMULATION"
        elif last_close < ema20v < ema50v:
            phase = "MARKDOWN"

        temp[ticker] = {
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
            "ema20": round(ema20v, 4),
            "ema50": round(ema50v, 4),
            "ema200": round(ema200v, 4),
            "support_20d": round(l20, 4) if pd.notna(l20) else np.nan,
            "resistance_60d": round(h60, 4) if pd.notna(h60) else np.nan,
            "atr14": round(float(atr), 4) if pd.notna(atr) else np.nan,
            "ret20": round(ret20, 4) if pd.notna(ret20) else np.nan,
            "ret5": round(ret5, 4) if pd.notna(ret5) else np.nan,
            "records_used": len(g),
        }

    if not temp:
        return pd.DataFrame()
    med_ret20 = np.nanmedian(ret20_all) if len(ret20_all) else np.nan
    for ticker, row in temp.items():
        rs = (row["ret20"] - med_ret20) if pd.notna(row.get("ret20")) and pd.notna(med_ret20) else np.nan
        row["relative_strength_20d"] = round(float(rs), 4) if pd.notna(rs) else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _add_metadata_and_sector_features(out: pd.DataFrame, metadata_df: pd.DataFrame | None) -> pd.DataFrame:
    if metadata_df is None or metadata_df.empty:
        out["sector"] = pd.NA
        out["sector_relative_strength_20d"] = np.nan
        return out
    meta = metadata_df.copy()
    if "ticker" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "ticker"})
    meta["ticker"] = meta["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
    keep = [c for c in ["ticker", "sector", "board", "name"] if c in meta.columns]
    meta = meta[keep].drop_duplicates("ticker")
    out = out.merge(meta, on="ticker", how="left")
    if "sector" in out.columns and out["sector"].notna().any():
        sector_med = out.groupby("sector", dropna=True)["ret20"].transform("median")
        out["sector_relative_strength_20d"] = out["ret20"] - sector_med
    else:
        out["sector_relative_strength_20d"] = np.nan
    return out


def _compute_rank_scores(out: pd.DataFrame) -> pd.DataFrame:
    rs20 = pd.to_numeric(out.get("relative_strength_20d", 0), errors="coerce").fillna(0.0)
    sec_rs = pd.to_numeric(out.get("sector_relative_strength_20d", 0), errors="coerce").fillna(0.0)
    broker = pd.to_numeric(out.get("broker_alignment_score", 0), errors="coerce").fillna(0.0)
    persistence = pd.to_numeric(out.get("broker_persistence_score", 0), errors="coerce").fillna(0.0)
    overhang = pd.to_numeric(out.get("overhang_score", 0), errors="coerce").fillna(0.0)
    dry = pd.to_numeric(out.get("dry_score_final", out.get("dry_score", 0)), errors="coerce").fillna(0.0)
    bi = pd.to_numeric(out.get("breakout_integrity", 0), errors="coerce").fillna(0.0)
    tq = pd.to_numeric(out.get("trend_quality", 0), errors="coerce").fillna(0.0)
    fb = pd.to_numeric(out.get("false_breakout_risk", 50), errors="coerce").fillna(50.0)
    conf = pd.to_numeric(out.get("score_confidence", 0), errors="coerce").fillna(0.0)
    market_bias = pd.to_numeric(out.get("market_bias_score", 50), errors="coerce").fillna(50.0)
    burst_up = pd.to_numeric(out.get("gulungan_up_score", 0), errors="coerce").fillna(0.0)
    burst_dn = pd.to_numeric(out.get("gulungan_down_score", 0), errors="coerce").fillna(0.0)
    absorb_up = pd.to_numeric(out.get("absorption_after_up_score", 0), errors="coerce").fillna(0.0)
    absorb_dn = pd.to_numeric(out.get("absorption_after_down_score", 0), errors="coerce").fillna(0.0)
    fu = pd.to_numeric(out.get("post_up_followthrough_score", 0), errors="coerce").fillna(0.0)
    fd = pd.to_numeric(out.get("post_down_followthrough_score", 0), errors="coerce").fillna(0.0)
    btrap = pd.to_numeric(out.get("bull_trap_score", 0), errors="coerce").fillna(0.0)
    beartrap = pd.to_numeric(out.get("bear_trap_score", 0), errors="coerce").fillna(0.0)

    out["long_rank_score"] = (
        0.18 * tq
        + 0.18 * bi
        + 0.12 * dry
        + 0.10 * broker
        + 0.08 * persistence
        + 0.08 * np.clip(rs20 * 1000, -20, 20)
        + 0.06 * np.clip(sec_rs * 1000, -15, 15)
        + 0.10 * conf
        + 0.05 * market_bias
        + 0.06 * burst_up
        + 0.05 * fu
        - 0.10 * fb
        - 0.05 * absorb_up
        - 0.05 * btrap
        - 0.04 * overhang
    ).round(2)
    out["risk_rank_score"] = (
        0.18 * pd.to_numeric(out.get("wet_score_final", out.get("wet_score", 0)), errors="coerce").fillna(0.0)
        + 0.14 * fb
        + 0.12 * overhang
        + 0.08 * burst_dn
        + 0.08 * absorb_dn
        + 0.06 * fd
        + 0.06 * beartrap
        + 0.08 * (100 - broker)
        + 0.08 * (50 - np.clip(rs20 * 500, -50, 50))
        + 0.10 * (100 - market_bias)
        + 0.14 * (100 - conf)
    ).round(2)
    return out


def compute_ticker_features(
    price_df: pd.DataFrame,
    broker_df: pd.DataFrame | None = None,
    done_df: pd.DataFrame | None = None,
    orderbook_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    broker_master_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = compute_price_side_features(price_df)
    if out.empty:
        return out

    out = _add_metadata_and_sector_features(out, metadata_df)

    market_ctx = compute_market_regime(price_df).iloc[0].to_dict()
    broker_ctx = compute_broker_context(broker_df if broker_df is not None else pd.DataFrame(), broker_master_df=broker_master_df)
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
        out["post_up_followthrough_score"] = np.nan
        out["post_down_followthrough_score"] = np.nan
        out["split_order_score"] = np.nan
        out["bull_trap_score"] = np.nan
        out["bear_trap_score"] = np.nan
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
        out["offer_refill_rate"] = np.nan
        out["bid_refill_rate"] = np.nan
        out["fake_wall_offer_score"] = np.nan
        out["fake_wall_bid_score"] = np.nan
    out = out.merge(drywet_ctx, on="ticker", how="left")

    out = _to_num(out, [
        "trend_quality", "breakout_integrity", "false_breakout_risk", "dry_score", "wet_score",
        "dry_score_final", "wet_score_final", "liquidity_mn", "broker_alignment_score",
        "institutional_support", "institutional_resistance", "institutional_support_low", "institutional_support_high", "institutional_resistance_low", "institutional_resistance_high", "overhang_score",
        "gulungan_up_score", "gulungan_down_score", "effort_result_up", "effort_result_down",
        "post_up_followthrough_score", "post_down_followthrough_score", "split_order_score", "bull_trap_score", "bear_trap_score",
        "bid_stack_quality", "offer_stack_quality", "absorption_after_up_score", "absorption_after_down_score",
        "tension_score", "fragility_score", "offer_refill_rate", "bid_refill_rate", "fake_wall_offer_score", "fake_wall_bid_score", "support_20d", "resistance_60d", "relative_strength_20d",
        "broker_persistence_score", "broker_concentration_score", "broker_acc_pressure", "broker_dist_pressure", "float_lock_score", "supply_overhang_score",
        "sector_relative_strength_20d"
    ])

    out["dry_score_final"] = out["dry_score_final"].fillna(out["dry_score"])
    out["wet_score_final"] = out["wet_score_final"].fillna(out["wet_score"])
    out["broker_alignment_score"] = out["broker_alignment_score"].fillna(0.0)
    out["overhang_score"] = out["overhang_score"].fillna(0.0)
    for c in ["broker_persistence_score", "broker_concentration_score", "broker_acc_pressure", "broker_dist_pressure", "float_lock_score", "supply_overhang_score"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["gulungan_up_score"] = out["gulungan_up_score"].fillna(0.0)
    out["gulungan_down_score"] = out["gulungan_down_score"].fillna(0.0)
    out["effort_result_up"] = out["effort_result_up"].fillna(0.0)
    out["effort_result_down"] = out["effort_result_down"].fillna(0.0)
    out["post_up_followthrough_score"] = out["post_up_followthrough_score"].fillna(0.0)
    out["post_down_followthrough_score"] = out["post_down_followthrough_score"].fillna(0.0)
    out["split_order_score"] = out["split_order_score"].fillna(0.0)
    out["bull_trap_score"] = out["bull_trap_score"].fillna(0.0)
    out["bear_trap_score"] = out["bear_trap_score"].fillna(0.0)
    out["absorption_after_up_score"] = out["absorption_after_up_score"].fillna(0.0)
    out["absorption_after_down_score"] = out["absorption_after_down_score"].fillna(0.0)
    out["offer_refill_rate"] = out["offer_refill_rate"].fillna(0.0)
    out["bid_refill_rate"] = out["bid_refill_rate"].fillna(0.0)
    out["fake_wall_offer_score"] = out["fake_wall_offer_score"].fillna(0.0)
    out["fake_wall_bid_score"] = out["fake_wall_bid_score"].fillna(0.0)

    out["market_regime"] = market_ctx.get("market_regime")
    out["execution_mode"] = market_ctx.get("execution_mode")
    out["market_breadth_pct"] = market_ctx.get("market_breadth_pct")
    out["market_bias_score"] = market_ctx.get("market_bias_score")
    out["market_vol_state"] = market_ctx.get("market_vol_state")
    out["liquidity_bucket"] = out["liquidity_mn"].apply(liquidity_bucket)

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
        persistence = row.get("broker_persistence_score", 0.0)
        dist_pressure = row.get("broker_dist_pressure", 0.0)
        float_lock = row.get("float_lock_score", 0.0)
        burst_bias = str(row.get("burst_bias", "NEUTRAL"))
        up = row["gulungan_up_score"]
        down = row["gulungan_down_score"]
        absorb_up = row["absorption_after_up_score"]
        absorb_down = row["absorption_after_down_score"]
        regime = row.get("market_regime", "CHOPPY")
        rs20 = row.get("relative_strength_20d", np.nan)
        sec_rs20 = row.get("sector_relative_strength_20d", np.nan)

        long_thr = get_long_thresholds(liq, regime)
        risk_thr = get_risk_thresholds(liq, regime)

        fu = row.get("post_up_followthrough_score", 0.0)
        fd = row.get("post_down_followthrough_score", 0.0)
        bull_trap = row.get("bull_trap_score", 0.0)
        bear_trap = row.get("bear_trap_score", 0.0)
        split_score = row.get("split_order_score", 0.0)
        fake_wall_offer = row.get("fake_wall_offer_score", 0.0)
        fake_wall_bid = row.get("fake_wall_bid_score", 0.0)
        verdict = "NEUTRAL"
        if liq < 5:
            verdict = "ILLIQUID"
        elif (
            tq >= long_thr["tq"] and bi >= long_thr["bi"] and fb <= long_thr["fb"]
            and dry >= long_thr["dry"] and broker >= long_thr["broker"] and persistence >= 48
            and (pd.isna(rs20) or rs20 >= -0.01) and (pd.isna(sec_rs20) or sec_rs20 >= -0.01)
            and not (up >= 60 and (absorb_up >= 60 or bull_trap >= 60)) and dist_pressure < 65 and fu >= 45
        ):
            verdict = "READY_LONG"
        elif phase in {"MARKUP", "ACCUMULATION", "PULLBACK_HEALTHY"} and bi >= max(42, long_thr["bi"] - 10) and fb <= min(45, long_thr["fb"] + 8) and broker >= max(45, long_thr["broker"] - 10):
            verdict = "WATCH"
        elif down >= 60 and (absorb_down >= 60 or bear_trap >= 60) and phase in {"MARKDOWN", "PULLBACK_HEALTHY"} and float_lock >= 40 and fd <= 55:
            verdict = "WATCH_REBOUND"
        elif phase == "MARKDOWN" and (wet >= risk_thr["wet"] or overhang >= risk_thr["overhang"] or burst_bias == "BEARISH" or dist_pressure >= 60 or fd >= 50):
            verdict = "TRIM"
        elif phase == "MARKDOWN" and (fb >= risk_thr["fb"] or broker < risk_thr["broker_low"] or fake_wall_offer >= 65):
            verdict = "AVOID"
        elif phase == "PULLBACK_HEALTHY" and dry >= long_thr["dry"] and broker >= max(50, long_thr["broker"] - 5) and persistence >= 45 and bull_trap < 65:
            verdict = "WATCH_REBOUND"
        elif up >= 60 and bull_trap >= 65:
            verdict = "WATCH" if phase in {"ACCUMULATION", "PULLBACK_HEALTHY"} else "TRIM"
        elif split_score >= 70 and fake_wall_offer >= 70 and up >= 55:
            verdict = "WATCH"
        verdicts.append(verdict)
    out["verdict"] = verdicts

    conf = compute_confidence(out)
    out = out.merge(conf, on="ticker", how="left")
    out = _compute_rank_scores(out)

    out["why_now"] = out.apply(build_why_now, axis=1)
    out["why_not_yet"] = out.apply(build_why_not_yet, axis=1)
    out["trigger"] = out.apply(build_trigger, axis=1)
    out["invalidator"] = out.apply(build_invalidator, axis=1)
    out["dominant_risk"] = out.apply(build_risk_note, axis=1)

    round_cols = [
        "close", "trend_quality", "breakout_integrity", "false_breakout_risk", "dry_score", "wet_score",
        "dry_score_final", "wet_score_final", "liquidity_mn", "broker_alignment_score", "institutional_support",
        "institutional_resistance", "overhang_score", "gulungan_up_score", "gulungan_down_score",
        "effort_result_up", "effort_result_down", "post_up_followthrough_score", "post_down_followthrough_score", "split_order_score", "bull_trap_score", "bear_trap_score", "bid_stack_quality", "offer_stack_quality",
        "absorption_after_up_score", "absorption_after_down_score", "tension_score", "fragility_score", "offer_refill_rate", "bid_refill_rate", "fake_wall_offer_score", "fake_wall_bid_score",
        "score_confidence", "data_completeness_score", "module_agreement_score", "support_20d", "resistance_60d",
        "market_breadth_pct", "market_bias_score", "market_vol_state", "relative_strength_20d",
        "sector_relative_strength_20d", "long_rank_score", "risk_rank_score"
    ]
    out = _to_num(out, round_cols)
    for c in round_cols:
        if c in out.columns:
            out[c] = out[c].round(2)

    sort_order = {"READY_LONG": 0, "WATCH": 1, "WATCH_REBOUND": 2, "NEUTRAL": 3, "TRIM": 4, "AVOID": 5, "ILLIQUID": 6}
    out["_sort"] = out["verdict"].map(sort_order).fillna(99)
    return out.sort_values(
        ["_sort", "long_rank_score", "score_confidence", "trend_quality", "breakout_integrity", "liquidity_mn"],
        ascending=[True, False, False, False, False, False],
    ).drop(columns=["_sort"]).reset_index(drop=True)
