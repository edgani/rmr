from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "ticker", "price", "lot"]


def normalize_done_detail(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED + ["side_aggressor"])
    out = df.copy()
    rename = {}
    lowered = {str(c).lower().strip(): c for c in out.columns}
    for want in ["timestamp", "ticker", "price", "lot", "buyer_broker", "seller_broker", "side_aggressor"]:
        if want in out.columns:
            continue
        if want in lowered:
            rename[lowered[want]] = want
        elif want == "timestamp":
            for cand in ["time", "datetime", "trade_time"]:
                if cand in lowered:
                    rename[lowered[cand]] = want
                    break
        elif want == "side_aggressor":
            for cand in ["side", "aggressor", "aggressor_side"]:
                if cand in lowered:
                    rename[lowered[cand]] = want
                    break
    out = out.rename(columns=rename)
    for c in REQUIRED:
        if c not in out.columns:
            out[c] = pd.NA
    if "side_aggressor" not in out.columns:
        out["side_aggressor"] = pd.NA

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["lot"] = pd.to_numeric(out["lot"], errors="coerce").fillna(0.0)
    out["side_aggressor"] = out["side_aggressor"].astype(str).str.upper().str.strip()
    out["side_aggressor"] = out["side_aggressor"].replace({
        "BUY": "BUY", "B": "BUY", "1": "BUY", "BOT": "BUY",
        "SELL": "SELL", "S": "SELL", "-1": "SELL", "SLD": "SELL",
        "NAN": pd.NA, "": pd.NA,
    })
    out = out.dropna(subset=["timestamp", "ticker", "price"]) 
    return out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)



def compute_done_bursts(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    df = normalize_done_detail(df)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "gulungan_up_score", "gulungan_down_score", "effort_result_up",
            "effort_result_down", "latest_event_label", "burst_bias", "last_burst_ts"
        ])
    rows = []
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.copy().sort_values("timestamp")
        if g.empty:
            continue
        g["bucket"] = g["timestamp"].dt.floor(window)
        g["buy_lot"] = np.where(g["side_aggressor"].eq("BUY"), g["lot"], 0.0)
        g["sell_lot"] = np.where(g["side_aggressor"].eq("SELL"), g["lot"], 0.0)
        by = g.groupby("bucket").agg(
            price_open=("price", "first"),
            price_close=("price", "last"),
            price_high=("price", "max"),
            price_low=("price", "min"),
            buy_lot=("buy_lot", "sum"),
            sell_lot=("sell_lot", "sum"),
            total_lot=("lot", "sum"),
            trade_count=("lot", "size"),
        ).reset_index()
        if by.empty:
            continue
        by["up_ticks"] = (by["price_close"] - by["price_open"]).clip(lower=0)
        by["down_ticks"] = (by["price_open"] - by["price_close"]).clip(lower=0)
        lot_scale = max(float(by["total_lot"].quantile(0.95)), 1.0)
        count_scale = max(float(by["trade_count"].quantile(0.95)), 1.0)
        up = (
            0.35 * np.clip(by["buy_lot"] / lot_scale, 0, 1)
            + 0.20 * np.clip((by["buy_lot"] - by["sell_lot"]) / np.maximum(by["total_lot"], 1.0), -1, 1).clip(lower=0)
            + 0.25 * np.clip(by["up_ticks"] / np.maximum((by["price_high"] - by["price_low"]).replace(0, np.nan).fillna(1.0), 1.0), 0, 1)
            + 0.20 * np.clip(by["trade_count"] / count_scale, 0, 1)
        )
        down = (
            0.35 * np.clip(by["sell_lot"] / lot_scale, 0, 1)
            + 0.20 * np.clip((by["sell_lot"] - by["buy_lot"]) / np.maximum(by["total_lot"], 1.0), -1, 1).clip(lower=0)
            + 0.25 * np.clip(by["down_ticks"] / np.maximum((by["price_high"] - by["price_low"]).replace(0, np.nan).fillna(1.0), 1.0), 0, 1)
            + 0.20 * np.clip(by["trade_count"] / count_scale, 0, 1)
        )
        by["up_score"] = (100 * up).round(1)
        by["down_score"] = (100 * down).round(1)
        by["effort_result_up"] = np.where(by["buy_lot"] > 0, 100 * by["up_ticks"] / np.log1p(by["buy_lot"]), 0.0)
        by["effort_result_down"] = np.where(by["sell_lot"] > 0, 100 * by["down_ticks"] / np.log1p(by["sell_lot"]), 0.0)
        last = by.iloc[-1]
        label = "NO_INTRADAY_SIGNAL"
        bias = "NEUTRAL"
        med_up = by["effort_result_up"].replace(0, np.nan).median()
        med_down = by["effort_result_down"].replace(0, np.nan).median()
        if last["up_score"] >= 60 and last["up_score"] > last["down_score"] * 1.15:
            is_cont = False if pd.isna(med_up) else last["effort_result_up"] >= med_up
            label = "UP_CONTINUATION_BURST" if is_cont else "UP_INITIATIVE_SWEEP"
            bias = "BULLISH"
        elif last["down_score"] >= 60 and last["down_score"] > last["up_score"] * 1.15:
            is_cont = False if pd.isna(med_down) else last["effort_result_down"] >= med_down
            label = "DOWN_CONTINUATION_BREAK" if is_cont else "DOWN_INITIATIVE_SWEEP"
            bias = "BEARISH"
        rows.append({
            "ticker": ticker,
            "gulungan_up_score": round(float(last["up_score"]), 1),
            "gulungan_down_score": round(float(last["down_score"]), 1),
            "effort_result_up": round(float(last["effort_result_up"]), 2),
            "effort_result_down": round(float(last["effort_result_down"]), 2),
            "latest_event_label": label,
            "burst_bias": bias,
            "last_burst_ts": last["bucket"],
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
