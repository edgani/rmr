from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "ticker", "price", "lot"]
ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "trade_time", "matched_at", "ts"],
    "ticker": ["ticker", "symbol", "kode", "stock", "security"],
    "price": ["price", "trade_price", "last", "matched_price"],
    "lot": ["lot", "qty", "volume", "trade_lot", "size"],
    "buyer_broker": ["buyer_broker", "buy_broker", "buyer", "broker_buy", "broker_b"],
    "seller_broker": ["seller_broker", "sell_broker", "seller", "broker_sell", "broker_s"],
    "side_aggressor": ["side_aggressor", "aggressor", "aggressor_side", "side", "initiator"],
}

SIDE_MAP = {
    "BUY": "BUY", "B": "BUY", "BOT": "BUY", "BUYER": "BUY", "1": "BUY", "UP": "BUY",
    "SELL": "SELL", "S": "SELL", "SLD": "SELL", "SELLER": "SELL", "-1": "SELL", "DOWN": "SELL",
}


def _first_matching(columns, aliases):
    lower = {str(c).lower().strip(): c for c in columns}
    for cand in aliases:
        if cand in lower:
            return lower[cand]
    return None


def _clean_numeric(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.replace(",", "", regex=False).str.replace("_", "", regex=False).str.strip()
    txt = txt.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")


def _infer_side(g: pd.DataFrame) -> pd.Series:
    side = g["side_aggressor"].copy()
    price = pd.to_numeric(g["price"], errors="coerce")
    diff = price.diff()
    inferred = np.where(diff > 0, "BUY", np.where(diff < 0, "SELL", np.nan))
    side = side.where(side.notna(), pd.Series(inferred, index=g.index))
    side = side.ffill().bfill()
    return side.fillna("UNKNOWN")


def normalize_done_detail(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED + ["buyer_broker", "seller_broker", "side_aggressor"])
    out = df.copy()
    rename = {}
    for want, aliases in ALIASES.items():
        if want not in out.columns:
            found = _first_matching(out.columns, aliases)
            if found is not None:
                rename[found] = want
    out = out.rename(columns=rename)
    for c in REQUIRED + ["buyer_broker", "seller_broker", "side_aggressor"]:
        if c not in out.columns:
            out[c] = pd.NA
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out["price"] = _clean_numeric(out["price"])
    out["lot"] = _clean_numeric(out["lot"]).fillna(0.0)
    for c in ["buyer_broker", "seller_broker"]:
        out[c] = out[c].astype(str).str.upper().str.strip().replace({"NAN": pd.NA, "": pd.NA})
    out["side_aggressor"] = out["side_aggressor"].astype(str).str.upper().str.strip().map(SIDE_MAP)
    out = out.dropna(subset=["timestamp", "ticker", "price"]).copy()
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{2,8}$", na=False)]
    out = out.sort_values(["ticker", "timestamp", "price"]).reset_index(drop=True)
    out["ts_sec"] = out["timestamp"].dt.floor("1s")
    pieces = []
    for ticker, g in out.groupby("ticker", sort=True):
        g = g.copy().sort_values("timestamp")
        g["side_aggressor"] = _infer_side(g)
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True) if pieces else out
    return out


def _safe_score(x) -> float:
    try:
        return float(np.clip(x, 0.0, 100.0))
    except Exception:
        return 0.0


def compute_done_bursts(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    df = normalize_done_detail(df)
    cols = [
        "ticker", "gulungan_up_score", "gulungan_down_score", "effort_result_up", "effort_result_down",
        "post_up_followthrough_score", "post_down_followthrough_score", "split_order_score",
        "bull_trap_score", "bear_trap_score", "latest_event_label", "burst_bias", "last_burst_ts"
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.copy().sort_values("timestamp")
        if g.empty:
            continue
        g["bucket"] = g["timestamp"].dt.floor(window)
        g["buy_lot"] = np.where(g["side_aggressor"].eq("BUY"), g["lot"], 0.0)
        g["sell_lot"] = np.where(g["side_aggressor"].eq("SELL"), g["lot"], 0.0)

        sec_cluster = g.groupby(["bucket", "ts_sec", "price", "side_aggressor"], as_index=False).agg(
            cluster_lot=("lot", "sum"), cluster_trades=("lot", "size")
        )
        sec_cluster["cluster_share"] = sec_cluster.groupby("bucket")["cluster_lot"].transform(lambda s: s / max(float(s.sum()), 1.0))
        cluster_summary = sec_cluster.groupby("bucket", as_index=False).agg(
            max_cluster_share=("cluster_share", "max"),
            same_second_cluster_count=("cluster_trades", lambda s: int((pd.Series(s) >= 2).sum())),
            max_cluster_trades=("cluster_trades", "max"),
        )

        by = g.groupby("bucket", as_index=False).agg(
            price_open=("price", "first"),
            price_close=("price", "last"),
            price_high=("price", "max"),
            price_low=("price", "min"),
            buy_lot=("buy_lot", "sum"),
            sell_lot=("sell_lot", "sum"),
            total_lot=("lot", "sum"),
            trade_count=("lot", "size"),
            unique_prices=("price", "nunique"),
        ).merge(cluster_summary, on="bucket", how="left")
        by[["max_cluster_share", "same_second_cluster_count", "max_cluster_trades"]] = by[["max_cluster_share", "same_second_cluster_count", "max_cluster_trades"]].fillna(0.0)
        if by.empty:
            continue
        spread = (by["price_high"] - by["price_low"]).replace(0, np.nan)
        by["up_progress"] = (by["price_close"] - by["price_open"]).clip(lower=0)
        by["down_progress"] = (by["price_open"] - by["price_close"]).clip(lower=0)
        by["close_strength"] = ((by["price_close"] - by["price_low"]) / spread).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        by["close_weakness"] = ((by["price_high"] - by["price_close"]) / spread).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        by["buy_dom"] = ((by["buy_lot"] - by["sell_lot"]) / np.maximum(by["total_lot"], 1.0)).clip(lower=0)
        by["sell_dom"] = ((by["sell_lot"] - by["buy_lot"]) / np.maximum(by["total_lot"], 1.0)).clip(lower=0)
        lot_scale = max(float(by["total_lot"].quantile(0.95)), 1.0)
        count_scale = max(float(by["trade_count"].quantile(0.95)), 1.0)
        cluster_scale = max(float(by["same_second_cluster_count"].quantile(0.95)), 1.0)

        by["split_order_score"] = 100 * np.clip(
            0.55 * by["max_cluster_share"].fillna(0.0) + 0.45 * np.clip(by["same_second_cluster_count"] / cluster_scale, 0, 1),
            0, 1,
        )
        by["gulungan_up_score"] = 100 * np.clip(
            0.28 * np.clip(by["buy_lot"] / lot_scale, 0, 1)
            + 0.18 * by["buy_dom"]
            + 0.18 * np.clip(by["up_progress"] / np.maximum(spread.fillna(1.0), 1.0), 0, 1)
            + 0.14 * np.clip(by["trade_count"] / count_scale, 0, 1)
            + 0.12 * by["close_strength"]
            + 0.10 * np.clip(by["unique_prices"] / 5.0, 0, 1),
            0, 1,
        )
        by["gulungan_down_score"] = 100 * np.clip(
            0.28 * np.clip(by["sell_lot"] / lot_scale, 0, 1)
            + 0.18 * by["sell_dom"]
            + 0.18 * np.clip(by["down_progress"] / np.maximum(spread.fillna(1.0), 1.0), 0, 1)
            + 0.14 * np.clip(by["trade_count"] / count_scale, 0, 1)
            + 0.12 * by["close_weakness"]
            + 0.10 * np.clip(by["unique_prices"] / 5.0, 0, 1),
            0, 1,
        )
        by["effort_result_up"] = np.where(by["buy_lot"] > 0, 100 * by["up_progress"] / np.log1p(by["buy_lot"]), 0.0)
        by["effort_result_down"] = np.where(by["sell_lot"] > 0, 100 * by["down_progress"] / np.log1p(by["sell_lot"]), 0.0)
        by["next_close"] = by["price_close"].shift(-1)
        by["next3_max"] = by["price_high"].shift(-1).rolling(3, min_periods=1).max()
        by["next3_min"] = by["price_low"].shift(-1).rolling(3, min_periods=1).min()
        by["post_up_followthrough_score"] = 100 * np.clip(
            0.45 * ((by["next_close"] - by["price_close"]).clip(lower=0) / np.maximum(spread.fillna(1.0), 1.0)).fillna(0)
            + 0.35 * ((by["next3_max"] - by["price_close"]).clip(lower=0) / np.maximum(spread.fillna(1.0), 1.0)).fillna(0)
            + 0.20 * (1 - ((by["price_close"] - by["next3_min"]).clip(lower=0) / np.maximum(spread.fillna(1.0), 1.0)).clip(0, 1)).fillna(0),
            0, 1,
        )
        by["post_down_followthrough_score"] = 100 * np.clip(
            0.45 * ((by["price_close"] - by["next_close"]).clip(lower=0) / np.maximum(spread.fillna(1.0), 1.0)).fillna(0)
            + 0.35 * ((by["price_close"] - by["next3_min"]).clip(lower=0) / np.maximum(spread.fillna(1.0), 1.0)).fillna(0)
            + 0.20 * (1 - ((by["next3_max"] - by["price_close"]).clip(lower=0) / np.maximum(spread.fillna(1.0), 1.0)).clip(0, 1)).fillna(0),
            0, 1,
        )
        med_up = by["effort_result_up"].replace(0, np.nan).median()
        med_down = by["effort_result_down"].replace(0, np.nan).median()
        by["bull_trap_score"] = 100 * np.clip(
            0.35 * np.clip(by["gulungan_up_score"] / 100.0, 0, 1)
            + 0.20 * (1 - np.clip(by["effort_result_up"] / max(float(med_up) if pd.notna(med_up) and med_up > 0 else 1.0, 1.0), 0, 1))
            + 0.20 * (1 - np.clip(by["post_up_followthrough_score"] / 100.0, 0, 1))
            + 0.15 * by["close_weakness"]
            + 0.10 * np.clip(by["split_order_score"] / 100.0, 0, 1),
            0, 1,
        )
        by["bear_trap_score"] = 100 * np.clip(
            0.35 * np.clip(by["gulungan_down_score"] / 100.0, 0, 1)
            + 0.20 * (1 - np.clip(by["effort_result_down"] / max(float(med_down) if pd.notna(med_down) and med_down > 0 else 1.0, 1.0), 0, 1))
            + 0.20 * (1 - np.clip(by["post_down_followthrough_score"] / 100.0, 0, 1))
            + 0.15 * by["close_strength"]
            + 0.10 * np.clip(by["split_order_score"] / 100.0, 0, 1),
            0, 1,
        )
        last = by.iloc[-1]
        label = "NO_INTRADAY_SIGNAL"
        bias = "NEUTRAL"
        if last["gulungan_up_score"] >= 60 and last["gulungan_up_score"] > last["gulungan_down_score"] * 1.12:
            if last["bull_trap_score"] >= 60:
                label = "UP_FALSE_BREAKOUT_RISK"
                bias = "BEARISH"
            elif last["post_up_followthrough_score"] >= 50 and last["effort_result_up"] >= (float(med_up) if pd.notna(med_up) else 0.0):
                label = "UP_CONTINUATION_BURST"
                bias = "BULLISH"
            else:
                label = "UP_INITIATIVE_SWEEP"
                bias = "BULLISH"
        elif last["gulungan_down_score"] >= 60 and last["gulungan_down_score"] > last["gulungan_up_score"] * 1.12:
            if last["bear_trap_score"] >= 60:
                label = "DOWN_CAPITULATION_RISK"
                bias = "BULLISH"
            elif last["post_down_followthrough_score"] >= 50 and last["effort_result_down"] >= (float(med_down) if pd.notna(med_down) else 0.0):
                label = "DOWN_CONTINUATION_BREAK"
                bias = "BEARISH"
            else:
                label = "DOWN_INITIATIVE_SWEEP"
                bias = "BEARISH"
        rows.append({
            "ticker": ticker,
            "gulungan_up_score": round(float(last["gulungan_up_score"]), 1),
            "gulungan_down_score": round(float(last["gulungan_down_score"]), 1),
            "effort_result_up": round(float(last["effort_result_up"]), 2),
            "effort_result_down": round(float(last["effort_result_down"]), 2),
            "post_up_followthrough_score": round(float(last["post_up_followthrough_score"]), 1),
            "post_down_followthrough_score": round(float(last["post_down_followthrough_score"]), 1),
            "split_order_score": round(float(last["split_order_score"]), 1),
            "bull_trap_score": round(float(last["bull_trap_score"]), 1),
            "bear_trap_score": round(float(last["bear_trap_score"]), 1),
            "latest_event_label": label,
            "burst_bias": bias,
            "last_burst_ts": last["bucket"],
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
