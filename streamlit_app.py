from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests

st.set_page_config(page_title="IDX EOD Scanner V3", layout="wide")


# =============================
# Context containers
# =============================


@dataclass
class BrokerContext:
    ticker: str
    dominant_accumulators: str = "-"
    dominant_distributors: str = "-"
    institutional_support: Optional[float] = None
    institutional_resistance: Optional[float] = None
    broker_alignment_score: float = 50.0
    broker_mode: str = "NO_BROKER_DATA"


@dataclass
class BurstContext:
    ticker: str
    latest_event_label: str = "NO_INTRADAY_DATA"
    burst_bias: str = "NEUTRAL"
    gulungan_up_score: float = 0.0
    gulungan_down_score: float = 0.0
    bullish_burst_score: float = 0.0
    bearish_burst_score: float = 0.0
    bull_trap_score: float = 0.0
    bear_trap_score: float = 0.0
    absorption_after_up_score: float = 0.0
    absorption_after_down_score: float = 0.0
    latest_event_time: Optional[pd.Timestamp] = None


# =============================
# Data loading helpers
# =============================


def _clean_symbol_list(values) -> list[str]:
    out = []
    for v in values:
        s = str(v).upper().strip().replace(".JK", "")
        if s and s != "NAN" and re.fullmatch(r"[A-Z0-9]{4,5}", s):
            out.append(s)
    return sorted(set(out))


def safe_read_universe(default_path: str = "data/idx_universe_sample.csv") -> list[str]:
    fallback = ["BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "ANTM", "PANI"]
    try:
        df = pd.read_csv(default_path)
        tickers = _clean_symbol_list(df.iloc[:, 0].tolist())
        return tickers or fallback
    except Exception:
        return fallback


def _extract_symbols_from_table(df: pd.DataFrame) -> list[str]:
    colmap = {str(c).strip().lower(): c for c in df.columns}
    for key in ["ticker", "symbol", "kode saham", "code", "stock code"]:
        if key in colmap:
            vals = _clean_symbol_list(df[colmap[key]].tolist())
            if len(vals) >= 50:
                return vals
    best: list[str] = []
    for c in df.columns:
        vals = _clean_symbol_list(df[c].tolist())
        if len(vals) > len(best):
            best = vals
    return best


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_full_ihsg_universe() -> tuple[list[str], str]:
    sample_vals: list[str] = []
    for local_path in ["data/idx_universe_full.csv", "data/idx_universe_sample.csv"]:
        try:
            df = pd.read_csv(local_path)
            vals = _clean_symbol_list(df.iloc[:, 0].tolist())
            if local_path.endswith("full.csv") and len(vals) >= 400:
                return vals, f"local:{local_path}"
            if local_path.endswith("sample.csv") and len(vals) >= 8:
                sample_vals = vals
        except Exception:
            pass

    headers = {"User-Agent": "Mozilla/5.0"}

    official_urls = [
        "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/",
        "https://www.idx.co.id/en/market-data/stocks-data/stock-list",
    ]
    for url in official_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.ok and resp.text:
                html = resp.text
                regex_vals = _clean_symbol_list(re.findall(r">\s*([A-Z0-9]{4,5})\s*<", html))
                if len(regex_vals) >= 400:
                    return regex_vals, f"official_idx_regex:{url}"
                for tbl in pd.read_html(io.StringIO(html)):
                    vals = _extract_symbols_from_table(tbl)
                    if len(vals) >= 400:
                        return vals, f"official_idx_table:{url}"
        except Exception:
            pass

    wiki_url = "https://en.wikipedia.org/wiki/IDX_Composite"
    try:
        resp = requests.get(wiki_url, headers=headers, timeout=20)
        if resp.ok and resp.text:
            html = resp.text
            segment = html
            m1 = re.search(r"Components.*?currently lists.*?(?:As for Sharia Index|ISSI)", html, re.I | re.S)
            if m1:
                segment = m1.group(0)
            vals = _clean_symbol_list(re.findall(r">\s*([A-Z0-9]{4,5})\s*<", segment))
            if len(vals) >= 400:
                return vals, f"wiki_html_regex:{wiki_url}"
            for tbl in pd.read_html(io.StringIO(html)):
                vals = _extract_symbols_from_table(tbl)
                if len(vals) >= 400:
                    return vals, f"wiki_html_table:{wiki_url}"
    except Exception:
        pass

    api_urls = [
        "https://en.wikipedia.org/w/index.php?title=IDX_Composite&action=raw",
        "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=IDX_Composite&rvslots=main&rvprop=content&format=json",
    ]
    for url in api_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if not resp.ok or not resp.text:
                continue
            txt = resp.text
            vals = _clean_symbol_list(re.findall(r"\|\s*\d+\s*\|\|\s*([A-Z0-9]{4,5})\s*\|\|", txt))
            if len(vals) >= 400:
                return vals, f"wiki_raw:{url}"
            vals = _clean_symbol_list(re.findall(r"\\n\|\s*\d+\s*\|\|\s*([A-Z0-9]{4,5})\s*\|\|", txt))
            if len(vals) >= 400:
                return vals, f"wiki_api_regex:{url}"
        except Exception:
            pass

    vals = sample_vals or safe_read_universe()
    return vals, "fallback_sample"

def _download_chunk(symbols: list[str], period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for sym in symbols:
            if sym not in available:
                continue
            df = raw[sym].copy().reset_index()
            if df.empty:
                continue
            df["ticker"] = sym.replace(".JK", "")
            frames.append(df)
    else:
        df = raw.reset_index().copy()
        df["ticker"] = symbols[0].replace(".JK", "")
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_values(["ticker", "date"])
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers: tuple[str, ...], period: str = "18mo", interval: str = "1d", chunk_size: int = 80) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    symbols = [f"{t}.JK" for t in tickers]
    frames: list[pd.DataFrame] = []
    for i in range(0, len(symbols), max(int(chunk_size), 1)):
        chunk = symbols[i:i + max(int(chunk_size), 1)]
        try:
            part = _download_chunk(chunk, period=period, interval=interval)
            if not part.empty:
                frames.append(part)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "date"], keep="last")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_broker_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "ticker", "broker_code", "buy_lot", "sell_lot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Broker CSV missing columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
    df["broker_code"] = df["broker_code"].astype(str).str.upper()
    for c in ["buy_lot", "sell_lot", "buy_value", "sell_value"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "ticker", "broker_code"])
    df["net_lot"] = df["buy_lot"] - df["sell_lot"]
    df["gross_lot"] = df["buy_lot"] + df["sell_lot"]
    df["avg_buy_price"] = np.where(df["buy_lot"] > 0, df["buy_value"] / df["buy_lot"], np.nan)
    df["avg_sell_price"] = np.where(df["sell_lot"] > 0, df["sell_value"] / df["sell_lot"], np.nan)
    return df


@st.cache_data(show_spinner=False)
def parse_done_detail_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"timestamp", "ticker", "price", "lot", "side"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Done detail CSV missing columns: {', '.join(sorted(missing))}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
    df["side"] = df["side"].astype(str).str.upper().str[0]
    df = df[df["side"].isin(["B", "S"])]
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["lot"] = pd.to_numeric(df["lot"], errors="coerce")
    for c in ["buyer_broker", "seller_broker"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    df = df.dropna(subset=["timestamp", "ticker", "price", "lot"]).sort_values(["ticker", "timestamp"])
    df["value"] = df["price"] * df["lot"]
    df["minute"] = df["timestamp"].dt.floor("min")
    return df


@st.cache_data(show_spinner=False)
def parse_orderbook_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"timestamp", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Orderbook CSV missing columns: {', '.join(sorted(missing))}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
    numeric_cols = [
        "bid_1_price", "bid_1_lot", "bid_2_price", "bid_2_lot", "bid_3_price", "bid_3_lot",
        "offer_1_price", "offer_1_lot", "offer_2_price", "offer_2_lot", "offer_3_price", "offer_3_lot",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    df = df.dropna(subset=["timestamp", "ticker"]).sort_values(["ticker", "timestamp"])
    df["minute"] = df["timestamp"].dt.floor("min")
    return df




def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_round(series: pd.Series, digits: int = 2) -> pd.Series:
    return _safe_numeric(series).round(digits)


# =============================
# EOD analytics
# =============================


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    df = prices.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", group_keys=False)

    for n in [20, 50, 200]:
        df[f"ema_{n}"] = g["close"].transform(lambda s: s.ewm(span=n, adjust=False).mean())

    df["ret_20"] = g["close"].pct_change(20)
    df["ret_5"] = g["close"].pct_change(5)
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = g.apply(lambda x: tr.loc[x.index].rolling(14).mean()).reset_index(level=0, drop=True)
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    df["vol_ma60"] = g["volume"].transform(lambda s: s.rolling(60).mean())
    df["hh_20"] = g["high"].transform(lambda s: s.rolling(20).max())
    df["hh_60"] = g["high"].transform(lambda s: s.rolling(60).max())
    df["ll_20"] = g["low"].transform(lambda s: s.rolling(20).min())
    df["ll_60"] = g["low"].transform(lambda s: s.rolling(60).min())
    df["range_20"] = (df["hh_20"] - df["ll_20"]) / df["close"].replace(0, np.nan)
    df["close_loc_20"] = (df["close"] - df["ll_20"]) / (df["hh_20"] - df["ll_20"]).replace(0, np.nan)
    df["close_vs_hh60"] = df["close"] / df["hh_60"].replace(0, np.nan) - 1
    df["close_vs_ll20"] = df["close"] / df["ll_20"].replace(0, np.nan) - 1
    df["base_maturity"] = g["range_20"].transform(lambda s: (s.rolling(15).mean().rsub(0.22).clip(lower=0) / 0.22).clip(0, 1))
    df["dry_score"] = ((1 - (df["vol_ma20"] / df["vol_ma60"].replace(0, np.nan)).clip(0, 2) / 2).fillna(0.5) * 100)
    df["wet_score"] = 100 - df["dry_score"]

    ema_stack = (
        (df["close"] > df["ema_20"]).astype(int)
        + (df["ema_20"] > df["ema_50"]).astype(int)
        + (df["ema_50"] > df["ema_200"]).astype(int)
    )
    df["trend_quality"] = (ema_stack / 3.0) * 100
    breakout_flag = (df["close"] > g["high"].shift(1).transform(lambda s: s.rolling(20).max())).astype(int)
    df["breakout_integrity"] = (
        30 * breakout_flag
        + 20 * (df["volume"] / df["vol_ma20"].replace(0, np.nan)).clip(0, 3).fillna(0)
        + 20 * df["base_maturity"].fillna(0)
        + 30 * df["close_loc_20"].clip(0, 1).fillna(0)
    ).clip(0, 100)
    df["false_breakout_risk"] = (
        40 * (df["close_vs_hh60"] > -0.01).astype(int)
        + 30 * (df["ret_5"] < -0.03).astype(int)
        + 30 * (df["close"] < df["ema_20"]).astype(int)
    ).clip(0, 100)
    df["liquidity_mn"] = (df["close"] * df["volume"]).rolling(20).mean() / 1_000_000_000
    return df


def classify_phase(row: pd.Series) -> str:
    if pd.isna(row.get("ema_20")):
        return "INSUFFICIENT_DATA"
    if row["trend_quality"] >= 85 and row["ret_20"] > 0.05:
        return "MARKUP"
    if row["trend_quality"] >= 66 and row["base_maturity"] > 0.5:
        return "EARLY_MARKUP"
    if row["ret_20"] < -0.08 and row["close"] < row["ema_50"]:
        return "MARKDOWN"
    if row["base_maturity"] > 0.65 and row["close_loc_20"] < 0.7:
        return "ACCUMULATION"
    if row["ret_5"] < -0.03 and row["close"] > row["ema_50"]:
        return "PULLBACK_HEALTHY"
    return "NEUTRAL"


# =============================
# Broker analytics
# =============================


def summarize_broker_context(broker_df: Optional[pd.DataFrame], ticker: str) -> BrokerContext:
    if broker_df is None or broker_df.empty:
        return BrokerContext(ticker=ticker)
    tdf = broker_df[broker_df["ticker"] == ticker].copy()
    if tdf.empty:
        return BrokerContext(ticker=ticker)
    latest_date = tdf["date"].max()
    tdf = tdf[tdf["date"] == latest_date].sort_values("gross_lot", ascending=False)
    acc = tdf[tdf["net_lot"] > 0].sort_values("net_lot", ascending=False)
    dist = tdf[tdf["net_lot"] < 0].sort_values("net_lot")
    top_acc = acc.head(3)
    top_dist = dist.head(3)

    dominant_accumulators = ", ".join([f"{r.broker_code}({int(r.net_lot):,})" for r in top_acc.itertuples()]) if not top_acc.empty else "-"
    dominant_distributors = ", ".join([f"{r.broker_code}({int(r.net_lot):,})" for r in top_dist.itertuples()]) if not top_dist.empty else "-"

    inst_support = None
    if not top_acc.empty and top_acc["buy_value"].sum() > 0:
        w = top_acc["buy_lot"].replace(0, np.nan)
        px = top_acc["buy_value"] / w
        valid = px.notna()
        if valid.any():
            inst_support = float(np.average(px[valid], weights=top_acc.loc[valid, "buy_lot"]))

    inst_res = None
    if not top_dist.empty and top_dist["sell_value"].sum() > 0:
        w = top_dist["sell_lot"].replace(0, np.nan)
        px = top_dist["sell_value"] / w
        valid = px.notna()
        if valid.any():
            inst_res = float(np.average(px[valid], weights=top_dist.loc[valid, "sell_lot"]))

    total_net = float(tdf["net_lot"].sum())
    gross = float(tdf["gross_lot"].sum())
    alignment = 50.0
    mode = "BALANCED"
    if gross > 0:
        ratio = total_net / gross
        alignment = float(np.clip(50 + ratio * 120, 0, 100))
        if ratio > 0.1:
            mode = "ACCUMULATION_DOMINANT"
        elif ratio < -0.1:
            mode = "DISTRIBUTION_DOMINANT"

    return BrokerContext(
        ticker=ticker,
        dominant_accumulators=dominant_accumulators,
        dominant_distributors=dominant_distributors,
        institutional_support=inst_support,
        institutional_resistance=inst_res,
        broker_alignment_score=alignment,
        broker_mode=mode,
    )


# =============================
# Intraday burst engine
# =============================


def _clip01(x: pd.Series | np.ndarray | float) -> pd.Series | float:
    return np.clip(x, 0, 1)


def compute_burst_features(done_df: Optional[pd.DataFrame], orderbook_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if done_df is None or done_df.empty:
        return pd.DataFrame()

    d = done_df.copy()
    grp = d.groupby(["ticker", "minute", "side"], observed=True)["lot"].sum().unstack(fill_value=0).reset_index()
    grp.columns.name = None
    if "B" not in grp.columns:
        grp["B"] = 0.0
    if "S" not in grp.columns:
        grp["S"] = 0.0

    p = d.groupby(["ticker", "minute"], observed=True).agg(
        open_price=("price", "first"),
        high_price=("price", "max"),
        low_price=("price", "min"),
        close_price=("price", "last"),
        trade_count=("price", "size"),
        traded_value=("value", "sum"),
    ).reset_index()

    price_levels = d.groupby(["ticker", "minute"], observed=True).agg(
        unique_price_levels=("price", pd.Series.nunique)
    ).reset_index()

    m = grp.merge(p, on=["ticker", "minute"], how="left").merge(price_levels, on=["ticker", "minute"], how="left")
    m["total_aggr_lot"] = m["B"] + m["S"]
    m["buy_share"] = m["B"] / m["total_aggr_lot"].replace(0, np.nan)
    m["sell_share"] = m["S"] / m["total_aggr_lot"].replace(0, np.nan)
    m["price_change"] = m["close_price"] - m["open_price"]
    m["range"] = (m["high_price"] - m["low_price"]).clip(lower=0)
    m["upper_wick_ratio"] = ((m["high_price"] - m[["open_price", "close_price"]].max(axis=1)) / m["range"].replace(0, np.nan)).fillna(0)
    m["lower_wick_ratio"] = (((m[["open_price", "close_price"]].min(axis=1) - m["low_price"])) / m["range"].replace(0, np.nan)).fillna(0)
    m["close_strength"] = ((m["close_price"] - m["low_price"]) / m["range"].replace(0, np.nan)).fillna(0.5)
    m["close_weakness"] = ((m["high_price"] - m["close_price"]) / m["range"].replace(0, np.nan)).fillna(0.5)

    m = m.sort_values(["ticker", "minute"]).reset_index(drop=True)
    g = m.groupby("ticker", group_keys=False)
    m["next_close"] = g["close_price"].shift(-1)
    m["close_plus_3"] = g["close_price"].shift(-3)
    m["next_min_low"] = g["low_price"].shift(-1)
    m["next_min_high"] = g["high_price"].shift(-1)

    # Orderbook merge if available
    if orderbook_df is not None and not orderbook_df.empty:
        ob = orderbook_df.copy()
        ob_agg = ob.groupby(["ticker", "minute"], observed=True).agg(
            bid_lot_top3=("bid_1_lot", "mean"),
            bid2=("bid_2_lot", "mean"),
            bid3=("bid_3_lot", "mean"),
            offer_lot_top3=("offer_1_lot", "mean"),
            off2=("offer_2_lot", "mean"),
            off3=("offer_3_lot", "mean"),
        ).reset_index()
        ob_agg["bid_lot_top3"] = ob_agg[["bid_lot_top3", "bid2", "bid3"]].sum(axis=1, skipna=True)
        ob_agg["offer_lot_top3"] = ob_agg[["offer_lot_top3", "off2", "off3"]].sum(axis=1, skipna=True)
        ob_agg = ob_agg[["ticker", "minute", "bid_lot_top3", "offer_lot_top3"]]
        m = m.merge(ob_agg, on=["ticker", "minute"], how="left")
    else:
        m["bid_lot_top3"] = np.nan
        m["offer_lot_top3"] = np.nan

    # Relative intensities per ticker
    for col in ["B", "S", "trade_count", "unique_price_levels", "traded_value", "range"]:
        med = g[col].transform(lambda s: s.rolling(30, min_periods=5).median())
        m[f"{col}_rel"] = (m[col] / med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    m["upward_price_levels_lifted"] = _clip01((m["price_change"] > 0).astype(float) * (m["unique_price_levels_rel"] / 3.0))
    m["downward_price_levels_broken"] = _clip01((m["price_change"] < 0).astype(float) * (m["unique_price_levels_rel"] / 3.0))
    m["price_displacement_up"] = _clip01((m["price_change"].clip(lower=0) / m["range"].replace(0, np.nan)).fillna(0))
    m["price_displacement_down"] = _clip01((-m["price_change"].clip(upper=0) / m["range"].replace(0, np.nan)).fillna(0))

    m["gulungan_up_score"] = (
        0.30 * _clip01(m["B_rel"] / 3)
        + 0.20 * _clip01(((m["B"] - m["S"]) / m["total_aggr_lot"].replace(0, np.nan)).fillna(0.0) + 0.5)
        + 0.20 * m["upward_price_levels_lifted"]
        + 0.15 * _clip01(m["trade_count_rel"] / 3)
        + 0.15 * m["price_displacement_up"]
    ) * 100
    m["gulungan_down_score"] = (
        0.30 * _clip01(m["S_rel"] / 3)
        + 0.20 * _clip01(((m["S"] - m["B"]) / m["total_aggr_lot"].replace(0, np.nan)).fillna(0.0) + 0.5)
        + 0.20 * m["downward_price_levels_broken"]
        + 0.15 * _clip01(m["trade_count_rel"] / 3)
        + 0.15 * m["price_displacement_down"]
    ) * 100

    m["effort_result_up"] = _clip01((m["price_displacement_up"] / np.log1p(m["B_rel"])) / 1.5).fillna(0) * 100
    m["effort_result_down"] = _clip01((m["price_displacement_down"] / np.log1p(m["S_rel"])) / 1.5).fillna(0) * 100

    m["post_up_followthrough_score"] = (
        0.50 * _clip01(((m["next_close"] - m["close_price"]).clip(lower=0) / m["range"].replace(0, np.nan)).fillna(0))
        + 0.50 * _clip01(((m["close_plus_3"] - m["close_price"]).clip(lower=0) / (m["range"].replace(0, np.nan) * 2)).fillna(0))
    ) * 100
    m["post_down_followthrough_score"] = (
        0.50 * _clip01(((m["close_price"] - m["next_close"]).clip(lower=0) / m["range"].replace(0, np.nan)).fillna(0))
        + 0.50 * _clip01(((m["close_price"] - m["close_plus_3"]).clip(lower=0) / (m["range"].replace(0, np.nan) * 2)).fillna(0))
    ) * 100

    offer_refill = (m["offer_lot_top3"] / (m["B"] + 1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    bid_refill = (m["bid_lot_top3"] / (m["S"] + 1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    m["absorption_after_up_score"] = (
        0.30 * _clip01(offer_refill / 3)
        + 0.25 * m["upper_wick_ratio"].clip(0, 1)
        + 0.25 * (1 - _clip01(m["effort_result_up"] / 100))
        + 0.20 * (1 - _clip01(m["post_up_followthrough_score"] / 100))
    ) * 100
    m["absorption_after_down_score"] = (
        0.30 * _clip01(bid_refill / 3)
        + 0.25 * m["lower_wick_ratio"].clip(0, 1)
        + 0.25 * (1 - _clip01(m["effort_result_down"] / 100))
        + 0.20 * (1 - _clip01(m["post_down_followthrough_score"] / 100))
    ) * 100

    m["bullish_burst_score"] = (
        0.28 * m["gulungan_up_score"]
        + 0.20 * m["effort_result_up"]
        + 0.20 * m["post_up_followthrough_score"]
        + 0.12 * (100 - m["absorption_after_up_score"])
        + 0.10 * (100 * m["close_strength"])
        + 0.10 * _clip01(m["B_rel"] / 3) * 100
    ).clip(0, 100)
    m["bearish_burst_score"] = (
        0.28 * m["gulungan_down_score"]
        + 0.20 * m["effort_result_down"]
        + 0.20 * m["post_down_followthrough_score"]
        + 0.12 * (100 - m["absorption_after_down_score"])
        + 0.10 * (100 * m["close_weakness"])
        + 0.10 * _clip01(m["S_rel"] / 3) * 100
    ).clip(0, 100)
    m["bull_trap_score"] = (
        0.30 * m["gulungan_up_score"]
        + 0.25 * m["absorption_after_up_score"]
        + 0.20 * (100 - m["effort_result_up"])
        + 0.15 * (100 - m["post_up_followthrough_score"])
        + 0.10 * (100 * m["upper_wick_ratio"].clip(0, 1))
    ).clip(0, 100)
    m["bear_trap_score"] = (
        0.30 * m["gulungan_down_score"]
        + 0.25 * m["absorption_after_down_score"]
        + 0.20 * (100 - m["effort_result_down"])
        + 0.15 * (100 - m["post_down_followthrough_score"])
        + 0.10 * (100 * m["lower_wick_ratio"].clip(0, 1))
    ).clip(0, 100)

    def event_label(r: pd.Series) -> str:
        if r["bullish_burst_score"] >= 72 and r["bull_trap_score"] < 58:
            return "UP_CONTINUATION_BURST"
        if r["gulungan_up_score"] >= 68 and r["bull_trap_score"] >= 62:
            return "UP_FALSE_BREAKOUT_RISK"
        if r["bearish_burst_score"] >= 72 and r["bear_trap_score"] < 58:
            return "DOWN_CONTINUATION_BREAK"
        if r["gulungan_down_score"] >= 68 and r["bear_trap_score"] >= 62:
            return "DOWN_CAPITULATION_RISK"
        if r["gulungan_up_score"] >= 60:
            return "UP_INITIATIVE_SWEEP"
        if r["gulungan_down_score"] >= 60:
            return "DOWN_INITIATIVE_SWEEP"
        return "NO_MAJOR_EVENT"

    m["event_label"] = m.apply(event_label, axis=1)
    return m


def summarize_burst_context(burst_df: Optional[pd.DataFrame], ticker: str) -> BurstContext:
    if burst_df is None or burst_df.empty:
        return BurstContext(ticker=ticker)
    tdf = burst_df[burst_df["ticker"] == ticker].copy()
    if tdf.empty:
        return BurstContext(ticker=ticker)
    latest = tdf.sort_values("minute").iloc[-1]
    burst_bias = "NEUTRAL"
    if latest["bullish_burst_score"] >= max(65, latest["bearish_burst_score"] + 8):
        burst_bias = "UP_BIAS"
    elif latest["bearish_burst_score"] >= max(65, latest["bullish_burst_score"] + 8):
        burst_bias = "DOWN_BIAS"
    elif latest["bull_trap_score"] >= 65:
        burst_bias = "UP_TRAP_RISK"
    elif latest["bear_trap_score"] >= 65:
        burst_bias = "DOWN_TRAP_RISK"

    return BurstContext(
        ticker=ticker,
        latest_event_label=str(latest["event_label"]),
        burst_bias=burst_bias,
        gulungan_up_score=float(latest["gulungan_up_score"]),
        gulungan_down_score=float(latest["gulungan_down_score"]),
        bullish_burst_score=float(latest["bullish_burst_score"]),
        bearish_burst_score=float(latest["bearish_burst_score"]),
        bull_trap_score=float(latest["bull_trap_score"]),
        bear_trap_score=float(latest["bear_trap_score"]),
        absorption_after_up_score=float(latest["absorption_after_up_score"]),
        absorption_after_down_score=float(latest["absorption_after_down_score"]),
        latest_event_time=latest["minute"],
    )


# =============================
# Final scan
# =============================


def build_latest_scan(df: pd.DataFrame, broker_df: Optional[pd.DataFrame], burst_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    latest = df.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    latest["phase"] = latest.apply(classify_phase, axis=1)
    broker_contexts = [summarize_broker_context(broker_df, t) for t in latest["ticker"]]
    burst_contexts = [summarize_burst_context(burst_df, t) for t in latest["ticker"]]
    latest = latest.merge(pd.DataFrame([c.__dict__ for c in broker_contexts]), on="ticker", how="left")
    latest = latest.merge(pd.DataFrame([c.__dict__ for c in burst_contexts]), on="ticker", how="left")

    numeric_merge_cols = [
        "institutional_support", "institutional_resistance", "broker_alignment_score",
        "gulungan_up_score", "gulungan_down_score", "bullish_burst_score", "bearish_burst_score",
        "bull_trap_score", "bear_trap_score", "absorption_after_up_score", "absorption_after_down_score",
    ]
    for c in numeric_merge_cols:
        if c in latest.columns:
            latest[c] = _safe_numeric(latest[c])

    if "institutional_support" in latest.columns:
        latest["institutional_support"] = _safe_round(latest["institutional_support"], 2)
    if "institutional_resistance" in latest.columns:
        latest["institutional_resistance"] = _safe_round(latest["institutional_resistance"], 2)

    latest["long_score"] = (
        0.22 * latest["trend_quality"].fillna(0)
        + 0.16 * latest["breakout_integrity"].fillna(0)
        + 0.12 * latest["base_maturity"].fillna(0) * 100
        + 0.08 * latest["dry_score"].fillna(50)
        + 0.12 * latest["broker_alignment_score"].fillna(50)
        + 0.15 * (100 - latest["false_breakout_risk"].fillna(0))
        + 0.15 * latest["bullish_burst_score"].fillna(0)
    )
    latest["sell_score"] = (
        0.24 * latest["false_breakout_risk"].fillna(0)
        + 0.14 * latest["wet_score"].fillna(50)
        + 0.16 * (100 - latest["trend_quality"].fillna(0))
        + 0.12 * (100 - latest["broker_alignment_score"].fillna(50))
        + 0.16 * latest["bearish_burst_score"].fillna(0)
        + 0.10 * latest["bull_trap_score"].fillna(0)
        + 0.08 * (latest["phase"].isin(["MARKDOWN"]).astype(int) * 100)
    )
    latest["rebound_watch_score"] = (
        0.35 * latest["bear_trap_score"].fillna(0)
        + 0.20 * latest["dry_score"].fillna(0)
        + 0.15 * (latest["phase"].isin(["MARKDOWN", "ACCUMULATION"]).astype(int) * 100)
        + 0.15 * (100 - latest["sell_score"].fillna(0))
        + 0.15 * latest["broker_alignment_score"].fillna(50)
    )

    def verdict(row: pd.Series) -> str:
        if pd.isna(row.get("liquidity_mn")) or row["liquidity_mn"] < 5:
            return "ILLIQUID"
        if row["sell_score"] >= 72:
            return "TRIM"
        if row["long_score"] >= 75 and row["phase"] in ["MARKUP", "EARLY_MARKUP", "ACCUMULATION"]:
            return "READY_LONG"
        if row["rebound_watch_score"] >= 66 and row["bear_trap_score"] >= 64:
            return "WATCH_REBOUND"
        if row["long_score"] >= 63:
            return "WATCH"
        if row["sell_score"] >= 60:
            return "AVOID"
        return "NEUTRAL"

    latest["verdict"] = latest.apply(verdict, axis=1)
    latest["support_20d"] = latest["ll_20"].round(2)
    latest["resistance_60d"] = latest["hh_60"].round(2)
    round_cols = [
        "close", "trend_quality", "breakout_integrity", "false_breakout_risk", "dry_score", "wet_score",
        "liquidity_mn", "broker_alignment_score", "long_score", "sell_score", "rebound_watch_score",
        "gulungan_up_score", "gulungan_down_score", "bullish_burst_score", "bearish_burst_score",
        "bull_trap_score", "bear_trap_score"
    ]
    for c in round_cols:
        if c in latest.columns:
            latest[c] = _safe_round(latest[c], 1)
    latest = latest.sort_values(["verdict", "long_score", "bullish_burst_score"], ascending=[True, False, False])
    return latest


# =============================
# Visualization helpers
# =============================


def make_chart(df: pd.DataFrame, ticker: str, broker_ctx: BrokerContext) -> go.Figure:
    tdf = df[df["ticker"] == ticker].copy().tail(220)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tdf["date"], open=tdf["open"], high=tdf["high"], low=tdf["low"], close=tdf["close"], name="Price"))
    for n in [20, 50, 200]:
        col = f"ema_{n}"
        if col in tdf.columns:
            fig.add_trace(go.Scatter(x=tdf["date"], y=tdf[col], mode="lines", name=f"EMA {n}"))
    if broker_ctx.institutional_support is not None:
        fig.add_hline(y=float(broker_ctx.institutional_support), line_dash="dash", annotation_text="Inst Support")
    if broker_ctx.institutional_resistance is not None:
        fig.add_hline(y=float(broker_ctx.institutional_resistance), line_dash="dot", annotation_text="Inst Resistance")
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_intraday_bars(burst_df: pd.DataFrame, ticker: str) -> go.Figure:
    tdf = burst_df[burst_df["ticker"] == ticker].copy().tail(120)
    fig = go.Figure()
    if tdf.empty:
        fig.update_layout(height=380)
        return fig
    fig.add_trace(go.Bar(x=tdf["minute"], y=tdf["gulungan_up_score"], name="Gulung Up"))
    fig.add_trace(go.Bar(x=tdf["minute"], y=-tdf["gulungan_down_score"], name="Gulung Down"))
    fig.add_trace(go.Scatter(x=tdf["minute"], y=tdf["bullish_burst_score"], mode="lines", name="Bullish Burst"))
    fig.add_trace(go.Scatter(x=tdf["minute"], y=-tdf["bearish_burst_score"], mode="lines", name="Bearish Burst"))
    fig.update_layout(height=420, barmode="relative", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def setup_text(row: pd.Series) -> str:
    verdict = row["verdict"]
    if verdict == "READY_LONG":
        return "Struktur naik sehat, breakout integrity oke, dan intraday burst masih confirm. Fokus jaga support dan follow-through."
    if verdict == "WATCH_REBOUND":
        return "Ada tanda flush/capitulation atau bear trap. Belum auto-long, tapi layak dipantau untuk rebound yang rapi."
    if verdict == "WATCH":
        return "Menarik, tapi belum bersih. Tunggu follow-through, pullback sehat, atau burst baru yang lebih kuat."
    if verdict == "TRIM":
        return "Risk reward mulai jelek. Trap/distribution risk naik, lebih aman rapikan exposure."
    if verdict == "AVOID":
        return "Belum menarik. Struktur dan microstructure belum mendukung."
    if verdict == "ILLIQUID":
        return "Likuiditas terlalu kecil untuk scanner ini."
    return "Belum ada edge yang cukup kuat."


# =============================
# Streamlit UI
# =============================

st.title("IDX EOD Scanner V3.3 — Full IHSG Scanner")
st.caption("Single-file IDX scanner: yfinance `.JK` untuk EOD real, optional broker summary + done detail + orderbook import untuk burst/gulungan up-down dan trap vs continuation. Sekarang support full universe IHSG via free auto-fetch + batching.")

with st.sidebar:
    st.header("Settings")
    universe_choice = st.radio("Universe", ["Full IHSG (free auto-fetch)", "Sample IDX universe", "Manual input"], index=0)
    sample_tickers = safe_read_universe()
    full_tickers, full_universe_source = fetch_full_ihsg_universe()
    default_manual = ", ".join(sample_tickers[:12])
    manual_text = st.text_area("Manual tickers (comma separated)", value=default_manual, height=150, disabled=universe_choice != "Manual input")
    if universe_choice == "Full IHSG (free auto-fetch)":
        st.caption(f"Universe source: {full_universe_source} | symbols: {len(full_tickers)}")
        if full_universe_source == "fallback_sample":
            st.warning("Full-universe source gagal kebaca, jadi app jatuh ke sample universe. Upload data/idx_universe_full.csv atau coba rerun.")
    period = st.selectbox("History", ["6mo", "12mo", "18mo", "24mo"], index=1)
    fetch_chunk = st.slider("yfinance batch size", 20, 120, 80, 10)
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, max_value=2000, value=0, step=50)
    min_liq = st.slider("Min average liquidity (IDR bn, proxy)", 0, 50, 5)
    only_best = st.checkbox("Only READY_LONG / WATCH / WATCH_REBOUND", value=False)
    st.markdown("---")
    uploaded_broker = st.file_uploader("Optional broker summary CSV", type=["csv"])
    uploaded_done = st.file_uploader("Optional done detail CSV", type=["csv"])
    uploaded_ob = st.file_uploader("Optional orderbook CSV", type=["csv"])
    run = st.button("Run scanner", type="primary")

if "scan_df" not in st.session_state:
    st.session_state.scan_df = None
    st.session_state.price_df = None
    st.session_state.broker_df = None
    st.session_state.done_df = None
    st.session_state.orderbook_df = None
    st.session_state.burst_df = None

if run:
    if universe_choice == "Full IHSG (free auto-fetch)":
        tickers = full_tickers
    elif universe_choice == "Sample IDX universe":
        tickers = sample_tickers
    else:
        tickers = [t.strip().upper().replace(".JK", "") for t in manual_text.split(",") if t.strip()]

    tickers = _clean_symbol_list(tickers)
    if max_tickers and max_tickers > 0:
        tickers = tickers[: int(max_tickers)]

    if not tickers:
        st.error("Ticker list kosong.")
    else:
        with st.spinner(f"Fetching EOD data for {len(tickers)} tickers..."):
            price_df = fetch_prices(tuple(tickers), period=period, chunk_size=int(fetch_chunk))
        if price_df.empty:
            st.error("Gagal ambil data harga. Coba lagi atau kurangi jumlah ticker.")
        else:
            price_df = compute_indicators(price_df)
            broker_df = None
            done_df = None
            orderbook_df = None
            burst_df = pd.DataFrame()
            if uploaded_broker is not None:
                try:
                    broker_df = parse_broker_csv(uploaded_broker.read())
                    broker_df = broker_df[broker_df["ticker"].isin(price_df["ticker"].unique())].copy()
                except Exception as e:
                    st.error(f"Broker CSV error: {e}")
            if uploaded_done is not None:
                try:
                    done_df = parse_done_detail_csv(uploaded_done.read())
                    done_df = done_df[done_df["ticker"].isin(price_df["ticker"].unique())].copy()
                except Exception as e:
                    st.error(f"Done detail CSV error: {e}")
            if uploaded_ob is not None:
                try:
                    orderbook_df = parse_orderbook_csv(uploaded_ob.read())
                    orderbook_df = orderbook_df[orderbook_df["ticker"].isin(price_df["ticker"].unique())].copy()
                except Exception as e:
                    st.error(f"Orderbook CSV error: {e}")
            if done_df is not None and not done_df.empty:
                burst_df = compute_burst_features(done_df, orderbook_df)
            scan_df = build_latest_scan(price_df, broker_df, burst_df)
            scan_df = scan_df[scan_df["liquidity_mn"].fillna(0) >= min_liq].copy()
            if only_best:
                scan_df = scan_df[scan_df["verdict"].isin(["READY_LONG", "WATCH", "WATCH_REBOUND"])]
            st.session_state.scan_df = scan_df
            st.session_state.price_df = price_df
            st.session_state.broker_df = broker_df
            st.session_state.done_df = done_df
            st.session_state.orderbook_df = orderbook_df
            st.session_state.burst_df = burst_df

scan_df = st.session_state.scan_df
price_df = st.session_state.price_df
broker_df = st.session_state.broker_df
burst_df = st.session_state.burst_df
orderbook_df = st.session_state.orderbook_df

if scan_df is None or scan_df.empty:
    st.info("Klik **Run scanner** untuk mulai. Broker/intraday layer sifatnya optional. Gunakan template di folder `data/` kalau mau test manual.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tickers scanned", int(scan_df["ticker"].nunique()))
c2.metric("READY_LONG", int((scan_df["verdict"] == "READY_LONG").sum()))
c3.metric("WATCH", int((scan_df["verdict"] == "WATCH").sum()))
c4.metric("WATCH_REBOUND", int((scan_df["verdict"] == "WATCH_REBOUND").sum()))
c5.metric("Intraday burst", "ON" if burst_df is not None and not burst_df.empty else "OFF")

if "full_universe_source" in locals() and universe_choice == "Full IHSG (free auto-fetch)":
    st.caption(f"Universe source used: {full_universe_source}. yfinance full-universe fetch is batched; some symbols can still fail or come back partial on a given run.")

view = st.radio("View", ["Scanner", "Ticker Detail", "Intraday Burst", "Data Audit"], horizontal=True)

if view == "Scanner":
    show_cols = [
        "ticker", "close", "verdict", "phase", "trend_quality", "breakout_integrity", "false_breakout_risk",
        "dry_score", "wet_score", "liquidity_mn", "broker_alignment_score", "institutional_support",
        "institutional_resistance", "burst_bias", "latest_event_label", "bullish_burst_score",
        "bearish_burst_score", "bull_trap_score", "bear_trap_score", "dominant_accumulators", "dominant_distributors",
    ]
    st.subheader("Scanner Output")
    st.dataframe(scan_df[[c for c in show_cols if c in scan_df.columns]], use_container_width=True, hide_index=True)
    st.download_button("Download scanner CSV", scan_df.to_csv(index=False).encode("utf-8"), file_name="idx_eod_scanner_v3.csv", mime="text/csv")

elif view == "Ticker Detail":
    ticker = st.selectbox("Ticker", scan_df["ticker"].tolist(), index=0)
    row = scan_df[scan_df["ticker"] == ticker].iloc[0]
    broker_ctx = summarize_broker_context(broker_df, ticker) if broker_df is not None else BrokerContext(ticker=ticker)
    burst_ctx = summarize_burst_context(burst_df, ticker) if burst_df is not None else BurstContext(ticker=ticker)

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(make_chart(price_df, ticker, broker_ctx), use_container_width=True)
    with right:
        st.subheader(f"{ticker} Summary")
        fields = [
            ("Verdict", row["verdict"]),
            ("Phase", row["phase"]),
            ("Close", row["close"]),
            ("Trend Quality", row["trend_quality"]),
            ("Breakout Integrity", row["breakout_integrity"]),
            ("False Breakout Risk", row["false_breakout_risk"]),
            ("Dry / Wet", f"{row['dry_score']} / {row['wet_score']}"),
            ("20D Support Proxy", row["support_20d"]),
            ("60D Resistance Proxy", row["resistance_60d"]),
            ("Institutional Support", row["institutional_support"] if pd.notna(row["institutional_support"]) else "-"),
            ("Institutional Resistance", row["institutional_resistance"] if pd.notna(row["institutional_resistance"]) else "-"),
            ("Broker Mode", row["broker_mode"]),
            ("Burst Bias", row.get("burst_bias", "NEUTRAL")),
            ("Latest Burst Event", row.get("latest_event_label", "NO_INTRADAY_DATA")),
            ("Bullish Burst", row.get("bullish_burst_score", 0)),
            ("Bearish Burst", row.get("bearish_burst_score", 0)),
            ("Bull Trap", row.get("bull_trap_score", 0)),
            ("Bear Trap", row.get("bear_trap_score", 0)),
            ("Dominant Accumulators", row["dominant_accumulators"]),
            ("Dominant Distributors", row["dominant_distributors"]),
        ]
        for k, v in fields:
            st.write(f"**{k}:** {v}")
        st.info(setup_text(row))
        if burst_ctx.latest_event_time is not None:
            st.caption(f"Latest burst timestamp: {burst_ctx.latest_event_time}")

    if burst_df is not None and not burst_df.empty:
        st.subheader("Intraday Burst Summary")
        st.plotly_chart(make_intraday_bars(burst_df, ticker), use_container_width=True)
        ev = burst_df[burst_df["ticker"] == ticker].copy().tail(40)
        keep = [
            "minute", "event_label", "gulungan_up_score", "gulungan_down_score", "bullish_burst_score",
            "bearish_burst_score", "bull_trap_score", "bear_trap_score", "absorption_after_up_score",
            "absorption_after_down_score"
        ]
        if not ev.empty:
            st.dataframe(ev[keep], use_container_width=True, hide_index=True)

elif view == "Intraday Burst":
    if burst_df is None or burst_df.empty:
        st.warning("Belum ada data intraday. Upload done detail CSV, dan optional orderbook CSV.")
    else:
        st.subheader("Latest Burst Events")
        latest_events = burst_df.sort_values(["ticker", "minute"]).groupby("ticker", as_index=False).tail(1)
        cols = [
            "ticker", "minute", "event_label", "gulungan_up_score", "gulungan_down_score", "bullish_burst_score",
            "bearish_burst_score", "bull_trap_score", "bear_trap_score", "absorption_after_up_score",
            "absorption_after_down_score"
        ]
        st.dataframe(latest_events[cols], use_container_width=True, hide_index=True)
        ticker = st.selectbox("Intraday ticker", latest_events["ticker"].tolist(), index=0, key="burst_ticker")
        st.plotly_chart(make_intraday_bars(burst_df, ticker), use_container_width=True)

else:
    st.subheader("Data Audit")
    audit = {
        "price_rows": int(len(price_df)) if price_df is not None else 0,
        "price_tickers": int(price_df["ticker"].nunique()) if price_df is not None else 0,
        "price_date_min": str(price_df["date"].min().date()) if price_df is not None and not price_df.empty else "-",
        "price_date_max": str(price_df["date"].max().date()) if price_df is not None and not price_df.empty else "-",
        "broker_rows": int(len(broker_df)) if broker_df is not None else 0,
        "broker_tickers": int(broker_df["ticker"].nunique()) if broker_df is not None and not broker_df.empty else 0,
        "broker_codes": int(broker_df["broker_code"].nunique()) if broker_df is not None and not broker_df.empty else 0,
        "burst_rows": int(len(burst_df)) if burst_df is not None else 0,
        "intraday_tickers": int(burst_df["ticker"].nunique()) if burst_df is not None and not burst_df.empty else 0,
        "orderbook_rows": int(len(orderbook_df)) if orderbook_df is not None else 0,
    }
    st.json(audit)
    if universe_choice == "Full IHSG (free auto-fetch)":
        st.write("**Universe source:**", full_universe_source)
        st.write("**Universe symbol count:**", len(full_tickers))
    if price_df is not None and not price_df.empty:
        st.write("**Tickers loaded:**", ", ".join(sorted(price_df["ticker"].unique())[:50]))
    if broker_df is not None and not broker_df.empty:
        st.write("**Brokers loaded:**", ", ".join(sorted(broker_df["broker_code"].unique())[:50]))
