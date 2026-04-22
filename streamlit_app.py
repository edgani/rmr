from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="IDX Buy-Side Front-Run Board v4 | Broker Intelligence", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BROKER_DIR = DATA_DIR / "broker"
UNIVERSE_PATH = DATA_DIR / "idx_universe_full.csv"

# ========================= BROKER ARCHETYPES =========================
# Referensi: Hengky Adinata, Remora, NeoBDM, Klinik Penyesalan
BROKER_ARCHETYPES = {
    # Institusi / Whale
    "AI": {"type": "institutional", "pattern": "sweeping", "weight": 1.0},
    "BK": {"type": "institutional", "pattern": "block_trade", "weight": 1.0},
    "YU": {"type": "institutional", "pattern": "gradual_accum", "weight": 1.0},
    "MS": {"type": "institutional", "pattern": "sweeping", "weight": 1.0},
    "ML": {"type": "institutional", "pattern": "block_trade", "weight": 1.0},
    "JP": {"type": "institutional", "pattern": "gradual_accum", "weight": 1.0},
    "NM": {"type": "institutional", "pattern": "sweeping", "weight": 1.0},
    "DX": {"type": "market_maker", "pattern": "mm_defend", "weight": 0.8},
    "EP": {"type": "market_maker", "pattern": "mm_defend", "weight": 0.8},
    # Retail Big (bisa jadi impostor)
    "YP": {"type": "retail_big", "pattern": "momentum_chase", "weight": 0.4},
    "XC": {"type": "retail_big", "pattern": "scalping", "weight": 0.4},
    "PD": {"type": "retail_big", "pattern": "scalping", "weight": 0.4},
    "KK": {"type": "retail_big", "pattern": "momentum_chase", "weight": 0.4},
    "XL": {"type": "retail_big", "pattern": "scalping", "weight": 0.4},
    "NI": {"type": "retail_big", "pattern": "momentum_chase", "weight": 0.4},
    "MG": {"type": "retail_big", "pattern": "scalping", "weight": 0.4},
    # Retail kecil
    "RD": {"type": "retail_small", "pattern": "noise", "weight": 0.1},
    "GR": {"type": "retail_small", "pattern": "noise", "weight": 0.1},
    "DM": {"type": "retail_small", "pattern": "noise", "weight": 0.1},
    "AK": {"type": "retail_small", "pattern": "noise", "weight": 0.1},
    "BS": {"type": "retail_small", "pattern": "noise", "weight": 0.1},
}

# ========================= HELPERS =========================

def _safe_float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return float(x)
    except Exception:
        return default


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _series_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:+.{digits}f}%"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def _ret(close: pd.Series, n: int) -> float:
    close = _series_num(close).dropna()
    if len(close) <= n:
        return np.nan
    prev = float(close.iloc[-1 - n])
    if not math.isfinite(prev) or prev == 0:
        return np.nan
    return float(close.iloc[-1] / prev - 1.0)


def _normalize_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


# ========================= UNIVERSE =========================

@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Missing {UNIVERSE_PATH}")
    df = pd.read_csv(UNIVERSE_PATH)
    df = _normalize_text_cols(df)

    if "ticker" not in df.columns:
        raise ValueError("idx_universe_full.csv must contain column 'ticker'")
    if "symbol_yf" not in df.columns:
        df["symbol_yf"] = df["ticker"].astype(str).str.upper().str.strip() + ".JK"

    for col in ["company_name", "sector", "board", "status", "listing_date", "shares_outstanding"]:
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["symbol_yf"] = df["symbol_yf"].astype(str).str.upper().str.strip()
    df["sector"] = df["sector"].fillna("").astype(str).str.strip()
    df["board"] = df["board"].fillna("").astype(str).str.strip()
    df["status"] = df["status"].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df


# ========================= DATA FETCH =========================

@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_prices(symbols: Tuple[str, ...], period: str = "18mo", interval: str = "1d", batch_size: int = 80) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    all_symbols = list(dict.fromkeys(symbols))

    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i + batch_size]
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            continue

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            top = set(raw.columns.get_level_values(0))
            for sym in batch:
                if sym not in top:
                    continue
                sub = raw[sym].copy()
                if sub.empty:
                    continue
                sub.columns = [str(c).strip().lower() for c in sub.columns]
                sub["symbol_yf"] = sym
                sub["date"] = sub.index
                frames.append(sub.reset_index(drop=True))
        else:
            sub = raw.copy()
            if not sub.empty:
                sub.columns = [str(c).strip().lower() for c in sub.columns]
                sub["symbol_yf"] = batch[0]
                sub["date"] = sub.index
                frames.append(sub.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close", "adj close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    return out


# ========================= BROKER DATA INGESTION =========================

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_broker_broksum() -> Optional[pd.DataFrame]:
    """Load broksum CSV: date,ticker,broker,buy_lot,sell_lot,buy_value,sell_value"""
    path = BROKER_DIR / "broksum.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _normalize_text_cols(df)
    for col in ["buy_lot", "sell_lot", "buy_value", "sell_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "ticker", "broker"])


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_broker_done_detail() -> Optional[pd.DataFrame]:
    """Load done detail CSV: time,ticker,broker,price,lot,type(BUY/SELL)"""
    path = BROKER_DIR / "done_detail.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _normalize_text_cols(df)
    df["lot"] = pd.to_numeric(df["lot"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time", "ticker", "broker", "type"])


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_broker_bid_offer() -> Optional[pd.DataFrame]:
    """Load bid-offer CSV: time,ticker,broker,side(BID/OFFER),price,lot,level"""
    path = BROKER_DIR / "bid_offer.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _normalize_text_cols(df)
    df["lot"] = pd.to_numeric(df["lot"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time", "ticker", "side"])


# ========================= MARKET CONTEXT =========================

def build_market_context(px: pd.DataFrame) -> Dict:
    jk = px[px["symbol_yf"] == "^JKSE"].copy().sort_values("date")
    if jk.empty:
        return {
            "market_bias": 0.0,
            "market_regime": "netral",
            "jkse_ret20": 0.0,
            "jkse_ret60": 0.0,
            "breadth_above_20": np.nan,
            "route_primary": "netral",
            "top_catalyst": "butuh data harga IHSG",
        }

    c = _series_num(jk["close"]).dropna()
    if len(c) < 80:
        return {
            "market_bias": 0.0,
            "market_regime": "netral",
            "jkse_ret20": _ret(c, 20),
            "jkse_ret60": _ret(c, 60),
            "breadth_above_20": np.nan,
            "route_primary": "netral",
            "top_catalyst": "butuh history lebih panjang",
        }

    ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = c.ewm(span=200, adjust=False).mean().iloc[-1]
    ret20 = _ret(c, 20)
    ret60 = _ret(c, 60)

    market_bias = 0.0
    market_bias += 0.22 if c.iloc[-1] > ema20 else -0.15
    market_bias += 0.24 if c.iloc[-1] > ema50 else -0.18
    market_bias += 0.26 if c.iloc[-1] > ema200 else -0.20
    market_bias += _clip(_safe_float(ret20, 0.0) * 2.5 + 0.5, 0, 1) * 0.18 - 0.09
    market_bias += _clip(_safe_float(ret60, 0.0) * 1.7 + 0.5, 0, 1) * 0.10 - 0.05
    market_bias = float(np.clip(market_bias, -1, 1))

    if market_bias > 0.25:
        market_regime = "risk-on"
        route_primary = "aggressive buy"
        top_catalyst = "fokus nama kuat, dekat trigger, dan leader sektor"
    elif market_bias > -0.10:
        market_regime = "netral"
        route_primary = "selective buy"
        top_catalyst = "pilih hanya struktur paling bersih dan near trigger"
    else:
        market_regime = "defensif"
        route_primary = "wait / selective rebound"
        top_catalyst = "hindari kejar, fokus kandidat early dan support kuat"

    return {
        "market_bias": market_bias,
        "market_regime": market_regime,
        "jkse_ret20": _safe_float(ret20, 0.0),
        "jkse_ret60": _safe_float(ret60, 0.0),
        "breadth_above_20": np.nan,
        "route_primary": route_primary,
        "top_catalyst": top_catalyst,
    }


# ========================= BROKER INTELLIGENCE ENGINE =========================

class BrokerIntelligence:
    def __init__(self, broksum_df: Optional[pd.DataFrame], done_df: Optional[pd.DataFrame]):
        self.broksum = broksum_df
        self.done = done_df
        self.archetypes = BROKER_ARCHETYPES

    def _classify_broker_row(self, broker_code: str, net_lot: float, buy_ratio: float) -> str:
        info = self.archetypes.get(broker_code, {"type": "unknown"})
        btype = info["type"]

        if btype in ["institutional", "market_maker"]:
            if net_lot > 0 and buy_ratio > 0.6:
                return "ACCUMULATOR"
            if net_lot < 0 and buy_ratio < 0.4:
                return "DISTRIBUTOR"
            return "NEUTRAL_INST"

        if btype == "retail_big":
            if abs(net_lot) > 500000 and buy_ratio > 0.7:
                return "RETAIL_BIG_BUY"
            if abs(net_lot) > 500000 and buy_ratio < 0.3:
                return "RETAIL_BIG_SELL"
            return "RETAIL_BIG_MIX"

        return "RETAIL_SMALL"

    def process_broksum(self, ticker: str, asof_date: datetime) -> Optional[Dict]:
        if self.broksum is None:
            return None
        df = self.broksum[self.broksum["ticker"] == ticker].copy()
        if df.empty:
            return None
        # Ambil data paling recent sebelum asof_date
        df = df[df["date"] <= asof_date]
        if df.empty:
            return None
        df = df.sort_values("date").tail(20)  # rolling 20 hari terakhir yang tersedia

        df["net_lot"] = df["buy_lot"] - df["sell_lot"]
        df["net_value"] = df["buy_value"] - df["sell_value"]
        df["buy_ratio"] = df["buy_lot"] / (df["buy_lot"] + df["sell_lot"] + 1e-9)
        df["total_lot"] = df["buy_lot"] + df["sell_lot"]

        # Classify per broker
        df["role"] = df.apply(lambda r: self._classify_broker_row(r["broker"], r["net_lot"], r["buy_ratio"]), axis=1)

        # Aggregate
        total_lot = df["total_lot"].sum()
        if total_lot == 0:
            return None

        accum_lot = df[df["role"] == "ACCUMULATOR"]["net_lot"].sum()
        distrib_lot = df[df["role"] == "DISTRIBUTOR"]["net_lot"].sum()
        inst_net = accum_lot + distrib_lot  # distrib negatif
        retail_big_buy = df[df["role"] == "RETAIL_BIG_BUY"]["net_lot"].sum()
        retail_big_sell = df[df["role"] == "RETAIL_BIG_SELL"]["net_lot"].sum()

        accum_dominance = max(0, accum_lot) / (total_lot + 1e-9)
        distrib_dominance = max(0, abs(distrib_lot)) / (total_lot + 1e-9)

        # Phase detection (FSM sederhana)
        net_total = df["net_lot"].sum()
        avg_buy_ratio = df["buy_ratio"].mean()

        if accum_dominance > 0.35 and net_total > 0:
            phase = "ACCUMULATION"
        elif distrib_dominance > 0.35 and net_total < 0:
            phase = "DISTRIBUTION"
        elif accum_dominance > 0.25 and net_total > total_lot * 0.1:
            phase = "MARK_UP"
        elif distrib_dominance > 0.25 and net_total < -total_lot * 0.1:
            phase = "MARK_DOWN"
        else:
            phase = "NEUTRAL"

        # Institutional cost basis (support/resistance)
        accum_df = df[df["role"] == "ACCUMULATOR"].copy()
        if not accum_df.empty:
            inst_support = (accum_df["buy_value"].sum()) / (accum_df["buy_lot"].sum() + 1e-9)
        else:
            inst_support = np.nan

        distrib_df = df[df["role"] == "DISTRIBUTOR"].copy()
        if not distrib_df.empty:
            inst_resistance = (distrib_df["sell_value"].sum()) / (distrib_df["sell_lot"].sum() + 1e-9)
        else:
            inst_resistance = np.nan

        # Broker list
        top_accum = df[df["role"] == "ACCUMULATOR"].groupby("broker")["net_lot"].sum().sort_values(ascending=False).head(3).index.tolist()
        top_distrib = df[df["role"] == "DISTRIBUTOR"].groupby("broker")["net_lot"].sum().sort_values(ascending=True).head(3).index.tolist()

        return {
            "accumulator_dominance": float(accum_dominance),
            "distributor_dominance": float(distrib_dominance),
            "institutional_net_flow": float(inst_net),
            "retail_big_net": float(retail_big_buy + retail_big_sell),
            "phase": phase,
            "inst_support": float(inst_support) if not pd.isna(inst_support) else None,
            "inst_resistance": float(inst_resistance) if not pd.isna(inst_resistance) else None,
            "top_accumulators": top_accum,
            "top_distributors": top_distrib,
            "total_broker_volume": int(total_lot),
        }

    def detect_fake_retail(self, ticker: str, asof_date: datetime) -> Optional[Dict]:
        if self.done is None:
            return None
        df = self.done[self.done["ticker"] == ticker].copy()
        if df.empty:
            return None
        df = df[df["time"] <= asof_date].sort_values("time").tail(1000)
        if len(df) < 10:
            return None

        # Statistik per broker per hari
        df["date"] = df["time"].dt.date
        stats = df.groupby(["date", "broker"]).agg(
            avg_lot=("lot", "mean"),
            std_lot=("lot", "std"),
            max_lot=("lot", "max"),
            freq=("lot", "count"),
            mode_lot=("lot", lambda x: pd.Series.mode(x).iloc[0] if len(pd.Series.mode(x)) > 0 else x.mean())
        ).reset_index()

        stats["std_lot"] = stats["std_lot"].fillna(0)
        stats["is_programmatic"] = stats["std_lot"] < stats["avg_lot"] * 0.15
        stats["is_rounded"] = stats["mode_lot"] % 100 == 0
        stats["is_splitter"] = (stats["freq"] > 50) & (stats["avg_lot"] < 10)
        stats["fake_score"] = (
            stats["is_programmatic"].astype(float) * 0.4 +
            stats["is_rounded"].astype(float) * 0.3 +
            stats["is_splitter"].astype(float) * 0.3
        )

        # Filter hanya retail broker
        retail_brokers = [k for k, v in self.archetypes.items() if v["type"] in ["retail_big", "retail_small"]]
        retail_stats = stats[stats["broker"].isin(retail_brokers)]

        if retail_stats.empty:
            return None

        avg_fake = retail_stats["fake_score"].mean()
        max_fake = retail_stats["fake_score"].max()
        suspicious_brokers = retail_stats[retail_stats["fake_score"] > 0.6]["broker"].unique().tolist()

        return {
            "fake_retail_score": float(avg_fake),
            "max_fake_score": float(max_fake),
            "suspicious_brokers": suspicious_brokers,
            "is_impostor_detected": len(suspicious_brokers) > 0,
        }

    def detect_crossing(self, ticker: str, asof_date: datetime) -> Optional[Dict]:
        if self.done is None:
            return None
        df = self.done[self.done["ticker"] == ticker].copy()
        if df.empty:
            return None
        df = df[df["time"] <= asof_date].sort_values("time").tail(2000)
        if len(df) < 20:
            return None

        df["time_rounded"] = df["time"].dt.floor("1min")
        crosses = []
        for (t, price), g in df.groupby(["time_rounded", "price"]):
            buys = g[g["type"] == "BUY"]
            sells = g[g["type"] == "SELL"]
            if len(buys) == 0 or len(sells) == 0:
                continue
            for bb in buys["broker"].unique():
                for sb in sells["broker"].unique():
                    if bb == sb:
                        continue
                    bb_vol = buys[buys["broker"] == bb]["lot"].sum()
                    sb_vol = sells[sells["broker"] == sb]["lot"].sum()
                    vol_ratio = min(bb_vol, sb_vol) / (max(bb_vol, sb_vol) + 1e-9)
                    if vol_ratio > 0.75 and min(bb_vol, sb_vol) > 10000:
                        crosses.append({
                            "buy_broker": bb,
                            "sell_broker": sb,
                            "lot_matched": min(bb_vol, sb_vol),
                            "ratio": float(vol_ratio),
                        })

        if not crosses:
            return {"crossing_detected": False, "cross_count": 0, "cross_pairs": []}

        cross_df = pd.DataFrame(crosses)
        return {
            "crossing_detected": True,
            "cross_count": len(cross_df),
            "cross_pairs": cross_df.groupby(["buy_broker", "sell_broker"])["lot_matched"].sum().sort_values(ascending=False).head(3).reset_index().to_dict("records"),
            "total_cross_lot": int(cross_df["lot_matched"].sum()),
        }


# ========================= VPA ENGINE =========================

class VPAEngine:
    @staticmethod
    def analyze(px_df: pd.DataFrame) -> Dict:
        df = px_df.copy().sort_values("date")
        if len(df) < 20:
            return {}

        close = _series_num(df["close"])
        high = _series_num(df["high"])
        low = _series_num(df["low"])
        open_ = _series_num(df["open"])
        volume = _series_num(df.get("volume", pd.Series([0] * len(df))))

        df["range"] = high - low
        df["body"] = (close - open_).abs()
        df["upper_shadow"] = high - close.combine(open_, max)
        df["lower_shadow"] = close.combine(open_, min) - low
        df["vol_avg20"] = volume.rolling(20).mean()
        df["body_pct"] = df["body"] / (df["range"] + 1e-9)
        df["vol_efficiency"] = df["body_pct"] * volume

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        # Climax: volume > 2.5x avg tapi body kecil
        climax = (last["volume"] > 2.5 * last["vol_avg20"]) and (last["body_pct"] < 0.3) if not pd.isna(last["vol_avg20"]) else False

        # No Demand: naik tipis, volume kering, upper shadow panjang
        no_demand = (last["close"] > last["open"]) and (last["volume"] < 0.7 * last["vol_avg20"]) and (last["upper_shadow"] > last["body"] * 2) if not pd.isna(last["vol_avg20"]) else False

        # No Supply: turun tipis, volume kering, lower shadow panjang
        no_supply = (last["close"] < last["open"]) and (last["volume"] < 0.7 * last["vol_avg20"]) and (last["lower_shadow"] > last["body"] * 2) if not pd.isna(last["vol_avg20"]) else False

        # Absorption: volume gede tapi range sempit
        absorption = (last["volume"] > 2 * last["vol_avg20"]) and (last["range"] < df["range"].tail(10).mean() * 0.5) if not pd.isna(last["vol_avg20"]) else False

        # Volume contraction (spring)
        vol_trend = df["volume"].tail(5).mean() / (df["volume"].tail(20).head(5).mean() + 1e-9)
        volume_drying = vol_trend < 0.8

        return {
            "climax_volume": bool(climax),
            "no_demand": bool(no_demand),
            "no_supply": bool(no_supply),
            "absorption": bool(absorption),
            "volume_drying": bool(volume_drying),
            "vol_efficiency": float(last["vol_efficiency"]) if not pd.isna(last["vol_efficiency"]) else 0.0,
            "dominant_pattern": (
                "CLIMAX" if climax else
                "ABSORPTION" if absorption else
                "NO_DEMAND" if no_demand else
                "NO_SUPPLY" if no_supply else
                "NORMAL"
            ),
        }

    @staticmethod
    def breakout_validity(px_df: pd.DataFrame, resistance: float, support: float) -> Dict:
        df = px_df.copy().sort_values("date")
        if len(df) < 2:
            return {"type": "NO_DATA", "confidence": 0.0}
        last = df.iloc[-1]
        c = last["close"]
        vol_avg = _series_num(df["volume"]).rolling(20).mean().iloc[-1]

        if c > resistance:
            vol_confirm = last["volume"] > vol_avg * 1.5 if not pd.isna(vol_avg) else False
            if vol_confirm:
                return {"type": "TRUE_BREAKOUT", "confidence": 0.85, "note": "Breakout + volume confirm"}
            else:
                return {"type": "FALSE_BREAKOUT", "confidence": 0.70, "note": "Breakout kering, hati-hati trap"}
        elif c < support:
            vol_confirm = last["volume"] > vol_avg * 1.5 if not pd.isna(vol_avg) else False
            if vol_confirm:
                return {"type": "TRUE_BREAKDOWN", "confidence": 0.85, "note": "Breakdown + volume confirm"}
            else:
                return {"type": "FALSE_BREAKDOWN", "confidence": 0.60, "note": "Breakdown tipis, bisa rebound"}
        return {"type": "NO_BREAK", "confidence": 0.0, "note": "Dalam range"}


# ========================= BID-OFFER ENGINE =========================

class BidOfferEngine:
    def __init__(self, bo_df: Optional[pd.DataFrame]):
        self.bo = bo_df
        self.retail_brokers = [k for k, v in BROKER_ARCHETYPES.items() if v["type"] in ["retail_big", "retail_small"]]
        self.inst_brokers = [k for k, v in BROKER_ARCHETYPES.items() if v["type"] in ["institutional", "market_maker"]]

    def tension_score(self, ticker: str, asof_date: datetime) -> Optional[Dict]:
        if self.bo is None:
            return None
        df = self.bo[self.bo["ticker"] == ticker].copy()
        if df.empty:
            return None
        df = df[df["time"] <= asof_date].sort_values("time").tail(500)
        if len(df) < 10:
            return None

        bids = df[df["side"] == "BID"]
        offers = df[df["side"] == "OFFER"]

        total_bid_lot = bids["lot"].sum() if not bids.empty else 0
        total_offer_lot = offers["lot"].sum() if not offers.empty else 0

        # Pintu 1: Struktur tipuan
        bid_levels = bids.groupby("price")["lot"].sum() if not bids.empty else pd.Series([0])
        offer_levels = offers.groupby("price")["lot"].sum() if not offers.empty else pd.Series([0])

        offer_conc = offer_levels.max() / (offer_levels.sum() + 1e-9) if offer_levels.sum() > 0 else 0
        bid_conc = bid_levels.max() / (bid_levels.sum() + 1e-9) if bid_levels.sum() > 0 else 0

        fake_offer_wall = (total_offer_lot > total_bid_lot * 1.5) and (offer_conc < 0.3)
        fake_bid_wall = (total_bid_lot > total_offer_lot * 1.5) and (bid_conc < 0.3)

        # Pintu 2: Frequency institusi
        bid_freq = bids["broker"].value_counts() if not bids.empty else pd.Series([0], index=["NA"])
        offer_freq = offers["broker"].value_counts() if not offers.empty else pd.Series([0], index=["NA"])

        inst_bid_freq = sum(bid_freq.get(b, 0) for b in self.inst_brokers)
        inst_offer_freq = sum(offer_freq.get(b, 0) for b in self.inst_brokers)
        freq_ratio = inst_bid_freq / (inst_offer_freq + 1e-9)

        # Pintu 3: Top 5 board
        top_bids = bids.nlargest(5, "lot") if not bids.empty else bids
        top_offers = offers.nlargest(5, "lot") if not offers.empty else offers

        # Offer eaten: offer berkurang drastis dari snapshot sebelumnya (proxy: top offer < 50% dari total)
        offer_eaten = (top_offers["lot"].sum() < total_offer_lot * 0.4) if not offers.empty else False

        tension = 0.0
        tension += 0.25 if fake_offer_wall else 0.0  # Tipuan offer = bullish
        tension += 0.25 if freq_ratio > 2.0 else 0.0
        tension += 0.30 if offer_eaten else 0.0
        tension += 0.20 if (top_bids["lot"].sum() > top_offers["lot"].sum() * 1.2) else 0.0

        return {
            "tension_score": float(_clip(tension)),
            "fake_offer_wall": bool(fake_offer_wall),
            "fake_bid_wall": bool(fake_bid_wall),
            "freq_ratio": float(freq_ratio),
            "offer_eaten": bool(offer_eaten),
            "top_bid_lot": int(top_bids["lot"].sum()) if not top_bids.empty else 0,
            "top_offer_lot": int(top_offers["lot"].sum()) if not top_offers.empty else 0,
            "interpretation": (
                "BREAKOUT IMMINENT" if tension > 0.75 else
                "ACCUMULATION_PHASE" if tension > 0.50 else
                "DISTRIBUTION_TRAP" if fake_bid_wall else
                "NEUTRAL"
            ),
        }


# ========================= GOD SCORE FUSION =========================

class GodScoreEngine:
    @staticmethod
    def fuse(
        price_features: Dict,
        broker_summary: Optional[Dict],
        vpa_signal: Optional[Dict],
        bo_signal: Optional[Dict],
    ) -> Dict:
        # Base dari price engine
        opp = _safe_float(price_features.get("opportunity_score"), 0.0)
        fr = _safe_float(price_features.get("front_run_score"), 0.0)
        conf = _safe_float(price_features.get("confidence"), 0.0)
        trend = _safe_float(price_features.get("trend_score"), 0.0)

        broker_boost = 0.0
        vpa_boost = 0.0
        bo_boost = 0.0
        notes = []

        # --- BROKER LAYER ---
        if broker_summary:
            if broker_summary["phase"] == "ACCUMULATION":
                broker_boost += 0.12
                notes.append("broker akumulasi")
            elif broker_summary["phase"] == "MARK_UP":
                broker_boost += 0.10
                notes.append("broker mark up")
            elif broker_summary["phase"] in ["DISTRIBUTION", "MARK_DOWN"]:
                broker_boost -= 0.15
                notes.append("broker distribusi")

            if broker_summary["accumulator_dominance"] > 0.4:
                broker_boost += 0.06
            if broker_summary.get("is_impostor_detected"):
                broker_boost -= 0.05
                notes.append("retail impostor detected")
            if broker_summary.get("crossing_detected"):
                broker_boost -= 0.04
                notes.append("crossing detected")

            # Inst support proximity (kalau dekat support institusi = bullish)
            inst_sup = broker_summary.get("inst_support")
            if inst_sup and price_features.get("close"):
                dist_to_support = (price_features["close"] - inst_sup) / price_features["close"]
                if 0 < dist_to_support < 0.05:
                    broker_boost += 0.05
                    notes.append("dekat inst support")

        # --- VPA LAYER ---
        if vpa_signal:
            if vpa_signal["absorption"]:
                vpa_boost += 0.08
                notes.append("absorption volume")
            if vpa_signal["no_supply"]:
                vpa_boost += 0.06
                notes.append("no supply")
            if vpa_signal["volume_drying"]:
                vpa_boost += 0.04
                notes.append("volume drying (spring)")
            if vpa_signal["climax_volume"]:
                vpa_boost -= 0.05  # Climax bisa distribusi
                notes.append("climax volume")

        # --- BID-OFFER LAYER ---
        if bo_signal:
            bo_boost += bo_signal["tension_score"] * 0.10
            if bo_signal["offer_eaten"]:
                notes.append("offer dimakan")
            if bo_signal["fake_offer_wall"]:
                notes.append("fake offer wall (tipuan jual)")

        # --- FINAL ---
        god_opp = _clip(opp + broker_boost + vpa_boost + bo_boost)
        god_fr = _clip(fr + broker_boost * 0.8 + vpa_boost * 0.6 + bo_boost * 0.8)
        god_conf = _clip(conf + abs(broker_boost) * 0.3 + abs(vpa_boost) * 0.2)

        # Classification
        if god_opp >= 0.70 and (bo_signal and bo_signal.get("offer_eaten")) and trend >= 0.6:
            classification = "GOD TIER: BREAKOUT IMMINENT"
            action = "FRONT_RUN_NOW"
        elif god_opp >= 0.60 and broker_summary and broker_summary["phase"] in ["ACCUMULATION", "MARK_UP"]:
            classification = "ALPHA: ACCUMULATION PHASE"
            action = "BUILD_POSITION"
        elif god_opp < 0.35 and broker_summary and broker_summary["phase"] in ["DISTRIBUTION", "MARK_DOWN"]:
            classification = "AVOID: DISTRIBUTION"
            action = "SELL_OR_IGNORE"
        else:
            classification = "NEUTRAL / WATCH"
            action = "WATCH"

        return {
            "god_opp": float(god_opp),
            "god_fr": float(god_fr),
            "god_conf": float(god_conf),
            "classification": classification,
            "action": action,
            "boost_notes": "; ".join(notes) if notes else "price only",
            "components": {
                "price": float(opp),
                "broker": float(broker_boost),
                "vpa": float(vpa_boost),
                "bo": float(bo_boost),
            },
        }


# ========================= SYMBOL FEATURES (ENHANCED) =========================

def compute_symbol_features(
    df_symbol: pd.DataFrame,
    bench20: float = 0.0,
    bench60: float = 0.0,
    market_bias: float = 0.0,
    broker_summary: Optional[Dict] = None,
    vpa_signal: Optional[Dict] = None,
    bo_signal: Optional[Dict] = None,
) -> Dict:
    s = df_symbol.sort_values("date").copy()
    if len(s) < 80:
        return {}

    close = _series_num(s["close"])
    high = _series_num(s["high"])
    low = _series_num(s["low"])
    volume = _series_num(s.get("volume", pd.Series([0] * len(s))))

    s["ema20"] = close.ewm(span=20, adjust=False).mean()
    s["ema50"] = close.ewm(span=50, adjust=False).mean()
    s["ema200"] = close.ewm(span=200, adjust=False).mean()
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    s["atr14"] = tr.rolling(14).mean()
    s["vol20"] = volume.rolling(20).mean()
    s["high20"] = high.rolling(20).max()
    s["high60"] = high.rolling(60).max()
    s["low20"] = low.rolling(20).min()
    s["low60"] = low.rolling(60).min()
    s["ret1"] = close.pct_change()

    c = _safe_float(close.iloc[-1])
    ema20 = _safe_float(s["ema20"].iloc[-1], c)
    ema50 = _safe_float(s["ema50"].iloc[-1], c)
    ema200 = _safe_float(s["ema200"].iloc[-1], ema50)
    atr14 = _safe_float(s["atr14"].iloc[-1], 0.0)
    high20 = _safe_float(s["high20"].iloc[-1], c)
    high60 = _safe_float(s["high60"].iloc[-1], c)
    low20 = _safe_float(s["low20"].iloc[-1], c)
    low60 = _safe_float(s["low60"].iloc[-1], c)
    vol_last = _safe_float(volume.iloc[-1], 0.0)
    vol20 = _safe_float(s["vol20"].iloc[-1], max(vol_last, 1.0))

    ret5 = _safe_float(_ret(close, 5), 0.0)
    ret20 = _safe_float(_ret(close, 20), 0.0)
    ret60 = _safe_float(_ret(close, 60), 0.0)
    rs20 = ret20 - _safe_float(bench20, 0.0)
    rs60 = ret60 - _safe_float(bench60, 0.0)
    rs_accel = rs20 - rs60 / 3.0

    above20 = 1 if c > ema20 else 0
    above50 = 1 if c > ema50 else 0
    above200 = 1 if c > ema200 else 0
    ema_stack = 1 if ema20 > ema50 > ema200 else 0
    trend_score = (above20 + above50 + above200 + ema_stack) / 4.0

    breakout_gap = max(high60 - c, 0.0) / max(c, 1e-9)
    breakout_nearness = _clip(1 - breakout_gap * 12)
    high20_gap = max(high20 - c, 0.0) / max(c, 1e-9)
    trigger_proximity = _clip(1 - high20_gap * 18)
    volume_expansion = vol_last / max(vol20, 1.0)

    atr_pct = atr14 / max(c, 1e-9) if c > 0 else 0.0
    dry_proxy = _clip(1 - atr_pct * 14)
    wet_proxy = 1 - dry_proxy
    extension = max(c / max(ema20, 1e-9) - 1, 0.0)
    too_late_risk = _clip(extension * 5 + max(ret20 - 0.15, 0) * 2.5)
    false_breakout_risk = _clip((1 - trend_score) * 0.50 + max(breakout_gap - 0.012, 0) * 10 + max(1 - volume_expansion, 0) * 0.10)

    liquidity_idr_bn = c * vol20 / 1e9
    liquidity_score = _clip((math.log10(max(liquidity_idr_bn, 0.01)) + 1.5) / 3.0)

    base_tightness = _clip(1 - atr_pct * 20)
    support_distance = (c - low20) / max(c, 1e-9)
    support_quality = _clip(1 - max(support_distance - 0.10, 0) * 5)

    opportunity_score = _clip(
        0.26 * trend_score
        + 0.17 * _clip(rs20 * 6 + 0.5)
        + 0.10 * _clip(rs60 * 4 + 0.5)
        + 0.12 * breakout_nearness
        + 0.10 * trigger_proximity
        + 0.10 * _clip(volume_expansion / 1.8)
        + 0.08 * dry_proxy
        + 0.07 * liquidity_score
        + 0.08 * _clip(market_bias + 0.5)
        - 0.12 * too_late_risk
        - 0.10 * false_breakout_risk
    )

    front_run_score = _clip(
        0.18 * _clip(rs20 * 5 + 0.45)
        + 0.15 * _clip(rs60 * 3 + 0.45)
        + 0.12 * _clip(rs_accel * 6 + 0.5)
        + 0.17 * trigger_proximity
        + 0.13 * base_tightness
        + 0.10 * dry_proxy
        + 0.06 * liquidity_score
        + 0.05 * support_quality
        + 0.10 * _clip(market_bias + 0.5)
        - 0.06 * too_late_risk
    )

    confidence = _clip(
        0.30 * trend_score
        + 0.15 * _clip(volume_expansion / 2.0)
        + 0.15 * _clip(rs20 * 6 + 0.5)
        + 0.10 * liquidity_score
        + 0.15 * _clip(market_bias + 0.5)
        + 0.15 * (1 - false_breakout_risk)
    )

    # ================= GOD SCORE FUSION =================
    base_feats = {
        "opportunity_score": opportunity_score,
        "front_run_score": front_run_score,
        "confidence": confidence,
        "trend_score": trend_score,
        "close": c,
    }
    god = GodScoreEngine.fuse(base_feats, broker_summary, vpa_signal, bo_signal)

    reasons = []
    if trend_score >= 0.75:
        reasons.append("trend sehat")
    elif trend_score >= 0.5:
        reasons.append("trend mulai rapi")
    if rs20 > 0.03:
        reasons.append("lebih kuat dari IHSG")
    elif rs20 > 0:
        reasons.append("mulai outperform IHSG")
    if trigger_proximity > 0.75:
        reasons.append("dekat trigger")
    if volume_expansion > 1.2:
        reasons.append("volume mulai masuk")
    if dry_proxy > 0.55:
        reasons.append("struktur ringan")
    if broker_summary and broker_summary["phase"] in ["ACCUMULATION", "MARK_UP"]:
        reasons.append("broker akumulasi")
    if vpa_signal and vpa_signal["absorption"]:
        reasons.append("absorption vol")
    if bo_signal and bo_signal["offer_eaten"]:
        reasons.append("offer dimakan")
    if not reasons:
        reasons.append("masih perlu konfirmasi")

    missing = []
    if trigger_proximity < 0.75:
        missing.append("trigger belum dekat")
    if trend_score < 0.5:
        missing.append("trend belum cukup rapi")
    if rs20 <= 0:
        missing.append("belum outperform IHSG")
    if volume_expansion < 1.0:
        missing.append("volume belum confirm")
    if too_late_risk > 0.55:
        missing.append("sudah agak telat")
    if broker_summary and broker_summary["phase"] in ["DISTRIBUTION", "MARK_DOWN"]:
        missing.append("broker distribusi")
    what_missing = ", ".join(missing[:2]) if missing else "sudah cukup lengkap"

    route = "risk-on" if market_bias > 0.2 else ("selective" if market_bias > -0.1 else "defensif")

    # Board assignment pakai god_opp
    god_opp = god["god_opp"]
    god_fr = god["god_fr"]

    if god_opp >= 0.62 and trigger_proximity >= 0.70 and too_late_risk < 0.70:
        board = "OPPORTUNITY SEKARANG"
    elif god_fr >= 0.52:
        board = "FRONT-RUN MARKET"
    else:
        board = "HIDDEN"

    if board == "OPPORTUNITY SEKARANG":
        if trigger_proximity >= 0.88 and too_late_risk < 0.30:
            status = "PALING DEKAT ENTRY"
            timing = "boleh dicicil"
        elif trend_score >= 0.75 and false_breakout_risk < 0.35:
            status = "STRUKTUR PALING BERSIH"
            timing = "boleh dicicil bertahap"
        else:
            status = "MASIH LAYAK BUY"
            timing = "boleh buy selektif"
    elif board == "FRONT-RUN MARKET":
        if trigger_proximity >= 0.80:
            status = "HAMPIR TRIGGER"
            timing = "tunggu pecah trigger"
        elif rs_accel > 0 and dry_proxy > 0.52:
            status = "PALING EARLY"
            timing = "masuk radar awal"
        else:
            status = "NEXT WAVE"
            timing = "tunggu giliran rotasi"
    else:
        status = "BELUM FOKUS BUY"
        timing = "belum prioritas"

    trigger = f"break {_fmt_num(high20 if high20 > c else high60)}"
    invalidator = f"close < {_fmt_num(min(ema20, low20))}"
    catalyst = "butuh break resistance" if board == "FRONT-RUN MARKET" else "ikuti momentum dan jaga support"

    # Micro note enhanced
    micro_parts = []
    if broker_summary:
        micro_parts.append(f"Phase: {broker_summary['phase']}")
        micro_parts.append(f"InstNet: {broker_summary['institutional_net_flow']:,.0f}")
    if vpa_signal:
        micro_parts.append(f"VPA: {vpa_signal['dominant_pattern']}")
    if bo_signal:
        micro_parts.append(f"BO: {bo_signal['interpretation']}")
    micro_note = " | ".join(micro_parts) if micro_parts else "Belum dinilai (price only)"

    return {
        "close": c,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "high20": high20,
        "high60": high60,
        "low20": low20,
        "low60": low60,
        "atr14": atr14,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "rs20": rs20,
        "rs60": rs60,
        "trend_score": trend_score,
        "volume_expansion": volume_expansion,
        "dry_proxy": dry_proxy,
        "wet_proxy": wet_proxy,
        "liquidity_idr_bn": liquidity_idr_bn,
        "liquidity_score": liquidity_score,
        "base_tightness": base_tightness,
        "breakout_gap": breakout_gap,
        "trigger_proximity": trigger_proximity,
        "opportunity_score": opportunity_score,
        "front_run_score": front_run_score,
        "too_late_risk": too_late_risk,
        "false_breakout_risk": false_breakout_risk,
        "confidence": confidence,
        # God score
        "god_opp": god_opp,
        "god_fr": god_fr,
        "god_conf": god["god_conf"],
        "god_classification": god["classification"],
        "god_action": god["action"],
        "god_notes": god["boost_notes"],
        "board": board,
        "status": status,
        "why_now": ", ".join(reasons[:3]),
        "what_missing": what_missing,
        "trigger": trigger,
        "invalidator": invalidator,
        "timing": timing,
        "route": route,
        "catalyst": catalyst,
        "micro_note": micro_note,
        # Raw broker/vpa/bo untuk advanced table
        "broker_phase": broker_summary.get("phase") if broker_summary else None,
        "broker_accum_dominance": broker_summary.get("accumulator_dominance") if broker_summary else None,
        "broker_inst_support": broker_summary.get("inst_support") if broker_summary else None,
        "broker_inst_resistance": broker_summary.get("inst_resistance") if broker_summary else None,
        "vpa_pattern": vpa_signal.get("dominant_pattern") if vpa_signal else None,
        "bo_tension": bo_signal.get("tension_score") if bo_signal else None,
        "bo_interpretation": bo_signal.get("interpretation") if bo_signal else None,
    }


# ========================= SCAN =========================

def assign_boards(scan: pd.DataFrame, market_bias: float) -> pd.DataFrame:
    if scan.empty:
        return scan
    out = scan.copy()

    # Opportunity candidates (pakai god_opp sekarang)
    opp_candidates = out[
        (out["trend_score"] >= 0.45)
        & (out["too_late_risk"] < 0.85)
        & (out["god_conf"] >= 0.35)
    ].copy()
    fr_candidates = out[
        (out["god_fr"] >= 0.35)
        & (out["god_conf"] >= 0.30)
    ].copy()

    if len(opp_candidates) > 0:
        opp_cut = float(max(0.60 if market_bias > 0 else 0.64, opp_candidates["god_opp"].quantile(0.93)))
    else:
        opp_cut = 0.65
    if len(fr_candidates) > 0:
        fr_cut = float(max(0.52 if market_bias > -0.15 else 0.56, fr_candidates["god_fr"].quantile(0.85)))
    else:
        fr_cut = 0.56

    out["board"] = "HIDDEN"
    out.loc[
        (out["god_opp"] >= opp_cut)
        & (out["trigger_proximity"] >= 0.60)
        & (out["too_late_risk"] < 0.80),
        "board",
    ] = "OPPORTUNITY SEKARANG"

    out.loc[
        (out["board"] == "HIDDEN")
        & (out["god_fr"] >= fr_cut)
        & (out["too_late_risk"] < 0.90),
        "board",
    ] = "FRONT-RUN MARKET"

    # Safety net
    if (out["board"] == "OPPORTUNITY SEKARANG").sum() == 0:
        topn = min(8, max(3, int(len(out) * 0.01)))
        idx = out.sort_values(["god_opp", "god_conf"], ascending=False).head(topn).index
        out.loc[idx, "board"] = "OPPORTUNITY SEKARANG"

    if (out["board"] == "FRONT-RUN MARKET").sum() == 0:
        remaining = out[out["board"] != "OPPORTUNITY SEKARANG"].copy()
        topn = min(20, max(8, int(len(out) * 0.03)))
        idx = remaining.sort_values(["god_fr", "trigger_proximity", "god_conf"], ascending=False).head(topn).index
        out.loc[idx, "board"] = "FRONT-RUN MARKET"

    return out


def run_scan(universe: pd.DataFrame, period: str, max_tickers: int, batch_size: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    use = universe.copy()
    if max_tickers > 0:
        use = use.head(max_tickers).copy()

    symbols = tuple(use["symbol_yf"].tolist() + ["^JKSE"])
    px = fetch_prices(symbols=symbols, period=period, batch_size=batch_size)
    if px.empty:
        return pd.DataFrame(), {
            "master_count": len(universe),
            "target_count": len(use),
            "loaded_count": 0,
            "failed_count": len(use),
            "coverage": 0.0,
            "failed_sample": use["ticker"].head(20).tolist(),
            "market_regime": "na",
            "market_bias": 0.0,
        }, pd.DataFrame()

    price_by_symbol = {sym: g.sort_values("date").copy() for sym, g in px.groupby("symbol_yf")}
    market = build_market_context(px)
    bench20 = market.get("jkse_ret20", 0.0)
    bench60 = market.get("jkse_ret60", 0.0)

    # Load broker data (kalau ada)
    broksum_df = load_broker_broksum()
    done_df = load_broker_done_detail()
    bo_df = load_broker_bid_offer()

    broker_engine = BrokerIntelligence(broksum_df, done_df)
    vpa_engine = VPAEngine()
    bo_engine = BidOfferEngine(bo_df)

    # As-of date = last date di price data
    asof = pd.to_datetime(px["date"].max())

    rows = []
    loaded, failed = [], []
    for _, meta in use.iterrows():
        sym = meta["symbol_yf"]
        tick = meta["ticker"]
        sub = price_by_symbol.get(sym)
        if sub is None or len(sub) < 80:
            failed.append(tick)
            continue

        # Broker signals
        bsum = broker_engine.process_broksum(tick, asof)
        fake = broker_engine.detect_fake_retail(tick, asof)
        cross = broker_engine.detect_crossing(tick, asof)
        if bsum and fake:
            bsum.update(fake)
        if bsum and cross:
            bsum.update(cross)

        vpa = vpa_engine.analyze(sub)
        bo = bo_engine.tension_score(tick, asof)

        feat = compute_symbol_features(sub, bench20=bench20, bench60=bench60, market_bias=market["market_bias"],
                                       broker_summary=bsum, vpa_signal=vpa, bo_signal=bo)
        if not feat:
            failed.append(tick)
            continue
        row = meta.to_dict()
        row.update(feat)
        rows.append(row)
        loaded.append(tick)

    scan = pd.DataFrame(rows)
    if not scan.empty:
        scan = assign_boards(scan, market_bias=market["market_bias"])

    audit = {
        "master_count": int(len(universe)),
        "target_count": int(len(use)),
        "loaded_count": int(len(loaded)),
        "failed_count": int(len(failed)),
        "failed_sample": failed[:30],
        "coverage": round(len(loaded) / max(len(use), 1), 4),
        "market_regime": market.get("market_regime", "na"),
        "market_bias": market.get("market_bias", 0.0),
        "route_primary": market.get("route_primary", "na"),
        "top_catalyst": market.get("top_catalyst", "na"),
        "broker_data_available": {
            "broksum": broksum_df is not None,
            "done_detail": done_df is not None,
            "bid_offer": bo_df is not None,
        },
        "asof_date": str(asof.date()),
    }
    return scan, audit, px


# ========================= PRESENTATION =========================

def board_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Ticker"] = out["ticker"]
    out["Sector"] = out["sector"].replace("", "—")
    out["Close"] = out["close"].map(_fmt_num)
    out["Status"] = out["status"]
    out["God Classification"] = out["god_classification"]
    out["God Score"] = (out["god_opp"] * 100).round(1).astype(str) + "%"
    out["Bid-Offer / Micro"] = out["micro_note"]
    out["Alasan Singkat"] = out["why_now"]
    out["Yang Masih Kurang"] = out["what_missing"]
    out["Trigger"] = out["trigger"]
    out["Invalidation"] = out["invalidator"]
    out["Timing"] = out["timing"]
    out["Confidence"] = (out["god_conf"].clip(0, 1) * 100).round(0).astype(int).astype(str) + "%"
    out["Route"] = out["route"]
    out["Catalyst"] = out["catalyst"]
    cols = [
        "Ticker", "Sector", "Close", "Status", "God Classification", "God Score",
        "Bid-Offer / Micro", "Alasan Singkat", "Yang Masih Kurang",
        "Trigger", "Invalidation", "Timing", "Confidence", "Route", "Catalyst"
    ]
    return out[cols]


def draw_price_chart(px: pd.DataFrame, symbol: str):
    sub = px[px["symbol_yf"] == symbol].copy().sort_values("date")
    if sub.empty:
        st.warning("Data harga tidak tersedia.")
        return
    sub["ema20"] = _series_num(sub["close"]).ewm(span=20, adjust=False).mean()
    sub["ema50"] = _series_num(sub["close"]).ewm(span=50, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=sub["date"], open=sub["open"], high=sub["high"], low=sub["low"], close=sub["close"], name="Price"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ema20"], mode="lines", name="EMA20", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ema50"], mode="lines", name="EMA50", line=dict(color="blue")))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ========================= UI =========================
st.title("IDX Buy-Side Front-Run Board v4")
st.caption("Price + Macro + Broker Intelligence + VPA + Bid-Offer Fusion Engine")

with st.sidebar:
    st.header("Scan Settings")
    period = st.selectbox("History", ["12mo", "18mo", "24mo", "36mo"], index=1)
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, value=0, step=100)
    batch_size = st.slider("Batch size yfinance", 20, 120, 80, 10)
    show_hidden = st.checkbox("Tampilkan juga yang belum fokus buy", value=False)
    st.divider()
    st.markdown("**Broker Data Folder:** `data/broker/`")
    st.markdown("- `broksum.csv`")
    st.markdown("- `done_detail.csv`")
    st.markdown("- `bid_offer.csv`")
    run = st.button("Run scan", type="primary", use_container_width=True)

try:
    universe = load_universe()
except Exception as e:
    st.error(f"Gagal load universe: {e}")
    st.stop()

st.info(f"Master universe siap pakai: **{len(universe):,} ticker**. File sumber: `data/idx_universe_full.csv`")

if "scan_df" not in st.session_state:
    st.session_state["scan_df"] = pd.DataFrame()
    st.session_state["audit"] = {}
    st.session_state["px"] = pd.DataFrame()

if run:
    with st.spinner("Sedang scan full universe + broker intelligence..."):
        scan_df, audit, px = run_scan(universe, period=period, max_tickers=max_tickers, batch_size=batch_size)
        st.session_state["scan_df"] = scan_df
        st.session_state["audit"] = audit
        st.session_state["px"] = px

scan_df = st.session_state["scan_df"]
audit = st.session_state["audit"]
px = st.session_state["px"]

# --- TOP METRICS ---
if audit:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target ticker", f'{audit.get("target_count", 0):,}')
    c2.metric("Loaded", f'{audit.get("loaded_count", 0):,}')
    c3.metric("Failed", f'{audit.get("failed_count", 0):,}')
    c4.metric("Coverage", f'{audit.get("coverage", 0) * 100:.1f}%')

    s1, s2 = st.columns(2)
    s1.info(f"Route sekarang: **{audit.get('route_primary', 'na')}** | Market regime: **{audit.get('market_regime', 'na')}**")
    s2.info(f"Catalyst summary: **{audit.get('top_catalyst', 'na')}** | Asof: **{audit.get('asof_date', 'na')}**")

    if audit.get("failed_count", 0) > 0:
        st.warning("Sebagian ticker gagal di-load dari yfinance. Ini normal kalau scan universe besar.")

    # Broker data status
    bavail = audit.get("broker_data_available", {})
    if not any(bavail.values()):
        st.info("💡 **Broker data belum tersedia.** Engine jalan mode *price-only*. Untuk full power, taruh CSV di `data/broker/`.")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Front-Run Boards",
    "🐋 Broker Intelligence",
    "⚡ Watchlist Good/Bad BO",
    "🔬 Advanced Table",
    "🗂️ Data Audit"
])

with tab1:
    if scan_df.empty:
        st.warning("Belum ada hasil scan. Klik **Run scan** dulu.")
    else:
        opp = scan_df[scan_df["board"] == "OPPORTUNITY SEKARANG"].copy().sort_values(["god_opp", "god_conf"], ascending=False)
        fr = scan_df[scan_df["board"] == "FRONT-RUN MARKET"].copy().sort_values(["god_fr", "trigger_proximity", "god_conf"], ascending=False)
        hidden = scan_df[scan_df["board"] == "HIDDEN"].copy().sort_values(["god_fr", "god_conf"], ascending=False)

        st.subheader("OPPORTUNITY SEKARANG")
        st.caption("Nama yang sudah paling dekat entry, struktur relatif bersih, dan masih layak buy sekarang.")
        if opp.empty:
            st.info("Belum ada nama di bucket ini di run sekarang.")
        else:
            st.dataframe(board_df(opp), use_container_width=True, hide_index=True)

        st.subheader("FRONT-RUN MARKET")
        st.caption("Nama yang belum obvious, tapi mulai align dan lebih cocok buat dipantau atau di-front-run.")
        if fr.empty:
            st.info("Belum ada nama di bucket ini di run sekarang.")
        else:
            st.dataframe(board_df(fr), use_container_width=True, hide_index=True)

        if show_hidden:
            st.subheader("BELUM FOKUS BUY")
            st.caption("Nama yang masih kalah prioritas buy di run sekarang.")
            st.dataframe(board_df(hidden), use_container_width=True, hide_index=True)

        with st.expander("Ticker Detail", expanded=False):
            pick = st.selectbox("Pilih ticker", scan_df["ticker"].tolist(), key="detail_tab1")
            row = scan_df[scan_df["ticker"] == pick].iloc[0]
            a, b, c, d = st.columns(4)
            a.metric("Status", row["status"])
            b.metric("God Score", f'{row["god_opp"] * 100:.1f}%')
            c.metric("Confidence", f'{row["god_conf"] * 100:.0f}%')
            d.metric("Close", _fmt_num(row["close"]))
            st.write(f"**Alasan singkat:** {row['why_now']}")
            st.write(f"**Yang masih kurang:** {row['what_missing']}")
            st.write(f"**Trigger:** {row['trigger']}")
            st.write(f"**Invalidation:** {row['invalidator']}")
            st.write(f"**Timing:** {row['timing']}")
            st.write(f"**God Classification:** {row['god_classification']} | **Action:** {row['god_action']}")
            st.write(f"**Route:** {row['route']} | **Catalyst:** {row['catalyst']}")
            st.write(f"**Micro Note:** {row['micro_note']}")
            draw_price_chart(px, row["symbol_yf"])

with tab2:
    st.subheader("Broker Intelligence Dashboard")
    if scan_df.empty:
        st.warning("Run scan dulu.")
    else:
        # Filter yang punya data broker
        brokered = scan_df[scan_df["broker_phase"].notna()].copy()
        if brokered.empty:
            st.info("Tidak ada data broker yang ter-load. Pastikan `data/broker/broksum.csv` tersedia.")
        else:
            st.caption(f"Menampilkan {len(brokered)} ticker dengan data broker.")

            # Tabel broker summary
            bo_cols = [
                "ticker", "company_name", "sector", "close", "broker_phase",
                "broker_accum_dominance", "broker_inst_support", "broker_inst_resistance",
                "god_opp", "god_classification"
            ]
            bo_disp = brokered[bo_cols].copy()
            bo_disp["broker_accum_dominance"] = (bo_disp["broker_accum_dominance"] * 100).round(1).astype(str) + "%"
            bo_disp["inst_support"] = bo_disp["broker_inst_support"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
            bo_disp["inst_resistance"] = bo_disp["broker_inst_resistance"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
            bo_disp["god_opp"] = (bo_disp["god_opp"] * 100).round(1).astype(str) + "%"
            st.dataframe(bo_disp.sort_values("god_opp", ascending=False), use_container_width=True, hide_index=True)

            # Pick ticker untuk detail broker
            pick2 = st.selectbox("Pilih ticker untuk detail broker", brokered["ticker"].tolist(), key="broker_detail")
            row2 = brokered[brokered["ticker"] == pick2].iloc[0]
            st.write(f"**Phase:** {row2['broker_phase']}")
            st.write(f"**Accumulator Dominance:** {row2['broker_accum_dominance'] * 100:.1f}%")
            st.write(f"**Institutional Support:** {_fmt_num(row2['broker_inst_support'])}")
            st.write(f"**Institutional Resistance:** {_fmt_num(row2['broker_inst_resistance'])}")
            st.write(f"**VPA Pattern:** {row2['vpa_pattern']}")
            st.write(f"**BO Tension:** {row2['bo_interpretation']} ({row2['bo_tension'] * 100:.1f}%)")
            st.write(f"**God Notes:** {row2['god_notes']}")

with tab3:
    st.subheader("Watchlist: Good vs Bad Bid-Offer")
    if scan_df.empty:
        st.warning("Run scan dulu.")
    else:
        # Good BO: accumulation phase + tension tinggi + god_opp bagus
        good_bo = scan_df[
            (scan_df["god_opp"] > 0.55)
            & (scan_df["bo_tension"] > 0.5)
            & (scan_df["broker_phase"].isin(["ACCUMULATION", "MARK_UP", None]))
        ].copy().sort_values("god_opp", ascending=False)

        # Bad BO: distribusi + tension jelek + god_opp rendah
        bad_bo = scan_df[
            (scan_df["god_opp"] < 0.40)
            & (
                scan_df["broker_phase"].isin(["DISTRIBUTION", "MARK_DOWN"])
                | (scan_df["bo_tension"] < 0.2)
            )
        ].copy().sort_values("god_opp", ascending=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Good Bid-Offer (Siap Naik / Front-Run)")
            if good_bo.empty:
                st.info("Belum ada kandidat good BO.")
            else:
                st.dataframe(
                    good_bo[["ticker", "sector", "close", "god_classification", "bo_interpretation", "broker_phase", "god_opp"]]
                    .assign(god_opp=lambda x: (x["god_opp"] * 100).round(1).astype(str) + "%"),
                    use_container_width=True, hide_index=True
                )
        with c2:
            st.markdown("### ❌ Bad Bid-Offer (Jual / Jangan Dilirik)")
            if bad_bo.empty:
                st.info("Belum ada kandidat bad BO.")
            else:
                st.dataframe(
                    bad_bo[["ticker", "sector", "close", "god_classification", "bo_interpretation", "broker_phase", "god_opp"]]
                    .assign(god_opp=lambda x: (x["god_opp"] * 100).round(1).astype(str) + "%"),
                    use_container_width=True, hide_index=True
                )

        # Yang tadinya watchlist tapi sekarang jelek
        if "prev_scan_df" in st.session_state and not st.session_state["prev_scan_df"].empty:
            prev = st.session_state["prev_scan_df"]
            prev_watch = prev[prev["board"].isin(["OPPORTUNITY SEKARANG", "FRONT-RUN MARKET"])]["ticker"].tolist()
            now_bad = bad_bo[bad_bo["ticker"].isin(prev_watch)]
            if not now_bad.empty:
                st.error("⚠️ **ALERT: Ticker yang tadinya di watchlist sekarang masuk zona BAD BO!**")
                st.dataframe(now_bad[["ticker", "god_classification", "broker_phase", "what_missing"]], use_container_width=True)

        # Save current sebagai prev untuk next run
        if not scan_df.empty and run:
            st.session_state["prev_scan_df"] = scan_df.copy()

with tab4:
    st.subheader("Advanced Table")
    if scan_df.empty:
        st.warning("Run scan dulu.")
    else:
        adv_cols = [
            "ticker", "company_name", "sector", "board", "status", "close", "ret5", "ret20", "ret60",
            "rs20", "rs60", "trend_score", "volume_expansion", "dry_proxy", "liquidity_idr_bn",
            "opportunity_score", "front_run_score", "trigger_proximity", "too_late_risk",
            "false_breakout_risk", "confidence", "god_opp", "god_fr", "god_conf",
            "broker_phase", "broker_accum_dominance", "vpa_pattern", "bo_tension", "bo_interpretation"
        ]
        adv = scan_df[[c for c in adv_cols if c in scan_df.columns]].copy()
        st.dataframe(adv.sort_values(["god_opp", "god_fr"], ascending=False), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Data Audit")
    st.json(audit)