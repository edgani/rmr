from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="IDX Buy-Side Front-Run Board", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "idx_universe_full.csv"

# ---------------------------- Helpers ----------------------------

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


# ---------------------------- Universe ----------------------------

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


# ---------------------------- Data fetch ----------------------------

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


# ---------------------------- Market context ----------------------------

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


# ---------------------------- Symbol features ----------------------------

def compute_symbol_features(df_symbol: pd.DataFrame, bench20: float = 0.0, bench60: float = 0.0, market_bias: float = 0.0) -> Dict:
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
    what_missing = ", ".join(missing[:2]) if missing else "sudah cukup lengkap"

    route = "risk-on" if market_bias > 0.2 else ("selective" if market_bias > -0.1 else "defensif")
    if opportunity_score >= 0.62 and trigger_proximity >= 0.70 and too_late_risk < 0.70:
        board = "OPPORTUNITY SEKARANG"
    elif front_run_score >= 0.52:
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
    micro_note = "Belum dinilai (price only)"

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
        "board": board,
        "status": status,
        "why_now": ", ".join(reasons[:3]),
        "what_missing": what_missing,
        "trigger": trigger,
        "invalidator": invalidator,
        "timing": timing,
        "confidence": confidence,
        "route": route,
        "catalyst": catalyst,
        "micro_note": micro_note,
    }


# ---------------------------- Scan ----------------------------

def assign_boards(scan: pd.DataFrame, market_bias: float) -> pd.DataFrame:
    if scan.empty:
        return scan
    out = scan.copy()

    # Opportunity candidates
    opp_candidates = out[
        (out["trend_score"] >= 0.45)
        & (out["too_late_risk"] < 0.85)
        & (out["confidence"] >= 0.35)
    ].copy()
    fr_candidates = out[
        (out["front_run_score"] >= 0.35)
        & (out["confidence"] >= 0.30)
    ].copy()

    # adaptive thresholds
    if len(opp_candidates) > 0:
        opp_cut = float(max(0.60 if market_bias > 0 else 0.64, opp_candidates["opportunity_score"].quantile(0.93)))
    else:
        opp_cut = 0.65
    if len(fr_candidates) > 0:
        fr_cut = float(max(0.52 if market_bias > -0.15 else 0.56, fr_candidates["front_run_score"].quantile(0.85)))
    else:
        fr_cut = 0.56

    out["board"] = "HIDDEN"
    out.loc[
        (out["opportunity_score"] >= opp_cut)
        & (out["trigger_proximity"] >= 0.60)
        & (out["too_late_risk"] < 0.80),
        "board",
    ] = "OPPORTUNITY SEKARANG"

    out.loc[
        (out["board"] == "HIDDEN")
        & (out["front_run_score"] >= fr_cut)
        & (out["too_late_risk"] < 0.90),
        "board",
    ] = "FRONT-RUN MARKET"

    # safety net: never leave top boards empty
    if (out["board"] == "OPPORTUNITY SEKARANG").sum() == 0:
        topn = min(8, max(3, int(len(out) * 0.01)))
        idx = out.sort_values(["opportunity_score", "confidence"], ascending=False).head(topn).index
        out.loc[idx, "board"] = "OPPORTUNITY SEKARANG"

    if (out["board"] == "FRONT-RUN MARKET").sum() == 0:
        remaining = out[out["board"] != "OPPORTUNITY SEKARANG"].copy()
        topn = min(20, max(8, int(len(out) * 0.03)))
        idx = remaining.sort_values(["front_run_score", "trigger_proximity", "confidence"], ascending=False).head(topn).index
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

    rows = []
    loaded, failed = [], []
    for _, meta in use.iterrows():
        sym = meta["symbol_yf"]
        sub = price_by_symbol.get(sym)
        if sub is None or len(sub) < 80:
            failed.append(meta["ticker"])
            continue
        feat = compute_symbol_features(sub, bench20=bench20, bench60=bench60, market_bias=market["market_bias"])
        if not feat:
            failed.append(meta["ticker"])
            continue
        row = meta.to_dict()
        row.update(feat)
        rows.append(row)
        loaded.append(meta["ticker"])

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
    }
    return scan, audit, px


# ---------------------------- Presentation ----------------------------

def board_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Ticker"] = out["ticker"]
    out["Sector"] = out["sector"].replace("", "—")
    out["Close"] = out["close"].map(_fmt_num)
    out["Status"] = out["status"]
    out["Bid-Offer / Micro"] = out["micro_note"]
    out["Alasan Singkat"] = out["why_now"]
    out["Yang Masih Kurang"] = out["what_missing"]
    out["Trigger"] = out["trigger"]
    out["Invalidation"] = out["invalidator"]
    out["Timing"] = out["timing"]
    out["Confidence"] = (pd.to_numeric(out["confidence"], errors="coerce").fillna(0).clip(0, 1) * 100).round(0).astype(int).astype(str) + "%"
    out["Route"] = out["route"]
    out["Catalyst"] = out["catalyst"]
    cols = [
        "Ticker", "Sector", "Close", "Status", "Bid-Offer / Micro", "Alasan Singkat",
        "Yang Masih Kurang", "Trigger", "Invalidation", "Timing", "Confidence", "Route", "Catalyst"
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
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ema20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ema50"], mode="lines", name="EMA50"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------- UI ----------------------------
st.title("IDX Buy-Side Front-Run Board")
st.caption("Full universe dari file IDX upload • price data via yfinance .JK • fokus buy-side only")

with st.sidebar:
    st.header("Scan Settings")
    period = st.selectbox("History", ["12mo", "18mo", "24mo", "36mo"], index=1)
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, value=0, step=100)
    batch_size = st.slider("Batch size yfinance", 20, 120, 80, 10)
    show_hidden = st.checkbox("Tampilkan juga yang belum fokus buy", value=False)
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
    with st.spinner("Sedang scan full universe..."):
        scan_df, audit, px = run_scan(universe, period=period, max_tickers=max_tickers, batch_size=batch_size)
        st.session_state["scan_df"] = scan_df
        st.session_state["audit"] = audit
        st.session_state["px"] = px

scan_df = st.session_state["scan_df"]
audit = st.session_state["audit"]
px = st.session_state["px"]

if audit:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target ticker", f'{audit.get("target_count", 0):,}')
    c2.metric("Loaded", f'{audit.get("loaded_count", 0):,}')
    c3.metric("Failed", f'{audit.get("failed_count", 0):,}')
    c4.metric("Coverage", f'{audit.get("coverage", 0) * 100:.1f}%')

    s1, s2 = st.columns(2)
    s1.info(f"Route sekarang: **{audit.get('route_primary', 'na')}** | Market regime: **{audit.get('market_regime', 'na')}**")
    s2.info(f"Catalyst summary: **{audit.get('top_catalyst', 'na')}**")

    if audit.get("failed_count", 0) > 0:
        st.warning("Sebagian ticker gagal di-load dari yfinance. Ini normal kalau scan universe besar. Cek Data Audit untuk sampel ticker gagal.")

if scan_df.empty:
    st.warning("Belum ada hasil scan. Klik **Run scan** dulu.")
    st.stop()

opp = scan_df[scan_df["board"] == "OPPORTUNITY SEKARANG"].copy().sort_values(["opportunity_score", "confidence"], ascending=False)
fr = scan_df[scan_df["board"] == "FRONT-RUN MARKET"].copy().sort_values(["front_run_score", "trigger_proximity", "confidence"], ascending=False)
hidden = scan_df[scan_df["board"] == "HIDDEN"].copy().sort_values(["front_run_score", "confidence"], ascending=False)

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
    pick = st.selectbox("Pilih ticker", scan_df["ticker"].tolist())
    row = scan_df[scan_df["ticker"] == pick].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Status", row["status"])
    b.metric("Confidence", f'{_clip(_safe_float(row["confidence"], 0.0)) * 100:.0f}%')
    c.metric("Close", _fmt_num(row["close"]))
    st.write(f"**Alasan singkat:** {row['why_now']}")
    st.write(f"**Yang masih kurang:** {row['what_missing']}")
    st.write(f"**Trigger:** {row['trigger']}")
    st.write(f"**Invalidation:** {row['invalidator']}")
    st.write(f"**Timing:** {row['timing']}")
    st.write(f"**Route:** {row['route']} | **Catalyst:** {row['catalyst']}")
    draw_price_chart(px, row["symbol_yf"])

with st.expander("Advanced Table", expanded=False):
    adv_cols = [
        "ticker", "company_name", "sector", "board", "status", "close", "ret5", "ret20", "ret60",
        "rs20", "rs60", "trend_score", "volume_expansion", "dry_proxy", "liquidity_idr_bn",
        "opportunity_score", "front_run_score", "trigger_proximity", "too_late_risk", "false_breakout_risk", "confidence"
    ]
    adv = scan_df[adv_cols].copy()
    st.dataframe(adv.sort_values(["opportunity_score", "front_run_score"], ascending=False), use_container_width=True, hide_index=True)

with st.expander("Data Audit", expanded=False):
    st.json(audit)
