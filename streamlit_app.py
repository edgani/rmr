
from __future__ import annotations
import math
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="IDX Front-Run Board", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "idx_universe_full.csv"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Missing {UNIVERSE_PATH}")
    df = pd.read_csv(UNIVERSE_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("idx_universe_full.csv must contain column: ticker")
    if "symbol_yf" not in df.columns:
        df["symbol_yf"] = df["ticker"].astype(str).str.upper().str.strip() + ".JK"
    for col in ["company_name", "sector", "board", "status", "listing_date"]:
        if col not in df.columns:
            df[col] = ""
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["symbol_yf"] = df["symbol_yf"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_prices(symbols: Tuple[str, ...], period: str = "18mo", interval: str = "1d", batch_size: int = 80) -> pd.DataFrame:
    frames = []
    for i in range(0, len(symbols), batch_size):
        batch = list(symbols[i:i + batch_size])
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
            for sym in batch:
                if sym not in raw.columns.get_level_values(0):
                    continue
                try:
                    sub = raw[sym].copy()
                except Exception:
                    continue
                if sub.empty:
                    continue
                sub = sub.rename(columns={c: str(c).lower() for c in sub.columns})
                sub["symbol_yf"] = sym
                sub["date"] = sub.index
                frames.append(sub.reset_index(drop=True))
        else:
            sub = raw.copy()
            sub = sub.rename(columns={c: str(c).lower() for c in sub.columns})
            sub["symbol_yf"] = batch[0]
            sub["date"] = sub.index
            frames.append(sub.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close", "adj close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def last_valid(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def compute_symbol_features(df_symbol: pd.DataFrame, bench20: float = 0.0, bench60: float = 0.0, market_bias: float = 0.0, bench_20: float | None = None, bench_60: float | None = None) -> Dict:
    s = df_symbol.sort_values("date").copy()
    if len(s) < 80:
        return {}
    close = pd.to_numeric(s["close"], errors="coerce")
    high = pd.to_numeric(s["high"], errors="coerce")
    low = pd.to_numeric(s["low"], errors="coerce")
    vol = pd.to_numeric(s.get("volume", 0), errors="coerce").fillna(0)

    s["ema20"] = close.ewm(span=20, adjust=False).mean()
    s["ema50"] = close.ewm(span=50, adjust=False).mean()
    s["ema200"] = close.ewm(span=200, adjust=False).mean()
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    s["atr14"] = tr.rolling(14).mean()
    s["vol20"] = vol.rolling(20).mean()
    s["high60"] = high.rolling(60).max()
    s["low20"] = low.rolling(20).min()
    s["low60"] = low.rolling(60).min()

    c = float(close.iloc[-1])
    ema20 = float(s["ema20"].iloc[-1])
    ema50 = float(s["ema50"].iloc[-1])
    ema200 = float(s["ema200"].iloc[-1]) if not math.isnan(float(s["ema200"].iloc[-1])) else ema50
    atr14 = float(s["atr14"].iloc[-1]) if not math.isnan(float(s["atr14"].iloc[-1])) else 0.0
    high60 = float(s["high60"].iloc[-1]) if not math.isnan(float(s["high60"].iloc[-1])) else c
    low20 = float(s["low20"].iloc[-1]) if not math.isnan(float(s["low20"].iloc[-1])) else c
    low60 = float(s["low60"].iloc[-1]) if not math.isnan(float(s["low60"].iloc[-1])) else c
    vol_last = float(vol.iloc[-1])
    vol20 = float(s["vol20"].iloc[-1]) if not math.isnan(float(s["vol20"].iloc[-1])) else vol_last

    def ret(n):
        if len(close) <= n:
            return np.nan
        prev = float(close.iloc[-1 - n])
        if prev == 0 or math.isnan(prev):
            return np.nan
        return c / prev - 1

    ret5 = ret(5); ret20 = ret(20); ret60 = ret(60)
    if bench_20 is not None:
        bench20 = bench_20
    if bench_60 is not None:
        bench60 = bench_60
    rs20 = safe_float(ret20) - safe_float(bench20, 0)
    rs60 = safe_float(ret60) - safe_float(bench60, 0)

    trend_ok = (c > ema20) + (ema20 > ema50) + (ema50 > ema200)
    trend_score = trend_ok / 3.0
    breakout_gap = (high60 - c) / max(c, 1e-9)
    breakout_integrity = np.clip((c / max(high60, 1e-9)), 0, 1)
    volume_expansion = vol_last / max(vol20, 1.0)
    dry_proxy = np.clip(1 - (s["atr14"].iloc[-1] / max(c, 1e-9)) * 12, 0, 1) if c > 0 else 0.0
    extension = max((c / max(ema20, 1e-9) - 1), 0)
    pullback_ok = (c - ema20) / max(atr14, 1e-9) if atr14 > 0 else 0

    opportunity_score = (
        0.28 * trend_score
        + 0.18 * np.clip(rs20 * 8 + 0.5, 0, 1)
        + 0.15 * np.clip(rs60 * 5 + 0.5, 0, 1)
        + 0.15 * np.clip(volume_expansion / 2.0, 0, 1)
        + 0.14 * np.clip((1 - breakout_gap * 8), 0, 1)
        + 0.10 * dry_proxy
    )
    front_run_score = (
        0.24 * np.clip(rs20 * 8 + 0.45, 0, 1)
        + 0.20 * np.clip(rs60 * 5 + 0.45, 0, 1)
        + 0.20 * np.clip((1 - max(breakout_gap - 0.02, 0) * 8), 0, 1)
        + 0.16 * dry_proxy
        + 0.10 * np.clip(volume_expansion / 1.5, 0, 1)
        + 0.10 * np.clip(market_bias + 0.5, 0, 1)
    )
    too_late_risk = np.clip(extension * 6 + max(safe_float(ret20, 0) - 0.18, 0) * 2, 0, 1)
    false_breakout_risk = np.clip((1 - trend_score) * 0.4 + max(breakout_gap - 0.01, 0) * 6 + max(1 - volume_expansion, 0) * 0.15, 0, 1)

    if opportunity_score >= 0.67 and too_late_risk < 0.5 and false_breakout_risk < 0.55:
        board = "OPPORTUNITY SEKARANG"
    elif front_run_score >= 0.58:
        board = "FRONT-RUN MARKET"
    else:
        board = "HIDDEN"

    if board == "OPPORTUNITY SEKARANG":
        if breakout_gap <= 0.01 and too_late_risk < 0.35:
            label = "PALING DEKAT ENTRY"
        elif trend_score > 0.8 and false_breakout_risk < 0.35:
            label = "STRUKTUR PALING BERSIH"
        else:
            label = "MASIH LAYAK BUY"
    elif board == "FRONT-RUN MARKET":
        if breakout_gap <= 0.03:
            label = "HAMPIR TRIGGER"
        elif rs20 > 0 and dry_proxy > 0.45:
            label = "PALING EARLY"
        else:
            label = "NEXT WAVE"
    else:
        label = "BELUM FOKUS BUY"

    why = []
    if trend_score > 0.66:
        why.append("trend sehat")
    if rs20 > 0:
        why.append("lebih kuat dari IHSG")
    if volume_expansion > 1.15:
        why.append("volume mulai masuk")
    if dry_proxy > 0.5:
        why.append("struktur relatif ringan")
    if not why:
        why.append("belum cukup jelas")
    why_now = ", ".join(why[:3])

    if board == "OPPORTUNITY SEKARANG":
        trigger = f"tetap di atas {high60:,.0f} / lanjut kuat"
    else:
        trigger = f"break {high60:,.0f}"
    invalid = f"close < {min(ema20, low20):,.0f}"
    timing = "boleh dicicil" if board == "OPPORTUNITY SEKARANG" else "tunggu trigger"

    confidence = np.clip(
        0.35 * trend_score
        + 0.20 * np.clip(volume_expansion / 2, 0, 1)
        + 0.15 * np.clip(rs20 * 8 + 0.5, 0, 1)
        + 0.15 * np.clip(market_bias + 0.5, 0, 1)
        + 0.15 * (1 - false_breakout_risk),
        0, 1
    )

    route = "risk-on" if market_bias > 0.15 else ("netral" if market_bias > -0.15 else "defensif")
    catalyst = "butuh break resistance" if board == "FRONT-RUN MARKET" else "ikuti momentum"

    return {
        "close": c,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
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
        "opportunity_score": opportunity_score,
        "front_run_score": front_run_score,
        "too_late_risk": too_late_risk,
        "false_breakout_risk": false_breakout_risk,
        "board": board,
        "status": label,
        "why_now": why_now,
        "trigger": trigger,
        "invalidator": invalid,
        "timing": timing,
        "confidence": confidence,
        "route": route,
        "catalyst": catalyst,
        "micro_note": "Belum dinilai (price only)",
    }

def build_market_context(price_df: pd.DataFrame) -> Dict:
    jk = price_df[price_df["symbol_yf"] == "^JKSE"].copy()
    if jk.empty:
        return {"market_bias": 0.0, "market_regime": "tidak tersedia", "breadth": np.nan}
    jk = jk.sort_values("date")
    c = pd.to_numeric(jk["close"], errors="coerce")
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = c.ewm(span=200, adjust=False).mean().iloc[-1]
    ret20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else 0.0
    market_bias = 0.0
    market_bias += 0.3 if c.iloc[-1] > ema50 else -0.2
    market_bias += 0.3 if c.iloc[-1] > ema200 else -0.2
    market_bias += np.clip(ret20 * 3, -0.3, 0.3)
    market_regime = "risk-on" if market_bias > 0.25 else ("netral" if market_bias > -0.2 else "defensif")
    return {"market_bias": float(np.clip(market_bias, -1, 1)), "market_regime": market_regime, "jkse_close": float(c.iloc[-1]), "jkse_ret20": float(ret20)}

def run_scan(universe: pd.DataFrame, period: str, max_tickers: int, batch_size: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    use = universe.copy()
    if max_tickers > 0:
        use = use.head(max_tickers).copy()
    symbols = tuple(use["symbol_yf"].tolist() + ["^JKSE"])
    px = fetch_prices(symbols=symbols, period=period, batch_size=batch_size)
    if px.empty:
        return pd.DataFrame(), {"master_count": len(universe), "loaded_count": 0, "failed_count": len(use), "coverage": 0.0}, pd.DataFrame()
    price_by_symbol = {sym: g.sort_values("date").copy() for sym, g in px.groupby("symbol_yf")}
    market = build_market_context(px)
    bench20 = market.get("jkse_ret20", 0.0)
    jk = price_by_symbol.get("^JKSE", pd.DataFrame())
    bench60 = np.nan
    if not jk.empty:
        c = pd.to_numeric(jk["close"], errors="coerce")
        if len(c) > 61:
            bench60 = c.iloc[-1] / c.iloc[-61] - 1
    rows = []
    loaded = []
    failed = []
    for _, meta in use.iterrows():
        sym = meta["symbol_yf"]
        sub = price_by_symbol.get(sym)
        if sub is None or len(sub) < 80:
            failed.append(meta["ticker"])
            continue
        feat = compute_symbol_features(sub, bench20=bench20, bench_60=bench60, market_bias=market["market_bias"])
        if not feat:
            failed.append(meta["ticker"])
            continue
        row = meta.to_dict()
        row.update(feat)
        rows.append(row)
        loaded.append(meta["ticker"])
    scan = pd.DataFrame(rows)
    audit = {
        "master_count": int(len(universe)),
        "target_count": int(len(use)),
        "loaded_count": int(len(loaded)),
        "failed_count": int(len(failed)),
        "failed_sample": failed[:30],
        "coverage": round(len(loaded) / max(len(use), 1), 4),
        "market_regime": market.get("market_regime", "na"),
        "market_bias": market.get("market_bias", 0.0),
    }
    return scan, audit, px

def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:+.1f}%"

def fmt_num(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"

def board_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Confidence"] = (out["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    out["Close"] = out["close"].map(fmt_num)
    out["Route"] = out["route"].astype(str)
    out["Ticker"] = out["ticker"]
    out["Status"] = out["status"]
    out["Alasan Singkat"] = out["why_now"]
    out["Trigger"] = out["trigger"]
    out["Invalidation"] = out["invalidator"]
    out["Timing"] = out["timing"]
    out["Catalyst"] = out["catalyst"]
    out["Bid-Offer / Micro"] = out["micro_note"]
    cols = ["Ticker", "Status", "Close", "Bid-Offer / Micro", "Alasan Singkat", "Trigger", "Invalidation", "Timing", "Confidence", "Route", "Catalyst"]
    return out[cols]

def draw_price_chart(px: pd.DataFrame, symbol: str):
    sub = px[px["symbol_yf"] == symbol].copy().sort_values("date")
    if sub.empty:
        st.warning("Data harga tidak tersedia.")
        return
    sub["ema20"] = pd.to_numeric(sub["close"], errors="coerce").ewm(span=20, adjust=False).mean()
    sub["ema50"] = pd.to_numeric(sub["close"], errors="coerce").ewm(span=50, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=sub["date"], open=sub["open"], high=sub["high"], low=sub["low"], close=sub["close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ema20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ema50"], mode="lines", name="EMA50"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.title("IDX Buy-Side Front-Run Board")
st.caption("Full universe dari file IDX upload • price data via yfinance .JK • fokus buy-side only")

with st.sidebar:
    st.header("Scan Settings")
    period = st.selectbox("History", ["12mo", "18mo", "24mo", "36mo"], index=1)
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, value=0, step=50)
    batch_size = st.slider("Batch size yfinance", 20, 120, 80, 10)
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
    c4.metric("Coverage", f'{audit.get("coverage", 0)*100:.1f}%')

    b1, b2 = st.columns(2)
    b1.info(f"Market regime: **{audit.get('market_regime', 'na')}**")
    b2.info(f"Market bias: **{audit.get('market_bias', 0):+.2f}**")

    if audit.get("failed_count", 0) > 0:
        st.warning("Sebagian ticker gagal di-load dari yfinance. Ini normal kalau scan universe besar. Cek Data Audit di bawah untuk sampel ticker gagal.")

if scan_df.empty:
    st.warning("Belum ada hasil scan. Klik **Run scan** dulu.")
    st.stop()

opp = scan_df[scan_df["board"] == "OPPORTUNITY SEKARANG"].copy().sort_values(["opportunity_score", "confidence"], ascending=False)
fr = scan_df[scan_df["board"] == "FRONT-RUN MARKET"].copy().sort_values(["front_run_score", "confidence"], ascending=False)

st.subheader("OPPORTUNITY SEKARANG")
if opp.empty:
    st.info("Belum ada nama yang lolos bucket ini di run sekarang.")
else:
    st.dataframe(board_df(opp), use_container_width=True, hide_index=True)

st.subheader("FRONT-RUN MARKET")
if fr.empty:
    st.info("Belum ada nama yang lolos bucket ini di run sekarang.")
else:
    st.dataframe(board_df(fr), use_container_width=True, hide_index=True)

with st.expander("Ticker Detail", expanded=False):
    pick = st.selectbox("Pilih ticker", scan_df["ticker"].tolist())
    row = scan_df[scan_df["ticker"] == pick].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Status", row["status"])
    b.metric("Confidence", f'{row["confidence"]*100:.0f}%')
    c.metric("Close", fmt_num(row["close"]))
    st.write(f"**Alasan singkat:** {row['why_now']}")
    st.write(f"**Trigger:** {row['trigger']}")
    st.write(f"**Invalidation:** {row['invalidator']}")
    st.write(f"**Timing:** {row['timing']}")
    st.write(f"**Route:** {row['route']} | **Catalyst:** {row['catalyst']}")
    draw_price_chart(px, row["symbol_yf"])

with st.expander("Advanced Table", expanded=False):
    adv = scan_df[[
        "ticker","company_name","sector","board","close","ret5","ret20","ret60","rs20","rs60",
        "trend_score","opportunity_score","front_run_score","too_late_risk","false_breakout_risk","status"
    ]].copy()
    st.dataframe(adv.sort_values(["opportunity_score", "front_run_score"], ascending=False), use_container_width=True, hide_index=True)

with st.expander("Data Audit", expanded=False):
    st.json(audit)
