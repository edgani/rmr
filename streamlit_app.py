import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

st.set_page_config(page_title="IDX EOD Smoke Test", layout="wide")

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "ICBP", "INDF",
    "ANTM", "MDKA", "UNTR", "AMRT", "PANI", "BRIS", "GOTO", "ADRO",
]


def normalize_tickers(raw: str) -> list[str]:
    seen = set()
    out: list[str] = []
    for token in raw.replace("\n", ",").split(","):
        t = token.strip().upper().replace(".JK", "")
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def add_jk_suffix(tickers: list[str]) -> list[str]:
    return [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]


def download_eod(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install requirements.txt first.")
    yf_tickers = add_jk_suffix(tickers)
    raw = yf.download(
        tickers=yf_tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("No data returned from yfinance.")

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0).unique())
        if set(level0) >= {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
            # Single ticker shape with OHLCV at level 0
            t = tickers[0]
            one = raw.copy().reset_index()
            one["Ticker"] = t
            one.columns = [c if isinstance(c, str) else c[0] for c in one.columns]
            frames.append(one)
        else:
            for t in yf_tickers:
                if t not in raw.columns.get_level_values(0):
                    continue
                one = raw[t].copy().reset_index()
                one["Ticker"] = t.replace(".JK", "")
                frames.append(one)
    else:
        one = raw.copy().reset_index()
        one["Ticker"] = tickers[0]
        frames.append(one)

    if not frames:
        raise RuntimeError("Parsed 0 ticker frames from yfinance response.")

    df = pd.concat(frames, ignore_index=True)
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    expected = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
    missing = [c for c in expected if c not in col_map]
    if missing:
        raise RuntimeError(f"Missing expected columns after fetch: {missing}")

    df = df.rename(columns={col_map[k]: k for k in expected})
    df = df[expected].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])  # type: ignore[assignment]
    g = out.groupby("ticker", group_keys=False)
    out["ret_1d"] = g["close"].pct_change()
    out["ma_20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["ma_50"] = g["close"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    out["vol_avg_20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["volume_burst"] = out["volume"] / out["vol_avg_20"].replace(0, np.nan)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["body_pct"] = (out["close"] - out["open"]).abs() / out["close"].replace(0, np.nan)
    out["trend_score"] = (
        50
        + 25 * np.tanh((out["close"] / out["ma_20"] - 1).fillna(0) * 10)
        + 25 * np.tanh((out["ma_20"] / out["ma_50"] - 1).fillna(0) * 10)
    )
    out["breakout_score"] = (
        50
        + 30 * np.tanh((out["ret_1d"].fillna(0)) * 15)
        + 20 * np.tanh((out["volume_burst"].fillna(1) - 1) * 1.5)
    ).clip(0, 100)
    out["risk_score"] = (
        50
        + 35 * np.tanh((out["range_pct"].fillna(0) - 0.03) * 12)
        - 15 * np.tanh((out["body_pct"].fillna(0) - 0.02) * 12)
    ).clip(0, 100)
    out["verdict_score"] = (0.55 * out["trend_score"] + 0.45 * out["breakout_score"] - 0.35 * out["risk_score"]).clip(0, 100)
    out["verdict"] = np.select(
        [
            out["verdict_score"] >= 55,
            out["verdict_score"].between(40, 55, inclusive="left"),
            out["verdict_score"] < 25,
        ],
        ["READY_LONG", "WATCH", "AVOID"],
        default="NEUTRAL",
    )
    return out


def latest_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    snap = (
        features.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .sort_values("verdict_score", ascending=False)
    )
    keep = [
        "date", "ticker", "close", "ret_1d", "volume", "volume_burst",
        "trend_score", "breakout_score", "risk_score", "verdict_score", "verdict",
    ]
    snap = snap[keep].copy()
    snap["ret_1d"] = (snap["ret_1d"] * 100).round(2)
    snap["volume_burst"] = snap["volume_burst"].round(2)
    for c in ["trend_score", "breakout_score", "risk_score", "verdict_score", "close"]:
        snap[c] = snap[c].round(2)
    return snap.reset_index(drop=True)


def plot_ticker(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df[df["ticker"] == ticker].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d["date"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name=ticker,
        )
    )
    if "ma_20" in d.columns:
        fig.add_trace(go.Scatter(x=d["date"], y=d["ma_20"], mode="lines", name="MA20"))
    if "ma_50" in d.columns:
        fig.add_trace(go.Scatter(x=d["date"], y=d["ma_50"], mode="lines", name="MA50"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("IDX EOD Smoke Test")
st.caption("Single-file Streamlit app. No multipage navigation. Built for clean GitHub deploy from zero.")

with st.sidebar:
    st.header("Settings")
    default_text = ", ".join(DEFAULT_TICKERS)
    tickers = normalize_tickers(st.text_area("IDX tickers", value=default_text, height=120))
    start = st.date_input("Start date", value=date.today() - timedelta(days=365))
    end = st.date_input("End date", value=date.today())
    run = st.button("Fetch data", type="primary", use_container_width=True)
    st.markdown("---")
    st.write("App file for Streamlit Cloud:")
    st.code("streamlit_app.py")
    st.write("Do not create a `pages/` folder in this repo.")

if "prices" not in st.session_state:
    st.session_state["prices"] = pd.DataFrame()
if "features" not in st.session_state:
    st.session_state["features"] = pd.DataFrame()
if "error" not in st.session_state:
    st.session_state["error"] = ""

if run:
    try:
        if not tickers:
            raise ValueError("Enter at least one ticker.")
        prices = download_eod(tickers, start, end)
        features = build_features(prices)
        st.session_state["prices"] = prices
        st.session_state["features"] = features
        st.session_state["error"] = ""
    except Exception as e:  # pragma: no cover
        st.session_state["error"] = str(e)

prices = st.session_state["prices"]
features = st.session_state["features"]
error = st.session_state["error"]

if error:
    st.error(error)

if prices.empty:
    st.info("Press 'Fetch data' to download EOD prices from yfinance for IDX tickers with .JK suffix.")
    st.stop()

snap = latest_snapshot(features)
col1, col2, col3 = st.columns(3)
col1.metric("Tickers loaded", int(snap["ticker"].nunique()))
col2.metric("Rows", int(len(prices)))
col3.metric("Latest date", str(snap["date"].max()))

st.subheader("Latest watchlist")
st.dataframe(snap, use_container_width=True, hide_index=True)
st.download_button("Download latest watchlist CSV", data=to_csv_bytes(snap), file_name="latest_watchlist.csv", mime="text/csv")

left, right = st.columns([2, 1])
with left:
    selected = st.selectbox("Ticker detail", options=snap["ticker"].tolist(), index=0)
    st.plotly_chart(plot_ticker(features, selected), use_container_width=True)
with right:
    latest = snap[snap["ticker"] == selected].iloc[0]
    st.markdown("### Snapshot")
    st.write(latest.to_frame("value"))
    raw = prices[prices["ticker"] == selected].tail(10).copy()
    st.markdown("### Last 10 rows")
    st.dataframe(raw, use_container_width=True, hide_index=True)

st.subheader("Raw prices")
st.dataframe(prices.tail(100), use_container_width=True, hide_index=True)
st.download_button("Download raw prices CSV", data=to_csv_bytes(prices), file_name="prices_daily_real.csv", mime="text/csv")
