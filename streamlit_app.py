from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_audit import audit_broker_summary, audit_done_detail, audit_orderbook, audit_prices, merge_audits
from src.fetch_prices import fetch_yf_prices_batched, retry_failed_tickers
from src.scoring import compute_ticker_features
from src.universe import resolve_universe_source

st.set_page_config(page_title="IDX Scanner V4.0", layout="wide")
BASE_DIR = Path(__file__).resolve().parent


def load_csv(upload, kind: str) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Gagal baca {kind} CSV: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_load_universe(mode: str, uploaded_universe: pd.DataFrame | None):
    return resolve_universe_source(mode=mode, base_dir=BASE_DIR, uploaded_df=uploaded_universe)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_prices(tickers: tuple[str, ...], start: str, batch_size: int):
    result = fetch_yf_prices_batched(list(tickers), start=start, batch_size=batch_size)
    if result.failed_tickers:
        retry_df = retry_failed_tickers(result.failed_tickers, start=start)
        if not retry_df.empty:
            merged = pd.concat([result.prices, retry_df], ignore_index=True)
            merged = merged.drop_duplicates(["ticker", "date"], keep="last")
            recovered = set(retry_df["ticker"].unique().tolist())
            result.failed_tickers = [t for t in result.failed_tickers if t not in recovered]
            result.prices = merged
    return result


def render_candles(price_df: pd.DataFrame, ticker: str):
    g = price_df[price_df["ticker"] == ticker].copy().sort_values("date")
    if g.empty:
        st.info("No price history for selected ticker.")
        return
    for span in [20, 50, 200]:
        g[f"ema{span}"] = g["close"].ewm(span=span, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name="Price"))
    for span in [20, 50, 200]:
        fig.add_trace(go.Scatter(x=g["date"], y=g[f"ema{span}"], name=f"EMA{span}"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


st.title("IDX Scanner V4.0 — Full Stack Base")
st.caption("Phase paling mentok yang masih deploy-safe: full-universe resolver, price-side EOD, broker truth layer, intraday burst + orderbook hooks, confidence, explainability.")

with st.sidebar:
    st.subheader("Universe")
    universe_mode = st.radio(
        "Universe mode",
        options=["full", "auto", "sample"],
        format_func=lambda x: {"full": "Full IHSG (local CSV first)", "auto": "Auto web fallback", "sample": "Sample only"}[x],
        index=0,
    )
    uploaded_universe = st.file_uploader("Optional universe CSV", type=["csv"], key="universe")

    st.subheader("Prices")
    history_months = st.slider("History (months)", min_value=6, max_value=36, value=12, step=3)
    batch_size = st.slider("yfinance batch size", min_value=10, max_value=120, value=80, step=10)
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, max_value=2500, value=0, step=50)

    st.subheader("Optional real data")
    broker_upload = st.file_uploader("Broker summary CSV", type=["csv"], key="broker")
    done_upload = st.file_uploader("Done detail CSV", type=["csv"], key="done")
    orderbook_upload = st.file_uploader("Orderbook CSV", type=["csv"], key="orderbook")

    run_scan = st.button("Run scanner", type="primary")

uploaded_universe_df = load_csv(uploaded_universe, "universe") if uploaded_universe else pd.DataFrame()
broker_df = load_csv(broker_upload, "broker")
done_df = load_csv(done_upload, "done_detail")
book_df = load_csv(orderbook_upload, "orderbook")

if run_scan:
    with st.spinner("Resolving universe..."):
        universe_df, universe_source, universe_warnings = cached_load_universe(
            universe_mode, uploaded_universe_df if not uploaded_universe_df.empty else None
        )
    if max_tickers and max_tickers > 0:
        universe_df = universe_df.head(int(max_tickers)).copy()
    tickers = universe_df["ticker"].astype(str).tolist()
    start = str(date.today() - timedelta(days=int(history_months * 31)))
    with st.spinner(f"Fetching EOD prices for {len(tickers)} tickers..."):
        fetch_result = cached_fetch_prices(tuple(tickers), start=start, batch_size=batch_size)
    price_df = fetch_result.prices.copy()
    scan_df = compute_ticker_features(price_df, broker_df=broker_df, done_df=done_df, orderbook_df=book_df)
    audit = merge_audits(
        audit_prices(price_df, attempted_tickers=len(fetch_result.attempted_tickers), failed_tickers=fetch_result.failed_tickers),
        audit_broker_summary(broker_df),
        audit_done_detail(done_df),
        audit_orderbook(book_df),
        universe_source,
        universe_warnings,
    )
    st.session_state["price_df"] = price_df
    st.session_state["scan_df"] = scan_df
    st.session_state["audit"] = audit

price_df = st.session_state.get("price_df", pd.DataFrame())
scan_df = st.session_state.get("scan_df", pd.DataFrame())
audit = st.session_state.get("audit", None)

if audit is not None:
    cols = st.columns(7)
    cols[0].metric("Tickers scanned", audit.get("ticker_count_loaded", 0))
    cols[1].metric("Attempted", audit.get("attempted_tickers", 0))
    cols[2].metric("Failed", audit.get("failed_ticker_count", 0))
    cols[3].metric("Source mode", audit.get("source_mode", "n/a"))
    cols[4].metric("Broker rows", audit.get("broker_rows", 0))
    cols[5].metric("Done rows", audit.get("done_rows", 0))
    cols[6].metric("Orderbook rows", audit.get("orderbook_rows", 0))
    st.caption(f"Universe source: {audit.get('universe_source', 'n/a')}")
    for w in audit.get("universe_warnings", []) or []:
        st.warning(w)
    if audit.get("failed_ticker_count", 0) > 0:
        st.info(f"Failed ticker preview: {audit.get('failed_tickers_preview', '')}")

view = st.radio("View", ["Scanner", "Ticker Detail", "Intraday / Broker", "Data Audit"], horizontal=True)

if view == "Scanner":
    st.subheader("Scanner Output")
    if scan_df.empty:
        st.info("Belum ada hasil scan. Klik Run scanner.")
    else:
        default_verdicts = [v for v in ["READY_LONG", "WATCH", "WATCH_REBOUND", "NEUTRAL", "TRIM", "AVOID", "ILLIQUID"] if v in scan_df["verdict"].unique()]
        verdict_filter = st.multiselect("Filter verdict", options=sorted(scan_df["verdict"].unique().tolist()), default=default_verdicts or sorted(scan_df["verdict"].unique().tolist()))
        show_complete = st.checkbox("Show only complete-data names", value=False)
        tmp = scan_df[scan_df["verdict"].isin(verdict_filter)].copy()
        if show_complete:
            tmp = tmp[tmp["data_completeness_score"] >= 70]
        cols = [
            "ticker","close","verdict","score_confidence","phase","trend_quality","breakout_integrity",
            "false_breakout_risk","dry_score_final","wet_score_final","liquidity_mn","broker_alignment_score",
            "broker_mode","institutional_support","institutional_resistance","gulungan_up_score","gulungan_down_score",
            "latest_event_label","why_now","why_not_yet","trigger","invalidator","dominant_risk"
        ]
        st.dataframe(tmp[cols], use_container_width=True, hide_index=True)
        st.download_button("Download scanner CSV", data=tmp.to_csv(index=False).encode("utf-8"), file_name="idx_scanner_v40.csv", mime="text/csv")

elif view == "Ticker Detail":
    st.subheader("Ticker Detail")
    if scan_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        ticker = st.selectbox("Ticker", options=scan_df["ticker"].tolist())
        row = scan_df.loc[scan_df["ticker"] == ticker].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Verdict", row["verdict"])
        c2.metric("Confidence", row["score_confidence"])
        c3.metric("Phase", row["phase"])
        c4.metric("Burst", row.get("latest_event_label", "-"))
        render_candles(price_df, ticker)
        detail_cols = [
            "trend_quality","breakout_integrity","false_breakout_risk","dry_score_final","wet_score_final",
            "broker_alignment_score","broker_mode","overhang_score","support_20d","resistance_60d",
            "institutional_support","institutional_resistance","gulungan_up_score","gulungan_down_score",
            "effort_result_up","effort_result_down","absorption_after_up_score","absorption_after_down_score",
            "bid_stack_quality","offer_stack_quality","data_completeness_score","module_agreement_score"
        ]
        detail_cols = [c for c in detail_cols if c in row.index]
        st.dataframe(pd.DataFrame([row[detail_cols]]), use_container_width=True, hide_index=True)
        st.markdown(f"**Dominant accumulator:** {row.get('dominant_accumulator', '-')}")
        st.markdown(f"**Dominant distributor:** {row.get('dominant_distributor', '-')}")
        st.markdown(f"**Why now:** {row['why_now']}")
        st.markdown(f"**Why not yet:** {row['why_not_yet']}")
        st.markdown(f"**Trigger:** {row['trigger']}")
        st.markdown(f"**Invalidator:** {row['invalidator']}")
        st.markdown(f"**Dominant risk:** {row['dominant_risk']}")

elif view == "Intraday / Broker":
    st.subheader("Intraday / Broker Context")
    if scan_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        cols = [
            "ticker", "broker_mode", "broker_alignment_score", "dominant_accumulator", "dominant_distributor",
            "institutional_support", "institutional_resistance", "overhang_score", "gulungan_up_score", "gulungan_down_score",
            "effort_result_up", "effort_result_down", "latest_event_label", "burst_bias", "bid_stack_quality",
            "offer_stack_quality", "absorption_after_up_score", "absorption_after_down_score", "tension_score", "fragility_score"
        ]
        cols = [c for c in cols if c in scan_df.columns]
        st.dataframe(scan_df[cols], use_container_width=True, hide_index=True)

else:
    st.subheader("Data Audit")
    if audit is None:
        st.info("Belum ada audit. Klik Run scanner.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("### Coverage")
            st.json({
                "ticker_count_loaded": audit.get("ticker_count_loaded"),
                "attempted_tickers": audit.get("attempted_tickers"),
                "failed_ticker_count": audit.get("failed_ticker_count"),
                "price_date_min": audit.get("price_date_min"),
                "price_date_max": audit.get("price_date_max"),
                "universe_source": audit.get("universe_source"),
                "source_mode": audit.get("source_mode"),
            })
        with right:
            st.markdown("### Optional Data Hooks")
            st.json({
                "broker_rows": audit.get("broker_rows"),
                "broker_count_loaded": audit.get("broker_count_loaded"),
                "broker_columns_ok": audit.get("broker_columns_ok"),
                "done_rows": audit.get("done_rows"),
                "done_columns_ok": audit.get("done_columns_ok"),
                "orderbook_rows": audit.get("orderbook_rows"),
                "orderbook_columns_ok": audit.get("orderbook_columns_ok"),
            })
