from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache_utils import (
    filter_cached_prices_for_universe,
    merge_cached_and_new_prices,
    persist_run_outputs,
    read_cached_prices,
    read_last_run,
)
from src.data_audit import audit_broker_summary, audit_done_detail, audit_orderbook, audit_prices, merge_audits
from src.fetch_prices import fetch_yf_prices_batched, retry_failed_tickers
from src.scoring import compute_ticker_features
from src.universe import resolve_universe_source
from src.route_overlay import derive_route_state, build_route_overlay
from src.normalizers import normalize_uploaded_csv

st.set_page_config(page_title="IDX Scanner V4.2 Maxed", layout="wide")
BASE_DIR = Path(__file__).resolve().parent


def load_csv(upload, kind: str) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Gagal baca {kind} CSV: {e}")
        return pd.DataFrame()
    try:
        return normalize_uploaded_csv(raw, kind)
    except Exception as e:
        st.warning(f"Normalisasi {kind} gagal, pakai raw CSV. Detail: {e}")
        return raw


@st.cache_data(show_spinner=False, ttl=3600)
def cached_load_universe(mode: str, uploaded_universe: pd.DataFrame | None):
    return resolve_universe_source(mode=mode, base_dir=BASE_DIR, uploaded_df=uploaded_universe)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_fetch_prices(tickers: tuple[str, ...], start: str, batch_size: int, pause_s: float):
    result = fetch_yf_prices_batched(list(tickers), start=start, batch_size=batch_size, pause_s=pause_s)
    retry_reports = []
    if result.failed_tickers:
        # second pass: single names
        retry_df = retry_failed_tickers(result.failed_tickers, start=start)
        if not retry_df.empty:
            merged = pd.concat([result.prices, retry_df], ignore_index=True)
            merged = merged.drop_duplicates(["ticker", "date"], keep="last")
            recovered = set(retry_df["ticker"].unique().tolist())
            still_failed = [t for t in result.failed_tickers if t not in recovered]
            retry_reports.append({
                "batch_no": "retry_single",
                "requested": len(result.failed_tickers),
                "loaded": len(recovered),
                "failed": len(still_failed),
                "failed_preview": ", ".join(still_failed[:10]),
                "error": None,
            })
            result.failed_tickers = still_failed
            result.prices = merged
    result.batch_reports.extend(retry_reports)
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


st.title("IDX Scanner V4.7 — Full Merge Final + Input Normalizer")
st.caption("Full-universe resolver + disk cache + retry + broker/intraday hardening + route-aware next-play overlay + explainability + ranking.")
last_run = read_last_run(BASE_DIR)
if last_run:
    st.caption(f"Last persisted run UTC: {last_run}")

with st.sidebar:
    st.subheader("Universe")
    universe_mode = st.radio(
        "Universe mode",
        options=["full", "auto", "sample"],
        format_func=lambda x: {"full": "Full IHSG (local CSV first)", "auto": "Auto web fallback", "sample": "Sample only"}[x],
        index=0,
    )
    uploaded_universe = st.file_uploader("Optional universe CSV / metadata CSV", type=["csv"], key="universe")

    st.subheader("Prices")
    history_months = st.slider("History (months)", min_value=6, max_value=36, value=12, step=3)
    batch_size = st.slider("yfinance batch size", min_value=10, max_value=120, value=80, step=10)
    batch_pause_s = st.slider("Pause between batches (s)", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, max_value=2500, value=0, step=50)
    use_cached_prices = st.checkbox("Use cached prices if available", value=True)
    refresh_mode = st.radio("Refresh mode", options=["full_refresh", "cache_then_fill"], index=1)

    st.subheader("Optional real data")
    broker_upload = st.file_uploader("Broker summary CSV", type=["csv"], key="broker")
    broker_master_upload = st.file_uploader("Broker master CSV (optional)", type=["csv"], key="broker_master")
    done_upload = st.file_uploader("Done detail CSV", type=["csv"], key="done")
    orderbook_upload = st.file_uploader("Orderbook CSV", type=["csv"], key="orderbook")
    route_events_upload = st.file_uploader("Route / catalyst events CSV (optional)", type=["csv"], key="route_events")

    run_scan = st.button("Run scanner", type="primary")

uploaded_universe_df = load_csv(uploaded_universe, "universe") if uploaded_universe else pd.DataFrame()
broker_df = load_csv(broker_upload, "broker")
broker_master_df = load_csv(broker_master_upload, "broker_master")
done_df = load_csv(done_upload, "done_detail")
book_df = load_csv(orderbook_upload, "orderbook")
route_events_df = load_csv(route_events_upload, "route_events")

if run_scan:
    with st.spinner("Resolving universe..."):
        universe_df, universe_source, universe_warnings = cached_load_universe(
            universe_mode, uploaded_universe_df if not uploaded_universe_df.empty else None
        )
    if max_tickers and max_tickers > 0:
        universe_df = universe_df.head(int(max_tickers)).copy()
    tickers = universe_df["ticker"].astype(str).tolist()
    start = str(date.today() - timedelta(days=int(history_months * 31)))

    cached_prices = read_cached_prices(BASE_DIR) if use_cached_prices else pd.DataFrame()
    cached_prices = filter_cached_prices_for_universe(cached_prices, tickers, min_start=start)
    price_df = pd.DataFrame()
    fetch_result = None

    with st.spinner(f"Fetching EOD prices for {len(tickers)} tickers..."):
        if refresh_mode == "cache_then_fill" and not cached_prices.empty:
            have_tickers = set(cached_prices["ticker"].astype(str).unique().tolist())
            missing_tickers = [t for t in tickers if t not in have_tickers]
            if missing_tickers:
                fetch_result = cached_fetch_prices(tuple(missing_tickers), start=start, batch_size=batch_size, pause_s=batch_pause_s)
                price_df = merge_cached_and_new_prices(cached_prices, fetch_result.prices)
            else:
                price_df = cached_prices.copy()
                class Dummy:
                    prices = price_df
                    failed_tickers = []
                    attempted_tickers = tickers
                    batch_reports = [{"batch_no": "cache_only", "requested": len(tickers), "loaded": len(tickers), "failed": 0, "failed_preview": "", "error": None}]
                fetch_result = Dummy()
        else:
            fetch_result = cached_fetch_prices(tuple(tickers), start=start, batch_size=batch_size, pause_s=batch_pause_s)
            price_df = merge_cached_and_new_prices(cached_prices, fetch_result.prices) if use_cached_prices else fetch_result.prices.copy()

    scan_df = compute_ticker_features(price_df, broker_df=broker_df, done_df=done_df, orderbook_df=book_df, metadata_df=universe_df, broker_master_df=broker_master_df)
    if not scan_df.empty:
        bias_series = pd.to_numeric(scan_df.get("market_bias_score", pd.Series(dtype=float)), errors="coerce") if "market_bias_score" in scan_df.columns else pd.Series(dtype=float)
        market_bias = float(bias_series.median()) if not bias_series.empty else 50.0
        regime = str(scan_df["market_regime"].mode().iloc[0]) if "market_regime" in scan_df.columns and not scan_df["market_regime"].dropna().empty else "CHOPPY"
        exec_mode = str(scan_df["execution_mode"].mode().iloc[0]) if "execution_mode" in scan_df.columns and not scan_df["execution_mode"].dropna().empty else "SELECTIVE"
        mh_count = int((scan_df.get("burst_bias", pd.Series(dtype=object)).astype(str).eq("BULLISH")).sum()) if "burst_bias" in scan_df.columns else 0
        catalyst_score = 0.0
        analog_label = "unknown"
        scenario_family = "unknown"
        if not route_events_df.empty:
            if "catalyst_score" in route_events_df.columns:
                catalyst_score = float(pd.to_numeric(route_events_df["catalyst_score"], errors="coerce").fillna(0).max())
            if "analog_label" in route_events_df.columns and route_events_df["analog_label"].notna().any():
                analog_label = str(route_events_df["analog_label"].dropna().iloc[0])
            if "scenario_family" in route_events_df.columns and route_events_df["scenario_family"].notna().any():
                scenario_family = str(route_events_df["scenario_family"].dropna().iloc[0])
        route_state = derive_route_state(market_regime=regime, execution_mode=exec_mode, market_bias_score=market_bias, most_hated_clear_count=mh_count, catalyst_window_score=catalyst_score, analog_label=analog_label, scenario_family=scenario_family)
        scan_df = build_route_overlay(scan_df, route_state, route_events_df if not route_events_df.empty else None)
        if "route_fit_score" in scan_df.columns:
            rel = pd.to_numeric(scan_df.get("relative_strength_20d", 0), errors="coerce").fillna(0.0)
            conf = pd.to_numeric(scan_df.get("score_confidence", 0), errors="coerce").fillna(0.0)
            catalyst = pd.to_numeric(scan_df.get("catalyst_window_score", 0), errors="coerce").fillna(0.0) * 100.0
            scan_df["route_rank_score"] = (scan_df["route_fit_score"].fillna(0.0) * 55.0 + conf * 0.25 + catalyst * 0.10 + rel.clip(-0.10, 0.10) * 100.0).round(2)
    audit = merge_audits(
        audit_prices(price_df, attempted_tickers=len(fetch_result.attempted_tickers), failed_tickers=fetch_result.failed_tickers, batch_reports=fetch_result.batch_reports),
        audit_broker_summary(broker_df),
        audit_done_detail(done_df),
        audit_orderbook(book_df),
        universe_source,
        universe_warnings,
    )
    audit["cache_mode"] = refresh_mode
    audit["cache_prices_rows"] = int(len(cached_prices)) if use_cached_prices else 0
    persist_info = persist_run_outputs(BASE_DIR, price_df, scan_df, audit)
    audit["persist_info"] = persist_info
    st.session_state["price_df"] = price_df
    st.session_state["scan_df"] = scan_df
    st.session_state["audit"] = audit
    st.session_state["universe_df"] = universe_df

price_df = st.session_state.get("price_df", pd.DataFrame())
scan_df = st.session_state.get("scan_df", pd.DataFrame())
audit = st.session_state.get("audit", None)
universe_df = st.session_state.get("universe_df", pd.DataFrame())

if audit is not None:
    cols = st.columns(10)
    cols[0].metric("Tickers scanned", audit.get("ticker_count_loaded", 0))
    cols[1].metric("Attempted", audit.get("attempted_tickers", 0))
    cols[2].metric("Failed", audit.get("failed_ticker_count", 0))
    cols[3].metric("Source mode", audit.get("source_mode", "n/a"))
    cols[4].metric("Broker rows", audit.get("broker_rows", 0))
    cols[5].metric("Done rows", audit.get("done_rows", 0))
    cols[6].metric("Orderbook rows", audit.get("orderbook_rows", 0))
    cols[7].metric("Universe source", audit.get("universe_source", "n/a"))
    cols[8].metric("Cache rows", audit.get("cache_prices_rows", 0))
    cols[9].metric("Refresh mode", audit.get("cache_mode", "-"))
    if audit.get("stale_warning"):
        st.warning(audit["stale_warning"])
    for w in audit.get("universe_warnings", []) or []:
        st.warning(w)
    if audit.get("universe_source") == "fallback_sample":
        st.error("Universe masih fallback sample. Itu sebabnya ticker sedikit. Tambah data/idx_universe_full.csv atau biarkan app cache universe besar lebih dulu.")
    if audit.get("failed_ticker_count", 0) > 0:
        with st.expander("Failed ticker report"):
            st.write(audit.get("failed_tickers_preview", ""))
            if audit.get("batch_reports"):
                st.dataframe(pd.DataFrame(audit["batch_reports"]), use_container_width=True, hide_index=True)

view = st.radio("View", ["Scanner", "Top Ranks", "Ticker Detail", "Intraday / Broker", "Data Audit"], horizontal=True)

if view == "Scanner":
    st.subheader("Scanner Output")
    if scan_df.empty:
        st.info("Belum ada hasil scan. Klik Run scanner.")
    else:
        default_verdicts = [v for v in ["READY_LONG", "WATCH", "WATCH_REBOUND", "NEUTRAL", "TRIM", "AVOID", "ILLIQUID"] if v in scan_df["verdict"].unique()]
        verdict_filter = st.multiselect("Filter verdict", options=sorted(scan_df["verdict"].unique().tolist()), default=default_verdicts or sorted(scan_df["verdict"].unique().tolist()))
        show_complete = st.checkbox("Show only complete-data names", value=False)
        sector_choices = sorted([str(s) for s in scan_df.get("sector", pd.Series(dtype=object)).dropna().unique().tolist()]) if "sector" in scan_df.columns else []
        selected_sectors = st.multiselect("Filter sector", options=sector_choices, default=[])
        tmp = scan_df[scan_df["verdict"].isin(verdict_filter)].copy()
        if show_complete:
            tmp = tmp[tmp["data_completeness_score"] >= 70]
        if selected_sectors and "sector" in tmp.columns:
            tmp = tmp[tmp["sector"].astype(str).isin(selected_sectors)]
        cols = [
            "ticker","sector","close","verdict","score_confidence","phase","market_regime","execution_mode","liquidity_bucket",
            "trend_quality","breakout_integrity","false_breakout_risk","dry_score_final","wet_score_final","drywet_state",
            "broker_alignment_score","broker_persistence_score","broker_mode","dominant_accumulator","dominant_distributor","institutional_support","institutional_resistance","institutional_support_low","institutional_support_high","institutional_resistance_low","institutional_resistance_high","float_lock_score","supply_overhang_score",
            "gulungan_up_score","gulungan_down_score","latest_event_label","relative_strength_20d","sector_relative_strength_20d",
            "long_rank_score","risk_rank_score","route_rank_score","route_primary","route_bias","forward_radar_bucket","top_catalyst_title","why_now","why_not_yet","trigger","invalidator","dominant_risk"
        ]
        cols = [c for c in cols if c in tmp.columns]
        st.dataframe(tmp[cols], use_container_width=True, hide_index=True)
        st.download_button("Download scanner CSV", data=tmp.to_csv(index=False).encode("utf-8"), file_name="idx_scanner_v42.csv", mime="text/csv")

elif view == "Top Ranks":
    st.subheader("Top Rankings")
    if scan_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        n = st.slider("Top N", min_value=10, max_value=100, value=25, step=5)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Top Long Candidates")
            long_cols = [c for c in ["ticker","sector","verdict","long_rank_score","route_rank_score","score_confidence","phase","route_primary","forward_radar_bucket","relative_strength_20d","sector_relative_strength_20d","why_now","trigger","invalidator"] if c in scan_df.columns]
            long_df = scan_df[scan_df["verdict"].isin(["READY_LONG","WATCH","WATCH_REBOUND","NEUTRAL"])].sort_values(["long_rank_score","score_confidence"], ascending=False).head(n)
            st.dataframe(long_df[long_cols], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### Top Risk / Avoid / Trim")
            risk_cols = [c for c in ["ticker","sector","verdict","risk_rank_score","score_confidence","phase","route_bias","dominant_risk","why_not_yet","invalidator"] if c in scan_df.columns]
            risk_df = scan_df.sort_values(["risk_rank_score","score_confidence"], ascending=False).head(n)
            st.dataframe(risk_df[risk_cols], use_container_width=True, hide_index=True)

elif view == "Ticker Detail":
    st.subheader("Ticker Detail")
    if scan_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        ticker = st.selectbox("Ticker", options=scan_df["ticker"].tolist())
        row = scan_df.loc[scan_df["ticker"] == ticker].iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Verdict", row["verdict"])
        c2.metric("Confidence", row["score_confidence"])
        c3.metric("Phase", row["phase"])
        c4.metric("Burst", row.get("latest_event_label", "-"))
        c5.metric("Market", row.get("market_regime", "-"))
        c6.metric("Sector", row.get("sector", "-"))
        if "route_primary" in row.index:
            st.caption(f"Route: {row.get('route_primary','-')} | Bias: {row.get('route_bias','-')} | Radar: {row.get('forward_radar_bucket','-')}")
        render_candles(price_df, ticker)
        detail_cols = [
            "trend_quality","breakout_integrity","false_breakout_risk","dry_score_final","wet_score_final",
            "broker_alignment_score","broker_mode","overhang_score","support_20d","resistance_60d",
            "institutional_support","institutional_resistance","gulungan_up_score","gulungan_down_score",
            "effort_result_up","effort_result_down","post_up_followthrough_score","post_down_followthrough_score","split_order_score","bull_trap_score","bear_trap_score","absorption_after_up_score","absorption_after_down_score",
            "bid_stack_quality","offer_stack_quality","offer_refill_rate","bid_refill_rate","fake_wall_offer_score","fake_wall_bid_score","data_completeness_score","module_agreement_score",
            "market_breadth_pct","market_bias_score","relative_strength_20d","sector_relative_strength_20d",
            "long_rank_score","risk_rank_score"
        ]
        detail_cols = [c for c in detail_cols if c in row.index]
        st.dataframe(pd.DataFrame([row[detail_cols]]), use_container_width=True, hide_index=True)
        st.markdown(f"**Dominant accumulator:** {row.get('dominant_accumulator', '-')}")
        st.markdown(f"**Dominant distributor:** {row.get('dominant_distributor', '-')}")
        st.markdown(f"**Broker mode:** {row.get('broker_mode', '-')} | Persistence: {row.get('broker_persistence_score', '-')} | Concentration: {row.get('broker_concentration_score', '-')}")
        st.markdown(f"**Why now:** {row['why_now']}")
        st.markdown(f"**Why not yet:** {row['why_not_yet']}")
        st.markdown(f"**Trigger:** {row['trigger']}")
        st.markdown(f"**Invalidator:** {row['invalidator']}")
        st.markdown(f"**Dominant risk:** {row['dominant_risk']}")

elif view == "Intraday / Broker":
    st.subheader("Intraday / Broker Context — Hardening")
    if scan_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        cols = [
            "ticker", "sector", "broker_mode", "broker_alignment_score", "broker_persistence_score", "broker_concentration_score", "dominant_accumulator", "dominant_distributor",
            "institutional_support", "institutional_support_low", "institutional_support_high", "institutional_resistance", "institutional_resistance_low", "institutional_resistance_high", "overhang_score", "float_lock_score", "supply_overhang_score", "gulungan_up_score", "gulungan_down_score",
            "effort_result_up", "effort_result_down", "post_up_followthrough_score", "post_down_followthrough_score", "split_order_score", "bull_trap_score", "bear_trap_score", "latest_event_label", "burst_bias", "bid_stack_quality",
            "offer_stack_quality", "absorption_after_up_score", "absorption_after_down_score", "offer_refill_rate", "bid_refill_rate", "fake_wall_offer_score", "fake_wall_bid_score", "tension_score", "fragility_score",
            "data_completeness_score"
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
                "persisted": audit.get("persist_info", {}),
                "cache_mode": audit.get("cache_mode"),
                "cache_prices_rows": audit.get("cache_prices_rows"),
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
