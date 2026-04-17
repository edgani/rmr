
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
from src.validation import build_price_side_validation_panel, run_walk_forward_validation

st.set_page_config(page_title="IDX Scanner V5.0", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

BUCKET_ORDER = [
    "SIAP NAIK SEKARANG",
    "HAMPIR SIAP — TUNGGU TRIGGER",
    "SUDAH NAIK — TUNGGU PULLBACK / SIGNAL BERIKUTNYA",
    "WATCH REBOUND",
    "JANGAN SENTUH DULU",
    "AVOID / BUANG",
]
BUCKET_HELP = {
    "SIAP NAIK SEKARANG": "Setup sudah paling rapi. Fokus kandidat entry terdekat.",
    "HAMPIR SIAP — TUNGGU TRIGGER": "Belum masuk sekarang. Tunggu trigger yang jelas.",
    "SUDAH NAIK — TUNGGU PULLBACK / SIGNAL BERIKUTNYA": "Arah masih oke, tapi entry sekarang rawan telat.",
    "WATCH REBOUND": "Pantulan taktis, bukan trend sehat utama.",
    "JANGAN SENTUH DULU": "Belum ada edge jelas. Simpan di radar saja.",
    "AVOID / BUANG": "Risiko distribusi / breakdown lebih dominan.",
}


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


def num(v, default=0.0) -> float:
    try:
        return float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return float(default)


def txt(v, default="-") -> str:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    s = str(v).strip()
    return s if s else default


def confidence_label(x: float) -> str:
    if x >= 80:
        return "tinggi"
    if x >= 60:
        return "sedang"
    return "rendah"


def micro_label(row: pd.Series, micro_available: bool) -> str:
    if not micro_available:
        return "Belum dinilai (price only)"
    burst_bias = txt(row.get("burst_bias"), "NEUTRAL")
    up = num(row.get("gulungan_up_score"))
    down = num(row.get("gulungan_down_score"))
    bull_trap = num(row.get("bull_trap_score"), 50)
    bear_trap = num(row.get("bear_trap_score"), 50)
    absorb_up = num(row.get("absorption_after_up_score"), 50)
    absorb_down = num(row.get("absorption_after_down_score"), 50)
    offer_q = num(row.get("offer_stack_quality"), 50)
    bid_q = num(row.get("bid_stack_quality"), 50)
    if burst_bias == "BULLISH" and up >= max(55, down) and bull_trap < 45 and absorb_up < 45 and offer_q >= 50:
        return "Bagus & mendukung naik"
    if burst_bias == "BULLISH" and bull_trap < 60:
        return "Mulai membaik"
    if burst_bias == "BEARISH" and down >= max(55, up) and bear_trap < 45 and absorb_down < 45 and bid_q < 50:
        return "Jelek / tekanan jual"
    return "Campuran / belum jelas"


def classify_bucket(row: pd.Series, micro_available: bool) -> tuple[str, str]:
    verdict = txt(row.get("verdict"), "NEUTRAL")
    conf = num(row.get("score_confidence"))
    trend = num(row.get("trend_quality"))
    breakout = num(row.get("breakout_integrity"))
    false_break = num(row.get("false_breakout_risk"))
    dry = num(row.get("dry_score_final"), num(row.get("dry_score")))
    wet = num(row.get("wet_score_final"), num(row.get("wet_score")))
    route_fit = num(row.get("route_fit_score"), num(row.get("route_rank_score")))
    rs20 = num(row.get("relative_strength_20d"))
    risk_rank = num(row.get("risk_rank_score"))
    long_rank = num(row.get("long_rank_score"))
    burst_bias = txt(row.get("burst_bias"), "NEUTRAL")
    bull_trap = num(row.get("bull_trap_score"), 50)
    bear_trap = num(row.get("bear_trap_score"), 50)
    absorb_up = num(row.get("absorption_after_up_score"), 50)

    if verdict in {"AVOID", "TRIM"} or risk_rank >= 75 or wet - dry >= 20:
        return "AVOID / BUANG", "Struktur/risiko masih jelek"
    if verdict == "WATCH_REBOUND":
        return "WATCH REBOUND", "Pantulan ada, tapi belum trend sehat utama"
    if verdict == "READY_LONG":
        if (rs20 > 0.08 and risk_rank >= 55) or false_break >= 45:
            return "SUDAH NAIK — TUNGGU PULLBACK / SIGNAL BERIKUTNYA", "Arah masih oke tapi entry sekarang rawan telat"
        if micro_available and burst_bias == "BULLISH" and bull_trap < 50 and absorb_up < 50:
            return "SIAP NAIK SEKARANG", "Trigger + microstructure mendukung"
        return "SIAP NAIK SEKARANG", "Setup sudah paling rapi sekarang"
    if verdict == "WATCH":
        if breakout >= 45 or route_fit >= 55 or long_rank >= 55:
            return "HAMPIR SIAP — TUNGGU TRIGGER", "Setup bagus tapi trigger belum lengkap"
        return "JANGAN SENTUH DULU", "Masih watchlist, belum ada edge kuat"
    if verdict == "NEUTRAL":
        if trend >= 60 and dry >= wet and (rs20 > 0.06 or long_rank >= 60):
            return "SUDAH NAIK — TUNGGU PULLBACK / SIGNAL BERIKUTNYA", "Arah cukup sehat tapi tunggu entry lebih rapi"
        if breakout >= 40 or route_fit >= 50:
            return "HAMPIR SIAP — TUNGGU TRIGGER", "Sudah mendekat ke trigger"
        return "JANGAN SENTUH DULU", "Belum ada alasan kuat buat masuk"
    if verdict == "ILLIQUID":
        return "AVOID / BUANG", "Likuiditas terlalu tipis"
    if micro_available and burst_bias == "BEARISH" and bear_trap < 50:
        return "AVOID / BUANG", "Tekanan jual masih dominan"
    if conf < 45:
        return "JANGAN SENTUH DULU", "Confidence terlalu rendah"
    return "JANGAN SENTUH DULU", "Belum cukup jelas"


def timing_text(bucket: str) -> str:
    return {
        "SIAP NAIK SEKARANG": "Boleh cicil / entry sekarang",
        "HAMPIR SIAP — TUNGGU TRIGGER": "Tunggu break / reclaim",
        "SUDAH NAIK — TUNGGU PULLBACK / SIGNAL BERIKUTNYA": "Tunggu pullback / retest",
        "WATCH REBOUND": "Taktikal saja, jangan agresif",
        "JANGAN SENTUH DULU": "Simpan di radar",
        "AVOID / BUANG": "Hindari dulu",
    }.get(bucket, "-")


def build_simple_table(scan_df: pd.DataFrame, audit: dict | None) -> pd.DataFrame:
    if scan_df.empty:
        return scan_df
    micro_available = bool((audit or {}).get("done_rows", 0) > 0 or (audit or {}).get("orderbook_rows", 0) > 0)
    out = scan_df.copy()
    statuses = out.apply(lambda r: classify_bucket(r, micro_available), axis=1)
    out["status_awam"] = [s[0] for s in statuses]
    out["alasan_singkat"] = [s[1] for s in statuses]
    out["timing"] = out["status_awam"].map(timing_text)
    out["bid_offer_state"] = out.apply(lambda r: micro_label(r, micro_available), axis=1)
    out["confidence_text"] = out.get("score_confidence", 0).apply(lambda x: confidence_label(num(x)))
    out["trigger_singkat"] = out.get("trigger", "-").astype(str).str.slice(0, 120)
    out["invalidator_singkat"] = out.get("invalidator", "-").astype(str).str.slice(0, 120)
    out["why_now_short"] = out.get("why_now", "-").astype(str).str.slice(0, 120)
    out["entry_score"] = (
        pd.to_numeric(out.get("long_rank_score", 0), errors="coerce").fillna(0.0) * 0.45
        + pd.to_numeric(out.get("route_rank_score", 0), errors="coerce").fillna(0.0) * 0.25
        + pd.to_numeric(out.get("score_confidence", 0), errors="coerce").fillna(0.0) * 0.20
        - pd.to_numeric(out.get("risk_rank_score", 0), errors="coerce").fillna(0.0) * 0.15
    ).round(2)
    order = {name: i for i, name in enumerate(BUCKET_ORDER)}
    out["bucket_order"] = out["status_awam"].map(order).fillna(999)
    out = out.sort_values(["bucket_order", "entry_score", "score_confidence"], ascending=[True, False, False]).reset_index(drop=True)
    return out


st.title("IDX Scanner V5.0 — Action View")
st.caption("Versi yang lebih gampang dibaca: fokus ke action bucket, trigger, invalidator, dan timing. Kolom debug dipindah ke advanced view.")
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

    st.subheader("Run")
    run_scan = st.button("Run scanner", type="primary")
    run_validation = st.button("Run validation")

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
    st.session_state["simple_df"] = build_simple_table(scan_df, audit)

price_df = st.session_state.get("price_df", pd.DataFrame())
scan_df = st.session_state.get("scan_df", pd.DataFrame())
audit = st.session_state.get("audit", None)
universe_df = st.session_state.get("universe_df", pd.DataFrame())
simple_df = st.session_state.get("simple_df", pd.DataFrame())
validation_result = st.session_state.get("validation_result", None)
validation_panel = st.session_state.get("validation_panel", pd.DataFrame())

if run_validation:
    if price_df.empty:
        st.warning("Run scanner dulu supaya price history ada.")
    else:
        with st.spinner("Building validation panel..."):
            panel = build_price_side_validation_panel(price_df, min_history=80, horizon=20)
            if panel.empty:
                st.warning("Validation panel kosong. Data belum cukup panjang untuk walk-forward.")
            else:
                score_col = "long_rank_score" if (isinstance(scan_df, pd.DataFrame) and "long_rank_score" in scan_df.columns) else None
                if score_col is None:
                    st.warning("long_rank_score belum tersedia.")
                else:
                    val = run_walk_forward_validation(panel, score_col=score_col, label_col="label_long_success", return_col="fwd_return", date_col="date", train_days=252*2, test_days=63, top_n=20)
                    st.session_state["validation_panel"] = panel
                    st.session_state["validation_result"] = val
                    validation_result = val
                    validation_panel = panel

if audit is not None:
    micro_live = audit.get("done_rows", 0) > 0 or audit.get("orderbook_rows", 0) > 0
    cols = st.columns(8)
    cols[0].metric("Tickers scanned", audit.get("ticker_count_loaded", 0))
    cols[1].metric("Ready sekarang", int((simple_df["status_awam"] == "SIAP NAIK SEKARANG").sum()) if isinstance(simple_df, pd.DataFrame) and not simple_df.empty else 0)
    cols[2].metric("Hampir siap", int((simple_df["status_awam"] == "HAMPIR SIAP — TUNGGU TRIGGER").sum()) if isinstance(simple_df, pd.DataFrame) and not simple_df.empty else 0)
    cols[3].metric("Tunggu pullback", int((simple_df["status_awam"] == "SUDAH NAIK — TUNGGU PULLBACK / SIGNAL BERIKUTNYA").sum()) if isinstance(simple_df, pd.DataFrame) and not simple_df.empty else 0)
    cols[4].metric("Watch rebound", int((simple_df["status_awam"] == "WATCH REBOUND").sum()) if isinstance(simple_df, pd.DataFrame) and not simple_df.empty else 0)
    cols[5].metric("Avoid / buang", int((simple_df["status_awam"] == "AVOID / BUANG").sum()) if isinstance(simple_df, pd.DataFrame) and not simple_df.empty else 0)
    cols[6].metric("Source mode", audit.get("source_mode", "n/a"))
    cols[7].metric("Bid-offer", "AKTIF" if micro_live else "BELUM AKTIF")
    if audit.get("stale_warning"):
        st.warning(audit["stale_warning"])
    for w in audit.get("universe_warnings", []) or []:
        st.warning(w)
    if audit.get("universe_source") == "fallback_sample":
        st.error("Universe masih sample fallback. Jadi hasil belum full IHSG.")
    if not micro_live:
        st.info("Run ini masih REAL_PRICES_ONLY. Jadi pengelompokan 'bid-offer bagus' belum boleh dianggap final. Yang sekarang jujurly baru grouping readiness price-side.")

view = st.radio("View", ["Action View", "Ticker Detail", "Advanced Table", "Validation", "Data Audit"], horizontal=True)

if view == "Action View":
    st.subheader("Action View — gampang dibaca")
    if simple_df.empty:
        st.info("Belum ada hasil scan. Klik Run scanner.")
    else:
        use_bid_offer = st.checkbox("Urutkan utamakan bid-offer / microstructure (kalau data ada)", value=False)
        work = simple_df.copy()
        if use_bid_offer and audit and (audit.get("done_rows", 0) > 0 or audit.get("orderbook_rows", 0) > 0):
            work = work.sort_values(["status_awam", "bid_offer_state", "entry_score", "score_confidence"], ascending=[True, True, False, False])
        top_bucket = st.selectbox("Fokus bucket", options=["Semua"] + BUCKET_ORDER, index=0)
        if top_bucket != "Semua":
            work = work[work["status_awam"] == top_bucket].copy()
        for bucket in BUCKET_ORDER:
            bucket_df = work[work["status_awam"] == bucket].copy()
            if bucket_df.empty:
                continue
            st.markdown(f"### {bucket}")
            st.caption(BUCKET_HELP[bucket])
            cols = [
                "ticker", "sector", "close", "status_awam", "bid_offer_state", "alasan_singkat", "trigger_singkat",
                "invalidator_singkat", "timing", "confidence_text", "route_primary", "top_catalyst_title"
            ]
            cols = [c for c in cols if c in bucket_df.columns]
            rename_map = {
                "ticker": "Ticker",
                "sector": "Sector",
                "close": "Close",
                "status_awam": "Status",
                "bid_offer_state": "Bid-Offer / Micro",
                "alasan_singkat": "Alasan Singkat",
                "trigger_singkat": "Trigger",
                "invalidator_singkat": "Invalidation",
                "timing": "Timing",
                "confidence_text": "Confidence",
                "route_primary": "Route",
                "top_catalyst_title": "Catalyst",
            }
            show_df = bucket_df[cols].rename(columns=rename_map)
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        with st.expander("Lihat advanced table"):
            adv_cols = [c for c in [
                "ticker","verdict","score_confidence","phase","trend_quality","breakout_integrity","false_breakout_risk",
                "dry_score_final","wet_score_final","broker_alignment_score","broker_mode","dominant_accumulator",
                "dominant_distributor","institutional_support","institutional_resistance","latest_event_label",
                "long_rank_score","risk_rank_score","route_rank_score","why_now","why_not_yet"
            ] if c in work.columns]
            st.dataframe(work[adv_cols], use_container_width=True, hide_index=True)
        st.download_button("Download action table CSV", data=work.to_csv(index=False).encode("utf-8"), file_name="idx_action_view.csv", mime="text/csv")

elif view == "Ticker Detail":
    st.subheader("Ticker Detail")
    if simple_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        picker_df = simple_df[["ticker", "status_awam"]].copy()
        picker_df["label"] = picker_df["ticker"] + " — " + picker_df["status_awam"]
        selected = st.selectbox("Ticker", options=picker_df["label"].tolist())
        ticker = selected.split(" — ")[0]
        row = simple_df.loc[simple_df["ticker"] == ticker].iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Status", row["status_awam"])
        c2.metric("Timing", row.get("timing", "-"))
        c3.metric("Confidence", row.get("confidence_text", "-"))
        c4.metric("Bid-Offer / Micro", row.get("bid_offer_state", "-"))
        c5.metric("Route", row.get("route_primary", "-"))
        c6.metric("Catalyst", txt(row.get("top_catalyst_title"), "-"))
        render_candles(price_df, ticker)
        st.markdown(f"**Alasan singkat:** {row.get('alasan_singkat', '-')}")
        st.markdown(f"**Why now:** {row.get('why_now', '-')}")
        st.markdown(f"**Why not yet:** {row.get('why_not_yet', '-')}")
        st.markdown(f"**Trigger:** {row.get('trigger', '-')}")
        st.markdown(f"**Invalidation:** {row.get('invalidator', '-')}")
        st.markdown(f"**Dominant risk:** {row.get('dominant_risk', '-')}")
        extra_cols = [c for c in [
            "trend_quality","breakout_integrity","false_breakout_risk","dry_score_final","wet_score_final",
            "broker_alignment_score","broker_mode","dominant_accumulator","dominant_distributor",
            "institutional_support","institutional_support_low","institutional_support_high",
            "institutional_resistance","institutional_resistance_low","institutional_resistance_high",
            "gulungan_up_score","gulungan_down_score","effort_result_up","effort_result_down",
            "post_up_followthrough_score","post_down_followthrough_score","bull_trap_score","bear_trap_score",
            "absorption_after_up_score","absorption_after_down_score","bid_stack_quality","offer_stack_quality",
            "score_confidence","data_completeness_score","module_agreement_score","long_rank_score","risk_rank_score","route_rank_score"
        ] if c in row.index]
        st.dataframe(pd.DataFrame([row[extra_cols]]), use_container_width=True, hide_index=True)

elif view == "Advanced Table":
    st.subheader("Advanced Table")
    if scan_df.empty:
        st.info("Belum ada hasil scan.")
    else:
        cols = [c for c in [
            "ticker","sector","verdict","status_awam","score_confidence","phase","market_regime","execution_mode",
            "trend_quality","breakout_integrity","false_breakout_risk","dry_score_final","wet_score_final","drywet_state",
            "broker_alignment_score","broker_persistence_score","broker_mode","dominant_accumulator","dominant_distributor",
            "institutional_support","institutional_resistance","gulungan_up_score","gulungan_down_score","latest_event_label",
            "relative_strength_20d","sector_relative_strength_20d","long_rank_score","risk_rank_score","route_rank_score",
            "route_primary","route_bias","forward_radar_bucket","top_catalyst_title","why_now","why_not_yet","trigger","invalidator","dominant_risk"
        ] if c in simple_df.columns]
        st.dataframe(simple_df[cols], use_container_width=True, hide_index=True)

elif view == "Validation":
    st.subheader("Walk-forward Validation")
    st.caption("Scaffold kejujuran: buat cek apakah ranking kelihatan bagus atau memang lumayan konsisten.")
    if validation_result is None:
        st.info("Klik Run validation setelah scanner jalan.")
    else:
        if validation_result.summary:
            m1, m2, m3 = st.columns(3)
            auc = validation_result.summary.get("auc", float("nan"))
            p20 = validation_result.summary.get("precision_at_20", float("nan"))
            exp20 = validation_result.summary.get("expectancy_at_20", float("nan"))
            m1.metric("Mean AUC", round(auc, 3) if pd.notna(auc) else "—")
            m2.metric("Mean Precision@20", round(p20, 3) if pd.notna(p20) else "—")
            m3.metric("Mean Expectancy@20", round(exp20, 4) if pd.notna(exp20) else "—")
        if not validation_result.fold_metrics.empty:
            st.markdown("### Fold metrics")
            st.dataframe(validation_result.fold_metrics, use_container_width=True, hide_index=True)
        if not validation_result.predictions.empty:
            st.markdown("### Top validation predictions")
            top_val = validation_result.predictions.sort_values(["fold", "long_rank_score"], ascending=[True, False]).groupby("fold").head(20)
            st.dataframe(top_val, use_container_width=True, hide_index=True)
        if isinstance(validation_panel, pd.DataFrame) and not validation_panel.empty:
            st.download_button("Download validation panel CSV", data=validation_panel.to_csv(index=False).encode("utf-8"), file_name="idx_validation_panel.csv", mime="text/csv")

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
