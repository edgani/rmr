
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.data_io import (
    load_universe_csv,
    fetch_prices_yfinance,
    build_price_health_report,
    load_optional_csv,
)
from src.scanner import compute_eod_scanner
from src.nextplay_overlay import (
    compute_macro_overlay,
    derive_route_state,
    build_asset_translation,
    build_upcoming_events,
    match_analog,
    build_most_hated_rally_monitor,
    build_top_drivers_now,
    build_forward_radar,
    merge_overlay_into_scan,
)

st.set_page_config(page_title="IDX EOD Scanner V4.3 — Next Play Overlay", layout="wide")

st.title("IDX EOD Scanner V4.3 — Next Play Overlay")
st.caption("Deploy-safe single-file app with next-play overlay inspired by the uploaded MacroRegime app.py.")

with st.sidebar:
    st.header("Settings")
    universe_path = st.text_input("Universe CSV", value="data/idx_universe_sample.csv")
    start = st.date_input("Start date", value=pd.Timestamp.today().normalize() - pd.Timedelta(days=365))
    max_tickers = st.number_input("Max tickers (0 = all)", min_value=0, max_value=5000, value=25, step=25)
    min_avg_value = st.number_input("Min avg traded value", min_value=0.0, value=5_000_000_000.0, step=1_000_000_000.0)
    use_cache = st.checkbox("Use local cache if present", value=True)
    run_scan = st.button("Run scan", type="primary")

    st.markdown("---")
    st.subheader("Optional uploads")
    broker_file = st.file_uploader("Broker summary CSV", type=["csv"])
    done_file = st.file_uploader("Done detail CSV", type=["csv"])
    book_file = st.file_uploader("Orderbook CSV", type=["csv"])

def _load_inputs():
    tickers_df = load_universe_csv(universe_path)
    if max_tickers > 0:
        tickers_df = tickers_df.head(max_tickers)
    tickers = tickers_df["ticker"].astype(str).tolist()
    price_df, failed = fetch_prices_yfinance(tickers, pd.Timestamp(start).strftime("%Y-%m-%d"), use_cache=use_cache)
    price_health = build_price_health_report(price_df, tickers)

    broker_df = load_optional_csv(broker_file)
    done_df = load_optional_csv(done_file)
    book_df = load_optional_csv(book_file)
    return tickers_df, price_df, failed, price_health, broker_df, done_df, book_df

if run_scan:
    tickers_df, price_df, failed, price_health, broker_df, done_df, book_df = _load_inputs()

    if price_df.empty:
        st.error("No prices loaded.")
        st.stop()

    scan_df = compute_eod_scanner(price_df, tickers_df, min_avg_value=min_avg_value)
    overlay = compute_macro_overlay(price_df, scan_df)
    route = derive_route_state(overlay)
    analog = match_analog(overlay)
    most_hated = build_most_hated_rally_monitor(overlay, route)
    asset_translation = build_asset_translation(route["route_state"], overlay, route)
    events = build_upcoming_events()
    drivers = build_top_drivers_now(overlay, route, most_hated, analog)
    radar = build_forward_radar(scan_df, overlay, route, top_n=20)
    final_df = merge_overlay_into_scan(scan_df, route, most_hated, analog, radar, events)

    st.session_state["scan_df"] = final_df
    st.session_state["overlay"] = overlay
    st.session_state["route"] = route
    st.session_state["analog"] = analog
    st.session_state["most_hated"] = most_hated
    st.session_state["asset_translation"] = asset_translation
    st.session_state["events"] = events
    st.session_state["drivers"] = drivers
    st.session_state["radar"] = radar
    st.session_state["audit"] = {
        "price_health": price_health,
        "failed_tickers": failed,
        "broker_rows": 0 if broker_df.empty else len(broker_df),
        "done_rows": 0 if done_df.empty else len(done_df),
        "orderbook_rows": 0 if book_df.empty else len(book_df),
    }

if "scan_df" not in st.session_state:
    st.info("Run scan first.")
    st.stop()

scan_df = st.session_state["scan_df"]
overlay = st.session_state["overlay"]
route = st.session_state["route"]
analog = st.session_state["analog"]
most_hated = st.session_state["most_hated"]
asset_translation = st.session_state["asset_translation"]
events = st.session_state["events"]
drivers = st.session_state["drivers"]
radar = st.session_state["radar"]
audit = st.session_state["audit"]

tab1, tab2, tab3, tab4 = st.tabs(["Scanner", "Next Play Overlay", "Forward Radar", "Data Audit"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", route["route_state"])
    c2.metric("Bias", route["route_bias"])
    c3.metric("Position cap", route["position_cap_label"])
    c4.metric("Confidence", f'{route["route_confidence"]:.0%}')

    view_cols = [
        "ticker", "sector", "close", "verdict", "phase", "trend_quality",
        "breakout_integrity", "false_breakout_risk", "dry_score", "wet_score",
        "long_rank_score", "risk_rank_score", "route_fit", "next_play_score",
        "why_now", "trigger", "invalidator"
    ]
    present = [c for c in view_cols if c in scan_df.columns]
    st.dataframe(scan_df[present], use_container_width=True, hide_index=True, height=560)
    st.download_button(
        "Download scanner CSV",
        data=scan_df.to_csv(index=False).encode("utf-8"),
        file_name="idx_eod_scanner_v43.csv",
        mime="text/csv",
    )

with tab2:
    a, b, c = st.columns([1,1,1])
    with a:
        st.subheader("Route")
        st.write(f'**Primary:** {route["route_state"]}')
        st.write(f'**Alt:** {route["alt_route"]}')
        st.write(f'**Invalidator:** {route["invalidator_route"]}')
        st.write(f'**Long allowed:** {route["long_allowed"]}')
        st.write(f'**Short allowed:** {route["short_allowed"]}')
        st.write(f'**Execution mode:** {route["execution_mode"]}')
    with b:
        st.subheader("Analog")
        st.write(f'**Label:** {analog["label"]}')
        st.write(f'**Family:** {analog["scenario_family"]}')
        st.write(f'**Similarity:** {analog["similarity"]:.0%}')
        st.write(f'**1M path:** {analog["path_1m"]}')
        st.write(f'**3M path:** {analog["path_3m"]}')
        st.write(f'**6M path:** {analog["path_6m"]}')
    with c:
        st.subheader("Most hated rally")
        st.write(f'**State:** {most_hated["state"]}')
        st.write(f'**Clear count:** {most_hated["clear_count"]}/4')
        st.write(f'**Relief score:** {most_hated["relief_squeeze_score"]:.0%}')
        st.write(f'**Risk-on switch:** {most_hated["risk_on_switch"]}')

    st.markdown("---")
    st.subheader("Top drivers now")
    if drivers:
        st.dataframe(pd.DataFrame(drivers), use_container_width=True, hide_index=True)
    else:
        st.info("No top drivers computed.")

    st.markdown("---")
    st.subheader("Asset translation")
    mcols = st.columns(len(asset_translation))
    for col, (market, setups) in zip(mcols, asset_translation.items()):
        with col:
            st.markdown(f"**{market}**")
            for s in setups[:3]:
                st.write(f'- **{s["bias"]}** | {s["setup"]}')
                st.caption(f'{s["why"]} Trigger: {s["trigger"]} | Invalidator: {s["invalidator"]}')

with tab3:
    st.subheader("Forward radar")
    if radar:
        st.dataframe(pd.DataFrame(radar), use_container_width=True, hide_index=True, height=420)
    else:
        st.info("No forward radar names.")
    st.markdown("---")
    st.subheader("Upcoming events")
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True, height=320)

with tab4:
    st.subheader("Data health")
    st.json(audit["price_health"], expanded=False)
    st.write(f'Failed tickers: {", ".join(audit["failed_tickers"][:40]) or "None"}')
    st.write(f'Broker rows: {audit["broker_rows"]}')
    st.write(f'Done rows: {audit["done_rows"]}')
    st.write(f'Orderbook rows: {audit["orderbook_rows"]}')
