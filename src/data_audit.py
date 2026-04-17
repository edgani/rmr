from __future__ import annotations

from datetime import date

import pandas as pd


def _normalize_dates(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[col], errors="coerce")


def audit_prices(price_df: pd.DataFrame, attempted_tickers: int = 0, failed_tickers: list[str] | None = None, batch_reports: list[dict] | None = None) -> dict:
    failed_tickers = failed_tickers or []
    batch_reports = batch_reports or []
    dates = _normalize_dates(price_df, "date")
    stale_warning = None
    if not dates.empty and dates.notna().any():
        last_dt = dates.max().date()
        days_stale = (date.today() - last_dt).days
        if days_stale > 5:
            stale_warning = f"Price data looks stale by {days_stale} days."
    return {
        "ticker_count_loaded": int(price_df["ticker"].nunique()) if price_df is not None and not price_df.empty else 0,
        "price_rows": int(len(price_df)) if price_df is not None else 0,
        "attempted_tickers": int(attempted_tickers),
        "failed_ticker_count": int(len(failed_tickers)),
        "failed_tickers": failed_tickers,
        "failed_tickers_preview": ", ".join((failed_tickers or [])[:25]),
        "batch_reports": batch_reports,
        "price_date_min": str(dates.min().date()) if not dates.empty and dates.notna().any() else None,
        "price_date_max": str(dates.max().date()) if not dates.empty and dates.notna().any() else None,
        "stale_warning": stale_warning,
    }


def audit_broker_summary(broker_df: pd.DataFrame) -> dict:
    if broker_df is None or broker_df.empty:
        return {"broker_rows": 0, "broker_count_loaded": 0, "broker_columns_ok": False}
    cols_ok = set(["date", "ticker", "broker_code", "buy_lot", "buy_value", "sell_lot", "sell_value"]).issubset(broker_df.columns)
    return {
        "broker_rows": int(len(broker_df)),
        "broker_count_loaded": int(broker_df["broker_code"].astype(str).nunique()) if "broker_code" in broker_df.columns else 0,
        "broker_columns_ok": bool(cols_ok),
    }


def audit_done_detail(done_df: pd.DataFrame) -> dict:
    if done_df is None or done_df.empty:
        return {"done_rows": 0, "done_columns_ok": False}
    cols_ok = set(["timestamp", "ticker", "price", "lot"]).issubset(done_df.columns)
    return {"done_rows": int(len(done_df)), "done_columns_ok": bool(cols_ok)}


def audit_orderbook(orderbook_df: pd.DataFrame) -> dict:
    if orderbook_df is None or orderbook_df.empty:
        return {"orderbook_rows": 0, "orderbook_columns_ok": False}
    cols_ok = set(["timestamp", "ticker"]).issubset(orderbook_df.columns)
    return {"orderbook_rows": int(len(orderbook_df)), "orderbook_columns_ok": bool(cols_ok)}


def merge_audits(price_audit: dict, broker_audit: dict, done_audit: dict, orderbook_audit: dict, universe_source: str, universe_warnings: list[str]) -> dict:
    source_mode = "REAL_PRICES_ONLY"
    if price_audit.get("ticker_count_loaded", 0) == 0:
        source_mode = "NO_PRICE_DATA"
    elif broker_audit.get("broker_rows", 0) > 0 and done_audit.get("done_rows", 0) > 0 and orderbook_audit.get("orderbook_rows", 0) > 0:
        source_mode = "REAL_PRICES_BROKER_INTRADAY_ORDERBOOK"
    elif broker_audit.get("broker_rows", 0) > 0 and (done_audit.get("done_rows", 0) > 0 or orderbook_audit.get("orderbook_rows", 0) > 0):
        source_mode = "REAL_PRICES_BROKER_PLUS_PARTIAL_INTRADAY"
    elif broker_audit.get("broker_rows", 0) > 0:
        source_mode = "REAL_PRICES_BROKER"
    elif done_audit.get("done_rows", 0) > 0 or orderbook_audit.get("orderbook_rows", 0) > 0:
        source_mode = "REAL_PRICES_PARTIAL_INTRADAY"

    out = {
        **price_audit,
        **broker_audit,
        **done_audit,
        **orderbook_audit,
        "source_mode": source_mode,
        "universe_source": universe_source,
        "universe_warnings": universe_warnings,
    }
    return out
