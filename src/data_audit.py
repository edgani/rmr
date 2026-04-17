from __future__ import annotations

from datetime import date

import pandas as pd


def _normalize_dates(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[col], errors="coerce")


def audit_prices(price_df: pd.DataFrame, attempted_tickers: int = 0, failed_tickers: list[str] | None = None) -> dict:
    failed_tickers = failed_tickers or []
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
        "failed_tickers_preview": ", ".join((failed_tickers or [])[:25]),
        "price_date_min": str(dates.min().date()) if not dates.empty and dates.notna().any() else None,
        "price_date_max": str(dates.max().date()) if not dates.empty and dates.notna().any() else None,
        "stale_warning": stale_warning,
    }


def audit_broker_summary(broker_df: pd.DataFrame) -> dict:
    if broker_df is None or broker_df.empty:
        return {
            "broker_rows": 0,
            "broker_count_loaded": 0,
            "broker_ticker_count": 0,
            "broker_columns_ok": False,
        }
    required = {"date", "ticker", "broker_code", "buy_lot", "buy_value", "sell_lot", "sell_value"}
    return {
        "broker_rows": int(len(broker_df)),
        "broker_count_loaded": int(broker_df["broker_code"].astype(str).nunique()) if "broker_code" in broker_df.columns else 0,
        "broker_ticker_count": int(broker_df["ticker"].astype(str).nunique()) if "ticker" in broker_df.columns else 0,
        "broker_columns_ok": required.issubset(set(broker_df.columns)),
    }


def audit_done_detail(done_df: pd.DataFrame) -> dict:
    if done_df is None or done_df.empty:
        return {"done_rows": 0, "done_tickers": 0, "done_columns_ok": False}
    required = {"timestamp", "ticker", "price", "lot"}
    return {
        "done_rows": int(len(done_df)),
        "done_tickers": int(done_df["ticker"].astype(str).nunique()) if "ticker" in done_df.columns else 0,
        "done_columns_ok": required.issubset(set(done_df.columns)),
    }


def audit_orderbook(book_df: pd.DataFrame) -> dict:
    if book_df is None or book_df.empty:
        return {"orderbook_rows": 0, "orderbook_tickers": 0, "orderbook_columns_ok": False}
    required = {"timestamp", "ticker", "bid_1_price", "bid_1_lot", "offer_1_price", "offer_1_lot"}
    return {
        "orderbook_rows": int(len(book_df)),
        "orderbook_tickers": int(book_df["ticker"].astype(str).nunique()) if "ticker" in book_df.columns else 0,
        "orderbook_columns_ok": required.issubset(set(book_df.columns)),
    }


def build_source_mode(prices_ok: bool, broker_ok: bool, intraday_ok: bool) -> str:
    if prices_ok and broker_ok and intraday_ok:
        return "real_prices_plus_broker_plus_intraday"
    if prices_ok and broker_ok:
        return "real_prices_plus_broker"
    if prices_ok:
        return "real_prices_only"
    return "demo_or_missing"


def merge_audits(price_audit: dict, broker_audit: dict, done_audit: dict, book_audit: dict, universe_source: str, universe_warnings: list[str]) -> dict:
    prices_ok = price_audit.get("ticker_count_loaded", 0) > 0
    broker_ok = broker_audit.get("broker_rows", 0) > 0 and broker_audit.get("broker_columns_ok", False)
    intraday_ok = (
        done_audit.get("done_rows", 0) > 0 and done_audit.get("done_columns_ok", False)
    ) or (
        book_audit.get("orderbook_rows", 0) > 0 and book_audit.get("orderbook_columns_ok", False)
    )
    out = {}
    out.update(price_audit)
    out.update(broker_audit)
    out.update(done_audit)
    out.update(book_audit)
    out["source_mode"] = build_source_mode(prices_ok, broker_ok, intraday_ok)
    out["universe_source"] = universe_source
    out["universe_warnings"] = universe_warnings
    out["fallback_warning"] = "sample" in universe_source
    return out
