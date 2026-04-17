from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable
import time

import pandas as pd
import yfinance as yf


@dataclass
class PriceFetchResult:
    prices: pd.DataFrame
    failed_tickers: list[str]
    attempted_tickers: list[str]
    batch_reports: list[dict] = field(default_factory=list)


def _chunked(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def normalize_yf_download(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    if not isinstance(raw.columns, pd.MultiIndex):
        tmp = raw.reset_index().rename(columns=str.lower)
        tmp["ticker"] = batch[0].replace(".JK", "")
        tmp = tmp.rename(columns={"adj close": "adj_close"})
        return tmp[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]

    frames = []
    for top in raw.columns.levels[0]:
        if top not in raw.columns.get_level_values(0):
            continue
        sub = raw[top].copy()
        if sub.dropna(how="all").empty:
            continue
        sub = sub.reset_index().rename(columns=str.lower)
        sub["ticker"] = str(top).replace(".JK", "")
        sub = sub.rename(columns={"adj close": "adj_close"})
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in sub.columns:
                sub[col] = pd.NA
        frames.append(sub[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    return pd.concat(frames, ignore_index=True)


def fetch_yf_prices_batched(
    tickers: list[str],
    start: str,
    end: str | None = None,
    batch_size: int = 80,
    pause_s: float = 0.0,
) -> PriceFetchResult:
    if end is None:
        end = str(date.today() + timedelta(days=1))

    attempted = list(dict.fromkeys([t.upper().replace(".JK", "") for t in tickers]))
    all_frames = []
    failed: list[str] = []
    batch_reports: list[dict] = []

    for idx, batch in enumerate(_chunked(attempted, max(1, batch_size)), start=1):
        symbols = [f"{t}.JK" for t in batch]
        err = None
        got = set()
        try:
            raw = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            norm = normalize_yf_download(raw, symbols)
            all_frames.append(norm)
            got = set(norm["ticker"].dropna().unique().tolist())
            failed.extend([t for t in batch if t not in got])
        except Exception as e:
            err = str(e)
            failed.extend(batch)
        batch_reports.append({
            "batch_no": idx,
            "requested": len(batch),
            "loaded": len(got),
            "failed": len([t for t in batch if t not in got]),
            "failed_preview": ", ".join([t for t in batch if t not in got][:10]),
            "error": err,
        })
        if pause_s > 0:
            time.sleep(pause_s)

    prices = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    )
    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            prices[c] = pd.to_numeric(prices[c], errors="coerce")
        prices = prices.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    failed = sorted(list(dict.fromkeys(failed)))
    return PriceFetchResult(prices=prices, failed_tickers=failed, attempted_tickers=attempted, batch_reports=batch_reports)


def retry_failed_tickers(
    failed_tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    if not failed_tickers:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    result = fetch_yf_prices_batched(failed_tickers, start=start, end=end, batch_size=1)
    return result.prices


def build_price_health_report(prices: pd.DataFrame, attempted_tickers: list[str], failed_tickers: list[str]) -> dict:
    if prices.empty:
        return {
            "loaded_tickers": 0,
            "rows": 0,
            "date_min": None,
            "date_max": None,
            "failed_tickers": failed_tickers,
            "attempted_tickers": len(attempted_tickers),
        }
    return {
        "loaded_tickers": int(prices["ticker"].nunique()),
        "rows": int(len(prices)),
        "date_min": str(pd.to_datetime(prices["date"]).min().date()),
        "date_max": str(pd.to_datetime(prices["date"]).max().date()),
        "failed_tickers": failed_tickers,
        "attempted_tickers": len(attempted_tickers),
    }
