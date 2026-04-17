from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


CACHE_PRICES = "latest_prices.csv"
CACHE_SCAN = "latest_scan.csv"
CACHE_AUDIT = "latest_audit.json"
CACHE_FAILED = "failed_tickers.csv"


def ensure_dirs(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    cache = base / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def read_cached_prices(base_dir: str | Path) -> pd.DataFrame:
    cache = ensure_dirs(base_dir)
    p = cache / CACHE_PRICES
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def read_cached_scan(base_dir: str | Path) -> pd.DataFrame:
    cache = ensure_dirs(base_dir)
    p = cache / CACHE_SCAN
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def merge_cached_and_new_prices(cached: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [cached, new] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    keep_cols = [c for c in ["ticker", "date"] if c in out.columns]
    if keep_cols:
        out = out.drop_duplicates(keep_cols, keep="last")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
    sort_cols = [c for c in ["ticker", "date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def filter_cached_prices_for_universe(cached: pd.DataFrame, tickers: list[str], min_start: str | None = None) -> pd.DataFrame:
    if cached is None or cached.empty:
        return pd.DataFrame()
    out = cached.copy()
    tickers = [str(t).upper().replace(".JK", "") for t in tickers]
    if "ticker" in out.columns:
        out = out[out["ticker"].astype(str).isin(tickers)]
    if min_start and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[out["date"] >= pd.to_datetime(min_start)]
    return out.reset_index(drop=True)


def persist_run_outputs(base_dir: str | Path, prices: pd.DataFrame, scan: pd.DataFrame, audit: dict) -> dict:
    cache = ensure_dirs(base_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    paths = {}
    if prices is not None and not prices.empty:
        p = cache / CACHE_PRICES
        prices.to_csv(p, index=False)
        paths["prices_csv"] = str(p)
    if scan is not None and not scan.empty:
        p = cache / CACHE_SCAN
        scan.to_csv(p, index=False)
        paths["scan_csv"] = str(p)
    if audit is not None:
        p = cache / CACHE_AUDIT
        with open(p, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
        paths["audit_json"] = str(p)
        fcsv = cache / CACHE_FAILED
        failed = audit.get("failed_tickers", []) or []
        pd.DataFrame({"ticker": failed}).to_csv(fcsv, index=False)
        paths["failed_tickers_csv"] = str(fcsv)
    stamp = cache / "last_run.txt"
    stamp.write_text(ts, encoding="utf-8")
    paths["last_run_utc"] = ts
    return paths


def read_last_run(base_dir: str | Path) -> str | None:
    cache = ensure_dirs(base_dir)
    stamp = cache / "last_run.txt"
    return stamp.read_text(encoding="utf-8").strip() if stamp.exists() else None
