
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import pandas as pd

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_universe_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sample = pd.DataFrame(
            {"ticker": ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "ANTM", "ICBP", "INDF", "MDKA", "ADRO", "AMRT"],
             "sector": ["Banks","Banks","Banks","Telecom","Auto","Metals","Consumer","Consumer","Metals","Energy","Retail"]}
        )
        return sample
    df = pd.read_csv(p)
    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must contain 'ticker' column")
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
    return df[["ticker","sector"]].drop_duplicates().reset_index(drop=True)

def _normalize_yf(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows = []
    if isinstance(df.columns, pd.MultiIndex):
        # expects level-0 OHLCV, level-1 ticker
        level0 = set(df.columns.get_level_values(0))
        for t in tickers:
            if ("Close", f"{t}.JK") not in df.columns:
                continue
            tmp = pd.DataFrame(index=df.index)
            for field in ["Open","High","Low","Close","Volume"]:
                col = (field, f"{t}.JK")
                if col in df.columns:
                    tmp[field.lower()] = pd.to_numeric(df[col], errors="coerce")
            tmp = tmp.dropna(subset=["close"])
            if tmp.empty:
                continue
            tmp = tmp.reset_index().rename(columns={"Date":"date", "index":"date"})
            tmp["ticker"] = t
            rows.append(tmp)
    else:
        # single ticker path
        tmp = df.copy().reset_index().rename(columns={"Date":"date","index":"date"})
        tmp.columns = [str(c).lower() for c in tmp.columns]
        if "close" in tmp.columns:
            tmp["ticker"] = tickers[0]
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out

def fetch_prices_yfinance(tickers: List[str], start: str, use_cache: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    cache_path = CACHE_DIR / "latest_prices.csv"
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            loaded = set(cached["ticker"].astype(str).unique().tolist())
            if loaded.issuperset(set(tickers)):
                return cached[cached["ticker"].isin(tickers)].copy(), []
        except Exception:
            pass

    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame(), tickers

    yf_tickers = [f"{t}.JK" for t in tickers]
    failed = []
    all_rows = []
    batch = 50
    for i in range(0, len(yf_tickers), batch):
        chunk = yf_tickers[i:i+batch]
        try:
            raw = yf.download(chunk, start=start, auto_adjust=False, progress=False, threads=True, group_by="column")
            norm = _normalize_yf(raw, [t.replace(".JK", "") for t in chunk])
            if not norm.empty:
                all_rows.append(norm)
        except Exception:
            failed.extend([t.replace(".JK", "") for t in chunk])

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not out.empty:
        out.to_csv(cache_path, index=False)
        seen = set(out["ticker"].unique())
        failed.extend([t for t in tickers if t not in seen])
    return out, sorted(set(failed))

def build_price_health_report(price_df: pd.DataFrame, expected_tickers: List[str]) -> dict:
    loaded = sorted(price_df["ticker"].unique().tolist()) if not price_df.empty else []
    return {
        "expected_ticker_count": len(expected_tickers),
        "loaded_ticker_count": len(loaded),
        "coverage": round(len(loaded) / max(len(expected_tickers), 1), 4),
        "failed_ticker_count": len([t for t in expected_tickers if t not in loaded]),
        "loaded_preview": loaded[:20],
        "date_min": None if price_df.empty else str(price_df["date"].min().date()),
        "date_max": None if price_df.empty else str(price_df["date"].max().date()),
    }

def load_optional_csv(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded)
    except Exception:
        return pd.DataFrame()
