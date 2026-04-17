from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

OFFICIAL_IDX_STOCK_LIST_URLS = [
    "https://www.idx.co.id/en/market-data/stocks-data/stock-list",
    "https://www.idx.co.id/data-pasar/data-saham/daftar-saham",
]

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/IDX_Composite",
    "https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Indonesia_Stock_Exchange",
]


def normalize_ticker_list(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ticker" not in out.columns:
        # try first column
        out = out.rename(columns={out.columns[0]: "ticker"})
    out["ticker"] = (
        out["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".JK", "", regex=False)
    )
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{2,8}$", na=False)]
    out = out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    return out[["ticker"]]


def _read_local_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return normalize_ticker_list(df)


def _scrape_idx_official(timeout: int = 20) -> Optional[pd.DataFrame]:
    for url in OFFICIAL_IDX_STOCK_LIST_URLS:
        try:
            html = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).text
            tables = pd.read_html(StringIO(html))
            found = []
            for tbl in tables:
                cols = [str(c).strip().lower() for c in tbl.columns]
                if "code" in cols:
                    col = tbl.columns[cols.index("code")]
                    found.append(pd.DataFrame({"ticker": tbl[col]}))
                elif "ticker code" in cols:
                    col = tbl.columns[cols.index("ticker code")]
                    found.append(pd.DataFrame({"ticker": tbl[col]}))
                elif "ticker" in cols:
                    col = tbl.columns[cols.index("ticker")]
                    found.append(pd.DataFrame({"ticker": tbl[col]}))
            if found:
                out = normalize_ticker_list(pd.concat(found, ignore_index=True))
                if len(out) >= 300:
                    return out
        except Exception:
            continue
    return None


def _scrape_wikipedia(timeout: int = 20) -> Optional[pd.DataFrame]:
    for url in WIKI_URLS:
        try:
            html = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).text
            tables = pd.read_html(StringIO(html))
            found = []
            for tbl in tables:
                cols = [str(c).strip().lower() for c in tbl.columns]
                candidates = ["ticker", "code", "trading code", "stock code", "symbol"]
                for cand in candidates:
                    if cand in cols:
                        col = tbl.columns[cols.index(cand)]
                        found.append(pd.DataFrame({"ticker": tbl[col]}))
                        break
            if found:
                out = normalize_ticker_list(pd.concat(found, ignore_index=True))
                if len(out) >= 100:
                    return out
        except Exception:
            continue
    return None


def resolve_universe_source(
    mode: str,
    base_dir: str | Path = ".",
    uploaded_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str, list[str]]:
    """
    Returns (tickers_df, source_label, warnings)
    """
    base_dir = Path(base_dir)
    warnings: list[str] = []

    if uploaded_df is not None and not uploaded_df.empty:
        return normalize_ticker_list(uploaded_df), "uploaded_csv", warnings

    if mode == "sample":
        sample = _read_local_csv(base_dir / "data" / "idx_universe_sample.csv")
        if sample is None or sample.empty:
            raise FileNotFoundError("Sample universe file missing.")
        warnings.append("Using sample universe only. This is not full IHSG.")
        return sample, "sample", warnings

    # Recommended stable local full CSV
    full_local = _read_local_csv(base_dir / "data" / "idx_universe_full.csv")
    if full_local is not None and not full_local.empty:
        return full_local, "local_full_csv", warnings

    if mode in {"full", "auto"}:
        idx_df = _scrape_idx_official()
        if idx_df is not None and not idx_df.empty:
            return idx_df, "official_idx_web", warnings

        wiki_df = _scrape_wikipedia()
        if wiki_df is not None and not wiki_df.empty:
            warnings.append("Universe loaded from Wikipedia/public web fallback. Validate coverage.")
            return wiki_df, "wikipedia_fallback", warnings

    sample = _read_local_csv(base_dir / "data" / "idx_universe_sample.csv")
    if sample is None or sample.empty:
        raise FileNotFoundError("No universe source available.")
    warnings.append("Fell back to sample universe. Add data/idx_universe_full.csv for stable full coverage.")
    return sample, "fallback_sample", warnings
