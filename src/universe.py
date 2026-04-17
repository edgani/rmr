from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional, Tuple
import re

import pandas as pd
import requests

OFFICIAL_IDX_STOCK_LIST_URLS = [
    "https://www.idx.co.id/en/market-data/stocks-data/stock-list",
    "https://www.idx.co.id/data-pasar/data-saham/daftar-saham",
]

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/IDX_Composite",
    "https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Indonesia_Stock_Exchange",
    "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia",
]

CACHE_UNIVERSE = Path("data/cache/latest_universe.csv")


def normalize_ticker_list(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lowered_map = {str(c).lower().strip(): c for c in out.columns}
    if "ticker" not in out.columns:
        for cand in ["ticker", "code", "symbol", "stock code", "ticker code"]:
            if cand in lowered_map:
                out = out.rename(columns={lowered_map[cand]: "ticker"})
                break
    if "ticker" not in out.columns:
        out = out.rename(columns={out.columns[0]: "ticker"})
    keep = [c for c in out.columns if str(c).lower().strip() in {"ticker", "sector", "board", "name", "listing_date"}]
    if "ticker" not in keep:
        keep = ["ticker"] + keep
    out = out[keep].copy()
    out["ticker"] = (
        out["ticker"].astype(str).str.upper().str.strip().str.replace(".JK", "", regex=False)
    )
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{2,8}$", na=False)]
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def _read_local_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return normalize_ticker_list(pd.read_csv(path))


def _extract_tables(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html))
    except Exception:
        return []


def _scrape_idx_official(timeout: int = 20) -> Optional[pd.DataFrame]:
    for url in OFFICIAL_IDX_STOCK_LIST_URLS:
        try:
            html = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).text
            tables = _extract_tables(html)
            found = []
            for tbl in tables:
                cols = [str(c).strip().lower() for c in tbl.columns]
                candidates = ["code", "ticker code", "ticker", "symbol"]
                hit = next((cand for cand in candidates if cand in cols), None)
                if hit is None:
                    continue
                sub = tbl.copy()
                sub = sub.rename(columns={tbl.columns[cols.index(hit)]: "ticker"})
                found.append(normalize_ticker_list(sub))
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
            tables = _extract_tables(html)
            found = []
            for tbl in tables:
                cols = [str(c).strip().lower() for c in tbl.columns]
                candidates = ["ticker", "code", "trading code", "stock code", "symbol"]
                for cand in candidates:
                    if cand in cols:
                        sub = tbl.copy().rename(columns={tbl.columns[cols.index(cand)]: "ticker"})
                        found.append(normalize_ticker_list(sub))
                        break
            if found:
                out = normalize_ticker_list(pd.concat(found, ignore_index=True))
                if len(out) >= 100:
                    return out
        except Exception:
            continue
    return None




def _read_cached_universe(base_dir: Path) -> Optional[pd.DataFrame]:
    p = base_dir / CACHE_UNIVERSE
    if not p.exists():
        return None
    try:
        df = normalize_ticker_list(pd.read_csv(p))
        return df if not df.empty else None
    except Exception:
        return None


def _write_cached_universe(base_dir: Path, df: pd.DataFrame) -> None:
    try:
        p = base_dir / CACHE_UNIVERSE
        p.parent.mkdir(parents=True, exist_ok=True)
        normalize_ticker_list(df).to_csv(p, index=False)
    except Exception:
        pass


def _scrape_raw_text_symbols(url: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    try:
        html = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).text
        syms = sorted(set(re.findall(r"\b[A-Z]{4}\b", html)))
        if len(syms) >= 200:
            return pd.DataFrame({"ticker": syms})
    except Exception:
        return None
    return None

def resolve_universe_source(
    mode: str,
    base_dir: str | Path = ".",
    uploaded_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str, list[str]]:
    base_dir = Path(base_dir)
    warnings: list[str] = []

    if uploaded_df is not None and not uploaded_df.empty:
        df = normalize_ticker_list(uploaded_df)
        if not df.empty:
            _write_cached_universe(base_dir, df)
        return df, "uploaded_csv", warnings

    if mode == "sample":
        sample = _read_local_csv(base_dir / "data" / "idx_universe_sample.csv")
        if sample is None or sample.empty:
            raise FileNotFoundError("Sample universe file missing.")
        warnings.append("Using sample universe only. This is not full IHSG.")
        return sample, "sample", warnings

    full_local = _read_local_csv(base_dir / "data" / "idx_universe_full.csv")
    if full_local is not None and not full_local.empty:
        _write_cached_universe(base_dir, full_local)
        return full_local, "local_full_csv", warnings

    cached = _read_cached_universe(base_dir)
    if cached is not None and not cached.empty and len(cached) >= 200:
        warnings.append("Using cached full universe from previous successful run.")
        return cached, "cached_full_universe", warnings

    if mode in {"full", "auto"}:
        idx_df = _scrape_idx_official()
        if idx_df is not None and not idx_df.empty:
            _write_cached_universe(base_dir, idx_df)
            return idx_df, "official_idx_web", warnings

        wiki_df = _scrape_wikipedia()
        if wiki_df is not None and not wiki_df.empty:
            warnings.append("Universe loaded from public web fallback. Validate coverage.")
            _write_cached_universe(base_dir, wiki_df)
            return wiki_df, "wikipedia_fallback", warnings

        for url in WIKI_URLS:
            raw_df = _scrape_raw_text_symbols(url)
            if raw_df is not None and not raw_df.empty:
                warnings.append("Universe reconstructed from raw text fallback. Validate coverage carefully.")
                _write_cached_universe(base_dir, raw_df)
                return normalize_ticker_list(raw_df), "raw_text_fallback", warnings

    sample = _read_local_csv(base_dir / "data" / "idx_universe_sample.csv")
    if sample is None or sample.empty:
        raise FileNotFoundError("No universe source available.")
    warnings.append("Fell back to sample universe. Add data/idx_universe_full.csv for stable full coverage.")
    warnings.append("Current run is NOT full IHSG if ticker count is small.")
    return sample, "fallback_sample", warnings
