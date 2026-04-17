
from __future__ import annotations
import argparse
import io
import re
from typing import List, Optional

import pandas as pd
import requests

OFFICIAL_URLS = [
    "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/",
    "https://www.idx.co.id/en/market-data/stocks-data/stock-list",
    "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat",
]
PUBLIC_FALLBACK_URLS = [
    "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia",
    "https://p2k.stekom.ac.id/ensiklopedia/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}

CODE_RE = re.compile(r"\b[A-Z]{4,5}\b")

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _pick_col(cols: List[str], patterns: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for p in patterns:
        for lc, orig in low.items():
            if p in lc:
                return orig
    return None

def _extract_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "company_name", "listing_date", "shares_outstanding", "board", "sector"])
    df = _clean_cols(df)

    # direct column matches
    code_col = _pick_col(list(df.columns), ["kode", "ticker", "symbol"])
    name_col = _pick_col(list(df.columns), ["nama perusahaan", "company", "perusahaan tercatat"])
    date_col = _pick_col(list(df.columns), ["tanggal", "listing date", "date listed"])
    shares_col = _pick_col(list(df.columns), ["jumlah saham", "shares"])
    board_col = _pick_col(list(df.columns), ["papan", "board"])
    sector_col = _pick_col(list(df.columns), ["sektor", "sector"])

    out = pd.DataFrame()
    if code_col:
        out["ticker"] = df[code_col].astype(str).str.extract(r"([A-Z]{4,5})", expand=False)
    else:
        # try all columns
        tmp = None
        for c in df.columns:
            s = df[c].astype(str).str.extract(r"\b([A-Z]{4,5})\b", expand=False)
            if s.notna().sum() >= max(3, len(df) // 8):
                tmp = s
                break
        out["ticker"] = tmp if tmp is not None else pd.Series([None] * len(df))

    out["company_name"] = df[name_col].astype(str) if name_col else ""
    out["listing_date"] = df[date_col].astype(str) if date_col else ""
    out["shares_outstanding"] = df[shares_col].astype(str) if shares_col else ""
    out["board"] = df[board_col].astype(str) if board_col else ""
    out["sector"] = df[sector_col].astype(str) if sector_col else ""
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out[out["ticker"].str.fullmatch(r"[A-Z]{4,5}", na=False)].copy()
    return out

def _read_html(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

def _scrape_sources(urls: List[str]) -> pd.DataFrame:
    parts = []
    for url in urls:
        try:
            dfs = _read_html(url)
        except Exception:
            continue
        for df in dfs:
            x = _extract_from_df(df)
            if not x.empty:
                x["source_url"] = url
                parts.append(x)
    if not parts:
        return pd.DataFrame(columns=["ticker", "company_name", "listing_date", "shares_outstanding", "board", "sector", "source_url"])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker"], keep="first").sort_values("ticker").reset_index(drop=True)
    return out

def build_universe() -> pd.DataFrame:
    official = _scrape_sources(OFFICIAL_URLS)
    public = _scrape_sources(PUBLIC_FALLBACK_URLS)
    frames = [df for df in [official, public] if not df.empty]
    if not frames:
        raise RuntimeError("Could not build universe from any source.")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "source_url"]).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    out["symbol_yf"] = out["ticker"].astype(str) + ".JK"
    out["status"] = "active"
    # normalize blanks
    for c in ["company_name", "listing_date", "shares_outstanding", "board", "sector"]:
        out[c] = out[c].fillna("").astype(str).str.strip()
    cols = ["ticker", "symbol_yf", "company_name", "listing_date", "shares_outstanding", "board", "sector", "status", "source_url"]
    return out[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/idx_universe_full.csv")
    args = ap.parse_args()
    out = build_universe()
    out.to_csv(args.output, index=False)
    print(f"saved {len(out)} tickers to {args.output}")

if __name__ == "__main__":
    main()
