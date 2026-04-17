from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia"

COLUMN_MAP = {
    "Kode": "ticker",
    "Papan pencatatan": "board",
    "Sektor": "sector",
    "Nama perusahaan": "company_name",
    "Tanggal pencatatan": "listing_date",
    "Jumlah Saham": "shares_outstanding",
}


def _normalize_board(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def _normalize_sector(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def build_dataframe() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; IDXUniverseBuilder/1.0)",
        "Accept-Language": "en-US,en;q=0.9,id-ID;q=0.8,id;q=0.7",
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)

    target = None
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        if "Kode" in cols and "Nama perusahaan" in cols and "Papan pencatatan" in cols:
            target = tbl.copy()
            break
    if target is None:
        raise RuntimeError("Could not find the IDX company table on the Wikipedia page.")

    keep = [c for c in target.columns if str(c) in COLUMN_MAP]
    out = target[keep].rename(columns=COLUMN_MAP)
    out["ticker"] = out["ticker"].astype(str).str.extract(r"([A-Z0-9.]+)", expand=False).fillna("")
    out["ticker"] = out["ticker"].str.replace(r"[^A-Z0-9.]", "", regex=True)
    out = out[out["ticker"].str.len() > 0].copy()
    out["symbol_yf"] = out["ticker"] + ".JK"
    if "board" in out.columns:
        out["board"] = out["board"].map(_normalize_board)
    if "sector" in out.columns:
        out["sector"] = out["sector"].map(_normalize_sector)
    out["status"] = "active"
    out = out.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return out[["ticker", "symbol_yf", "company_name", "listing_date", "shares_outstanding", "board", "sector", "status"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local IDX universe CSV from Wikipedia IDX-listed companies page.")
    parser.add_argument("--output", default="data/idx_universe_full.csv", help="Output CSV path")
    args = parser.parse_args()

    df = build_dataframe()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} tickers to {out_path}")


if __name__ == "__main__":
    main()
