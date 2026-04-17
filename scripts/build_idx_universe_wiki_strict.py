from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

WIKI_URLS = [
    "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia",
    "https://p2k.stekom.ac.id/ensiklopedia/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia",
]

TICKER_RE = re.compile(r"^[A-Z]{4,5}$")


def normalize_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def find_table_with_tickers(url: str):
    tables = pd.read_html(url)
    best = None
    best_count = -1
    for t in tables:
        df = t.copy()
        df.columns = [normalize_col(c) for c in df.columns]
        candidates = [c for c in df.columns if any(k in c for k in ["kode", "ticker", "code", "bei"]) ]
        if not candidates:
            continue
        code_col = candidates[0]
        vals = df[code_col].astype(str).str.extract(r"([A-Z]{4,5})", expand=False)
        count = vals.notna().sum()
        if count > best_count:
            best_count = count
            best = (df, code_col)
    return best


def build(urls):
    frames = []
    for url in urls:
        try:
            found = find_table_with_tickers(url)
            if not found:
                continue
            df, code_col = found
            out = pd.DataFrame()
            out["ticker"] = df[code_col].astype(str).str.extract(r"([A-Z]{4,5})", expand=False)
            name_candidates = [c for c in df.columns if "nama" in c or "company" in c]
            board_candidates = [c for c in df.columns if "papan" in c or "board" in c]
            sector_candidates = [c for c in df.columns if "sector" in c or "sektor" in c]
            out["company_name"] = df[name_candidates[0]] if name_candidates else ""
            out["board"] = df[board_candidates[0]] if board_candidates else ""
            out["sector"] = df[sector_candidates[0]] if sector_candidates else ""
            out["status"] = "active"
            out = out[out["ticker"].astype(str).str.match(TICKER_RE, na=False)]
            frames.append(out)
            print(f"[OK] {url} -> {len(out)} tickers")
        except Exception as e:
            print(f"[FAIL] {url}: {e}")
    if not frames:
        raise SystemExit("No usable public universe source could be parsed.")
    final = pd.concat(frames, ignore_index=True)
    final = final.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    final["symbol_yf"] = final["ticker"] + ".JK"
    return final[["ticker", "symbol_yf", "company_name", "sector", "board", "status"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = build(WIKI_URLS)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    if len(out) < 900:
        print("WARNING: Result is still below 900 rows. This is not a full official master universe.")


if __name__ == "__main__":
    main()
