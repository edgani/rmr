from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("code")
    if not ticker_col:
        raise ValueError(f"No ticker column found in {path}")
    df = df.rename(columns={ticker_col: "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(r"[^A-Z0-9.]", "", regex=True)
    if "symbol_yf" not in df.columns:
        df["symbol_yf"] = df["ticker"] + ".JK"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple IDX universe CSVs into one de-duplicated master file.")
    parser.add_argument("inputs", nargs="+", help="Input CSV files")
    parser.add_argument("--output", default="data/idx_universe_full.csv", help="Output CSV path")
    args = parser.parse_args()

    frames = [load_csv(Path(p)) for p in args.inputs]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["ticker"], keep="first").sort_values("ticker").reset_index(drop=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Merged {len(frames)} files into {len(out):,} unique tickers -> {out_path}")


if __name__ == "__main__":
    main()
