
from __future__ import annotations
import argparse
import pandas as pd

def norm(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError("input must have ticker column")
    out = df.copy()
    out["ticker"] = out[cols["ticker"]].astype(str).str.upper().str.extract(r"([A-Z]{4,5})", expand=False)
    if "symbol_yf" not in cols:
        out["symbol_yf"] = out["ticker"] + ".JK"
    out = out[out["ticker"].notna()].copy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_csv")
    ap.add_argument("extra_csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    a = norm(pd.read_csv(args.base_csv))
    b = norm(pd.read_csv(args.extra_csv))
    out = pd.concat([a, b], ignore_index=True).sort_values("ticker").drop_duplicates("ticker", keep="first")
    out.to_csv(args.output, index=False)
    print(f"saved {len(out)} tickers to {args.output}")

if __name__ == "__main__":
    main()
