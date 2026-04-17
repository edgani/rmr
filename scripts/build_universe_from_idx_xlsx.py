
import argparse
import pandas as pd

MONTH_MAP = {"Jan":"Jan","Feb":"Feb","Mar":"Mar","Apr":"Apr","Mei":"May","Jun":"Jun","Jul":"Jul","Agu":"Aug","Sep":"Sep","Okt":"Oct","Nov":"Nov","Des":"Dec"}

def parse_id_date(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    parts = s.split()
    if len(parts) == 3:
        d, m, y = parts
        m = MONTH_MAP.get(m, m)
        try:
            return pd.to_datetime(f"{d} {m} {y}", dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            return s
    return s

def main(inp: str, outp: str):
    df = pd.read_excel(inp)
    out = pd.DataFrame({
        "ticker": df["Kode"].astype(str).str.upper().str.strip(),
        "symbol_yf": df["Kode"].astype(str).str.upper().str.strip() + ".JK",
        "company_name": df["Nama Perusahaan"].astype(str).str.strip(),
        "listing_date": df["Tanggal Pencatatan"].apply(parse_id_date),
        "shares_outstanding": df["Saham"].astype(str).str.replace(".", "", regex=False).str.replace(",", "", regex=False),
        "board": df["Papan Pencatatan"].astype(str).str.strip(),
        "status": "active",
        "sector": "",
    })
    out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    out.to_csv(outp, index=False)
    print(f"saved {len(out)} tickers -> {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    main(a.input, a.output)
