from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    df = pd.read_excel(args.input)
    cols = {str(c).strip().lower(): c for c in df.columns}
    out = pd.DataFrame()
    out['ticker'] = df[cols['kode']].astype(str).str.upper().str.strip()
    out['symbol_yf'] = out['ticker'] + '.JK'
    out['company_name'] = df[cols.get('nama perusahaan', cols.get('nama emiten'))].astype(str)
    out['listing_date'] = pd.to_datetime(df[cols.get('tanggal pencatatan')], errors='coerce').dt.date.astype(str)
    saham_col = cols.get('saham')
    if saham_col:
        out['shares_outstanding'] = pd.to_numeric(df[saham_col].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    else:
        out['shares_outstanding'] = pd.NA
    out['board'] = df[cols.get('papan pencatatan', '')].astype(str) if cols.get('papan pencatatan') else ''
    out['status'] = out['board'].map(lambda x: 'special_monitoring' if 'khusus' in str(x).lower() else 'active')
    out['sector'] = ''
    out = out.drop_duplicates(subset=['ticker']).sort_values('ticker').reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'wrote {len(out)} rows to {args.output}')

if __name__ == '__main__':
    main()
