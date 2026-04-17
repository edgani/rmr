# IDX Buy-Side Front-Run Board

Deploy-safe single-file Streamlit app.

## What it does
- Uses `data/idx_universe_full.csv` as source-of-truth universe.
- Pulls EOD prices from Yahoo Finance (`.JK`).
- Splits names into two buy-side boards:
  - **OPPORTUNITY SEKARANG**
  - **FRONT-RUN MARKET**
- Shows audit for target / loaded / failed / coverage.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Rebuild universe from IDX Excel

```bash
python scripts/build_universe_from_idx_xlsx.py --input "Daftar Saham - 20260417.xlsx" --output data/idx_universe_full.csv
```
