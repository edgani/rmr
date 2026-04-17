# IDX Front-Run From Scratch (Full Universe Ready)

Paket ini sudah include `data/idx_universe_full.csv` hasil dari file IDX yang diupload user
dengan total 957 ticker.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Apa yang ada
- full universe lokal: `data/idx_universe_full.csv`
- price fetch via `yfinance` `.JK`
- 2 papan utama:
  - OPPORTUNITY SEKARANG
  - FRONT-RUN MARKET
- Data Audit:
  - target ticker
  - loaded
  - failed
  - coverage

## Catatan
- Harga tetap tergantung coverage / keberhasilan fetch `yfinance`
- Universe master sudah dari file IDX upload, jadi tidak lagi fallback ke 364 / sample kecil
- Sector masih kosong karena file IDX sumber tidak punya kolom sector
