# IDX Front-Run Scanner — From Scratch Pack

Ini paket paling lengkap yang sudah digabung jadi satu repo bersih dari nol.

## Isi utama
- `streamlit_app.py` — app utama Streamlit
- `src/` — logic scanner, broker, intraday, macro filter, ranking, explainability
- `data/` — starter universe + template CSV
- `scripts/` — builder/merge universe + audit helper

## Run cepat
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Yang penting buat full universe
App akan paling stabil kalau baca universe lokal dari:
- `data/idx_universe_full.csv`

Kalau file itu belum lengkap, pakai builder:
```bash
python scripts/build_idx_universe_all.py --output data/idx_universe_full.csv
```

Atau strict builder:
```bash
python scripts/build_idx_universe_wiki_strict.py --output data/idx_universe_full.csv
```

## Audit universe
```bash
python scripts/export_universe_audit_demo.py
```

## Data optional yang bisa di-upload di app
- broker summary CSV
- broker master CSV
- done detail CSV
- orderbook CSV
- route/catalyst events CSV

## Status jujur
- Ini paket paling lengkap untuk base deploy-safe + front-run aware scanner.
- Harga pakai yfinance `.JK`.
- Universe starter yang ikut di dalam belum otomatis 100% full IDX kecuali lu build/replace `data/idx_universe_full.csv`.
- Broker/intraday makin kuat kalau lu upload data real.
