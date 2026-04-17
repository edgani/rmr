# IDX Universe All-Ticker Builder

Tujuan pack ini: bantu bikin `data/idx_universe_full.csv` yang jauh lebih lengkap daripada seed 300-an/374.

Builder akan coba:
1. halaman resmi IDX `Daftar Saham` / `Stock List` / `Profil Perusahaan Tercatat`
2. fallback publik dari Wikipedia / mirror publik

## Run
```bash
pip install pandas requests lxml html5lib
python scripts/build_idx_universe_all.py --output data/idx_universe_full.csv
```

## Merge file tambahan sendiri
```bash
python scripts/merge_universe_csvs.py data/idx_universe_full.csv my_extra.csv --output data/idx_universe_full.csv
```

## Output columns
- ticker
- symbol_yf
- company_name
- listing_date
- shares_outstanding
- board
- sector
- status
- source_url

## Catatan
- builder ini ditujukan buat jalan di environment yang punya internet.
- hasilnya tetap harus diaudit, karena sumber publik bisa berubah format.
- kalau lu punya export resmi IDX stock list, merge saja ke file hasil builder ini dan jadikan itu source of truth app.
