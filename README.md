# IDX Universe Builder Pack

This pack helps you create a **local master universe** for the scanner so the app does not depend on unstable web fallbacks.

## Build a full baseline universe from Wikipedia

```bash
pip install pandas requests lxml html5lib
python scripts/build_idx_universe_from_wiki.py --output data/idx_universe_full.csv
```

This will build a local CSV with columns:
- `ticker`
- `symbol_yf`
- `company_name`
- `listing_date`
- `shares_outstanding`
- `board`
- `sector`
- `status`

## Merge with your own additions / corrections

```bash
python scripts/merge_universe_csvs.py data/idx_universe_full.csv my_extra_tickers.csv --output data/idx_universe_full.csv
```

## Recommended workflow

1. Build the baseline universe.
2. Open the scanner app.
3. Make the app read `data/idx_universe_full.csv` as the source of truth.
4. Periodically rebuild / merge updates.

## Note

This is a practical way to get a much more complete universe than the small fallback sample. It is still best to replace this with a direct official export when you have one.
