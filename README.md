
# IDX EOD Scanner V4.3 — Next Play Overlay

Single-file Streamlit app with a deploy-safe scanner plus a next-play overlay inspired by the uploaded `app(209).py`.

What it ports from the uploaded macro app:
- route state overlay
- asset translation
- analog prior
- most hated rally monitor
- upcoming events
- top drivers now
- forward radar
- data honesty banner

What it does **not** claim:
- full MacroRegime parity
- broker/intraday truth without your real uploads
- production alpha

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Universe
Prefer putting your own full list in `data/idx_universe_full.csv` with:
```csv
ticker,sector
BBCA,Banks
BBRI,Banks
...
```

Then point the sidebar input to that file.
