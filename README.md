# IDX Scanner V4.0 — Full Stack Base

Deploy-safe single-file Streamlit app for scanning IDX names with:
- full-universe loader (local CSV first, public web fallback, sample last)
- batched yfinance `.JK` EOD fetch
- price-side EOD scoring
- optional broker summary import
- optional done detail import
- optional orderbook import
- broker-aware dry/wet
- institutional support / resistance proxy
- intraday burst / gulungan up-down labels
- confidence and explainability

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Optional files

Put a stable full-universe file in:

`data/idx_universe_full.csv`

with one column:

```csv
ticker
BBCA
BBRI
BMRI
...
```

## Notes

This is still a research scanner, not a production execution engine.
- EOD price-side uses free `yfinance`
- broker/intraday layers require your own real CSV data
- institutional levels are broker-flow proxies, not custody truth
