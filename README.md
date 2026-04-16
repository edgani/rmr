# IDX EOD Smoke Test (Clean GitHub Deploy)

This repo is intentionally minimal so it can be deployed from zero without Streamlit multipage issues.

## What this app does
- Fetches EOD prices for IDX tickers using `yfinance` with the `.JK` suffix
- Builds a simple smoke-test watchlist
- Shows a candlestick chart and basic scores
- Exports raw prices and the latest watchlist to CSV

## Important
- The Streamlit app entrypoint is **`streamlit_app.py`**
- Do **not** create a `pages/` folder in this repo
- This is a **price-side EOD smoke test**, not a full broker-summary / bid-offer / done-detail engine yet

## Local run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud
- Deploy this repo from a **fresh GitHub repo**
- Set the **App file** to:

```text
streamlit_app.py
```

## If you still see `_navigation(...)`
That means Streamlit is still treating your deploy as a multipage app.
Check that the deployed repo does not contain a `pages/` directory.

## Default ticker universe
BBCA, BBRI, BMRI, BBNI, TLKM, ASII, ICBP, INDF, ANTM, MDKA, UNTR, AMRT, PANI, BRIS, GOTO, ADRO
