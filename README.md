# IDX Scanner V4.5 — Intraday Hardening Max

Deploy-safe single-file Streamlit scanner for IDX using free yfinance `.JK` EOD plus optional broker summary, done detail, and orderbook uploads.

## Added in V4.5
- stronger done detail normalization and aggressor inference
- same-second split-order clustering
- bidirectional burst engine with follow-through and trap scoring
- orderbook refill / fake wall / absorption proxies
- stronger confidence penalties when intraday context is missing
- scanner verdicts now react to trap vs continuation, not just burst direction

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
