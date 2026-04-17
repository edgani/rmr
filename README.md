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


## V4.7 fixes
- safer rank-score handling when broker columns are missing
- cached full universe fallback at `data/cache/latest_universe.csv`
- optional `Route / catalyst events CSV` upload for next-play overlay
- hard warning when the app falls back to the 19-name sample universe


## Input normalizer hardening
This build adds aggressive CSV normalization for universe/metadata, broker summary, broker master, done detail, orderbook, and route/catalyst events. Common alias columns, mixed casing, noisy currency text, and decimal comma formats are normalized automatically before scoring.


## V5.0 Action View
Default view sekarang dibikin lebih gampang dibaca untuk orang awam: 6 bucket action, trigger, invalidation, timing, dan confidence.
