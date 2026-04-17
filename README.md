# IDX EOD Scanner V3.2 — Full IHSG Ready

Single-file Streamlit app untuk:
- EOD IDX via `yfinance` ticker `.JK`
- Full universe IHSG / IDX Composite via free auto-fetch + batching
- Optional broker summary CSV import
- Optional done detail CSV import
- Optional orderbook CSV import
- Burst / gulungan up-down, bullish/bearish burst, trap risk, dan watch rebound

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Universe modes

- **Full IHSG (free auto-fetch)**
  - coba local `data/idx_universe_full.csv`
  - lalu official IDX stock list pages
  - lalu fallback ke Wikipedia `IDX_Composite`
- **Sample IDX universe**
- **Manual input**

Untuk full universe, app download data secara **batch** supaya lebih tahan terhadap limit / partial failure di `yfinance`.

## Optional CSV schema

### broker summary
- `date`
- `ticker`
- `broker_code`
- `buy_lot`
- `sell_lot`
- optional: `buy_value`, `sell_value`

### done detail
- `timestamp`
- `ticker`
- `price`
- `lot`
- `side` (`B` / `S`)
- optional: `buyer_broker`, `seller_broker`

### orderbook
- `timestamp`
- `ticker`
- `bid_1_price`, `bid_1_lot`, `bid_2_price`, `bid_2_lot`, `bid_3_price`, `bid_3_lot`
- `offer_1_price`, `offer_1_lot`, `offer_2_price`, `offer_2_lot`, `offer_3_price`, `offer_3_lot`

## Catatan jujur

- EOD tetap bergantung pada `yfinance`.
- Full universe gratis itu **praktis**, tapi tidak akan sebersih data berlisensi / official feed.
- Broker / done detail / orderbook layer masih bergantung pada file yang lu upload.
- Ini belum full production Hengky-style engine, tapi sudah naik dari EOD price-side ke broker + intraday burst layer.


## V3.3 full-universe fetch
Full-universe loader sekarang mencoba official IDX stock-list pages, Wikipedia HTML, dan MediaWiki raw/API sebelum jatuh ke sample universe.
