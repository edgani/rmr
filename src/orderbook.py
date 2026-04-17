from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # best-effort normalization
    rename = {}
    lowered = {str(c).lower().strip(): c for c in out.columns}
    if "ticker" not in out.columns and "ticker" in lowered:
        rename[lowered["ticker"]] = "ticker"
    if "timestamp" not in out.columns:
        for cand in ["timestamp", "time", "datetime"]:
            if cand in lowered:
                rename[lowered[cand]] = "timestamp"
                break
    out = out.rename(columns=rename)
    if "ticker" not in out.columns or "timestamp" not in out.columns:
        return pd.DataFrame()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    numeric_cols = [c for c in out.columns if any(k in c.lower() for k in ["bid_", "offer_", "spread", "mid"])]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return out



def compute_orderbook_context(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_orderbook(df)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "bid_stack_quality", "offer_stack_quality", "absorption_after_up_score",
            "absorption_after_down_score", "tension_score", "fragility_score"
        ])
    rows = []
    bid_qty_cols = [c for c in df.columns if c.lower().startswith("bid_") and ("lot" in c.lower() or "qty" in c.lower())]
    off_qty_cols = [c for c in df.columns if c.lower().startswith("offer_") and ("lot" in c.lower() or "qty" in c.lower())]
    spread_col = next((c for c in df.columns if c.lower() == "spread"), None)
    for ticker, g in df.groupby("ticker", sort=True):
        last = g.iloc[-1]
        bid_sum = float(pd.to_numeric(last[bid_qty_cols], errors="coerce").fillna(0).sum()) if bid_qty_cols else 0.0
        off_sum = float(pd.to_numeric(last[off_qty_cols], errors="coerce").fillna(0).sum()) if off_qty_cols else 0.0
        spread = float(last[spread_col]) if spread_col and pd.notna(last[spread_col]) else np.nan
        total = max(bid_sum + off_sum, 1.0)
        bid_stack_quality = 100 * bid_sum / total
        offer_stack_quality = 100 * off_sum / total
        fragility = 100 * (off_sum / max(bid_sum, 1.0)) if bid_sum > 0 else 100.0
        tension = 100 * min(total / max(np.nanmedian((g[bid_qty_cols + off_qty_cols].sum(axis=1)).replace(0, np.nan)), 1.0), 2.0) / 2.0 if (bid_qty_cols or off_qty_cols) else 0.0
        rows.append({
            "ticker": ticker,
            "bid_stack_quality": round(float(np.clip(bid_stack_quality, 0, 100)), 1),
            "offer_stack_quality": round(float(np.clip(offer_stack_quality, 0, 100)), 1),
            "absorption_after_up_score": round(float(np.clip(offer_stack_quality * 0.8 + (0 if np.isnan(spread) else min(spread, 20) * 2), 0, 100)), 1),
            "absorption_after_down_score": round(float(np.clip(bid_stack_quality * 0.8 + (0 if np.isnan(spread) else min(spread, 20) * 2), 0, 100)), 1),
            "tension_score": round(float(np.clip(tension, 0, 100)), 1),
            "fragility_score": round(float(np.clip(fragility, 0, 100)), 1),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
