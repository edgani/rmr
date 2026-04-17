from __future__ import annotations

import re
import numpy as np
import pandas as pd


def _first_matching(columns, aliases):
    lower = {str(c).lower().strip(): c for c in columns}
    for cand in aliases:
        if cand in lower:
            return lower[cand]
    return None


def _clean_numeric(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.replace(",", "", regex=False).str.replace("_", "", regex=False).str.strip()
    txt = txt.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")


def normalize_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename = {}
    lower = {str(c).lower().strip(): c for c in out.columns}
    for want, aliases in {
        "ticker": ["ticker", "symbol", "stock", "kode"],
        "timestamp": ["timestamp", "time", "datetime", "ts"],
        "mid_price": ["mid_price", "mid", "midprice"],
        "spread": ["spread", "sprd"],
    }.items():
        if want not in out.columns:
            found = _first_matching(out.columns, aliases)
            if found is not None:
                rename[found] = want
    out = out.rename(columns=rename)
    if "ticker" not in out.columns or "timestamp" not in out.columns:
        return pd.DataFrame()

    # normalize common depth column patterns into bid_i_price/lot and offer_i_price/lot
    colmap = {}
    for c in out.columns:
        lc = str(c).lower().strip()
        m = re.match(r"bid[_ ]?(\d+)(?:[_ ]?(price|px|lot|qty|volume))?$", lc)
        if m:
            idx, kind = m.group(1), m.group(2) or "price"
            colmap[c] = f"bid_{idx}_{'lot' if kind in {'lot','qty','volume'} else 'price'}"
            continue
        m = re.match(r"offer[_ ]?(\d+)(?:[_ ]?(price|px|lot|qty|volume))?$", lc)
        if m:
            idx, kind = m.group(1), m.group(2) or "price"
            colmap[c] = f"offer_{idx}_{'lot' if kind in {'lot','qty','volume'} else 'price'}"
            continue
        m = re.match(r"ask[_ ]?(\d+)(?:[_ ]?(price|px|lot|qty|volume))?$", lc)
        if m:
            idx, kind = m.group(1), m.group(2) or "price"
            colmap[c] = f"offer_{idx}_{'lot' if kind in {'lot','qty','volume'} else 'price'}"
    out = out.rename(columns=colmap)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    depth_cols = [c for c in out.columns if c.startswith("bid_") or c.startswith("offer_") or c in {"spread", "mid_price"}]
    for c in depth_cols:
        out[c] = _clean_numeric(out[c])
    out = out.dropna(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return out


def compute_orderbook_context(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_orderbook(df)
    cols = [
        "ticker", "bid_stack_quality", "offer_stack_quality", "absorption_after_up_score",
        "absorption_after_down_score", "tension_score", "fragility_score", "offer_refill_rate",
        "bid_refill_rate", "fake_wall_offer_score", "fake_wall_bid_score"
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    bid_qty_cols = [c for c in df.columns if c.startswith("bid_") and c.endswith("_lot")]
    off_qty_cols = [c for c in df.columns if c.startswith("offer_") and c.endswith("_lot")]
    rows = []
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.copy().sort_values("timestamp")
        if not bid_qty_cols and not off_qty_cols:
            continue
        g["bid_sum"] = pd.to_numeric(g[bid_qty_cols], errors="coerce").fillna(0).sum(axis=1) if bid_qty_cols else 0.0
        g["off_sum"] = pd.to_numeric(g[off_qty_cols], errors="coerce").fillna(0).sum(axis=1) if off_qty_cols else 0.0
        g["total_depth"] = g["bid_sum"] + g["off_sum"]
        if "mid_price" not in g.columns:
            bp = next((c for c in g.columns if c == "bid_1_price"), None)
            op = next((c for c in g.columns if c == "offer_1_price"), None)
            if bp and op:
                g["mid_price"] = (pd.to_numeric(g[bp], errors="coerce") + pd.to_numeric(g[op], errors="coerce")) / 2.0
            else:
                g["mid_price"] = np.nan
        if "spread" not in g.columns:
            bp = next((c for c in g.columns if c == "bid_1_price"), None)
            op = next((c for c in g.columns if c == "offer_1_price"), None)
            if bp and op:
                g["spread"] = pd.to_numeric(g[op], errors="coerce") - pd.to_numeric(g[bp], errors="coerce")
            else:
                g["spread"] = np.nan
        g["mid_chg"] = pd.to_numeric(g["mid_price"], errors="coerce").diff()
        g["offer_refill_pos"] = np.where((g["mid_chg"] >= 0) & (g["off_sum"].diff() > 0), g["off_sum"].diff(), 0.0)
        g["bid_refill_pos"] = np.where((g["mid_chg"] <= 0) & (g["bid_sum"].diff() > 0), g["bid_sum"].diff(), 0.0)
        last = g.iloc[-1]
        bid_sum = float(last["bid_sum"])
        off_sum = float(last["off_sum"])
        total = max(bid_sum + off_sum, 1.0)
        median_depth = max(float(g["total_depth"].replace(0, np.nan).median()) if g["total_depth"].notna().any() else 1.0, 1.0)
        spread = float(pd.to_numeric(pd.Series([last.get("spread", np.nan)]), errors="coerce").iloc[0])
        top_bid = float(pd.to_numeric(pd.Series([last.get("bid_1_lot", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        top_offer = float(pd.to_numeric(pd.Series([last.get("offer_1_lot", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        fake_wall_bid = 100 * np.clip(top_bid / max(bid_sum, 1.0), 0, 1)
        fake_wall_offer = 100 * np.clip(top_offer / max(off_sum, 1.0), 0, 1)
        bid_refill = 100 * np.clip(g["bid_refill_pos"].replace(0, np.nan).mean() / max(g["bid_sum"].replace(0, np.nan).median() or 1.0, 1.0), 0, 1)
        offer_refill = 100 * np.clip(g["offer_refill_pos"].replace(0, np.nan).mean() / max(g["off_sum"].replace(0, np.nan).median() or 1.0, 1.0), 0, 1)
        bid_stack_quality = 100 * np.clip(bid_sum / total, 0, 1)
        offer_stack_quality = 100 * np.clip(off_sum / total, 0, 1)
        tension = 100 * np.clip((bid_sum + off_sum) / median_depth / 2.0, 0, 1)
        fragility = 100 * np.clip(min(bid_sum, off_sum) / max(bid_sum, off_sum, 1.0), 0, 1)
        absorption_up = 100 * np.clip(0.45 * (offer_stack_quality / 100) + 0.25 * (offer_refill / 100) + 0.20 * (fake_wall_offer / 100) + 0.10 * min(max(spread, 0.0) / 10.0, 1.0), 0, 1)
        absorption_down = 100 * np.clip(0.45 * (bid_stack_quality / 100) + 0.25 * (bid_refill / 100) + 0.20 * (fake_wall_bid / 100) + 0.10 * min(max(spread, 0.0) / 10.0, 1.0), 0, 1)
        rows.append({
            "ticker": ticker,
            "bid_stack_quality": round(float(bid_stack_quality), 1),
            "offer_stack_quality": round(float(offer_stack_quality), 1),
            "absorption_after_up_score": round(float(absorption_up), 1),
            "absorption_after_down_score": round(float(absorption_down), 1),
            "tension_score": round(float(tension), 1),
            "fragility_score": round(float(fragility), 1),
            "offer_refill_rate": round(float(offer_refill), 1),
            "bid_refill_rate": round(float(bid_refill), 1),
            "fake_wall_offer_score": round(float(fake_wall_offer), 1),
            "fake_wall_bid_score": round(float(fake_wall_bid), 1),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
