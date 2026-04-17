from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = ["date", "ticker", "broker_code", "buy_lot", "buy_value", "sell_lot", "sell_value"]


def normalize_broker_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED + ["avg_buy", "avg_sell", "net_lot", "gross_lot"])

    out = df.copy()
    # flexible rename
    rename_map = {}
    lowered = {str(c).lower().strip(): c for c in out.columns}
    for want in REQUIRED:
        if want in out.columns:
            continue
        if want in lowered:
            rename_map[lowered[want]] = want
        elif want == "broker_code":
            for cand in ["broker", "brokercode", "code"]:
                if cand in lowered:
                    rename_map[lowered[cand]] = want
                    break
    out = out.rename(columns=rename_map)

    for c in REQUIRED:
        if c not in out.columns:
            out[c] = pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out["broker_code"] = out["broker_code"].astype(str).str.upper().str.strip()
    for c in ["buy_lot", "buy_value", "sell_lot", "sell_value"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["date"]) 
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{2,8}$", na=False)]
    out = out[out["broker_code"].str.match(r"^[A-Z0-9]{1,8}$", na=False)]

    out["avg_buy"] = np.where(out["buy_lot"] > 0, out["buy_value"] / out["buy_lot"], np.nan)
    out["avg_sell"] = np.where(out["sell_lot"] > 0, out["sell_value"] / out["sell_lot"], np.nan)
    out["net_lot"] = out["buy_lot"] - out["sell_lot"]
    out["gross_lot"] = out["buy_lot"] + out["sell_lot"]
    return out.sort_values(["ticker", "date", "broker_code"]).reset_index(drop=True)



def _fmt_top(df: pd.DataFrame, label_col: str, value_col: str, n: int = 3) -> str:
    if df.empty:
        return "-"
    tmp = df.sort_values(value_col, ascending=False).head(n)
    return ", ".join([f"{r[label_col]} ({r[value_col]:.0f})" for _, r in tmp.iterrows()])



def compute_broker_context(df: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    df = normalize_broker_summary(df)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "broker_alignment_score", "broker_mode", "dominant_accumulator",
            "dominant_distributor", "institutional_support", "institutional_resistance",
            "overhang_score", "broker_concentration_score", "broker_data_days"
        ])

    latest = df["date"].max()
    start = latest - pd.Timedelta(days=lookback_days - 1)
    work = df[df["date"].between(start, latest)].copy()

    rows = []
    for ticker, g in work.groupby("ticker", sort=True):
        by_broker = g.groupby("broker_code", as_index=False).agg(
            buy_lot=("buy_lot", "sum"),
            buy_value=("buy_value", "sum"),
            sell_lot=("sell_lot", "sum"),
            sell_value=("sell_value", "sum"),
            gross_lot=("gross_lot", "sum"),
            days=("date", "nunique"),
        )
        by_broker["net_lot"] = by_broker["buy_lot"] - by_broker["sell_lot"]
        by_broker["avg_buy"] = np.where(by_broker["buy_lot"] > 0, by_broker["buy_value"] / by_broker["buy_lot"], np.nan)
        by_broker["avg_sell"] = np.where(by_broker["sell_lot"] > 0, by_broker["sell_value"] / by_broker["sell_lot"], np.nan)

        pos = by_broker[by_broker["net_lot"] > 0].copy()
        neg = by_broker[by_broker["net_lot"] < 0].copy()

        total_gross = float(by_broker["gross_lot"].sum())
        total_pos = float(pos["net_lot"].sum()) if not pos.empty else 0.0
        total_neg = float((-neg["net_lot"]).sum()) if not neg.empty else 0.0
        net_dom = 0.0 if total_gross <= 0 else abs(total_pos - total_neg) / total_gross
        # HHI-like concentration on gross lot
        shares = by_broker["gross_lot"] / max(total_gross, 1e-9)
        concentration = float((shares.pow(2)).sum()) if total_gross > 0 else 0.0
        broker_alignment = float(np.clip(100 * (0.55 * net_dom + 0.45 * min(concentration * 8, 1)), 0, 100))

        institutional_support = np.nan
        if not pos.empty:
            w = pos["net_lot"].clip(lower=0)
            vals = pos["avg_buy"].fillna(0)
            institutional_support = float(np.average(vals, weights=w)) if w.sum() > 0 else np.nan

        institutional_resistance = np.nan
        if not neg.empty:
            w = (-neg["net_lot"]).clip(lower=0)
            vals = neg["avg_sell"].fillna(0)
            institutional_resistance = float(np.average(vals, weights=w)) if w.sum() > 0 else np.nan

        overhang = 0.0
        if total_gross > 0:
            overhang = float(np.clip(100 * total_neg / max(total_gross, 1e-9), 0, 100))

        mode = "BALANCED"
        if total_pos > total_neg * 1.2 and total_pos > 0:
            mode = "ACCUMULATION_DOMINANT"
        elif total_neg > total_pos * 1.2 and total_neg > 0:
            mode = "DISTRIBUTION_DOMINANT"

        rows.append({
            "ticker": ticker,
            "broker_alignment_score": round(broker_alignment, 1),
            "broker_mode": mode,
            "dominant_accumulator": _fmt_top(pos.assign(abs_net=pos["net_lot"]), "broker_code", "abs_net", 3),
            "dominant_distributor": _fmt_top(neg.assign(abs_net=(-neg["net_lot"])), "broker_code", "abs_net", 3),
            "institutional_support": institutional_support,
            "institutional_resistance": institutional_resistance,
            "overhang_score": round(overhang, 1),
            "broker_concentration_score": round(float(np.clip(concentration * 400, 0, 100)), 1),
            "broker_data_days": int(g["date"].nunique()),
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
