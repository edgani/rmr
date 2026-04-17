from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = ["date", "ticker", "broker_code", "buy_lot", "buy_value", "sell_lot", "sell_value"]

COLUMN_ALIASES = {
    "date": ["date", "trade_date", "asof", "tanggal", "dt"],
    "ticker": ["ticker", "symbol", "stock", "kode", "security", "saham"],
    "broker_code": ["broker_code", "broker", "brokercode", "code", "member", "buyer_broker", "seller_broker"],
    "buy_lot": ["buy_lot", "buylot", "b_lot", "blot", "buy_volume", "bvol", "lot_buy", "lotb"],
    "buy_value": ["buy_value", "buyvalue", "b_value", "bval", "buy_amount", "nval_b", "value_buy"],
    "sell_lot": ["sell_lot", "selllot", "s_lot", "slot", "sell_volume", "svol", "lot_sell", "lots"],
    "sell_value": ["sell_value", "sellvalue", "s_value", "sval", "sell_amount", "nval_s", "value_sell"],
}

MASTER_ALIASES = {
    "broker_code": ["broker_code", "broker", "code", "member"],
    "broker_name": ["broker_name", "name", "participant", "broker", "firm"],
    "style_hint": ["style_hint", "style", "hint", "category", "class"],
    "desk_hint": ["desk_hint", "desk", "type", "desk_type"],
}


def _clean_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    txt = s.astype(str).str.replace(",", "", regex=False).str.replace("_", "", regex=False).str.strip()
    txt = txt.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")


def _first_matching(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower_map = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def normalize_broker_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED + ["avg_buy", "avg_sell", "net_lot", "gross_lot"])

    out = df.copy()
    rename_map: dict[str, str] = {}
    for want, aliases in COLUMN_ALIASES.items():
        if want in out.columns:
            continue
        found = _first_matching(out.columns, aliases)
        if found is not None:
            rename_map[found] = want
    out = out.rename(columns=rename_map)

    for c in REQUIRED:
        if c not in out.columns:
            out[c] = pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out["broker_code"] = out["broker_code"].astype(str).str.upper().str.strip()

    for c in ["buy_lot", "buy_value", "sell_lot", "sell_value"]:
        out[c] = _clean_numeric(out[c]).fillna(0.0)

    out = out.dropna(subset=["date"])
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{2,8}$", na=False)]
    out = out[out["broker_code"].str.match(r"^[A-Z0-9]{1,8}$", na=False)]

    out["avg_buy"] = np.where(out["buy_lot"] > 0, out["buy_value"] / out["buy_lot"], np.nan)
    out["avg_sell"] = np.where(out["sell_lot"] > 0, out["sell_value"] / out["sell_lot"], np.nan)
    out["net_lot"] = out["buy_lot"] - out["sell_lot"]
    out["gross_lot"] = out["buy_lot"] + out["sell_lot"]
    return out.sort_values(["ticker", "date", "broker_code"]).reset_index(drop=True)


def normalize_broker_master(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["broker_code", "broker_name", "style_hint", "desk_hint"])

    out = df.copy()
    rename_map: dict[str, str] = {}
    for want, aliases in MASTER_ALIASES.items():
        if want in out.columns:
            continue
        found = _first_matching(out.columns, aliases)
        if found is not None:
            rename_map[found] = want
    out = out.rename(columns=rename_map)
    if "broker_code" not in out.columns:
        return pd.DataFrame(columns=["broker_code", "broker_name", "style_hint", "desk_hint"])

    for c in ["broker_name", "style_hint", "desk_hint"]:
        if c not in out.columns:
            out[c] = pd.NA

    out["broker_code"] = out["broker_code"].astype(str).str.upper().str.strip()
    out["broker_name"] = out["broker_name"].astype(str).str.strip()
    out["style_hint"] = out["style_hint"].astype(str).str.upper().str.strip()
    out["desk_hint"] = out["desk_hint"].astype(str).str.upper().str.strip()
    out = out[out["broker_code"].str.match(r"^[A-Z0-9]{1,8}$", na=False)]
    return out[["broker_code", "broker_name", "style_hint", "desk_hint"]].drop_duplicates("broker_code")


def _fmt_top(df: pd.DataFrame, n: int = 3) -> str:
    if df.empty:
        return "-"
    parts = []
    for _, r in df.head(n).iterrows():
        label = str(r.get("broker_code", "-")).strip()
        if pd.notna(r.get("broker_name", np.nan)) and str(r.get("broker_name", "")).strip() not in {"", "nan"}:
            label = f"{label}/{str(r['broker_name']).strip()[:16]}"
        if pd.notna(r.get("style_hint", np.nan)) and str(r.get("style_hint", "")).strip() not in {"", "nan", "NONE"}:
            label = f"{label}[{str(r['style_hint']).strip()[:8]}]"
        val = float(r.get("abs_net", 0.0))
        days = int(r.get("days", 0))
        parts.append(f"{label} ({val:.0f};{days}d)")
    return ", ".join(parts)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _institutional_band(level: float, concentration_score: float) -> tuple[float, float]:
    if not np.isfinite(level):
        return np.nan, np.nan
    width_pct = 0.008 + 0.015 * (1 - np.clip(concentration_score / 100.0, 0, 1))
    return float(level * (1 - width_pct)), float(level * (1 + width_pct))


def compute_broker_context(df: pd.DataFrame, broker_master_df: pd.DataFrame | None = None, lookback_days: int = 20) -> pd.DataFrame:
    df = normalize_broker_summary(df)
    master = normalize_broker_master(broker_master_df)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "broker_alignment_score", "broker_mode", "dominant_accumulator",
            "dominant_distributor", "institutional_support", "institutional_resistance",
            "institutional_support_low", "institutional_support_high",
            "institutional_resistance_low", "institutional_resistance_high",
            "overhang_score", "broker_concentration_score", "broker_persistence_score",
            "broker_days_active", "active_broker_count", "accumulator_breadth", "distributor_breadth",
            "broker_acc_pressure", "broker_dist_pressure", "broker_data_days", "broker_name_coverage",
            "broker_style_bias"
        ])

    latest = df["date"].max()
    start = latest - pd.Timedelta(days=max(lookback_days - 1, 1))
    work = df[df["date"].between(start, latest)].copy()

    rows = []
    for ticker, g in work.groupby("ticker", sort=True):
        by_day_broker = g.groupby(["date", "broker_code"], as_index=False).agg(
            buy_lot=("buy_lot", "sum"),
            buy_value=("buy_value", "sum"),
            sell_lot=("sell_lot", "sum"),
            sell_value=("sell_value", "sum"),
            gross_lot=("gross_lot", "sum"),
            net_lot=("net_lot", "sum"),
        )
        by_broker = by_day_broker.groupby("broker_code", as_index=False).agg(
            buy_lot=("buy_lot", "sum"),
            buy_value=("buy_value", "sum"),
            sell_lot=("sell_lot", "sum"),
            sell_value=("sell_value", "sum"),
            gross_lot=("gross_lot", "sum"),
            net_lot=("net_lot", "sum"),
            days=("date", "nunique"),
        )
        if not master.empty:
            by_broker = by_broker.merge(master, on="broker_code", how="left")
        else:
            by_broker["broker_name"] = pd.NA
            by_broker["style_hint"] = pd.NA
            by_broker["desk_hint"] = pd.NA

        by_broker["avg_buy"] = np.where(by_broker["buy_lot"] > 0, by_broker["buy_value"] / by_broker["buy_lot"], np.nan)
        by_broker["avg_sell"] = np.where(by_broker["sell_lot"] > 0, by_broker["sell_value"] / by_broker["sell_lot"], np.nan)

        daily_sign = by_day_broker.assign(pos=(by_day_broker["net_lot"] > 0).astype(int), neg=(by_day_broker["net_lot"] < 0).astype(int))
        sign_summary = daily_sign.groupby("broker_code", as_index=False).agg(
            pos_days=("pos", "sum"), neg_days=("neg", "sum"), active_days=("date", "nunique")
        )
        by_broker = by_broker.merge(sign_summary, on="broker_code", how="left")
        by_broker[["pos_days", "neg_days", "active_days"]] = by_broker[["pos_days", "neg_days", "active_days"]].fillna(0)
        by_broker["persistence"] = np.where(
            by_broker["active_days"] > 0,
            np.maximum(by_broker["pos_days"], by_broker["neg_days"]) / by_broker["active_days"],
            0.0,
        )

        total_gross = float(by_broker["gross_lot"].sum())
        shares = by_broker["gross_lot"] / max(total_gross, 1e-9)
        concentration = float((shares.pow(2)).sum()) if total_gross > 0 else 0.0
        concentration_score = float(np.clip(concentration * 400, 0, 100))

        pos = by_broker[by_broker["net_lot"] > 0].copy().sort_values("net_lot", ascending=False)
        neg = by_broker[by_broker["net_lot"] < 0].copy().assign(abs_net=lambda x: -x["net_lot"]).sort_values("abs_net", ascending=False)
        pos["abs_net"] = pos["net_lot"].abs()

        total_pos = float(pos["net_lot"].sum()) if not pos.empty else 0.0
        total_neg = float(neg["abs_net"].sum()) if not neg.empty else 0.0
        net_dom = 0.0 if total_gross <= 0 else abs(total_pos - total_neg) / total_gross

        top_acc_share = float(pos.head(3)["net_lot"].sum() / max(total_gross, 1e-9)) if not pos.empty else 0.0
        top_dist_share = float(neg.head(3)["abs_net"].sum() / max(total_gross, 1e-9)) if not neg.empty else 0.0
        persistence_score = float(np.clip((0.55 * by_broker["persistence"].mean() + 0.45 * by_broker["persistence"].max()) * 100, 0, 100))
        alignment_score = float(np.clip(100 * (0.35 * net_dom + 0.25 * min(concentration * 8, 1) + 0.20 * top_acc_share + 0.20 * persistence_score / 100), 0, 100))

        support = _weighted_mean(pos["avg_buy"], pos["net_lot"].clip(lower=0))
        resistance = _weighted_mean(neg["avg_sell"], neg["abs_net"].clip(lower=0))
        support_low, support_high = _institutional_band(support, concentration_score)
        resistance_low, resistance_high = _institutional_band(resistance, concentration_score)

        overhang = float(np.clip(100 * total_neg / max(total_gross, 1e-9), 0, 100)) if total_gross > 0 else 0.0
        acc_pressure = float(np.clip(100 * total_pos / max(total_gross, 1e-9), 0, 100)) if total_gross > 0 else 0.0
        dist_pressure = float(np.clip(100 * total_neg / max(total_gross, 1e-9), 0, 100)) if total_gross > 0 else 0.0

        mode = "BALANCED"
        if total_pos > total_neg * 1.20 and persistence_score >= 52:
            mode = "ACCUMULATION_DOMINANT"
        elif total_neg > total_pos * 1.20 and persistence_score >= 52:
            mode = "DISTRIBUTION_DOMINANT"
        elif concentration_score >= 55 and persistence_score < 45:
            mode = "CHURN_HEAVY"

        style_bias = "UNKNOWN"
        if not by_broker["style_hint"].dropna().empty:
            s = by_broker["style_hint"].astype(str).str.upper().str.strip()
            top_style = s[s.ne("") & s.ne("NAN")].value_counts()
            if not top_style.empty:
                style_bias = str(top_style.index[0])

        rows.append({
            "ticker": ticker,
            "broker_alignment_score": round(alignment_score, 1),
            "broker_mode": mode,
            "dominant_accumulator": _fmt_top(pos, 4),
            "dominant_distributor": _fmt_top(neg, 4),
            "institutional_support": support,
            "institutional_resistance": resistance,
            "institutional_support_low": support_low,
            "institutional_support_high": support_high,
            "institutional_resistance_low": resistance_low,
            "institutional_resistance_high": resistance_high,
            "overhang_score": round(overhang, 1),
            "broker_concentration_score": round(concentration_score, 1),
            "broker_persistence_score": round(persistence_score, 1),
            "broker_days_active": int(by_broker["active_days"].max()) if not by_broker.empty else 0,
            "active_broker_count": int(by_broker["broker_code"].nunique()),
            "accumulator_breadth": int((pos["net_lot"] > 0).sum()),
            "distributor_breadth": int((neg["abs_net"] > 0).sum()),
            "broker_acc_pressure": round(acc_pressure, 1),
            "broker_dist_pressure": round(dist_pressure, 1),
            "broker_data_days": int(g["date"].nunique()),
            "broker_name_coverage": round(float(by_broker["broker_name"].notna().mean() * 100), 1),
            "broker_style_bias": style_bias,
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
