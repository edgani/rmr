from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd

from .broker import normalize_broker_master, normalize_broker_summary
from .done_detail import normalize_done_detail
from .orderbook import normalize_orderbook


def _canon(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [_canon(c) for c in out.columns]
    return out


def _find_first(cols: list[str], aliases: list[str]) -> str | None:
    canon_map = {_canon(c): c for c in cols}
    for a in aliases:
        key = _canon(a)
        if key in canon_map:
            return canon_map[key]
    return None


def _clean_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    txt = s.astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    txt = txt.str.replace(r"(?i)rp\.?", "", regex=True)
    txt = txt.str.replace(r"(?i)idr", "", regex=True)
    txt = txt.str.replace(r"(?i)lot", "", regex=True)
    txt = txt.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    txt = txt.str.replace(" ", "", regex=False)
    has_dot = txt.str.contains("\\.", regex=True)
    has_comma = txt.str.contains(",", regex=False)
    euro = has_dot & has_comma & txt.str.rfind(".").lt(txt.str.rfind(","))
    txt.loc[euro] = txt.loc[euro].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    us = has_dot & has_comma & ~euro
    txt.loc[us] = txt.loc[us].str.replace(",", "", regex=False)
    comma_only = ~has_dot & has_comma
    long_thousands = comma_only & txt.str.match(r"^-?\d{1,3}(,\d{3})+$", na=False)
    txt.loc[long_thousands] = txt.loc[long_thousands].str.replace(",", "", regex=False)
    decimal_comma = comma_only & ~long_thousands
    txt.loc[decimal_comma] = txt.loc[decimal_comma].str.replace(",", ".", regex=False)
    txt = txt.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")


def normalize_universe_metadata(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "sector", "board", "name", "symbol_yf"])
    out = clean_column_names(df)
    aliases = {
        "ticker": ["ticker", "code", "symbol", "kode", "stock_code", "ticker_code"],
        "sector": ["sector", "industry", "sektor", "sector_name", "industry_group"],
        "board": ["board", "papan", "listing_board"],
        "name": ["name", "company", "issuer", "issuer_name", "company_name", "nama"],
        "symbol_yf": ["symbol_yf", "yf", "yf_symbol", "yfinance", "symboljk"],
    }
    rename = {}
    for want, opts in aliases.items():
        if want not in out.columns:
            f = _find_first(list(out.columns), opts)
            if f is not None:
                rename[f] = want
    out = out.rename(columns=rename)
    if "ticker" not in out.columns and len(out.columns) > 0:
        out = out.rename(columns={out.columns[0]: "ticker"})
    if "ticker" not in out.columns:
        return pd.DataFrame(columns=["ticker", "sector", "board", "name", "symbol_yf"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{2,8}$", na=False)].copy()
    if "symbol_yf" not in out.columns:
        out["symbol_yf"] = out["ticker"] + ".JK"
    else:
        yf = out["symbol_yf"].astype(str).str.upper().str.strip()
        out["symbol_yf"] = np.where(yf.str.endswith(".JK"), yf, out["ticker"] + ".JK")
    for c in ["sector", "board", "name"]:
        if c not in out.columns:
            out[c] = pd.NA
        else:
            out[c] = out[c].astype(str).replace({"nan": pd.NA, "": pd.NA}).str.strip()
    return out[[c for c in ["ticker", "sector", "board", "name", "symbol_yf"] if c in out.columns]].drop_duplicates("ticker").reset_index(drop=True)


def normalize_route_events(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["event_date", "title", "family", "catalyst_score", "analog_label", "scenario_family"])
    out = clean_column_names(df)
    aliases = {
        "event_date": ["event_date", "date", "when", "event_time", "countdown_date"],
        "title": ["title", "event", "name", "headline"],
        "family": ["family", "type", "event_family", "catalyst_family"],
        "catalyst_score": ["catalyst_score", "score", "priority", "impact_score", "weight"],
        "analog_label": ["analog_label", "analog", "scenario", "analogy"],
        "scenario_family": ["scenario_family", "family_label", "route_family"],
    }
    rename = {}
    for want, opts in aliases.items():
        if want not in out.columns:
            f = _find_first(list(out.columns), opts)
            if f is not None:
                rename[f] = want
    out = out.rename(columns=rename)
    for c in ["title", "family", "analog_label", "scenario_family"]:
        if c not in out.columns:
            out[c] = pd.NA
    out["event_date"] = pd.to_datetime(out.get("event_date", pd.Series(dtype=object)), errors="coerce")
    out["catalyst_score"] = _clean_numeric(out.get("catalyst_score", pd.Series(dtype=object))).fillna(0.0)
    return out[["event_date", "title", "family", "catalyst_score", "analog_label", "scenario_family"]].reset_index(drop=True)


def normalize_uploaded_csv(df: pd.DataFrame | None, kind: str) -> pd.DataFrame:
    kind = str(kind or "").lower().strip()
    if df is None or df.empty:
        return pd.DataFrame()
    if kind in {"universe", "metadata", "universe_metadata"}:
        return normalize_universe_metadata(df)
    if kind in {"broker", "broker_summary"}:
        return normalize_broker_summary(clean_column_names(df))
    if kind in {"broker_master", "brokermaster"}:
        return normalize_broker_master(clean_column_names(df))
    if kind in {"done", "done_detail", "trade_prints", "matched"}:
        return normalize_done_detail(clean_column_names(df))
    if kind in {"orderbook", "book", "depth"}:
        return normalize_orderbook(clean_column_names(df))
    if kind in {"route_events", "events", "catalyst"}:
        return normalize_route_events(df)
    return clean_column_names(df)
