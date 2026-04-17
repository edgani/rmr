
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd


def _clean_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def _norm_ticker(x: object) -> str:
    s = str(x or "").strip().upper()
    s = s.replace(".JK", "")
    return s


def _safe_list(values: Optional[Iterable[object]]) -> list[str]:
    if values is None:
        return []
    return [v for v in (_norm_ticker(x) for x in values) if v]


def load_master_universe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "symbol_yf", "sector", "board", "status"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "symbol_yf", "sector", "board", "status"])

    df.columns = [_clean_col(c) for c in df.columns]

    ticker_col = None
    for c in ["ticker", "symbol", "code", "stock_code"]:
        if c in df.columns:
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("Master universe file harus punya kolom ticker/symbol/code/stock_code")

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].map(_norm_ticker)
    out["symbol_yf"] = df.get("symbol_yf", out["ticker"] + ".JK")
    out["sector"] = df.get("sector", "")
    out["board"] = df.get("board", "")
    out["status"] = df.get("status", "active")
    out = out[out["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out


def build_universe_audit(
    master_df: pd.DataFrame,
    loaded_tickers: Optional[Iterable[object]],
    failed_tickers: Optional[Iterable[object]] = None,
    source_mode: str = "REAL_PRICES_ONLY",
) -> dict:
    loaded = sorted(set(_safe_list(loaded_tickers)))
    failed = sorted(set(_safe_list(failed_tickers)))

    if master_df is None or master_df.empty:
        summary = {
            "status": "missing_full_universe",
            "master_count": 0,
            "loaded_count": len(loaded),
            "failed_count": len(failed),
            "missing_count": 0,
            "coverage": 0.0,
            "source_mode": source_mode,
            "message": "File master universe belum ada atau kosong.",
        }
        return {
            "summary": summary,
            "missing_df": pd.DataFrame(columns=["ticker"]),
            "failed_df": pd.DataFrame({"ticker": failed}),
            "loaded_df": pd.DataFrame({"ticker": loaded}),
            "coverage_df": pd.DataFrame(),
        }

    master = sorted(set(master_df["ticker"].map(_norm_ticker)))
    master_set = set(master)
    loaded_set = set(loaded)
    failed_set = set(failed)

    missing = sorted(master_set - loaded_set)
    only_loaded = sorted(loaded_set - master_set)
    valid_loaded = sorted(loaded_set & master_set)

    coverage = len(valid_loaded) / max(len(master_set), 1)

    if coverage < 0.60:
        status = "critical_incomplete"
        message = "Loaded tickers masih jauh di bawah master universe."
    elif coverage < 0.85:
        status = "incomplete"
        message = "Loaded tickers belum penuh. Masih ada gap universe."
    else:
        status = "mostly_ok"
        message = "Coverage universe sudah lumayan, tapi tetap cek failed/missing."

    summary = {
        "status": status,
        "master_count": len(master_set),
        "loaded_count": len(valid_loaded),
        "failed_count": len(failed_set),
        "missing_count": len(missing),
        "extra_loaded_count": len(only_loaded),
        "coverage": round(coverage, 4),
        "source_mode": source_mode,
        "message": message,
    }

    missing_df = pd.DataFrame({"ticker": missing}).merge(master_df, how="left", on="ticker")
    failed_df = pd.DataFrame({"ticker": sorted(failed_set)})
    loaded_df = pd.DataFrame({"ticker": valid_loaded}).merge(master_df, how="left", on="ticker")
    extra_loaded_df = pd.DataFrame({"ticker": only_loaded})

    coverage_df = pd.DataFrame(
        [
            ("master", len(master_set)),
            ("loaded_valid", len(valid_loaded)),
            ("missing", len(missing)),
            ("failed", len(failed_set)),
            ("extra_loaded", len(only_loaded)),
        ],
        columns=["bucket", "count"],
    )

    return {
        "summary": summary,
        "missing_df": missing_df,
        "failed_df": failed_df,
        "loaded_df": loaded_df,
        "extra_loaded_df": extra_loaded_df,
        "coverage_df": coverage_df,
    }


def render_universe_audit_markdown(summary: dict) -> str:
    cov = 100.0 * float(summary.get("coverage", 0.0))
    return (
        f"**Universe audit**\n\n"
        f"- Status: `{summary.get('status','unknown')}`\n"
        f"- Source mode: `{summary.get('source_mode','unknown')}`\n"
        f"- Master tickers: `{summary.get('master_count',0)}`\n"
        f"- Loaded valid: `{summary.get('loaded_count',0)}`\n"
        f"- Missing: `{summary.get('missing_count',0)}`\n"
        f"- Failed: `{summary.get('failed_count',0)}`\n"
        f"- Coverage: `{cov:.1f}%`\n\n"
        f"{summary.get('message','')}"
    )
