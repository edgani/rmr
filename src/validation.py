from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: Dict[str, float]


def build_forward_labels(price_df: pd.DataFrame,
                         horizon: int = 20,
                         up_target: float = 0.08,
                         down_target: float = -0.08,
                         price_col: str = "close") -> pd.DataFrame:
    df = price_df.copy().sort_values(["ticker", "date"])
    out = []
    for t, g in df.groupby("ticker", sort=False):
        g = g.copy()
        px = pd.to_numeric(g[price_col], errors="coerce")
        fut = px.shift(-horizon) / px - 1.0
        g["fwd_return"] = fut
        g["label_long_success"] = (g["fwd_return"] >= up_target).astype(int)
        g["label_down_break"] = (g["fwd_return"] <= down_target).astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _precision_at_top(y_true: np.ndarray, y_score: np.ndarray, top_n: int) -> float:
    if len(y_true) == 0:
        return np.nan
    idx = np.argsort(-y_score)[:max(1, min(top_n, len(y_true)))]
    return float(np.mean(y_true[idx]))


def _expectancy(y_true: np.ndarray, fwd_returns: np.ndarray, y_score: np.ndarray, top_n: int) -> float:
    if len(fwd_returns) == 0:
        return np.nan
    idx = np.argsort(-y_score)[:max(1, min(top_n, len(fwd_returns)))]
    return float(np.nanmean(fwd_returns[idx]))


def run_walk_forward_validation(scan_with_labels: pd.DataFrame,
                                score_col: str = "long_rank_score",
                                label_col: str = "label_long_success",
                                return_col: str = "fwd_return",
                                date_col: str = "date",
                                train_days: int = 252 * 2,
                                test_days: int = 63,
                                top_n: int = 20) -> ValidationResult:
    df = scan_with_labels.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    if df.empty:
        return ValidationResult(pd.DataFrame(), pd.DataFrame(), {})

    dates = pd.Series(sorted(df[date_col].dropna().unique()))
    fold_rows: List[Dict] = []
    pred_rows: List[pd.DataFrame] = []

    start_idx = train_days
    fold = 0
    while start_idx + test_days <= len(dates):
        fold += 1
        train_start = dates.iloc[start_idx - train_days]
        train_end = dates.iloc[start_idx - 1]
        test_end = dates.iloc[start_idx + test_days - 1]

        train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        test_mask = (df[date_col] > train_end) & (df[date_col] <= test_end)
        test_df = df.loc[test_mask].copy()
        if test_df.empty:
            start_idx += test_days
            continue

        y_true = pd.to_numeric(test_df[label_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_score = pd.to_numeric(test_df[score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        fwd_ret = pd.to_numeric(test_df[return_col], errors="coerce").to_numpy(dtype=float)

        auc = _safe_auc(y_true, y_score)
        p_top = _precision_at_top(y_true, y_score, top_n=top_n)
        exp_top = _expectancy(y_true, fwd_ret, y_score, top_n=top_n)

        fold_rows.append({
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_end": test_end,
            "test_rows": int(len(test_df)),
            "auc": auc,
            f"precision_at_{top_n}": p_top,
            f"expectancy_at_{top_n}": exp_top,
        })

        pred_rows.append(test_df[[date_col, "ticker", score_col, label_col, return_col]].assign(fold=fold))
        start_idx += test_days

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = {}
    if not fold_metrics.empty:
        metric_cols = [c for c in fold_metrics.columns if c in {"auc", f"precision_at_{top_n}", f"expectancy_at_{top_n}"}]
        for c in metric_cols:
            summary[c] = float(pd.to_numeric(fold_metrics[c], errors="coerce").mean())
    return ValidationResult(fold_metrics=fold_metrics, predictions=predictions, summary=summary)
