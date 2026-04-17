from __future__ import annotations

import math
from typing import Optional

import pandas as pd


def _safe(v: Optional[float], default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def build_rank_scores(scan_df: pd.DataFrame) -> pd.DataFrame:
    df = scan_df.copy()
    if df.empty:
        return df

    long_scores = []
    risk_scores = []
    route_scores = []

    for _, row in df.iterrows():
        acc = _safe(row.get("accumulation_quality_score"), 50)
        brk = _safe(row.get("breakout_integrity_score"), 50)
        dry = _safe(row.get("dry_score"), 50)
        wet = _safe(row.get("wet_score"), 50)
        dist = _safe(row.get("distribution_risk_score"), 50)
        route_fit = _safe(row.get("route_fit_score"), 0.5) * 100.0
        conf = _safe(row.get("score_confidence"), 0.5) * 100.0
        catalyst = _safe(row.get("catalyst_window_score"), 0.0) * 100.0
        rel = _safe(row.get("relative_strength_20d"), 0.0)
        rel_norm = max(0.0, min(100.0, 50.0 + rel * 250.0))
        broker_align = _safe(row.get("broker_alignment_score"), 50)
        burst = max(_safe(row.get("bullish_burst_score"), 0), _safe(row.get("bear_trap_score"), 0))
        trap = max(_safe(row.get("bull_trap_score"), 0), _safe(row.get("bearish_burst_score"), 0))

        long_rank = (
            0.18 * acc +
            0.16 * brk +
            0.10 * dry +
            0.10 * rel_norm +
            0.12 * route_fit +
            0.08 * catalyst +
            0.10 * conf +
            0.08 * broker_align +
            0.08 * (burst * 100.0) -
            0.12 * dist -
            0.08 * wet -
            0.06 * (trap * 100.0)
        )

        risk_rank = (
            0.22 * dist +
            0.14 * wet +
            0.12 * (trap * 100.0) +
            0.10 * (100.0 - conf) +
            0.08 * (100.0 - route_fit) +
            0.08 * max(0.0, 50.0 - rel_norm) +
            0.08 * max(0.0, 50.0 - broker_align) +
            0.06 * max(0.0, 50.0 - acc) +
            0.06 * max(0.0, 50.0 - brk) +
            0.06 * max(0.0, 50.0 - dry)
        )

        route_rank = (
            0.45 * route_fit +
            0.20 * catalyst +
            0.15 * conf +
            0.10 * rel_norm +
            0.10 * broker_align
        )

        long_scores.append(_clip(long_rank))
        risk_scores.append(_clip(risk_rank))
        route_scores.append(_clip(route_rank))

    df["long_rank_score"] = long_scores
    df["risk_rank_score"] = risk_scores
    df["route_rank_score"] = route_scores

    df["rank_bucket"] = pd.cut(
        df["long_rank_score"],
        bins=[-1, 40, 55, 70, 85, 101],
        labels=["LOW", "WATCHLIST", "GOOD", "STRONG", "TOP"],
    ).astype(str)
    return df
