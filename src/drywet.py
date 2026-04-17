from __future__ import annotations

import numpy as np
import pandas as pd


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_broker_aware_drywet(price_scan_df: pd.DataFrame, broker_ctx: pd.DataFrame) -> pd.DataFrame:
    if price_scan_df is None or price_scan_df.empty:
        return pd.DataFrame(columns=["ticker", "dry_score_final", "wet_score_final"])
    base = price_scan_df[["ticker", "dry_score", "wet_score", "liquidity_mn"]].copy()
    if broker_ctx is None or broker_ctx.empty:
        base["dry_score_final"] = base["dry_score"]
        base["wet_score_final"] = base["wet_score"]
        return base[["ticker", "dry_score_final", "wet_score_final"]]
    out = base.merge(
        broker_ctx[["ticker", "broker_alignment_score", "overhang_score", "broker_concentration_score", "broker_mode"]],
        on="ticker", how="left"
    )
    out[["broker_alignment_score", "overhang_score", "broker_concentration_score"]] = out[["broker_alignment_score", "overhang_score", "broker_concentration_score"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    align_boost = out["broker_alignment_score"] / 100.0
    concentration = out["broker_concentration_score"] / 100.0
    overhang = out["overhang_score"] / 100.0
    dry_final = out["dry_score"] * (0.70 + 0.15 * align_boost + 0.15 * concentration) - 20 * overhang
    wet_final = out["wet_score"] * (0.75 + 0.25 * overhang)
    out["dry_score_final"] = np.clip(dry_final, 0, 100).round(1)
    out["wet_score_final"] = np.clip(wet_final, 0, 100).round(1)
    return out[["ticker", "dry_score_final", "wet_score_final"]]
