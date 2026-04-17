from __future__ import annotations

import numpy as np
import pandas as pd


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_broker_aware_drywet(price_scan_df: pd.DataFrame, broker_ctx: pd.DataFrame) -> pd.DataFrame:
    if price_scan_df is None or price_scan_df.empty:
        return pd.DataFrame(columns=[
            "ticker", "dry_score_final", "wet_score_final", "drywet_state",
            "supply_overhang_score", "float_lock_score"
        ])

    base = price_scan_df[["ticker", "dry_score", "wet_score", "liquidity_mn"]].copy()
    if broker_ctx is None or broker_ctx.empty:
        base["dry_score_final"] = base["dry_score"]
        base["wet_score_final"] = base["wet_score"]
        base["drywet_state"] = np.where(base["dry_score"] >= 60, "DRY", np.where(base["wet_score"] >= 60, "WET", "NEUTRAL"))
        base["supply_overhang_score"] = 0.0
        base["float_lock_score"] = 0.0
        return base[["ticker", "dry_score_final", "wet_score_final", "drywet_state", "supply_overhang_score", "float_lock_score"]]

    use_cols = [
        "ticker", "broker_alignment_score", "overhang_score", "broker_concentration_score",
        "broker_mode", "broker_persistence_score", "broker_acc_pressure", "broker_dist_pressure",
        "accumulator_breadth", "distributor_breadth"
    ]
    use_cols = [c for c in use_cols if c in broker_ctx.columns]
    out = base.merge(broker_ctx[use_cols], on="ticker", how="left")
    num_cols = [c for c in out.columns if c not in {"ticker", "broker_mode"}]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    out[num_cols] = out[num_cols].fillna(0.0)

    align = out["broker_alignment_score"] / 100.0
    conc = out["broker_concentration_score"] / 100.0
    overhang = out["overhang_score"] / 100.0
    persistence = out.get("broker_persistence_score", pd.Series(0.0, index=out.index)) / 100.0
    acc_pressure = out.get("broker_acc_pressure", pd.Series(0.0, index=out.index)) / 100.0
    dist_pressure = out.get("broker_dist_pressure", pd.Series(0.0, index=out.index)) / 100.0
    acc_breadth = pd.to_numeric(out.get("accumulator_breadth", 0.0), errors="coerce").fillna(0.0)
    dist_breadth = pd.to_numeric(out.get("distributor_breadth", 0.0), errors="coerce").fillna(0.0)
    breadth_balance = (acc_breadth - dist_breadth).clip(-10, 10) / 10.0

    float_lock_score = 100 * np.clip(0.45 * conc + 0.30 * persistence + 0.25 * acc_pressure, 0, 1)
    supply_overhang_score = 100 * np.clip(0.50 * overhang + 0.25 * dist_pressure + 0.25 * np.maximum(-breadth_balance, 0), 0, 1)

    dry_final = (
        out["dry_score"] * (0.55 + 0.15 * align + 0.15 * conc + 0.15 * persistence)
        + 10 * np.maximum(breadth_balance, 0)
        - 22 * overhang
        - 10 * dist_pressure
    )
    wet_final = (
        out["wet_score"] * (0.55 + 0.20 * overhang + 0.15 * dist_pressure + 0.10 * np.maximum(-breadth_balance, 0))
        + 10 * overhang
    )

    out["dry_score_final"] = np.clip(dry_final, 0, 100).round(1)
    out["wet_score_final"] = np.clip(wet_final, 0, 100).round(1)
    out["float_lock_score"] = np.round(float_lock_score, 1)
    out["supply_overhang_score"] = np.round(supply_overhang_score, 1)

    conditions = [
        out["dry_score_final"] >= 72,
        (out["dry_score_final"] >= 58) & (out["dry_score_final"] < 72),
        out["wet_score_final"] >= 72,
        (out["wet_score_final"] >= 58) & (out["wet_score_final"] < 72),
    ]
    choices = ["DRY", "SEMI_DRY", "WET", "SEMI_WET"]
    out["drywet_state"] = np.select(conditions, choices, default="NEUTRAL")
    return out[["ticker", "dry_score_final", "wet_score_final", "drywet_state", "supply_overhang_score", "float_lock_score"]]
