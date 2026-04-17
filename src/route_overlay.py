from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import math
import pandas as pd


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if pd.isna(x):
        return lo
    return max(lo, min(hi, float(x)))


def _safe(v: Optional[float], default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


@dataclass
class RouteState:
    primary: str
    alt: str
    invalidator: str
    bias: str
    position_cap: float
    long_allowed: bool
    short_allowed: bool
    market_regime: str
    execution_mode: str
    most_hated_clear_count: int = 0
    most_hated_active: bool = False
    catalyst_window_score: float = 0.0
    analog_label: str = "unknown"
    scenario_family: str = "unknown"


def derive_route_state(market_regime: str,
                       execution_mode: str,
                       market_bias_score: float,
                       most_hated_clear_count: int = 0,
                       catalyst_window_score: float = 0.0,
                       analog_label: str = "unknown",
                       scenario_family: str = "unknown") -> RouteState:
    regime = (market_regime or "UNKNOWN").upper()
    exec_mode = (execution_mode or "SELECTIVE").upper()
    bias_score = _safe(market_bias_score)

    if regime in {"RISK_ON", "Q1", "Q2"}:
        primary = "risk_on_rotation"
        alt = "quality_growth_breakout"
        invalidator = "risk_off_reversal"
        bias = "LONG_BIAS"
        long_allowed, short_allowed = True, False
    elif regime in {"RISK_OFF", "Q3", "Q4"}:
        primary = "defensive_capital_preservation"
        alt = "commodity_or_gold_barbell"
        invalidator = "breadth_recovery"
        bias = "DEFENSIVE_BIAS" if bias_score >= -0.15 else "SHORT_BIAS"
        long_allowed, short_allowed = bias != "SHORT_BIAS", True
    else:
        primary = "selective_stock_picking"
        alt = "wait_for_confirmation"
        invalidator = "broad_breakdown"
        bias = "MIXED_BIAS"
        long_allowed, short_allowed = True, True

    position_cap = 1.0
    if exec_mode == "DEFENSIVE":
        position_cap = 0.35
    elif exec_mode == "SELECTIVE":
        position_cap = 0.55
    elif exec_mode == "AGGRESSIVE":
        position_cap = 0.85

    mh_active = most_hated_clear_count >= 3 and bias_score > 0
    if mh_active:
        primary = "relief_squeeze_rotation"
        alt = primary if alt == "quality_growth_breakout" else alt
        bias = "TACTICAL_RISK_ON"
        long_allowed = True
        short_allowed = False
        position_cap = max(position_cap, 0.7)

    return RouteState(
        primary=primary,
        alt=alt,
        invalidator=invalidator,
        bias=bias,
        position_cap=position_cap,
        long_allowed=long_allowed,
        short_allowed=short_allowed,
        market_regime=regime,
        execution_mode=exec_mode,
        most_hated_clear_count=int(most_hated_clear_count or 0),
        most_hated_active=mh_active,
        catalyst_window_score=_clip(catalyst_window_score),
        analog_label=analog_label or "unknown",
        scenario_family=scenario_family or "unknown",
    )


def build_route_overlay(scan_df: pd.DataFrame,
                        route_state: RouteState,
                        events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = scan_df.copy()
    if df.empty:
        return df

    event_score = 0.0
    event_headline = "none"
    if events_df is not None and not events_df.empty:
        e = events_df.copy()
        if "catalyst_score" in e.columns:
            event_score = float(pd.to_numeric(e["catalyst_score"], errors="coerce").fillna(0).max())
        if "title" in e.columns:
            try:
                event_headline = str(e.sort_values("catalyst_score", ascending=False).iloc[0]["title"])
            except Exception:
                event_headline = str(e.iloc[0]["title"])

    df["route_primary"] = route_state.primary
    df["route_alt"] = route_state.alt
    df["route_invalidator"] = route_state.invalidator
    df["route_bias"] = route_state.bias
    df["position_cap"] = route_state.position_cap
    df["long_allowed"] = route_state.long_allowed
    df["short_allowed"] = route_state.short_allowed
    df["market_regime"] = route_state.market_regime
    df["execution_mode"] = route_state.execution_mode
    df["most_hated_clear_count"] = route_state.most_hated_clear_count
    df["most_hated_active"] = route_state.most_hated_active
    df["catalyst_window_score"] = max(route_state.catalyst_window_score, event_score)
    df["top_catalyst_title"] = event_headline
    df["analog_label"] = route_state.analog_label
    df["scenario_family"] = route_state.scenario_family

    # soft route-fit score
    route_fit = []
    for _, row in df.iterrows():
        verdict = str(row.get("verdict", "")).upper()
        sector = str(row.get("sector", "Unknown"))
        dry = _safe(row.get("dry_score_final", row.get("dry_score", 0.0)), 0.0) / 100.0
        risk = _safe(row.get("risk_rank_score", row.get("false_breakout_risk", 50.0)), 50.0) / 100.0
        breakout = _safe(row.get("breakout_integrity", row.get("breakout_integrity_score", 50.0)), 50.0) / 100.0
        rel = _safe(row.get("relative_strength_20d"), 0.0)
        score = 0.5
        if route_state.bias in {"LONG_BIAS", "TACTICAL_RISK_ON"}:
            score += 0.20 if verdict in {"READY_LONG", "WATCH"} else -0.10
            score += 0.15 * _clip(dry)
            score += 0.10 * _clip(breakout)
            score -= 0.15 * _clip(risk)
            score += 0.10 if rel > 0 else -0.05
        elif route_state.bias in {"SHORT_BIAS", "DEFENSIVE_BIAS"}:
            score += 0.20 if verdict in {"TRIM", "AVOID", "WATCH_REBOUND"} else -0.05
            score += 0.12 * _clip(risk)
            score -= 0.10 * _clip(dry)
            score -= 0.05 if sector.lower() in {"banks", "consumer discretionary"} else 0.0
        else:
            score += 0.08 if verdict in {"WATCH", "READY_LONG"} else 0.0
            score -= 0.08 if verdict in {"TRIM", "AVOID"} else 0.0
        route_fit.append(_clip(score))
    df["route_fit_score"] = route_fit

    def _radar_bucket(row: pd.Series) -> str:
        verdict = str(row.get("verdict", "")).upper()
        fit = _safe(row.get("route_fit_score"), 0.0)
        b = _safe(row.get("breakout_integrity_score"), 0.0)
        if verdict == "READY_LONG" and fit >= 0.6:
            return "ACTIVE"
        if verdict in {"WATCH", "WATCH_REBOUND"} and fit >= 0.55 and b >= 0.45:
            return "NEAR_TRIGGER"
        return "NOT_YET"

    df["forward_radar_bucket"] = df.apply(_radar_bucket, axis=1)
    return df
