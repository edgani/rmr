from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_LABELS = {"", "-", "NO_INTRADAY_SIGNAL", "NO_BROKER_UPLOAD"}


def _present_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").notna().astype(float)


def compute_confidence(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "data_completeness_score", "module_agreement_score", "score_confidence"])

    out = df[["ticker"]].copy()
    price_present = pd.Series(1.0, index=df.index)

    broker_mode = df.get("broker_mode", pd.Series("NO_BROKER_UPLOAD", index=df.index)).astype(str)
    broker_present = (~broker_mode.isin(["NO_BROKER_UPLOAD", "UNKNOWN", ""])) .astype(float)

    latest_event = df.get("latest_event_label", pd.Series("NO_INTRADAY_SIGNAL", index=df.index)).astype(str)
    burst_present = (~latest_event.isin(MISSING_LABELS)).astype(float)
    order_present = _present_numeric(df.get("bid_stack_quality", pd.Series(np.nan, index=df.index)))

    broker_fields = [
        _present_numeric(df.get("institutional_support", pd.Series(np.nan, index=df.index))),
        _present_numeric(df.get("institutional_resistance", pd.Series(np.nan, index=df.index))),
        _present_numeric(df.get("broker_alignment_score", pd.Series(np.nan, index=df.index))),
        _present_numeric(df.get("broker_persistence_score", pd.Series(np.nan, index=df.index))),
        _present_numeric(df.get("broker_concentration_score", pd.Series(np.nan, index=df.index))),
        _present_numeric(df.get("broker_acc_pressure", pd.Series(np.nan, index=df.index))),
    ]
    broker_field_present = sum(broker_fields) / max(len(broker_fields), 1)

    complete = (
        0.30 * price_present
        + 0.20 * broker_present
        + 0.20 * broker_field_present
        + 0.15 * burst_present
        + 0.15 * order_present
    )
    completeness = 100 * complete

    verdict = df.get("verdict", pd.Series("NEUTRAL", index=df.index)).astype(str)
    phase = df.get("phase", pd.Series("NEUTRAL", index=df.index)).astype(str)
    burst_bias = df.get("burst_bias", pd.Series("NEUTRAL", index=df.index)).astype(str)
    market_regime = df.get("market_regime", pd.Series("CHOPPY", index=df.index)).astype(str)
    rs20 = pd.to_numeric(df.get("relative_strength_20d", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0.0)
    drywet_state = df.get("drywet_state", pd.Series("NEUTRAL", index=df.index)).astype(str)
    broker_mode = broker_mode.fillna("BALANCED")

    agreement = []
    for i in range(len(df)):
        score = 0.0
        v = verdict.iloc[i]
        p = phase.iloc[i]
        b = burst_bias.iloc[i]
        bm = broker_mode.iloc[i]
        rg = market_regime.iloc[i]
        rs = rs20.iloc[i]
        dw = drywet_state.iloc[i]

        if v in {"READY_LONG", "WATCH"} and p in {"MARKUP", "ACCUMULATION", "PULLBACK_HEALTHY"}:
            score += 0.24
        elif v in {"TRIM", "AVOID"} and p == "MARKDOWN":
            score += 0.24
        else:
            score += 0.10

        if b == "BULLISH" and v in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
            score += 0.15
        elif b == "BEARISH" and v in {"TRIM", "AVOID"}:
            score += 0.15
        else:
            score += 0.05

        if bm == "ACCUMULATION_DOMINANT" and v in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
            score += 0.18
        elif bm == "DISTRIBUTION_DOMINANT" and v in {"TRIM", "AVOID"}:
            score += 0.18
        elif bm == "CHURN_HEAVY" and v in {"WATCH", "NEUTRAL"}:
            score += 0.12
        else:
            score += 0.05

        if dw in {"DRY", "SEMI_DRY"} and v in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
            score += 0.12
        elif dw in {"WET", "SEMI_WET"} and v in {"TRIM", "AVOID"}:
            score += 0.12
        else:
            score += 0.04

        if rg in {"RISK_ON", "UPTREND_SELECTIVE"} and v in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
            score += 0.15
        elif rg in {"RISK_OFF", "DOWNTREND_SELECTIVE"} and v in {"TRIM", "AVOID"}:
            score += 0.15
        else:
            score += 0.05

        if (v in {"READY_LONG", "WATCH"} and rs >= 0) or (v in {"TRIM", "AVOID"} and rs <= 0):
            score += 0.16
        else:
            score += 0.05
        agreement.append(score)

    agreement = 100 * np.clip(np.array(agreement), 0, 1)
    strong_verdict_penalty = np.where(verdict.isin(["READY_LONG", "TRIM", "AVOID"]) & (completeness < 70), 18.0, 0.0)
    broker_missing_penalty = np.where(verdict.isin(["READY_LONG", "TRIM", "AVOID"]) & broker_present.eq(0), 12.0, 0.0)
    confidence = 0.48 * completeness + 0.52 * agreement - strong_verdict_penalty - broker_missing_penalty

    out["data_completeness_score"] = np.round(completeness, 1)
    out["module_agreement_score"] = np.round(agreement, 1)
    out["score_confidence"] = np.round(np.clip(confidence, 0, 100), 1)
    return out
