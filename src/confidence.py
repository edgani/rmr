from __future__ import annotations

import numpy as np
import pandas as pd


def compute_confidence(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "data_completeness_score", "module_agreement_score", "score_confidence"])
    out = df[["ticker"]].copy()
    broker_present = df.get("broker_alignment_score", pd.Series(0, index=df.index)).notna().astype(float)
    burst_present = df.get("latest_event_label", pd.Series("", index=df.index)).astype(str).ne("").astype(float)
    orderbook_present = df.get("bid_stack_quality", pd.Series(np.nan, index=df.index)).notna().astype(float)
    price_present = pd.Series(1.0, index=df.index)

    completeness = 100 * (0.45 * price_present + 0.25 * broker_present + 0.20 * burst_present + 0.10 * orderbook_present)

    verdict = df.get("verdict", pd.Series("NEUTRAL", index=df.index)).astype(str)
    phase = df.get("phase", pd.Series("NEUTRAL", index=df.index)).astype(str)
    burst_bias = df.get("burst_bias", pd.Series("NEUTRAL", index=df.index)).astype(str)
    broker_mode = df.get("broker_mode", pd.Series("BALANCED", index=df.index)).astype(str)

    agreement = []
    for i in range(len(df)):
        score = 0.0
        if verdict.iloc[i] in {"READY_LONG", "WATCH"} and phase.iloc[i] in {"MARKUP", "ACCUMULATION", "PULLBACK_HEALTHY"}:
            score += 0.4
        elif verdict.iloc[i] in {"TRIM", "AVOID"} and phase.iloc[i] in {"MARKDOWN"}:
            score += 0.4
        else:
            score += 0.2
        if burst_bias.iloc[i] == "BULLISH" and verdict.iloc[i] in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
            score += 0.3
        elif burst_bias.iloc[i] == "BEARISH" and verdict.iloc[i] in {"TRIM", "AVOID"}:
            score += 0.3
        else:
            score += 0.1
        if broker_mode.iloc[i] == "ACCUMULATION_DOMINANT" and verdict.iloc[i] in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
            score += 0.3
        elif broker_mode.iloc[i] == "DISTRIBUTION_DOMINANT" and verdict.iloc[i] in {"TRIM", "AVOID"}:
            score += 0.3
        else:
            score += 0.1
        agreement.append(score)
    agreement = 100 * np.clip(np.array(agreement), 0, 1)
    confidence = 0.55 * completeness + 0.45 * agreement
    out["data_completeness_score"] = np.round(completeness, 1)
    out["module_agreement_score"] = np.round(agreement, 1)
    out["score_confidence"] = np.round(np.clip(confidence, 0, 100), 1)
    return out
