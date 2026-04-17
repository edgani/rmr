from __future__ import annotations

import pandas as pd

from src.route_overlay import derive_route_state, build_route_overlay
from src.ranking import build_rank_scores
from src.validation import build_forward_labels, run_walk_forward_validation


def test_route_overlay_and_ranking():
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "verdict": ["READY_LONG", "WATCH"],
        "sector": ["Banks", "Energy"],
        "dry_score": [70, 55],
        "wet_score": [20, 35],
        "accumulation_quality_score": [72, 61],
        "breakout_integrity_score": [75, 58],
        "distribution_risk_score": [22, 40],
        "relative_strength_20d": [0.10, -0.03],
        "score_confidence": [0.8, 0.6],
        "broker_alignment_score": [68, 44],
    })
    state = derive_route_state("RISK_ON", "AGGRESSIVE", 0.25, most_hated_clear_count=3)
    out = build_route_overlay(df, state)
    out = build_rank_scores(out)
    assert "route_primary" in out.columns
    assert "long_rank_score" in out.columns
    assert out["long_rank_score"].max() <= 100


def test_validation_scaffold():
    rows = []
    dates = pd.date_range("2022-01-01", periods=700, freq="B")
    for t in ["AAA", "BBB"]:
        px = 100.0
        for i, d in enumerate(dates):
            px *= 1.0005 if t == "AAA" else 1.0
            rows.append({"date": d, "ticker": t, "close": px, "long_rank_score": 60 if t == "AAA" else 40})
    df = pd.DataFrame(rows)
    labeled = build_forward_labels(df, horizon=20, up_target=0.005)
    res = run_walk_forward_validation(labeled, top_n=5)
    assert hasattr(res, "fold_metrics")
