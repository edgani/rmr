# IDX EOD Scanner V4.6 — Route-Aware Ranking + Validation Patch

This patch adds the two missing higher-value blocks:
1. route-aware next-play ranking
2. walk-forward / false-signal validation helpers

What this is:
- a patch pack you can drop into the current clean scanner repo
- Python modules for route overlay, ranking, and validation
- designed to be import-safe and deploy-safe

What this is not:
- a claim of live alpha
- a substitute for real broker / done detail / orderbook data
- a finished production validation report

## Files
- `src/route_overlay.py`
- `src/ranking.py`
- `src/validation.py`
- `tests/test_route_validation.py`
- `data/example_route_events.csv`

## Integration target
Import these modules into the current scanner app and call:
- `build_route_overlay(...)`
- `build_rank_scores(...)`
- `run_walk_forward_validation(...)`

## Notes
- Thresholds are rule-based and should be recalibrated on real data.
- Walk-forward is scaffolded for honest evaluation, not curve-fit optimization.
