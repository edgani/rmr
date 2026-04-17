from __future__ import annotations


def liquidity_bucket(liq_mn: float) -> str:
    if liq_mn < 10:
        return "micro"
    if liq_mn < 50:
        return "small"
    if liq_mn < 200:
        return "mid"
    return "large"


def regime_boost(regime: str) -> float:
    regime = str(regime or "CHOPPY")
    if regime == "RISK_ON":
        return 1.0
    if regime == "UPTREND_SELECTIVE":
        return 0.6
    if regime == "CHOPPY":
        return 0.0
    if regime == "DOWNTREND_SELECTIVE":
        return -0.6
    if regime == "RISK_OFF":
        return -1.0
    return 0.0


def get_long_thresholds(liq_mn: float, regime: str) -> dict:
    bucket = liquidity_bucket(liq_mn)
    # more demanding for thinner names; easier in strong market
    base = {
        "micro": {"tq": 78, "bi": 66, "fb": 28, "dry": 58, "broker": 60},
        "small": {"tq": 72, "bi": 60, "fb": 32, "dry": 55, "broker": 57},
        "mid": {"tq": 66, "bi": 55, "fb": 35, "dry": 52, "broker": 55},
        "large": {"tq": 60, "bi": 50, "fb": 38, "dry": 48, "broker": 50},
    }[bucket].copy()
    boost = regime_boost(regime)
    base["tq"] -= 3 * boost
    base["bi"] -= 4 * boost
    base["fb"] += 3 * boost
    base["dry"] -= 2 * boost
    base["broker"] -= 2 * boost
    return base


def get_risk_thresholds(liq_mn: float, regime: str) -> dict:
    bucket = liquidity_bucket(liq_mn)
    base = {
        "micro": {"wet": 62, "overhang": 55, "fb": 45, "broker_low": 42},
        "small": {"wet": 60, "overhang": 55, "fb": 45, "broker_low": 40},
        "mid": {"wet": 58, "overhang": 52, "fb": 42, "broker_low": 38},
        "large": {"wet": 55, "overhang": 48, "fb": 40, "broker_low": 35},
    }[bucket].copy()
    boost = regime_boost(regime)
    # stricter in risk-off
    base["wet"] -= 2 * boost
    base["overhang"] -= 2 * boost
    base["fb"] -= 2 * boost
    base["broker_low"] += 2 * boost
    return base
