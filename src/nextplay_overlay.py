
from __future__ import annotations
import math
from typing import Dict, List
import pandas as pd
import numpy as np

def _safe_mean(vals):
    vals = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else 0.0

def compute_macro_overlay(price_df: pd.DataFrame, scan_df: pd.DataFrame) -> Dict:
    # price-side only proxy overlay for deploy-safe scanner
    def _ret(ticker: str, n: int = 20):
        g = price_df[price_df["ticker"] == ticker].sort_values("date")
        if len(g) < n + 1:
            return float("nan")
        b = float(g["close"].iloc[-(n+1)])
        return float(g["close"].iloc[-1] / b - 1.0) if b else float("nan")

    breadth = float((scan_df["ret_20d"] > 0).mean()) if "ret_20d" in scan_df.columns and len(scan_df) else 0.5
    market_return = _safe_mean(scan_df["ret_20d"].tolist()) if "ret_20d" in scan_df.columns else 0.0
    banks = scan_df[scan_df["sector"].str.contains("Bank", case=False, na=False)]
    commodities = scan_df[scan_df["sector"].str.contains("Energy|Metal|Coal|Mining", case=False, na=False)]
    consumer = scan_df[scan_df["sector"].str.contains("Consumer|Retail", case=False, na=False)]

    bank_lead = _safe_mean(banks["ret_20d"].tolist()) if len(banks) else 0.0
    comm_lead = _safe_mean(commodities["ret_20d"].tolist()) if len(commodities) else 0.0
    consumer_lead = _safe_mean(consumer["ret_20d"].tolist()) if len(consumer) else 0.0

    stress = max(0.0, 0.5 - breadth) + max(0.0, -market_return)
    risk_on = max(0.0, breadth - 0.5) + max(0.0, market_return)
    slowdown_flags = 1.0 if breadth < 0.4 and market_return < -0.03 else (0.5 if breadth < 0.48 else 0.0)
    inf_shock = 1.0 if comm_lead > 0.08 and bank_lead < 0.03 else (0.5 if comm_lead > 0.05 else 0.0)

    return {
        "market_breadth_pct": breadth,
        "market_bias_score": risk_on - stress,
        "bank_lead_20d": bank_lead,
        "commodity_lead_20d": comm_lead,
        "consumer_lead_20d": consumer_lead,
        "slowdown_flags": slowdown_flags,
        "inf_shock": inf_shock,
        "data_source_mode": "Price-side only",
        "macro_source_quality": 0.35,
        "market_return_20d": market_return,
    }

def derive_route_state(overlay: Dict) -> Dict:
    breadth = float(overlay.get("market_breadth_pct", 0.5))
    bias = float(overlay.get("market_bias_score", 0.0))
    shock = float(overlay.get("inf_shock", 0.0))
    slowdown = float(overlay.get("slowdown_flags", 0.0))

    if breadth < 0.32 and bias < -0.08:
        route = "panic_crash"
        alt = "deflationary_riskoff"
        invalid = "vshape_rebound"
        pos_cap = 0.25
    elif slowdown > 0.7 and shock < 0.4:
        route = "deflationary_riskoff"
        alt = "quality_disinflation"
        invalid = "reflation_reaccel"
        pos_cap = 0.50
    elif shock > 0.7:
        route = "stagflation_persist"
        alt = "growth_scare"
        invalid = "quality_disinflation"
        pos_cap = 0.60
    elif breadth > 0.58 and bias > 0.04:
        route = "reflation_reaccel"
        alt = "quality_disinflation"
        invalid = "deflationary_riskoff"
        pos_cap = 1.00
    elif breadth > 0.52 and bias > 0.0:
        route = "quality_disinflation"
        alt = "reflation_reaccel"
        invalid = "growth_scare"
        pos_cap = 0.75
    else:
        route = "growth_scare"
        alt = "quality_disinflation"
        invalid = "stagflation_persist"
        pos_cap = 0.50

    route_bias = "risk-on" if route in {"reflation_reaccel", "quality_disinflation"} else ("risk-off" if route in {"panic_crash","deflationary_riskoff"} else "mixed")
    return {
        "route_state": route,
        "alt_route": alt,
        "invalidator_route": invalid,
        "route_bias": route_bias,
        "position_cap": pos_cap,
        "position_cap_label": f"{pos_cap:.2f}x",
        "long_allowed": route not in {"panic_crash"},
        "short_allowed": route in {"panic_crash","deflationary_riskoff","growth_scare","stagflation_persist"},
        "execution_mode": "aggressive" if pos_cap >= 0.95 else ("selective" if pos_cap >= 0.60 else "defensive"),
        "route_confidence": min(0.9, 0.5 + abs(bias) + abs(breadth - 0.5)),
    }

def build_asset_translation(route_state: str, overlay: Dict, route: Dict) -> Dict:
    cap = route.get("position_cap_label", "0.50x")
    translations = {
        "reflation_reaccel": {
            "IHSG": [
                {"bias":"LONG","setup":"Banks + domestic beta","why":"Breadth membaik dan risk appetite hidup.","trigger":"Breadth > 55% dan leaders tembus high.","invalidator":"Breadth balik jatuh.","size_cap":cap},
                {"bias":"WATCH","setup":"Commodity laggards","why":"Masih bisa ikut fase kedua kalau reflation meluas.","trigger":"Commodity leaders ikut konfirmasi.","invalidator":"Leadership makin sempit.","size_cap":"0.50x"},
            ],
            "FX": [{"bias":"SHORT USD","setup":"IDR relief","why":"Risk-on branch biasanya bantu EM FX.","trigger":"Domestic beta dan breadth konfirmasi.","invalidator":"Shock minyak kembali.","size_cap":"0.50x"}],
        },
        "quality_disinflation": {
            "IHSG": [
                {"bias":"LONG","setup":"Bank quality + defensif selektif","why":"Risk-on ada, tapi jangan broad beta semua.","trigger":"Bank relative strength tetap dominan.","invalidator":"Breadth rusak lagi.","size_cap":cap},
            ],
            "FX": [{"bias":"MIXED","setup":"Carry selective","why":"Perlu breadth tetap sehat.","trigger":"Risk-on bertahan.","invalidator":"USD squeeze.","size_cap":"0.25x"}],
        },
        "stagflation_persist": {
            "IHSG": [
                {"bias":"LONG","setup":"Energy / metals selective","why":"Inflation shock bikin exporter lebih tahan.","trigger":"Commodity leadership tetap hidup.","invalidator":"Oil rollback keras.","size_cap":cap},
                {"bias":"AVOID","setup":"Domestic high beta","why":"Margin dan funding sensitif.","trigger":"N/A","invalidator":"Shock reda.","size_cap":"0.25x"},
            ],
            "FX": [{"bias":"LONG USD","setup":"Defensive dollar","why":"Shock branch bikin dollar sensitif naik.","trigger":"Risk-off meluas.","invalidator":"Shock reda cepat.","size_cap":"0.50x"}],
        },
        "deflationary_riskoff": {
            "IHSG": [{"bias":"AVOID","setup":"Broad domestic beta","why":"Growth scare dominan.","trigger":"N/A","invalidator":"Breadth pulih.","size_cap":"0.25x"}],
            "FX": [{"bias":"LONG USD","setup":"Safety","why":"Risk-off branch.","trigger":"Breadth dan returns tetap lemah.","invalidator":"V-shape rebound.","size_cap":"0.50x"}],
        },
        "growth_scare": {
            "IHSG": [{"bias":"SELECTIVE","setup":"High quality only","why":"Belum cukup sehat untuk broad beta.","trigger":"Need stronger breadth.","invalidator":"Shock naik.","size_cap":"0.50x"}],
            "FX": [{"bias":"MIXED","setup":"Wait","why":"Belum ada branch bersih.","trigger":"Route clearer.","invalidator":"N/A","size_cap":"0.25x"}],
        },
        "panic_crash": {
            "IHSG": [{"bias":"AVOID","setup":"Capital preservation","why":"Crash branch aktif.","trigger":"N/A","invalidator":"Panic reda.","size_cap":"0.25x"}],
            "FX": [{"bias":"LONG USD","setup":"Funding stress hedge","why":"Crash regime.","trigger":"Stress naik.","invalidator":"Fast relief.","size_cap":"0.50x"}],
        },
    }
    return translations.get(route_state, translations["growth_scare"])

def build_upcoming_events() -> List[Dict]:
    return [
        {"title":"US CPI", "family":"inflation", "countdown":"T-3d", "impact":"Panas = rates shock / cold = relief."},
        {"title":"Fed speaker / policy", "family":"policy", "countdown":"T-7d", "impact":"Can reprice duration and risk appetite."},
        {"title":"OJK / MSCI / domestic catalyst", "family":"flow", "countdown":"T-10d", "impact":"IHSG flow-sensitive branch."},
        {"title":"NFP", "family":"labor", "countdown":"T-16d", "impact":"Growth scare vs relief switch."},
    ]

ANALOGS = [
    {"label":"Mixed slowdown", "scenario_family":"mixed_slowdown", "path_1m":"rotation without panic", "path_3m":"selective leadership", "path_6m":"await cleaner impulse", "vector": {"breadth":0.50, "bias":0.00, "shock":0.20}},
    {"label":"Commodity shock", "scenario_family":"commodity_shock", "path_1m":"energy/metals lead", "path_3m":"dispersion broadens", "path_6m":"policy bites later", "vector": {"breadth":0.42, "bias":-0.03, "shock":0.80}},
    {"label":"Relief squeeze", "scenario_family":"relief_squeeze", "path_1m":"laggards squeeze", "path_3m":"leadership broadens if confirmed", "path_6m":"can fail if breadth weakens", "vector": {"breadth":0.60, "bias":0.06, "shock":0.10}},
    {"label":"Deflationary risk-off", "scenario_family":"riskoff", "path_1m":"quality / cash / defense", "path_3m":"beta still fragile", "path_6m":"need policy relief", "vector": {"breadth":0.30, "bias":-0.10, "shock":0.10}},
]

def match_analog(overlay: Dict) -> Dict:
    cur = np.array([overlay.get("market_breadth_pct", 0.5), overlay.get("market_bias_score", 0.0), overlay.get("inf_shock", 0.0)], dtype=float)
    best = None
    best_sim = -1.0
    for a in ANALOGS:
        v = np.array([a["vector"]["breadth"], a["vector"]["bias"], a["vector"]["shock"]], dtype=float)
        sim = float(np.dot(cur, v) / ((np.linalg.norm(cur) + 1e-9) * (np.linalg.norm(v) + 1e-9)))
        sim = max(0.0, min(1.0, 0.5 + 0.5 * sim))
        if sim > best_sim:
            best_sim = sim
            best = dict(a)
            best["similarity"] = sim
    return best

def build_most_hated_rally_monitor(overlay: Dict, route: Dict) -> Dict:
    breadth = overlay.get("market_breadth_pct", 0.5)
    bank_lead = overlay.get("bank_lead_20d", 0.0)
    comm = overlay.get("commodity_lead_20d", 0.0)
    bias = overlay.get("market_bias_score", 0.0)
    clear = 0
    clear += 1 if breadth > 0.52 else 0
    clear += 1 if bank_lead > 0.03 else 0
    clear += 1 if bias > 0.02 else 0
    clear += 1 if comm < 0.10 else 0
    state = "ACTIVE" if clear >= 4 else ("PRE-CONFIRMED" if clear >= 3 else ("TRANSITION" if clear >= 2 else "OFF"))
    return {
        "clear_count": clear,
        "state": state,
        "risk_on_switch": state in {"ACTIVE", "PRE-CONFIRMED"},
        "relief_squeeze_score": min(1.0, max(0.0, 0.25 * clear + 0.5 * max(0.0, bias))),
    }

def build_top_drivers_now(overlay: Dict, route: Dict, most_hated: Dict, analog: Dict) -> List[Dict]:
    drivers = []
    def add(label, score, tone, why):
        if score <= 0.05:
            return
        drivers.append({"label": label, "score": round(min(1.0, score), 3), "tone": tone, "why": why})
    add("Breadth regime", abs(overlay.get("market_breadth_pct", 0.5) - 0.5) * 2, "good" if overlay.get("market_breadth_pct", 0.5) > 0.5 else "bad", "Lebar sempitnya partisipasi market.")
    add("Inflation shock", overlay.get("inf_shock", 0.0), "warn" if overlay.get("inf_shock", 0.0) < 0.7 else "bad", "Commodity-led shock branch.")
    add("Most hated rally", most_hated.get("relief_squeeze_score", 0.0), "good" if most_hated.get("risk_on_switch") else "warn", "Relief squeeze / risk-on switch.")
    add("Analog fit", analog.get("similarity", 0.0), "good", f"Current tape mirip {analog.get('label')}.")
    return sorted(drivers, key=lambda x: x["score"], reverse=True)

def build_forward_radar(scan_df: pd.DataFrame, overlay: Dict, route: Dict, top_n: int = 20) -> List[Dict]:
    if scan_df.empty:
        return []
    radar = []
    for _, row in scan_df.iterrows():
        route_fit = 0.0
        if route["route_state"] in {"reflation_reaccel", "quality_disinflation"} and row["verdict"] in {"WATCH","READY_LONG"}:
            route_fit += 0.4
        if route["route_state"] == "stagflation_persist" and str(row.get("sector","")).lower() in {"energy","metals","mining"}:
            route_fit += 0.3
        score = 0.45 * float(row.get("long_rank_score", 0)) / 100 + 0.25 * route_fit + 0.15 * (1 - float(row.get("false_breakout_risk", 50)) / 100) + 0.15 * float(row.get("dry_score", 50)) / 100
        status = "ACTIVE" if row["verdict"] == "READY_LONG" else ("NEAR_TRIGGER" if score >= 0.58 else "NOT_YET")
        radar.append({
            "ticker": row["ticker"],
            "sector": row.get("sector","Unknown"),
            "status": status,
            "trigger": "Breakout integrity > 65 + route still valid",
            "why_not_yet": "Need stronger confirmation" if status != "ACTIVE" else "",
            "signal_quality": round(score, 3),
            "momentum_1m": round(float(row.get("ret_20d", 0.0)) * 100, 2) if pd.notna(row.get("ret_20d", np.nan)) else None,
        })
    radar = sorted(radar, key=lambda x: x["signal_quality"], reverse=True)
    return radar[:top_n]

def merge_overlay_into_scan(scan_df: pd.DataFrame, route: Dict, most_hated: Dict, analog: Dict, radar: List[Dict], events: List[Dict]) -> pd.DataFrame:
    df = scan_df.copy()
    radar_map = {r["ticker"]: r for r in radar}
    next_play_score = []
    why_now = []
    why_not_yet = []
    trigger = []
    invalidator = []
    dominant_risk = []
    route_fit = []

    for _, row in df.iterrows():
        rf = 0.0
        sec = str(row.get("sector","")).lower()
        if route["route_state"] == "reflation_reaccel" and sec in {"banks", "retail", "consumer", "telecom"}:
            rf += 0.35
        if route["route_state"] == "quality_disinflation" and sec in {"banks", "consumer", "telecom"}:
            rf += 0.30
        if route["route_state"] == "stagflation_persist" and sec in {"energy", "metals", "mining"}:
            rf += 0.40
        if route["route_state"] in {"deflationary_riskoff","panic_crash"}:
            rf -= 0.20

        mh_boost = 0.10 if most_hated["risk_on_switch"] and row["verdict"] in {"WATCH","READY_LONG"} else 0.0
        analog_boost = 0.05 if analog["scenario_family"] in {"relief_squeeze","mixed_slowdown"} else 0.0
        nps = 0.55 * float(row.get("long_rank_score", 0)) / 100 - 0.35 * float(row.get("risk_rank_score", 0)) / 100 + rf + mh_boost + analog_boost

        route_fit.append(round(rf, 3))
        next_play_score.append(round(nps, 3))
        if row["verdict"] == "READY_LONG":
            why_now.append("Price-side already qualified and route still supportive.")
            why_not_yet.append("")
        elif row["verdict"] == "WATCH":
            why_now.append("Leadership acceptable but still needs trigger.")
            why_not_yet.append("Need cleaner breakout / stronger route confirmation.")
        elif row["verdict"] == "AVOID":
            why_now.append("")
            why_not_yet.append("Risk rank too high for current route.")
        else:
            why_now.append("")
            why_not_yet.append("Not yet a clear route-fit setup.")

        trigger.append("Breakout integrity > 65 and route not invalidated")
        invalidator.append(route["invalidator_route"])
        dr = "Macro hostile" if route["route_bias"] == "risk-off" else ("False breakout risk" if float(row.get("false_breakout_risk", 50)) > 60 else "Need confirmation")
        dominant_risk.append(dr)

    df["route_primary"] = route["route_state"]
    df["route_alt"] = route["alt_route"]
    df["route_invalidator"] = route["invalidator_route"]
    df["route_fit"] = route_fit
    df["next_play_score"] = next_play_score
    df["mh_clear_count"] = most_hated["clear_count"]
    df["analog_label"] = analog["label"]
    df["analog_family"] = analog["scenario_family"]
    df["why_now"] = why_now
    df["why_not_yet"] = why_not_yet
    df["trigger"] = trigger
    df["invalidator"] = invalidator
    df["dominant_risk"] = dominant_risk
    return df.sort_values(["next_play_score", "long_rank_score"], ascending=[False, False]).reset_index(drop=True)
