from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def _num(v, default: float = 0.0) -> float:
    try:
        x = float(pd.to_numeric(v, errors='coerce'))
        return x if pd.notna(x) else default
    except Exception:
        return default


def _txt(v, default: str = '') -> str:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    s = str(v).strip()
    return s if s else default


LONG_SECTOR_MAP: Dict[str, set[str]] = {
    'risk_on_rotation': {'technology', 'tech', 'banks', 'financials', 'consumer discretionary', 'consumer cyclical', 'property', 'industrials', 'materials'},
    'quality_growth_breakout': {'technology', 'tech', 'banks', 'financials', 'consumer discretionary', 'industrials'},
    'relief_squeeze_rotation': {'technology', 'tech', 'property', 'consumer discretionary', 'industrials', 'materials'},
    'selective_stock_picking': {'banks', 'telecom', 'consumer staples', 'healthcare', 'industrials'},
    'wait_for_confirmation': {'banks', 'telecom', 'consumer staples', 'healthcare'},
    'defensive_capital_preservation': {'consumer staples', 'healthcare', 'telecom', 'utilities', 'energy', 'gold', 'coal'},
    'commodity_or_gold_barbell': {'energy', 'gold', 'coal', 'materials', 'utilities', 'telecom'},
}

AVOID_SECTOR_MAP: Dict[str, set[str]] = {
    'defensive_capital_preservation': {'technology', 'tech', 'consumer discretionary', 'property', 'industrials'},
    'commodity_or_gold_barbell': {'technology', 'tech', 'consumer discretionary', 'property'},
    'risk_on_rotation': {'utilities'},
}


def _sector_align(route_primary: str, sector: str) -> float:
    route = _txt(route_primary, 'selective_stock_picking').lower()
    sector_l = _txt(sector, 'unknown').lower()
    favored = LONG_SECTOR_MAP.get(route, set())
    avoid = AVOID_SECTOR_MAP.get(route, set())
    if sector_l in favored:
        return 1.0
    if sector_l in avoid:
        return -1.0
    return 0.0


def annotate_macro_filter(scan_df: pd.DataFrame, strictness: str = 'balanced') -> pd.DataFrame:
    if scan_df is None or scan_df.empty:
        return pd.DataFrame() if scan_df is None else scan_df.copy()
    out = scan_df.copy()
    strict = _txt(strictness, 'balanced').lower()

    states = []
    gate_scores = []
    notes = []
    play_nows = []
    next_play = []
    route_summary = []
    adjusted_entry = []

    for _, row in out.iterrows():
        verdict = _txt(row.get('verdict'), 'NEUTRAL').upper()
        route_primary = _txt(row.get('route_primary'), 'selective_stock_picking')
        route_bias = _txt(row.get('route_bias'), 'MIXED_BIAS').upper()
        regime = _txt(row.get('market_regime'), 'UNKNOWN').upper()
        exec_mode = _txt(row.get('execution_mode'), 'SELECTIVE').upper()
        sector = _txt(row.get('sector'), 'Unknown')
        route_fit = _num(row.get('route_fit_score'), _num(row.get('route_rank_score')) / 100.0)
        catalyst = _num(row.get('catalyst_window_score'), 0.0)
        market_bias = _num(row.get('market_bias_score'), 50.0) / 100.0
        dry = _num(row.get('dry_score_final'), _num(row.get('dry_score'), 50.0)) / 100.0
        wet = _num(row.get('wet_score_final'), _num(row.get('wet_score'), 50.0)) / 100.0
        risk = _num(row.get('risk_rank_score'), _num(row.get('false_breakout_risk'), 50.0)) / 100.0
        long_rank = _num(row.get('long_rank_score'), 50.0) / 100.0
        conf = _num(row.get('score_confidence'), 50.0) / 100.0
        radar = _txt(row.get('forward_radar_bucket'), 'NOT_YET').upper()
        sector_align = _sector_align(route_primary, sector)
        sector_bonus = 0.12 * sector_align

        gate = 0.38 * route_fit + 0.18 * catalyst + 0.14 * market_bias + 0.10 * dry - 0.10 * risk + 0.10 * long_rank + 0.10 * conf + sector_bonus
        if exec_mode == 'DEFENSIVE':
            gate -= 0.05
        elif exec_mode == 'AGGRESSIVE':
            gate += 0.04
        if route_bias in {'DEFENSIVE_BIAS', 'SHORT_BIAS'}:
            gate -= 0.08 if verdict in {'READY_LONG', 'WATCH'} else 0.0
            gate += 0.06 if verdict in {'TRIM', 'AVOID', 'WATCH_REBOUND'} else 0.0
        elif route_bias in {'LONG_BIAS', 'TACTICAL_RISK_ON'}:
            gate += 0.08 if verdict in {'READY_LONG', 'WATCH'} else 0.0
            gate -= 0.05 if verdict in {'TRIM', 'AVOID'} else 0.0

        gate = max(0.0, min(1.0, gate))
        state = 'CAUTION'
        note = 'Macro campuran, jangan agresif.'
        play_now_flag = False
        next_play_flag = False

        if route_bias in {'LONG_BIAS', 'TACTICAL_RISK_ON', 'MIXED_BIAS'}:
            if gate >= 0.66 and sector_align >= 0 and verdict in {'READY_LONG', 'WATCH'}:
                state = 'ALIGNED'
                note = 'Align sama route/macro sekarang.'
                play_now_flag = verdict == 'READY_LONG'
            elif gate >= 0.52 and radar in {'NEAR_TRIGGER', 'ACTIVE'}:
                state = 'NEXT_ROUTE'
                note = 'Belum play utama, tapi dekat trigger next play.'
                next_play_flag = True
            elif gate < (0.40 if strict == 'loose' else 0.47 if strict == 'balanced' else 0.55):
                state = 'OFFSIDE'
                note = 'Bagus sendiri, tapi lawan route/macro utama.'
        else:
            # defensive / short-biased environment
            if verdict in {'TRIM', 'AVOID'} and gate >= 0.45:
                state = 'ALIGNED'
                note = 'Sesuai route defensif / risk-off.'
                play_now_flag = True
            elif verdict == 'WATCH_REBOUND' and gate >= 0.50:
                state = 'NEXT_ROUTE'
                note = 'Rebound mungkin ada, tapi masih tactical.'
                next_play_flag = True
            elif verdict in {'READY_LONG', 'WATCH'}:
                state = 'OFFSIDE'
                note = 'Nama long ini belum align dengan regime defensif.'
            else:
                state = 'CAUTION'
                note = 'Mode defensif: pilih kualitas / sabar.'

        route_summary.append(f"{route_primary} | {route_bias} | {exec_mode}")
        states.append(state)
        gate_scores.append(round(gate * 100.0, 1))
        notes.append(note)
        play_nows.append(play_now_flag)
        next_play.append(next_play_flag)
        adjusted_entry.append(round((_num(row.get('entry_score'), 0.0) * 0.65) + (gate * 35.0), 2))

    out['macro_filter_state'] = states
    out['macro_gate_score'] = gate_scores
    out['macro_note'] = notes
    out['preferred_play_now'] = play_nows
    out['next_play_candidate'] = next_play
    out['route_summary'] = route_summary
    out['entry_score_macro'] = adjusted_entry
    return out


def apply_macro_bucket_override(row: pd.Series, current_bucket: str, strictness: str = 'balanced') -> tuple[str, str]:
    state = _txt(row.get('macro_filter_state'), 'CAUTION').upper()
    note = _txt(row.get('macro_note'), '')
    strict = _txt(strictness, 'balanced').lower()
    bucket = current_bucket
    if state == 'OFFSIDE':
        if strict == 'strict':
            bucket = 'JANGAN SENTUH DULU' if current_bucket not in {'AVOID / BUANG', 'WATCH REBOUND'} else current_bucket
        elif strict == 'balanced':
            if current_bucket in {'SIAP NAIK SEKARANG', 'HAMPIR SIAP — TUNGGU TRIGGER'}:
                bucket = 'JANGAN SENTUH DULU'
        else:  # loose
            if current_bucket == 'SIAP NAIK SEKARANG':
                bucket = 'HAMPIR SIAP — TUNGGU TRIGGER'
    elif state == 'NEXT_ROUTE':
        if current_bucket == 'SIAP NAIK SEKARANG':
            bucket = 'HAMPIR SIAP — TUNGGU TRIGGER'
    return bucket, note
