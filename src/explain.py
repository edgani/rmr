from __future__ import annotations

import pandas as pd


def _safe(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _txt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    s = str(v).strip()
    return s if s else "-"



def build_why_now(row: pd.Series) -> str:
    verdict = str(row.get("verdict", "NEUTRAL"))
    bits: list[str] = []
    tq = _safe(row.get("trend_quality"))
    bi = _safe(row.get("breakout_integrity"))
    dry = _safe(row.get("dry_score_final", row.get("dry_score")))
    broker = _safe(row.get("broker_alignment_score"))
    burst = _txt(row.get("latest_event_label"))
    phase = str(row.get("phase", "NEUTRAL"))

    if tq >= 60:
        bits.append("trend cukup rapi")
    if bi >= 55:
        bits.append("breakout integrity sehat")
    if dry >= 55:
        bits.append("supply relatif kering")
    if broker >= 60:
        bits.append("broker align")
    if burst not in {"-", "NO_INTRADAY_SIGNAL", "NO_ORDERBOOK_UPLOAD"} and "UP_" in burst:
        bits.append(f"burst {burst.lower()}")
    if phase not in {"NEUTRAL", "MARKDOWN"}:
        bits.append(f"fase {phase.lower()}")

    if verdict in {"READY_LONG", "WATCH", "WATCH_REBOUND"}:
        return ", ".join(bits[:4]) if bits else "setup mulai mendukung"
    if verdict in {"TRIM", "AVOID"}:
        return "struktur lemah atau tekanan distribusi masih dominan"
    return ", ".join(bits[:3]) if bits else "belum ada alasan kuat untuk agresif"



def build_why_not_yet(row: pd.Series) -> str:
    verdict = str(row.get("verdict", "NEUTRAL"))
    reasons = []
    tq = _safe(row.get("trend_quality"))
    bi = _safe(row.get("breakout_integrity"))
    fb = _safe(row.get("false_breakout_risk"))
    wet = _safe(row.get("wet_score_final", row.get("wet_score")))
    broker = _safe(row.get("broker_alignment_score"))

    if verdict == "READY_LONG":
        return "tetap tunggu invalidation jelas dan jangan kejar terlalu telat"
    if tq < 50:
        reasons.append("trend belum cukup rapi")
    if bi < 55:
        reasons.append("breakout belum cukup meyakinkan")
    if fb >= 40:
        reasons.append("risiko false breakout masih ada")
    if wet >= 55:
        reasons.append("supply masih agak basah")
    if 0 < broker < 55:
        reasons.append("broker belum cukup align")
    return ", ".join(reasons[:4]) if reasons else "konfirmasi tambahan masih dibutuhkan"



def build_trigger(row: pd.Series) -> str:
    resistance = row.get("resistance_60d")
    support = row.get("support_20d")
    inst_res = row.get("institutional_resistance")
    inst_sup = row.get("institutional_support")
    verdict = str(row.get("verdict", "NEUTRAL"))
    if verdict in {"READY_LONG", "WATCH"}:
        if pd.notna(inst_res):
            return f"butuh close tegas di atas institutional resistance {float(inst_res):,.0f}"
        if pd.notna(resistance):
            return f"butuh close tegas di atas resistance {float(resistance):,.0f}"
    if verdict == "WATCH_REBOUND":
        if pd.notna(inst_sup):
            return f"butuh rebound bertahan di atas institutional support {float(inst_sup):,.0f}"
        if pd.notna(support):
            return f"butuh rebound bertahan di atas support {float(support):,.0f}"
    return "tunggu struktur lebih jelas dan volume mendukung"



def build_invalidator(row: pd.Series) -> str:
    support = row.get("support_20d")
    inst_sup = row.get("institutional_support")
    phase = str(row.get("phase", "NEUTRAL"))
    if pd.notna(inst_sup):
        return f"invalid kalau jebol institutional support {float(inst_sup):,.0f}"
    if pd.notna(support):
        return f"invalid kalau jebol support {float(support):,.0f}"
    if phase == "MARKDOWN":
        return "invalid kalau gagal reclaim area breakdown"
    return "invalid kalau struktur makin rusak"



def build_risk_note(row: pd.Series) -> str:
    notes = []
    wet = _safe(row.get("wet_score_final", row.get("wet_score")))
    fb = _safe(row.get("false_breakout_risk"))
    liq = _safe(row.get("liquidity_mn"))
    overhang = _safe(row.get("overhang_score"))
    burst = _txt(row.get("latest_event_label"))
    if wet >= 60:
        notes.append("masih basah")
    if fb >= 45:
        notes.append("rawan false breakout")
    if liq < 20:
        notes.append("likuiditas tipis")
    if overhang >= 55:
        notes.append("overhang distribusi")
    if burst.startswith("DOWN_"):
        notes.append("intraday bias masih bearish")
    return ", ".join(notes) if notes else "risiko relatif normal untuk setup saat ini"
