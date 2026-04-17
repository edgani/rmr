from __future__ import annotations
import argparse
from pathlib import Path

PATCH = '''
# --- strict full universe patch start ---
from pathlib import Path as _Path
import pandas as _pd

def _strict_universe_check(path="data/idx_universe_full.csv", threshold=900):
    p = _Path(path)
    if not p.exists():
        return {"ok": False, "reason": "missing_full_universe", "count": 0, "threshold": threshold}
    try:
        df = _pd.read_csv(p)
        count = len(df)
        if count < threshold:
            return {"ok": False, "reason": "universe_too_small", "count": count, "threshold": threshold}
        return {"ok": True, "reason": "ok", "count": count, "threshold": threshold}
    except Exception:
        return {"ok": False, "reason": "bad_universe_file", "count": 0, "threshold": threshold}

_STRICT_UNIVERSE = _strict_universe_check(threshold={threshold})
# --- strict full universe patch end ---
'''

NOTICE = '''
# --- strict full universe notice start ---
try:
    if not _STRICT_UNIVERSE["ok"]:
        st.error(
            f"FULL universe unavailable: {_STRICT_UNIVERSE['reason']} | "
            f"master_count={_STRICT_UNIVERSE['count']} threshold={_STRICT_UNIVERSE['threshold']}"
        )
except Exception:
    pass
# --- strict full universe notice end ---
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--threshold", type=int, default=900)
    args = ap.parse_args()
    app = Path(args.app)
    txt = app.read_text(encoding="utf-8")
    if "strict full universe patch start" not in txt:
        txt = PATCH.format(threshold=args.threshold) + "\n" + txt
    if "strict full universe notice start" not in txt:
        marker = "st.set_page_config"
        idx = txt.find(marker)
        if idx != -1:
            # insert notice after first blank line following page config call if possible
            end = txt.find("\n", idx)
            if end != -1:
                txt = txt[:end+1] + NOTICE + txt[end+1:]
            else:
                txt += "\n" + NOTICE
        else:
            txt += "\n" + NOTICE
    app.write_text(txt, encoding="utf-8")
    print(f"Patched {app}")


if __name__ == "__main__":
    main()
