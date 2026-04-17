#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import string
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

import requests

ALPHABET = string.ascii_uppercase
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
UA = {"User-Agent": "Mozilla/5.0 IDXUniverseBuilder/1.0"}


def chunks(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def generate_symbols(length: int = 4, suffix: str = ".JK") -> Iterable[str]:
    for tup in itertools.product(ALPHABET, repeat=length):
        yield ''.join(tup) + suffix


def load_resume_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sym = (row.get("symbol_yf") or row.get("symbol") or "").strip().upper()
                if sym:
                    seen.add(sym)
    except Exception:
        pass
    return seen


def validate_batch(symbols: List[str], timeout: float = 20.0) -> List[Dict[str, Any]]:
    params = {"symbols": ",".join(symbols)}
    r = requests.get(QUOTE_URL, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("quoteResponse", {}).get("result", []):
        symbol = str(item.get("symbol", "")).upper()
        if not symbol.endswith(".JK"):
            continue
        if item.get("quoteType") not in {"EQUITY", "MUTUALFUND", "ETF", None}:
            continue
        ex = str(item.get("fullExchangeName", "")) + "|" + str(item.get("exchange", ""))
        if "Indonesia" not in ex and "JKT" not in ex and item.get("exchange") not in {"JKT", "JKT-IND"}:
            # keep some unknowns if suffix is .JK and name exists
            if not (item.get("shortName") or item.get("longName")):
                continue
        out.append({
            "ticker": symbol.replace(".JK", ""),
            "symbol_yf": symbol,
            "short_name": item.get("shortName", ""),
            "long_name": item.get("longName", ""),
            "exchange": item.get("exchange", ""),
            "full_exchange_name": item.get("fullExchangeName", ""),
            "quote_type": item.get("quoteType", ""),
            "market_cap": item.get("marketCap", ""),
            "currency": item.get("currency", ""),
        })
    return out


def append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def dedup_sort(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for row in rows:
        sym = (row.get("symbol_yf") or "").upper()
        if sym and sym not in by:
            by[sym] = row
    ordered = sorted(by.values(), key=lambda r: r["symbol_yf"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered[0].keys()) if ordered else ["ticker","symbol_yf"])
        w.writeheader()
        for row in ordered:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Brute-force discover Yahoo .JK symbols in batches.")
    ap.add_argument("--output", default="data/idx_universe_bruteforce.csv")
    ap.add_argument("--length", type=int, default=4, help="Symbol length to brute-force. Default 4.")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all batches")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--audit-json", default="data/idx_universe_bruteforce.audit.json")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    seen = load_resume_seen(out_path) if args.resume else set()

    attempted = 0
    kept = 0
    batches = 0
    errors = 0

    for batch in chunks(list(generate_symbols(args.length)), args.batch_size):
        batches += 1
        if args.max_batches and batches > args.max_batches:
            break
        batch = [s for s in batch if s not in seen]
        if not batch:
            continue
        attempted += len(batch)
        try:
            rows = validate_batch(batch)
            fresh = [r for r in rows if r["symbol_yf"] not in seen]
            if fresh:
                append_rows(out_path, fresh)
                for r in fresh:
                    seen.add(r["symbol_yf"])
                kept += len(fresh)
        except Exception:
            errors += 1
        time.sleep(args.sleep)

    dedup_sort(out_path)
    final_rows = len(load_resume_seen(out_path))
    audit = {
        "mode": "bruteforce_yahoo_quote",
        "length": args.length,
        "attempted_symbols": attempted,
        "kept_in_run": kept,
        "final_rows": final_rows,
        "batches_seen": batches,
        "errors": errors,
        "output": str(out_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
