
from pathlib import Path
import pandas as pd

from src.universe_completeness import load_master_universe, build_universe_audit

root = Path(__file__).resolve().parents[1]
master_path = root / "data" / "idx_universe_full_template.csv"

master = load_master_universe(master_path)
loaded = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "GOTO"]
failed = ["ANTM", "MDKA"]

audit = build_universe_audit(master, loaded_tickers=loaded, failed_tickers=failed)

out_dir = root / "data"
audit["missing_df"].to_csv(out_dir / "missing_tickers_demo.csv", index=False)
audit["failed_df"].to_csv(out_dir / "failed_tickers_demo.csv", index=False)
audit["coverage_df"].to_csv(out_dir / "coverage_demo.csv", index=False)

print(audit["summary"])
