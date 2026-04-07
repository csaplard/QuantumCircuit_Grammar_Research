"""
Merge Fisher summary CSVs: fm-phi multiseed (factual, mathematical, philosophical)
+ creative multiseed block. Adds llm_seed from source_csv (…_seed42_…).

Example:
  python code/merge_llm_fisher_multiseed_summaries.py
  python code/merge_llm_fisher_multiseed_summaries.py --out results/llm_entropy/fisher_runs/custom.csv
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FISHER = os.path.join(REPO_ROOT, "results", "llm_entropy", "fisher_runs")

_REGIME_ORDER = ["factual", "creative", "mathematical", "philosophical"]
_CONTROL_ORDER = {"none": 0, "shuffled": 1, "random_uniform": 2}
_SEED_ORDER = {42: 0, 99: 1, 7: 2}


def _llm_seed_from_path(source_csv: str) -> int:
    m = re.search(r"_seed(\d+)_", source_csv.replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot parse llm seed from: {source_csv}")
    return int(m.group(1))


def _add_llm_seed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["llm_seed"] = out["source_csv"].map(_llm_seed_from_path)
    return out


def merge_fmphi_and_creative(
    fmphi_csv: str,
    creative_csv: str,
) -> pd.DataFrame:
    base = pd.read_csv(fmphi_csv)
    if "llm_seed" not in base.columns:
        base = _add_llm_seed(base)
    creative = pd.read_csv(creative_csv)
    creative = _add_llm_seed(creative)
    out = pd.concat([base, creative], ignore_index=True)
    out["_r"] = out["regime"].map({r: i for i, r in enumerate(_REGIME_ORDER)})
    out["_s"] = out["llm_seed"].map(_SEED_ORDER)
    out["_c"] = out["control"].map(_CONTROL_ORDER)
    out = out.sort_values(["_r", "_s", "_c"]).drop(columns=["_r", "_s", "_c"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge fm-phi + creative Fisher multiseed summaries")
    ap.add_argument(
        "--fmphi",
        default=os.path.join(FISHER, "llm_entropy_fisher_summary_fmphi_multiseed_10k.csv"),
        help="27-row fmphi summary (with llm_seed)",
    )
    ap.add_argument(
        "--creative",
        default=os.path.join(FISHER, "llm_entropy_fisher_summary_creative_multiseed_10k_grammar2.csv"),
        help="9-row creative summary (grammar seed fixed)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(FISHER, "llm_entropy_fisher_summary_all_regimes_multiseed_10k.csv"),
        help="Output path (36 rows)",
    )
    args = ap.parse_args()

    merged = merge_fmphi_and_creative(args.fmphi, args.creative)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
