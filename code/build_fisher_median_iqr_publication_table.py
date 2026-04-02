"""
Build a sorted publication-ready table: median N* and IQR (q25–q75) across Fisher seeds.

Reads: results/fisher_estimated_thresholds_median_seeds012.csv (or --input).
Writes: results/fisher_threshold_median_iqr_publication.csv and .md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

TOPO_ORDER = {"1D_Snake": 0, "2D_Block": 1, "Bulk_Full": 2}


def q_num(label: str) -> int:
    return int(str(label).replace("q", "").strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=RESULTS / "fisher_estimated_thresholds_median_seeds012.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS / "fisher_threshold_median_iqr_publication.csv",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS / "fisher_threshold_median_iqr_publication.md",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["q_int"] = df["q_label"].map(q_num)
    df["topo_rank"] = df["topology"].map(TOPO_ORDER)
    df = df.sort_values(["topo_rank", "q_int"]).reset_index(drop=True)

    df["IQR_width"] = df["estimated_N_threshold_q75"] - df["estimated_N_threshold_q25"]
    df["stable_3seeds"] = df["IQR_width"] == 0

    out = df[
        [
            "topology",
            "q_label",
            "file",
            "estimated_N_threshold",
            "estimated_N_threshold_q25",
            "estimated_N_threshold_q75",
            "IQR_width",
            "stable_3seeds",
        ]
    ].rename(
        columns={
            "estimated_N_threshold": "median_N_star",
            "estimated_N_threshold_q25": "N_star_q25",
            "estimated_N_threshold_q75": "N_star_q75",
        }
    )

    out.to_csv(args.out_csv, index=False)

    # Markdown: compact numeric columns
    md_lines = [
        "# Fisher threshold N* (median across seeds 0, 1, 2)",
        "",
        "Per-readout estimate from `fisher_information_analysis` phase-transition heuristic. "
        "**median_N_star** = median of `estimated_N_threshold` over three grammar-learner seeds; "
        "**N_star_q25 / N_star_q75** = quartiles across the same three values (IQR = q75 − q25).",
        "",
        "| Topology | Q | median N* | q25 | q75 | IQR width | stable (3 seeds) |",
        "|----------|---|-----------|-----|-----|-----------|------------------|",
    ]
    for _, r in out.iterrows():
        stab = "yes" if r["stable_3seeds"] else "no"
        md_lines.append(
            f"| {r['topology']} | {r['q_label']} | {int(r['median_N_star'])} | "
            f"{int(r['N_star_q25'])} | {int(r['N_star_q75'])} | {int(r['IQR_width'])} | {stab} |"
        )
    md_lines.append("")
    md_lines.append(f"Source CSV: `{args.input.name}`")
    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
