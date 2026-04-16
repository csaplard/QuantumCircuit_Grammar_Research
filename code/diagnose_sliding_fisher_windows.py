"""
Diagnose sliding Fisher stability across window sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sliding_fisher import rolling_fisher_trace_series


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan, "q10": np.nan, "q50": np.nan, "q90": np.nan}
    mean = float(np.mean(v))
    std = float(np.std(v))
    return {
        "n": int(v.size),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if abs(mean) > 1e-12 else np.nan,
        "q10": float(np.quantile(v, 0.10)),
        "q50": float(np.quantile(v, 0.50)),
        "q90": float(np.quantile(v, 0.90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sliding Fisher window diagnostics")
    ap.add_argument("--input-csv", required=True, help="Entropy CSV with column 'entropy'")
    ap.add_argument("--windows", default="32,64,128")
    ap.add_argument("--alphabet-size", type=int, default=7)
    ap.add_argument("--sax-behavior", choices=("quantile", "gaussian"), default="quantile")
    ap.add_argument("--fisher-epsilon", type=float, default=1e-10)
    ap.add_argument("--include-log-trace", action="store_true")
    ap.add_argument("--out-prefix", default="results/llm_entropy/fisher_runs/sliding_fisher_diag")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "entropy" not in df.columns:
        raise SystemExit("Input CSV must contain 'entropy' column")
    entropy = df["entropy"].values.astype(float)
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    out_df = pd.DataFrame({"step": np.arange(len(entropy)), "entropy": entropy})
    summary_rows = []
    for w in windows:
        tr = rolling_fisher_trace_series(
            entropy_series=entropy,
            window_size=w,
            alphabet_size=args.alphabet_size,
            sax_behavior=args.sax_behavior,
            fisher_epsilon=args.fisher_epsilon,
        )
        col = f"fisher_w{w}"
        out_df[col] = tr
        s = _summary_stats(tr)
        row = {"window": w, "fisher_epsilon": float(args.fisher_epsilon), **s}
        if args.include_log_trace:
            log_tr = np.where(np.isfinite(tr) & (tr > 0.0), np.log(tr), np.nan)
            out_df[f"log_{col}"] = log_tr
            row.update({f"log_{k}": v for k, v in _summary_stats(log_tr).items()})
        summary_rows.append(row)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_csv = Path(str(out_prefix) + "_traces.csv")
    summary_csv = Path(str(out_prefix) + "_summary.csv")
    plot_png = Path(str(out_prefix) + "_plot.png")

    out_df.to_csv(trace_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        for w in windows:
            c = f"fisher_w{w}"
            plt.plot(out_df["step"], out_df[c], label=f"W={w}", linewidth=1.3)
        plt.xlabel("Token step")
        plt.ylabel("Sliding Fisher trace")
        plt.title("Sliding Fisher diagnostics across windows")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_png, dpi=200)
        print(f"Wrote {plot_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Wrote {trace_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
