"""
IBM Fisher threshold sweep on existing raw shot files.

Runs an N-prefix sweep (e.g. 500..40000) for each matching IBM shot file,
extracts grammar transition matrices, computes Fisher/KL metrics, and estimates
threshold per file and per backend via max |d(Fisher trace)/dN|.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from grammar_learner import extract_grammar, train_model

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SHOTS_DIR = os.path.join(REPO_ROOT, "results", "ibm_raw_shots")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def parse_ibm_filename(name: str) -> Tuple[str, str, int, int] | None:
    m = re.match(
        r"^ibm_([^_]+)_(hadamard|ghz|layers|identity)_(\d+)q_(\d+)shots\.txt$",
        name,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1), m.group(2).lower(), int(m.group(3)), int(m.group(4))


def list_ibm_shot_files(
    shots_dir: str, backend_prefix: str | None, filename_substr: str | None
) -> List[str]:
    files = [f for f in os.listdir(shots_dir) if f.endswith(".txt") and "shots" in f]
    if backend_prefix:
        files = [f for f in files if f.startswith(f"{backend_prefix}_")]
    if filename_substr:
        files = [f for f in files if filename_substr in f]

    def sort_key(fn: str) -> Tuple[str, int, str]:
        p = parse_ibm_filename(fn)
        if p is None:
            return ("", 0, fn)
        backend, circuit, nq, _ = p
        return (backend, circuit, nq)

    files.sort(key=sort_key)
    return files


def compute_fisher_matrix(transition_matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    n = transition_matrix.shape[0]
    t = np.maximum(transition_matrix.copy(), epsilon)
    t = t / t.sum(axis=1, keepdims=True)

    eigvals, eigvecs = np.linalg.eig(t.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.real(eigvecs[:, idx])
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()

    fisher = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            fisher[j, j] += stationary[i] / t[i, j]
    return fisher


def fisher_scalar(fisher_matrix: np.ndarray) -> float:
    return float(np.trace(fisher_matrix))


def fisher_determinant(fisher_matrix: np.ndarray) -> float:
    return float(np.linalg.det(fisher_matrix))


def fisher_max_eigenvalue(fisher_matrix: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(fisher_matrix)
    return float(np.max(vals))


def compute_kl_divergence(t1: np.ndarray, t2: np.ndarray, epsilon: float = 1e-10) -> float:
    t1 = np.maximum(t1, epsilon)
    t2 = np.maximum(t2, epsilon)
    t1 = t1 / t1.sum(axis=1, keepdims=True)
    t2 = t2 / t2.sum(axis=1, keepdims=True)
    kl_per_row = np.sum(t1 * np.log(t1 / t2), axis=1)
    return float(np.mean(kl_per_row))


def estimate_threshold(lengths: np.ndarray, traces: np.ndarray) -> Tuple[float, float, int, int]:
    d_trace = np.diff(traces) / np.diff(lengths)
    idx = int(np.argmax(np.abs(d_trace)))
    left_n = int(lengths[idx])
    right_n = int(lengths[idx + 1])
    est = float((left_n + right_n) / 2)
    return est, float(d_trace[idx]), left_n, right_n


def summarize_thresholds(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_file_summary: List[Dict[str, object]] = []
    for fn, g in detail_df.groupby("file"):
        g = g.sort_values("n_points")
        lengths = g["n_points"].to_numpy(dtype=np.float64)
        traces = g["fisher_trace"].to_numpy(dtype=np.float64)
        if len(lengths) < 2:
            continue
        est, rate, left_n, right_n = estimate_threshold(lengths, traces)
        per_file_summary.append(
            {
                "file": fn,
                "backend": g["backend"].iloc[0],
                "circuit": g["circuit"].iloc[0],
                "n_qubits": int(g["n_qubits"].iloc[0]),
                "threshold_estimate": est,
                "max_rate_of_change": rate,
                "between_n_left": left_n,
                "between_n_right": right_n,
            }
        )
    summary_df = pd.DataFrame(per_file_summary)
    if summary_df.empty:
        backend_agg = pd.DataFrame(columns=["backend", "mean", "std", "min", "max", "count"])
    else:
        backend_agg = (
            summary_df.groupby("backend")["threshold_estimate"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
    return summary_df, backend_agg


def main() -> None:
    parser = argparse.ArgumentParser(description="IBM Fisher threshold sweep")
    parser.add_argument("--shots-dir", default=DEFAULT_SHOTS_DIR)
    parser.add_argument("--backend-prefix", default=None, help="e.g. ibm_marrakesh")
    parser.add_argument(
        "--filename-substr",
        default="40960",
        help="Only include files with this substring (e.g. 40960, 163840).",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000, 8000, 12000, 20000, 30000, 40000],
        help="N-prefix lengths to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alphabet-size", type=int, default=7)
    parser.add_argument("--output-tag", default="ibm40960")
    parser.add_argument(
        "--detail-csv",
        default=None,
        help="If set, skip training and only build threshold summary from this detail CSV.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    detail_path = os.path.join(RESULTS_DIR, f"ibm_fisher_sweep_{args.output_tag}.csv")
    summary_path = os.path.join(RESULTS_DIR, f"ibm_fisher_thresholds_{args.output_tag}.csv")
    report_path = os.path.join(RESULTS_DIR, f"ibm_fisher_threshold_report_{args.output_tag}.txt")

    if args.detail_csv:
        detail_df = pd.read_csv(args.detail_csv)
        summary_df, backend_agg = summarize_thresholds(detail_df)
        summary_df.to_csv(summary_path, index=False)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== IBM FISHER THRESHOLD SWEEP REPORT ===\n\n")
            f.write(f"detail_csv: {args.detail_csv}\n")
            f.write(f"files analyzed: {len(summary_df)}\n\n")
            if not summary_df.empty:
                f.write("--- Per-file thresholds ---\n")
                for _, row in summary_df.sort_values(["backend", "circuit", "n_qubits"]).iterrows():
                    f.write(
                        f"{row['file']}: threshold~{row['threshold_estimate']:.0f} "
                        f"(between {int(row['between_n_left'])}-{int(row['between_n_right'])})\n"
                    )
                f.write("\n--- Backend aggregates ---\n")
                for _, row in backend_agg.iterrows():
                    f.write(
                        f"{row['backend']}: mean={row['mean']:.1f}, std={row['std']:.1f}, "
                        f"min={row['min']:.0f}, max={row['max']:.0f}, n={int(row['count'])}\n"
                    )
            else:
                f.write("No threshold estimates produced.\n")
        print(f"Saved: {summary_path}", flush=True)
        print(f"Saved: {report_path}", flush=True)
        return

    files = list_ibm_shot_files(args.shots_dir, args.backend_prefix, args.filename_substr)
    if not files:
        print(
            f"No matching IBM shot files found in {args.shots_dir} "
            f"(prefix={args.backend_prefix}, substr={args.filename_substr}).",
            flush=True,
        )
        sys.exit(1)

    params = {"hidden_dim": 32, "seq_len": 16, "epochs": args.epochs, "lr": 0.01}
    lengths_req = sorted(list(set(int(x) for x in args.lengths)))

    all_rows: List[Dict[str, object]] = []
    per_file_summary: List[Dict[str, object]] = []

    for fn in files:
        parsed = parse_ibm_filename(fn)
        if parsed is None:
            continue
        backend, circuit, n_qubits, total_shots = parsed
        path = os.path.join(args.shots_dir, fn)
        df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
        signal_full = df["output"].values.astype(np.float64)
        max_n = len(signal_full)
        lengths = [n for n in lengths_req if n <= max_n]
        if len(lengths) < 2:
            continue

        print(f"\n=== {fn} | max_n={max_n} ===", flush=True)
        prev_t: np.ndarray | None = None
        traces: List[float] = []
        used_lengths: List[int] = []

        for n_pts in lengths:
            signal = signal_full[:n_pts]
            print(f"  N={n_pts}...", end=" ", flush=True)
            _, _, model, val_data = train_model(
                signal,
                f"{fn}_{n_pts}",
                alphabet_size=args.alphabet_size,
                data_is_array=True,
                seed=args.seed,
                **params,
            )
            if model is None:
                print("FAILED", flush=True)
                continue

            t = extract_grammar(model, val_data, seq_len=params["seq_len"])
            fisher = compute_fisher_matrix(t)
            f_trace = fisher_scalar(fisher)
            f_det = fisher_determinant(fisher)
            f_maxeig = fisher_max_eigenvalue(fisher)
            kl_prev = 0.0 if prev_t is None else compute_kl_divergence(prev_t, t)

            row = {
                "file": fn,
                "backend": backend,
                "circuit": circuit,
                "n_qubits": int(n_qubits),
                "total_shots": int(total_shots),
                "n_points": int(n_pts),
                "fisher_trace": f_trace,
                "fisher_det": f_det,
                "fisher_max_eigenvalue": f_maxeig,
                "kl_from_previous": kl_prev,
            }
            all_rows.append(row)
            traces.append(f_trace)
            used_lengths.append(n_pts)
            prev_t = t.copy()
            print(f"trace={f_trace:.2f}, KL={kl_prev:.4f}", flush=True)

        if len(used_lengths) >= 2:
            lengths_np = np.array(used_lengths, dtype=np.float64)
            traces_np = np.array(traces, dtype=np.float64)
            est, rate, left_n, right_n = estimate_threshold(lengths_np, traces_np)
            per_file_summary.append(
                {
                    "file": fn,
                    "backend": backend,
                    "circuit": circuit,
                    "n_qubits": int(n_qubits),
                    "threshold_estimate": est,
                    "max_rate_of_change": rate,
                    "between_n_left": left_n,
                    "between_n_right": right_n,
                }
            )
            print(
                f"  => threshold ~ {est:.0f} (between {left_n} and {right_n})",
                flush=True,
            )

    if not all_rows:
        print("No sweep rows were produced.", flush=True)
        sys.exit(1)

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(detail_path, index=False)
    summary_df, backend_agg = summarize_thresholds(detail_df)
    summary_df.to_csv(summary_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== IBM FISHER THRESHOLD SWEEP REPORT ===\n\n")
        f.write(f"shots_dir: {args.shots_dir}\n")
        f.write(f"backend_prefix: {args.backend_prefix}\n")
        f.write(f"filename_substr: {args.filename_substr}\n")
        f.write(f"seed: {args.seed}, epochs: {args.epochs}\n")
        f.write(f"lengths: {lengths_req}\n")
        f.write(f"files analyzed: {len(summary_df)}\n\n")

        if not summary_df.empty:
            f.write("--- Per-file thresholds ---\n")
            for _, row in summary_df.sort_values(["backend", "circuit", "n_qubits"]).iterrows():
                f.write(
                    f"{row['file']}: threshold~{row['threshold_estimate']:.0f} "
                    f"(between {int(row['between_n_left'])}-{int(row['between_n_right'])})\n"
                )
            f.write("\n--- Backend aggregates ---\n")
            for _, row in backend_agg.iterrows():
                f.write(
                    f"{row['backend']}: mean={row['mean']:.1f}, std={row['std']:.1f}, "
                    f"min={row['min']:.0f}, max={row['max']:.0f}, n={int(row['count'])}\n"
                )
        else:
            f.write("No threshold estimates produced.\n")

    print(f"\nSaved: {detail_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
