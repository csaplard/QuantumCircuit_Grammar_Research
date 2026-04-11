"""
Fisher Information Metric for Grammar Fingerprinting Transition Matrices
========================================================================
Computes the Fisher information metric from SAX transition matrices at
different data lengths to detect geometric phase transitions at the
information threshold (~8000 points for Sycamore data).

Theory:
  The transition matrix T[i,j] = P(symbol_j | symbol_i) lives on a
  statistical manifold. The Fisher information metric measures the
  "distance" between nearby probability distributions. If there's a
  phase transition at the threshold, the Fisher metric should show
  a discontinuity or sharp peak there.

What we compute:
  1. Transition matrices at different data lengths (500, 1000, 2000, ..., 40000)
  2. Fisher information matrix for each transition matrix
  3. Scalar curvature (trace of Fisher matrix) as function of data length
  4. Rate of change of Fisher metric between consecutive data lengths
  5. Detect if there's a sharp transition near the known threshold

Usage:
  python code/fisher_information_analysis.py              # all *_readout_raw_data.txt (default ~28)
  python code/fisher_information_analysis.py --quick      # one representative file per topology (3)

Input directory: same as ``build_sycamore_readout_fingerprints.py`` — prefers repo-root
``readout_raw_data/``, else ``results/readout_raw_data/``.

Output (full run):
  results/fisher_metric_vs_datalength_all_readouts.csv
  results/fisher_information_analysis_all_readouts.txt
  results/fisher_phase_transition_all_readouts.png
  results/transition_matrices_sycamore_all_readouts.npz  (7×7 row-stochastic T(N) per grid row; see index CSV)
  results/transition_matrices_index_all_readouts.csv

Output (--quick):
  results/fisher_metric_vs_datalength.csv
  results/fisher_information_analysis.txt
  results/fisher_phase_transition.png

Multi-seed (unique filenames, no overwrite):
  python code/fisher_information_analysis.py --tag-with-seed --seed 0
  python code/fisher_information_analysis.py --tag-with-seed --seed 1
  python code/aggregate_fisher_threshold_seeds.py results/fisher_estimated_thresholds_per_readout_all_readouts_seed0.csv ...
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from grammar_learner import train_model, extract_grammar
from build_sycamore_readout_fingerprints import default_readout_dir, label_from_filename

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def compute_fisher_matrix(transition_matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute the Fisher Information Matrix from a row-stochastic transition matrix.
    
    For a transition matrix T where T[i,j] = P(j|i), the Fisher information
    for row i is: F_i[j,k] = sum over observations of (d log P / d theta_j)(d log P / d theta_k)
    
    For categorical distributions (each row of the transition matrix),
    the Fisher information matrix for row i is:
        F_i[j,k] = delta(j,k) / T[i,j]  (diagonal)
    
    We compute the average Fisher information across all rows (states),
    weighted by the stationary distribution.
    """
    n = transition_matrix.shape[0]
    
    # Ensure no zeros (add small epsilon)
    T = transition_matrix.copy()
    T = np.maximum(T, epsilon)
    # Re-normalize rows
    T = T / T.sum(axis=1, keepdims=True)
    
    # Compute stationary distribution (left eigenvector of T)
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    # Find eigenvector corresponding to eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()
    
    # Fisher information matrix (averaged over stationary distribution)
    # For each row i of transition matrix, Fisher is diag(1/T[i,:])
    # Average: F[j,k] = sum_i pi_i * delta(j,k) / T[i,j]
    fisher = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            fisher[j, j] += stationary[i] / T[i, j]
    
    return fisher


def fisher_scalar(fisher_matrix: np.ndarray) -> float:
    """Scalar summary of Fisher information: trace of the matrix."""
    return float(np.trace(fisher_matrix))


def fisher_determinant(fisher_matrix: np.ndarray) -> float:
    """Determinant of Fisher matrix (volume of information)."""
    return float(np.linalg.det(fisher_matrix))


def fisher_max_eigenvalue(fisher_matrix: np.ndarray) -> float:
    """Maximum eigenvalue of Fisher matrix (dominant information direction)."""
    eigenvalues = np.linalg.eigvalsh(fisher_matrix)
    return float(np.max(eigenvalues))


def compute_kl_divergence(T1: np.ndarray, T2: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL divergence between two transition matrices (averaged over rows).
    """
    T1 = np.maximum(T1, epsilon)
    T2 = np.maximum(T2, epsilon)
    T1 = T1 / T1.sum(axis=1, keepdims=True)
    T2 = T2 / T2.sum(axis=1, keepdims=True)
    
    kl_per_row = np.sum(T1 * np.log(T1 / T2), axis=1)
    return float(np.mean(kl_per_row))


def analyze_threshold(data_lengths, fisher_traces, fisher_dets, fisher_maxeigs, kl_divs):
    """
    Detect phase transition by finding the point of maximum change.
    """
    # Rate of change of Fisher trace
    d_trace = np.diff(fisher_traces) / np.diff(data_lengths)
    
    # Rate of change of KL divergence between consecutive matrices
    # (already computed as kl_divs)
    
    # Find maximum rate of change
    max_rate_idx = np.argmax(np.abs(d_trace))
    threshold_estimate = (data_lengths[max_rate_idx] + data_lengths[max_rate_idx + 1]) / 2
    
    return {
        "threshold_estimate": threshold_estimate,
        "max_rate_of_change": float(d_trace[max_rate_idx]),
        "d_trace": d_trace,
    }


def _representative_three_files(data_dir: str, ground_truth: dict) -> list[tuple[str, str, str]]:
    """Return list of (filename, topology, q_label) — one file per topology, first in sorted order."""
    files = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(data_dir, "*_readout_raw_data.txt"))
    )
    picked = {"1D_Snake": None, "2D_Block": None, "Bulk_Full": None}
    for f in files:
        lab = label_from_filename(f)
        if lab not in ground_truth:
            continue
        topo = ground_truth[lab]
        if topo in picked and picked[topo] is None:
            picked[topo] = (f, topo, lab)
    out = [picked[k] for k in ("1D_Snake", "2D_Block", "Bulk_Full") if picked[k] is not None]
    return out


def _all_readout_jobs(data_dir: str, ground_truth: dict, max_files: int | None) -> list[tuple[str, str, str]]:
    """All *_readout_raw_data.txt with known q-label in GROUND_TRUTH."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*_readout_raw_data.txt")))
    jobs: list[tuple[str, str, str]] = []
    for p in paths:
        fname = os.path.basename(p)
        lab = label_from_filename(fname)
        if lab not in ground_truth:
            print(f"  skip (unknown label): {fname}")
            continue
        topo = ground_truth[lab]
        jobs.append((fname, topo, lab))
    if max_files is not None:
        jobs = jobs[: max(0, max_files)]
    return jobs


def _resolve_output_suffix(base_suffix: str, seed: int, output_tag: str | None, tag_with_seed: bool) -> str:
    """base_suffix is '' or '_all_readouts'. Optional tag avoids clobbering across seeds."""
    if output_tag:
        return f"{base_suffix}_{output_tag}" if base_suffix else f"_{output_tag}"
    if tag_with_seed:
        return f"{base_suffix}_seed{seed}" if base_suffix else f"_seed{seed}"
    return base_suffix


def save_transition_matrices_npz(
    all_results: list[dict],
    all_matrices: dict[tuple[str, int], np.ndarray],
    npz_path: str,
    index_csv_path: str,
) -> int:
    """
    Persist stacked transition matrices T(N) in row-major sync with all_results.
    NPZ contains array ``T`` of shape (K, 7, 7); index CSV lists file, topology, n_points, row_index.
    """
    Ts: list[np.ndarray] = []
    idx_rows: list[dict] = []
    for row_index, r in enumerate(all_results):
        key = (r["file"], int(r["n_points"]))
        T = all_matrices.get(key)
        if T is None:
            continue
        Ts.append(np.asarray(T, dtype=np.float64))
        idx_rows.append(
            {
                "row_index": len(Ts) - 1,
                "file": r["file"],
                "q_label": r.get("q_label", ""),
                "topology": r.get("topology", ""),
                "n_points": int(r["n_points"]),
            }
        )
    if not Ts:
        return 0
    stack = np.stack(Ts, axis=0)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError(f"Expected transition matrices (K, d, d); got {stack.shape}")
    out_dir = os.path.dirname(os.path.abspath(npz_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(npz_path, T=stack, alphabet_size=np.int32(stack.shape[-1]))
    pd.DataFrame(idx_rows).to_csv(index_csv_path, index=False)
    return len(Ts)


def _estimated_threshold_n(topo_df: pd.DataFrame) -> float:
    topo_df = topo_df.sort_values("n_points")
    lengths = topo_df["n_points"].values.astype(float)
    traces = topo_df["fisher_trace"].values
    if len(traces) < 3:
        return float("nan")
    d_trace = np.diff(traces) / np.diff(lengths)
    max_idx = int(np.argmax(np.abs(d_trace)))
    return float((lengths[max_idx] + lengths[max_idx + 1]) / 2)


def main():
    ap = argparse.ArgumentParser(description="Fisher information vs data length (Sycamore readouts)")
    ap.add_argument(
        "--input-dir",
        default=None,
        help="Folder with *_readout_raw_data.txt (default: same resolution as build_sycamore_readout_fingerprints).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only one representative file per topology (3 files).",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap number of readout files processed (full mode only).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for grammar learner (each N run). Default 42.",
    )
    ap.add_argument(
        "--tag-with-seed",
        action="store_true",
        help="Append _seed<SEED> to output filenames (use when running multiple seeds).",
    )
    ap.add_argument(
        "--output-tag",
        default=None,
        metavar="TAG",
        help="Append _TAG to output filenames (overrides --tag-with-seed). Example: run_a.",
    )
    ap.add_argument(
        "--no-save-transition-matrices",
        action="store_true",
        help="Skip writing transition_matrices_sycamore*.npz and index CSV (default: save).",
    )
    ap.add_argument(
        "--transition-matrices-npz",
        default=None,
        metavar="PATH",
        help="Override path for stacked T(N) arrays (.npz). Default: results/transition_matrices_sycamore<SUFFIX>.npz",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="LSTM epochs per (file, N) run. Default 50 (publication); use 3–5 for smoke tests only.",
    )
    ap.add_argument(
        "--alphabet-size",
        type=int,
        default=7,
        metavar="K",
        help="SAX alphabet size (transition matrix is K×K). Default 7; use 5 or 9 for alphabet sweeps.",
    )
    args = ap.parse_args()

    from run_validation_pipeline import GROUND_TRUTH

    data_dir = args.input_dir or default_readout_dir()
    if args.quick:
        jobs = _representative_three_files(data_dir, GROUND_TRUTH)
        base_suffix = ""
    else:
        jobs = _all_readout_jobs(data_dir, GROUND_TRUTH, args.max_files)
        base_suffix = "_all_readouts"
    suffix = _resolve_output_suffix(base_suffix, args.seed, args.output_tag, args.tag_with_seed)
    if args.alphabet_size != 7:
        suffix = f"{suffix}_alphabet{args.alphabet_size}"

    if not jobs:
        print(f"No readout files found in {data_dir!r} (expected *_readout_raw_data.txt with known q-labels).")
        return

    # Data lengths to test
    data_lengths = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
                    8000, 9000, 10000, 15000, 20000, 30000, 40000]

    print(f"Input directory: {data_dir}")
    print(f"Alphabet size (SAX K): {args.alphabet_size}")
    print(f"Seed: {args.seed}  |  Output file suffix: {suffix!r}")
    print(f"Mode: {'quick (3 files)' if args.quick else 'all readouts'} — {len(jobs)} file(s)")
    for fname, topo, lab in jobs:
        print(f"  {lab} ({topo}): {fname}")

    n_train_total = len(jobs) * sum(1 for n in data_lengths if n <= 110_000)
    print(
        f"Approx training runs: {n_train_total} (each {args.epochs} epochs); "
        "full 28-file run can take many hours."
    )

    params = {"hidden_dim": 32, "seq_len": 16, "epochs": args.epochs, "lr": 0.01}

    all_results = []
    all_matrices = {}  # {(file, length): matrix}

    for fname, topo, q_lab in jobs:
        fpath = os.path.join(data_dir, fname)
        raw_df = pd.read_csv(fpath, sep=" ", on_bad_lines="skip")
        full_signal = raw_df["output"].values.astype(np.float64)
        max_available = len(full_signal)
        
        print(f"\n{'='*60}")
        print(f"File: {fname} ({topo}), total points: {max_available}")
        print(f"{'='*60}")
        
        prev_matrix = None
        
        for n_pts in data_lengths:
            if n_pts > max_available:
                print(f"  N={n_pts}: skipped (only {max_available} available)")
                continue
                
            signal = full_signal[:n_pts]
            
            print(f"  N={n_pts}...", end=" ", flush=True)
            
            _, _, model, val_data = train_model(
                signal, f"{fname}_{n_pts}",
                alphabet_size=args.alphabet_size,
                data_is_array=True,
                seed=args.seed,
                **params,
            )
            
            if model is None:
                print("FAILED")
                continue
            
            # Extract transition matrix
            T = extract_grammar(model, val_data, seq_len=params["seq_len"])
            all_matrices[(fname, n_pts)] = T
            
            # Compute Fisher information
            F = compute_fisher_matrix(T)
            f_trace = fisher_scalar(F)
            f_det = fisher_determinant(F)
            f_maxeig = fisher_max_eigenvalue(F)
            
            # KL divergence from previous length
            kl_div = 0.0
            if prev_matrix is not None:
                kl_div = compute_kl_divergence(prev_matrix, T)
            
            # Matrix entropy (how uniform is the transition matrix)
            T_safe = np.maximum(T, 1e-10)
            T_norm = T_safe / T_safe.sum(axis=1, keepdims=True)
            entropy = -np.sum(T_norm * np.log(T_norm)) / T.shape[0]
            
            # Frobenius norm (how far from uniform)
            uniform = np.ones_like(T) / T.shape[0]
            frob_from_uniform = np.sqrt(np.sum((T - uniform) ** 2))
            
            result = {
                "file": fname,
                "q_label": q_lab,
                "topology": topo,
                "n_points": n_pts,
                "fisher_trace": f_trace,
                "fisher_det": f_det,
                "fisher_max_eigenvalue": f_maxeig,
                "kl_from_previous": kl_div,
                "matrix_entropy": entropy,
                "frobenius_from_uniform": frob_from_uniform,
            }
            all_results.append(result)
            
            print(f"trace={f_trace:.2f}, det={f_det:.2e}, KL={kl_div:.4f}, entropy={entropy:.4f}")
            
            prev_matrix = T.copy()
    
    # Save results
    df = pd.DataFrame(all_results)
    if df.empty:
        print("No successful training runs; no CSV/report/plot written.")
        return

    csv_name = f"fisher_metric_vs_datalength{suffix}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    if not args.no_save_transition_matrices and all_matrices:
        if args.transition_matrices_npz:
            npz_path = args.transition_matrices_npz
            index_csv = os.path.splitext(npz_path)[0] + "_index.csv"
        else:
            npz_path = os.path.join(RESULTS_DIR, f"transition_matrices_sycamore{suffix}.npz")
            index_csv = os.path.join(RESULTS_DIR, f"transition_matrices_index{suffix}.csv")
        k_saved = save_transition_matrices_npz(all_results, all_matrices, npz_path, index_csv)
        if k_saved:
            print(f"Saved: {npz_path}  ({k_saved} matrices, shape (K,7,7) in array 'T')")
            print(f"Saved: {index_csv}")

    # Per-file estimated threshold (from Fisher trace slope)
    thresh_rows = []
    for fname in df["file"].unique():
        g = df[df["file"] == fname]
        topo = g["topology"].iloc[0]
        ql = g["q_label"].iloc[0] if "q_label" in g.columns else ""
        thresh_rows.append(
            {
                "file": fname,
                "q_label": ql,
                "topology": topo,
                "estimated_N_threshold": _estimated_threshold_n(g),
            }
        )
    thresh_df = pd.DataFrame(thresh_rows).sort_values(["topology", "q_label"])
    thresh_csv = os.path.join(RESULTS_DIR, f"fisher_estimated_thresholds_per_readout{suffix}.csv")
    thresh_df.to_csv(thresh_csv, index=False)
    print(f"Saved: {thresh_csv}")

    # Generate report
    report_path = os.path.join(RESULTS_DIR, f"fisher_information_analysis{suffix}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== FISHER INFORMATION METRIC ANALYSIS ===\n\n")
        f.write(f"Files processed: {len(jobs)}\n")
        f.write(f"Input directory: {data_dir}\n\n")

        for fname in sorted(df["file"].unique()):
            file_df = df[df["file"] == fname].sort_values("n_points")
            topo = file_df["topology"].iloc[0]
            ql = file_df["q_label"].iloc[0] if "q_label" in file_df.columns else ""
            f.write(f"\n--- {ql} ({topo}) :: {fname} ---\n")
            f.write(
                f"{'N':>8} {'Fisher_Tr':>12} {'Fisher_Det':>14} {'KL_prev':>10} {'Entropy':>10} {'Frob_Unif':>10}\n"
            )
            f.write("-" * 70 + "\n")
            for _, row in file_df.iterrows():
                f.write(
                    f"{int(row['n_points']):>8} {row['fisher_trace']:>12.2f} {row['fisher_det']:>14.2e} "
                    f"{row['kl_from_previous']:>10.4f} {row['matrix_entropy']:>10.4f} {row['frobenius_from_uniform']:>10.4f}\n"
                )
            est = _estimated_threshold_n(file_df)
            if not np.isnan(est):
                f.write(f"\nEstimated phase transition (max |d(trace)/dN|): N ~ {int(est)}\n")
            f.write("\n")

        f.write("\n=== SUMMARY: estimated N by topology ===\n")
        for topo in ["1D_Snake", "2D_Block", "Bulk_Full"]:
            sub = thresh_df[thresh_df["topology"] == topo]["estimated_N_threshold"].dropna()
            if sub.empty:
                continue
            f.write(
                f"{topo}: n={len(sub)}, mean={sub.mean():.1f}, std={sub.std():.1f}, "
                f"min={sub.min():.0f}, max={sub.max():.0f}\n"
            )

        f.write("\n=== INTERPRETATION ===\n")
        f.write(
            "If the Fisher trace shows a sharp change near N~8000 (the Grammar Fingerprinting\n"
            "classification threshold), this suggests a geometric phase transition in the\n"
            "information manifold — supporting the hypothesis that the threshold represents\n"
            "a physical boundary where noise structure becomes geometrically organized.\n"
        )

    print(f"Saved: {report_path}")

    # Generate plot (per-topology: single bold line if one file, else thin per-file + mean)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = {"1D_Snake": "#e74c3c", "2D_Block": "#3498db", "Bulk_Full": "#27ae60"}
        metrics = [
            ("fisher_trace", "Fisher Trace", axes[0, 0]),
            ("kl_from_previous", "KL Divergence (from previous N)", axes[0, 1]),
            ("matrix_entropy", "Transition Matrix Entropy", axes[1, 0]),
            ("frobenius_from_uniform", "Frobenius Distance from Uniform", axes[1, 1]),
        ]

        for col, ylab, ax in metrics:
            for topo in ["1D_Snake", "2D_Block", "Bulk_Full"]:
                topo_df = df[df["topology"] == topo]
                if topo_df.empty:
                    continue
                base = colors[topo]
                n_files = topo_df["file"].nunique()
                if n_files == 1:
                    g = topo_df.sort_values("n_points")
                    ax.plot(
                        g["n_points"],
                        g[col],
                        "o-",
                        color=base,
                        label=topo,
                        linewidth=2,
                    )
                else:
                    for fn, g in topo_df.groupby("file"):
                        g = g.sort_values("n_points")
                        ax.plot(
                            g["n_points"],
                            g[col],
                            "-",
                            color=base,
                            alpha=0.28,
                            linewidth=1,
                        )
                    mean_s = topo_df.groupby("n_points", sort=True)[col].mean()
                    ax.plot(
                        mean_s.index,
                        mean_s.values,
                        "o-",
                        color=base,
                        linewidth=2.2,
                        label=f"{topo} (mean)",
                    )

        axes[0, 0].set_xlabel("Data Length (N)")
        axes[0, 0].set_ylabel("Fisher Trace")
        axes[0, 0].set_title("Fisher Information (Trace) vs Data Length")
        axes[0, 0].legend()
        axes[0, 0].axvline(x=8000, color="gray", linestyle="--", alpha=0.5)

        axes[0, 1].set_xlabel("Data Length (N)")
        axes[0, 1].set_ylabel("KL Divergence (from previous N)")
        axes[0, 1].set_title("Rate of Grammar Change")
        axes[0, 1].legend()
        axes[0, 1].axvline(x=8000, color="gray", linestyle="--", alpha=0.5)

        axes[1, 0].set_xlabel("Data Length (N)")
        axes[1, 0].set_ylabel("Transition Matrix Entropy")
        axes[1, 0].set_title("Grammar Entropy vs Data Length")
        axes[1, 0].legend()
        axes[1, 0].axvline(x=8000, color="gray", linestyle="--", alpha=0.5)

        axes[1, 1].set_xlabel("Data Length (N)")
        axes[1, 1].set_ylabel("Frobenius Distance from Uniform")
        axes[1, 1].set_title("Grammar Specialization vs Data Length")
        axes[1, 1].legend()
        axes[1, 1].axvline(x=8000, color="gray", linestyle="--", alpha=0.5)

        plt.suptitle("Fisher Information Metric: Phase Transition Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        fig_path = os.path.join(RESULTS_DIR, f"fisher_phase_transition{suffix}.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved: {fig_path}")

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
