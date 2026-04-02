"""
IBM Quantum — same logical protocol as Sycamore validation pipeline:

- LSTM + SAX (grammar_learner) + Frobenius fingerprints + Ward + k=3 clusters
- Original (unshuffled), Shuffled control, LSB (single-bit) control
- Baseline Logistic Regression + Random Forest on simple time-series features (LOOCV)
- Multiple seeds for LSTM runs

IBM filenames: ibm_<backend>_<circuit>_<N>q_<totalshots>shots.txt
Ground truth (physics3): hadamard+layers -> HadamardLayers, ghz -> GHZ, identity -> Identity
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from grammar_learner import extract_grammar, train_model

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SHOTS_DIR = os.path.join(REPO_ROOT, "results", "ibm_raw_shots")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def frobenius_distance(m1: np.ndarray, m2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((m1 - m2) ** 2)))


def parse_ibm_filename(name: str) -> Tuple[str, str, int, int] | None:
    """
    Returns (backend, circuit, n_qubits, total_shots) or None.
    Example: ibm_marrakesh_hadamard_10q_163840shots.txt
    """
    m = re.match(
        r"^ibm_([^_]+)_(hadamard|ghz|layers|identity)_(\d+)q_(\d+)shots\.txt$",
        name,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1), m.group(2).lower(), int(m.group(3)), int(m.group(4))


def physics3_regime(circuit: str) -> str:
    if circuit in ("hadamard", "layers"):
        return "HadamardLayers"
    if circuit == "ghz":
        return "GHZ"
    if circuit == "identity":
        return "Identity"
    raise ValueError(circuit)


def list_ibm_shot_files(
    shots_dir: str, backend_prefix: str | None, filename_substr: str | None
) -> List[str]:
    files = [f for f in os.listdir(shots_dir) if f.endswith(".txt") and "shots" in f]
    if backend_prefix:
        files = [f for f in files if f.startswith(f"{backend_prefix}_")]
    if filename_substr:
        files = [f for f in files if filename_substr in f]
    # Stable sort: circuit type, qubits, name
    def sort_key(fn: str) -> Tuple[str, int, str]:
        p = parse_ibm_filename(fn)
        if p is None:
            return ("", 0, fn)
        _, circ, nq, _ = p
        return (circ, nq, fn)

    files.sort(key=sort_key)
    return files


def build_ground_truth(files: List[str]) -> Dict[str, str]:
    gt: Dict[str, str] = {}
    for fn in files:
        p = parse_ibm_filename(fn)
        if p is None:
            continue
        backend, circuit, nq, _ = p
        label = f"{backend}_{circuit}_{nq}q"
        gt[label] = physics3_regime(circuit)
    return gt


def extract_features(signal: np.ndarray, max_pts: int) -> np.ndarray:
    """Simple baseline features for LR/RF (same max_pts truncation as LSTM path)."""
    x = signal.astype(np.float64)
    if len(x) > max_pts:
        x = x[:max_pts]
    if len(x) < 2:
        return np.zeros(16, dtype=np.float64)
    xm = float(np.mean(x))
    xs = float(np.std(x) + 1e-12)
    z = (x - xm) / xs
    d = np.diff(x)
    feats = [
        xm,
        xs,
        float(np.median(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(skew(z)),
        float(kurtosis(z, fisher=True)),
        float(np.mean(np.abs(d))),
        float(np.std(d)),
        float(np.mean(x % 2)),
        float(np.mean((x.astype(np.int64) & 1))),
    ]
    # A few bin counts on normalized values (coarse spectrum)
    hist, _ = np.histogram(z, bins=5, range=(-3, 3))
    feats.extend((hist / (len(z) + 1e-12)).astype(np.float64).tolist())
    out = np.array(feats, dtype=np.float64)
    if out.shape[0] < 16:
        out = np.pad(out, (0, 16 - out.shape[0]))
    return out[:16]


def run_lstm_cluster_accuracy(
    *,
    files: List[str],
    shots_dir: str,
    ground_truth: Dict[str, str],
    seed: int,
    epochs: int,
    max_pts: int,
    shuffle: bool,
    output_bit_index: int | None,
) -> Tuple[float, List[str]]:
    params = {"hidden_dim": 32, "seq_len": 16, "epochs": epochs, "lr": 0.01}
    rng = np.random.default_rng(seed) if shuffle else None
    fingerprints: List[np.ndarray] = []
    labels: List[str] = []

    for fn in files:
        p = parse_ibm_filename(fn)
        if p is None:
            continue
        backend, circuit, nq, _ = p
        label = f"{backend}_{circuit}_{nq}q"
        path = os.path.join(shots_dir, fn)
        df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
        raw = df["output"].values
        if len(raw) > max_pts:
            raw = raw[:max_pts]
        if output_bit_index is not None:
            k = int(output_bit_index)
            signal = np.array([((int(v) >> k) & 1) for v in raw], dtype=np.float64)
        else:
            signal = raw.astype(np.float64)
        if shuffle:
            signal = rng.permutation(signal)

        _, _, model, val_data = train_model(
            signal,
            label,
            alphabet_size=7,
            data_is_array=True,
            seed=seed,
            **params,
        )
        if model is None:
            continue
        fp = extract_grammar(model, val_data, seq_len=params["seq_len"])
        fingerprints.append(fp)
        labels.append(label)

    n_f = len(fingerprints)
    if n_f == 0:
        return 0.0, []

    dist = np.zeros((n_f, n_f), dtype=np.float64)
    for i in range(n_f):
        for j in range(n_f):
            dist[i, j] = frobenius_distance(fingerprints[i], fingerprints[j])
    Z = linkage(squareform(dist), method="ward")
    clusters = fcluster(Z, t=3, criterion="maxclust")

    results = pd.DataFrame(
        {
            "q_label": labels,
            "cluster": clusters,
            "true_regime": [ground_truth[q] for q in labels],
        }
    )

    correct = 0
    cluster_lines: List[str] = []
    for cluster_id in range(1, 4):
        cluster_data = results[results["cluster"] == cluster_id]
        if cluster_data.empty:
            cluster_lines.append(f"Cluster {cluster_id}: EMPTY (Purity: 0.0%)")
            continue
        majority_regime = cluster_data["true_regime"].value_counts().idxmax()
        majority_count = cluster_data["true_regime"].value_counts().max()
        purity = float((majority_count / len(cluster_data)) * 100)
        correct += int(majority_count)
        cluster_lines.append(
            f"Cluster {cluster_id}: {majority_regime} (Purity: {purity:.1f}%)"
        )

    accuracy = float((correct / n_f) * 100)
    return accuracy, cluster_lines


def run_lr_rf_baseline(
    *,
    files: List[str],
    shots_dir: str,
    y_labels: List[str],
    max_pts: int,
    seeds: List[int],
    rf_n_estimators: int,
) -> Dict[str, float]:
    """LOOCV accuracy for LR and RF; averaged over seeds (RF randomness)."""
    X_list: List[np.ndarray] = []
    for fn in files:
        path = os.path.join(shots_dir, fn)
        df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
        raw = df["output"].astype(np.float64).values
        X_list.append(extract_features(raw, max_pts))
    X = np.vstack(X_list)
    y = np.array(y_labels)
    loo = LeaveOneOut()
    lr_scores: List[float] = []
    rf_scores: List[float] = []
    for sd in seeds:
        lr = LogisticRegression(max_iter=5000, random_state=sd)
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=sd, n_jobs=-1
        )
        lr_scores.append(float(np.mean(cross_val_score(lr, X, y, cv=loo, scoring="accuracy"))))
        rf_scores.append(float(np.mean(cross_val_score(rf, X, y, cv=loo, scoring="accuracy"))))
    return {
        "lr_mean": float(np.mean(lr_scores)),
        "lr_std": float(np.std(lr_scores)),
        "rf_mean": float(np.mean(rf_scores)),
        "rf_std": float(np.std(rf_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IBM data: Sycamore-style validation protocol")
    parser.add_argument("--shots-dir", default=DEFAULT_SHOTS_DIR)
    parser.add_argument("--backend-prefix", default=None, help="e.g. ibm_marrakesh")
    parser.add_argument(
        "--filename-substr",
        default="163840",
        help="Only include shot files containing this substring (e.g. 163840 for reps=20).",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Tag for report filename (default: filename-substr or 'protocol').",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-pts", type=int, default=100_000)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 42, 123, 456, 999],
        help="Match validation_criteria.json default seed list unless overridden.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=200,
        help="RandomForest tree count (LOOCV can be slow if very large).",
    )
    parser.add_argument(
        "--modes",
        default="original,shuffled,lsb,baseline",
        help="Comma-separated: original, shuffled, lsb, baseline (LR/RF).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output report path (default: results/ibm_protocol_<tag>_<backend>.txt)",
    )
    args = parser.parse_args()

    files = list_ibm_shot_files(args.shots_dir, args.backend_prefix, args.filename_substr)
    if not files:
        print(
            f"No matching IBM shot files in {args.shots_dir} "
            f"(prefix={args.backend_prefix}, substr={args.filename_substr}).",
            flush=True,
        )
        sys.exit(1)

    gt = build_ground_truth(files)
    y_regimes: List[str] = []
    for fn in files:
        p = parse_ibm_filename(fn)
        if p is None:
            continue
        backend, circuit, nq, _ = p
        y_regimes.append(gt[f"{backend}_{circuit}_{nq}q"])

    backend_name = args.backend_prefix or "ibm_multi"
    tag = args.output_tag or (args.filename_substr if args.filename_substr else "protocol")
    out_path = args.out or os.path.join(
        RESULTS_DIR,
        f"ibm_protocol_{tag}_{backend_name.replace('ibm_', '')}.txt",
    )

    lines: List[str] = []
    lines.append("=== IBM QUANTUM — SYCAMORE-STYLE PROTOCOL (physics3 regimes) ===\n\n")
    lines.append(f"shots_dir: {args.shots_dir}\n")
    lines.append(f"files ({len(files)}):\n")
    for fn in files:
        lines.append(f"  {fn}\n")
    mode_set = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    lines.append(f"\nepochs={args.epochs}, max_pts={args.max_pts}, seeds={list(args.seeds)}\n")
    lines.append(f"modes={sorted(mode_set)}\n")
    lines.append("Regimes: HadamardLayers (hadamard+layers), GHZ, Identity\n\n")

    # --- Original (unshuffled) LSTM path ---
    if "original" in mode_set:
        lines.append("--- ORIGINAL (unshuffled, full integer readout) ---\n")
        for sd in args.seeds:
            acc, cl = run_lstm_cluster_accuracy(
                files=files,
                shots_dir=args.shots_dir,
                ground_truth=gt,
                seed=sd,
                epochs=args.epochs,
                max_pts=args.max_pts,
                shuffle=False,
                output_bit_index=None,
            )
            lines.append(f"seed={sd} accuracy={acc:.2f}%\n")
            for x in cl:
                lines.append(f"  {x}\n")
            lines.append("\n")
            print(f"[original] seed={sd} acc={acc:.2f}%", flush=True)

    # --- Shuffled control ---
    if "shuffled" in mode_set:
        lines.append("--- SHUFFLED CONTROL (per-file permutation destroys temporal order) ---\n")
        for sd in args.seeds:
            acc, cl = run_lstm_cluster_accuracy(
                files=files,
                shots_dir=args.shots_dir,
                ground_truth=gt,
                seed=sd,
                epochs=args.epochs,
                max_pts=args.max_pts,
                shuffle=True,
                output_bit_index=None,
            )
            lines.append(f"seed={sd} accuracy={acc:.2f}%\n")
            for x in cl:
                lines.append(f"  {x}\n")
            lines.append("\n")
            print(f"[shuffled] seed={sd} acc={acc:.2f}%", flush=True)

    # --- LSB control ---
    if "lsb" in mode_set:
        lines.append("--- LSB CONTROL (single-bit time series: (output>>0)&1) ---\n")
        for sd in args.seeds:
            acc, cl = run_lstm_cluster_accuracy(
                files=files,
                shots_dir=args.shots_dir,
                ground_truth=gt,
                seed=sd,
                epochs=args.epochs,
                max_pts=args.max_pts,
                shuffle=False,
                output_bit_index=0,
            )
            lines.append(f"seed={sd} accuracy={acc:.2f}%\n")
            for x in cl:
                lines.append(f"  {x}\n")
            lines.append("\n")
            print(f"[lsb] seed={sd} acc={acc:.2f}%", flush=True)

    # --- LR / RF baseline ---
    if "baseline" in mode_set:
        lines.append(
            "--- BASELINE: Logistic Regression + Random Forest (LOOCV, hand-crafted features) ---\n"
        )
        br = run_lr_rf_baseline(
            files=files,
            shots_dir=args.shots_dir,
            y_labels=y_regimes,
            max_pts=args.max_pts,
            seeds=list(args.seeds),
            rf_n_estimators=int(args.rf_n_estimators),
        )
        lines.append(
            f"LR LOOCV: mean={br['lr_mean']:.4f} std={br['lr_std']:.4f}\n"
            f"RF LOOCV: mean={br['rf_mean']:.4f} std={br['rf_std']:.4f}\n"
        )
        print(f"[baseline] LR={br['lr_mean']:.4f} RF={br['rf_mean']:.4f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
