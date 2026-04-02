import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from grammar_learner import extract_grammar, train_model


def qubits_from_label(label: str) -> int | None:
    m = re.search(r"_(\d+)q_", label)
    if not m:
        return None
    return int(m.group(1))


def group_label_from_filename(name: str) -> str | None:
    # Examples:
    #   ibm_marrakesh_ghz_10q_40960shots.txt
    #   ibm_marrakesh_hadamard_10q_40960shots.txt
    #   ibm_marrakesh_layers_20q_40960shots.txt
    if "_ghz_" in name:
        return "ghz"
    if "_hadamard_" in name or "_layers_" in name:
        return "hadamard_plus_layers"
    return None


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def purity_from_clusters(
    clusters: np.ndarray, eval_labels: List[str]
) -> Tuple[float, Dict[int, Dict[str, int]]]:
    cluster_ids = sorted(set(int(c) for c in clusters))
    counts_by_cluster: Dict[int, Dict[str, int]] = {}
    correct = 0
    for cid in cluster_ids:
        idx = [i for i, c in enumerate(clusters) if int(c) == cid]
        cnt: Dict[str, int] = {}
        for i in idx:
            t = eval_labels[i]
            cnt[t] = cnt.get(t, 0) + 1
        counts_by_cluster[cid] = cnt
        correct += max(cnt.values()) if cnt else 0
    return correct / len(eval_labels), counts_by_cluster


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold analysis: GHZ vs Hadamard+layers separation using varying max_pts."
    )
    parser.add_argument("--backend-prefix", required=True, type=str)
    parser.add_argument("--only-qubits", type=str, default="10,20")
    parser.add_argument("--max-points", type=str, default="4000,8000,16000")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--alphabet-size", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filename-substr",
        type=str,
        default=None,
        help="If set, only use shot files whose name contains this (e.g. 163840).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional tag appended to output CSV name (e.g. reps20).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(SCRIPT_DIR)
    shots_dir = os.path.join(repo_root, "results", "ibm_raw_shots")
    tag_sfx = f"_{args.output_tag}" if args.output_tag else ""
    out_csv = os.path.join(
        repo_root,
        "results",
        f"{args.backend_prefix}_ghz_vs_hadamardlayers_threshold{tag_sfx}.csv",
    )

    only_q = {int(x.strip()) for x in args.only_qubits.split(",") if x.strip()}
    max_pts_list = [int(x.strip()) for x in args.max_points.split(",") if x.strip()]

    pattern = os.path.join(shots_dir, f"{args.backend_prefix}_*shots.txt")
    # Use pandas directly; glob via os.listdir to keep it simple/consistent.
    all_files = [
        fn
        for fn in os.listdir(shots_dir)
        if fn.startswith(f"{args.backend_prefix}_") and fn.endswith("shots.txt")
    ]
    if args.filename_substr:
        all_files = [fn for fn in all_files if args.filename_substr in fn]

    # Filter to qubit range and to GHZ/Hadamard/Layers groups.
    items: List[Tuple[str, str]] = []  # (filename, eval_label)
    for fn in sorted(all_files):
        base = fn.replace(".txt", "")
        q = qubits_from_label(base)
        if q is None or q not in only_q:
            continue
        g = group_label_from_filename(fn)
        if g is None:
            continue
        items.append((fn, g))

    if not items:
        raise RuntimeError(f"No input files matched for backend={args.backend_prefix}")

    # We want stable ordering: group label then qubits (10 then 20).
    def sort_key(item: Tuple[str, str]) -> Tuple[str, int]:
        fn, _ = item
        base = fn.replace(".txt", "")
        q = qubits_from_label(base) or -1
        g = group_label_from_filename(fn) or "unknown"
        return (g, q)

    items = sorted(items, key=sort_key)
    filenames = [fn for fn, _ in items]
    eval_labels = [lab for _, lab in items]

    print(f"Files ({len(items)}):")
    for fn, lab in items:
        print(f"  {fn} -> {lab}")

    # Constant training params (except max_pts).
    params = {
        "alphabet_size": args.alphabet_size,
        "hidden_dim": args.hidden_dim,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "data_is_array": True,
        "seed": args.seed,
    }

    rows = []
    for max_pts in max_pts_list:
        fps: List[np.ndarray] = []
        for fn, _ in items:
            path = os.path.join(shots_dir, fn)
            df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
            signal = df["output"].astype(np.float64).values
            if len(signal) > max_pts:
                signal = signal[:max_pts]

            label = fn.replace(".txt", "")
            _, _, model, val_data = train_model(signal, label, **params)
            if model is None:
                raise RuntimeError(f"Training failed for {fn} at max_pts={max_pts}")
            fp = extract_grammar(model, val_data, seq_len=params["seq_len"])
            fps.append(fp)

        n = len(fps)
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                dist[i, j] = frobenius_distance(fps[i], fps[j])

        Z = linkage(squareform(dist), method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")
        purity, counts_by_cluster = purity_from_clusters(clusters, eval_labels)

        # Within vs between averages (for interpretability).
        within = []
        between = []
        for i in range(n):
            for j in range(i + 1, n):
                if eval_labels[i] == eval_labels[j]:
                    within.append(dist[i, j])
                else:
                    between.append(dist[i, j])

        rows.append(
            {
                "backend_prefix": args.backend_prefix,
                "max_pts": max_pts,
                "n_files": n,
                "purity_k2": purity,
                "within_mean": float(np.mean(within)) if within else float("nan"),
                "between_mean": float(np.mean(between)) if between else float("nan"),
                "within_to_between_ratio": float(np.mean(within)) / float(np.mean(between))
                if within and between
                else float("nan"),
                "cluster_counts": str(counts_by_cluster),
            }
        )

        print(
            f"[{args.backend_prefix}] max_pts={max_pts}: purity_k2={purity:.4f} "
            f"(within_mean={np.mean(within):.4f}, between_mean={np.mean(between):.4f})"
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved threshold report: {out_csv}")


if __name__ == "__main__":
    main()

