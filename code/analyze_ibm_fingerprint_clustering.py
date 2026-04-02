import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def frobenius_distance(m1: np.ndarray, m2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((m1 - m2) ** 2)))


def circuit_type_from_label(label: str) -> str:
    # e.g. ibm_torino_hadamard_10q_40960shots -> hadamard
    m = re.search(r"_(hadamard|ghz|layers|identity)_", label)
    if not m:
        return "unknown"
    return m.group(1)


def map_type(circuit_type: str, grouping: str) -> str:
    if grouping == "physics3":
        if circuit_type in {"hadamard", "layers"}:
            return "hadamard_plus_layers"
        return circuit_type
    if grouping == "identity2":
        if circuit_type == "identity":
            return "identity"
        return "non_identity"
    return circuit_type


def qubits_from_label(label: str) -> int | None:
    m = re.search(r"_(\d+)q_", label)
    if not m:
        return None
    return int(m.group(1))


def map_type_with_qubits(circuit_type: str, qubits: int | None, grouping: str) -> str:
    if grouping == "ghz10_low_vs_rest_high":
        # Low group: identity (any) + ghz_10q
        # High group: hadamard/layers (any) + ghz_20q
        if circuit_type == "identity":
            return "identity_or_ghz_low"
        if circuit_type == "ghz" and qubits == 10:
            return "identity_or_ghz_low"
        return "hadamard_layers_or_ghz_high"
    return map_type(circuit_type, grouping)


def analyze(
    npz_path: str,
    out_prefix: str,
    k: int | None,
    grouping: str,
    exclude_substr: str | None,
    only_qubits: str | None,
) -> None:
    data = np.load(npz_path)
    labels = sorted(data.files)
    if exclude_substr:
        labels = [lbl for lbl in labels if exclude_substr not in lbl]
    if only_qubits:
        allowed = {int(x.strip()) for x in only_qubits.split(",") if x.strip()}
        filtered = []
        for lbl in labels:
            q = qubits_from_label(lbl)
            if q is None:
                continue
            if q in allowed:
                filtered.append(lbl)
        labels = filtered
    mats = [data[k] for k in labels]
    raw_types = [circuit_type_from_label(k) for k in labels]
    if grouping == "ghz10_low_vs_rest_high":
        types = [map_type_with_qubits(t, qubits_from_label(lbl), grouping) for lbl, t in zip(labels, raw_types)]
    else:
        types = [map_type(t, grouping) for t in raw_types]

    n = len(labels)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = frobenius_distance(mats[i], mats[j])

    Z = linkage(squareform(dist), method="ward")
    n_clusters = int(k) if k is not None else len(sorted(set(types)))
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    cluster_to_types: dict[int, list[str]] = defaultdict(list)
    for c, t in zip(clusters, types):
        cluster_to_types[int(c)].append(t)

    total_majority = 0
    purity_rows = []
    for c in sorted(cluster_to_types):
        c_types = cluster_to_types[c]
        cnt = Counter(c_types)
        maj_type, maj_count = cnt.most_common(1)[0]
        purity = maj_count / len(c_types)
        total_majority += maj_count
        purity_rows.append((c, len(c_types), maj_type, purity, dict(cnt)))

    overall_purity = total_majority / n

    out_matrix_csv = os.path.join(RESULTS_DIR, f"{out_prefix}_frobenius_distance_matrix.csv")
    with open(out_matrix_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label"] + labels)
        for i, lbl in enumerate(labels):
            w.writerow([lbl] + [f"{dist[i, j]:.10f}" for j in range(n)])

    out_assign_csv = os.path.join(RESULTS_DIR, f"{out_prefix}_ward_clusters.csv")
    with open(out_assign_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "raw_circuit_type", "eval_circuit_type", "cluster"])
        for lbl, raw_t, t, c in zip(labels, raw_types, types, clusters):
            w.writerow([lbl, raw_t, t, int(c)])

    out_report = os.path.join(RESULTS_DIR, f"{out_prefix}_ward_report.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(f"Input: {npz_path}\n")
        f.write(f"Items: {n}\n")
        f.write(f"Grouping mode: {grouping}\n")
        f.write(f"Clusters (Ward maxclust): {n_clusters}\n")
        f.write(f"Overall type-separation purity: {overall_purity:.4f}\n\n")
        f.write("Cluster composition by circuit type:\n")
        for c, size, maj_type, purity, cnt in purity_rows:
            f.write(
                f"  Cluster {c}: size={size}, majority={maj_type}, purity={purity:.4f}, counts={cnt}\n"
            )

    print(f"Saved: {out_matrix_csv}")
    print(f"Saved: {out_assign_csv}")
    print(f"Saved: {out_report}")
    print(f"Overall type-separation purity: {overall_purity:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=str)
    parser.add_argument("--out-prefix", required=True, type=str)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--grouping",
        type=str,
        default="raw4",
        choices=["raw4", "physics3", "identity2", "ghz10_low_vs_rest_high"],
        help="raw4: 4 types; physics3: hadamard+layers merged; identity2: identity vs non_identity; ghz10_low_vs_rest_high: identity+ghz10 vs hadamard+layers+ghz20",
    )
    parser.add_argument("--exclude-substr", type=str, default=None)
    parser.add_argument(
        "--only-qubits",
        type=str,
        default="10,20",
        help="Comma-separated qubit counts to keep (e.g. '10,20' keeps only _10q_ and _20q_ labels).",
    )
    args = parser.parse_args()
    analyze(args.npz, args.out_prefix, args.k, args.grouping, args.exclude_substr, args.only_qubits)


if __name__ == "__main__":
    main()
