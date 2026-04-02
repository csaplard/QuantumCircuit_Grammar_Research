"""
Cross-platform universality (Item 1): same Grammar Fingerprint pipeline on
IBM shot data vs Sycamore readout data — Frobenius distances, Ward clustering,
purity vs known regimes.

IBM regimes: physics3 (HadamardLayers, GHZ, Identity) from filename.
Sycamore regimes: topology family from validation GROUND_TRUTH (1D_Snake, 2D_Block, Bulk_Full).

Writes:
  results/cross_platform_universality_report.txt
  results/cross_platform_universality_summary.csv
  results/cross_platform_regime_pair_frobenius.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Same as run_validation_pipeline.GROUND_TRUTH (topology label per Nq)
SYCAMORE_TOPOLOGY: Dict[str, str] = {
    "14q": "1D_Snake",
    "18q": "1D_Snake",
    "22q": "1D_Snake",
    "28q": "1D_Snake",
    "32q": "1D_Snake",
    "47q": "1D_Snake",
    "49q": "1D_Snake",
    "12q": "2D_Block",
    "16q": "2D_Block",
    "20q": "2D_Block",
    "24q": "2D_Block",
    "30q": "2D_Block",
    "34q": "2D_Block",
    "39q": "2D_Block",
    "40q": "2D_Block",
    "41q": "2D_Block",
    "42q": "2D_Block",
    "43q": "2D_Block",
    "44q": "2D_Block",
    "45q": "2D_Block",
    "50q": "2D_Block",
    "26q": "Bulk_Full",
    "36q": "Bulk_Full",
    "38q": "Bulk_Full",
    "46q": "Bulk_Full",
    "48q": "Bulk_Full",
    "51q": "Bulk_Full",
    "53q": "Bulk_Full",
}


def frobenius_distance(m1: np.ndarray, m2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((m1 - m2) ** 2)))


def ibm_physics3_regime(label: str) -> str:
    m = re.search(r"_(hadamard|ghz|layers|identity)_", label, re.IGNORECASE)
    if not m:
        return "unknown"
    c = m.group(1).lower()
    if c in ("hadamard", "layers"):
        return "HadamardLayers"
    if c == "ghz":
        return "GHZ"
    if c == "identity":
        return "Identity"
    return "unknown"


def load_npz(path: str) -> Tuple[List[str], List[np.ndarray]]:
    data = np.load(path)
    keys = sorted(data.files)
    return keys, [data[k] for k in keys]


def ward_purity(
    labels: List[str],
    mats: List[np.ndarray],
    regimes: List[str],
    k: int,
) -> Tuple[float, np.ndarray, List[str]]:
    n = len(labels)
    if n < 2:
        return 1.0, np.ones(n, dtype=np.int32), regimes
    k = max(2, min(int(k), n))
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = frobenius_distance(mats[i], mats[j])
    Z = linkage(squareform(dist), method="ward")
    clusters = fcluster(Z, t=k, criterion="maxclust")
    total_majority = 0
    for c in range(1, int(clusters.max()) + 1):
        idx = [i for i in range(n) if clusters[i] == c]
        if not idx:
            continue
        sub = [regimes[i] for i in idx]
        cnt = Counter(sub)
        total_majority += cnt.most_common(1)[0][1]
    purity = float(total_majority / n) if n else 0.0
    return purity, clusters, regimes


def within_between_frobenius(mats: List[np.ndarray], regimes: List[str]) -> Tuple[float, float, int, int]:
    """Mean distance for same regime vs different regime pairs."""
    n = len(mats)
    same_sum, same_n = 0.0, 0
    diff_sum, diff_n = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = frobenius_distance(mats[i], mats[j])
            if regimes[i] == regimes[j]:
                same_sum += d
                same_n += 1
            else:
                diff_sum += d
                diff_n += 1
    mean_same = float(same_sum / same_n) if same_n else float("nan")
    mean_diff = float(diff_sum / diff_n) if diff_n else float("nan")
    return mean_same, mean_diff, same_n, diff_n


def regime_pair_frobenius_table(
    mats: List[np.ndarray], regimes: List[str]
) -> List[Tuple[str, str, float, int]]:
    """Unordered regime pairs (A<=B lexicographically): mean Frobenius, pair count."""
    n = len(mats)
    regs = sorted(set(regimes))
    acc: dict[Tuple[str, str], list[float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = regimes[i], regimes[j]
            if a > b:
                a, b = b, a
            key = (a, b)
            acc.setdefault(key, []).append(frobenius_distance(mats[i], mats[j]))
    rows: List[Tuple[str, str, float, int]] = []
    for a in regs:
        for b in regs:
            if a > b:
                continue
            vals = acc.get((a, b), [])
            if not vals:
                continue
            rows.append((a, b, float(np.mean(vals)), len(vals)))
    return rows


def subset_indices(regimes: List[str], keep: set[str]) -> List[int]:
    return [i for i, r in enumerate(regimes) if r in keep]


def per_regime_within_frobenius(mats: List[np.ndarray], regimes: List[str]) -> List[Tuple[str, float, int]]:
    """Mean distance over pairs with both endpoints in the same named regime."""
    out: List[Tuple[str, float, int]] = []
    for r in sorted(set(regimes)):
        idx = [i for i, x in enumerate(regimes) if x == r]
        if len(idx) < 2:
            out.append((r, float("nan"), 0))
            continue
        s, c = 0.0, 0
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                s += frobenius_distance(mats[idx[ii]], mats[idx[jj]])
                c += 1
        out.append((r, float(s / c), c))
    return out


def append_regime_breakdown_section(
    lines: List[str],
    platform: str,
    mats: List[np.ndarray],
    regimes: List[str],
) -> None:
    lines.append(f"--- Regime breakdown: {platform} ---\n")
    unk = sum(1 for r in regimes if r == "unknown")
    if unk:
        lines.append(f"  (warning: {unk} points labeled unknown)\n")

    lines.append("  Within-regime mean Frobenius (same label pairs):\n")
    for r, mean_d, npairs in per_regime_within_frobenius(mats, regimes):
        if npairs == 0:
            lines.append(f"    {r}: n<2, no pairs\n")
        else:
            lines.append(f"    {r}: mean={mean_d:.6g} (pairs={npairs})\n")

    lines.append("  Unordered regime-pair mean Frobenius (all contributing pairs):\n")
    for a, b, mean_d, npairs in regime_pair_frobenius_table(mats, regimes):
        lines.append(f"    {a} | {b}: mean={mean_d:.6g} (pairs={npairs})\n")
    lines.append("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-platform IBM vs Sycamore fingerprint report")
    ap.add_argument(
        "--sycamore-npz",
        default=os.path.join(RESULTS_DIR, "sycamore_readout_grammar_fingerprints.npz"),
        help="Path to Sycamore grammar fingerprints .npz",
    )
    args = ap.parse_args()

    ibm_npz = os.path.join(RESULTS_DIR, "ibm_grammar_fingerprints.npz")
    ibm2_npz = os.path.join(RESULTS_DIR, "ibm_grammar_fingerprints_torino.npz")
    syc_npz = os.path.abspath(args.sycamore_npz)

    out_txt = os.path.join(RESULTS_DIR, "cross_platform_universality_report.txt")
    out_csv = os.path.join(RESULTS_DIR, "cross_platform_universality_summary.csv")
    out_pairs_csv = os.path.join(RESULTS_DIR, "cross_platform_regime_pair_frobenius.csv")

    pair_csv_rows: List[dict] = []
    lines: List[str] = []
    lines.append("=== Cross-platform universality (Grammar Fingerprint) ===\n\n")
    lines.append(
        "Protocol: LSTM + SAX (grammar_learner), alphabet_size=7, hidden_dim=32, seq_len=16, lr=0.01.\n"
        "  IBM fingerprints: epochs=20 (run_ibm_grammar_learning.py).\n"
        "  Sycamore fingerprints: epochs=50 (original validation / build_sycamore_readout_fingerprints.py default).\n\n"
    )

    rows_out: List[dict] = []

    # --- IBM: merge Marrakesh + Torino if present ---
    ibm_labels, ibm_mats = [], []
    for path, tag in [(ibm_npz, "marrakesh"), (ibm2_npz, "torino")]:
        if not os.path.isfile(path):
            lines.append(f"Missing (skip): {path}\n")
            continue
        keys, mats = load_npz(path)
        for k, m in zip(keys, mats):
            ibm_labels.append(f"{tag}::{k}")
            ibm_mats.append(m)

    if ibm_mats:
        ibm_regimes = [ibm_physics3_regime(k.split("::")[-1]) for k in ibm_labels]
        purity_ibm, _, _ = ward_purity(ibm_labels, ibm_mats, ibm_regimes, k=3)
        ms, md, sn, dn = within_between_frobenius(ibm_mats, ibm_regimes)
        lines.append(f"IBM (combined): n={len(ibm_labels)} fingerprints\n")
        lines.append(f"  Ward k=3 purity vs physics3: {purity_ibm:.4f}\n")
        lines.append(
            f"  Frobenius: mean same-regime={ms:.6g} (pairs={sn}) "
            f"mean diff-regime={md:.6g} (pairs={dn})\n\n"
        )
        rows_out.append(
            {
                "platform": "IBM_marrakesh_torino",
                "n": len(ibm_labels),
                "ward_k3_purity": f"{purity_ibm:.6f}",
                "frobenius_mean_same_regime": f"{ms:.8g}" if sn else "",
                "frobenius_mean_diff_regime": f"{md:.8g}" if dn else "",
            }
        )
        append_regime_breakdown_section(lines, "IBM (physics3)", ibm_mats, ibm_regimes)
        for a, b, mean_d, npairs in regime_pair_frobenius_table(ibm_mats, ibm_regimes):
            pair_csv_rows.append(
                {
                    "platform": "IBM_marrakesh_torino",
                    "regime_a": a,
                    "regime_b": b,
                    "mean_frobenius": f"{mean_d:.10g}",
                    "n_pairs": str(npairs),
                }
            )
    else:
        lines.append("IBM: no fingerprints found. Run run_ibm_grammar_learning.py\n\n")

    # --- Sycamore ---
    if os.path.isfile(syc_npz):
        keys, mats = load_npz(syc_npz)
        syc_regimes: List[str] = []
        for k in keys:
            m = re.search(r"sycamore_(\d+q)", k)
            q = m.group(1) if m else ""
            syc_regimes.append(SYCAMORE_TOPOLOGY.get(q, "unknown"))
        k_use = min(3, len(keys), max(2, len(set(syc_regimes))))
        purity_s, _, _ = ward_purity(keys, mats, syc_regimes, k=k_use)
        ms, md, sn, dn = within_between_frobenius(mats, syc_regimes)
        lines.append(f"Sycamore readout: n={len(keys)} fingerprints\n")
        lines.append(f"  Ward k={k_use} purity vs topology family: {purity_s:.4f}\n")
        lines.append(
            f"  Frobenius: mean same-regime={ms:.6g} (pairs={sn}) "
            f"mean diff-regime={md:.6g} (pairs={dn})\n\n"
        )
        rows_out.append(
            {
                "platform": "Sycamore_readout",
                "n": len(keys),
                "ward_k3_purity": f"{purity_s:.6f}",
                "frobenius_mean_same_regime": f"{ms:.8g}" if sn else "",
                "frobenius_mean_diff_regime": f"{md:.8g}" if dn else "",
            }
        )

        append_regime_breakdown_section(lines, "Sycamore (topology labels)", mats, syc_regimes)
        for a, b, mean_d, npairs in regime_pair_frobenius_table(mats, syc_regimes):
            pair_csv_rows.append(
                {
                    "platform": "Sycamore_readout",
                    "regime_a": a,
                    "regime_b": b,
                    "mean_frobenius": f"{mean_d:.10g}",
                    "n_pairs": str(npairs),
                }
            )

        # Subset Ward: discrete families vs boundary-heavy subsets
        lines.append("--- Sycamore subset Ward purity (sanity: where does global purity come from?) ---\n")
        subs: List[Tuple[str, List[int], int]] = []
        i12 = subset_indices(syc_regimes, {"1D_Snake", "2D_Block"})
        i23 = subset_indices(syc_regimes, {"2D_Block", "Bulk_Full"})
        i13 = subset_indices(syc_regimes, {"1D_Snake", "Bulk_Full"})
        if len(i12) >= 2 and len(set(syc_regimes[i] for i in i12)) >= 2:
            subs.append(("1D_Snake + 2D_Block only", i12, 2))
        if len(i23) >= 2 and len(set(syc_regimes[i] for i in i23)) >= 2:
            subs.append(("2D_Block + Bulk_Full only", i23, 2))
        if len(i13) >= 2 and len(set(syc_regimes[i] for i in i13)) >= 2:
            subs.append(("1D_Snake + Bulk_Full only", i13, 2))
        for name, idxs, k_sub in subs:
            sub_m = [mats[i] for i in idxs]
            sub_r = [syc_regimes[i] for i in idxs]
            sub_l = [keys[i] for i in idxs]
            p_sub, _, _ = ward_purity(sub_l, sub_m, sub_r, k=k_sub)
            lines.append(f"  {name}: n={len(idxs)} Ward k={k_sub} purity={p_sub:.4f}\n")
        lines.append("\n")
    else:
        lines.append(
            f"Sycamore: no {os.path.basename(syc_npz)} — run: "
            f"python code/build_sycamore_readout_fingerprints.py [--only-qubits 12,14,16]\n\n"
        )

    lines.append("Interpretation (evidence-linked):\n")
    lines.append(
        "  - Compare regime-pair Frobenius means above: IBM circuit families are disjoint; "
        "Sycamore topology labels can show small Bulk<->2D_Block distances if configurations overlap.\n"
    )
    lines.append(
        "  - If 1D+2D subset purity >> full k=3 purity while 2D+Bulk is weaker, global mixing "
        "is driven by boundary physics, not a broken pipeline.\n"
    )
    lines.append(
        "  - Cross-platform: same code path; different ground-truth geometry (discrete circuits vs "
        "overlapping topology classes).\n"
    )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    with open(out_pairs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["platform", "regime_a", "regime_b", "mean_frobenius", "n_pairs"]
        )
        w.writeheader()
        w.writerows(pair_csv_rows)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "platform",
                "n",
                "ward_k3_purity",
                "frobenius_mean_same_regime",
                "frobenius_mean_diff_regime",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    print(out_txt)
    print(out_csv)
    print(out_pairs_csv)


if __name__ == "__main__":
    main()
