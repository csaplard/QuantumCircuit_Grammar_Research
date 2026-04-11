"""
Print (or optionally run) the embedding + fit chain for Sycamore Fisher at SAX alphabet K.

Full Fisher re-run per K is expensive (same order as the k=7 overnight job). Typical workflow:

1) Train + save T(N) matrices:
     python code/fisher_information_analysis.py --alphabet-size K

   This appends ``_alphabetK`` to result filenames when K != 7 (k=7 keeps legacy names).

2) Fisher speed + log-det along each curve:
     python code/compute_fisher_ricci_curve.py --skip-ricci \\
       --npz results/transition_matrices_sycamore_all_readouts_alphabetK.npz \\
       --out results/fisher_ricci_and_embedding_vs_n_alphabetK.csv

3) Fit exp/pow + A*tau vs k(k-1)*pi/2:
     python code/fit_fisher_speed_foti_scale.py \\
       --embedding-csv results/fisher_ricci_and_embedding_vs_n_alphabetK.csv \\
       --thresholds-csv results/fisher_estimated_thresholds_per_readout_all_readouts_alphabetK.csv \\
       --alphabet-k K \\
       --out results/fisher_speed_foti_scale_fit_alphabetK.csv

Invariant check (hypothesis): median(A*tau) ~ k*(k-1)*pi/2 across alphabet sizes when other settings match.

Usage:
  python code/sycamore_alphabet_k_pipeline.py
  python code/sycamore_alphabet_k_pipeline.py --ks 5 9
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
PY = sys.executable
CODE = os.path.join(REPO_ROOT, "code")


def lines_for_k(k: int) -> list[str]:
    if k == 7:
        suf = "_all_readouts"
        npz = os.path.join(RESULTS, f"transition_matrices_sycamore{suf}.npz")
        emb = os.path.join(RESULTS, "fisher_ricci_and_embedding_vs_n.csv")
        thr = os.path.join(RESULTS, f"fisher_estimated_thresholds_per_readout{suf}.csv")
        fit_out = os.path.join(RESULTS, "fisher_speed_foti_scale_fit.csv")
    else:
        suf = f"_all_readouts_alphabet{k}"
        npz = os.path.join(RESULTS, f"transition_matrices_sycamore{suf}.npz")
        emb = os.path.join(RESULTS, f"fisher_ricci_and_embedding_vs_n_alphabet{k}.csv")
        thr = os.path.join(RESULTS, f"fisher_estimated_thresholds_per_readout{suf}.csv")
        fit_out = os.path.join(RESULTS, f"fisher_speed_foti_scale_fit_alphabet{k}.csv")
    out: list[str] = []
    out.append(
        f"{PY} {os.path.join(CODE, 'fisher_information_analysis.py')} --alphabet-size {k}"
    )
    out.append(
        f"{PY} {os.path.join(CODE, 'compute_fisher_ricci_curve.py')} --skip-ricci "
        f'--npz "{npz}" --out "{emb}"'
    )
    out.append(
        f"{PY} {os.path.join(CODE, 'fit_fisher_speed_foti_scale.py')} "
        f'--embedding-csv "{emb}" --thresholds-csv "{thr}" --alphabet-k {k} --out "{fit_out}"'
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sycamore alphabet-K Fisher embedding pipeline commands")
    ap.add_argument("--ks", type=int, nargs="*", default=[5, 7, 9], help="Alphabet sizes to emit")
    args = ap.parse_args()

    inv = lambda k: k * (k - 1) * 3.141592653589793 / 2.0
    print("Hypothesis scale k*(k-1)*pi/2:")
    for k in args.ks:
        print(f"  k={k}: {inv(k):.6g}")
    print()

    for k in args.ks:
        print(f"=== K = {k} ===")
        for ln in lines_for_k(k):
            print(ln)
        print()


if __name__ == "__main__":
    main()
