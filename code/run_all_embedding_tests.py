"""
Run embedding / Fisher-speed tests in sequence.

  Default (--quick): fast validation path
    1) LLM: run_llm_entropy_grammar_fisher --smoke + transition npz
    2) LLM: compute_fisher_ricci_curve --skip-ricci + fit_fisher_speed --alphabet-k 7
    3) Sycamore: fisher_information_analysis --quick --alphabet-size 5 (3 readouts)
    4) Sycamore: embedding + fit for k=5
    5) Sycamore: fisher_information_analysis --quick --alphabet-size 9
    6) Sycamore: embedding + fit for k=9

  Full (--full): production runs (many hours to days each — run overnight)
    - fisher_information_analysis --alphabet-size 5|9 (28 readouts)
    - run_llm_entropy_grammar_fisher (12 files, full grid, 50 epochs)

Usage:
  python code/run_all_embedding_tests.py
  python code/run_all_embedding_tests.py --full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")
LLM_RUNS = os.path.join(RESULTS, "llm_entropy", "fisher_runs")
PY = sys.executable


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}\n{desc}\n{'='*60}", flush=True)
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sequential LLM + Sycamore embedding tests")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Full Sycamore (28 files) and full LLM (12 files); days of CPU time.",
    )
    ap.add_argument(
        "--skip-sycamore",
        action="store_true",
        help="Only LLM part (useful if readout_raw_data missing).",
    )
    ap.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only Sycamore alphabet k=5,9 part.",
    )
    args = ap.parse_args()

    os.makedirs(LLM_RUNS, exist_ok=True)
    tag = "full" if args.full else "smoke"
    llm_npz = os.path.join(LLM_RUNS, f"transition_matrices_llm_{tag}.npz")
    llm_emb = os.path.join(LLM_RUNS, f"fisher_ricci_embedding_llm_{tag}_k7.csv")
    llm_fit = os.path.join(LLM_RUNS, f"fisher_speed_foti_scale_fit_llm_{tag}_k7.csv")

    if not args.skip_llm:
        llm_args = [os.path.join(SCRIPT_DIR, "run_llm_entropy_grammar_fisher.py")]
        if not args.full:
            llm_args.append("--smoke")
        llm_args.extend(
            [
                "--transition-npz",
                llm_npz,
                "--summary-csv",
                f"llm_entropy_fisher_summary_{tag}.csv",
            ]
        )
        run([PY] + llm_args, "1/6 LLM Fisher + save T(N) npz")

        run(
            [
                PY,
                os.path.join(SCRIPT_DIR, "compute_fisher_ricci_curve.py"),
                "--skip-ricci",
                "--npz",
                llm_npz,
                "--out",
                llm_emb,
            ],
            "2/6 LLM Fisher speed + log det G",
        )
        run(
            [
                PY,
                os.path.join(SCRIPT_DIR, "fit_fisher_speed_foti_scale.py"),
                "--embedding-csv",
                llm_emb,
                "--include-non-measured",
                "--alphabet-k",
                "7",
                "--out",
                llm_fit,
            ],
            "3/6 LLM A*tau fit (k=7)",
        )

    if args.skip_sycamore:
        print("Skipping Sycamore (--skip-sycamore). Done.")
        return

    for k in (5, 9):
        # Quick mode writes transition_matrices_sycamore_alphabetK.npz; full uses _all_readouts_alphabetK.
        suf = f"_all_readouts_alphabet{k}" if args.full else f"_alphabet{k}"
        if args.full:
            fa = [
                os.path.join(SCRIPT_DIR, "fisher_information_analysis.py"),
                "--alphabet-size",
                str(k),
            ]
            desc = f"Sycamore full Fisher K={k} (28 readouts, many hours)"
        else:
            fa = [
                os.path.join(SCRIPT_DIR, "fisher_information_analysis.py"),
                "--quick",
                "--alphabet-size",
                str(k),
            ]
            desc = f"Sycamore quick Fisher K={k} (3 readouts)"

        run([PY] + fa, desc)

        npz = os.path.join(RESULTS, f"transition_matrices_sycamore{suf}.npz")
        emb = os.path.join(RESULTS, f"fisher_ricci_and_embedding_vs_n_alphabet{k}.csv")
        thr = os.path.join(RESULTS, f"fisher_estimated_thresholds_per_readout{suf}.csv")
        fit_out = os.path.join(RESULTS, f"fisher_speed_foti_scale_fit_alphabet{k}.csv")

        run(
            [
                PY,
                os.path.join(SCRIPT_DIR, "compute_fisher_ricci_curve.py"),
                "--skip-ricci",
                "--npz",
                npz,
                "--out",
                emb,
            ],
            f"Embedding curve K={k}",
        )
        run(
            [
                PY,
                os.path.join(SCRIPT_DIR, "fit_fisher_speed_foti_scale.py"),
                "--embedding-csv",
                emb,
                "--thresholds-csv",
                thr,
                "--alphabet-k",
                str(k),
                "--out",
                fit_out,
            ],
            f"A*tau fit K={k}",
        )

    print("\nAll requested steps finished.")


if __name__ == "__main__":
    main()
