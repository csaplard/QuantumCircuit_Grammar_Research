"""
After ``run_llm_entropy_grammar_fisher.py`` has saved ``--transition-npz`` (same SAX k=7 as Sycamore):

  python code/compute_fisher_ricci_curve.py --skip-ricci \\
    --npz results/llm_entropy/fisher_runs/transition_matrices_llm_seed42.npz \\
    --out results/llm_entropy/fisher_runs/fisher_ricci_embedding_llm_k7.csv

  python code/fit_fisher_speed_foti_scale.py \\
    --embedding-csv results/llm_entropy/fisher_runs/fisher_ricci_embedding_llm_k7.csv \\
    --thresholds-csv <optional N* table if you build one> \\
    --include-non-measured \\
    --alphabet-k 7 \\
    --out results/llm_entropy/fisher_runs/fisher_speed_foti_scale_fit_llm_k7.csv

``--include-non-measured`` is needed if ``data_source`` is absent or not ``measured``.

Usage:
  python code/llm_embedding_speed_pipeline.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CODE = os.path.join(REPO_ROOT, "code")
RUNS = os.path.join(REPO_ROOT, "results", "llm_entropy", "fisher_runs")


def main() -> None:
    npz = os.path.join(RUNS, "transition_matrices_llm_seed42.npz")
    emb = os.path.join(RUNS, "fisher_ricci_embedding_llm_k7.csv")
    fit_out = os.path.join(RUNS, "fisher_speed_foti_scale_fit_llm_k7.csv")
    print("=== LLM k=7: Fisher speed fit (A*tau vs k(k-1)*pi/2) ===\n")
    print(
        f'{PY} {os.path.join(CODE, "compute_fisher_ricci_curve.py")} --skip-ricci '
        f'--npz "{npz}" --out "{emb}"'
    )
    print(
        f'{PY} {os.path.join(CODE, "fit_fisher_speed_foti_scale.py")} '
        f'--embedding-csv "{emb}" --include-non-measured --alphabet-k 7 --out "{fit_out}"'
    )


if __name__ == "__main__":
    main()
