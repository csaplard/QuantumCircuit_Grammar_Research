"""
After ``fisher_information_analysis.py`` has written ``transition_matrices_sycamore*.npz``:

  python code/run_measured_embedding_pipeline.py

Runs ``compute_fisher_ricci_curve.py --skip-ricci`` then ``plot_fisher_embedding_curves.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main() -> None:
    py = sys.executable
    subprocess.check_call(
        [py, os.path.join(SCRIPT_DIR, "compute_fisher_ricci_curve.py"), "--skip-ricci"],
        cwd=REPO_ROOT,
    )
    subprocess.check_call(
        [py, os.path.join(SCRIPT_DIR, "plot_fisher_embedding_curves.py")],
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
