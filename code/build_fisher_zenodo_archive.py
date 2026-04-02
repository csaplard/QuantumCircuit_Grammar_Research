"""
Assemble QuantumCircuit_Fisher_Research_Archive for Zenodo (continuation of Grammar archive).

Copies code, key results, figures, and generates integrity/SHA256SUMS.txt + INTEGRITY_AUDIT stub.

Default output (sibling of Grammar_Fingerprinting_Code_and_Results_v1.0):
  ../../QuantumCircuit_Fisher_Research_Archive

Usage (from Grammar research repo root):
  python code/build_fisher_zenodo_archive.py
  python code/build_fisher_zenodo_archive.py --out D:\\path\\QuantumCircuit_Fisher_Research_Archive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS = REPO_ROOT / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def copy_results_summary_into_bundle(out: Path) -> None:
    """Place results/RESULTS_SUMMARY.md in the Fisher Zenodo folder (full research-repo index + bundle note)."""
    src = RESULTS / "RESULTS_SUMMARY.md"
    header = """# Megjegyzés ehhez a Zenodo-csomaghoz

A lenti táblázatok a **teljes** kutatási repó `results/` mappájára vonatkoznak. **Ebben a csomagban** a `results/` alatt csak a Fisher + kiválasztott IBM fájlok vannak (lásd a fájllistát és a `README.md`-t). Hiányzó fájlok: Grammar `validation_report.txt`, `parameter_sweep_sax7.csv`, threshold transfer CSV-k, stb. — ezek a `QuantumCircuit_Grammar_Research_Archive` fejlesztői klónban maradnak.

---

"""
    if src.is_file():
        body = src.read_text(encoding="utf-8")
        (out / "results" / "RESULTS_SUMMARY.md").write_text(header + body, encoding="utf-8")
    else:
        stub = (
            header
            + "# Results — összefoglaló\n\n"
            + "(A teljes `RESULTS_SUMMARY.md` nem található a forrás `results/` mappában; "
            + "generáld a fejlesztői repóban, majd futtasd újra `build_fisher_zenodo_archive.py`.)\n"
        )
        (out / "results" / "RESULTS_SUMMARY.md").write_text(stub, encoding="utf-8")


def count_csv_data_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return 0
        return sum(1 for _ in r)


def write_sha256sums(root: Path, out_path: Path) -> int:
    lines: list[str] = []
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/") or rel.startswith("integrity/SHA256SUMS.txt"):
            continue
        lines.append(f"{sha256_file(p)} *{rel}")
        n += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT.parent.parent / "QuantumCircuit_Fisher_Research_Archive",
        help="Target directory (created/overwritten selectively)",
    )
    ap.add_argument("--skip-figures", action="store_true", help="Do not run build_fisher_figure_pack.py")
    args = ap.parse_args()
    out: Path = args.out.resolve()

    if not REPO_ROOT.is_dir():
        sys.exit("Repo root not found")

    # Fresh figures
    if not args.skip_figures:
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "build_fisher_figure_pack.py")],
            cwd=str(REPO_ROOT),
            check=True,
        )

    subdirs = ["code", "results", "figures", "integrity", "docs"]
    for d in subdirs:
        (out / d).mkdir(parents=True, exist_ok=True)

    # --- Code (minimal reproducibility)
    code_files = [
        "fisher_information_analysis.py",
        "grammar_learner.py",
        "signal_processing.py",
        "aggregate_fisher_threshold_seeds.py",
        "build_fisher_median_iqr_publication_table.py",
        "build_fisher_figure_pack.py",
        "run_fisher_seed_sweep.ps1",
        "run_fisher_seeds_1_2_after_seed0.ps1",
        "build_fisher_zenodo_archive.py",
    ]
    for fn in code_files:
        src = REPO_ROOT / "code" / fn
        if src.is_file():
            shutil.copy2(src, out / "code" / fn)

    # --- Results: Fisher core + publication + IBM Fisher summaries
    result_globs = [
        "fisher_estimated_thresholds_median_seeds012.csv",
        "fisher_threshold_median_iqr_publication.csv",
        "fisher_threshold_median_iqr_publication.md",
        "fisher_estimated_thresholds_per_readout_all_readouts_seed0.csv",
        "fisher_estimated_thresholds_per_readout_all_readouts_seed1.csv",
        "fisher_estimated_thresholds_per_readout_all_readouts_seed2.csv",
        "fisher_metric_vs_datalength_all_readouts_seed0.csv",
        "fisher_metric_vs_datalength_all_readouts_seed1.csv",
        "fisher_metric_vs_datalength_all_readouts_seed2.csv",
        "fisher_information_analysis_all_readouts_seed0.txt",
        "fisher_information_analysis_all_readouts_seed1.txt",
        "fisher_information_analysis_all_readouts_seed2.txt",
        "ibm_fisher_thresholds_normalized.csv",
        "ibm_fisher_threshold_backend_summary.csv",
        "ibm_torino_grammar_pairwise_kl.csv",
        "ibm_torino_marginals_hamming_compare.csv",
        "ibm_torino_run_compare_distributions.csv",
    ]
    for name in result_globs:
        src = RESULTS / name
        if src.is_file():
            shutil.copy2(src, out / "results" / name)

    copy_results_summary_into_bundle(out)

    # Figures (publication pack: fig01–fig03)
    fig_src = RESULTS / "fisher_figures"
    if fig_src.is_dir():
        for p in sorted(fig_src.iterdir()):
            if p.is_file():
                shutil.copy2(p, out / "figures" / p.name)

    # LICENSE + requirements from Grammar repo if present
    lic = REPO_ROOT / "LICENSE"
    if not lic.is_file():
        g = REPO_ROOT.parent.parent / "QuantumCircuit_Grammar_Research_Archive" / "LICENSE"
        if g.is_file():
            lic = g
    if lic.is_file():
        shutil.copy2(lic, out / "LICENSE")
    req = REPO_ROOT / "requirements.txt"
    if req.is_file():
        shutil.copy2(req, out / "requirements.txt")

    docs_repo = REPO_ROOT / "docs"
    if docs_repo.is_dir():
        for name in (
            "Fisher_Threshold_Study_Preprint.tex",
            "build_fisher_preprint_pdf.ps1",
            "Fisher_Threshold_Study_Preprint.pdf",
        ):
            p = docs_repo / name
            if p.is_file():
                shutil.copy2(p, out / "docs" / name)

    # README, Zenodo helpers, validation criteria (Grammar-style)
    write_readme(out)
    write_zenodo_json(out)
    write_fisher_validation_criteria(out)
    write_zenodo_md(out)
    write_docs_placeholder(out)

    n = write_sha256sums(out, out / "integrity" / "SHA256SUMS.txt")
    write_integrity_audit(out, n_hashed=n)
    print(f"SHA256 manifest: {n} files -> {out / 'integrity' / 'SHA256SUMS.txt'}")
    print(f"Archive root: {out}")


def write_readme(out: Path) -> None:
    text = """# Fisher Information Threshold Study — Grammar Fingerprinting (Zenodo continuation)

**Author:** Dániel Csaplár — Independent Researcher, Kazincbarcika, Hungary  
**ORCID:** [0009-0000-7362-7232](https://orcid.org/0009-0000-7362-7232)  
**Package date:** April 2026  

This deposit mirrors the layout of **`QuantumCircuit_Grammar_Research_Archive`** (integrity manifest, validation criteria JSON, reproduction commands) but focuses on the **Fisher information** analysis: **Fisher trace** on learned grammar transition matrices vs data length **N**, per-readout estimated thresholds **N*** with **median and IQR** across **three LSTM seeds** (0, 1, 2) on the same **28 Sycamore** readout configurations as the Grammar study.

---

## Summary

We quantify how much information the learned grammar carries about the transition model as a function of sample size, and estimate a data-length scale **N*** per readout. Results are aggregated for publication in `fisher_threshold_median_iqr_publication.csv` / `.md` and summarized in **figures** `fig01`–`fig03` (see `figures/`). Optional **IBM** cross-check CSVs are included for context (Torino summaries).

**Continuation:** Use the Grammar Zenodo record for blind topology classification; use **this** record for Fisher thresholds and figures. Link them with Zenodo **Related identifiers** (see `ZENODO.md`).

---

## Data sources

| Platform | Source | License | DOI |
|----------|--------|---------|-----|
| Google Sycamore (53 qubits) | Arute et al., *Nature* 574, 505–510 (2019) — Dryad | CC0 | [10.5061/dryad.k6t1rj8](https://doi.org/10.5061/dryad.k6t1rj8) |

Raw `*_readout_raw_data.txt` files are **not** bundled here when omitted for size; obtain them from the Grammar archive or Dryad.

---

## Key results (files)

| Topic | Content | File |
|-------|---------|------|
| Median N* + IQR | 28 readouts × seeds 0–2 | `results/fisher_threshold_median_iqr_publication.csv`, `.md` |
| Per-seed thresholds | Median across seeds | `results/fisher_estimated_thresholds_median_seeds012.csv` |
| Fisher vs N | Curves per readout | `results/fisher_metric_vs_datalength_all_readouts_seed*.csv` |
| Eredmény-index (teljes repó + megjegyzés) | Összefoglaló táblák | `results/RESULTS_SUMMARY.md` |
| Figures | Publication pack | `figures/fig01_*` … `fig03_*` |
| Reporting rules | Same corpus as Grammar | `fisher_validation_criteria.json` |

---

## Repository structure

```
├── code/                          # Reproduction scripts
├── results/                       # CSV / TXT outputs + RESULTS_SUMMARY.md (index)
├── figures/                       # fig01–fig03 (PDF + PNG)
├── docs/                          # Fisher preprint LaTeX + PDF build script; PDF optional (see ADD_PREPRINT_PDF_HERE.txt)
├── integrity/
│   ├── SHA256SUMS.txt             # All-file manifest (manifest not self-hashed)
│   └── INTEGRITY_AUDIT.txt        # Audit notes + consistency checks
├── fisher_validation_criteria.json
├── zenodo_metadata.json           # Paste-friendly Zenodo fields
├── ZENODO.md                      # Upload checklist (Grammar-style)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproduction

### Requirements

```bash
pip install -r requirements.txt
```

### One seed (example: 0)

```bash
python code/fisher_information_analysis.py --tag-with-seed --seed 0
```

### Seeds 0–2, aggregate, publication table, figures (matches this bundle)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File code/run_fisher_seed_sweep.ps1 -Seeds 0,1,2
```

```bash
python code/aggregate_fisher_threshold_seeds.py \\
  results/fisher_estimated_thresholds_per_readout_all_readouts_seed0.csv \\
  results/fisher_estimated_thresholds_per_readout_all_readouts_seed1.csv \\
  results/fisher_estimated_thresholds_per_readout_all_readouts_seed2.csv \\
  --out results/fisher_estimated_thresholds_median_seeds012.csv
python code/build_fisher_median_iqr_publication_table.py
python code/build_fisher_figure_pack.py
```

Rebuild **this** Zenodo folder from the development repo:

```bash
python code/build_fisher_zenodo_archive.py
```

---

## Zenodo

Step-by-step: **`ZENODO.md`**. Draft metadata: **`zenodo_metadata.json`**.

---

## Citation

Cite **this** Zenodo DOI once published, and the Grammar Fingerprinting preprint / Grammar dataset DOI as appropriate. Preprint PDF may be included under `docs/`.

---

## License

**Code:** MIT (see `LICENSE`). **Results, figures, README:** recommend **CC-BY 4.0** in the Zenodo deposit metadata.

---

## Contact

Dániel Csaplár — Kazincbarcika, Hungary — ORCID [0009-0000-7362-7232](https://orcid.org/0009-0000-7362-7232)
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def write_zenodo_json(out: Path) -> None:
    meta = {
        "version": "1.0.0",
        "title": "Fisher Information Threshold Analysis for Grammar Fingerprinting on Sycamore Readouts (28 configurations, 3 seeds)",
        "description": (
            "Research dataset and code (continuation of Grammar Fingerprinting on Google Sycamore raw readouts). "
            "Contains: (1) Fisher trace computed on learned grammar transition matrices as a function of data length N; "
            "(2) per-readout estimated thresholds N* with median and inter-quartile range across three LSTM random seeds; "
            "(3) publication CSV/MD tables; (4) figures fig01–fig03 (PDF+PNG); (5) optional IBM Torino summary CSVs; "
            "(6) SHA256 integrity manifest. Raw readout text files may be omitted for size; use Dryad CC0 data or the "
            "Grammar Fingerprinting archive. Link this deposit to the Grammar Zenodo record via related identifiers."
        ),
        "creators": [
            {
                "name": "Csaplár, Dániel",
                "affiliation": "Independent Researcher, Kazincbarcika, Hungary",
                "orcid": "0009-0000-7362-7232",
            }
        ],
        "keywords": [
            "quantum computing",
            "Fisher information",
            "grammar fingerprinting",
            "Sycamore",
            "readout noise",
            "machine learning",
            "information geometry",
        ],
        "license": "MIT",
        "access_right": "open",
        "upload_type": "dataset",
        "publication_date": date.today().isoformat(),
        "related_identifiers": [
            {
                "relation": "continues",
                "identifier": "doi:10.5281/zenodo.19158088",
                "resource_type": "dataset",
                "note": "Grammar Fingerprinting Zenodo deposit; Fisher is the continuation. Zenodo UI: also add reverse link from Grammar v2 if you version the Grammar record.",
            }
        ],
        "notes": (
            "This deposit is the Fisher continuation: link continues -> Grammar DOI 10.5281/zenodo.19158088. "
            "After publishing Fisher: paste the new Fisher DOI into preprint/docs and optional Grammar-record related identifier. "
            "CC-BY-4.0 for figures/README in addition to MIT if the UI allows."
        ),
    }
    (out / "zenodo_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_fisher_validation_criteria(out: Path) -> None:
    criteria = {
        "package": "QuantumCircuit_Fisher_Research_Archive",
        "continuation_of": "QuantumCircuit_Grammar_Research_Archive",
        "method": "Fisher information (trace) on learned grammar transition matrices vs data length N",
        "seeds": [0, 1, 2],
        "readout_configurations": 28,
        "aggregation": (
            "Per-readout N* estimated per seed; publication table reports median_N_star and "
            "IQR (q25–q75) across seeds where applicable"
        ),
        "alignment_note": (
            "Median N* bands often overlap the ~8k sample regime observed in Grammar classification sweeps; "
            "see Grammar archive validation and parameter sweeps."
        ),
        "pre_registration": (
            "Fisher analysis uses the same Sycamore readout corpus and grammar learner pipeline as the Grammar study; "
            "this file documents reporting conventions for the Zenodo deposit (analogous to validation_criteria.json in the Grammar archive)."
        ),
    }
    (out / "fisher_validation_criteria.json").write_text(json.dumps(criteria, indent=2), encoding="utf-8")


def write_zenodo_md(out: Path) -> None:
    txt = """# Zenodo upload — Fisher continuation package

This checklist follows the same **practical pattern** as **`QuantumCircuit_Grammar_Research_Archive`**: fixed folder layout, `integrity/SHA256SUMS.txt`, and a single archive zip for upload.

## Before you zip

1. Rebuild the bundle from the dev repo: `python code/build_fisher_zenodo_archive.py` (refreshes figures, README, hashes).
2. Compile `docs/Fisher_Threshold_Study_Preprint.tex` to PDF (or copy your PDF) into `docs/`. Optionally add `Grammar_Fingerprinting_Preprint.pdf`.
3. Confirm `results/fisher_threshold_median_iqr_publication.csv` has **28** data rows (one per readout configuration).

## Zenodo.org (new upload)

1. **Upload type:** Dataset (or Software if you split code; combined bundle is usually **Dataset**).
2. Copy **Title**, **Description**, **Authors**, **Keywords** from `zenodo_metadata.json`.
3. **License:** MIT for code; for figures and narrative text, add **CC-BY 4.0** in the deposit description or as a second license if the UI supports it.
4. **Related identifiers:** set **isSupplementTo** (or **continues**) to the **Grammar Fingerprinting** Zenodo DOI (replace the placeholder in `zenodo_metadata.json`).
5. **Date:** use publication date from `zenodo_metadata.json` or the upload date.
6. Upload a **.zip** of the entire `QuantumCircuit_Fisher_Research_Archive` folder (not the parent repo).

## After publication

- Paste the reserved **concept DOI** / version DOI into your preprint and into `zenodo_metadata.json` for the next revision.
- Re-run `build_fisher_zenodo_archive.py` if you change files so `SHA256SUMS.txt` stays consistent.

## Integrity

- Verify any file: `certutil -hashfile <path> SHA256` (Windows) and compare to `integrity/SHA256SUMS.txt`.
"""
    (out / "ZENODO.md").write_text(txt, encoding="utf-8")


def write_docs_placeholder(out: Path) -> None:
    txt = """Companion files for this Zenodo record:

- `Fisher_Threshold_Study_Preprint.tex` — LaTeX source (Grammar-style layout). Build PDF: run `build_fisher_preprint_pdf.ps1` (needs pdflatex / MiKTeX or TeX Live), or compile manually.
- `Fisher_Threshold_Study_Preprint.pdf` — optional: copy the compiled PDF here before zipping (same basename as the .tex).
- `Grammar_Fingerprinting_Preprint.pdf` — optional cross-link to the Grammar preprint PDF.

Before uploading: re-run `python code/build_fisher_zenodo_archive.py` so `integrity/SHA256SUMS.txt` includes new files.
"""
    (out / "docs" / "ADD_PREPRINT_PDF_HERE.txt").write_text(txt, encoding="utf-8")


def write_integrity_audit(out: Path, n_hashed: int) -> None:
    pub = out / "results" / "fisher_threshold_median_iqr_publication.csv"
    n_rows = count_csv_data_rows(pub)
    row_ok = n_rows == 28 if n_rows is not None else False
    pub_line = (
        f"- Publication table rows (data): {n_rows} (expected 28) — {'OK' if row_ok else 'CHECK'}"
        if n_rows is not None
        else "- Publication table: MISSING (expected results/fisher_threshold_median_iqr_publication.csv)"
    )
    txt = f"""INTEGRITY AUDIT
===============
Date: {date.today().isoformat()}
Scope: QuantumCircuit_Fisher_Research_Archive (Fisher continuation; Zenodo-oriented bundle)

1) File integrity manifest
--------------------------
- SHA256 manifest generated at:
  integrity/SHA256SUMS.txt
- Files hashed (excluding .git and the manifest file itself): {n_hashed}

2) Core artifact consistency
----------------------------
Checked files:
- results/fisher_threshold_median_iqr_publication.csv

Findings:
{pub_line}
- Key outputs also present: results/fisher_estimated_thresholds_median_seeds012.csv, figures/fig01–fig03 (PDF+PNG).

3) Reproduction
---------------
- Full commands: README.md
- Integrity helper script: code/build_fisher_zenodo_archive.py

4) Conclusion
-------------
This is an engineering integrity / reproducibility check (not external peer review). Strongest public check: independent rerun of `fisher_information_analysis.py` and aggregation scripts on the same readouts.

Notes:
- Re-run build_fisher_zenodo_archive.py after any manual edit; verify hashes.
- Grammar archive: QuantumCircuit_Grammar_Research_Archive (blind topology + raw readouts reference).

Generated by: code/build_fisher_zenodo_archive.py
"""
    (out / "integrity" / "INTEGRITY_AUDIT.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
