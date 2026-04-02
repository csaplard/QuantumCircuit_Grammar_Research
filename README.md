# Grammar Fingerprinting Research Archive

**Blind topology classification, Fisher-threshold analysis, IBM cross-platform data, and transfer modelling from raw quantum readouts.**

**Author:** Dániel Csaplár — Independent Researcher, Kazincbarcika, Hungary  
**ORCID:** [0009-0000-7362-7232](https://orcid.org/0009-0000-7362-7232)  
**Archive date:** March 2026  

---

## What this repository contains

1. **Sycamore (Dryad):** Grammar fingerprints from raw readout sequences; blind Ward clustering vs. hardware topology; strong controls (shuffle, LSB, baselines); **data-length threshold near ~8k samples** for classification signal.
2. **Fisher information geometry:** Per-readout sweeps over data length; scalar Fisher metrics and **estimated transition thresholds** (N*); **multi-seed** runs with **median + IQR** reporting.
3. **IBM Quantum:** Raw shot collection (`collect_ibm_shots.py`), grammar learning, IBM-specific Fisher sweeps, optional Marrakesh vs. Torino comparisons.
4. **Temporal drift (IBM):** Distribution-level and **grammar-matrix KL** comparisons between archived and refreshed shot files; marginal / Hamming summaries for high-q regimes.
5. **Transfer / universality:** Ridge-style models linking IBM and Sycamore threshold estimates; clustering / purity reports; optional DOCX report builders.

**Not tracked in Git** (see `.gitignore`): Dryad-scale `readout_raw_data/`; IBM `results/ibm_raw_shots/`; run logs; large `*.npz` caches; exploratory IBM subset analyses (`10_20`, `no5q`, `raw4` filename patterns); extra Fisher phase-transition PNGs (publication figures remain under `results/fisher_figures/`).

---

## Data sources

| Platform | Role | Notes |
|----------|------|--------|
| **Google Sycamore** | Primary readouts for validation & Fisher | Arute et al., *Nature* 574, 505–510 (2019) — Dryad **CC0**, [10.5061/dryad.k6t1rj8](https://doi.org/10.5061/dryad.k6t1rj8). Files: `readout_raw_data/*_readout_raw_data.txt` or `results/readout_raw_data/`. |
| **IBM Quantum** | Cross-platform shots & thresholds | Collected via `qiskit-ibm-runtime` (not in minimal `requirements.txt`). Use **`QISKIT_IBM_TOKEN`** (see `code/local_ibm_env.ps1.example` then copy to `local_ibm_env.ps1`, gitignored). |

---

## Core pipeline (all platforms)

1. Load integer **readout / shot** column (`output`).
2. Z-normalize then **SAX** (default alphabet 7, quantile breakpoints; `signal_processing.py`).
3. Train **character LSTM** (`grammar_learner.py`: hidden, seq_len, epochs as per script).
4. **Extract grammar:** row-normalized transition matrix T (next SAX symbol given current).
5. Downstream: clustering (Frobenius + Ward), Fisher metric on T, KL between successive N, or comparison across runs.

Ground-truth topology labels are used **only for evaluation**, not for training.

---

## Sycamore: blind topology validation (summary)

- **Claim:** Temporal structure in raw noise encodes **1D Snake / 2D Block / Bulk** topology; **~84.5%** mean Ward purity (multiple seeds) vs. chance and strong controls.
- **Controls:** Shuffled time order to chance; LSB-only bitstream to chance; static SAX + LR/RF to chance; short training / low max_pts to no signal.
- **Data-length sweep:** Sharp emergence of signal around **~8,000** samples (see `results/parameter_sweep_sax7.csv`, `validation_report.txt`).

**Main driver:** `code/run_validation_pipeline.py`  
**Criteria:** `validation_criteria.json`

---

## Fisher information analysis (Sycamore, 28 readouts)

**Script:** `code/fisher_information_analysis.py`

- Trains the grammar learner at fixed **N** grid: 500 to 40000 (see script for exact steps).
- Builds Fisher information matrix from T; traces Fisher trace / det / max eigenvalue, KL between successive T(N), entropy, Frobenius distance from uniform.
- Estimates per-file **N*** (phase-transition-style heuristic; see script and `fisher_information_analysis_*.txt` reports).
- **Multi-seed:** `--tag-with-seed --seed 0|1|2` gives output suffix `_seed{k}`.
- **Queue helper:** `code/run_fisher_seeds_1_2_after_seed0.ps1` waits for seed 0, then runs 1 and 2.
- **Aggregate:** `code/aggregate_fisher_threshold_seeds.py` gives **median + q25/q75** of N* across seeds.
- **Publication table:** `code/build_fisher_median_iqr_publication_table.py` writes `results/fisher_threshold_median_iqr_publication.csv` and `.md`.

**Typical outputs:**  
`results/fisher_metric_vs_datalength_all_readouts_seed*.csv`,  
`results/fisher_estimated_thresholds_per_readout_all_readouts_seed*.csv`,  
`results/fisher_phase_transition_all_readouts_seed*.png`,  
`results/fisher_estimated_thresholds_median_seeds012.csv`.

**Quick test:** `python code/fisher_information_analysis.py --quick`

---

## IBM Quantum workflows

| Script | Purpose |
|--------|---------|
| `code/collect_ibm_shots.py` | SamplerV2 jobs; saves `results/ibm_raw_shots/ibm_<backend>_*.txt` + metadata. `--resume` skips complete files. |
| `code/run_ibm_grammar_learning.py` | Fingerprints for all shot files to CSV + NPZ. |
| `code/run_ibm_fisher_threshold_sweep.py` | Per-circuit Fisher-style sweep on IBM shots. |
| `code/run_ibm_ghz_vs_hadamardlayers_threshold.py` | GHZ vs Hadamard/layers threshold comparison. |
| `code/run_ibm_sycamore_protocol.py` | Protocol-style report from IBM filenames. |
| `code/analyze_ibm_fingerprint_clustering.py` | IBM fingerprint clustering analyses. |

**Drift / two-run comparison (same backend, two dates):**

| Script | Purpose |
|--------|---------|
| `code/compare_ibm_shot_runs.py` | Raw outcome histogram: TV, Jensen-Shannon, sym-KL (sparse at high q). |
| `code/compare_ibm_shot_marginals.py` | Per-qubit P(1) drift + Hamming-weight TV/JS (interpretable at 20q). |
| `code/compare_ibm_grammar_pairwise_kl.py` | Train grammar on archive vs current; KL between T matrices. |

---

## Cross-platform and transfer modelling

- `code/cross_platform_universality_report.py` — combined IBM + Sycamore fingerprint summaries (Ward purity, regime tables).
- `code/fit_threshold_transfer_model.py` — OLS/Ridge-style transfer of threshold estimates across sources.
- `code/fisher_robustness_visualization.py` — Boxplots / curves for Fisher robustness.
- `code/drift_monitor_kl.py` — KL vs a reference grammar matrix from NPZ + rolling windows (prototype).
- `code/build_ibm_threshold_report_docx.py`, `code/build_transfer_threshold_docx.py` — Word reports from CSVs (optional python-docx).

---

## Key results (headline numbers; full tables in results/)

The README lists **summary numbers** only. **Authoritative rows** are in `results/*.csv` and `results/*.txt`.

*(Magyarul: a README-ben **fő számok** és **fájlnevek** vannak; a **teljes táblázatok** a `results/` mappában — nem másoljuk be ide mind a 28 readout sort.)*

### Sycamore blind topology

| Result | Value / note | File |
|--------|----------------|------|
| Mean Ward purity (3 seeds) | **~84.5%** | `results/robustness_audit_sax7.txt`, `results/validation_report.txt` |
| Shuffled control | **50%** | `results/shuffled_control_sax7.txt` |
| LSB single-bit control | **50%** | `results/readout_lsb_bit_control_sax7.txt` |
| Data-length sweep | Signal near **8k** (e.g. **92.86%** at max_pts 8000 in 60-combo sweep) | `results/parameter_sweep_sax7.csv` |
| Pre-registered criteria | **PASS** | `validation_criteria.json`, `results/validation_report.txt` |

### Fisher (28 readouts, seeds 0, 1, 2)

| Result | Note | File |
|--------|------|------|
| Median N* + IQR | Many readouts in **~6k-10k** band (aligned with ~8k regime); spread by readout | `results/fisher_threshold_median_iqr_publication.csv`, `.md` |
| Stable example | **46q Bulk:** N* = **750** all three seeds | same (`stable_3seeds` column) |
| Seed-sensitive | Large IQR e.g. 28q, 30q, 32q, 36q, 53q | `results/fisher_estimated_thresholds_median_seeds012.csv` |
| Curves | Fisher vs data length | `results/fisher_metric_vs_datalength_all_readouts_seed*.csv`, `results/fisher_phase_transition_all_readouts_seed*.png` |

### IBM Torino (archive vs refreshed shots, example)

| Result | Representative value | File |
|--------|---------------------|------|
| Grammar sym-KL (11 circuits) | Mean **~0.008**; mean Frobenius **~0.13** | `results/ibm_torino_grammar_pairwise_kl.csv` |
| 20q Hamming-weight TV | Mean **~0.031** | `results/ibm_torino_marginals_hamming_compare.csv` |
| Raw joint histogram | Inflated TV/KL at 20q; use marginals + grammar | `results/ibm_torino_run_compare_distributions.csv` |

### IBM Fisher and transfer

| Topic | File |
|-------|------|
| Backend sweeps | `results/ibm_fisher_sweep_marrakesh40960.csv`, `results/ibm_fisher_sweep_torino40960.csv` |
| Normalized thresholds | `results/ibm_fisher_thresholds_normalized.csv`, `results/ibm_fisher_threshold_backend_summary.csv` |
| Transfer model | `results/threshold_transfer_model_report.txt`, `results/threshold_transfer_predictions.csv` |

---

## Repository layout (high level)

```
code/                  # All Python tooling
readout_raw_data/      # Preferred: Sycamore *_readout_raw_data.txt
results/               # All outputs: validation, Fisher, IBM, plots, DOCX, logs
  ibm_raw_shots/       # IBM shots; archive_* folders for drift baselines
validation_criteria.json
requirements.txt
code/local_ibm_env.ps1.example
README.md
```

**Other `code/` utilities:** `run_validation_pipeline.py`, `run_readout_lsb_bit_experiment.py`, `build_sycamore_readout_fingerprints.py`, `aggregate_fisher_threshold_seeds.py`, `test_threshold_transfer_cv.py`, ...

---

## Reproduction (cheat sheet)

```bash
pip install -r requirements.txt
# IBM: pip install qiskit-ibm-runtime qiskit  (and account / token)
```

**Sycamore validation**

```bash
python code/run_validation_pipeline.py --tasks sweep,report
python code/run_validation_pipeline.py --tasks shuffled
python code/run_readout_lsb_bit_experiment.py --output-bit-index 0 --epochs 50 --max-pts 10000 --seeds 0 123 999
```

**Fisher (full 28 files, one seed)**

```bash
python code/fisher_information_analysis.py --tag-with-seed --seed 0
```

**Fisher seeds 0-2 + median table**

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File code/run_fisher_seed_sweep.ps1 -Seeds 0,1,2
python code/aggregate_fisher_threshold_seeds.py results/fisher_estimated_thresholds_per_readout_all_readouts_seed0.csv results/fisher_estimated_thresholds_per_readout_all_readouts_seed1.csv results/fisher_estimated_thresholds_per_readout_all_readouts_seed2.csv --out results/fisher_estimated_thresholds_median_seeds012.csv
python code/build_fisher_median_iqr_publication_table.py
```

**IBM shots (after local_ibm_env.ps1 or env)**

```bash
. ./code/local_ibm_env.ps1   # PowerShell
python code/collect_ibm_shots.py --backend ibm_torino --resume
python code/run_ibm_grammar_learning.py
```

---

## Negative / null results (reported)

- **IBM daily calibration summaries:** No above-chance grammar topology signal from pre-aggregated calibration tables; dense raw shots are needed.
- **Cosmological parameter encoding (Beck hypothesis):** Tested; not supported.

---

## Method note

Grammar Fingerprinting adapts industrial **condition monitoring** (SAX + sequence learning) to quantum readout streams. Development used AI-assisted coding (e.g. Cursor); **design, runs, and interpretation** remain the author's responsibility.

---

## License

**Code:** MIT  
**Results / documentation:** CC-BY 4.0  

---

## Contact

Dániel Csaplár  
Kazincbarcika, Hungary  
ORCID: [0009-0000-7362-7232](https://orcid.org/0009-0000-7362-7232)
