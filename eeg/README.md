# EEG Grammar Fingerprinting

Cross-substrate sequential-grammar pipeline applied to human EEG during
mental arithmetic. Companion to the prior Sycamore and LLM Grammar
Fingerprinting work in this repository.

**Preprint**: `paper/EEG_Grammar_Fingerprinting.md` (markdown source) and
`paper/EEG_Grammar_Fingerprinting.docx` (compiled). Self-archived on Zenodo.

## Contents

```
eeg/
├── code/                   # Pipeline source
│   ├── eeg_io.py           # EDF loading + MNE preprocessing (filter, notch, resample)
│   ├── controls.py         # Surrogate generators (shuffled, AAFT, random_uniform)
│   ├── run_eeg_grammar.py  # Main runner; iterates subjects x regimes x channels
│   ├── analyze_results.py  # Per-channel + global paired tests, LOSO CV
│   ├── compare_real_aaft.py# Real vs AAFT regime-specific comparison
│   ├── make_paper_figures.py # Figures 1, 2, 3, 4
│   ├── make_aaft_figure.py # Figure 5 (AAFT specifically)
│   ├── parse_log.py        # Utility: parse perplexity values from runner logs
│   └── test_pipeline_synthetic.py # End-to-end sanity check on synthetic signal
├── results/
│   ├── full_real/          # 36 subjects x 2 regimes x 19 channels (real signal)
│   │   ├── eeg_grammar_K7.csv         # Per-stream perplexity + metadata
│   │   ├── eeg_grammar_K7_meta.json   # Pipeline parameters
│   │   └── eeg_fingerprints_K7.npz    # All 7x7 fingerprint matrices
│   ├── full_aaft/          # Same shape, AAFT surrogate of each stream
│   └── full_combined.csv   # Merged real+AAFT for direct comparison
└── paper/
    ├── EEG_Grammar_Fingerprinting.md   # Preprint markdown source
    ├── EEG_Grammar_Fingerprinting.docx # Compiled Word document
    ├── zenodo_metadata.json            # Zenodo upload metadata template
    ├── generate_paper_docx.py          # md -> docx compiler
    └── figures/                        # All 5 preprint figures (PNG)
```

## Dataset

PhysioNet eegmat 1.0.0 (Zyma et al. 2019, doi:10.13026/C2JQ1P).
36 healthy adult volunteers, 19-channel 10-20 montage, 500 Hz, eyes closed.
- `Subject*_1.edf` — 180 s resting baseline
- `Subject*_2.edf` — first 60 s of serial mental arithmetic
- `subject-info.csv` — behavioral metadata (good/bad counter classification)

The dataset is publicly downloadable from
https://physionet.org/content/eegmat/1.0.0/ ; raw EDFs are not redistributed
in this repository.

## Reproducibility

Real signal pipeline (~24 h on a single laptop CPU, 36 subjects):

```bash
python eeg/code/run_eeg_grammar.py \
    --data_dir <path-to-eegmat> \
    --target_fs 100 \
    --lfreq 1 --hfreq 40 --notch 50 \
    --K 7 --hidden_dim 16 --seq_len 20 --epochs 50 --lr 0.01 \
    --seed 42 \
    --out_dir eeg/results/full_real
```

AAFT surrogate pipeline (~24 h on a single laptop CPU):

```bash
python eeg/code/run_eeg_grammar.py \
    [same flags as above] \
    --skip_real --surrogates aaft \
    --out_dir eeg/results/full_aaft
```

Analysis and figures:

```bash
python eeg/code/analyze_results.py \
    --csv eeg/results/full_real/eeg_grammar_K7.csv

python eeg/code/compare_real_aaft.py \
    --real_csv eeg/results/full_real/eeg_grammar_K7.csv \
    --aaft_log <aaft-runner-log>   # OR use the merged CSV directly

python eeg/code/make_paper_figures.py \
    --csv eeg/results/full_real/eeg_grammar_K7.csv \
    --npz eeg/results/full_real/eeg_fingerprints_K7.npz

python eeg/code/make_aaft_figure.py \
    --combined eeg/results/full_combined.csv
```

Sanity check on synthetic data (~1 min, no EDF required):

```bash
python eeg/code/test_pipeline_synthetic.py
```

## Pipeline parameters

All parameters identical to the Sycamore and LLM applications in this
repository — no per-substrate tuning:

| Parameter | Value | Notes |
|---|---|---|
| SAX alphabet K | 7 | Empirical-quantile breakpoints |
| LSTM hidden dim | 16 | Pure-NumPy implementation in `code/grammar_learner.py` |
| LSTM seq_len | 20 | 200 ms context at 100 Hz target_fs |
| Epochs | 50 | Adam optimizer, lr = 0.01 |
| Random seed | 42 | Single deterministic seed for all reported results |
| Band-pass | 1-40 Hz | Standard EEG, FIR zero-phase |
| Notch | 50 Hz | European mains frequency |
| Target sample rate | 100 Hz | Nyquist-protected by band-limit at 40 Hz |

## Headline results (n = 36 subjects, 1,368 streams)

- **Regime discrimination**: Wilcoxon p = 9.14e-83, Cohen's d_paired = 0.99,
  597/684 pairs (87.3%) showed higher perplexity during arithmetic.
- **Topography**: all 19 channels significant at FDR q < 0.01;
  effect-size gradient posterior–parietal–midline (d > 1.4) → frontopolar
  (d ≈ 0.55) — recapitulates classical alpha-suppression topography.
- **Generalization**: leave-one-subject-out logistic regression on flattened
  fingerprint vectors achieves 75.2% ± 10.6% accuracy across 36 folds
  (chance = 50%); 33/36 folds above chance.
- **AAFT surrogate** (probe of nonlinear residue beyond power spectrum):
  rest delta = +0.022, d = 0.12, p = 1.6e-4;
  arithmetic delta = +0.002, d = 0.006, p = 0.17;
  regime × surrogate interaction p = 0.47 (n.s.).
  Discrimination is therefore mediated by linear power-spectral content
  re-represented as sequential symbolic grammar.

## Dependencies

- Python 3.9+
- numpy, scipy, pandas, matplotlib, scikit-learn
- mne >= 1.10 (for EDF I/O and topographic plotting)
- python-docx (only for `generate_paper_docx.py`)

Install with: `pip install numpy scipy pandas matplotlib scikit-learn mne python-docx`

## Citation

If you use this pipeline, please cite the corresponding Zenodo preprint
(DOI assigned at upload) as well as the parent Grammar Fingerprinting
series listed in the main repository README.
