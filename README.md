# Grammar Fingerprinting: Blind Topology Classification of Quantum Processors from Raw Measurement Noise

**Author:** Dániel Csaplár — Independent Researcher, Kazincbarcika, Hungary  
**ORCID:** [0009-0000-7362-7232](https://orcid.org/0009-0000-7362-7232)  
**Date:** March 2026  

---

## Summary

This repository demonstrates that the temporal dynamics of raw quantum processor measurement noise encode the physical hardware topology. Using a method called **Grammar Fingerprinting** — Symbolic Aggregate Approximation (SAX) combined with a character-level LSTM — we extract learned transition matrices ("grammar fingerprints") from raw readout sequences and cluster them via Frobenius distance and Ward linkage.

**Key result:** Blind classification of 28 Google Sycamore quantum supremacy circuit configurations into three topology classes (1D Snake, 2D Block, Full Chip) achieves **84.5% mean accuracy** across three random seeds, while six independent controls confirm the signal is real.

---

## Data Sources

| Platform | Source | License | DOI |
|----------|--------|---------|-----|
| Google Sycamore (53 qubits) | Arute et al., *Nature* 574, 505–510 (2019) — Dryad | CC0 | [10.5061/dryad.k6t1rj8](https://doi.org/10.5061/dryad.k6t1rj8) |

---

## Validation Results

### Primary Result
- **Mean accuracy:** 84.52% (seeds 0, 123, 999)
- **Seed std:** 1.68%

### Control Tests (all PASS)

| Control | Purpose | Result |
|---------|---------|--------|
| **Shuffled control** | Destroy temporal order, preserve distribution | 50.00% (3 seeds, std 0) |
| **LSB single-bit control** | Use only 1 bit per readout (0/1 series) | 50.00% (3 seeds, std 0) |
| **Logistic Regression baseline** | Static SAX features, no sequence | ~36–44% (chance level) |
| **Random Forest baseline** | Static SAX features, no sequence | ~33–39% (chance level) |
| **Quick epochs (10 ep, 5k pts)** | Reduced training | 50.00% (3 seeds, std 0) |

### Parameter Sweep (60 combinations)
Epochs [10, 20, 30, 40, 50] × max_pts [3000, 5000, 8000, 10000] × seeds [0, 123, 999]

| max_pts | Accuracy (all epochs) | Std |
|---------|----------------------|-----|
| 3,000 | 50.00% | 0.00 |
| 5,000 | 50.00% | 0.00 |
| 8,000 | **92.86%** | 0.00 |
| 10,000 | 82–89% | 0–4% |

**Sharp threshold at ~8,000 data points:** below it, the model is blind; above it, classification accuracy jumps to 85–93%. This threshold is independent of training epochs, confirming the signal emerges from data quantity, not training duration.

### Decision Criteria (defined before experiments)
```json
{
  "minimum_accuracy_original": 75.0,
  "maximum_seed_std": 5.0,
  "maximum_shuffled_accuracy": 60.0,
  "minimum_accuracy_difference_vs_shuffled": 20.0
}
```
**VERDICT: SIGNAL IS REAL** (all criteria PASS, accuracy difference = 34.52%)

---

## What the Controls Rule Out

- **Shuffled = 50%** → Signal is in the temporal order, not the value distribution
- **LSB = 50%** → Signal is in multi-qubit correlations, not single-qubit statistics
- **LR/RF = chance** → Signal requires sequential modeling (LSTM), not static features
- **Quick training = 50%** → Signal is non-trivial, requires sufficient learning
- **Threshold at 8k** → Signal is a long-range temporal structure, not short-range pattern

---

## Repository Structure

```
├── code/
│   ├── grammar_learner.py          # SAX + LSTM pipeline, grammar extraction
│   ├── signal_processing.py        # SAX encoding (quantile-based)
│   ├── run_validation_pipeline.py  # Full validation: shuffled, sweep, report
│   └── run_readout_lsb_bit_experiment.py  # LSB single-bit control
├── results/
│   ├── readout_raw_data/           # Sycamore raw readout files (28 configs)
│   ├── parameter_sweep_sax7.csv    # Full 60-combination sweep
│   ├── validation_report.txt       # Automated PASS/FAIL report
│   ├── shuffled_control_sax7.txt   # Shuffled control results
│   ├── readout_lsb_bit_control_sax7.txt  # LSB control results
│   ├── robustness_audit_sax7.txt   # Original 3-seed audit
│   └── sycamore_sax_lr_rf_summary.* # Baseline classifier results
├── validation_criteria.json        # Pre-defined decision thresholds
├── requirements.txt
└── README.md
```

---

## Reproduction

### Requirements
```bash
pip install -r requirements.txt
```

### Run Full Validation
```bash
# Original robustness audit (3 seeds)
python code/run_validation_pipeline.py --tasks sweep,report

# Shuffled temporal-order control
python code/run_validation_pipeline.py --tasks shuffled

# LSB single-bit control
python code/run_readout_lsb_bit_experiment.py \
  --output-bit-index 0 --epochs 50 --max-pts 10000 \
  --seeds 0 123 999
```

### Pipeline Overview
1. Load raw readout `output` column (integer multi-qubit state per shot)
2. Z-normalize → SAX encoding (alphabet=7, quantile breakpoints)
3. Train character-level LSTM (hidden=32, seq_len=16) on next-symbol prediction
4. Extract learned transition matrix (grammar fingerprint)
5. Compute pairwise Frobenius distances between fingerprints
6. Ward hierarchical clustering into 3 groups
7. Evaluate cluster purity against ground-truth topology labels

**Note:** Ground-truth labels are used **only** for evaluation, never during training. The model receives only the raw signal — no labels, no file names, no hardware metadata.

---

## Negative Results (reported for completeness)

- **IBM daily calibration data:** Grammar Fingerprinting did not achieve above-chance accuracy on pre-processed daily calibration summaries (33–60%). The method requires dense raw measurement data, not aggregated metrics.
- **Cosmological parameter encoding hypothesis (Beck):** Tested and rejected (p = 0.76, not significant).

---

## Method Origin

Grammar Fingerprinting applies industrial vibration diagnostics logic to quantum processor noise: the same principle by which a machine's acoustic signature reveals mechanical faults can identify quantum hardware structure from measurement noise. The method was developed independently; AI tools (Claude, Cursor) were used for code development and validation design. All experimental decisions, execution, and interpretation are the author's own work.

---

## License

Code: MIT  
Results and documentation: CC-BY 4.0  

---

## Contact

Dániel Csaplár  
Kazincbarcika, Hungary  
ORCID: [0009-0000-7362-7232](https://orcid.org/0009-0000-7362-7232)
