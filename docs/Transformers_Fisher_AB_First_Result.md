# Transformers Fisher A/B (First Result)

This note documents the first live token-by-token Fisher-guided adaptive decoding test run using:

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Backend: `transformers` (direct Python loop, GPU)
- Horizon: 256 generated tokens
- Fisher: sliding window `W=32`, `epsilon=0.01`
- Control law: adaptive temperature with `mean+-sigma` thresholds

## Key takeaway

The control signal is active online (not only offline):

- Factual run: down-dominant intervention pattern
- Mathematical run: up-dominant intervention pattern

This mirrors the offline probe behavior and supports the claim that Fisher trace can act as a control signal, not just a diagnostic metric.

## Repro

1. Run baseline+adaptive experiment:
   - `python code/run_transformers_ab.py`
2. Build report table + figure:
   - `python code/build_transformers_ab_report.py ...` (pass the 4 trace + 4 text paths)

Generated outputs include:

- `transformers_ab_summary.csv`
- `transformers_ab_summary.md`
- `transformers_ab_trace_plot.png`
