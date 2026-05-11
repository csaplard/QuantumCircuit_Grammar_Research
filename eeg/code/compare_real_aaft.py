"""
Real vs AAFT osszehasonlitas - a kulcs tudomanyos teszt.
Hipotezis: rest-en real ~ AAFT (linearis grammatika), arithmetic-on real < AAFT
(nem-linearis tartalom amit AAFT nem fed).
"""

import sys
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LABEL_RX = re.compile(r"^\[S(\d+)_(\w+?)_(\w+)_(real|aaft_r\d+)\]")
PPL_RX = re.compile(r"^\s*Perplexity:\s*([\d\.]+)")


def parse_log(log_path):
    rows = []
    cur = None
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = LABEL_RX.match(line)
            if m:
                cond = 'real' if m.group(4) == 'real' else 'aaft'
                cur = (int(m.group(1)), m.group(2), m.group(3), cond)
                continue
            m = PPL_RX.match(line)
            if m and cur is not None:
                rows.append({
                    'subject': cur[0], 'regime': cur[1], 'channel': cur[2],
                    'condition': cur[3], 'perplexity': float(m.group(1)),
                })
                cur = None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_csv', required=True)
    ap.add_argument('--aaft_log', required=True)
    args = ap.parse_args()

    real = pd.read_csv(args.real_csv)
    real = real[real.condition == 'real'][['subject','regime','channel','perplexity']]
    real['condition'] = 'real'

    aaft = parse_log(args.aaft_log)
    aaft = aaft[aaft.condition == 'aaft']

    print(f"Real rows: {len(real)}")
    print(f"AAFT rows (so far): {len(aaft)}, alanyok: {sorted(aaft['subject'].unique())}\n")

    # Csak a parositott (subject, regime, channel) hibrideken
    merged = real.merge(aaft, on=['subject','regime','channel'],
                        suffixes=('_real','_aaft'))
    merged['delta'] = merged['perplexity_aaft'] - merged['perplexity_real']
    print(f"Parositott (subj, regime, ch) triplet: {len(merged)}\n")

    print("=== Mean perplexity by regime, condition ===")
    summary = merged.groupby('regime')[['perplexity_real','perplexity_aaft','delta']].agg(['mean','std','count'])
    print(summary.round(3))

    print("\n=== Per-regime paired test (AAFT - real) ===")
    for regime in ['rest', 'arithmetic']:
        sub = merged[merged.regime == regime]
        if len(sub) < 6:
            print(f"  {regime}: tul keves adat ({len(sub)})")
            continue
        d = sub['delta'].values
        try:
            stat, p = wilcoxon(d)
        except ValueError:
            stat, p = float('nan'), 1.0
        cohens = d.mean() / (d.std() + 1e-10)
        print(f"  {regime}: n={len(d)}, mean delta = {d.mean():+.3f}, "
              f"d = {cohens:.3f}, Wilcoxon p = {p:.3g}, "
              f"{(d>0).mean()*100:.1f}% positive")

    # A KULCS TESZT: regime x condition interakcio
    # Hipotezis: rest-en delta ~ 0, arith-on delta > 0 (real < AAFT)
    print("\n=== Regime x condition interakcio (mean delta) ===")
    rest_d = merged[merged.regime=='rest']['delta'].values if len(merged[merged.regime=='rest']) else []
    arith_d = merged[merged.regime=='arithmetic']['delta'].values if len(merged[merged.regime=='arithmetic']) else []
    if len(rest_d) > 5 and len(arith_d) > 5:
        # Mann-Whitney U: arith deltak > rest deltak?
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(arith_d, rest_d, alternative='greater')
        print(f"  rest mean delta : {rest_d.mean():+.3f} (n={len(rest_d)})")
        print(f"  arith mean delta: {arith_d.mean():+.3f} (n={len(arith_d)})")
        print(f"  arith > rest delta? Mann-Whitney p = {p:.3g}")
        if p < 0.05:
            print("  --> REGIME-SPECIFIKUS NEM-LINEARITAS detektalva")
        else:
            print("  --> regime-specifikus elteres meg nem mutatkozik szignifikansan")


if __name__ == "__main__":
    main()
