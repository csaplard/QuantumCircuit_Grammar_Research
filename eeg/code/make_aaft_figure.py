"""
Figure 6: AAFT vs real comparison (per-regime + interaction).
"""
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combined', default='results/full_combined.csv')
    ap.add_argument('--out', default='paper/figures/fig6_aaft.png')
    args = ap.parse_args()

    df = pd.read_csv(args.combined)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A. Boxplot of deltas per regime
    ax = axes[0]
    rest_d = df[df.regime=='rest']['delta'].values
    arith_d = df[df.regime=='arithmetic']['delta'].values
    bp = ax.boxplot([rest_d, arith_d],
                     tick_labels=[f'Rest\n(n={len(rest_d)})',
                                  f'Arithmetic\n(n={len(arith_d)})'],
                     widths=0.5, patch_artist=True, showfliers=False,
                     medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#9ecae1')
    bp['boxes'][1].set_facecolor('#fdae6b')
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_ylabel('Delta perplexity (AAFT - real)', fontsize=11)
    ax.set_title('A. AAFT - real perplexity gap per regime\n'
                 f'rest mean = +{rest_d.mean():.3f}, arith mean = +{arith_d.mean():.3f}',
                 fontsize=11, loc='left')
    ax.grid(axis='y', alpha=0.3)

    # B. Scatter: real vs AAFT per regime
    ax = axes[1]
    for regime, color, label in [('rest', '#3182bd', 'Rest'),
                                   ('arithmetic', '#e6550d', 'Arithmetic')]:
        sub = df[df.regime==regime]
        ax.scatter(sub['perplexity_real'], sub['perplexity_aaft'],
                    alpha=0.3, s=15, color=color, label=label, edgecolor='none')
    lo = min(df['perplexity_real'].min(), df['perplexity_aaft'].min()) - 0.2
    hi = max(df['perplexity_real'].max(), df['perplexity_aaft'].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.7, label='y = x')
    ax.set_xlabel('Real signal perplexity', fontsize=11)
    ax.set_ylabel('AAFT surrogate perplexity', fontsize=11)
    ax.set_title('B. Real vs AAFT perplexity by regime', fontsize=11, loc='left')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # C. Per-channel mean delta by regime
    ax = axes[2]
    by_ch = df.groupby(['channel','regime'])['delta'].mean().unstack()
    ch_order = by_ch.mean(axis=1).sort_values().index
    by_ch = by_ch.reindex(ch_order)
    y = np.arange(len(by_ch))
    width = 0.4
    ax.barh(y - width/2, by_ch['rest'].values, height=width,
             color='#3182bd', label='Rest', alpha=0.85)
    ax.barh(y + width/2, by_ch['arithmetic'].values, height=width,
             color='#e6550d', label='Arithmetic', alpha=0.85)
    ax.axvline(0, color='black', lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ch_order, fontsize=8)
    ax.set_xlabel('Mean delta (AAFT - real)', fontsize=11)
    ax.set_title('C. Per-channel AAFT - real delta by regime', fontsize=11, loc='left')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
