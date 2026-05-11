"""
Paper-szintu abrak generalasa a real-only futasbol.

Kimeneti abrak (paper/figures/):
  fig2_main_effect.png      - boxplot + scatter (rest vs arithmetic)
  fig3_topography.png       - csatorna-delta topografikus terkep
  fig4_loso.png             - LOSO accuracy histogram + per-fold
  fig5_fingerprints.png     - atlogolt fingerprint matrixok + diff
  fig6_subject_distribution - per-alany delta distribuciok (24/12 visualization)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


CHANNEL_ORDER_1020 = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                      'T3','C3','Cz','C4','T4',
                      'T5','P3','Pz','P4','T6',
                      'O1','O2']


def load_montage_pos():
    """1020 standardhoz pozicio (head-coords)."""
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']
    # MNE T3/T4/T5/T6 -> T7/T8/P7/P8 az ujabb labelek;
    # eegmat regi labeleket hasznal, alias-table:
    alias = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    out = {}
    for ch in CHANNEL_ORDER_1020:
        key = alias.get(ch, ch)
        if key in pos:
            out[ch] = pos[key][:2]  # x, y (top-down view)
    return out


def fig2_main_effect(df, out_path):
    real = df[df.condition == 'real']
    pivot = real.pivot_table(index=['subject','channel'], columns='regime',
                              values='perplexity').dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    bp = ax.boxplot([pivot['rest'].values, pivot['arithmetic'].values],
                     labels=['Rest\n(eyes closed)', 'Mental arithmetic'],
                     widths=0.5, patch_artist=True,
                     boxprops=dict(linewidth=1.2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.2))
    bp['boxes'][0].set_facecolor('#9ecae1')
    bp['boxes'][1].set_facecolor('#fdae6b')
    ax.set_ylabel('LSTM perplexity (K=7)', fontsize=12)
    ax.set_title(f'A. Grammar perplexity by regime (n={len(pivot)} streams)',
                  fontsize=12, loc='left')
    ax.grid(axis='y', alpha=0.3)

    # Scatter
    ax = axes[1]
    ax.scatter(pivot['rest'], pivot['arithmetic'], alpha=0.4, s=20,
                color='#3182bd', edgecolor='none')
    lo, hi = pivot.min().min() - 0.2, pivot.max().max() + 0.2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label='y = x')
    n_above = (pivot['arithmetic'] > pivot['rest']).sum()
    ax.set_xlabel('Perplexity (rest)', fontsize=12)
    ax.set_ylabel('Perplexity (mental arithmetic)', fontsize=12)
    ax.set_title(f'B. Per (subject, channel) pair: {n_above}/{len(pivot)} '
                  f'({100*n_above/len(pivot):.1f}%) above diagonal',
                  fontsize=12, loc='left')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def fig3_topography(df, out_path):
    real = df[df.condition == 'real']
    pivot = real.pivot_table(index=['subject','channel'], columns='regime',
                              values='perplexity').dropna()
    pivot['delta'] = pivot['arithmetic'] - pivot['rest']
    by_ch = pivot.groupby(level='channel')['delta'].agg(['mean', 'sem'])
    by_ch = by_ch.reindex(CHANNEL_ORDER_1020).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Topography - MNE plot_topomap-pel (helyes geometriaval)
    ax = axes[0]
    # Atnevezzuk a regi (T3/T4/T5/T6) cimkeket az MNE altal kezelt T7/T8/P7/P8-ra
    alias = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    chs = list(by_ch.index)
    chs_for_mne = [alias.get(c, c) for c in chs]
    vals = by_ch['mean'].values

    info = mne.create_info(ch_names=chs_for_mne, sfreq=100.0, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(np.abs(vals)))
    im, _ = mne.viz.plot_topomap(
        vals, info, axes=ax, show=False,
        cmap='RdBu_r', vlim=(-vmax, vmax),
        names=chs,  # eredeti label-ek (T3 stb.) megjelenitve
        sensors=True, contours=6,
        sphere='auto',
    )
    ax.set_title('A. Topographic delta map (arithmetic - rest)\n'
                 'Mean perplexity difference per electrode (n=36 subjects)',
                 fontsize=11, loc='left')
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label('Mean delta (perplexity)', fontsize=10)

    # Bar chart
    ax = axes[1]
    by_ch_sorted = by_ch.sort_values('mean')
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in by_ch_sorted['mean']]
    y = np.arange(len(by_ch_sorted))
    ax.barh(y, by_ch_sorted['mean'], xerr=by_ch_sorted['sem'],
            color=colors, alpha=0.8, edgecolor='black', linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(by_ch_sorted.index, fontsize=9)
    ax.axvline(0, color='k', lw=0.7)
    ax.set_xlabel('Mean delta (perplexity, arithmetic - rest)', fontsize=11)
    ax.set_title(f'B. Per-channel mean delta with SEM, sorted',
                 fontsize=11, loc='left')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def fig4_loso(npz_path, out_path):
    fps = np.load(npz_path)
    X, y, groups = [], [], []
    for label in fps.files:
        if not label.endswith('_real'):
            continue
        parts = label.split('_')
        subj = int(parts[0][1:])
        regime = parts[1]
        X.append(fps[label].flatten())
        y.append(0 if regime == 'rest' else 1)
        groups.append(subj)
    X = np.array(X); y = np.array(y); groups = np.array(groups)

    logo = LeaveOneGroupOut()
    fold_accs = []
    all_y_true, all_y_pred = [], []
    for tr, te in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[te]))
        fold_accs.append((pred == y[te]).mean())
        all_y_true.extend(y[te])
        all_y_pred.extend(pred)

    fold_accs = np.array(fold_accs)
    cm = confusion_matrix(all_y_true, all_y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(fold_accs, bins=15, color='#3182bd', edgecolor='black', alpha=0.85)
    ax.axvline(0.5, color='red', lw=1.5, ls='--', label='chance')
    ax.axvline(fold_accs.mean(), color='black', lw=2,
                label=f'mean = {fold_accs.mean():.3f} ± {fold_accs.std():.3f}')
    ax.set_xlabel('LOSO fold accuracy', fontsize=12)
    ax.set_ylabel('Number of folds', fontsize=12)
    ax.set_title(f'A. Leave-one-subject-out CV accuracy distribution\n'
                 f'(36 folds, logistic regression on flattened fingerprints)',
                 fontsize=11, loc='left')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # Confusion matrix
    ax = axes[1]
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Rest', 'Arithmetic'])
    ax.set_yticklabels(['Rest', 'Arithmetic'])
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    for i in range(2):
        for j in range(2):
            txt_color = 'white' if cm_pct[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_pct[i,j]:.1f}%\n(n={cm[i,j]})',
                    ha='center', va='center', color=txt_color,
                    fontsize=12, fontweight='bold')
    ax.set_title('B. Aggregated confusion matrix (row-normalized)\n'
                 'Pooled across all LOSO folds',
                 fontsize=11, loc='left')
    plt.colorbar(im, ax=ax, fraction=0.046, label='% of true class')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def fig5_fingerprints(npz_path, out_path):
    fps = np.load(npz_path)
    rest_mats, arith_mats = [], []
    for label in fps.files:
        if not label.endswith('_real'):
            continue
        if '_rest_' in label:
            rest_mats.append(fps[label])
        elif '_arithmetic_' in label:
            arith_mats.append(fps[label])
    rest_avg = np.mean(rest_mats, axis=0)
    arith_avg = np.mean(arith_mats, axis=0)
    diff = arith_avg - rest_avg
    K = diff.shape[0]
    syms = list('abcdefghijklmnop'[:K])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax_main = max(rest_avg.max(), arith_avg.max())

    for ax, mat, ttl in [
        (axes[0], rest_avg, f'A. Rest fingerprint\n(avg over {len(rest_mats)} streams)'),
        (axes[1], arith_avg, f'B. Arithmetic fingerprint\n(avg over {len(arith_mats)} streams)'),
    ]:
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=vmax_main)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(syms); ax.set_yticklabels(syms)
        ax.set_xlabel('Next symbol', fontsize=11)
        ax.set_ylabel('Current symbol', fontsize=11)
        ax.set_title(ttl, fontsize=11, loc='left')
        plt.colorbar(im, ax=ax, fraction=0.046, label='P(next | current)')

    ax = axes[2]
    vmax_d = np.abs(diff).max()
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(syms); ax.set_yticklabels(syms)
    ax.set_xlabel('Next symbol', fontsize=11)
    ax.set_ylabel('Current symbol', fontsize=11)
    ax.set_title(f'C. Difference (arithmetic - rest)\n'
                 f'max |diff| = {vmax_d:.4f}', fontsize=11, loc='left')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Delta probability')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/full_real/eeg_grammar_K7.csv')
    ap.add_argument('--npz', default='results/full_real/eeg_fingerprints_K7.npz')
    ap.add_argument('--out_dir', default='paper/figures')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, generating paper figures into {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    fig2_main_effect(df, os.path.join(args.out_dir, 'fig2_main_effect.png'))
    fig3_topography(df, os.path.join(args.out_dir, 'fig3_topography.png'))
    fig4_loso(args.npz, os.path.join(args.out_dir, 'fig4_loso.png'))
    fig5_fingerprints(args.npz, os.path.join(args.out_dir, 'fig5_fingerprints.png'))


if __name__ == "__main__":
    main()
