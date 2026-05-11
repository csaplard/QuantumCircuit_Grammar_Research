"""
Statisztikai elemzes az eeg_grammar_K*.csv eredmenyen.

Mit csinal:
  1. Per-csatorna paired test (Wilcoxon signed-rank) rest vs arithmetic perplexitas-ra,
     alanyok kozott parositva.
  2. Effect size (Cohen's d-szeru, paired).
  3. Globalis paired test az osszes (subject, channel) paron.
  4. Fingerprint-alapu regime klasszifikacio: leave-one-subject-out CV,
     logistic regression a fingerprint flatten vektoran.
  5. Topografikus delta-terkep export-rea kesz tablazat.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def per_channel_test(df):
    """Subject-szintu parositott Wilcoxon teszt csatornankent."""
    real = df[df.condition == 'real']
    rows = []
    for ch, g in real.groupby('channel'):
        rest = g[g.regime == 'rest'].set_index('subject')['perplexity']
        arith = g[g.regime == 'arithmetic'].set_index('subject')['perplexity']
        common = rest.index.intersection(arith.index)
        if len(common) < 6:
            continue
        rest = rest.loc[common].values
        arith = arith.loc[common].values
        diff = arith - rest
        try:
            stat, p = wilcoxon(diff)
        except ValueError:
            stat, p = np.nan, 1.0
        d = diff.mean() / (diff.std() + 1e-10)  # Cohen's d (paired)
        rows.append({
            'channel': ch,
            'n_subjects': len(common),
            'mean_delta': diff.mean(),
            'sd_delta': diff.std(),
            'cohens_d': d,
            'wilcoxon_W': stat,
            'p_value': p,
        })
    if not rows:
        return pd.DataFrame(columns=['channel','n_subjects','mean_delta',
                                      'sd_delta','cohens_d','wilcoxon_W',
                                      'p_value','p_fdr'])
    out = pd.DataFrame(rows).sort_values('p_value')
    # FDR (Benjamini-Hochberg) korrekcio
    p = out['p_value'].values
    n = len(p)
    order = np.argsort(p)
    ranked = np.arange(1, n+1)
    fdr = (p[order] * n / ranked)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr_full = np.empty(n)
    fdr_full[order] = fdr
    out['p_fdr'] = np.clip(fdr_full, 0, 1)
    return out


def global_test(df):
    """Globalis paired test az osszes (subject, channel) paron."""
    real = df[df.condition == 'real']
    pivot = real.pivot_table(index=['subject', 'channel'], columns='regime',
                              values='perplexity').dropna()
    diff = pivot['arithmetic'] - pivot['rest']
    stat, p = wilcoxon(diff)
    return {
        'n_pairs': len(diff),
        'mean_delta': diff.mean(),
        'sd_delta': diff.std(),
        'cohens_d_paired': diff.mean() / (diff.std() + 1e-10),
        'wilcoxon_W': stat,
        'p_value': p,
        'frac_positive': (diff > 0).mean(),
    }


def fingerprint_classification(npz_path, df, K, n_per_class_min=10):
    """
    Logistic regression LOSO CV: fingerprint mint feature, regime mint cimke.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    if not os.path.exists(npz_path):
        return None

    fps = np.load(npz_path)
    X, y, groups = [], [], []
    for label in fps.files:
        # label format: S{subj}_{regime}_{channel}_real
        if not label.endswith('_real'):
            continue
        parts = label.split('_')
        # S00 _ rest _ Fp1 _ real
        subj = int(parts[0][1:])
        regime = parts[1]
        # channel name might contain underscore? not in this dataset
        X.append(fps[label].flatten())
        y.append(0 if regime == 'rest' else 1)
        groups.append(subj)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return None

    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y[tr])
        accs.append(clf.score(Xte, y[te]))
    return {
        'n_streams': len(X),
        'n_subjects': len(np.unique(groups)),
        'mean_accuracy': float(np.mean(accs)),
        'sd_accuracy': float(np.std(accs)),
        'fold_accs': accs,
        'chance_level': 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--npz', default=None,
                    help='fingerprint NPZ path (csv-bol kovetkeztetjuk ha None)')
    ap.add_argument('--out_dir', default='results/analysis')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    K = int(df['K'].iloc[0])

    if args.npz is None:
        guess = args.csv.replace('eeg_grammar_K', 'eeg_fingerprints_K').replace('.csv', '.npz')
        args.npz = guess

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] CSV: {args.csv}  ({len(df)} sor)")
    print(f"[*] NPZ: {args.npz} ({'OK' if os.path.exists(args.npz) else 'NEM TALALT'})")
    print(f"[*] K={K}, alanyok: {df['subject'].nunique()}, csatornak: {df['channel'].nunique()}")
    print()

    # Globalis
    print("=== Globalis paired test (rest vs arithmetic, real only) ===")
    g = global_test(df)
    for k, v in g.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.4g}")
        else:
            print(f"  {k:20s} = {v}")
    print()

    # Per-csatorna
    print("=== Per-csatorna paired test (FDR korrigalt) ===")
    pc = per_channel_test(df)
    print(pc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    pc.to_csv(os.path.join(out_dir, 'per_channel_test.csv'), index=False)
    print(f"\n  -> {out_dir}/per_channel_test.csv")
    print()

    # Klasszifikacio
    print("=== Fingerprint-alapu LOSO klasszifikacio ===")
    cls = fingerprint_classification(args.npz, df, K)
    if cls is None:
        print("  (kihagyva - sklearn vagy NPZ nem elerheto)")
    else:
        print(f"  n_streams: {cls['n_streams']}")
        print(f"  n_subjects: {cls['n_subjects']}")
        print(f"  mean accuracy: {cls['mean_accuracy']:.3f} +/- {cls['sd_accuracy']:.3f}")
        print(f"  chance level : {cls['chance_level']:.3f}")
        print(f"  per-fold     : {[f'{a:.2f}' for a in cls['fold_accs']]}")

    # Surrogate kontrollok ha vannak
    if 'condition' in df.columns and df['condition'].nunique() > 1:
        print("\n=== Condition-szintu osszehasonlitas (mean perplexity) ===")
        print(df.groupby(['regime', 'condition'])['perplexity']
                .agg(['mean', 'std', 'count']).round(3).to_string())


if __name__ == "__main__":
    main()
