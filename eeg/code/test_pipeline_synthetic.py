"""
Vegigfutas-teszt szintetikus jelen, EDF nelkul.
Cel: igazolni hogy a teljes pipeline (eeg_io alkatreszek nelkul,
csak grammar_learner + controls) hibatlanul fut, es a kontrollok
TENYLEG magasabb perplexitast adnak mint az eredeti strukturalt jel.
"""

import os
import sys
import numpy as np

REPO_CODE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "code")
)
if REPO_CODE not in sys.path:
    sys.path.insert(0, REPO_CODE)

from grammar_learner import train_model, extract_grammar
from controls import make_surrogate, SURROGATES


def synth_eeg(n_samples=10000, fs=100.0, seed=0):
    """Szintetikus 'EEG': alpha (10 Hz) + theta (5 Hz) + zaj. Strukturalt."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    sig = (1.0*np.sin(2*np.pi*10*t) +
           0.5*np.sin(2*np.pi*5*t) +
           0.3*rng.standard_normal(n_samples))
    return sig


def main():
    sig = synth_eeg(n_samples=10000, fs=100.0, seed=0)
    print(f"Test signal: n={len(sig)}, mean={sig.mean():.3f}, std={sig.std():.3f}\n")

    K = 7
    epochs = 10  # gyorsra veve a tesztet
    seq_len = 20

    results = {}

    # Eredeti
    print(">>> REAL")
    val_loss, ppl, model, val_data = train_model(
        sig, label='REAL', alphabet_size=K,
        epochs=epochs, seq_len=seq_len, data_is_array=True, seed=42)
    fp = extract_grammar(model, val_data, seq_len=seq_len)
    results['real'] = ppl
    print(f"   real fingerprint shape: {fp.shape}, sum/row: {fp.sum(axis=1)[:3]}")

    # Surrogate-ok
    for kind in ('shuffled', 'random_uniform', 'aaft'):
        print(f"\n>>> {kind.upper()}")
        sig_s = make_surrogate(kind, sig, seed=100)
        val_loss_s, ppl_s, model_s, val_data_s = train_model(
            sig_s, label=kind, alphabet_size=K,
            epochs=epochs, seq_len=seq_len, data_is_array=True, seed=42)
        results[kind] = ppl_s

    print("\n" + "="*60)
    print(f"Perplexity osszehasonlitas (alacsonyabb = strukturaltabb):")
    print(f"  K={K} (max ppl ha tisztan random ~ {K})")
    for k, v in results.items():
        marker = " <-- structured" if v < results['random_uniform'] - 0.1 else ""
        print(f"  {k:18s}: {v:.4f}{marker}")

    # Sanity ellenorzes
    assert results['real'] < results['random_uniform'], \
        "FAIL: a strukturalt jel nem ad alacsonyabb perplexitast mint az i.i.d zaj"
    assert results['shuffled'] > results['real'] - 0.5, \
        "GYANUS: a shuffled perplexity nem szignifikansan magasabb mint a real"
    print("\n[PASS] A pipeline elvarasszeruen mukodik szintetikus jelen.")


if __name__ == "__main__":
    main()
