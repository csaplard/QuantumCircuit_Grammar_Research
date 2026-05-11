"""
EEG Grammar Fingerprinting fo futtato.

Vegigmegy a PhysioNet eegmat alanyokon, mindket regime-en (rest/arithmetic),
csatornankent meghivja a grammar_learner.train_model + extract_grammar parost,
es elmenti a fingerprint matrixokat + perplexity-ket egy konszolidalt CSV/NPZ-be.

Hasznalat:
    python run_eeg_grammar.py --data_dir <eegmat_dir> [--subjects 0-5] [--K 7]
    python run_eeg_grammar.py --data_dir <eegmat_dir> --quick      # 1 alany sanity
"""

import os
import sys
import json
import argparse
import time
import numpy as np

# A grammar_learner es signal_processing az archive-ban van.
REPO_CODE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "code")
)
if REPO_CODE not in sys.path:
    sys.path.insert(0, REPO_CODE)

from grammar_learner import train_model, extract_grammar  # noqa: E402

from eeg_io import discover_subjects, load_subject_regime  # noqa: E402
from controls import make_surrogate, SURROGATES  # noqa: E402


def parse_subject_spec(spec, available):
    """'0-5' vagy '0,3,7' vagy 'all' -> list of int subject ids."""
    if spec in (None, '', 'all'):
        return sorted(available)
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out & set(available))


def run_one_channel(signal, label, K, hidden_dim, seq_len, epochs, lr, seed):
    """Egy 1D jelre lefuttatja a teljes pipeline-t. Visszater dict-tel."""
    t0 = time.time()
    val_loss, ppl, model, val_data = train_model(
        signal,
        label=label,
        alphabet_size=K,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        epochs=epochs,
        lr=lr,
        data_is_array=True,
        seed=seed,
    )
    fingerprint = extract_grammar(model, val_data, seq_len=seq_len)
    return {
        'label': label,
        'val_loss': float(val_loss),
        'perplexity': float(ppl),
        'fingerprint': fingerprint.astype(np.float32),
        'wall_seconds': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='eegmat EDF konyvtar')
    ap.add_argument('--out_dir', default='results',
                    help='kimeneti mappa (relativan a script-hez)')
    ap.add_argument('--subjects', default='all',
                    help="pl. '0-5' vagy '0,3,7' vagy 'all'")
    ap.add_argument('--regimes', default='rest,arithmetic',
                    help='rest, arithmetic vagy mindketto')
    ap.add_argument('--channels', default='all',
                    help='vesszovel elvalasztott neveklista vagy all')
    ap.add_argument('--target_fs', type=float, default=None,
                    help='Cel mintaveteli frekvencia Hz-ben. None = nincs reszamplalas. '
                         'EEG-re ajanlott: 100 Hz (lasd Methods).')
    ap.add_argument('--lfreq', type=float, default=1.0, help='bandpass also hatara Hz')
    ap.add_argument('--hfreq', type=float, default=40.0, help='bandpass felso hatara Hz')
    ap.add_argument('--notch', type=float, default=50.0, help='notch frekvencia Hz (None=skip)')
    ap.add_argument('--keep_refs', action='store_true',
                    help='ne dobja ki az A1/A2 referencia csatornakat')
    ap.add_argument('--K', type=int, default=7, help='SAX alphabet meret')
    ap.add_argument('--hidden_dim', type=int, default=16)
    ap.add_argument('--seq_len', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip_real', action='store_true',
                    help='Ne futtassa a real signalt - csak surrogate-okat. '
                         'Hasznos ha mar megvan a real adat es csak surrogate-okat akarsz.')
    ap.add_argument('--surrogates', default='',
                    help=f"vesszovel elvalasztott lista; ures = csak az eredeti jel. "
                         f"Elerheto: {','.join(SURROGATES.keys())}. "
                         f"Pl. 'shuffled,aaft,random_uniform'.")
    ap.add_argument('--n_surrogate_reps', type=int, default=1,
                    help='Hany kulonbozo seed-del generaljuk minden surrogate-ot (atlagolasra)')
    ap.add_argument('--quick', action='store_true',
                    help='1 alany, 1 csatorna, 5 epoch -- sanity check')
    args = ap.parse_args()

    if args.quick:
        args.subjects = '0'
        args.epochs = 5

    out_dir = os.path.join(os.path.dirname(__file__), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    found = discover_subjects(args.data_dir)
    available_subjects = sorted({s for s, _, _ in found})
    if not available_subjects:
        print(f"[!] Nem talaltam Subject*_*.edf-et itt: {args.data_dir}")
        sys.exit(2)

    subjects = parse_subject_spec(args.subjects, available_subjects)
    regimes = [r.strip() for r in args.regimes.split(',')]
    channel_filter = None
    if args.channels != 'all':
        channel_filter = {c.strip() for c in args.channels.split(',')}

    print(f"[*] Alanyok: {subjects}")
    print(f"[*] Regime-ek: {regimes}")
    print(f"[*] K={args.K}, hidden={args.hidden_dim}, seq_len={args.seq_len}, "
          f"epochs={args.epochs}, lr={args.lr}, seed={args.seed}")
    print(f"[*] Filt: lfreq={args.lfreq}, hfreq={args.hfreq}, notch={args.notch}, "
          f"target_fs={args.target_fs}, drop_refs={not args.keep_refs}")

    filt_kwargs = dict(
        lfreq=args.lfreq if args.lfreq > 0 else None,
        hfreq=args.hfreq if args.hfreq > 0 else None,
        notch=args.notch if args.notch and args.notch > 0 else None,
        target_fs=args.target_fs,
        drop_refs=not args.keep_refs,
    )

    rows = []          # CSV-be: skalar metrikak
    fingerprints = {}  # NPZ-be: {label: KxK matrix}

    for subj in subjects:
        for regime in regimes:
            print(f"\n=== Subject {subj:02d} / {regime} ===")
            try:
                channels, sfreq = load_subject_regime(args.data_dir, subj, regime, **filt_kwargs)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue
            print(f"  sfreq={sfreq} Hz, n_channels={len(channels)}, "
                  f"n_samples={len(next(iter(channels.values())))}")

            for ch_name, sig in channels.items():
                if channel_filter is not None and ch_name not in channel_filter:
                    continue
                if args.quick and len(rows) >= 1:
                    break

                # Eredeti jel (kihagyhato --skip_real-lal)
                if not args.skip_real:
                    label = f"S{subj:02d}_{regime}_{ch_name}_real"
                    try:
                        res = run_one_channel(
                            sig, label,
                            K=args.K, hidden_dim=args.hidden_dim,
                            seq_len=args.seq_len, epochs=args.epochs,
                            lr=args.lr, seed=args.seed,
                        )
                    except Exception as ex:
                        print(f"  [!] {label} hiba: {ex}")
                        continue

                    rows.append({
                        'subject': subj, 'regime': regime, 'channel': ch_name,
                        'condition': 'real', 'rep': 0,
                        'K': args.K, 'sfreq': float(sfreq), 'n_samples': int(len(sig)),
                        'val_loss': res['val_loss'], 'perplexity': res['perplexity'],
                        'wall_seconds': res['wall_seconds'],
                    })
                    fingerprints[label] = res['fingerprint']

                # Surrogate kontrollok
                surr_list = [s.strip() for s in args.surrogates.split(',') if s.strip()]
                for surr_kind in surr_list:
                    for rep in range(args.n_surrogate_reps):
                        # Determinisztikus seed surrogate-onkent
                        s_seed = args.seed + 1000 * (1 + list(SURROGATES).index(surr_kind)) + rep
                        try:
                            sig_surr = make_surrogate(surr_kind, sig, seed=s_seed)
                        except Exception as ex:
                            print(f"  [!] surrogate {surr_kind} hiba: {ex}")
                            continue
                        s_label = f"S{subj:02d}_{regime}_{ch_name}_{surr_kind}_r{rep}"
                        try:
                            res_s = run_one_channel(
                                sig_surr, s_label,
                                K=args.K, hidden_dim=args.hidden_dim,
                                seq_len=args.seq_len, epochs=args.epochs,
                                lr=args.lr, seed=args.seed,
                            )
                        except Exception as ex:
                            print(f"  [!] {s_label} hiba: {ex}")
                            continue
                        rows.append({
                            'subject': subj, 'regime': regime, 'channel': ch_name,
                            'condition': surr_kind, 'rep': rep,
                            'K': args.K, 'sfreq': float(sfreq), 'n_samples': int(len(sig)),
                            'val_loss': res_s['val_loss'], 'perplexity': res_s['perplexity'],
                            'wall_seconds': res_s['wall_seconds'],
                        })
                        fingerprints[s_label] = res_s['fingerprint']

    # Mentes
    if not rows:
        print("[!] Semmi nem futott le.")
        sys.exit(3)

    import csv
    csv_path = os.path.join(out_dir, f"eeg_grammar_K{args.K}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    npz_path = os.path.join(out_dir, f"eeg_fingerprints_K{args.K}.npz")
    np.savez_compressed(npz_path, **fingerprints)

    meta_path = os.path.join(out_dir, f"eeg_grammar_K{args.K}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n[OK] {len(rows)} csatorna feldolgozva.")
    print(f"     CSV : {csv_path}")
    print(f"     NPZ : {npz_path}")
    print(f"     META: {meta_path}")


if __name__ == "__main__":
    main()
