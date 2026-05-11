"""
EDF betoltes + minimal eloprocesszalas a PhysioNet eegmat dataset-hez.
A kimenet 1D float numpy array csatornankent, kesz a grammar_learner.train_model
data_is_array=True hivasahoz.
"""

import os
import re
import numpy as np

try:
    import mne
except ImportError as e:
    raise ImportError(
        "MNE-Python kell. Telepites: pip install mne"
    ) from e


# Default sav-szuro hatarok (Hz). Az eegmat dataset 500 Hz-en van.
# 1-40 Hz lefedi delta..gamma (alacsony) savot, kiszuri a halozati zumogest es a DC-driftet.
DEFAULT_LFREQ = 1.0
DEFAULT_HFREQ = 40.0
DEFAULT_NOTCH = 50.0  # Europai halozati frekvencia (Ukrajna).


def load_edf(path, lfreq=DEFAULT_LFREQ, hfreq=DEFAULT_HFREQ, notch=DEFAULT_NOTCH,
             target_fs=None, drop_refs=True, verbose=False):
    """
    Beolvas egy EDF fajlt, savszurot/notch-ot/reszamplalast alkalmaz,
    visszaad egy mne.io.Raw objektumot.

    Args:
        path: EDF fajl utvonala.
        lfreq, hfreq: bandpass hatarok Hz-ben (None=skip).
        notch: notch frekvencia Hz (None=skip).
        target_fs: cel mintaveteli frekvencia Hz-ben. Ha None, marad az eredeti
                   (eegmat eseteben 500 Hz). EEG-re EZ A FONTOS PARAMETER:
                   100 Hz ajanlott amikor a SAX+LSTM seq_len=20 ablakot
                   neuralis ritmus-idoskalara akarjuk illeszteni
                   (200 ms = ~2 alpha-ciklus + 1 theta-ciklus).
                   A reszamplalas a sav-szuro UTAN tortenik (Nyquist OK 40 Hz hatarra).
        drop_refs: ha True, eldobja az A1/A2 fulreferenciat (csak 19 csatorna marad).
        verbose: MNE log szint.

    Returns:
        mne.io.Raw - preload-olva, szurve, opcionalisan reszamplalva.
    """
    log = 'INFO' if verbose else 'WARNING'
    raw = mne.io.read_raw_edf(path, preload=True, verbose=log)

    # Eldobando csatornak:
    #   - linked-ear referencia (eegmat-ben 'EEG A2-A1' kombinalt forma, ~zero jel)
    #   - ECG (szivritmus, nem agyi)
    # Az 'EEG ' prefixet eltavolitjuk a csatornanevekrol a tisztabb logoles erdekeben.
    drop_list = []
    for ch in raw.ch_names:
        ch_low = ch.lower().replace('eeg ', '').strip()
        if ch_low.startswith('ecg') or ch_low.startswith('emg'):
            drop_list.append(ch)
        elif drop_refs and ch_low in ('a1', 'a2', 'a2-a1', 'a1-a2'):
            drop_list.append(ch)
    if drop_list:
        raw.drop_channels(drop_list)

    # 'EEG Fp1' -> 'Fp1' atnevezes
    rename = {ch: ch.replace('EEG ', '').strip()
              for ch in raw.ch_names if ch.startswith('EEG ')}
    if rename:
        raw.rename_channels(rename)

    if notch is not None:
        raw.notch_filter(freqs=notch, verbose=log)
    if lfreq is not None or hfreq is not None:
        raw.filter(l_freq=lfreq, h_freq=hfreq, verbose=log)

    if target_fs is not None and target_fs < raw.info['sfreq']:
        # MNE resample: alkalmaz antialias-szurot (ami a band-limit miatt OK ha hfreq < target_fs/2)
        raw.resample(sfreq=target_fs, verbose=log)

    return raw


def raw_to_channel_arrays(raw):
    """
    Raw objektumot at-formaz: dict {channel_name: 1D float array}.
    Az ertekek mikrovolt egysegben (MNE alapertelmezett a Volt, *1e6).
    """
    data = raw.get_data() * 1e6  # Volt -> microVolt
    ch_names = raw.ch_names
    return {name: data[i].astype(np.float64) for i, name in enumerate(ch_names)}


SUBJECT_RX = re.compile(r"Subject(\d+)_([12])\.edf$", re.IGNORECASE)


def discover_subjects(data_dir):
    """
    Vegigmegy a data_dir-on, visszaad egy listat:
        [(subject_id, regime, full_path), ...]
    ahol regime 'rest' vagy 'arithmetic'.
    """
    out = []
    for fname in sorted(os.listdir(data_dir)):
        m = SUBJECT_RX.search(fname)
        if not m:
            continue
        subj = int(m.group(1))
        flag = m.group(2)
        regime = 'rest' if flag == '1' else 'arithmetic'
        out.append((subj, regime, os.path.join(data_dir, fname)))
    return out


def load_subject_regime(data_dir, subject_id, regime, **filt_kwargs):
    # filt_kwargs tovabbitva a load_edf-be: lfreq, hfreq, notch, target_fs, drop_refs, verbose
    """
    Egy konkret alany + regime betoltese, vissza dict {ch_name: signal}.
    """
    flag = '1' if regime == 'rest' else '2'
    candidates = [
        f"Subject{subject_id:02d}_{flag}.edf",
        f"subject{subject_id:02d}_{flag}.edf",
    ]
    for c in candidates:
        path = os.path.join(data_dir, c)
        if os.path.exists(path):
            raw = load_edf(path, **filt_kwargs)
            return raw_to_channel_arrays(raw), raw.info['sfreq']
    raise FileNotFoundError(
        f"Nem talaltam: subject={subject_id}, regime={regime} a {data_dir}-ben"
    )


if __name__ == "__main__":
    # Sanity check: ha argumentumot kapunk, betoltunk egy EDF-et es kiirunk parameterek.
    import sys
    if len(sys.argv) < 2:
        print("Hasznalat: python eeg_io.py <path_to_edf>")
        sys.exit(1)
    raw = load_edf(sys.argv[1], verbose=True)
    print(f"sfreq         : {raw.info['sfreq']} Hz")
    print(f"n_channels    : {len(raw.ch_names)}")
    print(f"channel_names : {raw.ch_names}")
    print(f"n_samples     : {raw.n_times}")
    print(f"duration      : {raw.times[-1]:.2f} s")
    chs = raw_to_channel_arrays(raw)
    first = next(iter(chs))
    s = chs[first]
    print(f"sample channel '{first}': mean={s.mean():.3f} uV, std={s.std():.3f} uV, "
          f"min={s.min():.2f}, max={s.max():.2f}")
