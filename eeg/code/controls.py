"""
Null-modell generatorok a Fisher/grammar pipeline kontrolljaihoz.

Harom standard surrogat amelyek egyre szigorubban megorzik az eredeti jel
strukturajat, es egyre erosebb a teszt arra hogy a pipeline-od TENYLEG
sub-amplitudo szekvencia-grammatikat detektal:

    shuffled        - temporalis sorrend roncsolasa (random permutacio)
                      megorzi: marginalis amplitudo-eloszlas
                      roncsolja: minden temporalis struktura
                      Ha pipeline ezen JOL klasszifikal, az hibajelzes.

    random_uniform  - tiszta i.i.d. zaj a [min, max] tartomanyban
                      megorzi: csak az amplitudo-tartomanyt
                      roncsol: marginalis eloszlast IS, temporalist IS
                      Pure null - perplexity az alphabet meretehez kell tartson.

    aaft            - Amplitude-Adjusted Fourier Transform (Theiler 1992)
                      megorzi: marginalis amplitudo-eloszlas + power spectrum
                      roncsolja: nem-linearis fazis-osszefuggeseket
                      Ez a LEGSZIGORUBB kontroll - ha a pipeline meg ezt is
                      megkulonbozteti az eredetitol, akkor a jelen valos,
                      nem-linearis temporalis grammatika van.

Hivatkozas: Theiler, J., et al. (1992). Testing for nonlinearity in time series:
the method of surrogate data. Physica D 58, 77-94.
"""

import numpy as np


def shuffled(signal, rng):
    """Random permutacio. Marginalis eloszlas megmarad, idosor szet van zilalva."""
    out = signal.copy()
    rng.shuffle(out)
    return out


def random_uniform(signal, rng):
    """i.i.d. uniform a [min, max] tartomanyban."""
    return rng.uniform(low=float(np.min(signal)),
                       high=float(np.max(signal)),
                       size=signal.shape).astype(signal.dtype)


def aaft(signal, rng):
    """
    Amplitude-Adjusted Fourier Transform surrogate.
    Megorzi a marginalis eloszlast es a power spectrumot, roncsolja a
    nem-linearis fazis-osszefuggeseket.

    Algoritmus (Theiler et al. 1992):
      1. Sorrendezzuk az eredeti jelet (rangok).
      2. Generaljunk Gaussian zajt es illesszuk az eredeti jel ranghoz
         (rank-matching: ugyanaz a sorrend mint az eredetinek).
      3. FFT, randomizaljuk a fazisokat (egyenletes a [-pi, pi]-n).
      4. Inverz FFT.
      5. Vissza-illesszuk az eredeti amplitudo-eloszlast (rank-matching ujra).
    """
    n = len(signal)
    # 1-2. Eredeti es Gaussian rang-egyezteteshez
    sorted_orig = np.sort(signal)
    ranks = np.argsort(np.argsort(signal))

    gauss = rng.standard_normal(n)
    sorted_gauss = np.sort(gauss)
    gauss_matched = sorted_gauss[ranks]

    # 3-4. Random fazisu surrogate a Gaussian-illesztett jelbol
    fft = np.fft.rfft(gauss_matched)
    mags = np.abs(fft)
    # Random fazisok, kivetel a DC es (paros n eseten) a Nyquist-binben
    phases = rng.uniform(-np.pi, np.pi, size=fft.shape)
    phases[0] = 0.0  # DC fazis fix
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist binnek valos kell legyen
    surrogate_gauss = np.fft.irfft(mags * np.exp(1j * phases), n=n)

    # 5. Eredeti amplitudo-eloszlas vissza-illesztese
    surrogate_ranks = np.argsort(np.argsort(surrogate_gauss))
    out = sorted_orig[surrogate_ranks]
    return out.astype(signal.dtype)


SURROGATES = {
    'shuffled':       shuffled,
    'random_uniform': random_uniform,
    'aaft':           aaft,
}


def make_surrogate(kind, signal, seed):
    """
    Egyseges API: kind, eredeti jel, seed -> surrogate jel.
    """
    if kind not in SURROGATES:
        raise ValueError(f"Ismeretlen surrogate: {kind}. "
                         f"Elerheto: {list(SURROGATES.keys())}")
    rng = np.random.default_rng(seed)
    return SURROGATES[kind](signal, rng)


if __name__ == "__main__":
    # Sanity: spectrum/marginal teszt
    rng = np.random.default_rng(0)
    # Tesztjel: kombinalt szinusz + zaj
    t = np.linspace(0, 10, 1000)
    sig = np.sin(2*np.pi*0.5*t) + 0.3*np.sin(2*np.pi*2*t) + 0.1*rng.standard_normal(1000)

    for kind in SURROGATES:
        s = make_surrogate(kind, sig, seed=42)
        print(f"{kind:16s}: mean={s.mean():+.3f} (orig {sig.mean():+.3f}), "
              f"std={s.std():.3f} (orig {sig.std():.3f}), "
              f"min={s.min():+.2f}, max={s.max():+.2f}")
        # AAFT-nek a spektruma kozel legyen az eredetihez
        if kind == 'aaft':
            ps_orig = np.abs(np.fft.rfft(sig))**2
            ps_surr = np.abs(np.fft.rfft(s))**2
            corr = np.corrcoef(ps_orig, ps_surr)[0, 1]
            print(f"                  AAFT spectrum corr w/ orig: {corr:.4f} (var ~1.0)")
