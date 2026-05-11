# Cross-Substrate Sequential-Grammar Pipeline Recovers Alpha-Suppression Topography in Human EEG: Convergent Validation on Mental Arithmetic Recordings

Continuation of Grammar Fingerprinting (doi:10.5281/zenodo.19158088), Fisher Information Threshold Study (doi:10.5281/zenodo.19394880), LLM Grammar Fingerprinting (doi:10.5281/zenodo.19461103), Curvature at the Fisher Threshold (Research Note, April 2026), Exponential Relaxation of Fisher Path Speed (doi:10.5281/zenodo.19519454)

Daniel Csaplár

Independent Researcher, Kazincbarcika, Hungary
ORCID: 0009-0000-7362-7232

May 2026

## Abstract

We test whether the substrate-agnostic sequential-grammar pipeline previously validated on superconducting quantum-processor readouts and on large-language-model logit-entropy traces also recovers state-discriminative information from human EEG. Using an unmodified pipeline (SAX alphabet K=7, LSTM hidden_dim=16, seq_len=20, 50 epochs), we analyze 36 healthy volunteers' eyes-closed rest versus first-minute mental arithmetic recordings (19-channel 10-20 montage; PhysioNet eegmat 1.0.0). Per-stream grammar perplexity was higher during arithmetic than rest in 87.3% of (subject, channel) pairs (Wilcoxon p = 9.1 × 10⁻⁸³, Cohen's d_paired = 0.99); all 19 channels survived Benjamini-Hochberg correction at q < 0.01. The spatial distribution of the effect — strongest at posterior, parietal, and midline electrodes; weakest at frontopolar — recapitulates the classical Berger-Klimesch alpha-suppression topography established by spectral analysis, despite the metric making no spectral assumption. Leave-one-subject-out classification of regime from the per-channel fingerprint matrices achieved 75.2% ± 10.6% accuracy across 36 folds (chance = 50%). Comparison with phase-randomized AAFT surrogates revealed no regime-specific nonlinear temporal grammar (regime × surrogate interaction p = 0.47): a small residual nonlinearity beyond linear power-spectral content was detected at rest (Δ = +0.022, d = 0.12, p = 1.6 × 10⁻⁴), consistent with phase-coupled alpha rhythm structure, and was absent during arithmetic (Δ = +0.002, d = 0.006, p = 0.17). The rest-arithmetic perplexity discrimination is therefore mediated by linear power-spectral content re-represented as sequential symbolic grammar — providing convergent validation that the same pipeline used across two prior substrates (quantum and linguistic) recovers substrate-appropriate, neurophysiologically interpretable state distinctions in human EEG.

## 1. Introduction

The Grammar Fingerprinting methodology (Csaplár, 2026a) established that quantum processor readout sequences carry statistically learnable temporal structure, detectable via the Fisher information trace of a SAX-discretized, LSTM-learned transition matrix. The Fisher Information Threshold Study (Csaplár, 2026b) showed that a data-length threshold N* exists below which no reliable structure is detectable, and that this threshold is consistent across 28 Sycamore readout configurations. The LLM extension (Csaplár, 2026c) demonstrated domain agnosticism: the same pipeline detects temporal grammar in large language model entropy series. The Exponential Relaxation note (Csaplár, 2026e) characterized the geometric dynamics of the learned transition matrices on the Fisher–Rao manifold.

A natural question is whether this cross-substrate consistency is incidental to the two substrates examined or whether it reflects a genuinely general property of complex stochastic systems. Among possible test cases, neuroelectrical recordings of the human cortex (EEG) are an attractive third substrate: they share with quantum readouts the property of being long, noisy, and produced by a system whose dynamics mix deterministic generators with strong stochastic forcing; they share with LLM logits the property of carrying state information at the level of moment-to-moment temporal sequencing rather than only in the amplitude marginal. They also have a 90-year body of established spectral landmarks — most notably the Berger-Klimesch alpha-suppression phenomenon — that any new metric can be benchmarked against.

In this work we apply the unchanged grammar-perplexity pipeline to an open EEG dataset of 36 healthy volunteers performing closed-eye serial mental arithmetic versus eyes-closed rest. We ask three concrete questions:

1. **Discrimination.** Does grammar perplexity, computed identically to the quantum and LLM applications, distinguish the resting and task-engaged regimes at the single-recording level?
2. **Convergence.** Does the spatial distribution of the perplexity difference, across the 19-electrode 10–20 montage, recapitulate the posterior alpha-suppression topography established by spectral analysis?
3. **Generalization.** Does the resulting per-channel grammar fingerprint support out-of-sample, between-subject classification of regime?

We additionally test whether the perplexity difference between regimes exceeds what a phase-randomized linear surrogate (AAFT; Theiler et al. 1992) of the same signal would produce — a question that probes whether cognitive task engagement induces nonlinear temporal grammar that lies outside the power-spectrum content.

The motivation is methodological rather than neurophysiological. The existence of alpha suppression during cognitive engagement is established beyond doubt. The contribution here is to demonstrate that the alpha-suppression topography is recovered by a pipeline that knows nothing about frequency, amplitude distribution, or any EEG-specific construct, and that this same pipeline has previously yielded equally interpretable signatures on quantum and linguistic substrates.

## 2. Methods

### 2.1 Dataset.

We analyzed the publicly available EEG During Mental Arithmetic Tasks dataset (Zyma et al. 2019; PhysioNet eegmat 1.0.0), comprising recordings from 36 healthy adult volunteers (ages 17–26). Each subject completed two sessions: a 3-minute eyes-closed resting baseline (`Subject*_1.edf`) and the first 60 seconds of an eyes-closed serial subtraction task (`Subject*_2.edf`), in which subjects mentally subtracted a two-digit number from a four-digit minuend in continuous fashion. EEG was recorded at 500 Hz from 19 scalp electrodes positioned according to the international 10–20 system (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2), with linked-ear reference. The combined linked-ear reference channel and the ECG channel were excluded from analysis. No subjects were excluded; all 36 are analyzed.

### 2.2 Preprocessing.

Preprocessing was performed with MNE-Python (v1.12) and consisted of: (i) notch filter at 50 Hz; (ii) zero-phase FIR band-pass filter at 1–40 Hz (Hamming window, passband ripple 0.0194 dB, stopband attenuation 53 dB); (iii) down-sampling to 100 Hz (anti-alias-protected by the band-limit at 40 Hz). The downsampling step normalizes the per-symbol time scale to ~10 ms, placing the LSTM context window (`seq_len=20`) at ~200 ms, which spans approximately two alpha cycles and one theta cycle — i.e., the temporal scale of working-memory neural rhythms. No artifact rejection (ICA or otherwise) was applied; eyes-closed recording suppresses blinks, and frontopolar artifact contamination is reported with the rest of the channels.

### 2.3 Grammar fingerprinting pipeline.

The pipeline is identical to that previously applied to Sycamore readouts (Csaplár 2026a, 2026b) and LLM entropy (Csaplár 2026c). For each `(subject, regime, channel)` triplet the 1-D signal is processed in four stages:

**SAX encoding.** The signal is z-score normalized and discretized into K=7 symbols using empirical-quantile breakpoints, enforcing equiprobable symbol marginals. The empirical-quantile choice deliberately discards amplitude-distribution information to isolate temporal sequence grammar, consistent with cross-substrate methodology. Each EEG sample maps to one symbol.

**LSTM next-symbol prediction.** A pure-NumPy character-level LSTM (hidden dim 16, Adam optimizer, lr = 0.01, 50 epochs, seq_len = 20, gradient clip ±5, forget-gate bias 1) is trained on an 80/20 train/validation split of the symbol sequence.

**Validation perplexity.** Per-stream perplexity is computed on the held-out validation set as exp(mean cross-entropy loss). With K = 7 the upper bound (uniform random sequence) is 7.

**Grammar fingerprint.** A K×K transition probability matrix P(next | current) is extracted by averaging the LSTM's predicted next-symbol distributions over the validation set, conditioned on the observed current symbol.

### 2.4 Surrogate controls.

We employ three null surrogates per stream (Theiler et al. 1992): shuffled (random permutation; preserves marginal, destroys temporal structure); random_uniform (i.i.d. uniform in min–max range); and Amplitude-Adjusted Fourier Transform (AAFT; rank-matched Gaussian surrogate with phase-randomized Fourier reconstruction back-mapped to the original amplitude distribution; preserves marginal and linear power spectrum, destroys nonlinear phase coupling). The AAFT surrogate is the discriminating control: any perplexity gap between the real signal and its AAFT surrogate quantifies nonlinear temporal grammar that lies outside spectral content.

### 2.5 Statistical analysis.

The unit of analysis is the (subject, channel, regime) triplet, yielding 36 × 19 = 684 paired (rest, arithmetic) observations. Per-channel paired test: Wilcoxon signed-rank, Benjamini–Hochberg FDR correction across 19 channels. Global paired test: Wilcoxon signed-rank across all 684 pairs. Effect size: Cohen's d for paired samples, d = mean(diff) / sd(diff). For surrogate comparisons the same paired structure is used with the surrogate replacing the rest baseline; the regime × surrogate interaction is tested by Mann-Whitney U on the per-pair AAFT-real deltas, comparing rest deltas to arithmetic deltas.

### 2.6 Generalization analysis.

We performed leave-one-subject-out (LOSO) classification of regime from the per-stream fingerprint matrix. The flattened K×K fingerprint (49-dimensional feature vector) was used as input to L2-regularized logistic regression (C = 1, scikit-learn 1.x). Features were standardized using training-fold statistics only. We report mean ± SD accuracy across 36 LOSO folds and the pooled confusion matrix.

## 3. Results

### 3.1 Grammar perplexity discriminates rest from mental arithmetic.

Across 36 subjects and 19 EEG channels (684 paired observations), grammar perplexity was significantly higher during mental arithmetic than during eyes-closed rest (Wilcoxon W = 1.75 × 10⁴, p = 9.14 × 10⁻⁸³; mean Δ = +0.65 perplexity units; Cohen's d_paired = 0.99). The shift was unidirectional in 597/684 pairs (87.3%; **Figure 1A,B**). Of the 19 channels analyzed individually, all 19 survived FDR correction at q < 0.01 (per-channel paired Wilcoxon tests across 36 subjects per channel). The most strongly affected channels (Cohen's d > 1.5) were the posterior alpha generators O1, O2, Pz, P4, and the midline electrode Cz; the least affected (but still significant) were the frontopolar electrodes Fp1 and Fp2 (d ≈ 0.55, p_FDR < 0.01).

**Figure 1: Main effect.** (A) Box plots of per-stream perplexity for the 684 (subject, channel) streams in each regime. Median perplexity is 4.01 at rest and 4.63 during arithmetic (means 4.05 and 4.69, respectively). (B) Per-stream pair scatter plot. Each point is one (subject, channel) pair; 87.3% lie above the y = x diagonal.

### 3.2 Topographic pattern recapitulates classical alpha suppression.

The spatial distribution of the perplexity difference is shown in **Figure 2**. The effect-size gradient runs from posterior–parietal–midline maxima (O1, O2, Pz, P4, Cz, T6; d > 1.4) to frontopolar minima (Fp1, Fp2; d ≈ 0.55), recapitulating the topography of task-related alpha suppression described by Berger and quantified spectrally by Klimesch (1999). This convergence is notable because the grammar-perplexity metric makes no spectral assumption: it operates exclusively on the symbol-transition statistics of the temporally discretized signal. The reproduction of classical alpha-suppression topography from a substrate-agnostic information-theoretic measure constitutes convergent validity with established EEG methodology.

The mechanistic interpretation is direct (**Figure 3**). The largest diff-fingerprint cell is the mid-amplitude self-transition P(d → d), which decreases from rest to arithmetic by 0.030 (rest = 0.30, arithmetic = 0.27). Resting alpha — a sustained, sinusoidal mid-amplitude oscillation — keeps the discretized signal in the central symbol class for many consecutive samples, producing high P(d → d). Task-related alpha suppression breaks this local persistence, scattering the signal into non-central symbol classes and reducing the diagonal mass.

**Figure 2: Topography.** (A) Topographic delta map (arithmetic − rest) interpolated across the 10-20 montage. All 19 electrodes show positive difference; the maximum spans the posterior–parietal–midline ring, the minimum is at frontopolar sites. (B) Per-channel mean delta with SEM, sorted.

**Figure 3: Average fingerprints.** (A) Average 7×7 grammar fingerprint matrix during eyes-closed rest. Rows: current SAX symbol; columns: predicted next SAX symbol; cell value: P(next | current). (B) Average fingerprint during mental arithmetic. (C) Difference matrix (arithmetic − rest); the dominant change is reduced self-persistence at the central symbol P(d → d).

### 3.3 Across-subject generalization.

For each held-out subject, an L2-regularized logistic regression was trained on the flattened 7×7 fingerprint matrices of the remaining 35 subjects (1,330 streams) and tested on the held-out subject's 38 streams (19 channels × 2 regimes). Mean LOSO accuracy was 75.2% ± 10.6% (chance = 50%; **Figure 4A**). Of 36 folds, 33 (92%) exceeded chance level; the modal accuracy was 75–85%. The pooled confusion matrix (**Figure 4B**) was symmetric: 76.3% of true-rest streams correctly classified, 74.1% of true-arithmetic streams correctly classified, indicating no systematic class bias. The above-chance LOSO accuracy demonstrates that the grammar-fingerprint pattern is sufficiently consistent across individuals to support out-of-sample classification — the metric captures a population-level signature, not subject-specific idiosyncrasies.

**Figure 4: Generalization.** (A) Distribution of leave-one-subject-out classification accuracies across the 36 folds. Mean = 75.2% ± 10.6% (chance = 50%). (B) Pooled confusion matrix across all folds, row-normalized.

### 3.4 No regime-specific nonlinear temporal grammar beyond linear spectral content.

We compared each (subject, channel, regime) real-signal perplexity to that of an AAFT surrogate of the same signal. AAFT preserves both the marginal amplitude distribution and the linear power spectrum but destroys nonlinear phase relationships. A regime × surrogate interaction would indicate regime-specific nonlinearity invisible to spectral analysis.

The full results across 36 subjects (n = 684 paired streams per regime) are summarized in **Table 1**.

**Table 1: AAFT surrogate vs real perplexity, per regime.**

| Regime | n | Mean Δ (AAFT − real) | SD | Cohen's d | Wilcoxon p |
|--------|---|---------------------|-----|-----------|------------|
| Rest | 684 | +0.022 | 0.176 | 0.123 | 1.6 × 10⁻⁴ |
| Arithmetic | 684 | +0.002 | 0.405 | 0.006 | 0.17 |

The rest-vs-arithmetic interaction was non-significant (Mann-Whitney one-sided "arith > rest" p = 0.235; two-sided p = 0.47), and the directional ordering was the opposite of an a-priori hypothesis: the residual nonlinearity was instead larger at rest. We observed (i) at rest, a small but significant nonlinear residual (d = 0.12, p = 1.6 × 10⁻⁴), consistent with phase-coupled alpha rhythm structure that AAFT phase randomization disrupts; (ii) during arithmetic, no detectable nonlinear residual (d = 0.006, p = 0.17), consistent with task-related alpha suppression yielding a more noise-like signal that AAFT reproduces accurately. We therefore interpret the rest-arithmetic perplexity discrimination as a sequential-symbolic representation of linear spectral content rather than as detection of a regime-specific nonlinear cognitive signature (**Figure 5**).

**Figure 5: AAFT surrogate test.** (A) Boxplots of per-stream AAFT − real perplexity delta for rest and arithmetic. (B) Real vs AAFT perplexity scatter by regime; the cluster sits on the y = x diagonal in both regimes. (C) Per-channel mean AAFT − real delta by regime.

## 4. Discussion

The principal interpretive observation is the convergence between the grammar-perplexity topography and the spectral alpha-suppression literature. Our pipeline operates on no spectral construct — there is no notion of frequency band, oscillation, or amplitude in the metric itself — yet the spatial distribution of the perplexity difference recovers the topographic signature established by Berger and quantified by Klimesch (1999) and successors. The strongest perplexity differences are at posterior, midline, and parietal sites where the thalamocortical alpha generators produce the most prominent rhythmic activity at rest; the weakest are at frontopolar sites where alpha is intrinsically less prominent. The mechanistic basis is visible in the average grammar fingerprints (Figure 3): at rest the central symbol class shows high self-persistence, reflecting the sustained mid-amplitude oscillation of the alpha rhythm; during arithmetic this self-persistence drops, scattering symbol transitions to non-central classes.

The same pipeline, with parameters fixed across substrates (K = 7, hidden = 16, seq_len = 20, epochs = 50), has previously yielded substrate-appropriate state-discrimination on superconducting quantum-processor readouts (Csaplár 2026a, 2026b) and large-language-model logit-entropy traces (Csaplár 2026c). The present EEG result extends the applicability of the pipeline to a third, independent class of stochastic temporal data. The convergence of the EEG result with existing electrophysiological knowledge supports the use of grammar perplexity as a generic, parameter-free diagnostic for state inference in temporally extended stochastic systems.

The non-significant regime × surrogate interaction means that the grammar-perplexity discrimination of rest from arithmetic is not driven by nonlinear temporal information beyond the linear power spectrum. It does not mean that the grammar metric is uninformative. Two observations qualify the null. First, the metric provides a single substrate-agnostic scalar that has demonstrated successful application across quantum, linguistic, and now neuroelectric data with no per-substrate tuning; spectral methods do not transfer in this manner. A symbolic-grammar reformulation of spectral content is therefore methodologically non-trivial when the goal is cross-substrate consistency. Second, the small nonlinear residual at rest (d = 0.12, p = 1.6 × 10⁻⁴) indicates that the metric is sensitive to phase-coherent oscillatory structure that AAFT disrupts; arithmetic abolishes this residual, consistent with alpha desynchronization. The metric thus tracks not only the gross amplitude/spectral suppression but also a measurable change in phase coherence between regimes — a property that warrants follow-up with finer-grained surrogate methods (iterated AAFT, twin surrogates) in future work.

Three limitations bear on interpretation. First, the dataset's arithmetic recordings are restricted to 60 seconds (vs. 180 seconds for rest), producing asymmetric data lengths; while the per-stream quantile-SAX and 80/20 train/val split are independent of stream length and the LSTM converges within 50 epochs in both regimes, the arithmetic fingerprint matrices are estimated from fewer transitions and are correspondingly noisier. Second, no artifact rejection (ICA or otherwise) was performed; eyes-closed recording suppresses blinks but does not eliminate muscle or skin-potential artifacts, particularly at frontopolar sites, where our smallest (though still significant) effects are observed. Third, our LOSO classifier is a flat logistic regression on the flattened fingerprint vector; richer architectures (graph-aware, channel-spatial) may achieve higher accuracy but were not pursued here for parsimony.

Three follow-ups would extend the present results. (i) Replication on the good-counter / bad-counter behavioral subgroups (eegmat metadata) would test whether grammar perplexity tracks behavioral performance, beyond regime label. (ii) Application to other cognitive tasks (working memory n-back, attention oddball) would test the generality of the grammar-EEG-task triplet beyond mental arithmetic. (iii) Within-substrate comparison with finer-grained nonlinear surrogates (iterated AAFT, twin surrogates, pseudo-periodic surrogates) would localize whether the small alpha phase-coupling residual we observed at rest is captured by these methods or persists.

## 5. Summary

1. The substrate-agnostic sequential-grammar pipeline (SAX K=7 + LSTM perplexity) discriminates eyes-closed rest from mental arithmetic in human EEG with Cohen's d_paired = 0.99 (Wilcoxon p = 9.1 × 10⁻⁸³, n = 684 paired observations across 36 subjects and 19 channels).

2. The topographic distribution of the regime-difference effect recapitulates the classical posterior alpha-suppression pattern (Berger, Klimesch 1999), with effect-size gradient from posterior–parietal–midline (d > 1.4) to frontopolar (d ≈ 0.55).

3. Leave-one-subject-out classification of regime from per-stream fingerprint matrices generalizes to held-out subjects at 75.2% ± 10.6% accuracy across 36 folds, with symmetric per-class performance.

4. AAFT surrogate analysis indicates that the rest-arithmetic discrimination is mediated by linear power-spectral content re-represented as sequential symbolic grammar; the regime × surrogate interaction was non-significant (p = 0.47). A small residual nonlinearity at rest (d = 0.12) is consistent with phase-coupled alpha rhythm structure; no nonlinear residual is detected during arithmetic.

5. The results constitute convergent validation that the cross-substrate sequential-grammar pipeline used previously on Sycamore quantum readouts and on LLM entropy series recovers substrate-appropriate, neurophysiologically interpretable state distinctions in human EEG.

## References

- Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. Archiv für Psychiatrie und Nervenkrankheiten, 87, 527–570.
- Csaplár, D. (2026a). Grammar Fingerprinting of Quantum Processor Topology. Zenodo. doi:10.5281/zenodo.19158088.
- Csaplár, D. (2026b). Fisher Information Threshold Study: Grammar Fingerprinting on Google Sycamore. Zenodo. doi:10.5281/zenodo.19394880.
- Csaplár, D. (2026c). Grammar Fingerprinting and Fisher Information Thresholds in Large Language Model Entropy Series. Zenodo. doi:10.5281/zenodo.19461103.
- Csaplár, D. (2026d). Curvature Structure at the Fisher Information Threshold. Research Note (April 2026).
- Csaplár, D. (2026e). Exponential Relaxation of Fisher Path Speed on the Statistical Manifold of Learned Transition Matrices. Zenodo. doi:10.5281/zenodo.19519454.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267.
- Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. Brain Research Reviews, 29, 169–195.
- Lin, J., Keogh, E., Wei, L. & Lonardi, S. (2007). Experiencing SAX: a novel symbolic representation of time series. Data Mining and Knowledge Discovery, 15(2), 107–144.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. Physica D, 58, 77–94.
- Zyma, I., Tukaev, S., Seleznov, I., Kiyono, K., Popov, A., Chernykh, M. & Shpenkov, O. (2019). Electroencephalograms during Mental Arithmetic Task Performance. Data, 4(1), 14.

## Data and Code Availability

EEG dataset: PhysioNet eegmat 1.0.0 (Zyma et al. 2019, doi:10.13026/C2JQ1P), publicly available at https://physionet.org/content/eegmat/1.0.0/. EEG-grammar pipeline source code (SAX encoder, LSTM trainer, surrogate generators, MNE preprocessing, statistical analysis, figure generation): https://github.com/csaplard/QuantumCircuit_Grammar_Research. All per-stream perplexities, fingerprint matrices (CSV + NPZ), per-subject metadata, statistical analysis outputs, and the figures of this preprint are maintained in the project repository.

— End of preprint —
