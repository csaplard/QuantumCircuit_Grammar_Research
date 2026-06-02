# Continuity A/B v1 — results summary

**Status:** v1 complete. Headline not supported. Methods-paper scope.
**Pre-registration:** `docs/continuity_ab/PREREG_v1.md`, git tag
`continuity-ab-v1-prereg-r1`.
**Confirmatory run:** `results/continuity_ab/v1_run/` (n=18 cells).

---

## TL;DR

A diagnostic apparatus was designed and pre-registered to distinguish two
behavior-matched architectures for multi-turn LLM continuity (arm A:
memory-reconstructed via summary + re-prefill; arm B: KV-cache state
carry across turn boundaries) using a Fisher-trace boundary
discontinuity metric `D_k`. The apparatus passed every pre-checks (Gate
0 turn-1 bitwise equivalence, fp32 CPU determinism on 18/18 cells), and
the structured null model behaved as designed. The pre-registered
headline test on n=18 confirmatory data did **not** support the
hypothesis: the two arms do not differ in `D_k` at any pre-registered
threshold. A small smoke-r1 trend (Cliff's δ = −0.111, n=3) collapsed to
near-zero (δ = −0.006) at proper sample size, exemplifying the false-
positive class that the pre-registration apparatus exists to filter.

The v1 verdict: **the metric in this form does not discriminate the two
mechanisms.** An unexpected and unregistered observation — an opposite-
direction arm-asymmetry on Markov-3 lexical surrogate prompts — is
reported as a candidate v2 hypothesis but is not a v1 finding.

---

## What was tested

**Model:** Qwen/Qwen2.5-1.5B-Instruct, fp32, CPU.
**Cells:** 3 regimes (factual, mathematical, creative) × 2 prompts/regime
× 3 seeds {17, 42, 1337} = 18 cells.
**Per cell:** 4 turns, 192 tokens/assistant turn, two arms (A: napló,
B: sodrás), JSD horizon = 20 turn-2 positions.
**Metric:** `D_k = |F_prefill_end[k+1] − F_last_gen[k]| / (1.4826 ×
MAD_gen(F))`, mid-layer (idx 14 of 28) last-token hidden state, sliding
window W=32. Per-session `D = mean_k D_k`. Pre-reg locked operational
definitions; see PREREG_v1.md §"Operational definitions".

**Null models:**
- Markov-3 lexical surrogate on the user-prompt corpus (~24 prompts);
  same conversation shape, recombinant prompts only from the real
  lexicon.
- AR(1) structure-respecting surrogate of the Fisher trace (session-
  pooled fit on gen phase only, independent marginal draws at each
  `prefill_end` event).

---

## The five headline checks at n=18

| Check | Pass criterion | Observed | Passed |
|---|---|---|---|
| `behavior_equivalence` | ≥12/18 cells in JSD band (mean≤0.05, max≤0.15 nats) | 9/18 | ✗ |
| `paired_significance` | Wilcoxon two-sided p < 0.01 on `D_A − D_B` | p = 0.734 | ✗ |
| `effect_size` | `|Cliff's δ|` ≥ 0.5 | 0.006 | ✗ |
| `markov_null_shrink` | `|D_A − D_B|` shrinks ≥ 50% on Markov vs real | shrink = −1.81 (gap GREW) | ✗ |
| `ar1_null_nonsig` | Paired Wilcoxon p ≥ 0.05 on AR(1) surrogate | p = 0.766 | ✓ |

`headline_supported: false` (1 of 5 checks passes, and that one is a null-
model integrity check, not a discrimination check).

Per-arm medians:

|  | real | Markov null | AR(1) null |
|---|---|---|---|
| `median_D_A` | 0.211 | 0.214 | 1.616 |
| `median_D_B` | 0.231 | 0.158 | 1.559 |
| Cliff's δ | −0.006 | **+0.333** | +0.056 |
| Wilcoxon p | 0.734 | **0.043** | 0.766 |

---

## The discipline arc (and why it matters for the conclusion)

The v1 pre-registration locked the operational definitions only after
three smoke iterations, each of which surfaced a specific failure mode
and prevented a specific false-positive class:

**Smoke r1 (apparatus debug).** The Qwen2.5 chat template in
`transformers ≥ 5.x` auto-inserts a default system message on every
`apply_chat_template` call, even with no system role in the input. For
arm A this is harmless (it re-prefills the whole conversation each turn,
so the system block re-enters identically). For arm B it would have
duplicated the system block into the carried KV-cache on every turn
boundary, producing a boundary anomaly attributable to template
duplication rather than to state continuity. The arm B continuation
renderer was patched to slice off the auto-inserted system block.
*Without this fix, the metric would have measured template-duplication
artifact, not anything about continuity.*

**Smoke r2 (null-model defect 1).** The originally-designed AR(1) null
was uniform across the entire trace (gen and prefill_end alike). This
made the surrogate structure-blind to the `prefill_end` events: at each
boundary, the surrogate produced an AR(1) step whose magnitude was
governed by per-arm fit parameters, inflating the synthesized null D in
arm-asymmetric ways for reasons that had nothing to do with the
no-continuity hypothesis. Replaced with a structure-respecting variant
(independent marginal draws at `prefill_end`).

**Smoke r3 (null-model defect 2).** The structure-respecting variant
still over-fit AR(1) per turn on short segments (a 15-event turn-2
segment in one cell produced `a ≈ 0.93`), inflating synthesized null D
for cells with short B-arm generation. This produced a tempting per-arm
asymmetric signature (`delta_A` ≈ 0, `delta_B` ≪ 0) on smoke n=3, which,
read post-hoc, would have re-framed the headline as a per-arm
"continuity vs reset" test. Recognizing this as test-shopping on the
data that surfaced the signal, the change was made one level deeper
(session-pooled AR(1) fit, eliminating small-sample over-fit), and the
headline test architecture was **not** rewritten. After stabilization,
`delta_A` flipped sign from +0.09 to −0.32 and the apparent per-arm
asymmetry vanished — confirming the smoke trend was a fit artifact.

**Confirmatory n=18.** The pre-reg was tag-locked
(`continuity-ab-v1-prereg-r1`) before the n=18 confirmatory run. On
n=18, the small smoke r1 arm-difference (δ = −0.111) collapsed to
δ = −0.006. The metric does not discriminate the arms.

The disciplined sequence is the methods-paper finding: an apparatus
that, in three separate places, would have produced a beautiful and
publishable false positive if the validation step had been skipped or
the test redesigned on the discovery data. The apparatus did its job
precisely because none of those rescues were performed.

---

## What v1 reports (defensibly)

1. **The apparatus is reproducible and the turn-1 equivalence gate is
   exact.** 18 of 18 cells have bitwise-identical generated tokens on
   turn 1 between the two arms (`tok_match: true` everywhere) and
   `fisher_max_abs_dev: 0.0` on the Fisher trace (fp32 CPU). The
   `turn1_jsd_sanity.max_over_cells = 0.0` — the next-token distribution
   is identical to floating-point precision. The design assumption
   (arms diverge only from turn 2 onward) holds at numerical precision.

2. **Behavioral equivalence is partial.** 9 of 18 cells fall in the
   pre-registered JSD equivalence band on turn 2 (mean ≤ 0.05 AND
   max ≤ 0.15 nats over the first 20 generated positions). The two
   architectures produce measurably different generated text in half
   the cells. This is below the pre-registered 12/18 threshold, so the
   "behavior-matched" precondition for the headline finding is not met
   at v1 parameters.

3. **The metric does not discriminate the arms.** Paired Wilcoxon
   p = 0.73, Cliff's δ = −0.006, abs effect size 0.006. The two arms
   are indistinguishable on `D_k` at n=18 power.

4. **The structured AR(1) null behaves correctly at n=18.** Arm-
   symmetric (δ = +0.056, p = 0.77) with both arms centered around
   `median_D_null ≈ 1.6`. The smoke r3 small-n asymmetry was a sample-
   size artifact, as the fit-stabilization predicted.

5. **The Markov-3 lexical surrogate produces an unexpected, opposite-
   direction arm asymmetry.** On real prompts arms are indistinguishable
   (δ = −0.006). On Markov surrogates `D_A > D_B`, δ = +0.333, p = 0.043.
   This is **not** a pre-registered test direction, and the p-value is
   above the pre-registered threshold for hypothesis tests (0.01) in
   any case. It is reported as observed, not claimed as a finding.

---

## What v1 cannot claim

- **No support for "state-carry produces measurably more boundary
  continuity than memory-reconstruction".** The metric in this form
  does not separate the two mechanisms.
- **No support for "the metric measures continuity".** Both arms have
  real D substantially smaller than the AR(1) null D (ratio ~0.13–0.15),
  but this is a non-pre-registered descriptive observation, not a
  hypothesis test. Whatever the metric detects, it does so symmetrically
  across arms.
- **No support for "Markov-3 reveals a hidden mechanism difference".**
  The Markov asymmetry was not pre-registered as a discrimination test.
  It is a candidate v2 hypothesis, not a v1 finding.

---

## v2 directions

Three concrete, narrowly-scoped follow-ups suggested by the v1 data:

1. **Robust metric.** `D_k`'s `MAD_gen` normalizer is sensitive to
   per-session gen-phase length and marginal-std. v2 should test a
   pooled reference normalizer (MAD computed across all cells of the
   real run, used as a fixed scale for both arms and both nulls), or a
   length-invariant alternative such as a z-score normalized boundary
   step relative to the within-arm step-to-step distribution.

2. **Layer-wise sweep.** v1 used a single mid-layer (idx 14 of 28). The
   discrimination signal may live at a different depth — early layers
   for input-encoding continuity, late layers for output-decoding
   commitment. Per-layer Fisher trace at all 28 layers, with
   Bonferroni-corrected layer-by-layer test, is a natural v2 metric.

3. **Markov-asymmetry as v2 hypothesis.** The opposite-direction
   asymmetry on incoherent prompts (`D_A > D_B` on Markov, vs no
   difference on real) suggests the candidate mechanism *the two arms
   handle incoherent input differently*, not *the two arms handle
   coherent input differently*. v2 can pre-register this as the primary
   test, with a properly-designed semantic null (external corpus,
   stronger word-level recombination, length-matched distractor
   prompts).

---

## Reproducibility

- Pre-registration: `docs/continuity_ab/PREREG_v1.md`, git tag
  `continuity-ab-v1-prereg-r1`.
- Code: `code/continuity_ab/{continuity_ab_poc,analyze_continuity_ab,
  continuity_ab_null}.py`.
- Data:
  - Real run: `results/continuity_ab/v1_run/` (36 sessions, 18
    JSD logits files, manifest, sessions.csv, decision.json).
  - Markov null: `results/continuity_ab/v1_run_markov/`.
  - AR(1) null: `results/continuity_ab/v1_run_ar1/`.
- Confirmatory run command: see PREREG_v1.md §"What v1 actually claims";
  exact parameters in `v1_run/manifest.json`.

Apparatus run time on a single CPU thread (Qwen2.5-1.5B-Instruct fp32):
~2:50 h for the real run, ~3:00 h for the Markov run. AR(1) surrogate
and analysis: seconds.

---

## Acknowledgement of process

This summary is itself part of the methods-paper finding. The three
smoke-driven null-model strengthenings (r1, r2, r3) and the explicit
withdrawal of a post-hoc per-arm test redesign are documented in the
pre-registration for replication. The v1 headline result was generated
by an apparatus that, at three separate points, would have produced a
publishable false positive if shortcut. It did not, because the
discipline was applied. That is what the v1 publishes.
