# Continuity A/B — pre-registration v1 (LOCK CANDIDATE after smoke r3)

**Status:** LOCK CANDIDATE. Three smoke iterations were run before lock and
each surfaced a specific operational decision that was made before this
version: smoke r1 (chain validation, render-template fix for Qwen2.5
auto-system-message), smoke r2 (naive AR(1) replaced with structure-
respecting variant), smoke r3 (per-turn AR(1) replaced with session-pooled
stabilization after the per-turn fit was shown to over-fit on short
segments). Each step strengthened the null without re-shaping the headline
test. The per-arm (real-vs-null) exploratory architecture was considered
and explicitly **withdrawn** as post-hoc test-shopping after smoke r2/r3.

This version is to be git-tagged as `continuity-ab-v1-prereg-r1` immediately
before the (B) scaled confirmatory run. Once tagged, no parameter or
decision rule changes; any further smoke findings go to v2.

---

## Question

Can an internal-state metric (sliding-window Fisher trace on a transformer
decoder layer's last-token hidden state) distinguish two architectures that are
behaviorally near-equivalent but differ in how multi-turn continuity is
implemented:

- **A — "napló" (journal / reconstruction).** Each turn starts from a fresh
  model state; prior turns enter only via a textual summary prepended to the
  next prompt.
- **B — "sodrás" (drift / state carry).** `past_key_values` (the KV-cache) is
  carried across turn boundaries; no re-prefill, no textual summary. Only the
  new user turn's tokens are appended.

The hypothesis is that B's Fisher trace is **continuous** across turn
boundaries while A's exhibits a **discontinuity** (reset-like jump), even when
the two arms' generated text is statistically indistinguishable.

This pre-reg fixes the metric, the comparison protocol, the null model, and
the decision rule **before** any production run.

---

## Model and decoding

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (primary). Secondary, optional:
  `Qwen/Qwen2.5-3B-Instruct`. No quantization for v1 (fp16/bf16 on GPU; CPU
  fallback allowed for the 1.5B variant).
- **Decoding:** top-p sampling, `top_p=0.95`, `temperature=0.8`, max 192 new
  tokens per turn. No adaptive feedback (`mode=baseline` semantics from
  `internal_fisher_poc.py`).
- **Chat template:** Qwen2.5 default `apply_chat_template` with `system=None`.
  Both arms use the same template builder so token-level format is matched.

## Fisher monitor

- One forward hook at `model.layers[mid]`, where `mid = n_layers // 2`.
- Sliding window `W = 32`. Metric per step:
  `F_t = mean_{k in window} mean_dim((h_k - h_{k-1})^2)` on the last-token
  hidden state at that step.
- Monitor is **never reset between turns** on either arm. It is reset only
  between **sessions** (different conversation instances).

## Arms

### A — napló
1. Maintain a running plaintext summary `S_k` after each assistant turn `k`.
   For v1, `S_k` is the **last assistant turn truncated to 60 tokens** plus
   `"... [continued]"` if truncated. (Deliberately simple — no learned
   summarizer in v1; that lives in a v2 ablation.)
2. Turn `k+1` prompt:
   `apply_chat_template([{role: "user", content: f"[Prior summary: {S_k}] {U_{k+1}}"}])`.
3. Generation: `past_key_values=None`, full prefill, then greedy/top-p loop.
   FisherMonitor accumulates across the whole prefill + generation; **not
   reset at the boundary**.

### B — sodrás
1. Turn 1: standard prefill + generation as in A.
2. Turn `k+1`: take previous `past_key_values`, tokenize the chat-template
   output for the new user message, then **slice off everything before the
   first `<|im_start|>user\n` marker** (Qwen2.5 IDs `[151644, 872, 198]`).
   This produces the per-turn delta. The slice is necessary because Qwen2.5
   in `transformers >= 5.x` auto-inserts a default system message on every
   `apply_chat_template` call; without slicing, the B arm would duplicate
   the system block into the carried cache and any turn-boundary anomaly
   would be confounded with that duplication, not with state continuity.
   Append the delta by one forward with the carried cache, then token-by-
   token generation.
3. No textual summary, no re-prefill. FisherMonitor continues uninterrupted.

### Behavior equivalence target
- Each session: 4 turns, ~60 user-token prompts, fixed across arms and seeds.
- Behavior similarity reported on **turn 2** (first place the arms can
  diverge): token-level Jensen–Shannon divergence between the two arms' logit
  distributions for the first 20 generated positions, conditioned on each
  arm's own context. **Target band:** mean JSD over the 20 positions ≤ 0.05
  (nats). If exceeded, the arms are not behavior-equivalent and the run is
  flagged; the headline metric is only reported when this band is met on at
  least 8/12 (regime, prompt) cells.

## Operational definitions (LOCKED — these must match the code in
`code/continuity_ab/analyze_continuity_ab.py`; if the code changes, this
section is the canonical record of the v1 definitions and any change
requires a new pre-reg tag).

### Boundary discontinuity `D_k`

For each turn boundary `k → k+1` (i.e. `k ≥ 1`):

```
before = F at the LAST 'gen' event of turn k
after  = F at the 'prefill_end' event of turn k+1
D_k    = |after - before| / (1.4826 * MAD_gen(F) + eps)
```

where `MAD_gen(F)` is the median absolute deviation of `F` over **generation
steps only** in that session (excludes `prefill_end` events, so the
normalizer is not contaminated by the very jumps being measured). The
`1.4826` factor scales MAD to a Gaussian-equivalent standard deviation, so
`D_k` reads approximately as a z-score.

Per-session: `D = mean_{k=1..T-1} D_k`. Turn 0 has no preceding boundary and
contributes nothing to `D`; turn 1 is handled by the dedicated equivalence
gate below.

### Behavior equivalence (JSD)

For each (regime, prompt, seed), at the start of turn 2:
1. Take arm A's generated tokens for turn 2 (call them `x_1, ..., x_N`).
2. Feed `x_1, ..., x_min(N, 20)` through both arms, each conditioned on its
   own pre-turn-2 state (arm A: `past=None` + summary; arm B: carried
   `past_key_values`). Teacher-forced — the same tokens go through both arms
   so logit positions are paired.
3. Read out next-token logit distributions `p_A[t]`, `p_B[t]` at each
   position `t`.
4. `jsd_t = JSD(p_A[t] || p_B[t])` in nats.

Pass band (both must hold): `mean_t jsd_t ≤ 0.05` AND `max_t jsd_t ≤ 0.15`.
The max-band is the stricter check; without it a single highly divergent
position could be hidden by an average over 20 positions.

### Turn-1 equivalence gate (HARD STOP)

By construction, arm A and arm B differ only from turn 2 onward — at turn 1
both arms do the same forward pass with `past_key_values=None`, the same
chat template, the same user text, the same seed, the same RNG state.
Therefore the analysis script **requires**:

- bitwise-identical generated `token_id` sequence on turn 1 for arm A and
  arm B
- `|F_A(t) - F_B(t)| < 1e-5` at every step of turn 1

A failure on this gate means the two arms are not equivalent at turn 1, so
any turn-2+ separation could be an artifact of the differing setup rather
than the carried state. Failure aborts the analysis and the run is
discarded; the script does not proceed to the headline tests.

### Group-level test (real prompts)

Per-arm: median `D` across (regime × prompt × seed), with IQR.

Paired test: Wilcoxon signed-rank on per-session `D_A − D_B`, since pairing
is per (regime, prompt, seed). Direction must be `D_A > D_B`. Effect size:
Cliff's delta on the per-session `D` distributions.

## Null models (structured, not just shuffled)

Both nulls were stress-tested on smoke r1 and re-specified before lock based
on what the validation surfaced.

### 1. Markov-3 prompt surrogate (LEXICAL null — v1 scope)

3-gram word model trained on the concatenated real user prompts from
`continuity_ab_poc.CONVERSATIONS` (~24 prompts, ~200 words). Same
conversation shape as real, recombinant prompts only from the real lexicon.

**Smoke r1 finding (LOCKED):** with this corpus size, the surrogate
prompts retain substantial substrings of real questions (e.g. "the second
derivative of sin(x)?" appears verbatim in a generated prompt), and the
model produces coherent answers. The v1 Markov-3 is therefore a **lexical
null**, not a semantic null. The pre-reg check ("shrink ≥ 50%") tests
whether the metric depends on prompt-level word-coherence; a passing
result means the effect survives lexical reshuffling, NOT that the metric
measures semantic continuity per se. A semantic null with a larger
external corpus is explicit v2 scope.

### 2. AR(1) structured null on Fisher trace (smoke r1 redesign)

The original AR(1) surrogate (uniform AR(1) across the whole trace,
single fit on session-wide gen-only F) was shown by smoke r1 to be
**structure-blind to `prefill_end` events**: it generated the boundary
F-value with the same dynamics as a gen step, inflating boundary jumps
purely as a generator artifact. The reported AR(1) `D_A` vs `D_B` gap
under that surrogate was not a fair baseline.

**Redesigned (v1-locked) AR(1) null — session-pooled stabilization:**
- Fit AR(1) on the gen phase **session-wide**, pooling all gen events
  across turns of one (regime, prompt, seed, arm) cell. This is the v2
  stabilization (smoke r2 → smoke r3): an earlier per-turn fit produced
  `a ≈ 0.93` on a 15-event segment, an unstable estimate that inflated
  the synthesized null `D`. Session-wide pooling removes the small-
  sample over-fit failure mode. Session-pooled fit parameters are saved
  per session in `manifest_ar1.json` for inspection.
- At each `prefill_end` position, sample F **independently from the
  session's gen-phase marginal distribution** (Gaussian with the
  gen-phase empirical mean and std). This models "no boundary
  continuity": the boundary value is independent of the previous gen
  value.
- The first gen event after `prefill_end` starts an AR(1) walk from the
  boundary value, using the session-pooled `(a, b, σ)`.

**Smoke r3 finding on the stabilized null (documented for v1 discussion,
NOT used to redesign the headline test):** After stabilization, the null
remains arm-asymmetric (`null Cliff's δ = −1.0`, ratio `null_D_B /
null_D_A ≈ 2.9×` on n=3). At the same time, both arms show `real_D <
null_D` (i.e. neither arm produces boundary jumps as large as
independent-draw would predict). The earlier per-turn-fit smoke
suggested an "A reset / B continuity" asymmetric signature; this
signature did **not survive** the stabilization (`delta_A` flipped sign
from +0.09 to −0.32). The per-arm exploratory architecture was
therefore **withdrawn** as a candidate headline-test redesign, on the
recognition that proposing it on data where the null defect had been
named was post-hoc test-shopping. The v1 headline test remains the
pre-registered "A vs B not significant on AR(1) surrogate" check, with
the explicit acknowledgement that on the n=18 confirmatory run this
check is unlikely to pass — and that this is an honest description of
the v1 metric's known limitation under arm-asymmetric gen-phase
properties, not a finding to be rescued by post-hoc re-architecture.

**Interpretation under H0 (no continuity):** real `D_k` should be
comparable to surrogate `D_k`. If real `D_k` is **smaller**, the boundary
is being "absorbed" by something — the candidate mechanism is state
continuity. If real `D_k` is **larger**, the boundary is more
discontinuous than independent draw (anti-continuity, also reportable).

**Headline check:** paired A-vs-B on per-session `D` in the surrogate
must not be significant at `p < 0.05`. Rationale: under the no-continuity
null, the two arms should not differ in `D` beyond what their per-arm
scale/normalizer differences imply. A significant arm-difference on the
structured null would mean the real arm-difference is a normalizer/scale
artifact, not a continuity measurement.

## Multi-seed

- Seeds `{17, 42, 1337}` for the primary table. Per-arm `D` reported as
  median over seeds; seed-variance reported.

## Pre-registered decision rule

**Pre-check (gate):** The turn-1 equivalence gate must pass on **all** cells.
A single failure aborts the headline tests entirely; the run is discarded.

### Direction semantics (LOCKED — revised after smoke r1)

The headline claim is "an internal-state metric **separates** the two arms
when behavior does not", not "arm A has more discontinuity than arm B". The
n=3 smoke run produced `cliffs_delta = -0.111` and `median(D_B) > median(D_A)`,
i.e. the empirical sign on smoke is the opposite of the initial intuition.
Locking the headline test to a specific sign would either invalidate a real
effect or require post-hoc justification. Neither is acceptable.

Therefore the headline test is **two-sided**:
- Paired Wilcoxon signed-rank on `D_A − D_B`, two-sided, `p < 0.01`.
- `|Cliff's delta| ≥ 0.5` (large effect, either sign).
- The observed sign of `median(D_A − D_B)` is reported as part of the
  result, with a fixed plain-English interpretation:
    - sign > 0 : "memory-reconstructed arm shows larger boundary
                  discontinuity than state-carry arm"
    - sign < 0 : "state-carry arm shows larger boundary discontinuity than
                  memory-reconstructed arm — possible mechanisms include
                  larger activation excursion when a short user-delta lands
                  on a long carried context"
  Both signs are reportable. Neither is the "expected" direction.

### Headline conditions (all must hold)

- Behavior-equivalence band met on ≥ 8/12 cells (mean ≤ 0.05 AND max ≤ 0.15
  nats over the 20 evaluated turn-2 positions).
- Paired two-sided Wilcoxon on per-session `D_A − D_B`, real prompts:
  `p < 0.01`.
- `|Cliff's delta| ≥ 0.5` on the per-session `D` distributions.
- On Markov-3 surrogate, `|median(D_A) − median(D_B)|` shrinks by ≥ 50%
  relative to real-prompt (absolute gap, sign-invariant).
- On AR(1) surrogate, paired A vs B is **not** significant at `p < 0.05`
  (two-sided).

If any of these fail, the result is reported truthfully as a negative or
partial finding; the script does not re-run with shifted thresholds.

### AR(1) calibration is the specific test for the normalizer artifact

A reasonable counter-explanation for an observed `D_A` vs `D_B` separation
is that the `MAD_gen` normalizer differs systematically between arms (e.g.,
arm B's turn-2 prefill is shorter, so its gen-phase Fisher variance
profile differs). Under this explanation, the separation is a structural
artifact, not a measurement of state continuity.

The AR(1) null is constructed to discriminate against this exactly:
- The AR(1) process is fit **per arm, per session**, on the gen-phase
  Fisher values only.
- The synthesized surrogate trace has the **same length**, the same
  `prefill_end` positions, the same per-arm AR(1) parameters `(a, b, σ)`.
- Therefore any `D_A` vs `D_B` difference that comes purely from length /
  variance / autocorrelation structure is preserved in the surrogate.
- If `D_A` vs `D_B` is still significant under AR(1), the real-data effect
  is a structural artifact and the headline cannot be claimed.
- If `D_A` vs `D_B` is **not** significant under AR(1), the real-data
  effect carries content above first-order autocorrelation — and the
  Markov-3 null then asks whether that content is *semantic continuity*
  vs *lexical-statistics-only*.

This calibration was locked **after** the n=3 smoke surfaced a direction
flip; the test exists to keep the most plausible "trivial" explanation
falsifiable before the scaled run.

## Outputs

- `results/continuity_ab/v1/<run_tag>/{arm}_{regime}_p{idx}_seed{s}_trace.csv`
- `results/continuity_ab/v1/<run_tag>/{arm}_{regime}_p{idx}_seed{s}_output.txt`
- `results/continuity_ab/v1/<run_tag>/summary.csv` — one row per session.
- `results/continuity_ab/v1/<run_tag>/figure_boundary.png` — Fisher trace with
  turn boundaries marked, A vs B overlaid for one example session.
- `results/continuity_ab/v1/<run_tag>/decision.json` — the five pre-registered
  checks with pass/fail booleans and observed values.

## What v1 actually claims (DEFENSIBLE SCOPE — locked after smoke r1)

The v1 headline statement is **deliberately narrower** than the original
framing, because smoke r1 surfaced two null-model limitations that bound
what the v1 data can support:

**Defensible v1 claim, in full:**
> "There exists an apparatus that, starting from a bitwise-identical
> turn-1 forward pass, produces a measurable per-session boundary
> discontinuity score `D_k` that differs between two behavior-matched
> arms (memory-reconstructed vs state-carry). The arm-difference is
> stable across seeds and regimes. Two structured nulls — Markov-3
> lexical reshuffling of prompts, and a structure-respecting AR(1) on
> the Fisher trace — bracket the *mechanism* of the difference: the v1
> Markov null tests lexical (not semantic) dependence, and the v1 AR(1)
> null tests whether the arm-difference survives the absence of
> across-boundary continuity. The v1 does **not** decide between
> 'semantic state continuity' and 'lexical-statistical structure' as the
> source of the effect — that distinction is explicit v2 scope and
> requires a stronger semantic null."

## What this v1 does **not** claim

- Nothing about phenomenal experience or "having a self".
- Nothing about which mechanism is "better".
- No claim that KV-cache continuation is the right substrate for state
  continuity in general — only that for this specific behavior-matched A/B,
  the metric does or does not separate them.
- **No claim that the metric measures semantic continuity.** The v1
  Markov null is too weak to discriminate semantic from lexical structure
  with the local 24-prompt corpus.
- **No claim that "AR(1) cannot reproduce the effect, therefore there is
  continuity".** The v1 AR(1) null is a structured surrogate for the
  no-continuity hypothesis at the boundary; a real `D` smaller than null
  `D` is consistent with continuity but not uniquely explained by it.

## Next iterations (out of scope for v1, listed to bound v1)

- v2: learned summarizer in arm A (closer to real "napló" systems).
- v3: hidden-state-level carrier (α-driver style) as a third arm.
- v4: cross-session transfer test (does the metric still see continuity when
  the cache survives a process restart via serialization?).
