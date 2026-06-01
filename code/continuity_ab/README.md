# Continuity A/B PoC (v1)

See `docs/continuity_ab/PREREG_v1.md` for the pre-registered protocol.

## What this is

A diagnostic experiment: can a Fisher-trace metric on the model's internal
state separate "memory-reconstructed continuity" (arm A — `napló`) from
"state-carry continuity" (arm B — `sodrás`) when the two arms are tuned to
behave near-equivalently from the outside?

This directory is the v1 minimal viable contrast:

- **Arm A** — `past_key_values=None` each turn; prior turns enter as a short
  summary in the prompt.
- **Arm B** — `past_key_values` carried across turn boundaries; no summary,
  no re-prefill.
- One Fisher monitor (sliding window on a mid-layer's last-token hidden
  state), **not reset between turns** on either arm.

The hypothesis is that the monitor's trace is continuous across turn
boundaries in B and discontinuous in A, while the generated text is
statistically indistinguishable between the arms.

## Files

- `continuity_ab_poc.py` — the runner. Loads the model, runs both arms over
  matched (regime × conversation × seed) sessions, writes per-session
  traces and generations. Does **not** yet compute the headline metric or
  null-model controls (those will live alongside as
  `analyze_continuity_ab.py` and `continuity_ab_null.py`).
- `README.md` — this file.

## Smoke run

```
python code/continuity_ab/continuity_ab_poc.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --tokens 64 --turns 2 \
    --prompts-per-regime 1 --seeds 42 --device cpu \
    --outdir results/continuity_ab/_smoke
```

If `--device cuda` and the model fits, this is much faster. On CPU, expect
several minutes for the smoke set.

## What needs to be true before promoting to a result run

**Gate 0 (must pass first — the cheapest error-catcher):**

0. `analyze_continuity_ab.py` reports the **turn-1 equivalence gate** as
   PASSING on every cell. Concretely:
   - arm A and arm B emit **bitwise-identical** `token_id` sequences on
     turn 1
   - `|F_A(t) - F_B(t)| < 1e-5` at every step of turn 1

   If turn 1 already separates, the two arms are not equivalently set up,
   and any turn-2+ separation is a false positive of the design itself.
   This must be clean before any turn-2 result is believed.

Then the rest:

1. Smoke run completes on both arms without exceptions; trace CSVs land.
2. Arm B turn-2 prefill length (visible in the `prefill_len` field of the
   per-turn step log) is **smaller** than arm A turn-2 prefill length —
   confirms the cache is doing its job.
3. The arms produce *different* text on turn 2+ (visible in `_output.txt`).
   If they produce identical text, the experiment is degenerate and the
   contrast needs to be revisited before the pre-reg is locked.

## Logit dump for JSD — implemented

`continuity_ab_poc.py` runs **three passes per cell** to produce the JSD
inputs that `analyze_continuity_ab.py` consumes:

1. **A natural** — arm A's normal session. Captures arm A's turn-1 and
   turn-2 first-20 next-token logits (already aligned with A's naturally-
   sampled tokens — these ARE the reference sequence).
2. **B natural** — arm B's normal session. Produces B's trace and output
   for the D_k measurement. No teacher-forcing.
3. **B replay** — arm B re-run with the same seed; at turn 2,
   teacher-forced on arm A's first-20 turn-2 tokens. Captures B's logits
   at each teacher-forced position. Turn 1 also captured for the turn-1
   JSD sanity (which must be ≈ 0 by design). This pass's Fisher trace is
   discarded.

Output file: `jsd_{regime}_p{idx}_seed{seed}_logits.npz` with keys:

```
tokens_ref         : (N,)  int64
turn2_jsd_logits_A : (N, V) float32
turn2_jsd_logits_B : (N, V) float32
turn1_jsd_logits_A : (M, V) float32
turn1_jsd_logits_B : (M, V) float32
```

**Asserts on write** (in `write_jsd_npz`) catch silent shape mismatches:
- `turn2` shapes must agree between arms,
- `turn2.shape[0]` must equal `len(tokens_ref)` — guarantees the
  teacher-forcing horizon and the captured-logit count are consistent.

### Teacher-forcing direction contract (the silent failure mode)

Inside `forward_tokens_and_generate` the loop invariant is:

> At loop entry, `logits` is the next-token distribution conditioned on
> `prefill + generated_ids[:step]`. It predicts position `step`.

When teacher-forcing, we **capture this distribution first**, *then* force
the chosen token to `tokens_ref[step]`, then advance. This guarantees that
`turn2_jsd_logits_A[t]` and `turn2_jsd_logits_B[t]` are both the
distributions predicting `tokens_ref[t]`, conditioned on `tokens_ref[:t]`.
An off-by-one here would silently break JSD pairing; the contract is
called out in a comment inside the loop so any future refactor sees it.

### Turn-1 sanity (cheapest fp-determinism check)

Because both arms do the exact same turn-1 forward pass with the same
seed, `turn1_jsd_logits_A` and `turn1_jsd_logits_B` should be bitwise
equal. The analysis reports `turn1_jsd_sanity.max_over_cells` in
`decision.json` — non-zero indicates fp non-determinism (still tolerable
in tiny amounts), values above ~1e-3 indicate a state-mismatch bug and
mean the turn-2 results are not trustable.

## Status

**Pre-lock.** The pre-reg in `docs/continuity_ab/PREREG_v1.md` is a draft.
No result-generating run, no figures, no claims until the pre-reg is tagged
`continuity-ab-v1-prereg-r1` in git.
