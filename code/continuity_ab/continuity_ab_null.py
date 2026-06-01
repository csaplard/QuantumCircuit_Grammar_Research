"""
Continuity A/B — v1 null models.

Two structured null surrogates, both pre-registered in
docs/continuity_ab/PREREG_v1.md:

  markov  : 3-gram word Markov surrogate on the user prompts. Same
            conversation shape, statistically plausible English, but no
            semantic continuity from turn to turn. Requires running the
            full PoC against the surrogate prompts (~9 min CPU per smoke).

  ar1     : AR(1) surrogate of the Fisher-trace itself. No model run.
            Per session, fits an AR(1) process to the gen-phase Fisher
            values and synthesizes a surrogate trace of the same shape.
            Structural boundary positions (prefill_end events) and trace
            lengths are preserved, so anything in D_k that comes purely
            from those structural features is preserved too — which is
            exactly the discrimination we want: a real D_A vs D_B effect
            that PERSISTS under AR(1) is a normalizer/structure artifact;
            an effect that SHRINKS under AR(1) has structural content
            above first-order autocorrelation.

Pre-registered decision rules these nulls feed (in PREREG_v1.md):
  - markov : |D_A - D_B| in surrogate must shrink by >= 50% vs real
             (after the direction-symmetric update; signed gap |...|).
  - ar1    : paired A-vs-B test on surrogate must be NOT significant at
             p < 0.05.

Both nulls write a run dir whose structure mirrors the real run, so
analyze_continuity_ab.py can consume them via --null-markov-dir /
--null-ar1-dir.

Usage:
  # Markov-3 surrogate prompts, then re-run PoC:
  python code/continuity_ab/continuity_ab_null.py markov \
      --out-dir results/continuity_ab/_smoke_markov \
      --seed 1729 \
      -- \
      --model Qwen/Qwen2.5-1.5B-Instruct --tokens 64 --turns 2 \
      --prompts-per-regime 1 --seeds 42 --device cpu --dtype float32

  # AR(1) surrogate of an existing real run (no model needed):
  python code/continuity_ab/continuity_ab_null.py ar1 \
      --real-run-dir results/continuity_ab/_smoke \
      --out-dir results/continuity_ab/_smoke_ar1 \
      --seed 1729
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# --------------------------------------------------------------------------- #
# Markov-3 prompt surrogate.
#
# Trained on the concatenated real user prompts (small corpus, ~200 words).
# This is deliberate and locked: a larger external corpus would change what
# the null tests (semantic vs lexical structure). The v1 null tests whether
# the metric's discriminative power survives when the prompts are recombinant
# from the same lexicon but lack semantic continuity. Bigger corpus = v2.
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def markov3_train(corpus_texts: list[str]) -> dict[tuple[str, str], list[str]]:
    """Build a word-level 3-gram model: (w_{i-2}, w_{i-1}) -> [w_i, ...]."""
    counts: dict[tuple[str, str], list[str]] = {}
    for text in corpus_texts:
        words = _tokenize_words(text)
        for i in range(len(words) - 2):
            key = (words[i], words[i + 1])
            counts.setdefault(key, []).append(words[i + 2])
    return counts


def markov3_sample(
    model: dict[tuple[str, str], list[str]],
    n_words: int,
    rng: random.Random,
) -> str:
    """Generate `n_words` words from the 3-gram model, then add a terminal '?'.

    The trailing '?' is intentional: real prompts in CONVERSATIONS are all
    questions, and we want the surrogate to share that surface form so the
    chat-template render path is exercised the same way."""
    keys = list(model.keys())
    if not keys:
        return "?" * 1
    start = keys[rng.randrange(len(keys))]
    out = list(start)
    while len(out) < n_words:
        key = (out[-2], out[-1])
        if key in model:
            choices = model[key]
            out.append(choices[rng.randrange(len(choices))])
        else:
            # fallback: jump to a random key, append its first word
            nk = keys[rng.randrange(len(keys))]
            out.append(nk[0])
    text = " ".join(out[:n_words]).rstrip(".!?,;:")
    return text + "?"


def build_markov_conversations(
    real_convs: dict[str, list[list[str]]], seed: int,
) -> dict[str, list[list[str]]]:
    """Generate a same-shape surrogate of CONVERSATIONS."""
    rng = random.Random(seed)
    corpus: list[str] = []
    for convs in real_convs.values():
        for conv in convs:
            corpus.extend(conv)
    model = markov3_train(corpus)
    surrogate: dict[str, list[list[str]]] = {}
    for regime, convs in real_convs.items():
        surrogate[regime] = []
        for conv in convs:
            new_conv: list[str] = []
            for turn in conv:
                # match real turn length in words, with a floor of 8
                n_w = max(8, len(_tokenize_words(turn)))
                new_conv.append(markov3_sample(model, n_w, rng))
            surrogate[regime].append(new_conv)
    return surrogate


def cmd_markov(args) -> int:
    """Build Markov-3 surrogate prompts and invoke continuity_ab_poc.py."""
    from continuity_ab_poc import CONVERSATIONS  # local import: heavy modules

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surrogate = build_markov_conversations(CONVERSATIONS, args.seed)
    conv_json = out_dir / "markov_conversations.json"
    conv_json.write_text(
        json.dumps(surrogate, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[markov] surrogate written: {conv_json}", flush=True)
    # Sanity preview
    first_regime = next(iter(surrogate))
    print(f"[markov] sample ({first_regime}, conv 0, turn 0): "
          f"{surrogate[first_regime][0][0]!r}", flush=True)

    poc_path = Path(SCRIPT_DIR) / "continuity_ab_poc.py"
    cmd = [
        sys.executable, str(poc_path),
        "--conversations-json", str(conv_json),
        "--outdir", str(out_dir),
    ]
    # Pass through everything the user gave after `--` to control the PoC
    # (model, device, dtype, tokens, turns, seeds, prompts-per-regime).
    cmd.extend(args.passthrough or [])
    print(f"[markov] launching PoC: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


# --------------------------------------------------------------------------- #
# Structure-respecting AR(1) Fisher-trace surrogate.
#
# Smoke r1 surfaced a defect in the naive AR(1) surrogate (uniform AR(1)
# across the whole trace): it generates a value at each prefill_end the same
# way it generates a gen step, which inflates the boundary jump |F_after -
# F_before| because the surrogate has no notion of boundary structure. The
# resulting "AR(1) D_k" was an artifact of the generator, not a fair
# baseline.
#
# v1 fix (this implementation):
#   - Fit AR(1) on the gen phase WITHIN each turn separately.
#   - At each prefill_end position, sample F independently from the gen-
#     phase marginal distribution (Gaussian with the gen-phase mean and
#     std of that session). This models "no boundary continuity":
#     F_boundary is independent of F_last_gen_of_previous_turn.
#   - The next gen step after prefill_end starts an AR(1) walk from the
#     boundary value, using the per-turn fit parameters.
#
# Interpretation: under H0 (no continuity), real D_k should be similar to
# the null D_k. If real D_k is SMALLER, the boundary is being "absorbed"
# by something (the candidate mechanism is state continuity). If real D_k
# is LARGER, the boundary is MORE discontinuous than independent draw.
# --------------------------------------------------------------------------- #

def fit_ar1(series: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """Return (a, b, sigma) for F_t = a*F_{t-1} + b + sigma*N(0,1)."""
    if len(series) < 3:
        m = float(np.mean(series)) if len(series) else 0.0
        s = float(np.std(series)) if len(series) else 0.0
        return 0.0, m, max(s, eps)
    x = series[:-1].astype(float)
    y = series[1:].astype(float)
    vx = float(np.var(x))
    if vx < eps:
        return 0.0, float(np.mean(y)), float(np.std(y) + eps)
    cov = float(np.cov(x, y, bias=True)[0, 1])
    a = cov / (vx + eps)
    a = float(np.clip(a, -0.99, 0.99))
    b = float(np.mean(y) - a * np.mean(x))
    resid = y - (a * x + b)
    sigma = float(np.std(resid) + eps)
    return a, b, sigma


def synth_ar1(
    a: float, b: float, sigma: float, length: int,
    rng: np.random.Generator, x0: float,
) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    out[0] = x0
    for t in range(1, length):
        out[t] = a * out[t - 1] + b + sigma * rng.standard_normal()
    return out


def synth_structured_surrogate(
    real_df: pd.DataFrame, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Session-wide AR(1) on gen phase, independent marginal draws at prefill_end.

    v2 stabilization (smoke r2 finding): the per-turn AR(1) fit produced
    `a ≈ 0.93` on a 15-event segment — an unstable estimate that inflated
    the synthesized null D for arms with short turn-2 generation, raising
    the suspicion that the per-arm asymmetry observed in null D was a
    fit-artifact rather than a real null property. To remove that
    confound BEFORE deciding whether the per-arm (real-vs-null) pattern
    has signal, the AR(1) fit is now session-wide (pooled across all
    turns' gen events). This eliminates the small-sample over-fit failure
    mode while preserving the structure-respecting property: at each
    prefill_end position, F is still drawn independently from the gen
    marginal, modeling "no boundary continuity".

    If the per-arm (real_D − null_D) pattern survives this stabilization,
    the smoke-r2 asymmetry has content beyond per-turn AR(1) instability.
    If it does not survive, the asymmetry was a null fit artifact, and
    the v1 headline falls back to the descriptive minimum.

    This change is a null-model stabilization (control-strengthening),
    NOT a headline-test redesign. The pre-reg AR(1) check architecture
    is NOT modified by this commit. Headline-test decisions wait on the
    stabilized null's output.

    Returns (new_fisher_values, fit_diagnostics).
    """
    out_F = np.full(len(real_df), np.nan, dtype=float)

    # Session-wide gen-phase stats: AR(1) fit + marginal for boundary draws.
    gen_all = real_df.loc[real_df["phase"] == "gen", "fisher_value"].to_numpy(
        dtype=float,
    )
    gen_mean = float(np.mean(gen_all)) if len(gen_all) > 0 else 0.0
    gen_std = float(np.std(gen_all)) if len(gen_all) > 1 else 1e-6
    if gen_std < 1e-9:
        gen_std = 1e-9  # avoid degenerate independent-draw distribution

    # Single session-wide AR(1) fit on all gen events concatenated.
    # Note: this concatenates across turn boundaries, which is fine because
    # we only use (a, b, sigma) as scalar parameters; the synthesized walk
    # restarts at each turn from the independent prefill_end draw.
    if len(gen_all) >= 3:
        a, b, sigma = fit_ar1(gen_all)
    else:
        a, b, sigma = 0.0, gen_mean, gen_std

    diags: dict[str, Any] = {
        "fit_scope": "session_wide_pooled_v2",
        "n_gen_total": int(len(gen_all)),
        "gen_marginal_mean": gen_mean,
        "gen_marginal_std": gen_std,
        "ar1_a": a, "ar1_b": b, "ar1_sigma": sigma,
    }

    turns = sorted(real_df["turn"].unique())
    for k in turns:
        turn_indices = real_df.index[real_df["turn"] == k].tolist()
        if not turn_indices:
            continue
        prev_val: float | None = None
        for idx in turn_indices:
            phase = real_df.loc[idx, "phase"]
            if phase == "prefill_end":
                val = gen_mean + gen_std * rng.standard_normal()
            elif phase == "gen":
                if prev_val is None:
                    val = gen_mean + gen_std * rng.standard_normal()
                else:
                    val = a * prev_val + b + sigma * rng.standard_normal()
            else:
                val = float(real_df.loc[idx, "fisher_value"])
            out_F[idx] = val
            prev_val = val

    nan_mask = np.isnan(out_F)
    if nan_mask.any():
        out_F[nan_mask] = real_df.loc[nan_mask, "fisher_value"].to_numpy(
            dtype=float,
        )

    return out_F, diags


def cmd_ar1(args) -> int:
    real_dir = Path(args.real_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_written = 0
    n_skipped = 0
    fit_log: list[dict[str, Any]] = []

    for trace_path in sorted(real_dir.glob("*_trace.csv")):
        df = pd.read_csv(trace_path)
        if "phase" not in df.columns or "fisher_value" not in df.columns:
            n_skipped += 1
            continue
        gen_F = df.loc[df["phase"] == "gen", "fisher_value"].to_numpy(dtype=float)
        if len(gen_F) < 4:
            n_skipped += 1
            continue

        new_F, diags = synth_structured_surrogate(df, rng)
        new_df = df.copy()
        new_df["fisher_value"] = new_F
        out_path = out_dir / trace_path.name
        new_df.to_csv(out_path, index=False)
        fit_log.append({
            "trace": trace_path.name,
            "n_gen_total": int(len(gen_F)),
            **diags,
        })
        n_written += 1

    # Also copy session-level metadata + JSD npz files so analyze can run
    # without complaining about missing siblings. The NPZ files are NOT used
    # by the AR(1) D_k computation, but their presence keeps the analysis
    # symmetric across run dirs.
    for ext_path in real_dir.glob("*"):
        if ext_path.suffix in (".json", ".npz") or ext_path.name == "sessions.csv":
            (out_dir / ext_path.name).write_bytes(ext_path.read_bytes())

    (out_dir / "manifest_ar1.json").write_text(json.dumps({
        "kind": "ar1_structured_surrogate_v2_session_pooled",
        "description": (
            "Session-wide pooled AR(1) on gen phase; independent marginal "
            "draw at each prefill_end. v2 stabilization (smoke r2): "
            "replaces per-turn AR(1) fit, which over-fit on a 15-event "
            "segment (a=0.93) and inflated synthesized null D, raising "
            "the suspicion that the observed per-arm null asymmetry was "
            "a fit artifact. The pre-reg AR(1) headline check is NOT "
            "modified by this change; only the null model is stabilized. "
            "Whether the per-arm pattern survives this stabilization is "
            "the next-checked question, not assumed."
        ),
        "source": str(real_dir),
        "seed": args.seed,
        "n_traces_written": n_written,
        "n_skipped": n_skipped,
        "fits": fit_log,
    }, indent=2))
    print(f"[ar1 ] wrote {n_written} surrogate traces to {out_dir} "
          f"(skipped {n_skipped})", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_m = sub.add_parser("markov", help="Markov-3 prompt surrogate null")
    p_m.add_argument("--out-dir", required=True,
                     help="run-dir to populate (will be passed as PoC --outdir)")
    p_m.add_argument("--seed", type=int, default=1729,
                     help="seed for the Markov sampler (NOT the model seed)")
    p_m.add_argument("passthrough", nargs=argparse.REMAINDER,
                     help="args after `--` go to continuity_ab_poc.py "
                          "(model, device, dtype, tokens, turns, seeds, etc.)")

    p_a = sub.add_parser("ar1", help="AR(1) hidden surrogate null")
    p_a.add_argument("--real-run-dir", required=True,
                     help="run-dir to clone (real PoC output)")
    p_a.add_argument("--out-dir", required=True,
                     help="run-dir to write AR(1) surrogates into")
    p_a.add_argument("--seed", type=int, default=1729)

    args = ap.parse_args()
    # argparse with REMAINDER keeps the leading "--" if present; strip it
    if getattr(args, "passthrough", None):
        if args.passthrough and args.passthrough[0] == "--":
            args.passthrough = args.passthrough[1:]

    if args.mode == "markov":
        return cmd_markov(args)
    elif args.mode == "ar1":
        return cmd_ar1(args)
    else:
        ap.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
