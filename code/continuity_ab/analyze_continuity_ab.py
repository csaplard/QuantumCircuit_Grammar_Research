"""
Continuity A/B — v1 analysis.

Operationalizes the metrics named in docs/continuity_ab/PREREG_v1.md.

Inputs (produced by continuity_ab_poc.py):
  results/continuity_ab/<run_tag>/
    manifest.json
    sessions.csv
    {arm}_{regime}_p{idx}_seed{s}_trace.csv         # per-step Fisher trace
    {arm}_{regime}_p{idx}_seed{s}_output.txt
    {arm}_{regime}_p{idx}_seed{s}_logits.npz        # OPTIONAL — needed for JSD
    {arm}_{regime}_p{idx}_seed{s}_tokens.json       # OPTIONAL — needed for JSD

Outputs (written next to inputs):
  decision.json     — the five pre-registered checks, pass/fail + observed
  per_session.csv   — one row per (regime, prompt_idx, seed) with D_A, D_B,
                      mean_JSD, max_JSD, turn1_token_match, turn1_fisher_max_dev
  figure_boundary.png — Fisher trace overlay for the first session, both arms,
                      with turn boundaries marked

Pre-registered metric definitions (LOCKED in PREREG_v1.md §"Operational
definitions"):

  D_k (boundary discontinuity for turn k -> k+1):
      before = F at the LAST 'gen' event of turn k
      after  = F at the 'prefill_end' event of turn k+1
      D_k    = |after - before| / (MAD(F over gen events only) + eps)

  Per-session D = mean over k in {1, ..., turns-1} of D_k.
      (Note: turn-1 has no preceding boundary, so D_0 is undefined and not
      averaged. The turn-1 equivalence check below is the dedicated sanity
      gate for turn 1.)

  JSD-equivalence sanity (behavior-equivalence test):
      For each (regime, prompt, seed), take arm A's generated tokens for
      turn 2, teacher-force them through BOTH arms' turn-2 starting states,
      read out the next-token distributions p_A[t], p_B[t] for the first
      min(20, gen_len) positions t.
      jsd_t = JSD(p_A[t] || p_B[t])  (in nats)
      Pass band: mean_t jsd_t <= 0.05  AND  max_t jsd_t <= 0.15.

  Turn-1 equivalence gate (HARD STOP):
      Arm A and arm B at turn 1 must produce bitwise-identical token IDs
      AND |F_A(t) - F_B(t)| < 1e-5 at every step. Failure aborts analysis
      and the run is discarded as not meeting the design assumption.

Usage:
  python code/continuity_ab/analyze_continuity_ab.py \
      --run-dir results/continuity_ab/<run_tag>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Optional plotting — fail soft if matplotlib is unavailable in the analysis
# environment.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:  # noqa: BLE001
    HAVE_PLT = False


EPS = 1e-9
TURN1_FISHER_TOL = 1e-5
JSD_MEAN_BAND = 0.05  # nats
JSD_MAX_BAND = 0.15   # nats
JSD_POSITIONS = 20

# Pre-registered headline thresholds (must match PREREG_v1.md).
HEADLINE_P_THRESHOLD = 0.01
HEADLINE_EFFECT_DELTA = 0.5  # Cliff's delta
NULL_SHRINK_TARGET = 0.5     # Markov-3: (D_A-D_B) must shrink >=50%
AR1_NULL_P_FLOOR = 0.05      # AR(1): A vs B must NOT be significant


# --------------------------------------------------------------------------- #
# Trace loading and per-session Fisher metrics.
# --------------------------------------------------------------------------- #

@dataclass
class SessionTrace:
    arm: str
    regime: str
    prompt_idx: int
    seed: int
    df: pd.DataFrame  # columns: turn, step, phase, token_id, fisher_value


def _parse_tag(name: str) -> tuple[str, str, int, int] | None:
    # e.g. A_factual_p0_seed42_trace.csv
    stem = name.replace("_trace.csv", "")
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    arm = parts[0]
    if arm not in ("A", "B"):
        return None
    regime = parts[1]
    try:
        prompt_idx = int(parts[2].lstrip("p"))
        seed = int(parts[3].lstrip("seed"))
    except ValueError:
        return None
    return arm, regime, prompt_idx, seed


def load_traces(run_dir: Path) -> list[SessionTrace]:
    out: list[SessionTrace] = []
    for path in sorted(run_dir.glob("*_trace.csv")):
        parsed = _parse_tag(path.name)
        if parsed is None:
            continue
        arm, regime, prompt_idx, seed = parsed
        df = pd.read_csv(path)
        out.append(SessionTrace(arm, regime, prompt_idx, seed, df))
    return out


def gen_only_mad(df: pd.DataFrame) -> float:
    """Median absolute deviation of Fisher values across generation steps
    only (excludes prefill_end events). Used as the per-session normalizer
    for D_k so the normalizer is not contaminated by the boundary jumps we
    are trying to measure."""
    gen = df[df["phase"] == "gen"]["fisher_value"].to_numpy(dtype=float)
    if gen.size == 0:
        return float("nan")
    med = float(np.median(gen))
    mad = float(np.median(np.abs(gen - med)))
    # Scale MAD to match a Gaussian standard deviation, so D_k reads roughly
    # like a z-score. (Standard convention; documented in PREREG.)
    return 1.4826 * mad


def boundary_indices(df: pd.DataFrame) -> list[tuple[int, int]]:
    """For each turn k > 0, return (idx_before, idx_after):
      idx_before = index in df of the LAST 'gen' event of turn k-1
      idx_after  = index in df of the 'prefill_end' event of turn k
    """
    pairs: list[tuple[int, int]] = []
    turns = sorted(df["turn"].unique())
    for k in turns:
        if k == 0:
            continue
        prev = df[(df["turn"] == k - 1) & (df["phase"] == "gen")]
        cur_pref = df[(df["turn"] == k) & (df["phase"] == "prefill_end")]
        if prev.empty or cur_pref.empty:
            continue
        pairs.append((int(prev.index[-1]), int(cur_pref.index[0])))
    return pairs


def session_D(df: pd.DataFrame) -> tuple[float, list[float]]:
    """Per-session D = mean of D_k over turn boundaries k in {1, ..., T-1}."""
    norm = gen_only_mad(df)
    if not math.isfinite(norm) or norm < EPS:
        return float("nan"), []
    Dk: list[float] = []
    for i_before, i_after in boundary_indices(df):
        f_before = float(df.loc[i_before, "fisher_value"])
        f_after = float(df.loc[i_after, "fisher_value"])
        Dk.append(abs(f_after - f_before) / (norm + EPS))
    if not Dk:
        return float("nan"), []
    return float(np.mean(Dk)), Dk


# --------------------------------------------------------------------------- #
# Turn-1 equivalence gate.
# --------------------------------------------------------------------------- #

def turn1_check(
    traces: list[SessionTrace], run_dir: Path,
) -> list[dict[str, Any]]:
    """For each (regime, prompt_idx, seed), require:
       - identical generated token IDs on turn 1
       - |F_A(t) - F_B(t)| < TURN1_FISHER_TOL at every step on turn 1
    """
    results: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, int], dict[str, SessionTrace]] = {}
    for t in traces:
        key = (t.regime, t.prompt_idx, t.seed)
        by_key.setdefault(key, {})[t.arm] = t
    for key, arms in sorted(by_key.items()):
        if "A" not in arms or "B" not in arms:
            results.append({"key": key, "ok": False, "reason": "missing arm"})
            continue
        a = arms["A"].df
        b = arms["B"].df
        a1 = a[a["turn"] == 0].reset_index(drop=True)
        b1 = b[b["turn"] == 0].reset_index(drop=True)
        # Token identity: compare 'gen' token_ids in order.
        a_tok = a1[a1["phase"] == "gen"]["token_id"].tolist()
        b_tok = b1[b1["phase"] == "gen"]["token_id"].tolist()
        tok_match = a_tok == b_tok
        # Fisher identity at the per-row level (rows are aligned by
        # construction: same prompt, same code path, same seed).
        n = min(len(a1), len(b1))
        if n == 0:
            fmax = float("nan")
            f_ok = False
        else:
            fa = a1["fisher_value"].to_numpy()[:n]
            fb = b1["fisher_value"].to_numpy()[:n]
            fmax = float(np.max(np.abs(fa - fb)))
            f_ok = fmax < TURN1_FISHER_TOL
        results.append({
            "key": key,
            "ok": bool(tok_match and f_ok),
            "tok_match": bool(tok_match),
            "fisher_max_abs_dev": fmax,
            "n_tokens_compared": len(a_tok),
        })
    return results


# --------------------------------------------------------------------------- #
# Behavior-equivalence (JSD) — requires saved logits, deferred if absent.
# --------------------------------------------------------------------------- #

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. p, q are probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    def _kl(x, y):
        mask = (x > 0) & (y > 0)
        return float(np.sum(x[mask] * (np.log(x[mask]) - np.log(y[mask]))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def load_logits_if_present(
    run_dir: Path, regime: str, p_idx: int, seed: int,
) -> dict[str, np.ndarray] | None:
    """Logits dump contract (produced by continuity_ab_poc.py):
       jsd_{regime}_p{idx}_seed{s}_logits.npz with arrays:
         tokens_ref         : (N,) int64  — arm A's first N turn-2 tokens
                                              (the teacher-forcing reference)
         turn2_jsd_logits_A : (N, V) f32  — arm A's natural next-token logits
                                              at turn-2 positions 0..N-1
         turn2_jsd_logits_B : (N, V) f32  — arm B teacher-forced on tokens_ref;
                                              its next-token logits at the same
                                              positions
         turn1_jsd_logits_A : (M, V) f32  — arm A's turn-1 natural logits
         turn1_jsd_logits_B : (M, V) f32  — arm B's turn-1 captured logits
                                              (should equal turn1_jsd_logits_A
                                              within fp tolerance by design)
    """
    combined = run_dir / f"jsd_{regime}_p{p_idx}_seed{seed}_logits.npz"
    if not combined.exists():
        return None
    data = np.load(combined)
    required = {"turn2_jsd_logits_A", "turn2_jsd_logits_B"}
    if not required <= set(data.files):
        return None
    out = {
        "A": data["turn2_jsd_logits_A"],
        "B": data["turn2_jsd_logits_B"],
    }
    if "turn1_jsd_logits_A" in data.files and "turn1_jsd_logits_B" in data.files:
        out["turn1_A"] = data["turn1_jsd_logits_A"]
        out["turn1_B"] = data["turn1_jsd_logits_B"]
    if "tokens_ref" in data.files:
        out["tokens_ref"] = data["tokens_ref"]
    return out


def turn1_jsd_sanity(logits: dict[str, np.ndarray]) -> dict[str, float]:
    """Turn-1 JSD must be ~0 by design (both arms do identical forward
    passes at turn 1). Non-zero turn-1 JSD signals fp non-determinism or
    a state-mismatch bug, and is reported separately from the headline."""
    if "turn1_A" not in logits or "turn1_B" not in logits:
        return {"turn1_jsd_max": float("nan"), "turn1_n": 0}
    a = logits["turn1_A"]
    b = logits["turn1_B"]
    n = min(len(a), len(b))
    if n == 0:
        return {"turn1_jsd_max": float("nan"), "turn1_n": 0}
    pa = softmax(a[:n], axis=-1)
    pb = softmax(b[:n], axis=-1)
    vals = [jsd(pa[t], pb[t]) for t in range(n)]
    return {"turn1_jsd_max": float(np.max(vals)), "turn1_n": int(n)}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + EPS)


def session_jsd(logits: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-position JSD on the first JSD_POSITIONS positions of turn 2."""
    a = logits["A"][:JSD_POSITIONS]
    b = logits["B"][:JSD_POSITIONS]
    n = min(len(a), len(b))
    if n == 0:
        return {"mean_jsd": float("nan"), "max_jsd": float("nan"), "n": 0}
    pa = softmax(a, axis=-1)
    pb = softmax(b, axis=-1)
    js = np.array([jsd(pa[t], pb[t]) for t in range(n)])
    return {
        "mean_jsd": float(np.mean(js)),
        "max_jsd": float(np.max(js)),
        "n": int(n),
    }


# --------------------------------------------------------------------------- #
# Group-level tests (paired A vs B on per-session D).
# --------------------------------------------------------------------------- #

def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size == 0 or ya.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in xa:
        gt += int(np.sum(xi > ya))
        lt += int(np.sum(xi < ya))
    return (gt - lt) / (xa.size * ya.size)


def paired_test(D_A: list[float], D_B: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank (paired, two-sided), since pairing is per
    (regime, prompt, seed). The two-sided test is pre-registered: see
    PREREG_v1.md §"Direction semantics (LOCKED ...)" — the headline is
    "metric separates arms", not "A > B"."""
    out: dict[str, float] = {
        "n_pairs": float(min(len(D_A), len(D_B))),
        "median_D_A": float(np.median(D_A)) if D_A else float("nan"),
        "median_D_B": float(np.median(D_B)) if D_B else float("nan"),
        "cliffs_delta": cliffs_delta(D_A, D_B),
    }
    out["abs_cliffs_delta"] = abs(out["cliffs_delta"]) \
        if math.isfinite(out["cliffs_delta"]) else float("nan")
    out["sign_of_gap"] = int(np.sign(out["median_D_A"] - out["median_D_B"])) \
        if math.isfinite(out["median_D_A"]) and math.isfinite(out["median_D_B"]) \
        else 0
    try:
        from scipy.stats import wilcoxon  # type: ignore
        if len(D_A) == len(D_B) and len(D_A) >= 2:
            stat, p = wilcoxon(D_A, D_B, zero_method="zsplit",
                               alternative="two-sided")
            out["wilcoxon_stat"] = float(stat)
            out["wilcoxon_p"] = float(p)
    except Exception:  # noqa: BLE001
        out["wilcoxon_p"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Decision rule application.
# --------------------------------------------------------------------------- #

def apply_decision_rule(
    *, paired: dict[str, float], jsd_pass_cells: int, jsd_total_cells: int,
    null_real: dict[str, float] | None,
    null_markov: dict[str, float] | None,
    null_ar1: dict[str, float] | None,
) -> dict[str, Any]:
    """Mirrors the five pre-registered checks in PREREG_v1.md."""
    checks = {}

    # 1) behavior-equivalence band met on >= 8/12 cells (or scaled equivalent).
    if jsd_total_cells > 0:
        frac = jsd_pass_cells / jsd_total_cells
        checks["behavior_equivalence"] = {
            "observed_pass_cells": jsd_pass_cells,
            "observed_total_cells": jsd_total_cells,
            "required_fraction": 8 / 12,
            "passed": bool(frac >= 8 / 12),
        }
    else:
        checks["behavior_equivalence"] = {"passed": None, "reason": "no JSD data"}

    # 2) paired two-sided test on real prompts: p<0.01.
    #    Direction is reported, not gated — see PREREG §Direction semantics.
    p = paired.get("wilcoxon_p", float("nan"))
    checks["paired_significance"] = {
        "p": p,
        "threshold": HEADLINE_P_THRESHOLD,
        "alternative": "two-sided",
        "observed_sign": paired.get("sign_of_gap", 0),
        "passed": bool(math.isfinite(p) and p < HEADLINE_P_THRESHOLD),
    }

    # 3) |Cliff's delta| >= 0.5 (sign-invariant)
    abs_cd = paired.get("abs_cliffs_delta", float("nan"))
    checks["effect_size"] = {
        "cliffs_delta": paired.get("cliffs_delta", float("nan")),
        "abs_cliffs_delta": abs_cd,
        "threshold": HEADLINE_EFFECT_DELTA,
        "passed": bool(math.isfinite(abs_cd) and abs_cd >= HEADLINE_EFFECT_DELTA),
    }

    # 4) Markov-3: |D_A - D_B| shrinks by >=50% relative to real (abs gap).
    if null_markov and null_real:
        real_gap = abs(null_real.get("median_D_A", 0.0)
                       - null_real.get("median_D_B", 0.0))
        null_gap = abs(null_markov.get("median_D_A", 0.0)
                       - null_markov.get("median_D_B", 0.0))
        shrink = float("nan") if real_gap == 0 else 1.0 - (null_gap / real_gap)
        checks["markov_null_shrink"] = {
            "real_abs_gap": real_gap, "null_abs_gap": null_gap,
            "shrink_fraction": shrink, "threshold": NULL_SHRINK_TARGET,
            "passed": bool(math.isfinite(shrink) and shrink >= NULL_SHRINK_TARGET),
        }
    else:
        checks["markov_null_shrink"] = {"passed": None, "reason": "null data missing"}

    # 5) AR(1): A vs B must NOT be significant at p<0.05.
    if null_ar1:
        p_ar1 = null_ar1.get("wilcoxon_p", float("nan"))
        checks["ar1_null_nonsig"] = {
            "p": p_ar1, "floor": AR1_NULL_P_FLOOR,
            "passed": bool(math.isfinite(p_ar1) and p_ar1 >= AR1_NULL_P_FLOOR),
        }
    else:
        checks["ar1_null_nonsig"] = {"passed": None, "reason": "null data missing"}

    all_passed = all(
        c.get("passed") is True for c in checks.values()
    )
    return {"checks": checks, "headline_supported": bool(all_passed)}


# --------------------------------------------------------------------------- #
# Plot.
# --------------------------------------------------------------------------- #

def plot_boundary(traces: list[SessionTrace], out_path: Path) -> None:
    if not HAVE_PLT or not traces:
        return
    # Pick first (regime, prompt, seed) cell where both arms exist.
    by_key: dict[tuple, dict[str, SessionTrace]] = {}
    for t in traces:
        by_key.setdefault((t.regime, t.prompt_idx, t.seed), {})[t.arm] = t
    target = next((v for v in by_key.values() if "A" in v and "B" in v), None)
    if target is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for arm_name, color in (("A", "tab:red"), ("B", "tab:blue")):
        df = target[arm_name].df.reset_index(drop=True)
        ax.plot(df.index, df["fisher_value"].to_numpy(),
                color=color, lw=1.0, label=f"arm {arm_name}")
        boundaries = df.index[df["phase"] == "prefill_end"].tolist()
        for b in boundaries:
            ax.axvline(b, color=color, alpha=0.2, ls="--")
    ax.set_xlabel("step (concatenated turns)")
    ax.set_ylabel("F (sliding-window Fisher trace)")
    ax.set_title("Fisher trace: arm A (napló) vs arm B (sodrás), with turn boundaries")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="directory produced by continuity_ab_poc.py")
    ap.add_argument("--null-markov-dir", default=None,
                    help="(optional) run-dir for Markov-3 surrogate")
    ap.add_argument("--null-ar1-dir", default=None,
                    help="(optional) run-dir for AR(1) surrogate")
    ap.add_argument("--strict-turn1", action="store_true", default=True,
                    help="hard-stop on turn-1 equivalence failure (default ON)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[err] run dir not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    traces = load_traces(run_dir)
    if not traces:
        print(f"[err] no _trace.csv files under {run_dir}", file=sys.stderr)
        sys.exit(2)

    # 0) Turn-1 equivalence gate.
    turn1 = turn1_check(traces, run_dir)
    n_fail = sum(1 for r in turn1 if not r["ok"])
    print(f"[turn1] cells={len(turn1)}  failing={n_fail}", flush=True)
    if n_fail > 0:
        for r in turn1:
            if not r["ok"]:
                print(f"  FAIL {r['key']}: "
                      f"tok_match={r.get('tok_match')} "
                      f"fmax={r.get('fisher_max_abs_dev')}", flush=True)
        if args.strict_turn1:
            (run_dir / "decision.json").write_text(json.dumps({
                "aborted": True,
                "reason": "turn1 equivalence gate failed",
                "turn1": turn1,
            }, indent=2))
            print("[abort] turn-1 equivalence gate failed; analysis halted.",
                  file=sys.stderr)
            sys.exit(3)

    # 1) Per-session D for each arm.
    per_session: list[dict[str, Any]] = []
    by_key: dict[tuple, dict[str, SessionTrace]] = {}
    for t in traces:
        by_key.setdefault((t.regime, t.prompt_idx, t.seed), {})[t.arm] = t

    jsd_pass = 0
    jsd_total = 0
    D_A_list: list[float] = []
    D_B_list: list[float] = []

    for key, arms in sorted(by_key.items()):
        regime, p_idx, seed = key
        row: dict[str, Any] = {"regime": regime, "prompt_idx": p_idx, "seed": seed}
        if "A" in arms:
            D_A, _ = session_D(arms["A"].df)
            row["D_A"] = D_A
            if math.isfinite(D_A):
                D_A_list.append(D_A)
        if "B" in arms:
            D_B, _ = session_D(arms["B"].df)
            row["D_B"] = D_B
            if math.isfinite(D_B):
                D_B_list.append(D_B)
        # JSD (turn-2 behavior equivalence) + turn-1 JSD sanity.
        logits = load_logits_if_present(run_dir, regime, p_idx, seed)
        if logits is not None:
            j = session_jsd(logits)
            row.update({"jsd_mean": j["mean_jsd"], "jsd_max": j["max_jsd"],
                        "jsd_n": j["n"]})
            t1 = turn1_jsd_sanity(logits)
            row.update({"turn1_jsd_max": t1["turn1_jsd_max"],
                        "turn1_jsd_n": t1["turn1_n"]})
            jsd_total += 1
            if (math.isfinite(j["mean_jsd"]) and j["mean_jsd"] <= JSD_MEAN_BAND
                    and math.isfinite(j["max_jsd"]) and j["max_jsd"] <= JSD_MAX_BAND):
                jsd_pass += 1
        else:
            row.update({"jsd_mean": float("nan"), "jsd_max": float("nan"),
                        "jsd_n": 0,
                        "turn1_jsd_max": float("nan"), "turn1_jsd_n": 0})
        per_session.append(row)

    df_session = pd.DataFrame(per_session)
    df_session.to_csv(run_dir / "per_session.csv", index=False)

    # 2) Paired test, real prompts.
    paired_real = paired_test(D_A_list, D_B_list)

    # 3) Null models, if provided. Same per_session + paired test logic on
    #    those run-dirs; we only need the paired summary here.
    null_real_summary = {
        "median_D_A": paired_real["median_D_A"],
        "median_D_B": paired_real["median_D_B"],
    }
    null_markov = None
    null_ar1 = None
    if args.null_markov_dir:
        try:
            null_markov = _summarize_external_run(Path(args.null_markov_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] markov null load failed: {exc}", flush=True)
    if args.null_ar1_dir:
        try:
            null_ar1 = _summarize_external_run(Path(args.null_ar1_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] ar1 null load failed: {exc}", flush=True)

    decision = apply_decision_rule(
        paired=paired_real,
        jsd_pass_cells=jsd_pass, jsd_total_cells=jsd_total,
        null_real=null_real_summary,
        null_markov=null_markov, null_ar1=null_ar1,
    )

    # Turn-1 JSD sanity (separate from the bitwise turn-1 gate above).
    # Should be ~0 by design; non-zero is fp non-determinism, not a headline
    # failure. Report max across all cells so the user sees if drift exists.
    t1_jsd_max = float("nan")
    t1_jsd_cells = [r.get("turn1_jsd_max", float("nan")) for r in per_session]
    t1_jsd_finite = [v for v in t1_jsd_cells if math.isfinite(v)]
    if t1_jsd_finite:
        t1_jsd_max = float(max(t1_jsd_finite))

    (run_dir / "decision.json").write_text(json.dumps({
        "turn1_gate": {"passed": n_fail == 0, "cells": turn1},
        "turn1_jsd_sanity": {
            "max_over_cells": t1_jsd_max,
            "expected": "~0 (design assumption: identical turn-1 forward pass)",
            "note": ("non-zero values indicate fp non-determinism; "
                     "above ~1e-3 suggests a state-mismatch bug, not a headline finding"),
        },
        "paired_real": paired_real,
        "jsd_pass_cells": jsd_pass, "jsd_total_cells": jsd_total,
        "null_markov": null_markov, "null_ar1": null_ar1,
        "decision": decision,
    }, indent=2))

    # 4) Plot.
    plot_boundary(traces, run_dir / "figure_boundary.png")

    # 5) Stdout summary.
    print("\n=== continuity_ab v1 analysis ===")
    print(f"sessions analyzed     : {len(per_session)}")
    print(f"median D_A / D_B      : "
          f"{paired_real['median_D_A']:.4f} / {paired_real['median_D_B']:.4f}")
    print(f"Cliff's delta (A-B)   : {paired_real.get('cliffs_delta', float('nan')):.3f}")
    print(f"Wilcoxon p            : {paired_real.get('wilcoxon_p', float('nan'))}")
    print(f"JSD cells passing band: {jsd_pass}/{jsd_total}")
    print(f"headline supported    : {decision['headline_supported']}")
    for name, c in decision["checks"].items():
        print(f"  [{name}] passed={c.get('passed')}  {c}")


def _summarize_external_run(run_dir: Path) -> dict[str, float]:
    """Helper: compute the paired summary for a null-model run dir, reusing
    the same machinery as the real one."""
    traces = load_traces(run_dir)
    by_key: dict[tuple, dict[str, SessionTrace]] = {}
    for t in traces:
        by_key.setdefault((t.regime, t.prompt_idx, t.seed), {})[t.arm] = t
    D_A: list[float] = []
    D_B: list[float] = []
    for key, arms in by_key.items():
        if "A" in arms:
            v, _ = session_D(arms["A"].df)
            if math.isfinite(v):
                D_A.append(v)
        if "B" in arms:
            v, _ = session_D(arms["B"].df)
            if math.isfinite(v):
                D_B.append(v)
    return paired_test(D_A, D_B)


if __name__ == "__main__":
    main()
