"""
Fisher Gating Proof of Concept — self-regulating transformer layer.

Inserts a differentiable FisherGate module into an early decoder layer via a
forward hook.  The gate measures Fisher information from hidden-state
transitions (sliding window), normalises the signal, and modulates the
residual stream through a learnable sigmoid gate.

Six-way comparison (with ablation controls):
  - baseline  : no hook at all
  - monitor   : hook inserted, gate fixed at 1.0 (pass-through, only logs F)
  - active    : hook inserted, gate = sigmoid(linear(z)) from Fisher
  - random    : hook inserted, gate = sigmoid(randn()) per step
  - constant  : hook inserted, gate fixed at 0.92 (midpoint of 85%-100%)
  - inverted  : hook inserted, gate = sigmoid(linear(-z)) (negated z)

Prompts: 7 per regime (factual / mathematical / creative) = 126 total runs.

Outputs (under results/fisher_gating_poc/):
  - per-generation: {mode}_{regime}_{prompt_idx}_output.txt / _trace.csv
  - summary.csv, gating_effect.csv, regime_separation.csv, ablation_comparison.csv
  - fisher_gate_traces.png, gate_distribution.png
  - regime_separation_by_mode.png, ablation_summary.png

Run:
    python code/fisher_gating_poc.py
    python code/fisher_gating_poc.py --quick
    python code/fisher_gating_poc.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --tokens 128
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --------------------------------------------------------------------------- #
# Constants / defaults
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "google/gemma-3-4b-it"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

MODES = ("baseline", "monitor", "active", "random", "constant", "inverted")

PROMPTS: dict[str, list[str]] = {
    "factual": [
        "Explain the causes and consequences of World War I.",
        "Describe how photosynthesis works in plants.",
        "What are the main differences between TCP and UDP protocols?",
        "Explain how a combustion engine works step by step.",
        "Describe the water cycle and its importance for Earth's climate.",
        "What were the key events of the French Revolution?",
        "Explain how vaccines work to protect against diseases.",
    ],
    "mathematical": [
        "Prove that the square root of 2 is irrational.",
        "Derive the quadratic formula from ax^2 + bx + c = 0.",
        "Prove that there are infinitely many prime numbers.",
        "Show that the sum of angles in a triangle is 180 degrees.",
        "Prove by induction that 1+2+...+n = n(n+1)/2.",
        "Derive the formula for the area of a circle using integration.",
        "Prove that e is irrational using its series expansion.",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
        "Write a poem about the last tree on Earth.",
        "Describe an alien civilization that communicates through colors.",
        "Write a monologue from the perspective of a dying star.",
        "Create a myth explaining why the moon changes shape.",
        "Write a conversation between two AI systems meeting for the first time.",
        "Describe a city that exists entirely underwater.",
    ],
}

WINDOW = 32
GEN_TOKENS = 256
BASE_TEMP = 0.9
TOP_P = 0.95
DAMPENING = 0.85
EMA_ALPHA = 0.1
SEED = 42


# --------------------------------------------------------------------------- #
# FisherGate module — the core innovation
# --------------------------------------------------------------------------- #

class FisherGate(nn.Module):
    """Differentiable self-monitoring layer.

    Measures Fisher information from hidden-state transitions (sliding window),
    normalises via running EMA statistics, and produces a scalar gate that
    modulates the residual stream.

    The gate is differentiable — backprop can flow through it during
    fine-tuning.  We do not fine-tune in this PoC, but the architecture
    supports it.
    """

    def __init__(self, window: int = WINDOW, dampening: float = DAMPENING,
                 alpha: float = EMA_ALPHA) -> None:
        super().__init__()
        self.window = window
        self.dampening = dampening
        self.alpha = alpha

        # Learnable mapping from z-score to gate pre-activation.
        # Initialised to identity-like: weight=1.0, bias=0.0.
        self.linear = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.linear.weight, 1.0)
        nn.init.constant_(self.linear.bias, 0.0)

        # Sliding window buffer and EMA state (non-persistent — reset per gen).
        self._buffer: deque[torch.Tensor] = deque(maxlen=window)
        self._f_ema: float = 0.0
        self._f_sq_ema: float = 0.0
        self._step: int = 0

        # Trace log for analysis.
        self.trace: list[dict[str, float]] = []

        # Mode control.
        self._mode: str = "active"
        self._rng: torch.Generator | None = None  # for random mode

    # -- public API --------------------------------------------------------- #

    def set_mode(self, mode: str, seed: int = 42) -> None:
        """Set operating mode and reset state for a new generation."""
        assert mode in MODES, f"unknown mode: {mode}"
        self._mode = mode
        self.reset()
        if mode == "random":
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)

    def reset(self) -> None:
        """Clear sliding window, EMA state, and trace log."""
        self._buffer.clear()
        self._f_ema = 0.0
        self._f_sq_ema = 0.0
        self._step = 0
        self.trace.clear()

    # -- forward ------------------------------------------------------------ #

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """Apply Fisher-based gating to hidden state tensor.

        Args:
            h_t: (batch, seq_len, hidden_dim) tensor from the decoder layer.

        Returns:
            Gated tensor of the same shape.
        """
        # Extract last-token hidden for Fisher measurement.
        h_last = h_t[:, -1, :].detach().to(torch.float32).squeeze(0).cpu()
        self._buffer.append(h_last)
        self._step += 1

        # Compute Fisher trace from sliding window.
        fisher_val = float("nan")
        z_val = float("nan")
        gate_val = float("nan")
        action = "warmup"

        if len(self._buffer) >= 2:
            deltas = []
            prev = None
            for t in self._buffer:
                if prev is not None:
                    deltas.append((t - prev).pow(2).mean().item())
                prev = t
            fisher_val = float(np.mean(deltas))

            # Update EMA statistics.
            if self._step == 2:
                # First valid measurement — initialise EMA directly.
                self._f_ema = fisher_val
                self._f_sq_ema = fisher_val ** 2
            else:
                self._f_ema = self.alpha * fisher_val + (1.0 - self.alpha) * self._f_ema
                self._f_sq_ema = self.alpha * (fisher_val ** 2) + (1.0 - self.alpha) * self._f_sq_ema

            f_var = max(0.0, self._f_sq_ema - self._f_ema ** 2)
            f_std = math.sqrt(f_var) + 1e-8
            z_val = (fisher_val - self._f_ema) / f_std

            # Compute gate value depending on mode.
            if self._mode == "monitor":
                gate_val = 1.0
                action = "pass"
            elif self._mode == "active":
                z_t = torch.tensor([[z_val]], dtype=torch.float32,
                                   device=self.linear.weight.device)
                gate_val = float(torch.sigmoid(self.linear(z_t)).item())
                action = "gate"
            elif self._mode == "random":
                rand_z = torch.randn(1, generator=self._rng).item()
                gate_val = float(torch.sigmoid(torch.tensor(rand_z)).item())
                action = "rand"
            elif self._mode == "constant":
                gate_val = 0.92
                action = "const"
            elif self._mode == "inverted":
                z_t = torch.tensor([[-z_val]], dtype=torch.float32,
                                   device=self.linear.weight.device)
                gate_val = float(torch.sigmoid(self.linear(z_t)).item())
                action = "inv"
            else:
                # baseline — should never reach here (no hook installed)
                gate_val = 1.0
                action = "none"
        else:
            # Not enough history yet — pass through.
            gate_val = 1.0
            action = "warmup"

        # Log trace.
        self.trace.append({
            "step": self._step - 1,
            "fisher_value": fisher_val,
            "fisher_ema": self._f_ema,
            "z_score": z_val,
            "gate_value": gate_val,
            "action": action,
        })

        # Apply gate: output = h_t * (dampening + (1 - dampening) * gate)
        scale = self.dampening + (1.0 - self.dampening) * gate_val
        return h_t * scale


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _cleanup_cuda() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def load_model(preferred: str):
    """Load model with 4-bit quantisation.  Falls back to FALLBACK_MODEL."""
    dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    candidates = [preferred]
    if preferred != FALLBACK_MODEL:
        candidates.append(FALLBACK_MODEL)

    last_err: Exception | None = None
    for name in candidates:
        try:
            print(f"[load] trying {name} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            mdl.eval()
            dev = next(mdl.parameters()).device
            print(f"[load] loaded {name}  device={dev}", flush=True)
            return name, tok, mdl
        except Exception as exc:
            msg = str(exc).splitlines()[0]
            print(f"[load] {name} failed: {msg}", flush=True)
            last_err = exc
            _cleanup_cuda()
    raise RuntimeError(f"Could not load any model. Last error: {last_err}")


def find_decoder_layers(model) -> torch.nn.ModuleList:
    """Best-effort lookup of the decoder ModuleList."""
    for path in (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "transformer.h",
    ):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return obj  # type: ignore[return-value]

    # Heuristic fallback: find longest ModuleList with "Layer"/"Block" children.
    print("[find_decoder_layers] auto-detect failed; scanning ...", flush=True)
    candidates: list[tuple[str, int, str]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            child0 = module[0].__class__.__name__
            candidates.append((name, len(module), child0))
    decoder_like = [
        c for c in candidates
        if "layer" in c[2].lower() or "block" in c[2].lower()
    ]
    if decoder_like:
        decoder_like.sort(key=lambda c: c[1], reverse=True)
        chosen_name = decoder_like[0][0]
        print(f"[find_decoder_layers] heuristic pick: {chosen_name}", flush=True)
        obj = model
        for part in chosen_name.split("."):
            obj = getattr(obj, part)
        return obj  # type: ignore[return-value]
    raise RuntimeError("Could not locate decoder layer list.")


# --------------------------------------------------------------------------- #
# Sampling / generation loop
# --------------------------------------------------------------------------- #

def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    temperature = max(1e-6, float(temperature))
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    s = sorted_probs.sum()
    if float(s) <= 0.0:
        return int(sorted_idx[0].item())
    sorted_probs = sorted_probs / s
    pick = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_idx.gather(-1, pick).item())


@torch.no_grad()
def generate(
    model, tokenizer, prompt: str,
    fisher_gate: FisherGate | None,
    *,
    mode: str,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Token-by-token generation with KV cache.

    For baseline mode, fisher_gate should be None (no hook installed).
    For all other modes, fisher_gate is set to the appropriate mode and its
    hook is active on the target layer.
    """
    # Seed for reproducibility within (regime, prompt).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if fisher_gate is not None:
        fisher_gate.set_mode(mode, seed=seed)

    # Device detection via input embedding weights.
    try:
        emb = model.get_input_embeddings()
        device = emb.weight.device
    except Exception:
        device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)

    past = None
    generated_ids: list[int] = []
    next_input = input_ids
    next_attn = attn

    for step in range(max_new_tokens):
        kwargs: dict[str, Any] = {"use_cache": True, "past_key_values": past}
        if next_attn is not None:
            kwargs["attention_mask"] = next_attn
        out = model(input_ids=next_input, **kwargs)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        token_id = top_p_sample(logits[0], temperature=BASE_TEMP, top_p=TOP_P)
        generated_ids.append(token_id)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        next_input = torch.tensor([[token_id]], device=device)
        if next_attn is not None:
            next_attn = torch.cat(
                [next_attn, torch.ones((1, 1), dtype=next_attn.dtype, device=device)],
                dim=1,
            )

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Build trace from FisherGate log (or NaN for baseline).
    if fisher_gate is not None and fisher_gate.trace:
        trace = fisher_gate.trace.copy()
    else:
        n = len(generated_ids)
        trace = [
            {"step": i, "fisher_value": float("nan"), "fisher_ema": float("nan"),
             "z_score": float("nan"), "gate_value": float("nan"), "action": "none"}
            for i in range(n)
        ]

    return {"text": text, "trace": trace, "n_tokens": len(generated_ids)}


# --------------------------------------------------------------------------- #
# Hook management
# --------------------------------------------------------------------------- #

def make_hook(fisher_gate: FisherGate):
    """Create a forward hook that modifies the layer output via FisherGate."""
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        gated_h = fisher_gate(h)
        if isinstance(output, tuple):
            return (gated_h,) + output[1:]
        return gated_h
    return hook_fn


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two samples."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_traces(all_traces: dict, outdir: Path, regimes: list[str]) -> None:
    """3x6 grid: rows=regimes, cols=modes.  F(t) with gate overlay."""
    fig, axes = plt.subplots(len(regimes), len(MODES), figsize=(24, 12),
                             sharex=True)
    for row, regime in enumerate(regimes):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            key = (mode, regime)
            df = all_traces.get(key)
            if df is not None and len(df) > 0:
                ax.plot(df["fisher_value"].values, lw=0.8, color="steelblue",
                        label="F(t)")
                ax2 = ax.twinx()
                ax2.plot(df["gate_value"].values, lw=0.8, color="crimson",
                         alpha=0.7, label="gate")
                ax2.set_ylim(0.0, 1.05)
                if col == len(MODES) - 1:
                    ax2.set_ylabel("gate", fontsize=8)
            ax.set_title(f"{mode} / {regime}", fontsize=8)
            if col == 0:
                ax.set_ylabel("F(t)", fontsize=8)
            if row == len(regimes) - 1:
                ax.set_xlabel("step", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
    fig.suptitle("Fisher Gate Traces (first prompt per regime)")
    fig.tight_layout()
    fig.savefig(outdir / "fisher_gate_traces.png", dpi=150)
    plt.close(fig)


def plot_gate_distribution(all_traces: dict, outdir: Path, regimes: list[str]) -> None:
    """Histogram of gate values per regime, overlaid for active/random/inverted."""
    compare_modes = ("active", "random", "inverted")
    fig, axes = plt.subplots(1, len(regimes), figsize=(14, 4), sharey=True)
    colors = {"active": "steelblue", "random": "orange", "inverted": "green"}
    for i, regime in enumerate(regimes):
        ax = axes[i]
        for mode in compare_modes:
            key = (mode, regime)
            df = all_traces.get(key)
            if df is not None:
                vals = df["gate_value"].dropna().values
                if len(vals) > 0:
                    ax.hist(vals, bins=30, alpha=0.45, label=mode,
                            color=colors.get(mode, "grey"))
        ax.set_title(regime)
        ax.set_xlabel("gate value")
        if i == 0:
            ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle("Gate Value Distribution (all prompts)")
    fig.tight_layout()
    fig.savefig(outdir / "gate_distribution.png", dpi=150)
    plt.close(fig)


def plot_regime_separation(sep_df: pd.DataFrame, outdir: Path) -> None:
    """Grouped bar chart of Cohen's d (factual vs math) per mode."""
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = sep_df["mode"].values
    d_vals = sep_df["cohens_d_fact_math"].values
    x = np.arange(len(modes))
    bars = ax.bar(x, d_vals, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylabel("Cohen's d (factual vs mathematical)")
    ax.set_title("Regime Separation by Mode")
    ax.grid(alpha=0.2, axis="y")
    for bar, val in zip(bars, d_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "regime_separation_by_mode.png", dpi=150)
    plt.close(fig)


def plot_ablation_summary(ablation_df: pd.DataFrame, sep_df: pd.DataFrame,
                          outdir: Path, regimes: list[str]) -> None:
    """Two-panel figure: (left) mean gate by regime per mode, (right) Cohen's d."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel — mean gate value by regime for each mode.
    modes_list = list(MODES)
    x = np.arange(len(modes_list))
    width = 0.25
    colors = {"factual": "steelblue", "mathematical": "darkorange", "creative": "seagreen"}
    for i, regime in enumerate(regimes):
        vals = []
        for mode in modes_list:
            row = ablation_df[(ablation_df["mode"] == mode) & (ablation_df["regime"] == regime)]
            vals.append(float(row["mean_gate"].iloc[0]) if len(row) > 0 else float("nan"))
        ax1.bar(x + i * width, vals, width, label=regime,
                color=colors.get(regime, "grey"), edgecolor="black", linewidth=0.3)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(modes_list, rotation=30, ha="right")
    ax1.set_ylabel("Mean Gate Value")
    ax1.set_title("Mean Gate Value by Regime and Mode")
    ax1.legend()
    ax1.grid(alpha=0.2, axis="y")

    # Right panel — Cohen's d by mode.
    d_vals = []
    mode_labels = []
    for _, row in sep_df.iterrows():
        mode_labels.append(row["mode"])
        d_vals.append(row["cohens_d_fact_math"])
    ax2.bar(range(len(mode_labels)), d_vals, color="steelblue",
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(mode_labels)))
    ax2.set_xticklabels(mode_labels, rotation=30, ha="right")
    ax2.set_ylabel("Cohen's d (factual vs mathematical)")
    ax2.set_title("Regime Separation by Mode")
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle("Ablation Summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "ablation_summary.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run() -> None:
    ap = argparse.ArgumentParser(description="Fisher Gating PoC")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    ap.add_argument("--tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="results/fisher_gating_poc")
    ap.add_argument("--quick", action="store_true",
                    help="Run only 2 prompts per regime (36 runs) for fast iteration")
    ap.add_argument("--prompts-per-regime", type=int, default=0,
                    help="Explicit prompt count per regime (0 = all, --quick overrides)")
    args = ap.parse_args()

    if args.quick:
        prompts_per_regime = 2
    elif args.prompts_per_regime > 0:
        prompts_per_regime = args.prompts_per_regime
    else:
        prompts_per_regime = 7  # all

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -- load model --------------------------------------------------------- #
    model_name, tok, mdl = load_model(args.model)
    layers = find_decoder_layers(mdl)
    n_layers = len(layers)
    target_idx = max(0, n_layers // 6)  # early layer — strongest regime signal
    print(f"[model] {model_name}  n_layers={n_layers}  hook_layer={target_idx}",
          flush=True)

    # -- create FisherGate module ------------------------------------------- #
    fisher_gate = FisherGate(window=WINDOW, dampening=DAMPENING, alpha=EMA_ALPHA)

    # -- build run schedule ------------------------------------------------- #
    regimes = list(PROMPTS.keys())
    schedule: list[tuple[str, str, int, str]] = []  # (mode, regime, prompt_idx, prompt)
    for mode in MODES:
        for regime in regimes:
            for p_idx, prompt in enumerate(PROMPTS[regime][:prompts_per_regime]):
                schedule.append((mode, regime, p_idx, prompt))

    total = len(schedule)
    print(f"[plan] {total} runs: {len(MODES)} modes x {len(regimes)} regimes "
          f"x {prompts_per_regime} prompts", flush=True)

    # -- generation loop ---------------------------------------------------- #
    summary_rows: list[dict[str, Any]] = []
    # Store first-prompt traces for the grid plot, all-prompt traces for histograms.
    first_traces: dict[tuple[str, str], pd.DataFrame] = {}
    all_gate_values: dict[tuple[str, str], list[float]] = {}
    # Per-mode Fisher values by regime (for Cohen's d).
    fisher_by_mode_regime: dict[tuple[str, str], list[float]] = {}

    hook_handle = None
    global_t0 = time.time()

    try:
        for run_idx, (mode, regime, p_idx, prompt) in enumerate(schedule):
            elapsed = time.time() - global_t0
            if run_idx > 0:
                rate = elapsed / run_idx
                eta = rate * (total - run_idx)
            else:
                eta = 0.0
            print(f"\n[{run_idx + 1}/{total}] {mode} / {regime} / prompt {p_idx} "
                  f"— elapsed {elapsed / 60:.0f}m, ETA {eta / 60:.0f}m", flush=True)

            # Install or remove hook depending on mode.
            if hook_handle is not None:
                hook_handle.remove()
                hook_handle = None

            gate_arg: FisherGate | None = None
            if mode != "baseline":
                hook_handle = layers[target_idx].register_forward_hook(
                    make_hook(fisher_gate)
                )
                gate_arg = fisher_gate

            t0 = time.time()
            result = generate(
                mdl, tok, prompt, gate_arg,
                mode=mode,
                max_new_tokens=args.tokens,
                seed=args.seed,
            )
            gen_elapsed = time.time() - t0

            # -- save per-generation files ---------------------------------- #
            tag = f"{mode}_{regime}_{p_idx}"
            (outdir / f"{tag}_output.txt").write_text(
                f"MODEL: {model_name}\nMODE: {mode}\nREGIME: {regime}\n"
                f"PROMPT_IDX: {p_idx}\nSEED: {args.seed}\n"
                f"PROMPT:\n{prompt}\n\n--- GENERATED ---\n{result['text']}\n",
                encoding="utf-8",
            )

            trace_df = pd.DataFrame(result["trace"])
            trace_df.to_csv(outdir / f"{tag}_trace.csv", index=False)

            # -- accumulate stats ------------------------------------------- #
            f_vals = trace_df["fisher_value"].dropna().values.astype(float)
            g_vals = trace_df["gate_value"].dropna().values.astype(float)
            f_mean = float(np.nanmean(f_vals)) if len(f_vals) > 0 else float("nan")
            f_std = float(np.nanstd(f_vals)) if len(f_vals) > 0 else float("nan")
            g_mean = float(np.nanmean(g_vals)) if len(g_vals) > 0 else float("nan")

            summary_rows.append({
                "mode": mode, "regime": regime, "prompt_idx": p_idx,
                "fisher_mean": f_mean, "fisher_std": f_std,
                "mean_gate": g_mean, "n_steps": result["n_tokens"],
                "elapsed_s": round(gen_elapsed, 2), "model": model_name,
            })

            # Store traces for plots.
            key = (mode, regime)
            if p_idx == 0:
                first_traces[key] = trace_df
            all_gate_values.setdefault(key, []).extend(g_vals.tolist())

            # Fisher values for Cohen's d calculation.
            fisher_by_mode_regime.setdefault(key, []).extend(
                f_vals[np.isfinite(f_vals)].tolist()
            )

            print(f"  done: {result['n_tokens']} tokens, F_mean={f_mean:.4g}, "
                  f"gate_mean={g_mean:.4f}, t={gen_elapsed:.1f}s", flush=True)

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # -- aggregate analysis ------------------------------------------------- #
    print("\n[analysis] computing aggregate statistics ...", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)

    # gating_effect.csv — per regime per mode.
    gating_rows: list[dict[str, Any]] = []
    for mode in MODES:
        for regime in regimes:
            sub = summary_df[(summary_df["mode"] == mode) &
                             (summary_df["regime"] == regime)]
            gating_rows.append({
                "mode": mode, "regime": regime,
                "mean_fisher": float(sub["fisher_mean"].mean()),
                "std_fisher": float(sub["fisher_mean"].std()),
                "mean_gate": float(sub["mean_gate"].mean()),
                "std_gate": float(sub["mean_gate"].std()),
                "n_prompts": len(sub),
            })
    gating_df = pd.DataFrame(gating_rows)
    gating_df.to_csv(outdir / "gating_effect.csv", index=False)

    # regime_separation.csv — Cohen's d (factual vs math) per mode.
    sep_rows: list[dict[str, Any]] = []
    for mode in MODES:
        f_fact = np.array(fisher_by_mode_regime.get((mode, "factual"), []))
        f_math = np.array(fisher_by_mode_regime.get((mode, "mathematical"), []))
        f_crea = np.array(fisher_by_mode_regime.get((mode, "creative"), []))
        sep_rows.append({
            "mode": mode,
            "cohens_d_fact_math": cohens_d(f_fact, f_math),
            "cohens_d_fact_creative": cohens_d(f_fact, f_crea),
            "cohens_d_math_creative": cohens_d(f_math, f_crea),
        })
    sep_df = pd.DataFrame(sep_rows)
    sep_df.to_csv(outdir / "regime_separation.csv", index=False)

    # ablation_comparison.csv — side-by-side table.
    ablation_rows: list[dict[str, Any]] = []
    for mode in MODES:
        for regime in regimes:
            sub = summary_df[(summary_df["mode"] == mode) &
                             (summary_df["regime"] == regime)]
            ablation_rows.append({
                "mode": mode, "regime": regime,
                "mean_fisher": float(sub["fisher_mean"].mean()),
                "mean_gate": float(sub["mean_gate"].mean()),
                "n_prompts": len(sub),
            })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(outdir / "ablation_comparison.csv", index=False)

    # -- plots -------------------------------------------------------------- #
    print("[plots] generating figures ...", flush=True)

    plot_traces(first_traces, outdir, regimes)

    # For gate distribution, build DataFrames from accumulated gate values.
    all_traces_for_hist: dict[tuple[str, str], pd.DataFrame] = {}
    for key, vals in all_gate_values.items():
        all_traces_for_hist[key] = pd.DataFrame({"gate_value": vals})
    plot_gate_distribution(all_traces_for_hist, outdir, regimes)

    plot_regime_separation(sep_df, outdir)
    plot_ablation_summary(ablation_df, sep_df, outdir, regimes)

    total_elapsed = time.time() - global_t0
    print(f"\n[done] {total} runs completed in {total_elapsed / 60:.1f} min", flush=True)
    print(f"[write] outputs written to {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        sys.exit(130)
