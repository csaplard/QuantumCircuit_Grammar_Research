"""
Internal Fisher-layer proof of concept.

Registers forward hooks on intermediate transformer layer(s) of a small
Gemma-family model, computes a sliding-window Fisher trace from hidden-state
deltas at the last token position, and optionally feeds that signal back into
the sampling temperature during generation.

A/B comparison:
  - baseline : monitor hooks active, no feedback (fixed temperature)
  - adaptive : monitor hooks active, temperature modulated by F(t)

Prompts: factual / mathematical / creative.

Outputs (under results/internal_fisher_poc/):
  - {mode}_{regime}_output.txt
  - {mode}_{regime}_fisher_trace.csv
  - summary.csv
  - internal_fisher_comparison.png

Run:
    python code/internal_fisher_poc.py
    python code/internal_fisher_poc.py --model google/gemma-3-1b-it --tokens 256
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

MODEL_CANDIDATES = [
    # Most-preferred first. Default is gemma-3-4b-it text-only — fits in 6GB GPU
    # under 4-bit quantization without any vision/audio offload bookkeeping.
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3n-E2B-it",
    "google/gemma-2-2b-it",
]
DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_QUANTIZE = "4bit"

PROMPTS: dict[str, list[str]] = {
    "factual": [
        "Explain the causes and consequences of World War I.",
        "Describe how photosynthesis works in plants.",
        "Summarize the main events of the French Revolution.",
        "Explain how a CPU executes machine instructions.",
        "Describe the structure and function of the human heart.",
        "Explain the theory of plate tectonics.",
        "Summarize the key ideas of Darwin's theory of evolution.",
    ],
    "mathematical": [
        "Prove that the square root of 2 is irrational.",
        "Prove that there are infinitely many prime numbers.",
        "Derive the quadratic formula from a x^2 + b x + c = 0 by completing the square.",
        "Prove that the sum of the first n positive integers equals n(n+1)/2.",
        "Show that if n^2 is even, then n is even.",
        "Prove by induction that 2^n > n for every positive integer n.",
        "Derive the formula for the area of a circle of radius r using integration.",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
        "Write a short story about a retired robot that secretly learns to paint.",
        "Write a short story about an old clockmaker searching for a missing pocket watch.",
        "Write a short story about a child who finds a hidden door at the base of a tree.",
        "Write a short story about a stranger arriving in a village snowed in for the winter.",
        "Write a short story about an astronaut who hears a faint voice on Mars.",
        "Write a short story about a violin that has been continuously played for two hundred years.",
    ],
}

WINDOW = 32
BASE_TEMP = 0.9
TOP_P = 0.95
TEMP_DELTA = 0.15
GEN_TOKENS = 256


# --------------------------------------------------------------------------- #
# Fisher monitor (forward hook)
# --------------------------------------------------------------------------- #

class FisherMonitor:
    """Sliding-window Fisher trace on a layer's last-token hidden state.

    F(t) = mean over the last (W-1) consecutive hidden-state deltas of
    mean((h_k - h_{k-1})^2) across hidden dims.
    """

    def __init__(self, window: int = WINDOW, name: str = "mid") -> None:
        self.window = int(window)
        self.name = name
        self.hiddens: deque[torch.Tensor] = deque(maxlen=self.window)
        self.history: list[float] = []
        self.last_value: float = 0.0

    def hook(self, module, inputs, output):
        # Gemma/Llama decoder layers may return a tuple; first element is hidden.
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor):
            return
        # h: (batch, seq_len, hidden_dim). Always take last token position.
        last = h[:, -1, :].detach().to(torch.float32).squeeze(0).cpu()
        self.hiddens.append(last)
        if len(self.hiddens) >= 2:
            deltas = []
            prev = None
            for t in self.hiddens:
                if prev is not None:
                    deltas.append((t - prev).pow(2).mean().item())
                prev = t
            self.last_value = float(np.mean(deltas))
        else:
            self.last_value = 0.0
        self.history.append(self.last_value)

    def reset(self) -> None:
        self.hiddens.clear()
        self.history.clear()
        self.last_value = 0.0


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _cleanup_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def load_model(preferred: str | None, dtype_str: str, device_pref: str = "cuda",
               quantize: str = "none"):
    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    bnb_config = None
    if quantize in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quantize == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,  # allow vision/audio on CPU
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )

    tried = [preferred] if preferred else []
    tried.extend([m for m in MODEL_CANDIDATES if m != preferred])

    last_err: Exception | None = None
    for name in tried:
        if not name:
            continue
        # Try the preferred placement first; on OOM, fall back through cpu.
        placements: list[str] = []
        if device_pref == "auto":
            placements = ["auto"]
        elif device_pref == "cpu":
            placements = ["cpu"]
        else:  # cuda
            if torch.cuda.is_available():
                placements = ["cuda", "auto", "cpu"]
            else:
                placements = ["cpu"]

        for placement in placements:
            try:
                print(f"[load] trying {name}  dtype={dtype}  placement={placement} ...", flush=True)
                tok = AutoTokenizer.from_pretrained(name)
                kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                }
                if bnb_config is not None:
                    kwargs["quantization_config"] = bnb_config
                    # bnb requires a device_map; auto puts it on GPU when it fits
                    kwargs["device_map"] = "auto" if placement != "cpu" else "cpu"
                elif placement == "auto":
                    kwargs["device_map"] = "auto"
                elif placement == "cpu":
                    kwargs["device_map"] = "cpu"
                mdl = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                mdl.eval()
                if placement == "cuda" and bnb_config is None:
                    mdl = mdl.to("cuda")
                dev = next(mdl.parameters()).device
                print(f"[load] loaded {name}  device={dev}  dtype={dtype}  placement={placement}", flush=True)
                return name, tok, mdl
            except torch.cuda.OutOfMemoryError as exc:
                print(f"[load] {name}@{placement} OOM: {exc}", flush=True)
                last_err = exc
                _cleanup_cuda()
                continue
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).splitlines()[0]
                print(f"[load] {name}@{placement} failed: {msg}", flush=True)
                last_err = exc
                _cleanup_cuda()
                if "out of memory" in str(exc).lower() or "CUBLAS" in str(exc):
                    continue
                break  # non-OOM error: try next model name, not next placement
    raise RuntimeError(f"Could not load any model candidate. Last error: {last_err}")


def find_decoder_layers(model) -> torch.nn.ModuleList:
    """Best-effort lookup of the decoder ModuleList across Gemma/Llama variants."""
    for path in (
        "model.layers",                   # Llama / Qwen / Gemma2 / Gemma3
        "model.model.layers",             # some wrappers
        "model.language_model.layers",    # Gemma 3n (multimodal)
        "language_model.model.layers",    # alternate
        "transformer.h",                  # GPT-2 style
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

    # Auto-detect failed: dump every ModuleList in the model so the user
    # (or future code) can see what the actual decoder path is.
    print("[find_decoder_layers] auto-detect failed; module-list inventory:", flush=True)
    candidates: list[tuple[str, int, str]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            child0 = module[0].__class__.__name__
            candidates.append((name, len(module), child0))
            print(f"  [{len(module):3d}]  {name:50s}  child={child0}", flush=True)
    # Heuristic: pick the longest ModuleList whose child class name contains
    # "Layer" or "Block" — that's almost always the decoder stack.
    decoder_like = [
        c for c in candidates
        if "layer" in c[2].lower() or "block" in c[2].lower()
    ]
    if decoder_like:
        decoder_like.sort(key=lambda c: c[1], reverse=True)
        chosen_name, chosen_len, chosen_class = decoder_like[0]
        print(f"[find_decoder_layers] heuristic pick: {chosen_name} "
              f"(n={chosen_len}, class={chosen_class})", flush=True)
        obj = model
        for part in chosen_name.split("."):
            obj = getattr(obj, part)
        return obj  # type: ignore[return-value]
    raise RuntimeError(
        "Could not locate decoder layer list on this model. "
        "See the inventory above and add the right path to find_decoder_layers."
    )


def pick_layer_indices(n_layers: int) -> dict[str, int]:
    return {
        "early": max(0, n_layers // 6),
        "mid":   max(0, n_layers // 2),
        "late":  max(0, n_layers - 2),
    }


# --------------------------------------------------------------------------- #
# Sampling + generation loop
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
def generate_with_monitor(
    model,
    tokenizer,
    prompt: str,
    monitors: dict[str, FisherMonitor],
    primary_key: str,
    *,
    mode: str,
    max_new_tokens: int,
    base_temp: float,
    top_p: float,
    temp_delta: float,
) -> dict[str, Any]:
    for m in monitors.values():
        m.reset()

    # With device_map="auto"+offload, modules can live on different devices.
    # Inputs should land where the input embeddings are.
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
    step_log: list[dict[str, Any]] = []
    recent_vals: list[float] = []

    primary = monitors[primary_key]

    next_input = input_ids
    next_attn = attn

    for step in range(max_new_tokens):
        kwargs: dict[str, Any] = {"use_cache": True, "past_key_values": past}
        if next_attn is not None:
            kwargs["attention_mask"] = next_attn
        out = model(input_ids=next_input, **kwargs)
        past = out.past_key_values
        logits = out.logits[:, -1, :]  # (1, vocab)

        f_val = float(primary.last_value) if np.isfinite(primary.last_value) else 0.0

        # Decide temperature
        if mode == "adaptive" and len(recent_vals) >= 4:
            mu = float(np.mean(recent_vals))
            sd = float(np.std(recent_vals) + 1e-8)
            if f_val > mu + sd:
                temp = base_temp * (1.0 - temp_delta)
                action = "down"
            elif f_val < mu - sd:
                temp = base_temp * (1.0 + temp_delta)
                action = "up"
            else:
                temp = base_temp
                action = "hold"
        else:
            mu = float("nan")
            sd = float("nan")
            temp = base_temp
            action = "hold"

        recent_vals.append(f_val)
        if len(recent_vals) > 256:
            recent_vals.pop(0)

        token_id = top_p_sample(logits[0], temperature=temp, top_p=top_p)
        generated_ids.append(token_id)

        step_log.append({
            "step": step,
            "token_id": token_id,
            "fisher_value": f_val,
            "running_mean": mu,
            "running_std": sd,
            "temperature": temp,
            "action": action,
            **{f"fisher_{name}": float(mon.last_value) for name, mon in monitors.items()},
        })

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        # Next iteration: single new token with extended attention mask.
        next_input = torch.tensor([[token_id]], device=device)
        if next_attn is not None:
            next_attn = torch.cat(
                [next_attn, torch.ones((1, 1), dtype=next_attn.dtype, device=device)],
                dim=1,
            )

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {"text": text, "steps": step_log}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default: gemma-3-4b-it)")
    ap.add_argument("--tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--base-temp", type=float, default=BASE_TEMP)
    ap.add_argument("--top-p", type=float, default=TOP_P)
    ap.add_argument("--temp-delta", type=float, default=TEMP_DELTA)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda",
                    help="placement: cuda (try GPU then fallback), cpu (force CPU), auto (device_map=auto)")
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"], default=DEFAULT_QUANTIZE,
                    help="bitsandbytes quantization (default: 4bit, set 'none' to disable)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed re-applied before each (regime, mode) generation for fair A/B comparison")
    ap.add_argument("--prompts-per-regime", type=int, default=0,
                    help="how many prompts per regime to actually run (0 = all). "
                         "Use a small value (e.g. 2) for a quick smoke test.")
    ap.add_argument("--outdir", default="results/internal_fisher_poc")
    ap.add_argument("--single-layer", action="store_true",
                    help="Only attach the 'mid' hook (saves memory).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_name, tok, mdl = load_model(args.model, args.dtype,
                                       device_pref=args.device,
                                       quantize=args.quantize)
    layers = find_decoder_layers(mdl)
    n_layers = len(layers)
    idx_map = pick_layer_indices(n_layers)
    if args.single_layer:
        idx_map = {"mid": idx_map["mid"]}
    print(f"[model] {model_name}  n_layers={n_layers}  hooks={idx_map}", flush=True)

    monitors: dict[str, FisherMonitor] = {}
    handles = []
    for name, idx in idx_map.items():
        mon = FisherMonitor(window=args.window, name=name)
        handle = layers[idx].register_forward_hook(mon.hook)
        monitors[name] = mon
        handles.append(handle)

    summary_rows: list[dict[str, Any]] = []
    fisher_traces: dict[tuple[str, str], list[float]] = {}

    try:
        for regime, prompt_list in PROMPTS.items():
            n_use = len(prompt_list) if args.prompts_per_regime <= 0 \
                else min(args.prompts_per_regime, len(prompt_list))
            for p_idx, prompt in enumerate(prompt_list[:n_use]):
                for mode in ("baseline", "adaptive"):
                    # Re-seed before every generation so baseline and adaptive
                    # for the same (regime, prompt) start from the same RNG state.
                    # Differences between the two modes then come purely from the
                    # temperature feedback.
                    torch.manual_seed(args.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(args.seed)
                    t0 = time.time()
                    print(f"\n=== {regime} / p{p_idx} / {mode}  (seed={args.seed}) ===",
                          flush=True)
                    result = generate_with_monitor(
                        mdl, tok, prompt, monitors, primary_key="mid",
                        mode=mode,
                        max_new_tokens=args.tokens,
                        base_temp=args.base_temp,
                        top_p=args.top_p,
                        temp_delta=args.temp_delta,
                    )
                    elapsed = time.time() - t0

                    tag = f"{mode}_{regime}_p{p_idx}"
                    (outdir / f"{tag}_output.txt").write_text(
                        f"MODEL: {model_name}\nMODE: {mode}\nREGIME: {regime}\n"
                        f"PROMPT_IDX: {p_idx}\nPROMPT:\n{prompt}\n\n"
                        f"--- GENERATED ---\n{result['text']}\n",
                        encoding="utf-8",
                    )

                    df = pd.DataFrame(result["steps"])
                    df.to_csv(outdir / f"{tag}_fisher_trace.csv", index=False)

                    f_mid = df["fisher_mid"].to_numpy(dtype=float) \
                        if "fisher_mid" in df.columns \
                        else df["fisher_value"].to_numpy(dtype=float)
                    # Keep only the first prompt's traces for the legacy comparison
                    # plot (one trace per (regime, mode) for visual clarity).
                    if p_idx == 0:
                        fisher_traces[(regime, mode)] = f_mid.tolist()
                    acts = df["action"].value_counts()
                    summary_rows.append({
                        "regime": regime,
                        "prompt_idx": p_idx,
                        "mode": mode,
                        "n_steps": int(len(df)),
                        "fisher_mean": float(np.nanmean(f_mid)),
                        "fisher_std": float(np.nanstd(f_mid)),
                        "n_down": int(acts.get("down", 0)),
                        "n_up": int(acts.get("up", 0)),
                        "n_hold": int(acts.get("hold", 0)),
                        "elapsed_s": round(elapsed, 2),
                        "model": model_name,
                    })
                    print(
                        f"[done] {regime}/p{p_idx}/{mode}: steps={len(df)}  "
                        f"mean F={np.nanmean(f_mid):.4g}  "
                        f"down/up/hold={acts.get('down',0)}/{acts.get('up',0)}/{acts.get('hold',0)}  "
                        f"t={elapsed:.1f}s",
                        flush=True,
                    )
    finally:
        for h in handles:
            h.remove()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)

    # Aggregate across prompts (per regime × mode): mean / std of per-prompt means,
    # plus total intervention counts.
    agg = summary_df.groupby(["regime", "mode"], as_index=False).agg(
        n_prompts=("prompt_idx", "nunique"),
        fisher_mean_mean=("fisher_mean", "mean"),
        fisher_mean_std=("fisher_mean", "std"),
        fisher_std_mean=("fisher_std", "mean"),
        n_down_total=("n_down", "sum"),
        n_up_total=("n_up", "sum"),
        n_hold_total=("n_hold", "sum"),
        total_steps=("n_steps", "sum"),
    )
    agg.to_csv(outdir / "aggregate_summary.csv", index=False)

    # 2x3 plot: rows = {baseline, adaptive}, cols = regimes.
    # Shows the FIRST prompt of each regime (for visual continuity with prior runs).
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    regimes = list(PROMPTS.keys())
    for col, regime in enumerate(regimes):
        for row, mode in enumerate(("baseline", "adaptive")):
            ax = axes[row, col]
            y = fisher_traces.get((regime, mode), [])
            ax.plot(y, lw=1.2)
            ax.set_title(f"{regime} / {mode}  (prompt 0)")
            ax.set_ylabel("F(t)")
            if row == 1:
                ax.set_xlabel("generation step")
            ax.grid(alpha=0.3)
    fig.suptitle(f"Internal Fisher trace (mid layer) — {model_name}")
    fig.tight_layout()
    fig.savefig(outdir / "internal_fisher_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\n[write] outputs written to {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        sys.exit(130)
