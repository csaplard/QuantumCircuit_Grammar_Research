"""
FisherGate Fine-Tuning — Training the Self-Regulating Layer.

Freezes the entire base model and trains ONLY the FisherGate's learnable
parameters (gate_linear: 1→1 linear, ~2 params) on GSM8K math reasoning.

The gradient path:
  frozen base layers → activations (grad flows through, not weights) →
  FisherGate.forward() [gate_linear has requires_grad=True] →
  subsequent frozen layers → logits → cross-entropy loss →
  backward to gate_linear.weight / gate_linear.bias only.

Outputs to results/fisher_gate_training/:
  training_log.csv, gate_weights_trajectory.png, loss_curve.png,
  eval_before.csv, eval_after.csv, comparison.csv, trained_gate_state.pt

Run:
    python code/train_fisher_gate.py
    python code/train_fisher_gate.py --steps 100 --eval-examples 25   # quick
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MODEL   = "google/gemma-3-1b-it"
FALLBACK_MODEL  = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

WINDOW      = 32
DAMPENING   = 0.85
EMA_ALPHA   = 0.1
SEED        = 42

TRAIN_EXAMPLES  = 500
EVAL_EXAMPLES   = 50
EPOCHS          = 2
GRAD_ACCUM      = 4
LR              = 1e-3
WARMUP_STEPS    = 20
MAX_SEQ_LEN     = 512
GEN_TOKENS      = 256


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# FisherGate — differentiable training version
# --------------------------------------------------------------------------- #

class FisherGate(nn.Module):
    """Differentiable self-monitoring gate for a transformer layer.

    The sliding-window Fisher measurement is done with detached tensors so
    history doesn't inflate the graph.  The gate_linear layer is the ONLY
    trainable component — gradients flow through it into the residual stream
    and onward to the loss.
    """

    def __init__(self, window: int = WINDOW, dampening: float = DAMPENING,
                 alpha: float = EMA_ALPHA) -> None:
        super().__init__()
        self.window    = window
        self.dampening = dampening
        self.alpha     = alpha

        # The trainable gate: z-score (scalar) → gate pre-activation (scalar).
        # Initialised identity-like: weight=1.0, bias=0.0.
        self.gate_linear = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.gate_linear.weight, 1.0)
        nn.init.constant_(self.gate_linear.bias,   0.0)

        # Optional stabiliser (can be disabled if OOM).
        # self.gate_norm = nn.LayerNorm(1)

        # Non-persistent sliding-window state (reset per example).
        self._buffer: deque[torch.Tensor] = deque(maxlen=window)
        self._f_ema:    float = 0.0
        self._f_sq_ema: float = 0.0
        self._step:     int   = 0

        # Trace for logging (plain Python — not part of graph).
        self.trace: list[dict[str, Any]] = []

    # -- public API --------------------------------------------------------- #

    def reset(self) -> None:
        """Reset per-example state."""
        self._buffer.clear()
        self._f_ema    = 0.0
        self._f_sq_ema = 0.0
        self._step     = 0
        self.trace.clear()

    # -- forward ------------------------------------------------------------ #

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (batch, seq_len, hidden_dim) — live activation tensor.
        Returns:
            Gated tensor with same shape and gradient path intact.
        """
        # ── Fisher measurement (fully detached — no graph) ──────────────────
        h_last = h_t[:, -1, :].detach().float().squeeze(0).cpu()
        self._buffer.append(h_last)
        self._step += 1

        fisher_val = 0.0
        z_val      = 0.0

        if len(self._buffer) >= 2:
            deltas = []
            prev = None
            for t in self._buffer:
                if prev is not None:
                    deltas.append((t - prev).pow(2).mean().item())
                prev = t
            fisher_val = float(np.mean(deltas))

            if self._step == 2:
                self._f_ema    = fisher_val
                self._f_sq_ema = fisher_val ** 2
            else:
                self._f_ema    = self.alpha * fisher_val + (1 - self.alpha) * self._f_ema
                self._f_sq_ema = self.alpha * fisher_val**2 + (1 - self.alpha) * self._f_sq_ema

            f_var  = max(0.0, self._f_sq_ema - self._f_ema**2)
            f_std  = math.sqrt(f_var) + 1e-8
            z_val  = (fisher_val - self._f_ema) / f_std

        # ── Gate computation (differentiable) ───────────────────────────────
        # z_val is a plain float; we wrap it in a tensor ON the gate_linear's
        # device so the linear layer can act on it with full grad support.
        z_tensor = torch.tensor(
            [[z_val]], dtype=torch.float32,
            device=self.gate_linear.weight.device,
        )
        # gate ∈ (0, 1), shape (1, 1), has grad_fn through gate_linear
        gate = torch.sigmoid(self.gate_linear(z_tensor))  # (1, 1)

        # ── Apply gate to LIVE activations ───────────────────────────────────
        # scale ∈ [dampening, 1] — still a tensor with grad_fn
        scale = self.dampening + (1.0 - self.dampening) * gate  # (1, 1)

        # Cast scale to h_t's dtype so forward works in both training (autocast)
        # and inference (no autocast, h_t may be bfloat16 on GPU).
        scale = scale.to(dtype=h_t.dtype, device=h_t.device)
        # h_t: (batch, seq, hidden); scale broadcasts over all dims
        output = h_t * scale   # grad flows: output -> scale -> gate -> gate_linear

        # ── Log (detached floats only) ───────────────────────────────────────
        self.trace.append({
            "step":        self._step - 1,
            "fisher_value": fisher_val,
            "fisher_ema":  self._f_ema,
            "z_score":     z_val,
            "gate_value":  gate.item(),
        })

        return output


# --------------------------------------------------------------------------- #
# Model + tokenizer loading
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


def load_model(model_name: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    for name in [model_name, FALLBACK_MODEL]:
        if not name:
            continue
        try:
            print(f"[load] {name} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            mdl.eval()
            dev = next(mdl.parameters()).device
            print(f"[load] OK  device={dev}", flush=True)
            return name, tok, mdl
        except Exception as exc:
            print(f"[load] {name} failed: {str(exc).splitlines()[0]}", flush=True)
            _cleanup_cuda()

    raise RuntimeError("No model could be loaded.")


# --------------------------------------------------------------------------- #
# Decoder-layer detection (same proven pattern)
# --------------------------------------------------------------------------- #

def find_decoder_layers(model) -> nn.ModuleList:
    for path in (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "transformer.h",
    ):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        else:
            if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
                return obj  # type: ignore[return-value]

    # Heuristic fallback
    best_name, best_obj = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            child = module[0].__class__.__name__.lower()
            if "layer" in child or "block" in child or "decoder" in child:
                if best_obj is None or len(module) > len(best_obj):
                    best_name, best_obj = name, module
    if best_obj is not None:
        print(f"[layers] heuristic: {best_name} (n={len(best_obj)})", flush=True)
        return best_obj  # type: ignore[return-value]

    raise RuntimeError("Could not locate decoder layer list.")


# --------------------------------------------------------------------------- #
# GSM8K helpers
# --------------------------------------------------------------------------- #

def extract_answer(text: str) -> str | None:
    """Extract the numerical answer from a GSM8K response."""
    # Standard GSM8K format: "#### <number>"
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last standalone number
    nums = re.findall(r"\b\d[\d,]*\.?\d*\b", text.replace(",", ""))
    return nums[-1] if nums else None


def format_prompt(tokenizer, question: str, answer: str | None = None,
                  max_len: int = MAX_SEQ_LEN) -> dict[str, torch.Tensor]:
    """Format a GSM8K example using the tokenizer's chat template."""
    messages = [{"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(answer is None),
        )
    except Exception:
        # Fallback for models without a chat template
        if answer is not None:
            text = f"Question: {question}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer:"

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    return enc


def get_prompt_length(tokenizer, question: str, max_len: int = MAX_SEQ_LEN) -> int:
    """Number of tokens in the question-only portion (for loss masking)."""
    messages = [{"role": "user", "content": question}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        text = f"Question: {question}\nAnswer:"
    enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
    return len(enc["input_ids"])


# --------------------------------------------------------------------------- #
# Generation (for eval)
# --------------------------------------------------------------------------- #

def top_p_sample(logits: torch.Tensor, temperature: float = 0.1,
                 top_p: float = 0.95) -> int:
    temperature = max(1e-6, temperature)
    logits = logits / temperature
    probs  = torch.softmax(logits, dim=-1)
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False
    sorted_p = sorted_p.masked_fill(mask, 0.0)
    s = sorted_p.sum()
    if float(s) <= 0.0:
        return int(sorted_idx[0].item())
    sorted_p = sorted_p / s
    pick = torch.multinomial(sorted_p, 1)
    return int(sorted_idx.gather(-1, pick).item())


@torch.no_grad()
def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = GEN_TOKENS, temperature: float = 0.1) -> str:
    try:
        emb = model.get_input_embeddings()
        device = emb.weight.device
    except Exception:
        device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn      = enc.get("attention_mask", None)
    past      = None
    generated: list[int] = []
    next_in   = input_ids
    next_attn = attn

    for _ in range(max_new_tokens):
        out   = model(input_ids=next_in, attention_mask=next_attn,
                      past_key_values=past, use_cache=True)
        past  = out.past_key_values
        logits = out.logits[:, -1, :]
        tid   = top_p_sample(logits[0], temperature=temperature)
        if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
            break
        generated.append(tid)
        next_in   = torch.tensor([[tid]], device=device)
        if next_attn is not None:
            next_attn = torch.cat(
                [next_attn, torch.ones((1, 1), dtype=next_attn.dtype, device=device)],
                dim=1,
            )
    return tokenizer.decode(generated, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, tokenizer, examples: list[dict], label: str,
             gate: FisherGate | None, hook_handle=None) -> list[dict]:
    """Run eval on GSM8K examples; returns per-example results."""
    rows = []
    correct = 0
    for i, ex in enumerate(examples):
        set_seeds(SEED + i)
        if gate is not None:
            gate.reset()

        q      = ex["question"]
        gold   = extract_answer(ex["answer"]) or ""

        messages = [{"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = f"Question: {q}\nAnswer:"

        gen  = generate(model, tokenizer, prompt)
        pred = extract_answer(gen) or ""
        ok   = pred.strip() == gold.strip()
        if ok:
            correct += 1
        rows.append({
            "idx":       i,
            "question":  q,
            "gold":      gold,
            "pred":      pred,
            "correct":   int(ok),
            "generated": gen[:300],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(examples)}  acc={correct/(i+1):.1%}",
                  flush=True)

    acc = correct / len(examples) if examples else 0.0
    print(f"  [{label}] FINAL acc = {acc:.1%} ({correct}/{len(examples)})", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default=DEFAULT_MODEL)
    ap.add_argument("--steps",         type=int, default=0,
                    help="max gradient steps (0 = auto from epochs × data)")
    ap.add_argument("--epochs",        type=int, default=EPOCHS)
    ap.add_argument("--eval-examples", type=int, default=EVAL_EXAMPLES)
    ap.add_argument("--train-examples",type=int, default=TRAIN_EXAMPLES)
    ap.add_argument("--seq-len",       type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lr",            type=float, default=LR)
    ap.add_argument("--grad-accum",    type=int, default=GRAD_ACCUM)
    ap.add_argument("--warmup",        type=int, default=WARMUP_STEPS)
    ap.add_argument("--seed",          type=int, default=SEED)
    ap.add_argument("--outdir",        default="results/fisher_gate_training")
    ap.add_argument("--skip-eval",     action="store_true",
                    help="Skip pre/post eval (faster iteration)")
    args = ap.parse_args()

    set_seeds(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model_name, tok, model = load_model(args.model)

    # ── Freeze base model completely ─────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False
    print(f"[freeze] all base params frozen", flush=True)

    # Enable gradient checkpointing to save VRAM during backward
    try:
        model.gradient_checkpointing_enable()
        print("[grad_ckpt] enabled", flush=True)
    except Exception as e:
        print(f"[grad_ckpt] not available: {e}", flush=True)

    # ── Insert FisherGate ────────────────────────────────────────────────────
    layers = find_decoder_layers(model)
    n_layers = len(layers)
    hook_layer_idx = max(0, n_layers // 6)
    print(f"[model] {model_name}  n_layers={n_layers}  gate_at={hook_layer_idx}",
          flush=True)

    gate = FisherGate(window=WINDOW, dampening=DAMPENING, alpha=EMA_ALPHA)

    # Move gate to same device as the gate_linear (CPU initially, then GPU)
    try:
        emb_device = model.get_input_embeddings().weight.device
        gate.gate_linear = gate.gate_linear.to(emb_device)
    except Exception:
        pass

    # Verify trainable params
    trainable = [(n, p) for n, p in gate.named_parameters() if p.requires_grad]
    print(f"[gate] trainable params: {[n for n,_ in trainable]}", flush=True)
    assert len(trainable) > 0, "No trainable params in FisherGate!"

    # Register hook — modifies the output of the chosen decoder layer
    handle: Any = None

    def make_hook(fisher_gate: FisherGate):
        def hook_fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            gated = fisher_gate(h)
            if isinstance(output, tuple):
                return (gated,) + output[1:]
            return gated
        return hook_fn

    def install_hook():
        nonlocal handle
        handle = layers[hook_layer_idx].register_forward_hook(make_hook(gate))

    def remove_hook():
        nonlocal handle
        if handle is not None:
            handle.remove()
            handle = None

    # ── Load GSM8K ───────────────────────────────────────────────────────────
    print("[data] loading GSM8K ...", flush=True)
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test  = load_dataset("gsm8k", "main", split="test")

    train_data = list(ds_train.select(range(min(args.train_examples, len(ds_train)))))
    eval_data  = list(ds_test.select(range(min(args.eval_examples, len(ds_test)))))
    print(f"[data] train={len(train_data)}  eval={len(eval_data)}", flush=True)

    # ── Device for inputs ────────────────────────────────────────────────────
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device

    # ── Pre-training evaluation ───────────────────────────────────────────────
    if not args.skip_eval:
        print("\n[eval] Pre-training (gate at init, no hook) ...", flush=True)
        remove_hook()
        gate.reset()
        pre_rows = evaluate(model, tok, eval_data, "pre-no-gate", gate=None)
        pd.DataFrame(pre_rows).to_csv(outdir / "eval_before.csv", index=False)
        pre_acc = sum(r["correct"] for r in pre_rows) / len(pre_rows)
    else:
        pre_acc = float("nan")
        print("[eval] skipped pre-training eval", flush=True)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        gate.gate_linear.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
    total_steps = (
        args.steps if args.steps > 0
        else (len(train_data) // args.grad_accum) * args.epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=total_steps,
    )
    print(f"[train] total_steps={total_steps}  lr={args.lr}  "
          f"grad_accum={args.grad_accum}", flush=True)

    # ── Training ──────────────────────────────────────────────────────────────
    install_hook()
    log_rows: list[dict] = []
    global_step = 0
    accum_loss  = 0.0
    accum_count = 0
    t_start     = time.time()

    optimizer.zero_grad()

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        for ex_idx, ex in enumerate(train_data):
            if global_step >= total_steps:
                break

            gate.reset()
            q   = ex["question"]
            ans = ex["answer"]

            # Tokenize full prompt+answer
            enc = format_prompt(tok, q, ans, max_len=args.seq_len)
            input_ids = enc["input_ids"].to(input_device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(input_device)

            if input_ids.shape[1] < 4:
                continue  # skip malformed examples

            # Prompt length (for loss masking)
            prompt_len = get_prompt_length(tok, q, max_len=args.seq_len)
            prompt_len = min(prompt_len, input_ids.shape[1] - 1)

            # Build labels: -100 on prompt tokens, real ids on answer tokens
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                              enabled=torch.cuda.is_available()):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=labels,
                        use_cache=False,
                    )
                loss = out.loss / args.grad_accum
                loss.backward()

                accum_loss  += loss.item() * args.grad_accum
                accum_count += 1

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] step {global_step}, skipping", flush=True)
                _cleanup_cuda()
                optimizer.zero_grad()
                accum_count = 0
                accum_loss  = 0.0
                continue

            # Gradient step every GRAD_ACCUM examples
            if (ex_idx + 1) % args.grad_accum == 0:
                # Verify gradients exist (diagnostic)
                gw_grad = gate.gate_linear.weight.grad
                if gw_grad is None:
                    print(f"[warn] step {global_step}: gate_linear.weight.grad is None "
                          f"— gate may be disconnected from graph", flush=True)

                nn.utils.clip_grad_norm_(gate.gate_linear.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                mean_loss = accum_loss / accum_count if accum_count else float("nan")
                accum_loss  = 0.0
                accum_count = 0

                gw  = gate.gate_linear.weight.item()
                gb  = gate.gate_linear.bias.item()
                mgv = float(np.mean([t["gate_value"] for t in gate.trace])) \
                      if gate.trace else float("nan")
                mfv = float(np.mean([t["fisher_value"] for t in gate.trace
                                     if not math.isnan(t["fisher_value"])])) \
                      if gate.trace else float("nan")

                elapsed_m = (time.time() - t_start) / 60.0
                eta_m     = (elapsed_m / global_step) * (total_steps - global_step) \
                            if global_step > 0 else 0.0

                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"[step {global_step:4d}/{total_steps}] "
                        f"ep={epoch+1} "
                        f"loss={mean_loss:.4f} "
                        f"gate_w={gw:.4f} gate_b={gb:.4f} "
                        f"mean_gate={mgv:.4f} mean_F={mfv:.2f} "
                        f"elapsed={elapsed_m:.1f}m ETA={eta_m:.1f}m",
                        flush=True,
                    )

                log_rows.append({
                    "step":        global_step,
                    "epoch":       epoch + 1,
                    "loss":        mean_loss,
                    "gate_weight": gw,
                    "gate_bias":   gb,
                    "mean_gate":   mgv,
                    "mean_fisher": mfv,
                    "elapsed_min": round(elapsed_m, 2),
                })

                if global_step >= total_steps:
                    break

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(outdir / "training_log.csv", index=False)

    # Save trained gate
    torch.save(gate.state_dict(), outdir / "trained_gate_state.pt")
    print(f"[save] gate state -> {outdir / 'trained_gate_state.pt'}", flush=True)

    # ── Training plots ────────────────────────────────────────────────────────
    if len(log_df) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        axes[0].plot(log_df["step"], log_df["loss"], lw=1.5)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(alpha=0.3)

        # Gate weight & bias
        axes[1].plot(log_df["step"], log_df["gate_weight"], label="weight", lw=1.5)
        axes[1].plot(log_df["step"], log_df["gate_bias"],   label="bias",   lw=1.5)
        axes[1].axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5, label="init_w=1")
        axes[1].axhline(0.0, color="gray", ls=":",  lw=0.8, alpha=0.5, label="init_b=0")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("parameter value")
        axes[1].set_title("Gate Parameters over Training")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        fig.suptitle(f"FisherGate Training — {model_name}")
        fig.tight_layout()
        fig.savefig(outdir / "loss_curve.png", dpi=150)
        plt.close(fig)

        # Separate gate_weights_trajectory plot
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(log_df["step"], log_df["gate_weight"], label="gate_weight", lw=1.5)
        ax2.plot(log_df["step"], log_df["gate_bias"],   label="gate_bias",   lw=1.5)
        ax2.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax2.axhline(0.0, color="gray", ls=":",  lw=0.8, alpha=0.5)
        ax2.set_xlabel("step")
        ax2.set_title("Gate Weight/Bias Trajectory")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(outdir / "gate_weights_trajectory.png", dpi=150)
        plt.close(fig2)

    # ── Post-training evaluation ──────────────────────────────────────────────
    if not args.skip_eval:
        print("\n[eval] Post-training (trained gate active) ...", flush=True)
        gate.reset()
        # hook is still installed
        post_rows_gate = evaluate(model, tok, eval_data,
                                  "post-with-gate", gate=gate)
        pd.DataFrame(post_rows_gate).to_csv(outdir / "eval_after.csv", index=False)
        post_acc_gate = sum(r["correct"] for r in post_rows_gate) / len(post_rows_gate)

        print("\n[eval] Post-training (gate removed) ...", flush=True)
        remove_hook()
        post_rows_nogm = evaluate(model, tok, eval_data,
                                  "post-no-gate", gate=None)
        post_acc_nogm = sum(r["correct"] for r in post_rows_nogm) / len(post_rows_nogm)

        comparison = pd.DataFrame([
            {"mode": "pre-training (no gate)",    "accuracy": pre_acc},
            {"mode": "post-training (no gate)",   "accuracy": post_acc_nogm},
            {"mode": "post-training (with gate)", "accuracy": post_acc_gate},
        ])
        comparison.to_csv(outdir / "comparison.csv", index=False)
        print("\n" + comparison.to_string(index=False))
    else:
        remove_hook()
        post_acc_gate = float("nan")
        post_acc_nogm = float("nan")

    # ── Quick regime-sensitivity check ───────────────────────────────────────
    print("\n[regime-check] Quick gate sensitivity on 3 math vs 3 factual prompts ...",
          flush=True)
    install_hook()
    regime_checks = {
        "mathematical": [
            "Prove that sqrt(2) is irrational.",
            "Show that there are infinitely many primes.",
            "Derive the quadratic formula.",
        ],
        "factual": [
            "Explain how photosynthesis works.",
            "Describe the causes of World War I.",
            "What is the water cycle?",
        ],
    }
    regime_gates: dict[str, list[float]] = {}
    for regime, prompts in regime_checks.items():
        gvals: list[float] = []
        for p in prompts:
            gate.reset()
            set_seeds(SEED)
            messages = [{"role": "user", "content": p}]
            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text = f"Question: {p}\nAnswer:"
            _ = generate(model, tok, text, max_new_tokens=64)
            if gate.trace:
                valid = [t["gate_value"] for t in gate.trace
                         if not math.isnan(t.get("gate_value", float("nan")))]
                if valid:
                    gvals.append(float(np.mean(valid)))
        regime_gates[regime] = gvals
        print(f"  {regime:15s}: mean gate = {np.mean(gvals):.4f} "
              f"(n={len(gvals)})", flush=True)
    remove_hook()

    # ── Final diagnostics ─────────────────────────────────────────────────────
    final_w = gate.gate_linear.weight.item()
    final_b = gate.gate_linear.bias.item()
    init_w, init_b = 1.0, 0.0

    w_shift = abs(final_w - init_w) / (abs(init_w) + 1e-8) * 100
    b_shift = abs(final_b - init_b) / (abs(init_b) + 1e-8) * 100

    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"1. Did the gate learn?")
    print(f"   gate_weight: {init_w:.4f} → {final_w:.4f}  ({w_shift:.1f}% shift)")
    print(f"   gate_bias:   {init_b:.4f} → {final_b:.4f}  ({b_shift:.2f} abs shift)")
    learned = (w_shift > 5.0) or (abs(final_b - init_b) > 0.05)
    print(f"   → {'YES — weights moved significantly' if learned else 'NO — weights barely moved'}")

    if len(log_df) > 4:
        first_loss = log_df["loss"].iloc[:3].mean()
        last_loss  = log_df["loss"].iloc[-3:].mean()
        print(f"\n2. Did loss decrease?")
        print(f"   First 3 steps mean: {first_loss:.4f}")
        print(f"   Last  3 steps mean: {last_loss:.4f}")
        print(f"   → {'YES' if last_loss < first_loss else 'NO / FLAT'}")

    if not args.skip_eval:
        print(f"\n3. Did accuracy improve?")
        print(f"   Pre-training  (no gate): {pre_acc:.1%}")
        print(f"   Post-training (no gate): {post_acc_nogm:.1%}")
        print(f"   Post-training (w/ gate): {post_acc_gate:.1%}")
        delta = post_acc_gate - pre_acc
        print(f"   Gate Δ vs baseline:      {delta:+.1%}")
        print(f"   → {'IMPROVEMENT' if delta > 0.01 else 'NEUTRAL' if delta >= -0.01 else 'REGRESSION'}")

    if regime_gates:
        print(f"\n4. Regime-dependent gate behaviour after training?")
        for r, gvals in regime_gates.items():
            print(f"   {r:15s}: {np.mean(gvals):.4f}")
        math_mean    = np.mean(regime_gates.get("mathematical", [0]))
        factual_mean = np.mean(regime_gates.get("factual", [0]))
        diff = abs(math_mean - factual_mean)
        print(f"   |math_gate - factual_gate| = {diff:.4f}")
        print(f"   → {'REGIME-SENSITIVE' if diff > 0.02 else 'NOT REGIME-SENSITIVE'}")

    print(f"\n[write] all outputs -> {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(130)
