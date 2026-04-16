"""
Post-training evaluation script for FisherGate.
Loads the trained gate from results/fisher_gate_training/trained_gate_state.pt
and runs pre/post accuracy comparison on GSM8K test set (first 50 examples).

Run:
    python code/eval_fisher_gate.py
"""
from __future__ import annotations
import math, re, random, sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

OUTDIR = Path("results/fisher_gate_training")
SEED = 42
GEN_TOKENS = 256
EVAL_N = 50

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------------------------------- #
# FisherGate (must match training version exactly)
# --------------------------------------------------------------------------- #
class FisherGate(nn.Module):
    def __init__(self, window=32, dampening=0.85, alpha=0.1):
        super().__init__()
        self._buffer: deque[torch.Tensor] = deque(maxlen=window)
        self._f_ema = 0.0; self._f_sq_ema = 0.0; self._step = 0
        self.dampening = dampening; self.alpha = alpha
        self.gate_linear = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.gate_linear.weight, 1.0)
        nn.init.constant_(self.gate_linear.bias, 0.0)
        self.trace: list[dict] = []

    def reset(self):
        self._buffer.clear(); self._f_ema = 0.0
        self._f_sq_ema = 0.0; self._step = 0; self.trace.clear()

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        h_last = h_t[:, -1, :].detach().float().squeeze(0).cpu()
        self._buffer.append(h_last)
        self._step += 1
        z_val = 0.0; f_val = 0.0
        if len(self._buffer) >= 2:
            deltas = []
            prev = None
            for t in self._buffer:
                if prev is not None:
                    deltas.append((t - prev).pow(2).mean().item())
                prev = t
            f_val = float(np.mean(deltas))
            if self._step == 2:
                self._f_ema = f_val; self._f_sq_ema = f_val ** 2
            else:
                self._f_ema = self.alpha * f_val + (1 - self.alpha) * self._f_ema
                self._f_sq_ema = self.alpha * f_val**2 + (1 - self.alpha) * self._f_sq_ema
            f_std = math.sqrt(max(0.0, self._f_sq_ema - self._f_ema**2)) + 1e-8
            z_val = (f_val - self._f_ema) / f_std

        z_t = torch.tensor([[z_val]], dtype=torch.float32,
                             device=self.gate_linear.weight.device)
        with torch.no_grad():
            gate = torch.sigmoid(self.gate_linear(z_t))
        gate_v = gate.item()
        self.trace.append({"step": self._step - 1, "fisher": f_val,
                           "z": z_val, "gate": gate_v})
        scale = self.dampening + (1.0 - self.dampening) * gate
        scale = scale.to(dtype=h_t.dtype, device=h_t.device)
        return h_t * scale


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    for name in ["google/gemma-3-1b-it", "Qwen/Qwen2.5-Coder-1.5B-Instruct"]:
        try:
            print(f"[load] {name} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=bnb, device_map="auto",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            mdl.eval()
            for p in mdl.parameters():
                p.requires_grad = False
            dev = next(mdl.parameters()).device
            print(f"[load] OK  device={dev}", flush=True)
            return name, tok, mdl
        except Exception as e:
            print(f"[load] failed: {str(e).splitlines()[0]}", flush=True)
    raise RuntimeError("No model loaded")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int = GEN_TOKENS) -> str:
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    iids = enc["input_ids"]
    attn = enc.get("attention_mask")
    past = None; out_ids = []; ni = iids; na = attn

    for _ in range(max_new):
        o = model(input_ids=ni, attention_mask=na,
                  past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :]
        probs = torch.softmax(logits / 0.1, -1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, -1)
        mask = cum > 0.95; mask[..., 0] = False
        sp = sp.masked_fill(mask, 0.0)
        s = sp.sum()
        if float(s) <= 0:
            tid = int(si[0].item())
        else:
            sp = sp / s
            tid = int(si.gather(-1, torch.multinomial(sp, 1)).item())
        if tokenizer.eos_token_id and tid == tokenizer.eos_token_id:
            break
        out_ids.append(tid)
        ni = torch.tensor([[tid]], device=device)
        if na is not None:
            na = torch.cat([na, torch.ones((1, 1), dtype=na.dtype, device=device)], 1)

    return tokenizer.decode(out_ids, skip_special_tokens=True)


def extract_answer(text: str) -> str | None:
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"\b\d[\d,]*\.?\d*\b", text)
    return nums[-1] if nums else None


def make_prompt(tok, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return f"Question: {question}\nAnswer:"


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
def run_eval(model, tok, data, label: str, gate: FisherGate | None) -> tuple[list[dict], float]:
    rows = []; correct = 0
    for i, ex in enumerate(data):
        torch.manual_seed(SEED + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + i)
        if gate is not None:
            gate.reset()
        prompt = make_prompt(tok, ex["question"])
        gen = generate(model, tok, prompt)
        pred = extract_answer(gen) or ""
        gold = extract_answer(ex["answer"]) or ""
        ok = pred.strip() == gold.strip()
        if ok:
            correct += 1
        gate_mean = float(np.mean([t["gate"] for t in gate.trace])) if gate and gate.trace else float("nan")
        f_mean = float(np.mean([t["fisher"] for t in gate.trace if t["fisher"] != 0.0])) if gate and gate.trace else float("nan")
        rows.append({
            "idx": i, "gold": gold, "pred": pred, "correct": int(ok),
            "gate_mean": gate_mean, "fisher_mean": f_mean,
            "generated": gen[:300],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(data)}  acc={correct/(i+1):.1%}", flush=True)
    acc = correct / len(data)
    print(f"  [{label}] FINAL acc={acc:.1%} ({correct}/{len(data)})", flush=True)
    return rows, acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    model_name, tok, mdl = load_model()

    # Locate decoder layers
    layers = mdl.model.layers
    hook_idx = max(0, len(layers) // 6)
    print(f"[model] {model_name}  n_layers={len(layers)}  gate_at={hook_idx}", flush=True)

    # Load trained gate
    gate_init = FisherGate()
    gate_trained = FisherGate()
    trained_state = torch.load(OUTDIR / "trained_gate_state.pt", weights_only=True)
    gate_trained.load_state_dict(trained_state)
    gate_trained.eval()
    print(f"[gate-trained] weight={gate_trained.gate_linear.weight.item():.6f}  "
          f"bias={gate_trained.gate_linear.bias.item():.6f}", flush=True)
    print(f"[gate-init]    weight={gate_init.gate_linear.weight.item():.6f}  "
          f"bias={gate_init.gate_linear.bias.item():.6f}", flush=True)

    # Load eval data
    ds_test = load_dataset("gsm8k", "main", split="test")
    eval_data = list(ds_test.select(range(min(EVAL_N, len(ds_test)))))

    # Load pre-training eval (already done)
    pre_df = pd.read_csv(OUTDIR / "eval_before.csv")
    pre_acc = float(pre_df["correct"].mean())
    print(f"[pre-acc] loaded from eval_before.csv: {pre_acc:.1%}", flush=True)

    # Hook helpers
    handle = None
    def install_hook(gate_module: FisherGate):
        nonlocal handle
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            g = gate_module(h)
            return (g,) + out[1:] if isinstance(out, tuple) else g
        handle = layers[hook_idx].register_forward_hook(hook_fn)

    def remove_hook():
        nonlocal handle
        if handle:
            handle.remove()
            handle = None

    # --- eval: no gate ---
    print("\n[eval] Post-training WITHOUT gate ...", flush=True)
    rows_nogate, acc_nogate = run_eval(mdl, tok, eval_data, "post-no-gate", gate=None)
    pd.DataFrame(rows_nogate).to_csv(OUTDIR / "eval_after_no_gate.csv", index=False)

    # --- eval: init gate ---
    print("\n[eval] Post-training WITH init gate (weight=1, bias=0) ...", flush=True)
    install_hook(gate_init)
    rows_init, acc_init = run_eval(mdl, tok, eval_data, "post-init-gate", gate=gate_init)
    pd.DataFrame(rows_init).to_csv(OUTDIR / "eval_after_init_gate.csv", index=False)
    remove_hook()

    # --- eval: trained gate ---
    print("\n[eval] Post-training WITH trained gate ...", flush=True)
    install_hook(gate_trained)
    rows_trained, acc_trained = run_eval(mdl, tok, eval_data, "post-trained-gate", gate=gate_trained)
    pd.DataFrame(rows_trained).to_csv(OUTDIR / "eval_after_trained_gate.csv", index=False)
    remove_hook()

    # --- regime sensitivity check ---
    print("\n[regime-check] Fisher gate sensitivity after training ...", flush=True)
    install_hook(gate_trained)
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
    regime_results = {}
    for regime, prompts in regime_checks.items():
        g_means = []
        f_means = []
        for p in prompts:
            gate_trained.reset()
            torch.manual_seed(SEED)
            text = make_prompt(tok, p)
            _ = generate(mdl, tok, text, max_new=64)
            if gate_trained.trace:
                gv = [t["gate"] for t in gate_trained.trace]
                fv = [t["fisher"] for t in gate_trained.trace if t["fisher"] != 0.0]
                if gv:
                    g_means.append(float(np.mean(gv)))
                if fv:
                    f_means.append(float(np.mean(fv)))
        gm = float(np.mean(g_means)) if g_means else float("nan")
        fm = float(np.mean(f_means)) if f_means else float("nan")
        regime_results[regime] = {"gate_mean": gm, "fisher_mean": fm}
        print(f"  {regime:15s}: mean_gate={gm:.4f}  mean_F={fm:.2f}", flush=True)
    remove_hook()

    # --- comparison table ---
    comp = pd.DataFrame([
        {"mode": "pre-training (no gate)",         "accuracy": pre_acc},
        {"mode": "post-training (no gate)",         "accuracy": acc_nogate},
        {"mode": "post-training (init gate)",       "accuracy": acc_init},
        {"mode": "post-training (trained gate)",    "accuracy": acc_trained},
    ])
    comp.to_csv(OUTDIR / "comparison.csv", index=False)

    # --- diagnostics ---
    final_w = gate_trained.gate_linear.weight.item()
    final_b = gate_trained.gate_linear.bias.item()
    w_shift = abs(final_w - 1.0) / 1.0 * 100
    b_shift = abs(final_b - 0.0)

    print("\n" + "="*58)
    print("DIAGNOSTIC SUMMARY")
    print("="*58)
    print(f"1. Did the gate learn?")
    print(f"   weight: 1.0000 -> {final_w:.6f}  ({w_shift:.2f}% shift)")
    print(f"   bias:   0.0000 -> {final_b:.6f}  ({b_shift:.4f} abs)")
    if w_shift > 5:
        print(f"   -> YES — weight moved significantly")
    elif b_shift > 0.05:
        print(f"   -> PARTIAL — only bias learned (weight gradient was zero due to z=0 during training)")
    else:
        print(f"   -> NO — weights barely moved")

    log_df = pd.read_csv(OUTDIR / "training_log.csv")
    first3 = log_df["loss"].iloc[:3].mean()
    last3  = log_df["loss"].iloc[-3:].mean()
    print(f"\n2. Did loss decrease?")
    print(f"   First 3 steps: {first3:.4f}  |  Last 3 steps: {last3:.4f}")
    print(f"   -> {'YES' if last3 < first3 else 'FLAT/NOISY'}")

    print(f"\n3. Accuracy comparison:")
    print(comp.to_string(index=False))
    delta_trained = acc_trained - pre_acc
    delta_init    = acc_init - pre_acc
    print(f"   Trained gate delta vs baseline: {delta_trained:+.1%}")
    print(f"   Init gate    delta vs baseline: {delta_init:+.1%}")
    if delta_trained > 0.02:
        print(f"   -> IMPROVEMENT")
    elif delta_trained >= -0.02:
        print(f"   -> NEUTRAL (within noise)")
    else:
        print(f"   -> REGRESSION")

    print(f"\n4. Regime-dependent gate behaviour (trained gate):")
    for r, v in regime_results.items():
        print(f"   {r:15s}: gate={v['gate_mean']:.4f}  F={v['fisher_mean']:.2f}")
    if len(regime_results) == 2:
        keys = list(regime_results.keys())
        diff = abs(regime_results[keys[0]]["gate_mean"] - regime_results[keys[1]]["gate_mean"])
        print(f"   |delta gate| = {diff:.4f}  -> {'REGIME-SENSITIVE' if diff > 0.02 else 'NOT SENSITIVE'}")

    print(f"\n[write] -> {OUTDIR.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        sys.exit(130)
