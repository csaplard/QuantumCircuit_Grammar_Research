#!/usr/bin/env python3
"""
dream_replay_poc_v2.py  —  Dream-analogue replay, B-architecture
=================================================================
Key fix vs v1: dropout is active during TRAINING (weight update),
not inference. This means the dream phase actually modifies the
LoRA adapter weights under the dropout constraint — analogous to
biological synaptic consolidation during REM.

Pipeline
--------
T0  : base model Fisher geometry
LoRA: fine-tune 100 steps on GSM8K training examples
T1  : post-training Fisher geometry
T2D       : dream fine-tune 100 steps, mid-layer dropout p=DROPOUT_P,
            content = GSM8K training examples  (structured + original)
T2C-struct: dream fine-tune 100 steps, mid-layer dropout p=DROPOUT_P,
            content = off-domain prompts       (structured + foreign)
T2C-content: dream fine-tune 100 steps, random-layer dropout,
            content = GSM8K training examples  (random + original)
T2C-rand  : dream fine-tune 100 steps, random-layer dropout,
            content = off-domain prompts       (random + foreign)

Each T2 condition runs from T1 state (state restored between conditions).
Bootstrap: BOOTSTRAP_K seeds per T2 condition.
Fisher measured WITHOUT dropout after each dream phase.

Metrics
-------
- Cohen's d per layer between all regime pairs
- Temporal participation ratio PR=(sum F)^2/sum(F^2)
- Fisher path speed = mean F(t)
- Nearest-centroid probe with LOO-CV (no overfitting)

Outputs  (results/dream_replay_v2/)
-------------------------------------
T0/T1 traces + layer_sep + metrics
T2D/T2C-struct/T2C-content/T2C-rand  layer_sep_mean + metrics
probe_loocv.json
summary.png
"""

import argparse
import copy
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────── constants ────────────────────────────────────────

MODEL_NAME   = "google/gemma-3-1b-it"
RESULTS_DIR  = Path("results/dream_replay_v2")
WINDOW       = 32
DROPOUT_P    = 0.03     # conservative — target: loosen, not destroy
LORA_STEPS   = 100
DREAM_STEPS  = 100      # 1:1 ratio with LORA_STEPS
BOOTSTRAP_K  = 3        # seeds per T2 condition (4 conditions × 3 = 12 T2 runs)
MAX_TOKENS   = 128
PROBE_LAYER  = 2

# Off-domain content for T2C conditions (clearly different from GSM8K math)
OFF_DOMAIN = [
    "Explain why the sky appears blue during the day.",
    "Describe the process of photosynthesis in simple terms.",
    "What are the main differences between mammals and reptiles?",
    "How does a rainbow form after rain?",
    "Why do leaves change color in autumn?",
    "Describe the water cycle from evaporation to precipitation.",
    "What causes the tides in the ocean?",
    "How do birds navigate during migration?",
    "Explain the difference between weather and climate.",
    "Why do we see lightning before we hear thunder?",
]

PROMPTS: dict[str, list[str]] = {
    "factual": [
        "Explain how vaccines work.",
        "What causes earthquakes?",
        "How does the human immune system function?",
        "Why is the sky blue?",
        "What is the greenhouse effect?",
        "How do computers store information?",
        "Explain the water cycle.",
    ],
    "mathematical": [
        "Solve: a train travels 60 km/h for 2.5 hours. How far does it go?",
        "What is the sum of all integers from 1 to 100?",
        "A rectangle has area 48 and perimeter 28. Find its dimensions.",
        "If 3x + 7 = 22, what is x?",
        "How many ways can you arrange 4 books on a shelf?",
        "A store offers 20% off then 10% off. What is the total discount?",
        "What is the derivative of x^3 + 2x^2 - 5x?",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Describe a city where time flows backwards.",
        "Write a poem about the first snowfall of winter.",
        "Imagine a world where music is forbidden. Describe one day in it.",
        "Write a letter from a tree to the forest around it.",
        "Describe what silence looks like.",
        "Write a myth explaining why stars appear at night.",
    ],
    "philosophical": [
        "What does it mean for something to truly exist?",
        "Is there a meaningful difference between memory and identity?",
        "Can a machine experience curiosity? What would that require?",
        "What is the relationship between language and thought?",
        "Is time a fundamental feature of reality or a construct of perception?",
        "What distinguishes knowledge from belief?",
        "If the universe is deterministic, what is the nature of choice?",
    ],
}

REGIMES = list(PROMPTS.keys())
REGIME_LABEL = {r: i for i, r in enumerate(REGIMES)}
N_PROMPTS = 7


# ─────────────────────────── model loading ────────────────────────────────────

def load_model(name: str = MODEL_NAME):
    print(f"[load] {name} ...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, device_map="auto",
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    mdl.eval()
    device = mdl.get_input_embeddings().weight.device
    print(f"[load] OK  device={device}", flush=True)
    return mdl, tok, device


def find_decoder_layers(model) -> nn.ModuleList:
    for path in ("model.layers", "model.model.layers",
                 "model.language_model.layers", "transformer.h"):
        obj = model
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False; break
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    best, best_n = None, 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_n:
            if any(x in type(mod[0]).__name__.lower() for x in ("layer", "block", "decoder")):
                best, best_n = mod, len(mod)
    if best:
        return best
    raise RuntimeError("Cannot find decoder layers")


# ─────────────────────────── Fisher monitor ───────────────────────────────────

class FisherMonitor:
    def __init__(self, window=WINDOW, probe_maxlen=32):
        self.window = window
        self.buffer: deque = deque(maxlen=window)
        self.history: list[float] = []
        self._probe: list[torch.Tensor] = []
        self._probe_maxlen = probe_maxlen
        self._handle = None

    def reset(self):
        self.buffer.clear(); self.history.clear(); self._probe.clear()

    def _hook(self, module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        vec = h[0, -1].detach().float().cpu()
        if len(self._probe) >= self._probe_maxlen:
            self._probe.pop(0)
        self._probe.append(vec)
        self.buffer.append(vec)
        if len(self.buffer) >= 2:
            deltas = [float(torch.mean((self.buffer[i] - self.buffer[i-1])**2))
                      for i in range(1, len(self.buffer))]
            self.history.append(float(np.mean(deltas)))
        return output

    def attach(self, layer): self._handle = layer.register_forward_hook(self._hook)
    def detach(self):
        if self._handle: self._handle.remove(); self._handle = None

    def probe_vector(self):
        if not self._probe: return None
        return torch.stack(self._probe).mean(0)


# ─────────────────────────── dropout hook ─────────────────────────────────────

class DropoutHookManager:
    """Adds activation dropout to specified layer outputs (active in train+eval)."""
    def __init__(self, layers, indices, p):
        self.layers = layers; self.indices = indices; self.p = p
        self._handles = []

    def _make_hook(self, p):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            h_d = F.dropout(h, p=p, training=True)   # always active
            return (h_d,) + output[1:] if isinstance(output, tuple) else h_d
        return hook

    def attach(self):
        for idx in self.indices:
            self._handles.append(
                self.layers[idx].register_forward_hook(self._make_hook(self.p))
            )

    def detach(self):
        for h in self._handles: h.remove()
        self._handles.clear()


def mid_layer_indices(n): return list(range(n // 3, 2 * n // 3))
def random_layer_indices(n, count, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=count, replace=False).tolist())


# ─────────────────────────── Fisher measurement ───────────────────────────────

@torch.no_grad()
def generate_tokens(model, tokenizer, prompt, device, max_new_tokens=MAX_TOKENS):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt").to(device)
    ids = enc["input_ids"]; mask = enc["attention_mask"]; past = None
    for _ in range(max_new_tokens):
        cur = ids[:, -1:] if past is not None else ids
        out = model(input_ids=cur, attention_mask=mask,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        mask = torch.cat([mask, torch.ones(1, 1, device=device, dtype=mask.dtype)], dim=1)
        if nxt.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_condition(model, tokenizer, device, layers, label,
                  seed=42, max_tokens=MAX_TOKENS):
    """Measure Fisher on all regimes × prompts. No dropout during measurement."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    n = len(layers)
    monitors = [FisherMonitor() for _ in range(n)]
    for m, layer in zip(monitors, layers): m.attach(layer)

    traces = defaultdict(lambda: defaultdict(dict))
    probe_vecs = defaultdict(dict)
    total = len(REGIMES) * N_PROMPTS; done = 0; t0 = time.time()

    for regime in REGIMES:
        for pi, prompt in enumerate(PROMPTS[regime]):
            for m in monitors: m.reset()
            torch.manual_seed(seed + pi * 13)
            generate_tokens(model, tokenizer, prompt, device, max_tokens)
            for li, m in enumerate(monitors):
                traces[regime][pi][li] = m.history[:]
            probe_vecs[regime][pi] = monitors[PROBE_LAYER].probe_vector()
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done < total else 0
            print(f"  [{label}] {done}/{total}  {regime} p={pi}"
                  f"  {elapsed/60:.1f}m  ETA {eta/60:.1f}m", flush=True)

    for m in monitors: m.detach()
    return {"traces": {k: dict(v) for k, v in traces.items()},
            "probe_vecs": dict(probe_vecs)}


# ─────────────────────────── dream fine-tuning (B arch) ───────────────────────

def dream_finetune(model, tokenizer, device, layers,
                   content_data, dropout_indices, dropout_p,
                   n_steps=DREAM_STEPS, seed=42):
    """
    Fine-tune LoRA weights with activation dropout active on specific layers.
    Dropout hooks modify forward activations DURING training → weight updates
    incorporate the dropout constraint (B architecture).

    content_data : list of dicts with "question" and "answer" keys
    """
    dropout_mgr = DropoutHookManager(layers, dropout_indices, dropout_p)
    dropout_mgr.attach()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("[dream] WARNING: no trainable parameters — is PEFT applied?", flush=True)
        dropout_mgr.detach()
        return

    opt = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    model.train()
    torch.manual_seed(seed)

    for step in range(n_steps):
        ex = content_data[step % len(content_data)]
        q = ex.get("question", ex) if isinstance(ex, dict) else ex
        a = ex.get("answer", "") if isinstance(ex, dict) else ""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q},
             {"role": "assistant", "content": a}],
            tokenize=False, add_generation_prompt=False,
        )
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        labels = enc["input_ids"].clone()
        labels[:, :labels.shape[1] // 2] = -100   # mask prompt tokens
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward()
        if (step + 1) % 4 == 0:
            opt.step(); opt.zero_grad()
        if (step + 1) % 25 == 0:
            print(f"  [dream] step {step+1}/{n_steps}  loss={loss.item():.4f}", flush=True)

    model.eval()
    dropout_mgr.detach()


def save_t1_state(model) -> dict:
    """Clone all trainable (LoRA) parameter tensors."""
    return {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_t1_state(model, state: dict):
    """Restore LoRA parameters to saved T1 state."""
    for n, p in model.named_parameters():
        if n in state:
            p.data.copy_(state[n])


# ─────────────────────────── LoRA fine-tuning ─────────────────────────────────

def lora_finetune(model, tokenizer, device, n_steps=LORA_STEPS):
    from peft import LoraConfig, TaskType, get_peft_model
    cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,    # no dropout during LoRA — only during dream phase
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    training_data = [ds[i] for i in range(50)]

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01,
    )
    model.train(); torch.manual_seed(42)
    for step in range(n_steps):
        ex = training_data[step % len(training_data)]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]},
             {"role": "assistant", "content": ex["answer"]}],
            tokenize=False, add_generation_prompt=False,
        )
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        labels = enc["input_ids"].clone()
        labels[:, :labels.shape[1] // 2] = -100
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward()
        if (step + 1) % 4 == 0:
            opt.step(); opt.zero_grad()
        if (step + 1) % 25 == 0:
            print(f"  [lora] step {step+1}/{n_steps}  loss={loss.item():.4f}", flush=True)

    model.eval()
    return model, training_data


# ─────────────────────────── metrics ──────────────────────────────────────────

def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2: return 0.0
    vp = ((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)
    return float(abs(a.mean() - b.mean()) / np.sqrt(vp + 1e-12))


def compute_layer_sep(traces, n_layers):
    rows = []
    for li in range(n_layers):
        rv = {}
        for regime in REGIMES:
            vals = []
            for pi in range(N_PROMPTS): vals.extend(traces[regime][pi].get(li, []))
            rv[regime] = np.array(vals) if vals else np.array([0.0])
        for i, ra in enumerate(REGIMES):
            for rb in REGIMES[i+1:]:
                rows.append({"layer": li, "regime_a": ra, "regime_b": rb,
                              "cohens_d": round(cohen_d(rv[ra], rv[rb]), 4)})
    return rows


def participation_ratio(traces):
    out = {}
    for regime in REGIMES:
        vals = []
        for pi in range(N_PROMPTS):
            for li in traces[regime][pi]: vals.extend(traces[regime][pi][li])
        f = np.array(vals) + 1e-12
        out[regime] = round(float(f.sum()**2 / (f**2).sum()), 4)
    return out


def path_speed(traces):
    out = {}
    for regime in REGIMES:
        vals = []
        for pi in range(N_PROMPTS):
            for li in traces[regime][pi]: vals.extend(traces[regime][pi][li])
        out[regime] = round(float(np.mean(vals)) if vals else 0.0, 4)
    return out


# ─────────────────────────── nearest-centroid probe (LOO-CV) ──────────────────

def nearest_centroid_loocv(probe_vecs: dict) -> float:
    """
    Leave-one-out cross-validated nearest-centroid classifier.
    No free parameters → no overfitting risk.
    Operates on raw hidden-state vectors at PROBE_LAYER.
    """
    X, y = [], []
    for regime in REGIMES:
        for pi in range(N_PROMPTS):
            vec = probe_vecs.get(regime, {}).get(pi)
            if vec is not None:
                X.append(vec.float().numpy())
                y.append(REGIME_LABEL[regime])
    if len(X) < 4: return float("nan")
    X = np.stack(X); y = np.array(y)
    correct = 0
    for i in range(len(X)):
        # Compute centroid for each class EXCLUDING point i
        centroids = {}
        for label in range(len(REGIMES)):
            mask = (y == label) & (np.arange(len(y)) != i)
            if mask.sum() > 0:
                centroids[label] = X[mask].mean(0)
        # Classify point i by nearest centroid
        dists = {label: np.linalg.norm(X[i] - c) for label, c in centroids.items()}
        pred = min(dists, key=dists.get)
        if pred == y[i]: correct += 1
    return round(correct / len(X), 4)


# ─────────────────────────── save helpers ─────────────────────────────────────

def save_traces(traces, path):
    rows = []
    for regime in REGIMES:
        for pi in range(N_PROMPTS):
            for li, vals in traces[regime][pi].items():
                for t, v in enumerate(vals):
                    rows.append({"regime": regime, "prompt": pi, "layer": li, "t": t, "F": v})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime","prompt","layer","t","F"])
        w.writeheader(); w.writerows(rows)


def save_layer_sep(rows, path):
    fields = ["layer", "regime_a", "regime_b", "cohens_d", "cohens_d_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)


def aggregate_sep(sep_list):
    combined = defaultdict(list)
    for sep in sep_list:
        for row in sep:
            combined[(row["layer"], row["regime_a"], row["regime_b"])].append(row["cohens_d"])
    return [{"layer": k[0], "regime_a": k[1], "regime_b": k[2],
             "cohens_d": round(float(np.mean(vs)), 4),
             "cohens_d_std": round(float(np.std(vs)), 4)}
            for k, vs in sorted(combined.items())]


# ─────────────────────────── plotting ─────────────────────────────────────────

def plot_summary(all_results, n_layers, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib missing", flush=True); return

    T2_CONDS = ["T2D", "T2C-struct", "T2C-content", "T2C-rand"]
    colors = {"T0": "black", "T1": "navy",
              "T2D": "steelblue", "T2C-struct": "mediumseagreen",
              "T2C-content": "tomato", "T2C-rand": "orange"}
    lss = {"T0": "-", "T1": "--", "T2D": "-", "T2C-struct": "-.",
           "T2C-content": ":", "T2C-rand": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Dream Replay v2 — B architecture (dropout during training)\n"
                 f"p={DROPOUT_P}  LoRA={LORA_STEPS}steps  Dream={DREAM_STEPS}steps",
                 fontsize=13, fontweight="bold")

    cond_order = [c for c in ["T0","T1","T2D","T2C-struct","T2C-content","T2C-rand"]
                  if c in all_results]

    # Panel 1: Cohen's d per layer
    ax = axes[0, 0]
    for cond in cond_order:
        rows = all_results[cond].get("layer_sep", [])
        if not rows: continue
        layers_idx = sorted(set(r["layer"] for r in rows))
        mean_d = [np.mean([r["cohens_d"] for r in rows if r["layer"]==li])
                  for li in layers_idx]
        ax.plot(layers_idx, mean_d, label=cond, color=colors.get(cond,"gray"),
                ls=lss.get(cond,"-"), lw=2)
        if cond in T2_CONDS:
            std_d = [np.mean([r.get("cohens_d_std",0) for r in rows if r["layer"]==li])
                     for li in layers_idx]
            ax.fill_between(layers_idx, np.array(mean_d)-np.array(std_d),
                            np.array(mean_d)+np.array(std_d),
                            alpha=0.12, color=colors.get(cond,"gray"))
    ax.axvspan(0, n_layers//3, alpha=0.05, color="green")
    ax.axvspan(n_layers//3, 2*n_layers//3, alpha=0.05, color="orange")
    ax.axvspan(2*n_layers//3, n_layers, alpha=0.05, color="purple")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Cohen's d")
    ax.set_title("Regime separation — 4 regimes × 6 pairs"); ax.legend(fontsize=8)

    # Panel 2: Participation ratio
    ax = axes[0, 1]
    x = np.arange(len(REGIMES)); bw = 0.8 / max(len(cond_order), 1)
    for ci, cond in enumerate(cond_order):
        pr = all_results[cond].get("participation", {})
        ax.bar(x + ci*bw, [pr.get(r,0) for r in REGIMES], bw,
               label=cond, color=colors.get(cond,"gray"), alpha=0.75)
    ax.set_xticks(x + bw*len(cond_order)/2)
    ax.set_xticklabels(REGIMES, rotation=15, fontsize=9)
    ax.set_ylabel("Participation ratio"); ax.set_title("Temporal PR (high=diffuse)")
    ax.legend(fontsize=8)

    # Panel 3: Path speed
    ax = axes[1, 0]
    for ci, cond in enumerate(cond_order):
        ps = all_results[cond].get("path_speed", {})
        ax.bar(x + ci*bw, [ps.get(r,0) for r in REGIMES], bw,
               label=cond, color=colors.get(cond,"gray"), alpha=0.75)
    ax.set_xticks(x + bw*len(cond_order)/2)
    ax.set_xticklabels(REGIMES, rotation=15, fontsize=9)
    ax.set_ylabel("Mean F(t)"); ax.set_title("Fisher path speed per regime")
    ax.legend(fontsize=8)

    # Panel 4: Probe LOO-CV
    ax = axes[1, 1]
    pt = all_results.get("probe_loocv", {})
    if pt:
        keys = list(pt.keys())
        vals = [pt[k] for k in keys]
        bcols = [colors.get(k.split("_s")[0], "gray") for k in keys]
        ax.bar(range(len(keys)), vals, color=bcols, alpha=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, fontsize=8)
        ax.axhline(1/len(REGIMES), color="gray", ls="--",
                   label=f"chance {1/len(REGIMES):.0%}")
        ax.set_ylim(0, 1.05); ax.set_ylabel("LOO-CV accuracy")
        ax.set_title("Nearest-centroid probe (LOO-CV)\nT0-trained, tested on T2 hidden states")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()
    print(f"[plot] -> {out}", flush=True)


# ─────────────────────────── main ─────────────────────────────────────────────

def mean_d_zone(rows, n_layers, zone):
    lo = {"early":0, "mid":n_layers//3, "late":2*n_layers//3}[zone]
    hi = {"early":n_layers//3, "mid":2*n_layers//3, "late":n_layers}[zone]
    v  = [r["cohens_d"] for r in rows if lo <= r["layer"] < hi]
    return round(float(np.mean(v)), 3) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default=MODEL_NAME)
    ap.add_argument("--bootstrap-k",  type=int,   default=BOOTSTRAP_K)
    ap.add_argument("--dropout-p",    type=float, default=DROPOUT_P)
    ap.add_argument("--lora-steps",   type=int,   default=LORA_STEPS)
    ap.add_argument("--dream-steps",  type=int,   default=DREAM_STEPS)
    ap.add_argument("--max-tokens",   type=int,   default=MAX_TOKENS)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()
    all_results: dict = {}
    probe_loocv: dict = {}

    # ── load ───────────────────────────────────────────────────────────────────
    model, tokenizer, device = load_model(args.model)
    layers   = find_decoder_layers(model)
    n_layers = len(layers)
    mid_idx  = mid_layer_indices(n_layers)
    print(f"[setup] n_layers={n_layers}  mid={mid_idx}", flush=True)

    # ── T0 ─────────────────────────────────────────────────────────────────────
    print("\n===== T0: base model =====", flush=True)
    t0 = run_condition(model, tokenizer, device, layers, "T0", seed=42,
                       max_tokens=args.max_tokens)
    save_traces(t0["traces"], RESULTS_DIR / "T0_traces.csv")
    t0_sep = compute_layer_sep(t0["traces"], n_layers)
    save_layer_sep(t0_sep, RESULTS_DIR / "T0_layer_sep.csv")
    all_results["T0"] = {
        "layer_sep":     t0_sep,
        "participation": participation_ratio(t0["traces"]),
        "path_speed":    path_speed(t0["traces"]),
    }
    save_json(all_results["T0"], RESULTS_DIR / "T0_metrics.json")
    probe_loocv["T0"] = nearest_centroid_loocv(t0["probe_vecs"])
    print(f"[T0] early={mean_d_zone(t0_sep,n_layers,'early')}  "
          f"mid={mean_d_zone(t0_sep,n_layers,'mid')}  "
          f"late={mean_d_zone(t0_sep,n_layers,'late')}  "
          f"probe_loocv={probe_loocv['T0']:.2%}", flush=True)

    # ── LoRA fine-tuning ────────────────────────────────────────────────────────
    print("\n===== LoRA fine-tuning =====", flush=True)
    model, gsm8k_data = lora_finetune(model, tokenizer, device, n_steps=args.lora_steps)
    layers = find_decoder_layers(model)
    mid_idx = mid_layer_indices(len(layers))

    # ── T1 ─────────────────────────────────────────────────────────────────────
    print("\n===== T1: post-training =====", flush=True)
    t1 = run_condition(model, tokenizer, device, layers, "T1", seed=42,
                       max_tokens=args.max_tokens)
    save_traces(t1["traces"], RESULTS_DIR / "T1_traces.csv")
    t1_sep = compute_layer_sep(t1["traces"], n_layers)
    save_layer_sep(t1_sep, RESULTS_DIR / "T1_layer_sep.csv")
    all_results["T1"] = {
        "layer_sep":     t1_sep,
        "participation": participation_ratio(t1["traces"]),
        "path_speed":    path_speed(t1["traces"]),
    }
    save_json(all_results["T1"], RESULTS_DIR / "T1_metrics.json")
    probe_loocv["T1"] = nearest_centroid_loocv(t1["probe_vecs"])
    print(f"[T1] early={mean_d_zone(t1_sep,n_layers,'early')}  "
          f"mid={mean_d_zone(t1_sep,n_layers,'mid')}  "
          f"late={mean_d_zone(t1_sep,n_layers,'late')}  "
          f"probe_loocv={probe_loocv['T1']:.2%}", flush=True)

    # Save T1 LoRA state for restoration between T2 conditions
    t1_state = save_t1_state(model)
    print(f"[T1] state saved ({len(t1_state)} tensors)", flush=True)

    # Off-domain content for T2C conditions (formatted like training data)
    off_domain_data = [{"question": q, "answer": "This is a factual question."}
                       for q in OFF_DOMAIN]

    # T2 factorial design
    t2_conditions = {
        "T2D":         {"dropout_idx_fn": lambda seed: mid_idx,
                        "content":        gsm8k_data,
                        "label":          "structured dropout + training content"},
        "T2C-struct":  {"dropout_idx_fn": lambda seed: mid_idx,
                        "content":        off_domain_data,
                        "label":          "structured dropout + off-domain content"},
        "T2C-content": {"dropout_idx_fn": lambda seed: random_layer_indices(
                            len(layers), len(mid_idx), seed=200+seed),
                        "content":        gsm8k_data,
                        "label":          "random dropout + training content"},
        "T2C-rand":    {"dropout_idx_fn": lambda seed: random_layer_indices(
                            len(layers), len(mid_idx), seed=200+seed),
                        "content":        off_domain_data,
                        "label":          "random dropout + off-domain content"},
    }

    # ── T2 bootstrap ───────────────────────────────────────────────────────────
    t2_seps: dict[str, list] = defaultdict(list)
    t2_pr:   dict[str, dict] = defaultdict(lambda: defaultdict(list))
    t2_ps:   dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for cond_name, cond_cfg in t2_conditions.items():
        for seed_i in range(args.bootstrap_k):
            print(f"\n===== {cond_name} seed={seed_i} | {cond_cfg['label']} =====",
                  flush=True)

            # Restore T1 state before each dream fine-tune
            restore_t1_state(model, t1_state)

            # Dream fine-tuning (B architecture: dropout active during weight update)
            dropout_idx = cond_cfg["dropout_idx_fn"](seed_i)
            dream_finetune(
                model, tokenizer, device, layers,
                content_data=cond_cfg["content"],
                dropout_indices=dropout_idx,
                dropout_p=args.dropout_p,
                n_steps=args.dream_steps,
                seed=100 + seed_i,
            )

            # Measure Fisher WITHOUT dropout
            data = run_condition(model, tokenizer, device, layers,
                                 label=f"{cond_name}_s{seed_i}",
                                 seed=100 + seed_i,
                                 max_tokens=args.max_tokens)
            save_traces(data["traces"], RESULTS_DIR / f"{cond_name}_traces_s{seed_i}.csv")
            sep = compute_layer_sep(data["traces"], n_layers)
            save_layer_sep(sep, RESULTS_DIR / f"{cond_name}_layer_sep_s{seed_i}.csv")
            t2_seps[cond_name].append(sep)

            pr = participation_ratio(data["traces"])
            ps = path_speed(data["traces"])
            for r in REGIMES:
                t2_pr[cond_name][r].append(pr.get(r, 0))
                t2_ps[cond_name][r].append(ps.get(r, 0))

            probe_acc = nearest_centroid_loocv(data["probe_vecs"])
            probe_loocv[f"{cond_name}_s{seed_i}"] = probe_acc
            print(f"  [{cond_name} s={seed_i}] "
                  f"early={mean_d_zone(sep,n_layers,'early')}  "
                  f"mid={mean_d_zone(sep,n_layers,'mid')}  "
                  f"probe={probe_acc:.2%}  "
                  f"elapsed={( time.time()-t_global)/60:.1f}m", flush=True)

    # Save T2 aggregated results
    for cond_name in t2_conditions:
        sep_mean = aggregate_sep(t2_seps[cond_name])
        save_layer_sep(sep_mean, RESULTS_DIR / f"{cond_name}_layer_sep_mean.csv")
        metrics = {
            "layer_sep":          sep_mean,
            "participation":      {r: round(float(np.mean(t2_pr[cond_name][r])),4) for r in REGIMES},
            "path_speed":         {r: round(float(np.mean(t2_ps[cond_name][r])),4) for r in REGIMES},
            "participation_std":  {r: round(float(np.std(t2_pr[cond_name][r])),4)  for r in REGIMES},
            "path_speed_std":     {r: round(float(np.std(t2_ps[cond_name][r])),4)  for r in REGIMES},
        }
        all_results[cond_name] = metrics
        save_json(metrics, RESULTS_DIR / f"{cond_name}_metrics.json")

    save_json(probe_loocv, RESULTS_DIR / "probe_loocv.json")
    all_results["probe_loocv"] = probe_loocv

    # ── summary ────────────────────────────────────────────────────────────────
    total_min = (time.time() - t_global) / 60
    print(f"\n{'='*65}\nSUMMARY  (total={total_min:.1f}m)\n{'='*65}", flush=True)
    for cond in ["T0","T1","T2D","T2C-struct","T2C-content","T2C-rand"]:
        if cond not in all_results: continue
        sep = all_results[cond].get("layer_sep", [])
        print(f"\n  {cond}:", flush=True)
        print(f"    Cohen's d  early={mean_d_zone(sep,n_layers,'early')}  "
              f"mid={mean_d_zone(sep,n_layers,'mid')}  "
              f"late={mean_d_zone(sep,n_layers,'late')}", flush=True)
        print(f"    PR:  {all_results[cond].get('participation',{})}", flush=True)
        print(f"    PS:  {all_results[cond].get('path_speed',{})}", flush=True)

    print("\n  Probe LOO-CV:", flush=True)
    for k, v in probe_loocv.items():
        print(f"    {k}: {v:.2%}", flush=True)

    print("\n  Key comparisons (mid-layer Cohen's d):", flush=True)
    for cond in ["T2D","T2C-struct","T2C-content","T2C-rand"]:
        if cond in all_results:
            sep = all_results[cond]["layer_sep"]
            d = mean_d_zone(sep, n_layers, "mid")
            t1_d = mean_d_zone(t1_sep, n_layers, "mid")
            print(f"    {cond}: {d}  (vs T1={t1_d}  delta={d-t1_d:+.3f})", flush=True)

    plot_summary(all_results, n_layers, RESULTS_DIR / "summary.png")
    print(f"\n[done] -> {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
