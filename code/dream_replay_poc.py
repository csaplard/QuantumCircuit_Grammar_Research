#!/usr/bin/env python3
"""
dream_replay_poc.py
====================
Dream-analogue replay: Fisher geometry under structured vs random
activation dropout.

Conditions
----------
T0  : base model, no dropout (baseline geometry)
T1  : after LoRA fine-tune on 50 GSM8K examples (T1 = T0 if --skip-training)
T2D : mid-layer activation dropout p=DROPOUT_P  ("dream" – structured)
T2C : random-layer activation dropout, same p and n_layers (control)

Bootstrap k=BOOTSTRAP_K seeds for T2D and T2C.

Metrics
-------
- Cohen's d per layer between all regime pairs
- Temporal participation ratio PR = (sum F_i)^2 / sum(F_i^2)
- Fisher path speed tau = mean(F(t))
- Linear probe transfer: trained on T0 hidden states, tested on T2D / T2C

Outputs  (results/dream_replay/)
---------------------------------
T0_traces.csv, T1_traces.csv
T2D_traces_s{k}.csv, T2C_traces_s{k}.csv
T0_layer_sep.csv, T1_layer_sep.csv
T2D_layer_sep_mean.csv, T2C_layer_sep_mean.csv
T0_metrics.json, T1_metrics.json, T2D_metrics.json, T2C_metrics.json
probe_transfer.json
summary.png
"""

import argparse
import csv
import json
import os
import sys
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
RESULTS_DIR  = Path("results/dream_replay")
WINDOW       = 32      # Fisher sliding window
DROPOUT_P    = 0.10    # activation dropout probability for T2 conditions
BOOTSTRAP_K  = 5       # random seeds for T2 bootstrap
MAX_TOKENS   = 128     # tokens generated per prompt
LORA_STEPS   = 50      # LoRA fine-tune steps
PROBE_LAYER  = 2       # layer index used for probe hidden-state collection

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
        "A store offers 20% off then an additional 10% off. Total discount?",
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
N_PROMPTS = 7   # prompts per regime


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
        name,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    mdl.eval()
    device = mdl.get_input_embeddings().weight.device
    print(f"[load] OK  device={device}", flush=True)
    return mdl, tok, device


def find_decoder_layers(model) -> nn.ModuleList:
    """Return the nn.ModuleList of decoder layers, with heuristic fallback."""
    for path in (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "transformer.h",
    ):
        obj = model
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    # Heuristic: longest ModuleList whose first element looks like a decoder block
    best, best_n = None, 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_n:
            cname = type(mod[0]).__name__.lower()
            if any(x in cname for x in ("layer", "block", "decoder")):
                best, best_n = mod, len(mod)
    if best is not None:
        return best
    raise RuntimeError("Cannot find decoder layers in model")


# ─────────────────────────── Fisher monitor ───────────────────────────────────

class FisherMonitor:
    """
    Attaches a forward hook to a single layer.
    Tracks F(t) = mean(delta_h^2) over a sliding window.
    Also accumulates hidden states for probe-transfer use.
    """

    def __init__(self, window: int = WINDOW, probe_maxlen: int = 32):
        self.window = window
        self.buffer: deque = deque(maxlen=window)
        self.history: list[float] = []
        self._probe_states: list[torch.Tensor] = []
        self._probe_maxlen = probe_maxlen
        self._handle = None

    def reset(self):
        self.buffer.clear()
        self.history.clear()
        self._probe_states.clear()

    def _hook(self, module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        vec = h[0, -1].detach().float().cpu()   # last-token hidden state

        # Probe accumulation (capped)
        if len(self._probe_states) >= self._probe_maxlen:
            self._probe_states.pop(0)
        self._probe_states.append(vec)

        # Fisher sliding window
        self.buffer.append(vec)
        if len(self.buffer) >= 2:
            deltas = [
                float(torch.mean((self.buffer[i] - self.buffer[i - 1]) ** 2))
                for i in range(1, len(self.buffer))
            ]
            self.history.append(float(np.mean(deltas)))
        return output

    def attach(self, layer):
        self._handle = layer.register_forward_hook(self._hook)

    def detach(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def probe_vector(self) -> torch.Tensor | None:
        if not self._probe_states:
            return None
        return torch.stack(self._probe_states).mean(dim=0)


# ─────────────────────────── dropout hook ─────────────────────────────────────

class DropoutHookManager:
    """
    Applies activation dropout to the outputs of specified layer indices.
    Dropout is applied with training=True so it is active during inference.

    Note: this is output/activation dropout, not attention-weight dropout.
    The effect is equivalent for our purposes: it forces information to route
    through alternative pathways in subsequent layers.
    """

    def __init__(self, layers: nn.ModuleList, indices: list[int], p: float):
        self.layers = layers
        self.indices = indices
        self.p = p
        self._handles = []

    def _make_hook(self, p: float):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            h_d = F.dropout(h, p=p, training=True)
            if isinstance(output, tuple):
                return (h_d,) + output[1:]
            return h_d
        return hook

    def attach(self):
        for idx in self.indices:
            h = self.layers[idx].register_forward_hook(self._make_hook(self.p))
            self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def mid_layer_indices(n: int) -> list[int]:
    """Central third of layers (indices lo..hi-1)."""
    lo, hi = n // 3, 2 * n // 3
    return list(range(lo, hi))


def random_layer_indices(n: int, count: int, seed: int) -> list[int]:
    """Random subset of `count` distinct layer indices."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=count, replace=False).tolist())


# ─────────────────────────── generation ───────────────────────────────────────

@torch.no_grad()
def generate_tokens(
    model, tokenizer, prompt: str, device,
    max_new_tokens: int = MAX_TOKENS,
) -> str:
    """Token-by-token autoregressive generation (hooks fire per token)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    past = None

    for _ in range(max_new_tokens):
        cur_ids = input_ids[:, -1:] if past is not None else input_ids
        out = model(
            input_ids=cur_ids,
            attention_mask=attn_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
        input_ids = torch.cat([input_ids, nxt], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)],
            dim=1,
        )
        if nxt.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(
        input_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )


# ─────────────────────────── condition runner ─────────────────────────────────

def run_condition(
    model, tokenizer, device,
    layers: nn.ModuleList,
    label: str,
    dropout_indices: list[int] | None = None,
    dropout_p: float = 0.0,
    seed: int = 42,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """
    Run all regimes x prompts with Fisher monitors on every layer.
    Returns:
      traces[regime][prompt_i][layer_i] = list[float]  (F(t) values)
      probe_vecs[regime][prompt_i]      = Tensor(hidden_dim) | None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n = len(layers)
    monitors = [FisherMonitor() for _ in range(n)]
    for m, layer in zip(monitors, layers):
        m.attach(layer)

    dropout_mgr = None
    if dropout_indices:
        dropout_mgr = DropoutHookManager(layers, dropout_indices, dropout_p)
        dropout_mgr.attach()

    traces: dict = defaultdict(lambda: defaultdict(dict))
    probe_vecs: dict = defaultdict(dict)

    total = len(REGIMES) * N_PROMPTS
    done = 0
    t0 = time.time()

    for regime in REGIMES:
        for pi, prompt in enumerate(PROMPTS[regime]):
            for m in monitors:
                m.reset()

            torch.manual_seed(seed + pi * 13)   # deterministic per prompt
            generate_tokens(model, tokenizer, prompt, device, max_tokens)

            for li, m in enumerate(monitors):
                traces[regime][pi][li] = m.history[:]

            probe_vecs[regime][pi] = monitors[PROBE_LAYER].probe_vector()

            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done < total else 0
            print(
                f"  [{label}] {done}/{total}  {regime} p={pi}"
                f"  {elapsed/60:.1f}m elapsed  ETA {eta/60:.1f}m",
                flush=True,
            )

    for m in monitors:
        m.detach()
    if dropout_mgr:
        dropout_mgr.detach()

    return {"traces": {k: dict(v) for k, v in traces.items()},
            "probe_vecs": dict(probe_vecs)}


# ─────────────────────────── metrics ──────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_p = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (
        len(a) + len(b) - 2
    )
    return float(abs(a.mean() - b.mean()) / np.sqrt(var_p + 1e-12))


def compute_layer_separation(traces: dict, n_layers: int) -> list[dict]:
    """Per-layer Cohen's d between every pair of regimes."""
    rows = []
    for li in range(n_layers):
        regime_vals: dict[str, np.ndarray] = {}
        for regime in REGIMES:
            vals = []
            for pi in range(N_PROMPTS):
                vals.extend(traces[regime][pi].get(li, []))
            regime_vals[regime] = np.array(vals) if vals else np.array([0.0])
        for i, ra in enumerate(REGIMES):
            for rb in REGIMES[i + 1:]:
                rows.append({
                    "layer": li,
                    "regime_a": ra,
                    "regime_b": rb,
                    "cohens_d": round(cohen_d(regime_vals[ra], regime_vals[rb]), 4),
                })
    return rows


def compute_participation_ratio(traces: dict) -> dict[str, float]:
    """
    Temporal participation ratio per regime:
      PR = (sum F_i)^2 / sum(F_i^2)
    High PR: signal diffuse across time. Low PR: spiky (few dominant steps).
    """
    out = {}
    for regime in REGIMES:
        vals = []
        for pi in range(N_PROMPTS):
            for li in traces[regime][pi]:
                vals.extend(traces[regime][pi][li])
        f = np.array(vals, dtype=float) + 1e-12
        out[regime] = round(float(f.sum() ** 2 / (f ** 2).sum()), 4)
    return out


def compute_path_speed(traces: dict) -> dict[str, float]:
    """Mean F(t) per regime (Fisher path speed proxy)."""
    out = {}
    for regime in REGIMES:
        vals = []
        for pi in range(N_PROMPTS):
            for li in traces[regime][pi]:
                vals.extend(traces[regime][pi][li])
        out[regime] = round(float(np.mean(vals)) if vals else 0.0, 4)
    return out


# ─────────────────────────── linear probe ─────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_probe_dataset(
    probe_vecs: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    X, y = [], []
    for regime in REGIMES:
        label = REGIME_LABEL[regime]
        for pi in range(N_PROMPTS):
            vec = probe_vecs.get(regime, {}).get(pi)
            if vec is not None:
                X.append(vec.float())
                y.append(label)
    if not X:
        return None, None
    return torch.stack(X), torch.tensor(y, dtype=torch.long)


def train_linear_probe(
    X: torch.Tensor, y: torch.Tensor, n_epochs: int = 500
) -> LinearProbe:
    probe = LinearProbe(X.shape[1], len(REGIMES))
    opt = torch.optim.Adam(probe.parameters(), lr=5e-4, weight_decay=1e-3)
    for _ in range(n_epochs):
        loss = F.cross_entropy(probe(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe


def eval_probe(probe: LinearProbe, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = probe(X).argmax(dim=1)
    return float((preds == y).float().mean())


# ─────────────────────────── LoRA fine-tuning ────────────────────────────────

def lora_finetune(model, tokenizer, device, n_steps: int = LORA_STEPS):
    """
    Fine-tune the model on 50 GSM8K training examples with LoRA.
    Requires: pip install peft datasets
    Falls back to no-op (T1 = T0) if either is missing.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        print("[lora] peft not installed — pip install peft to enable training", flush=True)
        print("[lora] Skipping: T1 = T0", flush=True)
        return model

    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
    except Exception as exc:
        print(f"[lora] Cannot load GSM8K ({exc}) — skipping training", flush=True)
        return model

    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.01,
    )

    model.train()
    torch.manual_seed(42)
    examples = [ds[i] for i in range(50)]

    for step in range(n_steps):
        ex = examples[step % len(examples)]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]},
             {"role": "assistant", "content": ex["answer"]}],
            tokenize=False, add_generation_prompt=False,
        )
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        labels = enc["input_ids"].clone()
        # Mask prompt tokens (crude heuristic: first half)
        labels[:, : labels.shape[1] // 2] = -100

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward()

        if (step + 1) % 4 == 0:
            opt.step()
            opt.zero_grad()

        if (step + 1) % 10 == 0:
            print(f"  [lora] step {step+1}/{n_steps}  loss={loss.item():.4f}", flush=True)

    model.eval()
    return model


# ─────────────────────────── save helpers ─────────────────────────────────────

def save_traces(traces: dict, path: Path):
    rows = []
    for regime in REGIMES:
        for pi in range(N_PROMPTS):
            for li, vals in traces[regime][pi].items():
                for t, v in enumerate(vals):
                    rows.append({"regime": regime, "prompt": pi, "layer": li, "t": t, "F": v})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "prompt", "layer", "t", "F"])
        w.writeheader(); w.writerows(rows)


def save_layer_sep(rows: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "regime_a", "regime_b", "cohens_d"])
        w.writeheader(); w.writerows(rows)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ─────────────────────────── bootstrap aggregation ────────────────────────────

def aggregate_sep(sep_list: list[list[dict]]) -> list[dict]:
    """Mean Cohen's d across bootstrap runs per (layer, pair)."""
    combined: dict = defaultdict(list)
    for sep in sep_list:
        for row in sep:
            combined[(row["layer"], row["regime_a"], row["regime_b"])].append(row["cohens_d"])
    return [
        {"layer": k[0], "regime_a": k[1], "regime_b": k[2],
         "cohens_d": round(float(np.mean(vs)), 4),
         "cohens_d_std": round(float(np.std(vs)), 4)}
        for k, vs in combined.items()
    ]


# ─────────────────────────── plotting ─────────────────────────────────────────

def plot_summary(all_results: dict, n_layers: int, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — skipping", flush=True)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dream Replay — Fisher Geometry Analysis", fontsize=14, fontweight="bold")

    cond_order = [c for c in ["T0", "T1", "T2D", "T2C"] if c in all_results]
    colors = {"T0": "black", "T1": "navy", "T2D": "steelblue", "T2C": "tomato"}
    lss    = {"T0": "-",    "T1": "--",   "T2D": "-",         "T2C": ":"}

    # Panel 1: Cohen's d per layer (mean across regime pairs)
    ax = axes[0, 0]
    for cond in cond_order:
        rows = all_results[cond].get("layer_sep", [])
        if not rows:
            continue
        layers_idx = sorted(set(r["layer"] for r in rows))
        mean_d = [np.mean([r["cohens_d"] for r in rows if r["layer"] == li])
                  for li in layers_idx]
        ax.plot(layers_idx, mean_d, label=cond, color=colors[cond], ls=lss[cond], lw=2)
    ax.axvspan(0, n_layers // 3, alpha=0.05, color="green", label="early")
    ax.axvspan(n_layers // 3, 2 * n_layers // 3, alpha=0.05, color="orange", label="mid")
    ax.axvspan(2 * n_layers // 3, n_layers, alpha=0.05, color="purple", label="late")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean Cohen's d (across pairs)")
    ax.set_title("Regime separation per layer")
    ax.legend(fontsize=8)

    # Panel 2: Participation ratio per regime
    ax = axes[0, 1]
    x = np.arange(len(REGIMES))
    bar_w = 0.8 / max(len(cond_order), 1)
    for ci, cond in enumerate(cond_order):
        pr = all_results[cond].get("participation", {})
        vals = [pr.get(r, 0) for r in REGIMES]
        ax.bar(x + ci * bar_w, vals, bar_w, label=cond,
               color=colors[cond], alpha=0.75)
    ax.set_xticks(x + bar_w * len(cond_order) / 2)
    ax.set_xticklabels(REGIMES, rotation=15, fontsize=9)
    ax.set_ylabel("Participation ratio")
    ax.set_title("Temporal participation ratio\n(high = diffuse signal)")
    ax.legend(fontsize=8)

    # Panel 3: Fisher path speed per regime
    ax = axes[1, 0]
    for ci, cond in enumerate(cond_order):
        ps = all_results[cond].get("path_speed", {})
        vals = [ps.get(r, 0) for r in REGIMES]
        ax.bar(x + ci * bar_w, vals, bar_w, label=cond,
               color=colors[cond], alpha=0.75)
    ax.set_xticks(x + bar_w * len(cond_order) / 2)
    ax.set_xticklabels(REGIMES, rotation=15, fontsize=9)
    ax.set_ylabel("Mean F(t)")
    ax.set_title("Fisher path speed")
    ax.legend(fontsize=8)

    # Panel 4: Probe transfer accuracy
    ax = axes[1, 1]
    pt = all_results.get("probe_transfer", {})
    if pt:
        labels_pt = list(pt.keys())
        vals_pt   = [pt[k] for k in labels_pt]
        bar_colors = ["black" if k == "T0_train" else
                      "steelblue" if k.startswith("T2D") else "tomato"
                      for k in labels_pt]
        ax.bar(range(len(labels_pt)), vals_pt, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(labels_pt)))
        ax.set_xticklabels(labels_pt, rotation=25, fontsize=8)
        ax.axhline(1 / len(REGIMES), color="gray", ls="--", label=f"chance ({1/len(REGIMES):.0%})")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title("Linear probe transfer\n(trained on T0, tested on T2D/T2C)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] -> {out}", flush=True)


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dream replay Fisher geometry experiment")
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip LoRA fine-tune (T1 = T0 measurements)")
    ap.add_argument("--model",       default=MODEL_NAME)
    ap.add_argument("--bootstrap-k", type=int,   default=BOOTSTRAP_K)
    ap.add_argument("--dropout-p",   type=float, default=DROPOUT_P)
    ap.add_argument("--max-tokens",  type=int,   default=MAX_TOKENS)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()
    all_results: dict = {}

    # ── load ───────────────────────────────────────────────────────────────────
    model, tokenizer, device = load_model(args.model)
    layers    = find_decoder_layers(model)
    n_layers  = len(layers)
    mid_idx   = mid_layer_indices(n_layers)
    print(f"[setup] n_layers={n_layers}", flush=True)
    print(f"[setup] mid-layer dropout indices: {mid_idx}", flush=True)

    # ── T0 ─────────────────────────────────────────────────────────────────────
    print("\n===== T0: base model, no dropout =====", flush=True)
    t0_data = run_condition(
        model, tokenizer, device, layers,
        label="T0", dropout_indices=None, seed=42,
        max_tokens=args.max_tokens,
    )
    save_traces(t0_data["traces"], RESULTS_DIR / "T0_traces.csv")
    t0_sep = compute_layer_separation(t0_data["traces"], n_layers)
    save_layer_sep(t0_sep, RESULTS_DIR / "T0_layer_sep.csv")
    all_results["T0"] = {
        "layer_sep":     t0_sep,
        "participation": compute_participation_ratio(t0_data["traces"]),
        "path_speed":    compute_path_speed(t0_data["traces"]),
    }
    save_json(all_results["T0"], RESULTS_DIR / "T0_metrics.json")
    print(f"[T0] participation: {all_results['T0']['participation']}", flush=True)
    print(f"[T0] path_speed:    {all_results['T0']['path_speed']}", flush=True)

    # ── train linear probe on T0 ───────────────────────────────────────────────
    probe_results: dict = {}
    probe_t0 = None
    X_t0, y_t0 = build_probe_dataset(t0_data["probe_vecs"])
    if X_t0 is not None:
        print(f"\n[probe] Training linear probe on T0 (n={len(y_t0)}) ...", flush=True)
        probe_t0 = train_linear_probe(X_t0, y_t0)
        acc = eval_probe(probe_t0, X_t0, y_t0)
        probe_results["T0_train"] = round(acc, 4)
        print(f"[probe] T0 train accuracy: {acc:.2%}  (should be well above {1/len(REGIMES):.0%})", flush=True)
    else:
        print("[probe] No probe vectors collected — skipping probe", flush=True)

    # ── LoRA fine-tuning ────────────────────────────────────────────────────────
    if not args.skip_training:
        print("\n===== LoRA fine-tuning =====", flush=True)
        model = lora_finetune(model, tokenizer, device)
        layers = find_decoder_layers(model)   # refresh after PEFT wraps model
        mid_idx = mid_layer_indices(len(layers))
    else:
        print("\n[skip] --skip-training: T1 will use base model measurements", flush=True)

    # ── T1 ─────────────────────────────────────────────────────────────────────
    print("\n===== T1: after training, no dropout =====", flush=True)
    t1_data = run_condition(
        model, tokenizer, device, layers,
        label="T1", dropout_indices=None, seed=42,
        max_tokens=args.max_tokens,
    )
    save_traces(t1_data["traces"], RESULTS_DIR / "T1_traces.csv")
    t1_sep = compute_layer_separation(t1_data["traces"], n_layers)
    save_layer_sep(t1_sep, RESULTS_DIR / "T1_layer_sep.csv")
    all_results["T1"] = {
        "layer_sep":     t1_sep,
        "participation": compute_participation_ratio(t1_data["traces"]),
        "path_speed":    compute_path_speed(t1_data["traces"]),
    }
    save_json(all_results["T1"], RESULTS_DIR / "T1_metrics.json")
    print(f"[T1] participation: {all_results['T1']['participation']}", flush=True)
    print(f"[T1] path_speed:    {all_results['T1']['path_speed']}", flush=True)

    # ── T2D and T2C bootstrap ───────────────────────────────────────────────────
    t2d_seps, t2c_seps = [], []
    t2d_pr: dict[str, list] = defaultdict(list)
    t2c_pr: dict[str, list] = defaultdict(list)
    t2d_ps: dict[str, list] = defaultdict(list)
    t2c_ps: dict[str, list] = defaultdict(list)

    for seed_i in range(args.bootstrap_k):

        # --- T2D: mid-layer (structured dream dropout) ---
        print(f"\n===== T2D seed={seed_i} | mid-layer dropout p={args.dropout_p} =====",
              flush=True)
        t2d = run_condition(
            model, tokenizer, device, layers,
            label=f"T2D_s{seed_i}",
            dropout_indices=mid_idx,
            dropout_p=args.dropout_p,
            seed=100 + seed_i,
            max_tokens=args.max_tokens,
        )
        save_traces(t2d["traces"], RESULTS_DIR / f"T2D_traces_s{seed_i}.csv")
        sep_d = compute_layer_separation(t2d["traces"], n_layers)
        save_layer_sep(sep_d, RESULTS_DIR / f"T2D_layer_sep_s{seed_i}.csv")
        t2d_seps.append(sep_d)
        pr_d = compute_participation_ratio(t2d["traces"])
        ps_d = compute_path_speed(t2d["traces"])
        for r in REGIMES:
            t2d_pr[r].append(pr_d.get(r, 0))
            t2d_ps[r].append(ps_d.get(r, 0))

        # Probe test on T2D
        if probe_t0 is not None:
            X, y = build_probe_dataset(t2d["probe_vecs"])
            if X is not None:
                acc = eval_probe(probe_t0, X, y)
                probe_results[f"T2D_s{seed_i}"] = round(acc, 4)
                print(f"[probe] T2D s={seed_i}: {acc:.2%}", flush=True)

        # --- T2C: random-layer (control dropout, matched intensity) ---
        rnd_idx = random_layer_indices(n_layers, len(mid_idx), seed=200 + seed_i)
        print(
            f"\n===== T2C seed={seed_i} | random-layer dropout {rnd_idx[:5]}... =====",
            flush=True,
        )
        t2c = run_condition(
            model, tokenizer, device, layers,
            label=f"T2C_s{seed_i}",
            dropout_indices=rnd_idx,
            dropout_p=args.dropout_p,
            seed=100 + seed_i,          # same base seed as T2D for matched comparison
            max_tokens=args.max_tokens,
        )
        save_traces(t2c["traces"], RESULTS_DIR / f"T2C_traces_s{seed_i}.csv")
        sep_c = compute_layer_separation(t2c["traces"], n_layers)
        save_layer_sep(sep_c, RESULTS_DIR / f"T2C_layer_sep_s{seed_i}.csv")
        t2c_seps.append(sep_c)
        pr_c = compute_participation_ratio(t2c["traces"])
        ps_c = compute_path_speed(t2c["traces"])
        for r in REGIMES:
            t2c_pr[r].append(pr_c.get(r, 0))
            t2c_ps[r].append(ps_c.get(r, 0))

        # Probe test on T2C
        if probe_t0 is not None:
            X, y = build_probe_dataset(t2c["probe_vecs"])
            if X is not None:
                acc = eval_probe(probe_t0, X, y)
                probe_results[f"T2C_s{seed_i}"] = round(acc, 4)
                print(f"[probe] T2C s={seed_i}: {acc:.2%}", flush=True)

        print(
            f"  [progress] bootstrap {seed_i+1}/{args.bootstrap_k} done  "
            f"elapsed {(time.time()-t_global)/60:.1f}m",
            flush=True,
        )

    # ── aggregate bootstrap ────────────────────────────────────────────────────
    t2d_sep_mean = aggregate_sep(t2d_seps)
    t2c_sep_mean = aggregate_sep(t2c_seps)
    save_layer_sep(t2d_sep_mean, RESULTS_DIR / "T2D_layer_sep_mean.csv")
    save_layer_sep(t2c_sep_mean, RESULTS_DIR / "T2C_layer_sep_mean.csv")

    all_results["T2D"] = {
        "layer_sep":     t2d_sep_mean,
        "participation": {r: round(float(np.mean(t2d_pr[r])), 4) for r in REGIMES},
        "path_speed":    {r: round(float(np.mean(t2d_ps[r])), 4) for r in REGIMES},
        "participation_std": {r: round(float(np.std(t2d_pr[r])), 4) for r in REGIMES},
        "path_speed_std":    {r: round(float(np.std(t2d_ps[r])), 4) for r in REGIMES},
    }
    all_results["T2C"] = {
        "layer_sep":     t2c_sep_mean,
        "participation": {r: round(float(np.mean(t2c_pr[r])), 4) for r in REGIMES},
        "path_speed":    {r: round(float(np.mean(t2c_ps[r])), 4) for r in REGIMES},
        "participation_std": {r: round(float(np.std(t2c_pr[r])), 4) for r in REGIMES},
        "path_speed_std":    {r: round(float(np.std(t2c_ps[r])), 4) for r in REGIMES},
    }
    all_results["probe_transfer"] = probe_results

    save_json(all_results["T2D"], RESULTS_DIR / "T2D_metrics.json")
    save_json(all_results["T2C"], RESULTS_DIR / "T2C_metrics.json")
    save_json(probe_results,      RESULTS_DIR / "probe_transfer.json")

    # ── summary printout ───────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for cond in ["T0", "T1", "T2D", "T2C"]:
        if cond not in all_results:
            continue
        sep  = all_results[cond].get("layer_sep", [])
        pr   = all_results[cond].get("participation", {})
        ps   = all_results[cond].get("path_speed", {})
        def mean_d(zone):
            lo = {"early": 0, "mid": n_layers//3, "late": 2*n_layers//3}[zone]
            hi = {"early": n_layers//3, "mid": 2*n_layers//3, "late": n_layers}[zone]
            v  = [r["cohens_d"] for r in sep if lo <= r["layer"] < hi]
            return round(float(np.mean(v)), 3) if v else 0.0
        print(f"\n  {cond}:", flush=True)
        print(f"    Cohen's d  early={mean_d('early')}  mid={mean_d('mid')}  late={mean_d('late')}", flush=True)
        print(f"    Participation: {pr}", flush=True)
        print(f"    Path speed:    {ps}", flush=True)

    print("\n  Probe transfer (T0-trained):", flush=True)
    for k, v in probe_results.items():
        print(f"    {k}: {v:.2%}", flush=True)

    total_min = (time.time() - t_global) / 60
    print(f"\n[done] total={total_min:.1f}m  results -> {RESULTS_DIR}", flush=True)

    # ── plot ───────────────────────────────────────────────────────────────────
    plot_summary(all_results, n_layers, RESULTS_DIR / "summary.png")


if __name__ == "__main__":
    main()
