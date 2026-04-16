# Self-Regulating Fisher Gating Layer — Proof of Concept

## Context

I'm building toward a self-regulating transformer architecture. I have already demonstrated that:

1. **Fisher information measured from hidden state transitions during generation is regime-dependent.** Mathematical reasoning produces systematically lower Fisher values than factual/creative text. Cohen's d = 3.14, validated across 3 models (Qwen 1.5B, Gemma 3 1B, Gemma 3 4B), 7 prompts per regime, 3 layer depths.

2. **The signal is strongest at early layers** (d = 2.52–3.74) and architecture-specific in its layer profile.

3. **An external adaptive sampler** using this signal to modulate temperature improves text coherence but not reasoning accuracy — because it acts from outside.

The next step: **move the Fisher computation inside the model as a differentiable gating mechanism**, so the model regulates its own processing based on its own information geometry. This is the proof of concept for that.

## Task

Build a single Python script (`fisher_gating_poc.py`) that:

### 1. Model loading
- Use `google/gemma-3-4b-it` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`
- If unavailable, fallback to `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Set `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` before each generation

### 2. Fisher Gating Module (the core innovation)

Create a PyTorch `nn.Module` called `FisherGate` that:

```python
class FisherGate(nn.Module):
    """
    A differentiable self-monitoring layer that:
    1. Measures Fisher information from hidden state transitions (sliding window)
    2. Uses that measurement to gate/modulate the residual stream
    
    This is NOT an external controller. It lives inside the forward pass
    and its output is part of the computational graph.
    """
```

**Internal mechanics:**
- Maintains a sliding window buffer (W=32) of recent hidden states (detached, to avoid memory explosion)
- At each forward pass:
  - Receives the current hidden state tensor `h_t` (shape: [batch, seq_len, hidden_dim])
  - Takes the last token position: `h = h_t[:, -1, :]`
  - Appends `h.detach()` to the sliding window buffer
  - If buffer has >= 2 entries, computes Fisher information estimate:
    - `deltas = h[t] - h[t-1]` for all consecutive pairs in window
    - `F(t) = mean(deltas ** 2)` — scalar Fisher trace
  - Computes a **gating factor** from F(t):
    - Maintain running EMA of F: `F_ema = alpha * F(t) + (1 - alpha) * F_ema` (alpha=0.1)
    - Maintain running EMA of F²: for variance estimate
    - `z = (F(t) - F_ema) / (F_std + eps)` — normalized deviation
    - `gate = sigmoid(linear(z))` where `linear` is a tiny learnable layer (1 → 1, initialized to identity-like behavior: weight=1.0, bias=0.0)
  - Applies the gate to the residual stream:
    - `output = h_t * gate + h_t * (1 - gate) * dampening_factor`
    - Simplified: `output = h_t * (dampening_factor + (1 - dampening_factor) * gate)`
    - Where `dampening_factor = 0.85` (so gate modulates between 85%-100% of signal)
  - Logs: step, F(t), F_ema, z, gate_value, for later analysis

**Key design decisions:**
- The gate is differentiable — backprop can flow through it during fine-tuning (we don't fine-tune in this PoC, but the architecture supports it)
- The window buffer stores detached tensors — no gradient through history
- The gate effect is subtle (15% range) — we're not trying to radically change activations, just modulate them
- The learnable parameter (linear layer) means the model could in principle learn HOW to use the Fisher signal

### 3. Insertion into the model

- Identify the early decoder layer (approximately layer N//6, where the regime signal is strongest)
- Insert the FisherGate AFTER this layer using a forward hook that:
  - Passes the layer output through FisherGate
  - Returns the gated output as the new layer output
  - This means all subsequent layers receive Fisher-gated activations

**Important:** The hook must MODIFY the output, not just observe it. Use `register_forward_hook` and return the modified tensor.

### 4. Six-way comparison (with ablation controls)

Run six modes for each prompt:
- **Baseline**: No FisherGate, normal generation
- **Monitor-only**: FisherGate inserted but gate fixed at 1.0 (pass-through, only logs F(t))
- **Active gating**: FisherGate inserted and active (gate modulates activations based on Fisher z-score)
- **Random gate** (ablation): FisherGate inserted but gate value is `sigmoid(random_normal())` each step instead of Fisher-derived. Same 85%-100% range. This answers: "does any perturbation work?"
- **Constant gate** (ablation): FisherGate inserted but gate fixed at 0.92 (midpoint of the 85%-100% range). This answers: "is it just a constant scaling effect?"
- **Inverted gate** (ablation): FisherGate inserted, Fisher is measured, but the z-score is negated before the sigmoid: `gate = sigmoid(linear(-z))`. This answers: "does the direction of the Fisher signal matter, or any correlated noise?"

This six-way design answers every reviewer question:
- Baseline vs Monitor: does inserting the module change anything even when passive?
- Monitor vs Active: does the Fisher-based gating change the output?
- Active vs Random: is the Fisher signal doing something that random noise can't?
- Active vs Constant: is the dynamic adaptation important, or is static scaling enough?
- Active vs Inverted: does the polarity of the Fisher signal matter?

If Active gating produces regime-dependent gate values AND differs meaningfully from Random/Constant/Inverted — that's the clean result. If Random works just as well, we know it's not the Fisher signal. If Inverted works just as well, we know the direction doesn't matter. These controls make the result bulletproof.

**Implementation note:** All six modes use the same seed per regime. The Random gate mode should use a separate `torch.Generator` for its random values so it doesn't interfere with the model's sampling PRNG.

### 5. Prompts

Use these 7 prompts per regime (same as my multi-prompt validation):

**Factual:**
1. "Explain the causes and consequences of World War I."
2. "Describe how photosynthesis works in plants."
3. "What are the main differences between TCP and UDP protocols?"
4. "Explain how a combustion engine works step by step."
5. "Describe the water cycle and its importance for Earth's climate."
6. "What were the key events of the French Revolution?"
7. "Explain how vaccines work to protect against diseases."

**Mathematical:**
1. "Prove that the square root of 2 is irrational."
2. "Derive the quadratic formula from ax^2 + bx + c = 0."
3. "Prove that there are infinitely many prime numbers."
4. "Show that the sum of angles in a triangle is 180 degrees."
5. "Prove by induction that 1+2+...+n = n(n+1)/2."
6. "Derive the formula for the area of a circle using integration."
7. "Prove that e is irrational using its series expansion."

**Creative:**
1. "Write a short story about a lighthouse keeper who discovers a message in a bottle."
2. "Write a poem about the last tree on Earth."
3. "Describe an alien civilization that communicates through colors."
4. "Write a monologue from the perspective of a dying star."
5. "Create a myth explaining why the moon changes shape."
6. "Write a conversation between two AI systems meeting for the first time."
7. "Describe a city that exists entirely underwater."

Generate 256 tokens per prompt.

### 6. Output and analysis

Save everything to `results/fisher_gating_poc/`:

**Per-generation files:**
- `{mode}_{regime}_{prompt_idx}_output.txt` — generated text
- `{mode}_{regime}_{prompt_idx}_trace.csv` — columns: step, fisher_value, fisher_ema, z_score, gate_value, action

Where mode is one of: baseline, monitor, active, random, constant, inverted

**Aggregate analysis:**
- `summary.csv` — per regime per mode: mean F, std F, mean gate value, n_steps
- `gating_effect.csv` — for each regime: mean gate value across all modes, compare Fisher distributions
- `regime_separation.csv` — Cohen's d between regimes for each mode, to verify that gating doesn't destroy the signal
- `ablation_comparison.csv` — key comparison table: for each regime, mean gate value and mean F for all 6 modes side by side. This is the ablation result table.

**Plots:**
- `fisher_gate_traces.png` — 3x6 grid (rows=regimes, cols=modes), showing F(t) trace with gate value overlaid on secondary y-axis. This will be large — save at high DPI.
- `gate_distribution.png` — histogram of gate values per regime, overlaid for active/random/inverted modes. This directly shows whether Fisher-based gating behaves differently from random.
- `regime_separation_by_mode.png` — grouped bar chart of Cohen's d (factual↔math) per mode. If Active has higher d than Random/Constant/Inverted, the Fisher signal adds value.
- `ablation_summary.png` — compact 2-panel figure suitable for a paper: (left) mean gate value by regime for each mode, (right) Cohen's d by mode. This is the money plot.

### 7. Key questions this PoC answers

1. **Does the Fisher gating module produce regime-dependent gate values?** (If mathematical prompts produce systematically different gate values than factual/creative, the module is "sensing" the regime)
2. **Does gating preserve the regime separation signal?** (If Cohen's d stays similar between monitor and active modes, the gating doesn't destroy what it measures)
3. **Does gating change the generated text?** (Compare outputs between modes — any coherence differences?)
4. **Is the gate behavior interpretable?** (Can we see the gate responding to specific text patterns?)
5. **Is it really the Fisher signal?** (ABLATION: If Active outperforms Random, Constant, and Inverted on regime-sensitivity and/or text quality, the Fisher information specifically — not any perturbation — is driving the effect. This is the reviewer-proof result.)

## Technical constraints
- Single Python file: `fisher_gating_poc.py`
- Libraries: `torch`, `transformers`, `bitsandbytes`, `matplotlib`, `pandas`, `numpy`
- 8GB RAM limit — 4-bit quantization mandatory
- CUDA preferred, CPU fallback
- Per-regime seed: `torch.manual_seed(42)` + `torch.cuda.manual_seed(42)` before each generation
- All outputs to `results/fisher_gating_poc/`
- Total runs: 6 modes × 3 regimes × 7 prompts = 126 generations. At ~80s each this is ~2.5 hours. Add a progress counter that prints `[42/126] active / mathematical / prompt 3 — elapsed 55m, ETA 70m` style updates.
- If runtime is a concern, offer a `--quick` flag that runs only 2 prompts per regime (36 runs, ~45 min) for fast iteration, with full 7-prompt run as default.

## Why this matters
This is the proof of concept for a self-regulating transformer architecture. If the FisherGate module produces regime-dependent gating behavior from the model's own hidden states — without any external measurement — that demonstrates the model can "sense" its own information geometry. That's the first step toward a model that adapts its own processing based on what it's doing. Not from outside. From within.
