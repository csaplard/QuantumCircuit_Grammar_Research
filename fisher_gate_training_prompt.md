# FisherGate Fine-Tuning — Training the Self-Regulating Layer

## Context

I have a proof-of-concept result: a `FisherGate` module inserted into a transformer measurably senses cognitive regime from the model's own hidden state dynamics (regime-dependent gate values, Cohen's d ≈ 2.7 between mathematical and factual prompts). But in inference-only mode, the gate's learnable parameters were never trained — they sit at initialization, so the gating behavior doesn't yet improve downstream performance.

This script trains ONLY the FisherGate parameters (a tiny linear(1→1) layer plus LayerNorm = a few hundred parameters total) while keeping the base model fully frozen. This is the minimum viable test: if the model can learn to use its own Fisher signal for better reasoning, the gate weights will shift away from initialization and downstream accuracy will improve.

## Task

Build a single Python script `train_fisher_gate.py` that:

### 1. Model loading
- Load `google/gemma-3-1b-it` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`
- Fallback: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Freeze everything**: `for p in model.parameters(): p.requires_grad = False`
- Set `model.eval()` to disable dropout on the base model

### 2. FisherGate module (trainable)

Reuse the same `FisherGate` module from the PoC, but make ONLY these parameters trainable:
- `self.gate_linear = nn.Linear(1, 1)` — initialized as `weight=1.0, bias=0.0` (identity-like)
- `self.gate_norm = nn.LayerNorm(1)` (optional stabilizer)

Key implementation details:
- Sliding window buffer (W=32) stays detached — no gradient through Fisher history
- The gate computation: `z = (F(t) - F_ema) / (F_std + eps)` is detached where possible, but the final `gate = sigmoid(gate_linear(z))` MUST be differentiable
- The residual modulation: `output = h * (dampening + (1 - dampening) * gate)` with `dampening = 0.85`
- Set `self.gate_linear.weight.requires_grad = True` and `self.gate_linear.bias.requires_grad = True`

Insert the FisherGate after the early decoder layer (layer N//6) using a forward hook that modifies the layer output.

### 3. Dataset

Use GSM8K via `datasets` library:
```python
from datasets import load_dataset
ds = load_dataset("gsm8k", "main", split="train")
```

Take the first 500 examples for training. Each example has `question` and `answer` (where answer ends with `#### <number>`).

Format each example as a chat-style prompt using the model's tokenizer chat template:
```
[user]: {question}
[assistant]: {answer}
```

For the loss, mask the prompt tokens (compute loss only on the answer portion).

### 4. Training loop

- Batch size: 1 (memory constraint)
- Gradient accumulation: 4 steps (effective batch 4)
- Max sequence length: 512 tokens
- Learning rate: 1e-3 (small model, tiny parameter count, can use higher LR)
- Optimizer: `torch.optim.AdamW` with weight_decay=0.0 (no regularization on such few params)
- Scheduler: cosine with 20 warmup steps
- Total steps: 500 examples / effective_batch_4 = ~125 steps per epoch
- Train for 2 epochs (total ~250 gradient updates)

Training loop structure:
```python
for epoch in range(2):
    for i, example in enumerate(train_dataset):
        # Tokenize with chat template
        # Forward pass through frozen model + active FisherGate
        # Compute cross-entropy loss on answer tokens only
        # Backward — gradients flow ONLY to FisherGate parameters
        # Accumulate and step every 4 examples
        # Log loss and current gate weight/bias values
```

**Critical:** After `loss.backward()`, verify that `gate_linear.weight.grad` is not None and not zero. If it's None, the gate is disconnected from the graph — fix the detach() calls in FisherGate.

### 5. Logging

Every 10 steps, print and log to `training_log.csv`:
- step, epoch, loss, gate_weight, gate_bias, mean_gate_value_this_batch, mean_fisher_this_batch, elapsed_time

This lets us track whether the gate weights are actually moving during training. If `gate_weight` stays at 1.0 and `gate_bias` at 0.0 after 50+ steps, training is broken.

### 6. Evaluation

Before training (step 0) and after training, evaluate on a held-out set:
- Use GSM8K test set, first 50 examples
- For each example, generate up to 256 tokens with:
  - Mode A: Base model (FisherGate removed)
  - Mode B: Base model + trained FisherGate
- Extract the numerical answer from the generation (look for the last number after `####` or the last number in the response)
- Compute accuracy for each mode
- Save results to `eval_before.csv` and `eval_after.csv`
- Print comparison table

### 7. Output files

Save to `results/fisher_gate_training/`:
- `training_log.csv` — per-step training metrics
- `gate_weights_trajectory.png` — plot of gate_weight and gate_bias over training steps
- `loss_curve.png` — training loss over steps
- `eval_before.csv` — accuracy before training (gate at initialization)
- `eval_after.csv` — accuracy after training (gate trained)
- `comparison.csv` — summary: baseline accuracy, pre-training accuracy, post-training accuracy
- `trained_gate_state.pt` — saved gate parameters (tiny file, ~1KB)

### 8. Key diagnostic questions

After training completes, the script should print:

1. **Did the gate learn?** Compare `gate_weight` and `gate_bias` before vs after. If they moved more than 5% from initialization, learning happened.

2. **Did loss decrease?** Training loss should trend downward over the 250 steps. If it's flat, the gate signal isn't informative enough.

3. **Did accuracy improve?** Compare eval_before vs eval_after. Even 1-2 percentage points on GSM8K is meaningful with only 2 trainable parameters.

4. **Does the gate respond differently to different regimes now?** After training, run a quick inference on 3 math + 3 factual prompts and report mean gate values. If the regime-dependent behavior strengthened, the gate learned to USE the Fisher signal more effectively.

## Technical constraints
- Single Python file: `train_fisher_gate.py`
- Libraries: `torch`, `transformers`, `bitsandbytes`, `datasets`, `peft` (optional for LoRA comparison), `matplotlib`, `pandas`, `numpy`
- 8GB RAM limit — 4-bit quantization mandatory, gradient checkpointing enabled: `model.gradient_checkpointing_enable()`
- Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` for mixed-precision forward
- If OOM occurs, reduce max_seq_length to 384 or 256 before reducing batch
- Set seeds: `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)`
- All outputs to `results/fisher_gate_training/`

## Expected runtime
- Gemma 3 1B, 250 steps, seq_len 512, batch 1, grad_accum 4
- On a modest GPU (6GB VRAM): ~2-3 hours for training + ~15 min for eval before/after
- On CPU: Much longer, probably not practical. If no GPU, reduce to 100 steps and 25 eval examples.

## Why this matters
The PoC showed the model can SENSE its own information geometry. This script tests whether it can USE that sensing — whether the FisherGate's tiny learnable parameters can learn to translate Fisher signal into better reasoning. Even a small improvement on GSM8K would be the first demonstration that self-regulation via internal information geometry actually works. Two trainable parameters, and if they shift the model's behavior measurably, that's the real proof of concept.

The null result is also informative: if the gate doesn't learn anything, we know the Fisher signal by itself isn't actionable without richer architectural support (e.g. multi-dimensional gates, multiple insertion points, or joint training with LoRA adapters).
