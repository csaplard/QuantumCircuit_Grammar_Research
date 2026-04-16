"""
Minimal token-by-token Fisher-adaptive generation with Hugging Face transformers.

Loop:
  logits -> entropy -> sliding Fisher(W=32, epsilon=0.01) -> temperature control by mean+-sigma band.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sliding_fisher import SlidingFisherTrace


def _entropy_from_logits(logits: torch.Tensor, temperature: float) -> float:
    scaled = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    p = probs.clamp_min(1e-12)
    h = -(p * torch.log(p)).sum()
    return float(h.item())


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    scaled = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = csum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    token_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(token_id.item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal transformers Fisher-adaptive sampler")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--window-size", type=int, default=32)
    ap.add_argument("--fisher-epsilon", type=float, default=0.01)
    ap.add_argument("--temperature-init", type=float, default=0.7)
    ap.add_argument("--temperature-low", type=float, default=0.4)
    ap.add_argument("--temperature-high", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stats-warmup-checks",
        type=int,
        default=8,
        help="Number of Fisher points before enabling mean+-sigma control",
    )
    ap.add_argument("--out-csv", default="results/llm_entropy/fisher_runs/transformers_minimal_trace.csv")
    ap.add_argument("--out-text", default="results/llm_entropy/fisher_runs/transformers_minimal_output.txt")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype)
    if args.dtype == "auto":
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model_id} on {device} dtype={torch_dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    tracker = SlidingFisherTrace(
        window_size=args.window_size,
        alphabet_size=7,
        sax_behavior="quantile",
        fisher_epsilon=args.fisher_epsilon,
    )
    fisher_history: list[float] = []

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "step",
                "token_id",
                "token_text",
                "entropy",
                "fisher_trace",
                "fisher_mean",
                "fisher_std",
                "temp_before",
                "temp_after",
                "action",
            ]
        )

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    output_token_ids: list[int] = []
    temperature = float(args.temperature_init)
    past_key_values = None

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            else:
                out = model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
        logits = out.logits[:, -1, :].squeeze(0)
        past_key_values = out.past_key_values

        temp_before = float(temperature)
        entropy = _entropy_from_logits(logits, temperature=temp_before)
        fisher_trace = tracker.update(entropy)

        fisher_mean = math.nan
        fisher_std = math.nan
        action = "hold"
        if fisher_trace is not None:
            fisher_history.append(float(fisher_trace))
            if len(fisher_history) >= args.stats_warmup_checks:
                fisher_mean = float(torch.tensor(fisher_history).mean().item())
                fisher_std = float(torch.tensor(fisher_history).std(unbiased=False).item())
                low = fisher_mean - fisher_std
                high = fisher_mean + fisher_std
                if fisher_trace < low:
                    temperature = float(args.temperature_low)
                    action = "down"
                elif fisher_trace > high:
                    temperature = float(args.temperature_high)
                    action = "up"

        next_id = _sample_next_token(logits, temperature=temp_before, top_p=args.top_p, generator=gen)
        output_token_ids.append(next_id)
        token_text = tokenizer.decode([next_id], skip_special_tokens=False)

        with out_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    step,
                    next_id,
                    token_text,
                    f"{entropy:.8f}",
                    "" if fisher_trace is None else f"{fisher_trace:.8f}",
                    "" if math.isnan(fisher_mean) else f"{fisher_mean:.8f}",
                    "" if math.isnan(fisher_std) else f"{fisher_std:.8f}",
                    f"{temp_before:.4f}",
                    f"{temperature:.4f}",
                    action,
                ]
            )

        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1)

        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    out_text = Path(args.out_text)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_text.write_text(text, encoding="utf-8")

    print(f"Wrote text:  {out_text}")
    print(f"Wrote trace: {out_csv}")


if __name__ == "__main__":
    main()
