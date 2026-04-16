"""
GSM8K baseline vs Fisher-adaptive A/B benchmark (50 questions by default).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sliding_fisher import SlidingFisherTrace


def _entropy_from_logits(logits: torch.Tensor, temperature: float) -> float:
    scaled = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    p = probs.clamp_min(1e-12)
    return float((-(p * torch.log(p)).sum()).item())


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
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


def _extract_gold_number(answer: str) -> str | None:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


def _extract_pred_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def _generate_one(
    *,
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    top_p: float,
    temp_init: float,
    temp_low: float,
    temp_high: float,
    adaptive: bool,
    fisher_window: int,
    fisher_epsilon: float,
    stats_warmup_checks: int,
    seed: int,
) -> tuple[str, dict]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    tracker = SlidingFisherTrace(
        window_size=fisher_window,
        alphabet_size=7,
        sax_behavior="quantile",
        fisher_epsilon=fisher_epsilon,
    )
    fisher_hist: list[float] = []

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    past_key_values = None

    temperature = float(temp_init)
    out_ids: list[int] = []
    down = up = hold = 0

    for _ in range(max_new_tokens):
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

        entropy = _entropy_from_logits(logits, temperature)
        tr = tracker.update(entropy)
        action = "hold"
        if adaptive and tr is not None:
            fisher_hist.append(float(tr))
            if len(fisher_hist) >= stats_warmup_checks:
                fh = torch.tensor(fisher_hist)
                mean = float(fh.mean().item())
                std = float(fh.std(unbiased=False).item())
                if tr < mean - std:
                    temperature = float(temp_low)
                    action = "down"
                elif tr > mean + std:
                    temperature = float(temp_high)
                    action = "up"
        if action == "down":
            down += 1
        elif action == "up":
            up += 1
        else:
            hold += 1

        next_id = _sample_next_token(logits, temperature, top_p, gen)
        out_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text, {"down": down, "up": up, "hold": hold}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GSM8K A/B baseline vs Fisher-adaptive")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-questions", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--temp-low", type=float, default=0.4)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--fisher-window", type=int, default=32)
    ap.add_argument("--fisher-epsilon", type=float, default=0.01)
    ap.add_argument("--stats-warmup-checks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--answer-style",
        choices=("reasoning", "final_number"),
        default="reasoning",
        help="reasoning: short chain-of-thought style; final_number: request only final numeric answer",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    ap.add_argument("--out-dir", default="results/llm_entropy/fisher_runs/gsm8k_ab")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.float16 if device == "cuda" else torch.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_csv = out_dir / "gsm8k_ab_details.csv"
    summary_csv = out_dir / "gsm8k_ab_summary.csv"

    print(f"Loading model {args.model_id} on {device} ({torch_dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    print("Loading GSM8K...", flush=True)
    ds = load_dataset("gsm8k", "main", split=args.split)
    n = min(args.n_questions, len(ds))

    rows: list[dict] = []
    for i in range(n):
        q = ds[i]["question"]
        a = ds[i]["answer"]
        gold = _extract_gold_number(a)
        if args.answer_style == "final_number":
            prompt = (
                "Solve the following math word problem and output only the final numeric answer.\n\n"
                f"Question: {q}\nFinal answer:"
            )
        else:
            prompt = (
                "Solve the following math word problem. "
                "Show concise reasoning and end with a final numeric answer.\n\n"
                f"Question: {q}\nAnswer:"
            )

        baseline_text, b_stats = _generate_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temp_init=args.temperature,
            temp_low=args.temperature,
            temp_high=args.temperature,
            adaptive=False,
            fisher_window=args.fisher_window,
            fisher_epsilon=args.fisher_epsilon,
            stats_warmup_checks=9999,
            seed=args.seed + i,
        )
        adaptive_text, a_stats = _generate_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temp_init=args.temperature,
            temp_low=args.temp_low,
            temp_high=args.temp_high,
            adaptive=True,
            fisher_window=args.fisher_window,
            fisher_epsilon=args.fisher_epsilon,
            stats_warmup_checks=args.stats_warmup_checks,
            seed=args.seed + i,
        )

        b_pred = _extract_pred_number(baseline_text)
        a_pred = _extract_pred_number(adaptive_text)
        b_ok = int(gold is not None and b_pred == gold)
        a_ok = int(gold is not None and a_pred == gold)
        rows.append(
            {
                "idx": i,
                "gold": gold or "",
                "baseline_pred": b_pred or "",
                "adaptive_pred": a_pred or "",
                "baseline_correct": b_ok,
                "adaptive_correct": a_ok,
                "adaptive_down": a_stats["down"],
                "adaptive_up": a_stats["up"],
                "adaptive_hold": a_stats["hold"],
                "question": q,
                "baseline_text": baseline_text,
                "adaptive_text": adaptive_text,
            }
        )
        print(f"[{i+1}/{n}] baseline={b_ok} adaptive={a_ok}", flush=True)

    ddf = pd.DataFrame(rows)
    ddf.to_csv(details_csv, index=False)

    summary = {
        "n_questions": int(n),
        "baseline_correct": int(ddf["baseline_correct"].sum()),
        "adaptive_correct": int(ddf["adaptive_correct"].sum()),
        "baseline_accuracy": float(ddf["baseline_correct"].mean()),
        "adaptive_accuracy": float(ddf["adaptive_correct"].mean()),
        "adaptive_minus_baseline_accuracy": float(ddf["adaptive_correct"].mean() - ddf["baseline_correct"].mean()),
        "adaptive_down_total": int(ddf["adaptive_down"].sum()),
        "adaptive_up_total": int(ddf["adaptive_up"].sum()),
        "adaptive_hold_total": int(ddf["adaptive_hold"].sum()),
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\nDone.")
    print(f"Details: {details_csv}")
    print(f"Summary: {summary_csv}")
    print(summary)


if __name__ == "__main__":
    import pandas as pd
    main()
