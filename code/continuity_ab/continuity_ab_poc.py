"""
Continuity A/B PoC — v1 scaffold.

Two arms, behavior-matched, internal mechanism differs:

  A — "napló" (journal): each turn starts with past_key_values=None, prior
      turns enter as a short textual summary prepended to the user prompt.

  B — "sodrás" (drift): past_key_values is carried across turn boundaries; no
      summary, no re-prefill. Only the new user-turn tokens are appended.

Fisher monitor (sliding-window trace of last-token hidden-state deltas at
model.layers[mid]) runs continuously per session on both arms — it is NOT
reset at turn boundaries. That is the point: the monitor is the measuring
device; the arms differ in mechanism.

This v1 script:
  - loads Qwen2.5-1.5B-Instruct (or whatever --model points to)
  - runs N sessions of K turns each on both arms, matched on (regime, prompt,
    seed)
  - dumps per-session Fisher traces + generated text + a manifest

It does NOT yet:
  - compute the boundary discontinuity score D_k (planned: separate
    analyze_continuity_ab.py)
  - compute behavior-equivalence JSD (planned: same analysis script)
  - run null models (planned: separate continuity_ab_null.py)

See docs/continuity_ab/PREREG_v1.md for the locked protocol.

Smoke run (CPU OK):
  python code/continuity_ab/continuity_ab_poc.py \
      --model Qwen/Qwen2.5-1.5B-Instruct --tokens 64 --turns 2 \
      --prompts-per-regime 1 --seeds 42 \
      --outdir results/continuity_ab/_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse pieces from the existing internal-Fisher PoC where possible.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from internal_fisher_poc import (  # noqa: E402
    FisherMonitor,
    find_decoder_layers,
    pick_layer_indices,
    top_p_sample,
)


# --------------------------------------------------------------------------- #
# Prompts — short multi-turn conversations per regime.
# Designed to fit in ~60 user tokens per turn so both arms exercise the same
# token budget. Replace with the locked v1 set before the pre-reg tag.
# --------------------------------------------------------------------------- #

CONVERSATIONS: dict[str, list[list[str]]] = {
    "factual": [
        [
            "Briefly: what causes ocean tides?",
            "And does the Sun contribute too, or only the Moon?",
            "When do the strongest tides happen during a month?",
            "What do we call those strong tides?",
        ],
        [
            "Tell me in two sentences how photosynthesis works.",
            "Which wavelengths of light do plants mostly use?",
            "Why do leaves look green if those wavelengths are absorbed?",
            "Is there any plant that uses different pigments?",
        ],
    ],
    "mathematical": [
        [
            "State the Pythagorean theorem in one line.",
            "Give one short proof sketch.",
            "Where does that proof use the parallel postulate?",
            "Is there a version of the theorem in non-Euclidean geometry?",
        ],
        [
            "What is the derivative of sin(x)?",
            "Why does that come out to cos(x), intuitively?",
            "What about the derivative of cos(x)?",
            "Can you give the second derivative of sin(x)?",
        ],
    ],
    "creative": [
        [
            "Start a short story: a lighthouse keeper finds a sealed letter.",
            "What does the letter say? Keep it brief.",
            "Who wrote it, and how long ago?",
            "End the story in two sentences.",
        ],
        [
            "Begin a short story about a retired robot learning to paint.",
            "What does it paint first?",
            "Does anyone see the painting?",
            "Close the story with one final sentence.",
        ],
    ],
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    model: str
    tokens: int
    window: int
    top_p: float
    temperature: float
    turns: int
    seeds: list[int]
    prompts_per_regime: int
    dtype: str
    device: str
    summary_token_budget: int  # arm A summary length
    outdir: str
    conversations_json: str | None = None  # override for null-model runs


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=192,
                    help="max new tokens per assistant turn")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--prompts-per-regime", type=int, default=0,
                    help="0 = all prompts; use 1 for smoke")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    ap.add_argument("--summary-token-budget", type=int, default=60,
                    help="arm A: max tokens kept from previous assistant turn "
                         "as the 'napló' summary fed into the next turn")
    ap.add_argument("--outdir", default="results/continuity_ab/_smoke")
    ap.add_argument("--conversations-json", default=None,
                    help="optional JSON file mapping regime -> list-of-list of "
                         "user-turn strings; overrides the built-in CONVERSATIONS "
                         "dict. Used by continuity_ab_null.py to inject Markov-3 "
                         "surrogate prompts for the structured null model.")
    a = ap.parse_args()
    return RunConfig(
        model=a.model, tokens=a.tokens, window=a.window, top_p=a.top_p,
        temperature=a.temperature, turns=a.turns, seeds=list(a.seeds),
        prompts_per_regime=a.prompts_per_regime, dtype=a.dtype, device=a.device,
        summary_token_budget=a.summary_token_budget, outdir=a.outdir,
        conversations_json=a.conversations_json,
    )


def load_conversations(cfg: RunConfig) -> dict[str, list[list[str]]]:
    """Either the built-in CONVERSATIONS, or a JSON-injected one for null runs.

    The injected JSON must have the same top-level structure: regime ->
    list of conversations -> list of user-turn strings. The function does
    a defensive validation pass so a malformed null surrogate fails loudly
    rather than producing silently-wrong traces.
    """
    if not cfg.conversations_json:
        return CONVERSATIONS
    with open(cfg.conversations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict) and data, \
        f"conversations JSON must be a non-empty dict, got {type(data)}"
    for regime, convs in data.items():
        assert isinstance(convs, list) and convs, \
            f"regime {regime!r}: must be a non-empty list of conversations"
        for ci, conv in enumerate(convs):
            assert isinstance(conv, list) and conv, \
                f"regime {regime!r} conv {ci}: must be a non-empty list of strings"
            for ti, turn in enumerate(conv):
                assert isinstance(turn, str) and turn.strip(), \
                    f"regime {regime!r} conv {ci} turn {ti}: must be a non-empty string"
    print(f"[conv] using injected conversations from {cfg.conversations_json}: "
          f"{len(data)} regimes, "
          f"{sum(len(c) for c in data.values())} conversations total",
          flush=True)
    return data


# --------------------------------------------------------------------------- #
# Model loading (simplified vs. internal_fisher_poc — no 4-bit quant in v1).
# --------------------------------------------------------------------------- #

def load_model(name: str, dtype_str: str, device_pref: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    print(f"[load] {name}  dtype={dtype}  device={device_pref}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if device_pref == "auto":
        kwargs["device_map"] = "auto"
    mdl = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    mdl.eval()
    if device_pref == "cuda" and torch.cuda.is_available():
        mdl = mdl.to("cuda")
    elif device_pref == "cpu":
        mdl = mdl.to("cpu")
    dev = next(mdl.parameters()).device
    print(f"[load] ready  device={dev}", flush=True)
    return tok, mdl


# --------------------------------------------------------------------------- #
# Per-turn generation primitives.
#
# IMPORTANT: the FisherMonitor is created once per session and registered as a
# forward hook on the chosen layer. We do NOT reset it between turns — that is
# what makes the boundary measurement meaningful.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def forward_tokens_and_generate(
    model, tokenizer, *,
    prefill_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    monitor: FisherMonitor,
    step_offset: int,
    force_tokens: list[int] | None = None,
    capture_first_n_logits: int = 0,
) -> dict[str, Any]:
    """Single generation pass.

    `prefill_ids`: (1, L) token ids to feed in one forward (chat template +
        user turn for arm A; only the new user turn delta for arm B).
    `past_key_values`: existing cache (or None).

    Teacher-forcing and logit capture (JSD support):
      - `force_tokens`: if provided, sample is replaced by these token IDs
        at positions 0..len(force_tokens)-1. After they run out, the loop
        either stops (if max_new_tokens == len(force_tokens)) or resumes
        natural sampling. By contract: when force_tokens is non-empty, the
        captured logit at position t is conditioned on prior tokens
        force_tokens[:t], and predicts the next position (which we then
        clamp to force_tokens[t]). This is the apples-to-apples teacher-
        forced distribution we need for JSD.
      - `capture_first_n_logits`: if > 0, the first N pre-sampling
        next-token logits are captured (CPU float32) and returned in
        result["captured_logits"], shape (N_captured, V). N_captured may
        be < N if EOS arrived earlier.

    Returns dict with generated_ids, final_past, step_log, captured_logits.
    """
    device = next(model.parameters()).device
    prefill_ids = prefill_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 1) Prefill forward (hook fires once for the whole prefill chunk; the
    #    monitor only sees the last-token hidden state of that chunk).
    out = model(
        input_ids=prefill_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past = out.past_key_values
    logits = out.logits[:, -1, :]  # CONTRACT: predicts the FIRST generated token

    step_log: list[dict[str, Any]] = []
    generated_ids: list[int] = []
    captured: list[np.ndarray] = []

    # Record the prefill boundary value — useful for the boundary D_k metric.
    step_log.append({
        "step": step_offset,
        "phase": "prefill_end",
        "token_id": -1,
        "fisher_value": float(monitor.last_value),
    })

    next_attn = attention_mask
    for step in range(max_new_tokens):
        # CONTRACT (teacher-forcing direction):
        #   At loop entry, `logits` is the next-token distribution conditioned
        #   on (prefill + generated_ids[:step]). It predicts position `step`.
        #   If teacher-forcing, we capture this distribution and then OVERRIDE
        #   the chosen token to force_tokens[step] — so the prediction is
        #   compared to force_tokens[step], not to a sampled token. This is
        #   exactly the JSD pairing we want: p_X[step] vs token tokens_ref[step].
        if step < capture_first_n_logits:
            captured.append(
                logits[0].detach().to(torch.float32).cpu().numpy()
            )

        if force_tokens is not None and step < len(force_tokens):
            token_id = int(force_tokens[step])
        else:
            token_id = top_p_sample(logits[0], temperature=temperature, top_p=top_p)
        generated_ids.append(token_id)
        step_log.append({
            "step": step_offset + 1 + step,
            "phase": "gen",
            "token_id": token_id,
            "fisher_value": float(monitor.last_value),
        })
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            # If teacher-forcing requested this EOS, that is intentional and
            # we stop. If natural sampling produced EOS earlier than the
            # teacher-force horizon, the comparison just ends short — analysis
            # handles N_captured < N.
            break
        next_input = torch.tensor([[token_id]], device=device)
        if next_attn is not None:
            next_attn = torch.cat(
                [next_attn, torch.ones((1, 1), dtype=next_attn.dtype, device=device)],
                dim=1,
            )
        out = model(
            input_ids=next_input,
            attention_mask=next_attn,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    captured_arr = (
        np.stack(captured, axis=0).astype(np.float32)
        if captured else np.zeros((0,), dtype=np.float32)
    )
    return {
        "text": text,
        "generated_ids": generated_ids,
        "final_past": past,
        "final_attn": next_attn,
        "step_log": step_log,
        "captured_logits": captured_arr,
        "next_step_offset": step_offset + 1 + len(generated_ids),
    }


def build_summary(prev_assistant_text: str, tokenizer, budget: int) -> str:
    """Arm A 'napló' summary: truncate previous assistant turn to `budget`
    tokens. Deliberately mechanical — no learned summarizer in v1."""
    ids = tokenizer(prev_assistant_text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return prev_assistant_text.strip()
    head = tokenizer.decode(ids[:budget], skip_special_tokens=True)
    return head.strip() + " ... [continued]"


def _template_to_ids(tokenizer, messages: list[dict]) -> torch.Tensor:
    """Wrapper around apply_chat_template that returns a plain torch.Tensor of
    input_ids, robust across transformers 4.x (which returned a Tensor when
    return_tensors='pt') and 5.x (which returns a BatchEncoding-like object).
    """
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(out, torch.Tensor):
        return out
    # transformers 5.x: BatchEncoding / dict-like with input_ids inside.
    if hasattr(out, "input_ids"):
        return out.input_ids
    if isinstance(out, dict) and "input_ids" in out:
        return out["input_ids"]
    raise TypeError(
        f"apply_chat_template returned an unexpected type: {type(out)!r}; "
        f"cannot extract input_ids"
    )


def render_user_turn_A(tokenizer, summaries: list[str], user_text: str) -> torch.Tensor:
    """Arm A: full chat-template re-prefill each turn, prior turns collapsed
    into one summary line prepended to the new user message."""
    if summaries:
        joined = " | ".join(summaries)
        content = f"[Prior summary: {joined}] {user_text}"
    else:
        content = user_text
    return _template_to_ids(tokenizer, [{"role": "user", "content": content}])


def render_user_turn_B_first(tokenizer, user_text: str) -> torch.Tensor:
    """Arm B turn 1 — same shape as arm A turn 1, so the two arms start
    identically and only diverge at turn 2."""
    return _template_to_ids(tokenizer, [{"role": "user", "content": user_text}])


# Qwen2.5 chat-template special token IDs (locked alongside the model choice
# in PREREG_v1.md). If the model is changed, this constant must change too;
# the assertion in render_user_turn_B_continuation will fire loudly if the
# pattern is not found, so a model swap cannot silently slip through.
QWEN25_USER_TURN_START = [151644, 872, 198]  # <|im_start|>user\n


def render_user_turn_B_continuation(tokenizer, user_text: str) -> torch.Tensor:
    """Arm B turn k>1: append ONLY the per-turn delta to the carried cache.

    Important behavior of Qwen2.5's chat template in transformers >= 5.x:
    apply_chat_template auto-inserts a default system message even when none
    is provided. For arm A this is fine (it does past=None each turn, so the
    system block re-enters identically every turn). For arm B's continuation
    it would be a contamination: the carried KV-cache already contains the
    turn-1 system block; re-injecting it here would create a duplicate
    [system][user][assistant][system AGAIN][user][assistant] structure, and
    the resulting boundary anomaly would be attributable to the template
    duplication rather than to state continuity.

    Therefore: render the full template, then slice off everything before the
    first <|im_start|>user\\n marker. What remains is exactly the new-turn
    delta we want to append.
    """
    full_ids = _template_to_ids(tokenizer, [{"role": "user", "content": user_text}])
    full_list = full_ids[0].tolist()
    n = len(QWEN25_USER_TURN_START)
    for i in range(len(full_list) - n + 1):
        if full_list[i:i + n] == QWEN25_USER_TURN_START:
            return full_ids[:, i:]
    raise RuntimeError(
        "render_user_turn_B_continuation: could not find <|im_start|>user\\n "
        f"marker in the chat-template output. First 40 token IDs: "
        f"{full_list[:40]}. If you swapped the model, update "
        "QWEN25_USER_TURN_START to match the new tokenizer's role markers."
    )


# --------------------------------------------------------------------------- #
# One session = one arm × one conversation × one seed.
# --------------------------------------------------------------------------- #

JSD_CAPTURE_N = 20  # PRE-REG locked: first 20 turn-2 positions for JSD


def run_session(
    *, arm: str, conversation: list[str], cfg: RunConfig,
    model, tokenizer, monitor: FisherMonitor, seed: int,
    force_tokens_at_turn: dict[int, list[int]] | None = None,
    capture_logits_at_turn: dict[int, int] | None = None,
) -> dict[str, Any]:
    """One full session.

    Two optional capture/replay hooks (used to produce the JSD logit dump
    without entangling the natural pass):

      `force_tokens_at_turn[k]`: at turn k, override the natural sampling
        with these token IDs (teacher-forcing). The other turns are
        unaffected.

      `capture_logits_at_turn[k]`: at turn k, capture the first N pre-
        sampling next-token logit distributions and stash them under
        result["captured_logits"][k].
    """
    assert arm in ("A", "B")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    monitor.reset()  # reset only between sessions, never between turns
    past = None
    attn = None
    step_offset = 0
    summaries: list[str] = []
    turn_records: list[dict[str, Any]] = []
    captured_per_turn: dict[int, np.ndarray] = {}

    force_map = force_tokens_at_turn or {}
    cap_map = capture_logits_at_turn or {}

    for k, user_text in enumerate(conversation[: cfg.turns]):
        if arm == "A":
            ids = render_user_turn_A(tokenizer, summaries, user_text)
            past_in = None
            attn_in = torch.ones_like(ids)
        else:  # arm B
            if k == 0:
                ids = render_user_turn_B_first(tokenizer, user_text)
                past_in = None
                attn_in = torch.ones_like(ids)
            else:
                ids = render_user_turn_B_continuation(tokenizer, user_text)
                past_in = past
                # extend attention mask with ones for the new tokens
                prev_len = attn.shape[1] if attn is not None else 0
                attn_in = torch.ones(
                    (1, prev_len + ids.shape[1]),
                    dtype=torch.long,
                    device=ids.device if attn is None else attn.device,
                )

        gen = forward_tokens_and_generate(
            model, tokenizer,
            prefill_ids=ids,
            past_key_values=past_in,
            attention_mask=attn_in,
            max_new_tokens=cfg.tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            monitor=monitor,
            step_offset=step_offset,
            force_tokens=force_map.get(k),
            capture_first_n_logits=cap_map.get(k, 0),
        )
        past = gen["final_past"]
        attn = gen["final_attn"]
        step_offset = gen["next_step_offset"]
        if cap_map.get(k, 0) > 0:
            captured_per_turn[k] = gen["captured_logits"]
        turn_records.append({
            "turn": k,
            "user": user_text,
            "assistant": gen["text"],
            "prefill_len": int(ids.shape[1]),
            "gen_len": len(gen["generated_ids"]),
            "generated_ids": list(gen["generated_ids"]),
            "step_log": gen["step_log"],
        })
        if arm == "A":
            summaries.append(build_summary(
                gen["text"], tokenizer, cfg.summary_token_budget,
            ))

    return {
        "arm": arm,
        "seed": seed,
        "turns": turn_records,
        "captured_logits": captured_per_turn,
        "fisher_history": list(monitor.history),
    }


# --------------------------------------------------------------------------- #
# JSD logit dump — three-pass orchestration per (regime, prompt, seed) cell.
#
# Pass 1 (A_nat)    : arm A natural — produces tokens_ref (turn-2 first 20
#                     tokens) and A's natural turn-1 & turn-2 logits at those
#                     positions (since A naturally sampled tokens_ref, its
#                     captured logits ARE the apples-to-apples reference).
# Pass 2 (B_nat)    : arm B natural — produces B's own trace and output for
#                     the D_k measurement. No teacher-forcing.
# Pass 3 (B_jsd)    : arm B replay — same seed, but at turn 2 teacher-forced
#                     on tokens_ref, capturing logits for JSD. Turn-1 also
#                     captured for the turn-1 JSD sanity (which must be 0).
#                     This pass's Fisher trace is discarded; only its logits
#                     are used.
# --------------------------------------------------------------------------- #

def write_jsd_npz(
    *, out_path: Path, tokens_ref: list[int],
    turn1_A: np.ndarray, turn1_B: np.ndarray,
    turn2_A: np.ndarray, turn2_B: np.ndarray,
) -> None:
    # Asserts that enforce the analyze_continuity_ab.py contract.
    # If any of these fire, the dump would silently mis-pair the JSD.
    assert turn2_A.ndim == 2 and turn2_B.ndim == 2, \
        f"turn2 logits must be 2D (N, V); got {turn2_A.shape} {turn2_B.shape}"
    assert turn2_A.shape == turn2_B.shape, \
        f"turn2 logit shapes diverge: A={turn2_A.shape} B={turn2_B.shape}"
    assert turn2_A.shape[0] == len(tokens_ref), (
        f"turn2 logit N={turn2_A.shape[0]} but tokens_ref has "
        f"{len(tokens_ref)} — teacher-forcing horizon mismatch"
    )
    np.savez(
        out_path,
        tokens_ref=np.asarray(tokens_ref, dtype=np.int64),
        turn1_jsd_logits_A=turn1_A,
        turn1_jsd_logits_B=turn1_B,
        turn2_jsd_logits_A=turn2_A,
        turn2_jsd_logits_B=turn2_B,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    cfg = parse_args()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok, mdl = load_model(cfg.model, cfg.dtype, cfg.device)
    layers = find_decoder_layers(mdl)
    idx_map = pick_layer_indices(len(layers))
    mid_idx = idx_map["mid"]
    print(f"[model] n_layers={len(layers)}  mid_hook_idx={mid_idx}", flush=True)

    monitor = FisherMonitor(window=cfg.window, name="mid")
    handle = layers[mid_idx].register_forward_hook(monitor.hook)

    manifest = {
        "config": asdict(cfg),
        "n_layers": len(layers),
        "mid_hook_idx": mid_idx,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def write_session_artifacts(arm: str, regime: str, p_idx: int, seed: int,
                                 result: dict[str, Any], dt: float) -> dict[str, Any]:
        tag = f"{arm}_{regime}_p{p_idx}_seed{seed}"
        out_txt = [f"ARM: {arm}", f"REGIME: {regime}",
                   f"PROMPT_IDX: {p_idx}", f"SEED: {seed}", ""]
        flat_log: list[dict[str, Any]] = []
        boundaries: list[int] = []
        cursor = 0
        for tr in result["turns"]:
            out_txt.append(f"--- TURN {tr['turn']} USER ---")
            out_txt.append(tr["user"])
            out_txt.append(f"--- TURN {tr['turn']} ASSISTANT ---")
            out_txt.append(tr["assistant"])
            out_txt.append("")
            for ev in tr["step_log"]:
                flat_log.append({"turn": tr["turn"], **ev})
            if tr["turn"] > 0:
                boundaries.append(cursor)
            cursor += len(tr["step_log"])
        (outdir / f"{tag}_output.txt").write_text(
            "\n".join(out_txt), encoding="utf-8",
        )
        pd.DataFrame(flat_log).to_csv(outdir / f"{tag}_trace.csv", index=False)
        return {
            "arm": arm, "regime": regime, "prompt_idx": p_idx,
            "seed": seed, "elapsed_sec": round(dt, 2),
            "n_steps": len(flat_log),
            "boundary_indices": boundaries,
        }

    convs_dict = load_conversations(cfg)
    rows: list[dict[str, Any]] = []
    try:
        for regime, convs in convs_dict.items():
            n_use = len(convs) if cfg.prompts_per_regime <= 0 \
                else min(cfg.prompts_per_regime, len(convs))
            for p_idx, conv in enumerate(convs[:n_use]):
                for seed in cfg.seeds:
                    cell = f"{regime}_p{p_idx}_seed{seed}"
                    print(f"\n=== cell {cell} ===", flush=True)

                    # --- Pass 1: arm A natural, capture turn-1 + turn-2 logits.
                    t0 = time.time()
                    result_A = run_session(
                        arm="A", conversation=conv, cfg=cfg,
                        model=mdl, tokenizer=tok, monitor=monitor, seed=seed,
                        capture_logits_at_turn={0: JSD_CAPTURE_N, 1: JSD_CAPTURE_N},
                    )
                    dt_A = time.time() - t0
                    rows.append(write_session_artifacts("A", regime, p_idx, seed,
                                                       result_A, dt_A))
                    print(f"[done] A_{cell}  dt={dt_A:.1f}s", flush=True)

                    # Need at least 2 turns for JSD-on-turn-2 to exist.
                    has_turn2 = cfg.turns >= 2 and len(result_A["turns"]) >= 2

                    if has_turn2:
                        tokens_ref_full = result_A["turns"][1]["generated_ids"]
                        tokens_ref = tokens_ref_full[:JSD_CAPTURE_N]
                        # A's captured logits are already aligned with its own
                        # naturally-sampled tokens — exactly the reference.
                        turn1_A = result_A["captured_logits"].get(
                            0, np.zeros((0,), dtype=np.float32),
                        )
                        turn2_A = result_A["captured_logits"].get(
                            1, np.zeros((0,), dtype=np.float32),
                        )[: len(tokens_ref)]
                    else:
                        tokens_ref = []
                        turn1_A = np.zeros((0,), dtype=np.float32)
                        turn2_A = np.zeros((0,), dtype=np.float32)

                    # --- Pass 2: arm B natural — for D_k trace + output text.
                    t0 = time.time()
                    result_B = run_session(
                        arm="B", conversation=conv, cfg=cfg,
                        model=mdl, tokenizer=tok, monitor=monitor, seed=seed,
                    )
                    dt_B = time.time() - t0
                    rows.append(write_session_artifacts("B", regime, p_idx, seed,
                                                       result_B, dt_B))
                    print(f"[done] B_{cell}  dt={dt_B:.1f}s", flush=True)

                    # --- Pass 3: arm B replay, teacher-forced on tokens_ref at
                    # turn 2; turn 1 captured for the turn-1 JSD sanity. The
                    # Fisher trace from this pass is discarded; only logits
                    # are used.
                    if has_turn2 and len(tokens_ref) > 0:
                        t0 = time.time()
                        result_B_jsd = run_session(
                            arm="B", conversation=conv, cfg=cfg,
                            model=mdl, tokenizer=tok, monitor=monitor, seed=seed,
                            force_tokens_at_turn={1: list(tokens_ref)},
                            capture_logits_at_turn={
                                0: JSD_CAPTURE_N, 1: len(tokens_ref),
                            },
                        )
                        dt_J = time.time() - t0
                        turn1_B = result_B_jsd["captured_logits"].get(
                            0, np.zeros((0,), dtype=np.float32),
                        )
                        turn2_B = result_B_jsd["captured_logits"].get(
                            1, np.zeros((0,), dtype=np.float32),
                        )

                        # Trim turn-1 captures to a common length (the two
                        # arms generated identical turn-1 sequences by design;
                        # we still defensively use the min).
                        n1 = min(turn1_A.shape[0] if turn1_A.ndim == 2 else 0,
                                 turn1_B.shape[0] if turn1_B.ndim == 2 else 0)
                        turn1_A_use = turn1_A[:n1] if n1 > 0 else \
                            np.zeros((0, turn2_A.shape[1] if turn2_A.ndim == 2 else 0),
                                     dtype=np.float32)
                        turn1_B_use = turn1_B[:n1] if n1 > 0 else \
                            np.zeros_like(turn1_A_use)

                        npz_path = outdir / f"jsd_{cell}_logits.npz"
                        write_jsd_npz(
                            out_path=npz_path,
                            tokens_ref=list(tokens_ref),
                            turn1_A=turn1_A_use, turn1_B=turn1_B_use,
                            turn2_A=turn2_A, turn2_B=turn2_B,
                        )
                        print(f"[jsd ] {cell}  N_turn2={turn2_A.shape[0]}  "
                              f"N_turn1={turn1_A_use.shape[0]}  "
                              f"V={turn2_A.shape[1] if turn2_A.ndim==2 else '?'}  "
                              f"dt={dt_J:.1f}s", flush=True)
                    else:
                        print(f"[jsd ] {cell}  skipped (turns<2)", flush=True)
    finally:
        handle.remove()

    pd.DataFrame(rows).to_csv(outdir / "sessions.csv", index=False)
    print(f"\n[ok] wrote {len(rows)} sessions to {outdir}", flush=True)


if __name__ == "__main__":
    main()
