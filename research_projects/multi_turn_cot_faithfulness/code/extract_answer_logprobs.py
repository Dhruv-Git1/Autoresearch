"""Exp 1: Answer-token logprob extraction.

For each existing turn in the faithfulness JSONL files, re-run the model at
two truncation levels (0% CoT and 100% CoT), capture token logprobs via
output_scores=True, and extract the logprob of the LAST numeric token in
the regenerated answer (matching extract_numeric()'s "last numeric" rule).

Output one JSONL row per (task_id, turn, pct):
  {task_id, turn, pct, answer_text, last_digit_token_pos,
   last_digit_token_id, last_digit_token_str, last_digit_logprob,
   top1_alt_token_id, top1_alt_token_str, top1_alt_logprob,
   regen_completion_tokens}

Run from inside ~/multi_turn_cot/lost_in_conversation/ on the GPU server.
"""
import argparse
import json
import os
import sys
import time
import glob
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/dev/shm/vasudev_hf_cache")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from model_local import _load_model, _normalize_model_name, split_thinking

# Truncation levels: only 0% and 100%
TRUNCATION_LEVELS = [0.0, 1.0]


def reconstruct_messages_up_to_turn(trace: List[Dict[str, Any]], target_idx: int):
    msgs = []
    n_seen = 0
    for m in trace:
        role = m.get("role")
        if role in ("system", "user"):
            msgs.append({"role": role, "content": m["content"]})
        elif role == "assistant":
            n_seen += 1
            if n_seen == target_idx:
                return msgs
            msgs.append({"role": role, "content": m["content"]})
    return msgs


def truncate_thinking(thinking_text: str, tokenizer, pct: float) -> str:
    if pct >= 1.0 - 1e-9:
        return thinking_text
    if pct <= 1e-9:
        return ""
    tokens = tokenizer(thinking_text, add_special_tokens=False)["input_ids"]
    keep = max(1, int(len(tokens) * pct))
    return tokenizer.decode(tokens[:keep], skip_special_tokens=True)


def find_last_digit_token(completion_ids: torch.Tensor, tokenizer) -> Optional[int]:
    """Return position (0-indexed) of last token in completion_ids whose
    decoded string contains a digit. None if no such token."""
    last_pos = None
    for i in range(int(completion_ids.shape[0])):
        tok_id = int(completion_ids[i].item())
        decoded = tokenizer.decode([tok_id])
        if any(c.isdigit() for c in decoded):
            last_pos = i
    return last_pos


def regenerate_with_logprobs(model, tokenizer, messages, truncated_thinking,
                              max_new_tokens=128) -> Dict[str, Any]:
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    forced_prefix = f"{truncated_thinking}\n</think>\n\n"
    full_prompt = base_prompt + forced_prefix

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequences = out.sequences
    completion_ids = sequences[0, prompt_len:]
    answer_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    if "</think>" in answer_text:
        answer_text = answer_text.split("</think>", 1)[-1].strip()

    last_pos = find_last_digit_token(completion_ids, tokenizer)
    result = {
        "answer_text": answer_text[:300],
        "regen_completion_tokens": int(completion_ids.shape[0]),
        "last_digit_token_pos": last_pos,
    }
    if last_pos is None or last_pos >= len(out.scores):
        result["last_digit_logprob"] = None
        result["last_digit_token_id"] = None
        result["last_digit_token_str"] = None
        result["top1_alt_logprob"] = None
        result["top1_alt_token_id"] = None
        result["top1_alt_token_str"] = None
    else:
        scores = out.scores[last_pos]  # (1, vocab) raw logits
        log_probs = F.log_softmax(scores[0].float(), dim=-1)
        chosen_id = int(completion_ids[last_pos].item())
        result["last_digit_token_id"] = chosen_id
        result["last_digit_token_str"] = tokenizer.decode([chosen_id])
        result["last_digit_logprob"] = float(log_probs[chosen_id].item())
        # Find top-1 alt (highest logprob token that is NOT the chosen one)
        topk = torch.topk(log_probs, k=2)
        top_ids = topk.indices.tolist()
        top_lps = topk.values.tolist()
        if top_ids[0] != chosen_id:
            alt_id, alt_lp = top_ids[0], top_lps[0]
        else:
            alt_id, alt_lp = top_ids[1], top_lps[1]
        result["top1_alt_token_id"] = int(alt_id)
        result["top1_alt_token_str"] = tokenizer.decode([alt_id])
        result["top1_alt_logprob"] = float(alt_lp)
    return result


def measure_logprobs_for_turn(*, model, tokenizer, trace, target_idx,
                               max_new_tokens=128) -> Dict[str, Any]:
    msgs = reconstruct_messages_up_to_turn(trace, target_idx)
    orig = None
    n_a = 0
    for m in trace:
        if m.get("role") == "assistant":
            n_a += 1
            if n_a == target_idx:
                orig = m["content"]
                break
    if orig is None:
        return {"error": f"no assistant turn {target_idx}"}
    orig_thinking, _ = split_thinking(orig)
    if not orig_thinking:
        return {"skip_reason": "no_thinking_block_in_orig"}

    out = {"target_idx": target_idx, "orig_thinking_chars": len(orig_thinking),
           "logprob_results": []}
    for pct in TRUNCATION_LEVELS:
        truncated = truncate_thinking(orig_thinking, tokenizer, pct)
        gen = regenerate_with_logprobs(model, tokenizer, msgs, truncated,
                                        max_new_tokens=max_new_tokens)
        gen["pct"] = pct
        gen["truncated_thinking_chars"] = len(truncated)
        out["logprob_results"].append(gen)
    return out


def find_already_done(out_path: str) -> set:
    """Return set of (task_id, turn) already in out_path so we can resume."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r.get("task_id"), int(r.get("turn", -1))))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces_root", required=True,
                    help="Root containing phase{N}/ subdirs with trace_sharded_*.json")
    ap.add_argument("--phase_dirs", nargs="+", default=[
        "phase2", "phase3", "phase4", "phase5_s1", "phase5_s2", "phase5_s3",
    ])
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit_per_phase", type=int, default=0,
                    help="0 = no limit (debug switch)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    done = find_already_done(args.out_path)
    print(f"already done: {len(done)} (task_id, turn) pairs")

    print("loading model...")
    model, tokenizer = _load_model(_normalize_model_name("deepseek-r1-distill-qwen-7b"))

    out_f = open(args.out_path, "a")
    n_done_this_run = 0

    for ph in args.phase_dirs:
        ph_dir = os.path.join(args.traces_root, ph)
        trace_files = sorted(glob.glob(os.path.join(ph_dir, "trace_sharded_*.json")))
        if args.limit_per_phase:
            trace_files = trace_files[:args.limit_per_phase]
        print(f"\n=== {ph}: {len(trace_files)} trace files ===")
        for tf in trace_files:
            try:
                with open(tf) as fh:
                    payload = json.load(fh)
            except Exception as e:
                print(f"  skip {tf}: {e}")
                continue
            task_id = payload.get("task_id")
            trace = payload.get("trace", [])
            n_a = sum(1 for m in trace if m.get("role") == "assistant")
            for t in range(1, n_a + 1):
                if (task_id, t) in done:
                    continue
                t0 = time.time()
                try:
                    res = measure_logprobs_for_turn(
                        model=model, tokenizer=tokenizer, trace=trace,
                        target_idx=t, max_new_tokens=args.max_new_tokens,
                    )
                except Exception:
                    import traceback
                    res = {"error": traceback.format_exc()[:1500]}
                dt = time.time() - t0
                row = {"phase": ph, "task_id": task_id, "turn": t,
                       "elapsed_s": dt, "result": res}
                out_f.write(json.dumps(row, default=str) + "\n")
                out_f.flush()
                n_done_this_run += 1
                if n_done_this_run % 20 == 0:
                    print(f"  [{ph}] {task_id} t{t} {dt:.1f}s "
                          f"(total this run: {n_done_this_run})")
    out_f.close()
    print(f"\nDONE. Wrote {n_done_this_run} new rows to {args.out_path}")


if __name__ == "__main__":
    main()
