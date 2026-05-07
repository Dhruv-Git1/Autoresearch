"""Day 1 — FULL vs SHARDED accuracy on N math samples.

Runs both simulation types on the same set of GSM8K math samples to validate:
  Sanity floor:   FULL accuracy on R1-Distill-7B should be in the published range
                  (R1-Distill-7B reports >85% on GSM8K; even our limited setup
                  should comfortably exceed 40%).
  Sanity ceiling: SHARDED accuracy < FULL accuracy on the same prompts
                  (Laban et al. 2025: avg -39%).

Saves per-turn data (assistant raw response, parsed thinking, parsed answer,
shard ids, system verifier classification, evaluator score) to a JSONL for
downstream faithfulness analysis.

Run from inside lost_in_conversation/ directory.
"""
import json
import os
import sys
import time
import argparse
import random
import traceback

os.environ.setdefault("HF_HOME", "/dev/shm/vasudev_hf_cache")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from simulator_full import ConversationSimulatorFull
from simulator_sharded import ConversationSimulatorSharded
from model_local import split_thinking


def extract_per_turn_records(sim_trace, conv_type, task_id, score, is_correct):
    """Walk a simulator trace and emit one record per assistant turn.

    Each record contains the parsed thinking and answer halves so downstream
    analysis can apply faithfulness measurements without re-parsing.
    """
    records = []
    user_turn_idx = 0  # how many user shards revealed before this assistant
    cum_shard_ids = []
    last_user_msg = None
    last_shard_revealed_idx = None
    last_verification = None
    last_eval = None

    for i, msg in enumerate(sim_trace):
        role = msg.get("role")
        if role == "user":
            user_turn_idx += 1
            last_user_msg = msg.get("content", "")
        elif role == "log":
            content = msg.get("content", {})
            if isinstance(content, dict):
                if content.get("type") == "shard_revealed":
                    cum_shard_ids.append(content.get("shard_id"))
                elif content.get("type") == "system-verification":
                    last_verification = content.get("response", {})
                elif content.get("type") == "answer-evaluation":
                    last_eval = content
        elif role == "assistant":
            raw = msg.get("content", "")
            thinking, answer_text = split_thinking(raw)
            records.append({
                "task_id": task_id,
                "conv_type": conv_type,
                "assistant_turn_idx": len([r for r in records if r["task_id"] == task_id and r["conv_type"] == conv_type]) + 1,
                "trace_idx": i,
                "user_turn_idx_at_this_assistant": user_turn_idx,
                "cum_shard_ids_revealed": list(cum_shard_ids),
                "n_shards_revealed_so_far": len(cum_shard_ids),
                "last_user_message": last_user_msg,
                "raw_response": raw,
                "thinking_chars": len(thinking),
                "thinking_text": thinking,
                "answer_chars": len(answer_text),
                "answer_text": answer_text,
                "verification_response_type": last_verification.get("response_type") if isinstance(last_verification, dict) else None,
                "extracted_answer": last_eval.get("exact_answer") if last_eval else None,
                "this_turn_is_correct": last_eval.get("is_correct") if last_eval else None,
                "this_turn_score": last_eval.get("score") if last_eval else None,
                "final_is_correct": is_correct,
                "final_score": score,
            })
            # Reset turn-local state so it only attaches to the *next* assistant
            last_verification = None
            last_eval = None
    return records


def run_one(sample, conv_type, models):
    """Run a single FULL or SHARDED sim. Returns (is_correct, score, trace, elapsed_s)."""
    t0 = time.time()
    if conv_type == "full":
        sim = ConversationSimulatorFull(
            sample=sample,
            assistant_model=models["assistant"],
            system_model=models["system"],
            run_concat=False,
            temperature=models["assistant_temp"],
            dataset_fn="data/sharded_instructions_600.json",
            log_folder="logs",
        )
        is_correct, score = sim.run(verbose=False, save_log=False)
        trace = sim_to_trace_full(sim)
    elif conv_type == "sharded":
        sim = ConversationSimulatorSharded(
            sample=sample,
            assistant_model=models["assistant"],
            system_model=models["system"],
            user_model=models["user"],
            assistant_temperature=models["assistant_temp"],
            user_temperature=models["user_temp"],
            dataset_fn="data/sharded_instructions_600.json",
            log_folder="logs",
        )
        is_correct, score = sim.run(verbose=False, save_log=False)
        trace = sim.trace
    else:
        raise ValueError(conv_type)
    return is_correct, score, trace, time.time() - t0


def sim_to_trace_full(sim):
    """ConversationSimulatorFull doesn't expose .trace; reconstruct minimally
    from public attributes after .run()."""
    # We don't have direct access; the run method built `trace` locally and
    # didn't store it. The simplest fix is to monkey-patch run, but for
    # baseline counts we don't need the trace (FULL has only 1 assistant turn).
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--task", type=str, default="math")
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--out_dir", type=str, default="../results/day1")
    ap.add_argument("--assistant_temp", type=float, default=0.6)  # R1 recommended
    ap.add_argument("--user_temp", type=float, default=1.0)
    ap.add_argument("--skip_full", action="store_true")
    ap.add_argument("--skip_sharded", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open("data/sharded_instructions_600.json") as f:
        all_data = json.load(f)
    pool = [d for d in all_data if d["task"] == args.task]
    rnd = random.Random(args.seed)
    rnd.shuffle(pool)
    samples = pool[: args.n_samples]
    print(f"selected {len(samples)} {args.task} samples (seed={args.seed})")

    models = {
        "assistant": "deepseek-r1-distill-qwen-7b",
        "system": "qwen2.5-7b-instruct",
        "user": "qwen2.5-7b-instruct",
        "assistant_temp": args.assistant_temp,
        "user_temp": args.user_temp,
    }

    summary = {"full": [], "sharded": []}
    per_turn_records = []

    for i, sample in enumerate(samples):
        print(f"\n--- [{i+1}/{len(samples)}] {sample['task_id']} ---")
        if not args.skip_full:
            try:
                ok, sc, _, dt = run_one(sample, "full", models)
                print(f"  FULL    : correct={ok}  score={sc}  time={dt:.1f}s")
                summary["full"].append({"task_id": sample["task_id"], "is_correct": ok, "score": sc, "elapsed_s": dt})
            except Exception:
                traceback.print_exc()
                summary["full"].append({"task_id": sample["task_id"], "error": traceback.format_exc()[:500]})

        if not args.skip_sharded:
            try:
                ok, sc, trace, dt = run_one(sample, "sharded", models)
                n_assist = sum(1 for m in trace if m.get("role") == "assistant")
                print(f"  SHARDED : correct={ok}  score={sc}  time={dt:.1f}s  assistant_turns={n_assist}")
                summary["sharded"].append({"task_id": sample["task_id"], "is_correct": ok, "score": sc, "elapsed_s": dt, "n_assistant_turns": n_assist})
                # extract per-turn records
                recs = extract_per_turn_records(trace, "sharded", sample["task_id"], sc, ok)
                per_turn_records.extend(recs)
                # also save full trace for this sample
                with open(os.path.join(args.out_dir, f"trace_sharded_{sample['task_id'].replace('/', '_')}.json"), "w") as f:
                    json.dump({"task_id": sample["task_id"], "is_correct": ok, "score": sc, "trace": trace, "gold_answer": sample["answer"]}, f, indent=2, default=str)
            except Exception:
                traceback.print_exc()
                summary["sharded"].append({"task_id": sample["task_id"], "error": traceback.format_exc()[:500]})

    # write summaries
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "models": models,
            "samples": [s["task_id"] for s in samples],
            "full": summary["full"],
            "sharded": summary["sharded"],
        }, f, indent=2)
    with open(os.path.join(args.out_dir, "per_turn_records.jsonl"), "w") as f:
        for r in per_turn_records:
            f.write(json.dumps(r, default=str) + "\n")

    # print headline numbers
    def ok_rate(rows):
        ok = [r for r in rows if r.get("is_correct") is not None]
        if not ok:
            return None
        return sum(1 for r in ok if r["is_correct"]) / len(ok)

    full_acc = ok_rate(summary["full"])
    sharded_acc = ok_rate(summary["sharded"])
    print("\n=== Day 1 Sanity ===")
    print(f"FULL    accuracy: {full_acc} on n={len([r for r in summary['full'] if 'error' not in r])}")
    print(f"SHARDED accuracy: {sharded_acc} on n={len([r for r in summary['sharded'] if 'error' not in r])}")
    if full_acc is not None and sharded_acc is not None:
        print(f"Drop  (sharded - full): {sharded_acc - full_acc:+.2f}")
    print(f"\nSaved outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
