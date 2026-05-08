"""Phase 2 batch runner — bistability hypothesis.

Runs SHARDED math conversations + Lanham-style counterfactual faithfulness
measurement per turn, saving incrementally so the run can be resumed.

Run from inside lost_in_conversation/ directory:

    LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
        python3 phase2_batch_runner.py --n_samples 8 --max_shards 5 \
        --out_dir ../multi_turn_cot_faithfulness/results/phase2
"""
import argparse
import json
import os
import random
import time
import traceback

os.environ.setdefault("HF_HOME", "/dev/shm/vasudev_hf_cache")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from simulator_sharded import ConversationSimulatorSharded
from model_local import _load_model, _normalize_model_name
from faithfulness_counterfactual import measure_faithfulness_for_turn, faithfulness_score_from_results
from tasks import get_task

# Task IDs already collected in Phase 1 (skip these)
PHASE1_IDS = {
    "sharded-GSM8K/965",
    "sharded-GSM8K/1019",
    "sharded-GSM8K/598",
    "sharded-GSM8K/547",
    "sharded-GSM8K/1246",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--task", type=str, default="math")
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument("--out_dir", type=str, default="../results/phase2")
    ap.add_argument("--max_shards", type=int, default=5,
                    help="Exclude samples with more shards than this (longer conversations).")
    ap.add_argument("--faith_tokens", type=int, default=512,
                    help="max_new_tokens for counterfactual regen (answer-only, 512 is enough).")
    ap.add_argument("--assistant_temp", type=float, default=0.6)
    ap.add_argument("--user_temp", type=float, default=1.0)
    ap.add_argument("--skip_faith", action="store_true")
    ap.add_argument("--exclude_dirs", nargs="*", default=[],
                    help="Dirs containing trace_sharded_*.json from previous runs — those task_ids will be excluded.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    faith_path = os.path.join(args.out_dir, "faithfulness.jsonl")

    # --- Build exclusion set (Phase 1 hardcoded + any prior-run dirs) ---
    exclude_ids = set(PHASE1_IDS)
    for d in args.exclude_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith("trace_sharded_") and fn.endswith(".json"):
                try:
                    with open(os.path.join(d, fn)) as fh:
                        tid = json.load(fh).get("task_id")
                    if tid:
                        exclude_ids.add(tid)
                except Exception:
                    pass
    if len(exclude_ids) > len(PHASE1_IDS):
        print(f"Excluding {len(exclude_ids)} task_ids ({len(exclude_ids)-len(PHASE1_IDS)} from prior runs)", flush=True)

    # --- Select samples ---
    with open("data/sharded_instructions_600.json") as fh:
        all_data = json.load(fh)

    pool = []
    for d in all_data:
        if d.get("task") != args.task:
            continue
        if d.get("task_id") in exclude_ids:
            continue
        n_shards = len(d.get("shards", []))
        if n_shards > args.max_shards:
            continue
        pool.append(d)

    rnd = random.Random(args.seed)
    rnd.shuffle(pool)
    samples = pool[: args.n_samples]
    print(f"Phase 2: {len(samples)} samples  (task={args.task}, max_shards={args.max_shards}, seed={args.seed})", flush=True)
    for s in samples:
        print(f"  {s['task_id']}  shards={len(s.get('shards', []))}", flush=True)

    # --- Resume state: which task_ids already have faithfulness records ---
    done_faith_ids = set()
    if os.path.exists(faith_path):
        for line in open(faith_path):
            line = line.strip()
            if line:
                try:
                    done_faith_ids.add(json.loads(line)["task_id"])
                except Exception:
                    pass
    print(f"resuming: {len(done_faith_ids)} task_ids already in faithfulness.jsonl", flush=True)

    # --- Pre-load BOTH models before any conversation starts ---
    # This fails fast if VRAM is insufficient, avoids OOM mid-run.
    print("\nLoading R1-Distill-7B...", flush=True)
    r1_model, r1_tok = _load_model(_normalize_model_name("deepseek-r1-distill-qwen-7b"))
    print("Loading Qwen2.5-7B-Instruct...", flush=True)
    _load_model(_normalize_model_name("qwen2.5-7b-instruct"))
    task_obj = get_task(args.task)
    models = {
        "assistant": "deepseek-r1-distill-qwen-7b",
        "system": "qwen2.5-7b-instruct",
        "user": "qwen2.5-7b-instruct",
        "assistant_temp": args.assistant_temp,
        "user_temp": args.user_temp,
    }

    summary_rows = []
    faith_f = open(faith_path, "a")

    for i, sample in enumerate(samples):
        tid = sample["task_id"]
        gold = sample.get("answer", "")
        n_shards_sample = len(sample.get("shards", []))
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(samples)}] {tid}  shards={n_shards_sample}", flush=True)
        print(f"{'='*60}", flush=True)

        trace_path = os.path.join(args.out_dir, f"trace_sharded_{tid.replace('/', '_')}.json")

        # --- Step 1: SHARDED conversation (skip if trace already saved) ---
        if os.path.exists(trace_path):
            print("  trace exists, loading...", flush=True)
            with open(trace_path) as fh:
                payload = json.load(fh)
            trace = payload["trace"]
            is_correct = payload.get("is_correct")
            score = payload.get("score")
            n_assistant = sum(1 for m in trace if m.get("role") == "assistant")
            print(f"  loaded: turns={n_assistant}, correct={is_correct}, score={score}", flush=True)
        else:
            t0 = time.time()
            try:
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
                print(f"  starting conversation (verbose — each turn printed below)...", flush=True)
                is_correct, score = sim.run(verbose=True, save_log=False)
                trace = sim.trace
                n_assistant = sum(1 for m in trace if m.get("role") == "assistant")
                elapsed = time.time() - t0
                print(f"  SHARDED: correct={is_correct}  score={score}  turns={n_assistant}  time={elapsed:.1f}s", flush=True)
                with open(trace_path, "w") as fh:
                    json.dump({
                        "task_id": tid,
                        "is_correct": is_correct,
                        "score": score,
                        "n_assistant_turns": n_assistant,
                        "gold_answer": gold,
                        "trace": trace,
                    }, fh, indent=2, default=str)
            except Exception:
                traceback.print_exc()
                summary_rows.append({"task_id": tid, "error": traceback.format_exc()[:400]})
                continue

        summary_rows.append({
            "task_id": tid,
            "is_correct": bool(is_correct),
            "score": score,
            "n_assistant_turns": n_assistant,
            "n_shards": n_shards_sample,
        })

        # --- Step 2: Faithfulness measurement ---
        if args.skip_faith:
            continue
        if tid in done_faith_ids:
            print(f"  faithfulness already done, skipping")
            continue

        n_assistant = sum(1 for m in trace if m.get("role") == "assistant")
        print(f"  measuring faithfulness on {n_assistant} turns...", flush=True)
        for t_idx in range(1, n_assistant + 1):
            t0 = time.time()
            try:
                res = measure_faithfulness_for_turn(
                    model=r1_model, tokenizer=r1_tok, task_obj=task_obj,
                    trace=trace, target_assistant_idx=t_idx,
                    gold_answer=gold,
                    max_new_tokens=args.faith_tokens,
                )
            except Exception:
                res = {"error": traceback.format_exc()[:800]}
            elapsed = time.time() - t0
            faith = faithfulness_score_from_results(res) if "truncation_results" in res else None
            # Show the actual answers so we can verify non-garbage output
            answers_preview = []
            if "truncation_results" in res:
                for tr in res["truncation_results"]:
                    txt = tr.get("regen_answer_preview", "")[:40].replace("\n", " ")
                    answers_preview.append(f"{int(tr['pct']*100)}%:'{txt}'")
            row = {
                "task_id": tid,
                "turn": t_idx,
                "elapsed_s": round(elapsed, 1),
                "faithfulness_score": faith,
                "result": res,
            }
            print(f"    turn {t_idx:2d}  faith={faith}  ({elapsed:.1f}s)  {' | '.join(answers_preview[:3])}", flush=True)
            faith_f.write(json.dumps(row, default=str) + "\n")
            faith_f.flush()

    faith_f.close()

    # --- Write summary ---
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump({
            "args": vars(args),
            "n_samples": len(samples),
            "samples": [s["task_id"] for s in samples],
            "rows": summary_rows,
            "n_correct": sum(1 for r in summary_rows if r.get("is_correct")),
            "accuracy": (sum(1 for r in summary_rows if r.get("is_correct")) /
                         max(1, len([r for r in summary_rows if "error" not in r]))),
        }, fh, indent=2)

    print(f"\nPhase 2 complete. Results in {args.out_dir}/", flush=True)
    correct = sum(1 for r in summary_rows if r.get("is_correct"))
    total = len([r for r in summary_rows if "error" not in r])
    print(f"Accuracy: {correct}/{total} = {correct/max(1,total):.1%}", flush=True)


if __name__ == "__main__":
    main()
