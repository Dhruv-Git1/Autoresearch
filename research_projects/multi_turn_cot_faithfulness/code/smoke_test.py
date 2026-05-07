"""End-to-end smoke test: 1 sharded GSM8K conversation with R1-Distill-7B.

Run from inside the lost_in_conversation/ directory.

Verifies:
  - Both models load on the GPU
  - The simulator runs a multi-turn conversation
  - Each assistant turn contains a parseable <think>...</think> block
  - The math evaluator scores the final answer
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/dev/shm/vasudev_hf_cache")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Import path: this file is run from lost_in_conversation/ dir
from simulator_sharded import ConversationSimulatorSharded
from model_local import split_thinking


def load_one_math_sample(idx=0):
    with open("data/sharded_instructions_600.json") as f:
        data = json.load(f)
    math = [d for d in data if d["task"] == "math"]
    return math[idx]


def main():
    sample = load_one_math_sample(0)
    print(f"=== sample task_id: {sample['task_id']} ===")
    print(f"question: {sample['question']}")
    print(f"gold answer: {sample['answer'].split('####')[-1].strip()}")
    print(f"shards ({len(sample['shards'])}):")
    for s in sample["shards"]:
        print(f"  {s['shard_id']}: {s['shard']}")
    print()

    sim = ConversationSimulatorSharded(
        sample=sample,
        assistant_model="deepseek-r1-distill-qwen-7b",
        system_model="qwen2.5-7b-instruct",
        user_model="qwen2.5-7b-instruct",
        assistant_temperature=0.6,
        user_temperature=1.0,
        dataset_fn="data/sharded_instructions_600.json",
        log_folder="logs",
    )

    t0 = time.time()
    is_correct, score = sim.run(verbose=True, save_log=True)
    print(f"\n=== conversation finished in {time.time()-t0:.1f}s ===")
    print(f"is_correct={is_correct}, score={score}")

    # Inspect thinking blocks per assistant turn
    print("\n=== <think> block inspection ===")
    for i, msg in enumerate(sim.trace):
        if msg["role"] == "assistant":
            thinking, answer = split_thinking(msg["content"])
            print(f"-- turn {i} (assistant) --")
            print(f"   thinking_chars={len(thinking)}, answer_chars={len(answer)}")
            print(f"   thinking preview: {thinking[:200]}...")
            print(f"   answer preview:   {answer[:200]}...")

    # Save smoke trace separately
    out_path = f"../results/smoke_trace_{sample['task_id'].replace('/', '_')}.json"
    os.makedirs("../results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "sample_task_id": sample["task_id"],
            "is_correct": is_correct,
            "score": score,
            "trace": sim.trace,
        }, f, indent=2, default=str)
    print(f"\nsaved trace to {out_path}")


if __name__ == "__main__":
    main()
