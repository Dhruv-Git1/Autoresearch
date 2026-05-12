"""Task 3 — Recover correctness labels from local trace files.

Extracts:
  - gold answer from the '#### <number>' line in gold_answer field
  - model's final exact_answer from the last 'answer-evaluation' log entry in trace
  - is_correct = (normalize(model_answer) == normalize(gold_answer))

Writes results/correctness_labels/labels.json.
"""
import json
import math
import os
import re
from collections import defaultdict

TRACE_DIRS = [
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase2",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase3",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase4",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s1",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s2",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s3",
]

FAITH_PATHS = [
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase2\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase3\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase4\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s1\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s2\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s3\faithfulness.jsonl",
]

OUT_DIR = r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\correctness_labels"
_GOLD_RE = re.compile(r"####\s*([\d,]+)")


def normalize_answer(text):
    if text is None:
        return None
    text = str(text).strip().replace(",", "").lstrip("$").rstrip(".")
    try:
        v = float(text)
        return int(v) if v == int(v) else v
    except (ValueError, OverflowError):
        return None


def extract_gold(gold_answer_text):
    if not gold_answer_text:
        return None
    m = _GOLD_RE.search(gold_answer_text)
    if not m:
        return None
    return normalize_answer(m.group(1))


def extract_model_final_answer(trace):
    """Return the last exact_answer from answer-evaluation log entries."""
    last_answer = None
    for entry in reversed(trace):
        content = entry.get("content", {})
        if isinstance(content, dict) and content.get("type") == "answer-evaluation":
            ea = content.get("exact_answer")
            if ea is not None:
                last_answer = ea
                break
    return last_answer


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load conversation IDs that appear in our faithfulness data
    faith_tids = set()
    for p in FAITH_PATHS:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    faith_tids.add(json.loads(line)["task_id"])
    print(f"Conversations in faithfulness data: {len(faith_tids)}")

    labels = {}
    seen_tids = set()

    for td in TRACE_DIRS:
        if not os.path.exists(td):
            continue
        for fn in os.listdir(td):
            if not (fn.startswith("trace_sharded_") and fn.endswith(".json")):
                continue
            fpath = os.path.join(td, fn)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    t = json.load(fh)
            except Exception as e:
                print(f"  WARNING: could not load {fn}: {e}")
                continue

            tid = t.get("task_id")
            if not tid or tid in seen_tids:
                continue
            if tid not in faith_tids:
                continue
            seen_tids.add(tid)

            gold_text = t.get("gold_answer", "")
            gold_val = extract_gold(gold_text)
            trace = t.get("trace", [])
            model_raw = extract_model_final_answer(trace)
            model_val = normalize_answer(model_raw)

            if gold_val is not None and model_val is not None:
                is_correct = (model_val == gold_val)
            else:
                is_correct = None

            labels[tid] = {
                "task_id": tid,
                "gold_answer_raw": gold_text[-100:] if gold_text else None,
                "gold_val": gold_val,
                "model_answer_raw": str(model_raw) if model_raw is not None else None,
                "model_val": model_val,
                "is_correct": is_correct,
                "original_is_correct": t.get("is_correct"),
            }

    n_recovered = sum(1 for v in labels.values() if v["is_correct"] is not None)
    n_correct = sum(1 for v in labels.values() if v["is_correct"] is True)
    n_wrong = sum(1 for v in labels.values() if v["is_correct"] is False)
    n_none = sum(1 for v in labels.values() if v["is_correct"] is None)

    summary = {
        "n_total": len(labels),
        "n_with_label": n_recovered,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_unrecoverable": n_none,
        "accuracy": round(n_correct / n_recovered, 3) if n_recovered else None,
        "labels": labels,
    }

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nRecovered labels for {n_recovered}/{len(labels)} conversations")
    print(f"  Correct: {n_correct}, Wrong: {n_wrong}, Unrecoverable: {n_none}")
    if n_recovered:
        print(f"  Accuracy: {n_correct/n_recovered:.1%}")
    print()
    for tid, v in list(labels.items())[:8]:
        print(f"  {tid}: gold={v['gold_val']} model={v['model_val']} -> is_correct={v['is_correct']}")


if __name__ == "__main__":
    main()
