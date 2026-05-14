"""
Prepare HotpotQA sharded dataset for the replication experiment.
Run from repo root:
  python research_projects/multi_turn_cot_faithfulness/code/prepare_hotpotqa.py

Outputs:
  research_projects/multi_turn_cot_faithfulness/data/sharded_hotpotqa_30.json
  research_projects/multi_turn_cot_faithfulness/data/hotpotqa_prep_report.txt
"""
import json
import random
import re
import string
from pathlib import Path

RANDOM_SEED = 42
N_POOL = 120     # filter to this many candidates, then keep first N_KEEP
N_KEEP = 30
MIN_SHARDS = 8
MAX_SHARDS = 14
MIN_SUPPORT_SENTS = 6

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "sharded_hotpotqa_30.json"
OUT_REPORT = OUT_DIR / "hotpotqa_prep_report.txt"


def normalize_answer(s: str) -> str:
    """HotpotQA official normalization (verbatim from eval script)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def build_shards(item: dict, rng: random.Random) -> list | None:
    """
    Returns a list of shard dicts {"shard_id": int, "shard": str}, or None if
    the item does not pass filters.

    Sharding strategy:
      - Collect all sentences from supporting paragraphs.
      - Label each as "essential" (in supporting_facts) or "distractor".
      - Target n_shards in [MIN_SHARDS, MAX_SHARDS].
      - Keep all essential sentences; fill to MIN_SHARDS with distractors.
      - Shuffle, then enforce >=1 essential shard in second half.

    HotpotQA datasets-library context format:
      context = {"title": [str, ...], "sentences": [[str, ...], ...]}
      supporting_facts = {"title": [str, ...], "sent_id": [int, ...]}
    """
    sf_raw = item.get("supporting_facts", {})
    if isinstance(sf_raw, dict) and "title" in sf_raw and "sent_id" in sf_raw:
        sf_set = set(zip(sf_raw["title"], sf_raw["sent_id"]))
    elif isinstance(sf_raw, dict):
        sf_set = {(title, sid) for title, sent_ids in sf_raw.items() for sid in sent_ids}
    else:
        sf_set = {tuple(x) for x in sf_raw}

    context = item.get("context", {})
    if isinstance(context, dict) and "title" in context and "sentences" in context:
        ctx_items = list(zip(context["title"], context["sentences"]))
    elif isinstance(context, dict):
        ctx_items = list(context.items())
    else:
        ctx_items = [(x[0], x[1]) for x in context]

    essential = []
    distractor = []

    for title, sentences in ctx_items:
        for sid, sent in enumerate(sentences):
            sent = str(sent).strip()
            if len(sent) < 10:
                continue
            if (title, sid) in sf_set:
                essential.append(sent)
            else:
                distractor.append(sent)

    if len(essential) < 2:
        return None

    # Pick a random target length in [MIN_SHARDS, MAX_SHARDS] to create
    # conversation-length variance for the length gradient analysis.
    target = rng.randint(MIN_SHARDS, MAX_SHARDS)
    rng.shuffle(distractor)
    if len(essential) >= target:
        chosen = essential[:target]
    else:
        needed = target - len(essential)
        chosen = essential + distractor[:needed]

    if len(chosen) < MIN_SHARDS:
        return None

    rng.shuffle(chosen)
    half = len(chosen) // 2
    second_half = chosen[half:]
    essential_set = set(essential)
    if not any(s in essential_set for s in second_half):
        first_half = chosen[:half]
        for i, s in enumerate(first_half):
            if s in essential_set:
                j = rng.randrange(half, len(chosen))
                chosen[i], chosen[j] = chosen[j], chosen[i]
                break

    # Convert to the expected {"shard_id": int, "shard": str} format
    return [{"shard_id": idx + 1, "shard": sent} for idx, sent in enumerate(chosen)]


def main():
    from datasets import load_dataset

    rng = random.Random(RANDOM_SEED)
    print("Loading HotpotQA dev set (distractor)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"  Total: {len(ds)} items")

    candidates = []
    for item in ds:
        if item.get("type") != "bridge":
            continue
        if item.get("level") not in {"medium", "hard"}:
            continue
        ans = item.get("answer", "").strip().lower()
        if ans in {"yes", "no"} or len(ans) < 2:
            continue
        sf = item.get("supporting_facts", {})
        n_sf = sum(len(v) for v in sf.values()) if isinstance(sf, dict) else len(sf)
        if n_sf < MIN_SUPPORT_SENTS:
            continue

        shards = build_shards(item, rng)
        if shards is None or len(shards) < MIN_SHARDS:
            continue

        candidates.append({
            "_id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "shards": shards,
            "n_shards": len(shards),
        })

        if len(candidates) >= N_POOL:
            break

    print(f"  Pool after filtering: {len(candidates)}")
    if len(candidates) < N_KEEP:
        print(f"WARNING: only {len(candidates)} candidates, wanted {N_KEEP}.")

    selected = candidates[:N_KEEP]

    out_items = []
    for i, c in enumerate(selected):
        out_items.append({
            "id": i,
            "task_id": f"hotpotqa_{i:04d}_{c['_id'][:8]}",
            "task": "hotpotqa",
            "question": c["question"],
            "gold_answer": normalize_answer(c["answer"]),
            "answer": c["answer"],
            "shards": c["shards"],
            "n_shards": c["n_shards"],
        })

    OUT_JSON.write_text(json.dumps(out_items, indent=2), encoding="utf-8")
    print(f"  Wrote {len(out_items)} items to {OUT_JSON}")

    shard_counts = [x["n_shards"] for x in out_items]
    report_lines = [
        "HotpotQA prep report",
        f"  Items: {len(out_items)}",
        f"  Shard count: min={min(shard_counts)} max={max(shard_counts)} "
        f"mean={sum(shard_counts)/len(shard_counts):.1f}",
        "",
        "5 example items:",
    ]
    for ex in out_items[:5]:
        report_lines.append(f"\n  task_id: {ex['task_id']}")
        report_lines.append(f"  question: {ex['question']}")
        report_lines.append(f"  raw_answer: {ex['answer']}")
        report_lines.append(f"  gold_answer (normalized): {ex['gold_answer']}")
        report_lines.append(f"  n_shards: {ex['n_shards']}")
        report_lines.append(f"  shard[0]: {ex['shards'][0]}")
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
