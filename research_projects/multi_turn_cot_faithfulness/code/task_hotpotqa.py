"""HotpotQA task implementation for the multi-turn CoT faithfulness pipeline."""
import re
import string
from typing import Dict, Any, Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from task_base import Task


SYSTEM_PROMPT = (
    "Answer the factual question using ONLY information provided in the conversation. "
    "Think step by step. The answer must be a SHORT entity: a name, place, title, "
    "or term (1-5 words). Do NOT write a full sentence. "
    "End your response with: **Answer:** <short entity answer>"
)


def normalize_answer(s: str) -> str:
    """HotpotQA official normalization (verbatim from eval script)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def parse_answer(text: str) -> str:
    """
    Extract final answer using 3-pattern fallback:
      1. \\boxed{...}
      2. **Answer:** ... line
      3. Last non-empty line <= 80 chars
    Returns normalized answer string.
    """
    if not text:
        return ""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return normalize_answer(m.group(1))
    m = re.search(r"^\s*\*\*Answer:\*\*\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        return normalize_answer(m.group(1).strip())
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines and len(lines[-1]) <= 80:
        return normalize_answer(lines[-1])
    return ""


class TaskHotpotQA(Task):
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.answer_extraction_strategy = "gen"

    def get_task_name(self) -> str:
        return "hotpotqa"

    def get_dataset_file(self) -> str:
        return "data/sharded_hotpotqa_30.json"

    def get_samples(self, filter=None) -> List[Dict[str, Any]]:
        import json
        with open(self.get_dataset_file()) as f:
            return json.load(f)

    def get_answer_description(self) -> str:
        return (
            "The answer should be a short entity or phrase (a person's name, "
            "a place, an organization, a year, etc.). It is NOT yes or no."
        )

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        gold_raw = sample.get("gold_answer") or sample.get("answer", "")
        gold_norm = normalize_answer(gold_raw)
        pred_norm = parse_answer(extracted_answer)
        is_correct = bool(pred_norm) and (pred_norm == gold_norm)
        return {
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "exact_answer": pred_norm,
        }

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return f"{self.system_prompt}\n\nQuestion: {sample['question']}"

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        shards_text = "\n".join(f"- {s['shard']}" for s in sample.get("shards", []))
        return f"{self.system_prompt}\n\nContext:\n{shards_text}\n\nQuestion: {sample['question']}"

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": sample["task_id"],
            "question": sample["question"],
            "answer": sample.get("answer", ""),
        }
