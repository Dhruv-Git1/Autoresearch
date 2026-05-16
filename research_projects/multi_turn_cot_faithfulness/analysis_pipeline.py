"""
analysis_pipeline.py - Faithfulness analysis and metrics pipeline.
Computes anchoring rates, turn-level gradients, and statistical tests.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy import stats

ANCHORING_THRESHOLD = 0.8
COT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.00]
MIN_CONV_SIZE = 5
LENGTH_BINS = {
    "short":  (0,  10),
    "medium": (10, 20),
    "long":   (20, float("inf")),
}


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_anchored(row: Dict, threshold: float = ANCHORING_THRESHOLD) -> bool:
    answers = [row.get(f"regen_{int(l * 100)}") for l in COT_LEVELS]
    answers = [a for a in answers if a is not None]
    if len(answers) < 3:
        return False
    majority = max(set(answers), key=answers.count)
    return answers.count(majority) / len(answers) >= threshold


def anchoring_rate(rows: List[Dict]) -> float:
    return sum(is_anchored(r) for r in rows) / len(rows) if rows else 0.0


def length_stratify(rows: List[Dict]) -> Dict[str, List[Dict]]:
    bins: Dict[str, List] = defaultdict(list)
    for r in rows:
        n = r.get("turn_index", 0)
        for label, (lo, hi) in LENGTH_BINS.items():
            if lo <= n < hi:
                bins[label].append(r)
                break
    return dict(bins)


def compute_gradient(rows: List[Dict], n_bins: int = 5) -> List[float]:
    turns = [r.get("turn_index", 0) for r in rows]
    if not turns:
        return []
    mx = max(turns) + 1
    bin_size = max(1, mx // n_bins)
    return [
        anchoring_rate([r for r in rows if b * bin_size <= r.get("turn_index", 0) < (b + 1) * bin_size])
        for b in range(n_bins)
    ]


def run_chi2_test(rows: List[Dict]) -> Tuple[float, float]:
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r.get("conv_id", "?")].append(is_anchored(r))
    obs = np.array([[sum(v), len(v) - sum(v)] for v in by_conv.values()])
    chi2, p, *_ = stats.chi2_contingency(obs)
    return float(chi2), float(p)


def icc_one_way(rows: List[Dict]) -> float:
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r.get("conv_id", "?")].append(float(is_anchored(r)))
    groups = list(by_conv.values())
    grand = np.mean([x for g in groups for x in g])
    k = len(groups)
    n = sum(len(g) for g in groups)
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups) / (n - k)
    return float((msb - msw) / (msb + msw))


def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


# EXTEND_ANALYSIS
