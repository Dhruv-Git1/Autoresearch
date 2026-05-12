"""Task 2 — H2b: Information integration test.

Tests the MECHANISM of anchoring without needing correctness labels.

For each turn, the 0%-truncation answer is what the conversation context alone
determines (no current-turn CoT).  Across consecutive turns in a conversation,
we measure whether this context-only answer changes — i.e., whether the model
updates its belief as new shards arrive.

answer_update_rate = fraction of consecutive-turn pairs where 0%-level answer changes.
Prediction: high anchoring_rate <-> low answer_update_rate  (Spearman rho < 0).
"""
import json
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_NUM_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

FAITH_PATHS = [
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase2\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase3\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase4\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s1\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s2\faithfulness.jsonl",
    r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase5_s3\faithfulness.jsonl",
]

OUT_DIR = r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\h2b"
FIG_PATH = r"d:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\paper\figures\h2b_integration.png"


def extract_numeric(text):
    if not text:
        return ""
    matches = _NUM_RE.findall(text)
    if not matches:
        return ""
    last = matches[-1]
    if isinstance(last, tuple):
        last = next((m for m in last if m), "")
    return last.strip().lstrip("$").rstrip(",.")


def is_anchored(row):
    tr = row.get("result", {})
    trunc_results = tr.get("truncation_results", [])
    if len(trunc_results) < 3:
        return None
    nums = [extract_numeric(t.get("regen_answer_preview", "")) for t in trunc_results]
    nums = [n for n in nums if n]
    if len(nums) < 3:
        return None
    return len(set(nums)) == 1


def get_zero_pct_answer(row):
    """Return numeric answer at 0% CoT truncation (context-only)."""
    tr = row.get("result", {})
    trunc_results = tr.get("truncation_results", [])
    if not trunc_results:
        return None
    zero = trunc_results[0]
    # Confirm it's the 0% entry
    if zero.get("pct", -1) != 0.0:
        # Find 0% entry explicitly
        for t in trunc_results:
            if t.get("pct", -1) == 0.0:
                zero = t
                break
        else:
            return None
    ans = extract_numeric(zero.get("regen_answer_preview", ""))
    return ans if ans else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for p in FAITH_PATHS:
        if not os.path.exists(p):
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows across {len({r['task_id'] for r in rows})} conversations")

    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["task_id"]].append(r)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda r: r["turn"])

    per_conv_data = []

    for tid, turn_rows in sorted(by_tid.items()):
        # Anchoring rate for this conversation
        anchored_vals = [is_anchored(r) for r in turn_rows]
        valid_anchored = [a for a in anchored_vals if a is not None]
        if len(valid_anchored) < 3:
            continue
        anchoring_rate = sum(valid_anchored) / len(valid_anchored)

        # 0%-level answers across turns
        zero_answers = [get_zero_pct_answer(r) for r in turn_rows]
        zero_answers_valid = [(i, a) for i, a in enumerate(zero_answers) if a is not None]

        if len(zero_answers_valid) < 3:
            continue

        # answer_update_rate: fraction of consecutive pairs where answer changes
        n_pairs = len(zero_answers_valid) - 1
        n_updates = sum(
            1 for i in range(n_pairs)
            if zero_answers_valid[i][1] != zero_answers_valid[i + 1][1]
        )
        answer_update_rate = n_updates / n_pairs if n_pairs > 0 else float("nan")

        if math.isnan(answer_update_rate):
            continue

        per_conv_data.append({
            "conv_id": tid,
            "anchoring_rate": round(anchoring_rate, 4),
            "answer_update_rate": round(answer_update_rate, 4),
            "n_turns": len(valid_anchored),
            "n_valid_zero_answers": len(zero_answers_valid),
        })

    print(f"Conversations with sufficient data: {len(per_conv_data)}")

    anchoring_rates = [c["anchoring_rate"] for c in per_conv_data]
    update_rates = [c["answer_update_rate"] for c in per_conv_data]

    if len(per_conv_data) >= 5:
        rho, p_val = spearmanr(anchoring_rates, update_rates)
        rho = float(rho)
        p_val = float(p_val)
    else:
        rho, p_val = float("nan"), float("nan")

    stats = {
        "n_conversations": len(per_conv_data),
        "spearman_rho": round(rho, 4) if not math.isnan(rho) else "nan",
        "spearman_p": round(p_val, 4) if not math.isnan(p_val) else "nan",
        "verdict": (
            f"Significant negative correlation (rho={rho:.3f}, p={p_val:.3f}): "
            "high anchoring predicts low answer update rate (belief-updating failure)"
            if (not math.isnan(rho) and rho < 0 and p_val < 0.05) else
            f"Not significant (rho={rho:.3f}, p={p_val:.3f})"
            if not math.isnan(rho) else "Insufficient data"
        ),
        "per_conv_data": per_conv_data,
    }

    with open(os.path.join(OUT_DIR, "integration_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_conv_data"}, indent=2))
    print(f"Verdict: {stats['verdict']}")

    # Figure: scatter plot with regression line
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(anchoring_rates, update_rates, alpha=0.7, color="#3498db",
               edgecolors="white", s=60, zorder=3)

    if len(per_conv_data) >= 5 and not math.isnan(rho):
        # Regression line (linear for visual aid)
        m, b = np.polyfit(anchoring_rates, update_rates, 1)
        xline = np.linspace(min(anchoring_rates), max(anchoring_rates), 100)
        ax.plot(xline, m * xline + b, "r--", lw=1.5, alpha=0.8,
                label=f"Linear fit (rho={rho:.3f}, p={p_val:.3f})")

    ax.set_xlabel("Anchoring rate (fraction of turns anchored)", fontsize=11)
    ax.set_ylabel("Answer update rate\n(fraction of consecutive-turn pairs where 0%-CoT answer changes)", fontsize=10)
    ax.set_title(
        f"H2b: Does anchoring predict belief-updating failure?\n"
        f"Spearman ρ = {rho:.3f}, p = {p_val:.3f}, N = {len(per_conv_data)} conversations",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {FIG_PATH}")


if __name__ == "__main__":
    main()
