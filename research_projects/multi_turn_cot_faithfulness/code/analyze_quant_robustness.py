"""Exp 4: Quantization robustness — compare INT8 baseline vs INT4 re-run.

Loads two faithfulness.jsonl files (INT8 = existing Phase 4, INT4 = freshly
re-run on the same trace files), labels each turn anchored vs exploring via
is_anchored(), and compares per-conversation anchored fractions.

Outputs:
  results/quant_robustness_int4/quant_analysis.json
  paper/figures/quant_robustness.png
"""
import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_bistability_analysis import is_anchored


def load_anchored_per_turn(path: str) -> Dict:
    """Return dict: task_id -> list of (turn, is_anchored_bool) sorted by turn,
    skipping turns where is_anchored() is None."""
    by_tid = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            tid = r.get("task_id")
            turn = r.get("turn")
            if tid is None or turn is None:
                continue
            a = is_anchored(r)
            if a is None:
                continue
            by_tid[tid].append((int(turn), bool(a)))
    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: x[0])
    return dict(by_tid)


def per_conv_fraction(by_tid: Dict) -> Dict[str, float]:
    return {tid: (sum(1 for _, a in turns if a) / max(1, len(turns)))
            for tid, turns in by_tid.items()}


def per_turn_agreement(int8: Dict, int4: Dict) -> Dict[str, Any]:
    """Per-turn agreement on anchored vs exploring, joined by (task_id, turn)."""
    n_match = 0
    n_total = 0
    n_int8_only_anchored = 0  # INT8 calls anchored, INT4 calls exploring
    n_int4_only_anchored = 0
    for tid in int8:
        if tid not in int4:
            continue
        int8_dict = dict(int8[tid])
        int4_dict = dict(int4[tid])
        common = sorted(set(int8_dict) & set(int4_dict))
        for t in common:
            n_total += 1
            a8 = int8_dict[t]
            a4 = int4_dict[t]
            if a8 == a4:
                n_match += 1
            elif a8 and not a4:
                n_int8_only_anchored += 1
            elif a4 and not a8:
                n_int4_only_anchored += 1
    return {
        "n_compared_turns": n_total,
        "n_agree": n_match,
        "agreement_rate": (n_match / n_total) if n_total else float("nan"),
        "int8_only_anchored": n_int8_only_anchored,
        "int4_only_anchored": n_int4_only_anchored,
    }


def make_figure(int8_frac: Dict[str, float], int4_frac: Dict[str, float],
                out_path: str):
    common = sorted(set(int8_frac) & set(int4_frac))
    x = [int8_frac[t] for t in common]
    y = [int4_frac[t] for t in common]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: paired-line plot
    ax = axes[0]
    for tid, a, b in zip(common, x, y):
        ax.plot([0, 1], [a, b], "-", color="#1a7a3a", alpha=0.6)
        ax.scatter([0, 1], [a, b], color=["#1a7a3a", "#a31515"], s=50, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["INT8\n(baseline)", "INT4\n(re-run)"])
    ax.set_ylabel("Per-conversation anchored fraction")
    ax.set_title(f"Paired comparison (n={len(common)} convs)")
    ax.grid(alpha=0.3, axis="y")

    # Right: scatter with y=x
    ax = axes[1]
    ax.scatter(x, y, s=80, color="#1a7a3a", alpha=0.7)
    lim = [0, max(max(x or [0]), max(y or [0])) * 1.1 + 0.05]
    ax.plot(lim, lim, "k--", alpha=0.4, label="$y=x$ (no shift)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("INT8 anchored fraction")
    ax.set_ylabel("INT4 anchored fraction")
    mean_shift = (np.mean(y) - np.mean(x)) if x else 0.0
    ax.set_title(f"INT4 vs INT8 anchored fraction\n(mean shift {mean_shift*100:+.1f} pp)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Quantization robustness: INT4 vs INT8 anchored classifications",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    base = "research_projects/multi_turn_cot_faithfulness/results"
    ap.add_argument("--int8_path", default=f"{base}/phase4/faithfulness.jsonl")
    ap.add_argument("--int4_path", default=f"{base}/quant_robustness_int4/faithfulness.jsonl")
    ap.add_argument("--out_dir", default=f"{base}/quant_robustness_int4")
    ap.add_argument("--fig_path", default="research_projects/multi_turn_cot_faithfulness/paper/figures/quant_robustness.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)

    print(f"Loading INT8 baseline: {args.int8_path}")
    int8 = load_anchored_per_turn(args.int8_path)
    print(f"  {len(int8)} convs, {sum(len(v) for v in int8.values())} labelled turns")

    print(f"Loading INT4 results: {args.int4_path}")
    int4 = load_anchored_per_turn(args.int4_path)
    print(f"  {len(int4)} convs, {sum(len(v) for v in int4.values())} labelled turns")

    common = sorted(set(int8) & set(int4))
    print(f"  {len(common)} common conversations")

    int8_f = per_conv_fraction({t: int8[t] for t in common})
    int4_f = per_conv_fraction({t: int4[t] for t in common})

    print("\n=== Per-conversation anchored fraction ===")
    print(f"  {'task_id':<32} {'INT8':>6} {'INT4':>6} {'shift_pp':>9}")
    for tid in common:
        print(f"  {tid:<32} {int8_f[tid]*100:>5.1f}% {int4_f[tid]*100:>5.1f}% "
              f"{(int4_f[tid]-int8_f[tid])*100:>+6.1f}")

    pair_stats = {}
    if len(common) >= 2:
        x = np.array([int8_f[t] for t in common])
        y = np.array([int4_f[t] for t in common])
        mean_int8 = float(np.mean(x))
        mean_int4 = float(np.mean(y))
        mean_shift = mean_int4 - mean_int8
        try:
            stat, p = wilcoxon(y - x)
            wp = float(p)
        except ValueError:
            wp = float("nan")
        # Bootstrap CI on mean shift
        rng = np.random.default_rng(42)
        boots = [np.mean(rng.choice(y - x, size=len(x), replace=True))
                 for _ in range(10000)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        pair_stats = {
            "n_paired_convs": len(common),
            "mean_int8": mean_int8,
            "mean_int4": mean_int4,
            "mean_shift": mean_shift,
            "mean_shift_ci95": [lo, hi],
            "wilcoxon_p": wp,
        }
        print(f"\n  Mean INT8 anchored = {mean_int8*100:.2f}%")
        print(f"  Mean INT4 anchored = {mean_int4*100:.2f}%")
        print(f"  Mean shift = {mean_shift*100:+.2f}pp (95% CI {lo*100:+.2f} to {hi*100:+.2f})")
        print(f"  Wilcoxon p = {wp:.4g}")

    agree = per_turn_agreement(int8, int4)
    print(f"\n  Per-turn agreement: {agree['n_agree']}/{agree['n_compared_turns']} "
          f"= {agree['agreement_rate']*100:.1f}%")
    print(f"  INT8-only anchored: {agree['int8_only_anchored']}")
    print(f"  INT4-only anchored: {agree['int4_only_anchored']}")

    out_json = os.path.join(args.out_dir, "quant_analysis.json")
    with open(out_json, "w") as f:
        json.dump({
            "int8_per_conv_fraction": int8_f,
            "int4_per_conv_fraction": int4_f,
            "paired_stats": pair_stats,
            "per_turn_agreement": agree,
        }, f, indent=2)
    print(f"\nwrote {out_json}")

    if int8_f and int4_f:
        make_figure(int8_f, int4_f, args.fig_path)
    print("DONE.")


if __name__ == "__main__":
    main()
