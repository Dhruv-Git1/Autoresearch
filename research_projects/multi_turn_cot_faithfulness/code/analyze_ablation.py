"""Exp: Prior-answer ablation — compare original INT8 baseline vs INT8 re-run
on the same conversations with prior assistant answers redacted.

This test isolates whether anchoring is driven by the model lexically reading
its own prior answers in the visible context (the "repetition confound"), or
by something deeper.

If anchored rate drops sharply when prior answers are redacted, lexical
copying was the mechanism. If anchored rate persists, the model is
internally committed via something the prior answer text was not carrying.

Outputs:
  results/ablation_no_prior_answer/ablation_analysis.json
  paper/figures/ablation_no_prior_answer.png
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_bistability_analysis import is_anchored


def load_anchored_per_turn(path: str) -> Dict:
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


def per_turn_agreement(base: Dict, abl: Dict) -> Dict[str, Any]:
    n_match = 0
    n_total = 0
    n_base_only_anchored = 0
    n_abl_only_anchored = 0
    for tid in base:
        if tid not in abl:
            continue
        bd = dict(base[tid])
        ad = dict(abl[tid])
        common = sorted(set(bd) & set(ad))
        for t in common:
            n_total += 1
            ab = bd[t]
            aa = ad[t]
            if ab == aa:
                n_match += 1
            elif ab and not aa:
                n_base_only_anchored += 1
            elif aa and not ab:
                n_abl_only_anchored += 1
    return {
        "n_compared_turns": n_total,
        "n_agree": n_match,
        "agreement_rate": (n_match / n_total) if n_total else float("nan"),
        "baseline_only_anchored": n_base_only_anchored,
        "ablated_only_anchored": n_abl_only_anchored,
    }


def make_figure(base_frac: Dict[str, float], abl_frac: Dict[str, float],
                out_path: str):
    common = sorted(set(base_frac) & set(abl_frac))
    x = [base_frac[t] for t in common]
    y = [abl_frac[t] for t in common]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for a, b in zip(x, y):
        ax.plot([0, 1], [a, b], "-", color="#1a7a3a", alpha=0.6)
        ax.scatter([0, 1], [a, b], color=["#1a7a3a", "#a31515"], s=50, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(prior answers visible)",
                        "Ablated\n(prior answers redacted)"])
    ax.set_ylabel("Per-conversation anchored fraction")
    ax.set_title(f"Paired comparison (n={len(common)} convs)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.scatter(x, y, s=80, color="#1a7a3a", alpha=0.7)
    lim = [0, max(max(x or [0]), max(y or [0])) * 1.1 + 0.05]
    ax.plot(lim, lim, "k--", alpha=0.4, label="$y=x$ (no change)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Baseline anchored fraction")
    ax.set_ylabel("Ablated anchored fraction")
    mean_shift = (np.mean(y) - np.mean(x)) if x else 0.0
    ax.set_title(f"Ablated vs Baseline\n(mean shift {mean_shift*100:+.1f} pp)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Prior-answer ablation: does anchoring persist without lexical context-copying?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    base = "research_projects/multi_turn_cot_faithfulness/results"
    ap.add_argument("--baseline_path", default=f"{base}/phase4/faithfulness.jsonl")
    ap.add_argument("--ablation_path", default=f"{base}/ablation_no_prior_answer/faithfulness.jsonl")
    ap.add_argument("--out_dir", default=f"{base}/ablation_no_prior_answer")
    ap.add_argument("--fig_path", default="research_projects/multi_turn_cot_faithfulness/paper/figures/ablation_no_prior_answer.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)

    print(f"Loading baseline (prior answers visible): {args.baseline_path}")
    baseline = load_anchored_per_turn(args.baseline_path)
    print(f"  {len(baseline)} convs, {sum(len(v) for v in baseline.values())} labelled turns")

    print(f"Loading ablation (prior answers redacted): {args.ablation_path}")
    ablation = load_anchored_per_turn(args.ablation_path)
    print(f"  {len(ablation)} convs, {sum(len(v) for v in ablation.values())} labelled turns")

    common = sorted(set(baseline) & set(ablation))
    print(f"  {len(common)} common conversations")

    base_f = per_conv_fraction({t: baseline[t] for t in common})
    abl_f = per_conv_fraction({t: ablation[t] for t in common})

    print("\n=== Per-conversation anchored fraction ===")
    print(f"  {'task_id':<32} {'baseline':>9} {'ablated':>8} {'shift_pp':>9}")
    for tid in common:
        print(f"  {tid:<32} {base_f[tid]*100:>8.1f}% {abl_f[tid]*100:>7.1f}% "
              f"{(abl_f[tid]-base_f[tid])*100:>+8.1f}")

    pair_stats = {}
    if len(common) >= 2:
        x = np.array([base_f[t] for t in common])
        y = np.array([abl_f[t] for t in common])
        mean_base = float(np.mean(x))
        mean_abl = float(np.mean(y))
        mean_shift = mean_abl - mean_base
        try:
            stat, p = wilcoxon(y - x)
            wp = float(p)
        except ValueError:
            wp = float("nan")
        rng = np.random.default_rng(42)
        boots = [np.mean(rng.choice(y - x, size=len(x), replace=True))
                 for _ in range(10000)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        pair_stats = {
            "n_paired_convs": len(common),
            "mean_baseline": mean_base,
            "mean_ablated": mean_abl,
            "mean_shift": mean_shift,
            "mean_shift_ci95": [lo, hi],
            "wilcoxon_p": wp,
        }
        print(f"\n  Mean baseline anchored = {mean_base*100:.2f}%")
        print(f"  Mean ablated  anchored = {mean_abl*100:.2f}%")
        print(f"  Mean shift = {mean_shift*100:+.2f}pp (95% CI {lo*100:+.2f} to {hi*100:+.2f})")
        print(f"  Wilcoxon p = {wp:.4g}")

    agree = per_turn_agreement(baseline, ablation)
    print(f"\n  Per-turn agreement: {agree['n_agree']}/{agree['n_compared_turns']} "
          f"= {agree['agreement_rate']*100:.1f}%")
    print(f"  baseline-only anchored (lost after ablation): {agree['baseline_only_anchored']}")
    print(f"  ablation-only anchored (gained after ablation): {agree['ablated_only_anchored']}")

    out_json = os.path.join(args.out_dir, "ablation_analysis.json")
    with open(out_json, "w") as f:
        json.dump({
            "baseline_per_conv_fraction": base_f,
            "ablated_per_conv_fraction": abl_f,
            "paired_stats": pair_stats,
            "per_turn_agreement": agree,
        }, f, indent=2)
    print(f"\nwrote {out_json}")

    if base_f and abl_f:
        make_figure(base_f, abl_f, args.fig_path)
    print("DONE.")


if __name__ == "__main__":
    main()
