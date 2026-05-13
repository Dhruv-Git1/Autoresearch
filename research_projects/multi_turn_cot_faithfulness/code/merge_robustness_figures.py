"""
Merged robustness panel figure (F11 in the consolidated paper).

Produces a single 2-row × 2-column figure:
  Row 1 — Prior-answer ablation (anchoring persists without lexical context-copying)
  Row 2 — Quantization robustness (INT4 vs INT8, identical results)

Each row:  left = paired-line comparison,  right = scatter on y=x diagonal.

Run from repo root:
    PYTHONIOENCODING=utf-8 python research_projects/multi_turn_cot_faithfulness/code/merge_robustness_figures.py

Output: research_projects/multi_turn_cot_faithfulness/paper/figures/robustness_panel.png
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_bistability_analysis import is_anchored

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research_projects/multi_turn_cot_faithfulness/results"
OUT     = REPO / "research_projects/multi_turn_cot_faithfulness/paper/figures/robustness_panel.png"

BASELINE_PATH = RESULTS / "phase4/faithfulness.jsonl"
ABLATION_PATH = RESULTS / "ablation_no_prior_answer/faithfulness.jsonl"
INT4_PATH     = RESULTS / "quant_robustness_int4/faithfulness.jsonl"

# ── Color palette ─────────────────────────────────────────────────────────────
GREEN  = "#1a7a3a"
RED    = "#a31515"
GREY   = "#888888"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_anchored_per_turn(path: Path) -> Dict:
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
            tid  = r.get("task_id")
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


def paired_stats(x_vals, y_vals):
    """Return (mean_shift_pp, wilcoxon_p) for paired x/y lists."""
    shifts = [b - a for a, b in zip(x_vals, y_vals)]
    mean_shift = float(np.mean(shifts)) * 100
    if len(shifts) >= 2 and not all(s == 0 for s in shifts):
        try:
            _, p = wilcoxon(shifts)
        except Exception:
            p = float("nan")
    else:
        p = 1.0
    return mean_shift, p


# ── Plotting helper ───────────────────────────────────────────────────────────

def _draw_row(axes_left, axes_right,
              x_vals, y_vals,
              left_labels, right_xlabel, right_ylabel,
              row_title, mean_shift, wilcox_p):
    """Draw one row of the 2×2 figure."""
    n = len(x_vals)

    # Left: paired-line plot
    ax = axes_left
    for a, b in zip(x_vals, y_vals):
        ax.plot([0, 1], [a, b], "-", color=GREEN, alpha=0.6, lw=1.2)
        ax.scatter([0], [a], color=GREEN, s=45, zorder=3)
        ax.scatter([1], [b], color=RED,   s=45, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(left_labels, fontsize=9)
    ax.set_ylabel("Per-conv anchored fraction", fontsize=9)
    ax.set_title(f"Paired comparison (n={n} convs)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: scatter on y = x
    ax = axes_right
    ax.scatter(x_vals, y_vals, s=65, color=GREEN, alpha=0.75, zorder=3)
    all_vals = x_vals + y_vals
    lo, hi = min(all_vals) - 0.02, max(all_vals) + 0.05
    lo = max(lo, 0)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1.1, label="$y=x$ (no change)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(right_xlabel, fontsize=9)
    ax.set_ylabel(right_ylabel, fontsize=9)
    p_str = f"p={wilcox_p:.3f}" if not np.isnan(wilcox_p) else "p=n/a"
    ax.set_title(f"Mean shift {mean_shift:+.1f} pp  ({p_str})", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Row label on the left
    axes_left.set_ylabel(axes_left.get_ylabel(), fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading baseline (phase4) …")
    baseline = load_anchored_per_turn(BASELINE_PATH)

    print("Loading ablation (prior answers redacted) …")
    ablation = load_anchored_per_turn(ABLATION_PATH)

    print("Loading INT4 re-run …")
    int4     = load_anchored_per_turn(INT4_PATH)

    # --- Ablation row data ---
    abl_common = sorted(set(baseline) & set(ablation))
    base_f_abl = per_conv_fraction({t: baseline[t] for t in abl_common})
    abl_f      = per_conv_fraction({t: ablation[t] for t in abl_common})
    x_abl = [base_f_abl[t] for t in abl_common]
    y_abl = [abl_f[t]      for t in abl_common]
    shift_abl, p_abl = paired_stats(x_abl, y_abl)

    # --- Quant row data ---
    q_common = sorted(set(baseline) & set(int4))
    base_f_q = per_conv_fraction({t: baseline[t] for t in q_common})
    int4_f   = per_conv_fraction({t: int4[t]     for t in q_common})
    x_q = [base_f_q[t] for t in q_common]
    y_q = [int4_f[t]   for t in q_common]
    shift_q, p_q = paired_stats(x_q, y_q)

    print(f"Ablation:  n={len(abl_common)}  shift={shift_abl:+.1f} pp  p={p_abl:.3f}")
    print(f"Quant:     n={len(q_common)}  shift={shift_q:+.1f} pp  p={p_q:.3f}")

    # --- Build figure ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    # Row 1: ablation
    _draw_row(
        axes[0, 0], axes[0, 1],
        x_abl, y_abl,
        left_labels=["Baseline\n(prior answers visible)", "Ablated\n(prior answers redacted)"],
        right_xlabel="Baseline anchored fraction",
        right_ylabel="Ablated anchored fraction",
        row_title="Prior-answer ablation",
        mean_shift=shift_abl,
        wilcox_p=p_abl,
    )

    # Row 2: quant
    _draw_row(
        axes[1, 0], axes[1, 1],
        x_q, y_q,
        left_labels=["INT8\n(baseline)", "INT4\n(re-run)"],
        right_xlabel="INT8 anchored fraction",
        right_ylabel="INT4 anchored fraction",
        row_title="Quantization robustness",
        mean_shift=shift_q,
        wilcox_p=p_q,
    )

    # Row annotations on the far left
    for row_idx, label in enumerate([
        "Row 1 — Prior-answer ablation\n(anchoring persists without lexical copying)",
        "Row 2 — Quantization robustness\n(INT4 vs INT8: identical classifications)",
    ]):
        axes[row_idx, 0].set_title(
            f"Paired comparison (n={len(abl_common) if row_idx == 0 else len(q_common)} convs)",
            fontsize=9
        )

    fig.suptitle(
        "Robustness controls: anchoring survives two independent confound checks\n"
        "(paired Phase 4 conversations, n=6 each)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
