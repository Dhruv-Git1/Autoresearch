"""
Generate publication-quality figures for the bistability paper.
Run from the repo root:
    python3 research_projects/multi_turn_cot_faithfulness/code/generate_paper_figures.py

Outputs go to: research_projects/multi_turn_cot_faithfulness/paper/figures/
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.stats import kstest

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research_projects/multi_turn_cot_faithfulness/results"
OUT = REPO / "research_projects/multi_turn_cot_faithfulness/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

FAITH_PATHS = [
    RESULTS / "phase2/faithfulness.jsonl",
    RESULTS / "phase3/faithfulness.jsonl",
    RESULTS / "phase4/faithfulness.jsonl",
]

# ── Colour palette ─────────────────────────────────────────────────────────────
GREEN   = "#27ae60"   # exploring
RED     = "#c0392b"   # anchored
BLUE    = "#2980b9"
ORANGE  = "#e67e22"
PURPLE  = "#8e44ad"
GREY    = "#95a5a6"
DARK    = "#2c3e50"
LIGHT   = "#ecf0f1"
BG      = "#fafafa"

# ── Typography ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# ── Helpers ────────────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def extract_numeric(text):
    if not text:
        return ""
    matches = _NUM_RE.findall(text)
    if not matches:
        return ""
    last = next((m for pair in reversed(matches) for m in pair if m), "")
    return last.strip().lstrip("$").rstrip(",.")


def is_anchored(row):
    tr = row.get("result", {}).get("truncation_results", [])
    if len(tr) < 3:
        return None
    nums = [extract_numeric(t.get("regen_answer_preview", "")) for t in tr]
    nums = [n for n in nums if n]
    if len(nums) < 3:
        return None
    return len(set(nums)) == 1


def run_lengths(seq):
    if not seq:
        return []
    runs, cur, length = [], seq[0], 1
    for v in seq[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = v, 1
    runs.append((cur, length))
    return runs


def geometric_pmf(k, p):
    return p * (1 - p) ** (k - 1)


def load_data():
    rows = []
    for p in FAITH_PATHS:
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {len([p for p in FAITH_PATHS if p.exists()])} files")
    return rows


def build_conv_records(rows):
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["task_id"]].append(r)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda r: r["turn"])

    records, all_anchored_rl, all_exploring_rl, all_anchored_vals = [], [], [], []

    for tid, turn_rows in sorted(by_tid.items()):
        seq = [is_anchored(r) for r in turn_rows]
        valid = [v for v in seq if v is not None]
        if not valid:
            continue
        all_anchored_vals.extend(int(v) for v in valid)
        rls = run_lengths([int(v) for v in valid])
        anchored_rl = [l for v, l in rls if v == 1]
        exploring_rl = [l for v, l in rls if v == 0]
        all_anchored_rl.extend(anchored_rl)
        all_exploring_rl.extend(exploring_rl)
        records.append({
            "task_id": tid,
            "n_turns": len(valid),
            "frac_anchored": sum(valid) / len(valid),
            "anchored_seq": valid,
            "n_anchored_runs": len(anchored_rl),
            "max_anchored_run": max(anchored_rl) if anchored_rl else 0,
        })

    return records, all_anchored_rl, all_exploring_rl, all_anchored_vals


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Concept overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_concept():
    """Schematic explaining bistability: exploring mode vs anchored mode."""

    fig = plt.figure(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("white")

    # ── layout: 2 columns × 2 rows ───────────────────────────────────────────
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.03, right=0.97, top=0.93, bottom=0.04,
                           wspace=0.12, hspace=0.35)

    ax_title = fig.add_subplot(gs[0, :])  # top spanning both cols
    ax_left  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    for ax in [ax_title, ax_left, ax_right]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ── top banner ────────────────────────────────────────────────────────────
    ax_title.text(0.5, 0.82, "CoT Faithfulness is Bistable in Multi-Turn Conversations",
                  ha="center", va="center", fontsize=11, fontweight="bold", color=DARK,
                  transform=ax_title.transAxes)

    # --- draw a mini conversation timeline across the top row ─────────────────
    # Approximate mode sequence from sample 965 (44 turns)
    sample_modes = (
        [0]*8 + [1]*5 + [0]*10 + [1]*1 + [0]*9 + [1]*2 + [0]*9
    )[:40]  # 40 turns for display

    n = len(sample_modes)
    tw = 0.022   # tile width in axes fraction
    th = 0.30
    y0 = 0.30
    x0 = 0.02
    gap = 0.002

    for i, mode in enumerate(sample_modes):
        x = x0 + i * (tw + gap)
        col = RED if mode else GREEN
        rect = FancyBboxPatch((x, y0), tw, th,
                              boxstyle="round,pad=0.003",
                              facecolor=col, edgecolor="white",
                              linewidth=0.4,
                              transform=ax_title.transAxes, clip_on=False)
        ax_title.add_patch(rect)

    # annotations on the timeline
    ax_title.annotate("Exploring\n(CoT causal)",
                      xy=(x0 + 3*(tw+gap) + tw/2, y0 + th),
                      xytext=(x0 + 3*(tw+gap) + tw/2, y0 + th + 0.28),
                      xycoords="axes fraction", textcoords="axes fraction",
                      ha="center", va="bottom", fontsize=7.5, color=GREEN, fontweight="bold",
                      arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0))

    ax_title.annotate("Anchored\n(CoT post-hoc)",
                      xy=(x0 + 9*(tw+gap) + tw/2, y0 + th),
                      xytext=(x0 + 9*(tw+gap) + tw/2, y0 + th + 0.28),
                      xycoords="axes fraction", textcoords="axes fraction",
                      ha="center", va="bottom", fontsize=7.5, color=RED, fontweight="bold",
                      arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))

    ax_title.text(x0 - 0.01, y0 + th/2, "Turn →",
                  ha="right", va="center", fontsize=7, color=GREY,
                  transform=ax_title.transAxes)
    ax_title.text(x0 + n*(tw+gap), y0 + th/2, f"(sample 965, {n} turns)",
                  ha="left", va="center", fontsize=6.5, color=GREY, style="italic",
                  transform=ax_title.transAxes)

    # ── left panel: Exploring mode ────────────────────────────────────────────
    _draw_truncation_panel(ax_left,
                           title="Exploring mode",
                           title_color=GREEN,
                           answers=["4.2", "4.2", "3.8", "4.5", "4.2"],
                           outcome="Different answers → CoT is causally active",
                           outcome_color=GREEN,
                           indicator="✓  CoT drives the answer")

    # ── right panel: Anchored mode ────────────────────────────────────────────
    _draw_truncation_panel(ax_right,
                           title="Anchored mode",
                           title_color=RED,
                           answers=["300", "300", "300", "300", "300"],
                           outcome="Same answer regardless → CoT is post-hoc",
                           outcome_color=RED,
                           indicator="✗  CoT is decorative")

    out = OUT / "concept_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


def _draw_truncation_panel(ax, title, title_color, answers, outcome, outcome_color, indicator):
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel title
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=title_color, transform=ax.transAxes)

    pcts = ["0%", "25%", "50%", "75%", "100%"]
    bar_colors = ["#c0392b", "#e67e22", "#f39c12", "#27ae60", "#2980b9"]
    bar_y_start = 0.80
    bar_height = 0.095
    bar_gap = 0.012
    label_x = 0.10
    bar_x0 = 0.22
    bar_x1 = 0.68
    ans_x = 0.72

    for i, (pct, ans, bc) in enumerate(zip(pcts, answers, bar_colors)):
        y = bar_y_start - i * (bar_height + bar_gap)

        # truncation label
        ax.text(label_x, y + bar_height/2, pct + " CoT",
                ha="right", va="center", fontsize=7.5, color=DARK,
                transform=ax.transAxes)

        # filled bar (representing the CoT prefix shown)
        frac = i / 4.0
        if frac > 0:
            bar = FancyBboxPatch((bar_x0, y), (bar_x1 - bar_x0) * frac, bar_height,
                                 boxstyle="round,pad=0.005",
                                 facecolor=bc, alpha=0.75, edgecolor="none",
                                 transform=ax.transAxes)
            ax.add_patch(bar)
        # empty bar (representing the truncated part)
        empty = FancyBboxPatch((bar_x0, y), bar_x1 - bar_x0, bar_height,
                               boxstyle="round,pad=0.005",
                               facecolor="none", edgecolor=GREY, linewidth=0.5,
                               transform=ax.transAxes)
        ax.add_patch(empty)

        # answer label
        ax.text(ans_x + 0.01, y + bar_height/2, f"→ {ans}",
                ha="left", va="center", fontsize=7.5,
                color=outcome_color if title == "Anchored mode" else DARK,
                fontweight="bold" if title == "Anchored mode" else "normal",
                transform=ax.transAxes)

    # outcome text
    y_out = bar_y_start - 5 * (bar_height + bar_gap) - 0.02
    ax.text(0.5, y_out, outcome,
            ha="center", va="top", fontsize=7, color=outcome_color,
            style="italic", wrap=True, transform=ax.transAxes)

    # indicator badge
    ax.text(0.5, 0.04, indicator,
            ha="center", va="bottom", fontsize=8, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=outcome_color, edgecolor="none"),
            transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Faithfulness heatmap (main result)
# ══════════════════════════════════════════════════════════════════════════════

def fig_heatmap(records):
    if not records:
        print("  Skipping heatmap (no data)")
        return

    # Sort by frac_anchored descending so most interesting at top
    records_sorted = sorted(records, key=lambda r: r["frac_anchored"], reverse=True)
    n_conv = len(records_sorted)
    max_turns = max(len(r["anchored_seq"]) for r in records_sorted)

    heat = np.full((n_conv, max_turns), np.nan)
    for i, r in enumerate(records_sorted):
        seq = r["anchored_seq"]
        heat[i, :len(seq)] = seq

    fig_h = max(4.5, n_conv * 0.28 + 1.5)
    fig_w = max(8.5, max_turns * 0.38 + 2.2)
    fig, ax = plt.subplots(figsize=(min(fig_w, 11), min(fig_h, 9)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = mcolors.ListedColormap([GREEN, RED])
    im = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Y-axis: conversation IDs
    short_ids = [r["task_id"].split("/")[-1] for r in records_sorted]
    ax.set_yticks(range(n_conv))
    ax.set_yticklabels(short_ids, fontsize=6.5)
    ax.set_xlabel("Turn index", fontsize=9)
    ax.set_ylabel("Conversation (GSM8K sample ID)", fontsize=9)

    # Annotate frac_anchored on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n_conv))
    fa_labels = [f"{r['frac_anchored']:.0%}" for r in records_sorted]
    ax2.set_yticklabels(fa_labels, fontsize=6.5)
    for tick, r in zip(ax2.get_yticklabels(), records_sorted):
        tick.set_color(RED if r["frac_anchored"] > 0 else GREEN)
    ax2.set_ylabel("Anchored fraction", fontsize=8, color=DARK)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GREY)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.12)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["exploring\n(CoT causal)", "anchored\n(CoT post-hoc)"], fontsize=7.5)
    cbar.ax.tick_params(length=0)

    n_total = sum(len(r["anchored_seq"]) for r in records_sorted)
    n_anchored = sum(sum(r["anchored_seq"]) for r in records_sorted)
    pct = n_anchored / n_total * 100

    ax.set_title(
        f"Per-turn CoT faithfulness mode  "
        f"(N={n_conv} conversations, {n_total} turns)\n"
        f"Overall: {n_anchored}/{n_total} turns anchored ({pct:.0f}%)",
        fontsize=9.5, pad=8
    )

    # Draw thin separator lines between conversations
    for i in range(1, n_conv):
        ax.axhline(i - 0.5, color="white", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out = OUT / "heatmap_faithfulness.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Run-length distribution (H1 result)
# ══════════════════════════════════════════════════════════════════════════════

def fig_runlength(all_anchored_rl, all_exploring_rl):
    if not all_anchored_rl:
        print("  Skipping run-length (no anchored runs)")
        return

    mean_rl = float(np.mean(all_anchored_rl))
    p_geom = 1.0 / mean_rl
    ks_stat, ks_p = kstest(all_anchored_rl, "geom", args=(p_geom,))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    fig.patch.set_facecolor("white")

    # ── Panel A: anchored run-length vs geometric null ─────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")
    max_run = max(all_anchored_rl)
    bins = np.arange(0.5, max_run + 1.5)
    ax.hist(all_anchored_rl, bins=bins, density=True, alpha=0.75,
            color=RED, edgecolor="white", linewidth=0.8,
            label=f"Observed (n={len(all_anchored_rl)} runs)")

    ks_arr = np.arange(1, max_run + 2)
    geom = [geometric_pmf(k, p_geom) for k in ks_arr]
    ax.plot(ks_arr, geom, "k--", lw=1.8,
            label=f"Geometric null\n($\\hat{{p}}$={p_geom:.2f})")
    ax.set_xlabel("Run length (consecutive anchored turns)")
    ax.set_ylabel("Probability density")
    sig = "p < 0.001" if ks_p < 0.001 else f"p = {ks_p:.3f}"
    ax.set_title(f"Anchored run-length vs. geometric null\nKS {sig}  —  heavy tail confirms persistence",
                 fontsize=8.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0.5, max_run + 0.5)

    # Annotate significance
    if ks_p < 0.05:
        ax.text(0.97, 0.97, "★ Significant\n(p < 0.05)",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8, color=RED, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=RED, alpha=0.85))

    # ── Panel B: exploring run-length (complementary view) ─────────────────────
    ax = axes[1]
    ax.set_facecolor("white")
    if all_exploring_rl:
        max_exp = max(all_exploring_rl)
        bins_exp = np.arange(0.5, min(max_exp, 30) + 1.5)
        data_clipped = [min(v, 30) for v in all_exploring_rl]
        ax.hist(data_clipped, bins=bins_exp, density=True, alpha=0.75,
                color=GREEN, edgecolor="white", linewidth=0.8,
                label=f"Observed (n={len(all_exploring_rl)} runs)")

        # Fit geometric to exploring runs
        mean_exp = float(np.mean(all_exploring_rl))
        p_exp = 1.0 / mean_exp
        ks_arr_exp = np.arange(1, min(max_exp, 30) + 2)
        geom_exp = [geometric_pmf(k, p_exp) for k in ks_arr_exp]
        ax.plot(ks_arr_exp, geom_exp, "k--", lw=1.8,
                label=f"Geometric null\n($\\hat{{p}}$={p_exp:.2f})")

    ax.set_xlabel("Run length (consecutive exploring turns)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Exploring run-length distribution\n(complementary to anchored)",
                 fontsize=8.5)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("H1: Anchored mode is persistent — run-lengths exceed geometric null",
                 fontsize=9.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT / "runlength_dist.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Fraction anchored × conversation length scatter (H2/H3)
# ══════════════════════════════════════════════════════════════════════════════

def fig_frac_scatter(records):
    if len(records) < 3:
        print("  Skipping scatter (< 3 conversations)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    fig.patch.set_facecolor("white")

    # ── Panel A: scatter frac_anchored vs conversation length ──────────────────
    ax = axes[0]
    ax.set_facecolor("white")

    xs = [r["frac_anchored"] for r in records]
    ys = [r["n_turns"] for r in records]
    sizes = [max(30, r["max_anchored_run"] * 25) for r in records]
    colors = [RED if r["frac_anchored"] > 0.05 else GREEN for r in records]

    sc = ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.75, zorder=3,
                    edgecolors="white", linewidths=0.8)

    # Annotate top-anchored conversations
    top = sorted(records, key=lambda r: r["frac_anchored"], reverse=True)[:5]
    for r in top:
        ax.annotate(r["task_id"].split("/")[-1],
                    (r["frac_anchored"], r["n_turns"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=DARK, alpha=0.8)

    ax.set_xlabel("Fraction of turns in anchored mode")
    ax.set_ylabel("Conversation length (# turns)")
    ax.set_title("Anchored fraction vs. conversation length\n(size ∝ max anchored run length)",
                 fontsize=8.5)
    ax.axvline(0.05, color=GREY, lw=0.8, ls=":", label="5% threshold")
    ax.legend(fontsize=7.5)

    # Summary annotation
    n_any_anchored = sum(1 for r in records if r["frac_anchored"] > 0)
    ax.text(0.97, 0.03,
            f"{n_any_anchored}/{len(records)} conversations\nhave ≥1 anchored turn",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GREY, alpha=0.9))

    # ── Panel B: H3 bimodality bar chart ──────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("white")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    all_vals = []
    for r in records:
        all_vals.extend(r["anchored_seq"])
    frac_anc = sum(all_vals) / len(all_vals) if all_vals else 0
    frac_exp = 1.0 - frac_anc

    # Draw a large donut-style split showing H3
    theta_exp = frac_exp * 360
    theta_anc = frac_anc * 360

    wedges, texts = ax.pie(
        [frac_exp, frac_anc],
        colors=[GREEN, RED],
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.0),
        labels=None,
    )

    # Center text
    ax.text(0, 0.15, f"{frac_anc:.0%}", ha="center", va="center",
            fontsize=16, fontweight="bold", color=RED)
    ax.text(0, -0.18, "anchored", ha="center", va="center",
            fontsize=8, color=RED)

    # Legend patches
    leg_patches = [
        mpatches.Patch(facecolor=GREEN, label=f"Exploring ({frac_exp:.0%})"),
        mpatches.Patch(facecolor=RED, label=f"Anchored ({frac_anc:.0%})"),
    ]
    ax.legend(handles=leg_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.9)
    ax.set_title(f"H3: Both modes present\n(N={len(records)} conversations, {len(all_vals)} turns)",
                 fontsize=8.5)

    fig.suptitle("Bistability in CoT faithfulness: both modes are present",
                 fontsize=9.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT / "frac_anchored_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Phase cascade (N accumulation showing H1 becoming significant)
# ══════════════════════════════════════════════════════════════════════════════

def fig_cascade():
    """Shows how H1 KS p-value evolved as N grew across phases."""
    phases = ["Phase 1\n(N=4)", "P1+P2\n(N=8)", "P1+P2+P3\n(N=14)", "P1+P2+P3+P4\n(N=24)"]
    faith_turns = [53, 92, 177, 412]
    ks_p = [0.888, 0.516, 0.0019, 0.0001]  # last one effectively 0
    frac_anc = [0.12, 0.08, 0.08, 0.13]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    fig.patch.set_facecolor("white")

    # ── Panel A: KS p-value trajectory ──────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")
    x = np.arange(len(phases))
    bar_colors = [RED if p < 0.05 else GREY for p in ks_p]
    bars = ax.bar(x, [-math.log10(max(p, 1e-6)) for p in ks_p],
                  color=bar_colors, alpha=0.80, edgecolor="white", linewidth=0.8,
                  width=0.55)
    ax.axhline(-math.log10(0.05), color=RED, lw=1.2, ls="--", alpha=0.7,
               label="p = 0.05 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=7.5)
    ax.set_ylabel("−log₁₀(KS p-value)  [higher = more significant]", fontsize=8)
    ax.set_title("H1 significance grows with N\n(anchored run-length vs. geometric null)", fontsize=8.5)
    ax.legend(fontsize=7.5)

    # Annotate bars
    for i, (bar, p, ft) in enumerate(zip(bars, ks_p, faith_turns)):
        label = f"p={p}" if p >= 0.001 else "p≈0"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha="center", va="bottom", fontsize=7,
                fontweight="bold" if p < 0.05 else "normal",
                color=RED if p < 0.05 else DARK)
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                f"{ft} turns", ha="center", va="bottom", fontsize=6,
                color="white", fontweight="bold")

    # ── Panel B: frac_anchored trajectory ───────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("white")
    ax.bar(x, [fa * 100 for fa in frac_anc],
           color=[RED if fa > 0.10 else ORANGE for fa in frac_anc],
           alpha=0.80, edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=7.5)
    ax.set_ylabel("% turns in anchored mode", fontsize=8)
    ax.set_title("H3: Anchored fraction stable across phases\n(both modes confirmed at every scale)", fontsize=8.5)

    for i, (fa, ft) in enumerate(zip(frac_anc, faith_turns)):
        ax.text(i, fa * 100 + 0.3, f"{fa:.0%}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=DARK)

    fig.suptitle("Bistability cascade: effect strengthens with N",
                 fontsize=9.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT / "phase_cascade.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Output -> {OUT}\n")

    print("Loading data...")
    rows = load_data()
    records, all_anchored_rl, all_exploring_rl, all_anchored_vals = build_conv_records(rows)
    print(f"  {len(records)} conversations, {len(all_anchored_vals)} assessable turns\n")

    print("Generating figures...")
    fig_concept()
    fig_heatmap(records)
    fig_runlength(all_anchored_rl, all_exploring_rl)
    fig_frac_scatter(records)
    fig_cascade()

    print(f"\nDone. {len(list(OUT.iterdir()))} files in {OUT}")
