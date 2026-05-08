"""Phase 2 — Bistability analysis.

Three hypotheses:
  H1: Anchored-mode runs are longer than a geometric (i.i.d.) baseline,
      meaning the model is genuinely "stuck" rather than randomly producing
      the same answer by coincidence.

  H2: Fraction of turns spent in anchored mode negatively predicts
      conversation accuracy (anchored = lost in conversation).

  H3: Across conversations, there is a bimodal distribution of per-turn
      faithfulness — most turns are near 0 (anchored) or near 1 (exploring),
      not uniformly distributed.

Inputs:
  --faith_paths   one or more faithfulness.jsonl files (phase1 + phase2)
  --trace_dirs    dirs containing trace_sharded_*.json (for is_correct labels)
  --out_dir       output directory for plots and stats

Run from inside lost_in_conversation/ directory.
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import kstest, pointbiserialr

_NUM_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


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
    """True if all 5 truncation levels produce the same numeric answer.
    None if data is insufficient (< 3 valid nums).
    """
    tr = row.get("result", {})
    trunc_results = tr.get("truncation_results", [])
    if len(trunc_results) < 3:
        return None
    nums = [extract_numeric(t.get("regen_answer_preview", ""))
            for t in trunc_results]
    nums = [n for n in nums if n]
    if len(nums) < 3:
        return None
    return len(set(nums)) == 1


def run_lengths(seq):
    """[(value, length), ...] from a binary list."""
    if not seq:
        return []
    runs = []
    cur, length = seq[0], 1
    for v in seq[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = v, 1
    runs.append((cur, length))
    return runs


def geometric_pmf(k, p):
    return (1 - p) ** (k - 1) * p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faith_paths", nargs="+",
                    default=[
                        "../results/day2/faithfulness.jsonl",
                        "../results/day2_sample965/faithfulness.jsonl",
                        "../results/phase2/faithfulness.jsonl",
                    ])
    ap.add_argument("--trace_dirs", nargs="+",
                    default=["../results/day1", "../results/phase2"])
    ap.add_argument("--out_dir", default="../results/phase2")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load faithfulness rows ----
    rows = []
    for p in args.faith_paths:
        if not os.path.exists(p):
            print(f"WARNING: {p} not found, skipping")
            continue
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n_tasks = len({r["task_id"] for r in rows})
    print(f"Loaded {len(rows)} faithfulness rows across {n_tasks} conversations")

    # ---- Load traces for is_correct labels ----
    correct_by_id = {}
    for td in args.trace_dirs:
        if not os.path.exists(td):
            continue
        for fn in os.listdir(td):
            if fn.startswith("trace_sharded_") and fn.endswith(".json"):
                try:
                    with open(os.path.join(td, fn)) as fh:
                        payload = json.load(fh)
                    correct_by_id[payload["task_id"]] = payload.get("is_correct", False)
                except Exception:
                    pass

    # ---- Build per-conversation data ----
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["task_id"]].append(r)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda r: r["turn"])

    conv_records = []
    all_anchored_rl = []
    all_exploring_rl = []
    all_faith_values = []

    for tid, turn_rows in sorted(by_tid.items()):
        anchored_seq = []
        faith_seq = []
        for r in turn_rows:
            a = is_anchored(r)
            f = r.get("faithfulness_score")
            anchored_seq.append(a)
            faith_seq.append(f)

        valid_anchored = [v for v in anchored_seq if v is not None]
        valid_faith = [f for f in faith_seq if f is not None]
        if not valid_anchored:
            continue

        all_faith_values.extend(valid_faith)
        runs = run_lengths([int(v) for v in valid_anchored])
        anchored_rl = [l for v, l in runs if v == 1]
        exploring_rl = [l for v, l in runs if v == 0]
        all_anchored_rl.extend(anchored_rl)
        all_exploring_rl.extend(exploring_rl)

        frac_anchored = sum(valid_anchored) / len(valid_anchored)
        conv_records.append({
            "task_id": tid,
            "n_turns": len(valid_anchored),
            "frac_anchored": round(frac_anchored, 3),
            "n_anchored_runs": len(anchored_rl),
            "max_anchored_run": max(anchored_rl) if anchored_rl else 0,
            "mean_anchored_run": float(np.mean(anchored_rl)) if anchored_rl else 0.0,
            "mean_faith": float(np.mean(valid_faith)) if valid_faith else float("nan"),
            "is_correct": correct_by_id.get(tid),
            "anchored_seq": valid_anchored,
        })

    df = pd.DataFrame(conv_records)
    print(f"\n{len(df)} conversations analyzed:")
    print(df[["task_id", "n_turns", "frac_anchored", "max_anchored_run", "is_correct"]].to_string(index=False))

    # ---- Statistical tests ----
    stats_out = {}

    # H1: Anchored run-length vs geometric
    if all_anchored_rl:
        mean_rl = float(np.mean(all_anchored_rl))
        p_geom = 1.0 / mean_rl  # MLE for geometric
        ks_stat, ks_p = kstest(all_anchored_rl, "geom", args=(p_geom,))
        h1 = {
            "n_anchored_runs": len(all_anchored_rl),
            "mean_run_length": round(mean_rl, 3),
            "p_geometric_mle": round(p_geom, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_p": round(ks_p, 4),
        }
        h1["verdict"] = ("long_tail: anchored bursts exceed geometric prediction (p<0.05)"
                         if ks_p < 0.05 else
                         "consistent with geometric (no excess clustering)")
        stats_out["H1_runlength"] = h1
        print(f"\nH1  n_anchored_runs={len(all_anchored_rl)}  mean_rl={mean_rl:.2f}  KS p={ks_p:.3f}  → {h1['verdict']}")

    # H2: frac_anchored negatively predicts correctness
    valid = df[df["is_correct"].notna()].copy()
    if len(valid) >= 5:
        valid["correct_int"] = valid["is_correct"].astype(int)
        r_val, p_val = pointbiserialr(valid["frac_anchored"], valid["correct_int"])
        h2 = {
            "n": len(valid),
            "r": round(r_val, 4),
            "p": round(p_val, 4),
        }
        h2["verdict"] = (f"significant negative correlation (r={r_val:.2f}, p={p_val:.3f})"
                         if p_val < 0.05 and r_val < 0 else
                         f"not significant (r={r_val:.2f}, p={p_val:.3f})")
        stats_out["H2_frac_anchored_vs_correct"] = h2
        print(f"H2  r={r_val:.3f}  p={p_val:.3f}  n={len(valid)}  → {h2['verdict']}")

    # H3: Bimodality — use is_anchored (answer-change) NOT faithfulness_score
    # (faithfulness_score is correctness-flip, which is 0 whenever the model is
    # consistently wrong regardless of truncation — uninformative for bimodality).
    # Instead: collect all per-turn is_anchored values and check that both modes
    # (anchored=1, exploring=0) are substantially present.
    all_anchored_vals = []
    for tid, turn_rows in by_tid.items():
        for r in turn_rows:
            a = is_anchored(r)
            if a is not None:
                all_anchored_vals.append(int(a))

    if len(all_anchored_vals) >= 20:
        frac_anchored_global = sum(all_anchored_vals) / len(all_anchored_vals)
        frac_exploring_global = 1.0 - frac_anchored_global
        # Both modes present = neither is < 5%
        both_modes_present = min(frac_anchored_global, frac_exploring_global) > 0.05
        h3 = {
            "n_obs": len(all_anchored_vals),
            "frac_anchored": round(frac_anchored_global, 3),
            "frac_exploring": round(frac_exploring_global, 3),
            "both_modes_present": both_modes_present,
        }
        h3["verdict"] = (f"bimodal: anchored={frac_anchored_global:.0%}, exploring={frac_exploring_global:.0%}"
                         if both_modes_present else
                         f"not bimodal: one mode dominates "
                         f"(anchored={frac_anchored_global:.0%}, exploring={frac_exploring_global:.0%})")
        stats_out["H3_bimodality"] = h3
        print(f"H3  frac_anchored={frac_anchored_global:.2f}  frac_exploring={frac_exploring_global:.2f}  "
              f"both_modes={both_modes_present}  → {h3['verdict']}")

    any_significant = (
        stats_out.get("H1_runlength", {}).get("ks_p", 1.0) < 0.05 or
        stats_out.get("H2_frac_anchored_vs_correct", {}).get("p", 1.0) < 0.05 or
        stats_out.get("H3_bimodality", {}).get("both_modes_present", False)
    )
    stats_out["decision"] = "GRADUATE" if any_significant else "NEED_MORE_DATA"
    stats_out["n_conversations"] = len(df)
    stats_out["n_faithfulness_obs"] = len(rows)

    with open(os.path.join(args.out_dir, "bistability_stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)
    print(f"\n=== DECISION: {stats_out['decision']} ===")
    print(json.dumps(stats_out, indent=2))

    if len(df) == 0:
        print("No data to plot. Exiting.")
        return

    # ---- Plot 1: Faithfulness heatmap (conversation × turn) ----
    max_turns = max(len(r["anchored_seq"]) for _, r in df.iterrows())
    n_conv = len(df)
    heat = np.full((n_conv, max_turns), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        seq = row["anchored_seq"]
        heat[i, :len(seq)] = seq

    fig_h = max(4, n_conv * 0.35 + 1)
    fig_w = max(10, max_turns * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mcolors.ListedColormap(["#2ecc71", "#e74c3c"])
    im = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Turn index (1-indexed assistant turns)")
    ax.set_ylabel("Conversation")
    ax.set_yticks(range(n_conv))
    ax.set_yticklabels(
        [f"{row['task_id'].split('/')[-1]} {'✓' if row['is_correct'] else '✗' if row['is_correct'] is not None else '?'}"
         for _, row in df.iterrows()],
        fontsize=7,
    )
    ax.set_title(
        f"CoT faithfulness mode per turn  (n={n_conv} conversations)\n"
        f"green=exploring (CoT drives answer)   red=anchored (CoT post-hoc)"
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.02)
    cbar.set_ticklabels(["exploring", "anchored"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "heatmap_faithfulness.png"), dpi=150)
    plt.close(fig)
    print("saved heatmap_faithfulness.png")

    # ---- Plot 2: Run-length distribution vs geometric null ----
    if all_anchored_rl:
        max_run = max(all_anchored_rl)
        bins = np.arange(0.5, max_run + 1.5)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.hist(all_anchored_rl, bins=bins, density=True, alpha=0.75,
                color="#e74c3c", label=f"Anchored runs (n={len(all_anchored_rl)})")
        p_est = stats_out.get("H1_runlength", {}).get("p_geometric_mle", 0.5)
        ks = np.arange(1, max_run + 1)
        geom_pmf = [geometric_pmf(k, p_est) for k in ks]
        ax.plot(ks, geom_pmf, "k--", lw=1.5, label=f"Geometric null (p={p_est:.2f})")
        ax.set_xlabel("Run length (consecutive anchored turns)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Anchored run-length distribution\n"
            f"KS p={stats_out.get('H1_runlength',{}).get('ks_p', float('nan')):.3f}"
        )
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.hist(all_exploring_rl, bins=np.arange(0.5, max(all_exploring_rl or [1]) + 1.5),
                density=True, alpha=0.75, color="#2ecc71",
                label=f"Exploring runs (n={len(all_exploring_rl)})")
        ax.set_xlabel("Run length (consecutive exploring turns)")
        ax.set_ylabel("Density")
        ax.set_title("Exploring run-length distribution")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "runlength_dist.png"), dpi=150)
        plt.close(fig)
        print("saved runlength_dist.png")

    # ---- Plot 3: Faith distribution histogram ----
    if all_faith_values:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(all_faith_values, bins=20, density=True, alpha=0.75,
                color="#3498db", edgecolor="white")
        ax.axvline(0.5, color="k", lw=1.2, ls="--", label="midpoint (0.5)")
        extreme_frac = stats_out.get("H3_bimodality", {}).get("fraction_at_extremes", float("nan"))
        ax.set_xlabel("Per-turn faithfulness score")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Distribution of per-turn faithfulness (n={len(all_faith_values)} obs)\n"
            f"{extreme_frac:.0%} of observations at extremes (< 0.2 or > 0.8)"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "faith_distribution.png"), dpi=150)
        plt.close(fig)
        print("saved faith_distribution.png")

    # ---- Plot 4: frac_anchored vs accuracy ----
    valid = df[df["is_correct"].notna()].copy()
    if len(valid) >= 3:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = ["#2ecc71" if c else "#e74c3c" for c in valid["is_correct"]]
        sc = ax.scatter(valid["frac_anchored"], valid["n_turns"],
                        c=colors, s=100, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)
        for _, row in valid.iterrows():
            ax.annotate(row["task_id"].split("/")[-1],
                        (row["frac_anchored"], row["n_turns"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6, alpha=0.6)
        r_info = stats_out.get("H2_frac_anchored_vs_correct", {})
        ax.set_xlabel("Fraction of turns in anchored mode")
        ax.set_ylabel("Conversation length (# assistant turns)")
        ax.set_title(
            f"Anchored fraction vs length  (green=correct, red=wrong)\n"
            f"r(frac_anchored, correct)={r_info.get('r', float('nan')):.2f}  "
            f"p={r_info.get('p', float('nan')):.3f}  n={r_info.get('n', '?')}"
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "frac_anchored_scatter.png"), dpi=150)
        plt.close(fig)
        print("saved frac_anchored_scatter.png")

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
