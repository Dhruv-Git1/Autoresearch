"""Exp 1 analysis: compare answer-token logprobs between anchored and exploring turns.

Input: answer_logprobs.jsonl (one row per (task_id, turn) with logprob_results
       containing per-pct results) plus faithfulness.jsonl files for label lookup.

Computes three comparisons across anchored vs exploring turns:
  - logprob_100pct  (full CoT)
  - logprob_0pct    (no CoT)
  - delta_logprob   = logprob_100pct - logprob_0pct

Reports mean, median, IQR, Mann-Whitney U test, Cohen's d for each.

Outputs:
  results/answer_logprobs/logprob_analysis.json
  paper/figures/logprob_distributions.png
"""
import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_bistability_analysis import is_anchored


def load_faith_labels(faith_paths: List[str]) -> Dict:
    """Build dict (task_id, turn) -> True/False/None anchored label."""
    labels = {}
    for p in faith_paths:
        if not os.path.exists(p):
            print(f"  WARN: missing {p}")
            continue
        with open(p) as f:
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
                labels[(tid, int(turn))] = a
    return labels


def load_logprob_rows(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def extract_per_turn_logprobs(logprob_rows: List[Dict[str, Any]]) -> List[Dict]:
    """Pivot to one row per turn with logprob and margin at 0%/100%.

    Margin = log P(chosen) - log P(top alternative) — much more sensitive
    than raw chosen-token logprob, which saturates at 0 under greedy decoding.
    """
    out = []
    for r in logprob_rows:
        res = r.get("result", {})
        if "error" in res or "skip_reason" in res:
            continue
        results = res.get("logprob_results", [])
        by_pct = {round(x.get("pct", -1), 4): x for x in results}
        z = by_pct.get(0.0)
        h = by_pct.get(1.0)
        if z is None or h is None:
            continue
        lp0 = z.get("last_digit_logprob")
        lp1 = h.get("last_digit_logprob")
        alt0 = z.get("top1_alt_logprob")
        alt1 = h.get("top1_alt_logprob")
        if lp0 is None or lp1 is None or alt0 is None or alt1 is None:
            continue
        m0 = float(lp0) - float(alt0)
        m1 = float(lp1) - float(alt1)
        out.append({
            "task_id": r.get("task_id"),
            "turn": int(r.get("turn", -1)),
            "logprob_0pct": float(lp0),
            "logprob_100pct": float(lp1),
            "delta_logprob": float(lp1) - float(lp0),
            "margin_0pct": m0,
            "margin_100pct": m1,
            "delta_margin": m1 - m0,
        })
    return out


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def summarize(name, anchored_vals, exploring_vals):
    a = np.asarray(anchored_vals)
    e = np.asarray(exploring_vals)
    try:
        u, p = mannwhitneyu(a, e, alternative="two-sided")
    except Exception:
        u, p = float("nan"), float("nan")
    return {
        "name": name,
        "n_anchored": int(len(a)),
        "n_exploring": int(len(e)),
        "anchored_mean": float(np.mean(a)) if len(a) else float("nan"),
        "anchored_median": float(np.median(a)) if len(a) else float("nan"),
        "anchored_iqr": [float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))] if len(a) else [float("nan")] * 2,
        "exploring_mean": float(np.mean(e)) if len(e) else float("nan"),
        "exploring_median": float(np.median(e)) if len(e) else float("nan"),
        "exploring_iqr": [float(np.quantile(e, 0.25)), float(np.quantile(e, 0.75))] if len(e) else [float("nan")] * 2,
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "cohens_d": cohen_d(a, e),
    }


def make_figure(per_turn_with_label, out_path):
    # Use margin (chosen logprob - top alternative logprob) — much more
    # discriminating than raw logprob under greedy decoding, where chosen
    # tokens are usually near 100% probability.
    a_m1 = [r["margin_100pct"] for r in per_turn_with_label if r["anchored"]]
    e_m1 = [r["margin_100pct"] for r in per_turn_with_label if not r["anchored"]]
    a_m0 = [r["margin_0pct"] for r in per_turn_with_label if r["anchored"]]
    e_m0 = [r["margin_0pct"] for r in per_turn_with_label if not r["anchored"]]
    a_d = [r["delta_margin"] for r in per_turn_with_label if r["anchored"]]
    e_d = [r["delta_margin"] for r in per_turn_with_label if not r["anchored"]]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    panels = [
        ("Margin at 0% CoT\n(no reasoning visible)", a_m0, e_m0),
        ("Margin at 100% CoT\n(full reasoning visible)", a_m1, e_m1),
        (r"$\Delta$ Margin (100% - 0%)" + "\n(reasoning impact on commitment)", a_d, e_d),
    ]
    colors = {"anchored": "#a31515", "exploring": "#1a7a3a"}
    for ax, (title, a, e) in zip(axes, panels):
        # Skip violinplot for any empty group; fall back to scatter
        valid = [(arr, pos, label) for arr, pos, label in
                 [(a, 1, "Anchored"), (e, 2, "Exploring")] if len(arr) > 0]
        if all(len(arr) >= 3 for arr, _, _ in valid):
            parts = ax.violinplot([arr for arr, _, _ in valid],
                                  positions=[pos for _, pos, _ in valid],
                                  widths=0.7, showmeans=True, showmedians=True)
            for i, body in enumerate(parts["bodies"]):
                col = colors["anchored"] if valid[i][2] == "Anchored" else colors["exploring"]
                body.set_facecolor(col)
                body.set_alpha(0.55)
        else:
            # Scatter fallback for tiny samples
            for arr, pos, label in valid:
                col = colors["anchored"] if label == "Anchored" else colors["exploring"]
                ax.scatter([pos] * len(arr), arr, color=col, alpha=0.7, s=40)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Anchored\n(n={len(a)})", f"Exploring\n(n={len(e)})"])
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("Logprob (nats)")
    fig.suptitle("Answer-token logprob distributions: anchored vs exploring",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprobs_path", required=True)
    ap.add_argument("--faith_paths", nargs="+", required=True)
    ap.add_argument("--out_dir", default="research_projects/multi_turn_cot_faithfulness/results/answer_logprobs")
    ap.add_argument("--fig_path", default="research_projects/multi_turn_cot_faithfulness/paper/figures/logprob_distributions.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)

    print("Loading anchoring labels...")
    labels = load_faith_labels(args.faith_paths)
    print(f"  loaded {len(labels)} (task_id, turn) labels")

    print("Loading logprob rows...")
    lp_rows = load_logprob_rows(args.logprobs_path)
    print(f"  loaded {len(lp_rows)} logprob rows")

    print("Pivoting per-turn...")
    per_turn = extract_per_turn_logprobs(lp_rows)
    print(f"  {len(per_turn)} turns with both 0% and 100% logprobs")

    # Join
    joined = []
    n_no_label = 0
    n_unlabeled = 0
    for r in per_turn:
        a = labels.get((r["task_id"], r["turn"]))
        if a is None:
            n_unlabeled += 1
            continue
        r["anchored"] = bool(a)
        joined.append(r)
    print(f"  joined: {len(joined)} (skipped {n_unlabeled} unlabeled)")

    n_a = sum(1 for r in joined if r["anchored"])
    n_e = len(joined) - n_a
    print(f"  anchored={n_a}, exploring={n_e}")

    summaries = []
    for name, key in [
        ("logprob_0pct", "logprob_0pct"),
        ("logprob_100pct", "logprob_100pct"),
        ("delta_logprob", "delta_logprob"),
        ("margin_0pct", "margin_0pct"),
        ("margin_100pct", "margin_100pct"),
        ("delta_margin", "delta_margin"),
    ]:
        a_vals = [r[key] for r in joined if r["anchored"]]
        e_vals = [r[key] for r in joined if not r["anchored"]]
        s = summarize(name, a_vals, e_vals)
        summaries.append(s)
        print(f"\n{name}:")
        print(f"  anchored mean={s['anchored_mean']:.3f} median={s['anchored_median']:.3f}")
        print(f"  exploring mean={s['exploring_mean']:.3f} median={s['exploring_median']:.3f}")
        print(f"  Mann-Whitney U={s['mannwhitney_u']:.1f} p={s['mannwhitney_p']:.4g} d={s['cohens_d']:.3f}")

    json_path = os.path.join(args.out_dir, "logprob_analysis.json")
    with open(json_path, "w") as f:
        json.dump({
            "n_joined": len(joined),
            "n_anchored": n_a,
            "n_exploring": n_e,
            "comparisons": summaries,
        }, f, indent=2)
    print(f"\nwrote {json_path}")

    make_figure(joined, args.fig_path)
    print("DONE.")


if __name__ == "__main__":
    main()
