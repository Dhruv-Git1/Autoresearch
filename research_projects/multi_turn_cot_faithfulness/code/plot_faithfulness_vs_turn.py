"""Day 3 decision plot: faithfulness vs turn-index.

Reads the JSONL emitted by faithfulness_counterfactual.py and produces:
  1. results/day3/faithfulness_vs_turn.png   — primary decision plot
  2. results/day3/decision_stats.json        — Pearson r, p-value, n
  3. results/day3/per_turn_summary.csv       — mean / std / n per turn
  4. (if available) length-stratified plot   — confound check

Decision rule (per the plan):
  |Pearson r| > 0.3 with p < 0.05  →  GRADUATE
  otherwise                         →  SHELVE / pivot
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_results(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean_ci(x, alpha=0.05):
    x = np.asarray([v for v in x if v is not None])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m, 1
    se = stats.sem(x)
    h = se * stats.t.ppf(1 - alpha / 2, len(x) - 1)
    return m, m - h, m + h, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", default="../results/day2/faithfulness.jsonl")
    ap.add_argument("--out_dir", default="../results/day3")
    ap.add_argument("--length_records", default="../results/day1/per_turn_records.jsonl",
                    help="for length-stratified confound check")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_results(args.results_path)
    print(f"loaded {len(rows)} (task_id, turn) faithfulness records")

    # build per-turn summary
    by_turn = defaultdict(list)
    pairs = []
    for r in rows:
        t = r.get("turn")
        f = r.get("faithfulness_score")
        if f is None:
            continue
        by_turn[t].append(f)
        pairs.append((t, f))

    summary = []
    for t in sorted(by_turn):
        m, lo, hi, n = mean_ci(by_turn[t])
        summary.append({"turn": t, "n": n, "mean_faithfulness": m, "ci95_lo": lo, "ci95_hi": hi})
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(args.out_dir, "per_turn_summary.csv"), index=False)
    print(df.to_string(index=False))

    # Pearson correlation between turn and per-conversation faithfulness
    turns = np.array([p[0] for p in pairs])
    faiths = np.array([p[1] for p in pairs])
    if len(pairs) >= 3:
        r, p = stats.pearsonr(turns, faiths)
        decision = "GRADUATE" if abs(r) > 0.3 and p < 0.05 else "SHELVE"
    else:
        r, p, decision = float("nan"), float("nan"), "INSUFFICIENT_DATA"
    stats_payload = {
        "n_total_observations": len(pairs),
        "n_unique_samples": len({r["task_id"] for r in rows}),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "decision": decision,
        "rule": "|r|>0.3 and p<0.05 -> GRADUATE",
    }
    with open(os.path.join(args.out_dir, "decision_stats.json"), "w") as f:
        json.dump(stats_payload, f, indent=2)
    print("\n=== Decision stats ===")
    print(json.dumps(stats_payload, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(df):
        ax.errorbar(
            df["turn"], df["mean_faithfulness"],
            yerr=[df["mean_faithfulness"] - df["ci95_lo"], df["ci95_hi"] - df["mean_faithfulness"]],
            marker="o", capsize=4, lw=1.6, label="mean ± 95% CI",
        )
    # scatter raw
    ax.scatter(turns, faiths, alpha=0.3, s=24, color="#888", label="per-conversation")
    ax.set_xlabel("Assistant turn-index in SHARDED conversation")
    ax.set_ylabel("CoT-deletion faithfulness score")
    ax.set_title(f"Faithfulness vs turn  (n={len(pairs)} obs, r={r:.2f}, p={p:.3f})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    out_png = os.path.join(args.out_dir, "faithfulness_vs_turn.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nsaved plot to {out_png}")

    # Length-stratified confound check (if records available)
    if os.path.exists(args.length_records):
        try:
            recs = [json.loads(l) for l in open(args.length_records)]
            keyed = {(r["task_id"], r["assistant_turn_idx"]): r.get("thinking_chars", 0) for r in recs}
            # Pair faithfulness with length
            length_rows = []
            for r in rows:
                key = (r["task_id"], r["turn"])
                length = keyed.get(key)
                if length is None or r.get("faithfulness_score") is None:
                    continue
                length_rows.append({"turn": r["turn"], "length": length, "faith": r["faithfulness_score"]})
            if length_rows:
                ldf = pd.DataFrame(length_rows)
                # quartile bins
                ldf["len_q"] = pd.qcut(ldf["length"], q=min(4, ldf["length"].nunique()), labels=False, duplicates="drop")
                fig2, ax2 = plt.subplots(figsize=(7, 4.5))
                for q in sorted(ldf["len_q"].dropna().unique()):
                    sub = ldf[ldf["len_q"] == q].groupby("turn")["faith"].mean().reset_index()
                    ax2.plot(sub["turn"], sub["faith"], marker="o", label=f"length q{int(q)}")
                ax2.set_xlabel("Turn"); ax2.set_ylabel("mean faithfulness")
                ax2.set_title("Length-stratified faithfulness (confound check)")
                ax2.grid(alpha=0.3); ax2.legend()
                fig2.tight_layout()
                fig2.savefig(os.path.join(args.out_dir, "faithfulness_length_stratified.png"), dpi=150)
                print(f"saved length-stratified plot")
        except Exception as e:
            print(f"length-stratified plot failed: {e}")

    print(f"\n=== DECISION: {decision} ===")
    print("(GRADUATE means proceed to Phase 2 confound work + N=50 sweep)")


if __name__ == "__main__":
    main()
