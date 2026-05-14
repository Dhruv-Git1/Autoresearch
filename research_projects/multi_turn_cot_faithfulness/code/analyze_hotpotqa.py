"""
Analyze HotpotQA replication results (tiered max_turns design) and generate
side-by-side gradient figure comparing GSM8K vs HotpotQA.

Tiered design:
  hotpotqa_s1: max_turns=25 -> "long" condition
  hotpotqa_s2: max_turns=10 -> "short" condition
  hotpotqa_s3: max_turns=17 -> "medium" condition

Run from repo root after downloading all 3 result dirs:
  python research_projects/multi_turn_cot_faithfulness/code/analyze_hotpotqa.py \
    --s1_faith  results/hotpotqa_s1/faithfulness.jsonl \
    --s2_faith  results/hotpotqa_s2/faithfulness.jsonl \
    --s3_faith  results/hotpotqa_s3/faithfulness.jsonl \
    --gsm8k_stats results/bistability_v3_combined/bistability_stats.json \
    --out_dir   results/bistability_hotpotqa

Outputs:
  results/bistability_hotpotqa/hotpotqa_gradient_stats.json
  paper/figures/gsm8k_hotpotqa_gradient.png
"""
import argparse
import json
import os
import re
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG_OUT = REPO / "research_projects/multi_turn_cot_faithfulness/paper/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_string_answer(text: str) -> str:
    """3-pattern fallback matching tasks/hotpotqa/task_hotpotqa.py."""
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


def is_anchored_str(row: dict):
    """True if all >=3 valid truncation levels produce the same normalized answer."""
    tr = row.get("result", {})
    trunc_results = tr.get("truncation_results", [])
    if len(trunc_results) < 3:
        return None
    answers = [extract_string_answer(t.get("regen_answer_preview", ""))
               for t in trunc_results]
    answers = [a for a in answers if a]
    if len(answers) < 3:
        return None
    return len(set(answers)) == 1


def load_and_compute(faith_path: str, label: str) -> dict:
    """Load faithfulness.jsonl, compute overall anchoring rate."""
    rows = []
    if not os.path.exists(faith_path):
        print(f"  WARNING: {faith_path} not found — skipping {label}")
        return {"label": label, "n_rows": 0, "n_scoreable_turns": 0,
                "n_anchored": 0, "frac_anchored": 0.0, "n_conversations": 0}
    with open(faith_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_conversations = len({r["task_id"] for r in rows})
    anchored_vals = []
    for r in rows:
        a = is_anchored_str(r)
        if a is not None:
            anchored_vals.append(int(a))

    frac = sum(anchored_vals) / len(anchored_vals) if anchored_vals else 0.0
    print(f"  {label}: {len(rows)} rows, {n_conversations} convs, "
          f"{sum(anchored_vals)}/{len(anchored_vals)} anchored = {frac:.1%}")
    return {
        "label": label,
        "n_rows": len(rows),
        "n_scoreable_turns": len(anchored_vals),
        "n_anchored": sum(anchored_vals),
        "frac_anchored": frac,
        "n_conversations": n_conversations,
    }


def _bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100) * 100
    hi = np.percentile(boots, (1 + ci) / 2 * 100) * 100
    return lo, hi


def load_all_is_anchored_str(faith_path: str) -> list:
    """Return list of int (0/1) is_anchored_str values from a file."""
    out = []
    if not os.path.exists(faith_path):
        return out
    with open(faith_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = is_anchored_str(row)
            if a is not None:
                out.append(int(a))
    return out


def plot_side_by_side(gsm8k_bins: dict, hpqa_conditions: list, out_path: Path):
    """
    Side-by-side bar chart:
      Left panel:  GSM8K (3 length bins from bistability_stats.json)
      Right panel: HotpotQA (3 max_turns conditions: short/medium/long)
    Both panels share y-axis.
    """
    BG   = "#fafafa"
    DARK = "#2c3e50"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    gsm8k_bin_keys = ["short (<10)", "medium (10-20)", "long (>20)"]
    gsm8k_labels   = ["Short\n(<10 turns)", "Medium\n(10-20 turns)", "Long\n(>20 turns)"]
    gsm8k_vals = [gsm8k_bins.get(k, {}).get("mean_anchored", 0) * 100
                  for k in gsm8k_bin_keys]
    gsm8k_n    = [gsm8k_bins.get(k, {}).get("n_conv", 0) for k in gsm8k_bin_keys]

    x = np.arange(3)
    w = 0.55
    colors_gsm8k = ["#3498db", "#e67e22", "#c0392b"]

    ax1.bar(x, gsm8k_vals, width=w, color=colors_gsm8k, alpha=0.75,
            edgecolor="white", linewidth=0.8)
    for xi, (v, n) in enumerate(zip(gsm8k_vals, gsm8k_n)):
        ax1.text(xi, v + 0.8, f"{v:.1f}%\n(n={n} convs)",
                 ha="center", va="bottom", fontsize=8, fontweight="bold", color=DARK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gsm8k_labels, fontsize=8.5)
    ax1.set_ylabel("Anchored fraction (%)", fontsize=9)
    ax1.set_title("GSM8K (Numeric Math)\nN=67 conversations", fontsize=9.5, pad=6)

    if gsm8k_vals[0] > 0 and gsm8k_vals[2] > 0:
        ratio = gsm8k_vals[2] / gsm8k_vals[0]
        y_top = max(gsm8k_vals) + 12
        ax1.annotate("", xy=(x[2], y_top), xytext=(x[0], y_top),
                     arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
        ax1.text(x[1], y_top + 0.8, f"{ratio:.1f}x gradient",
                 ha="center", va="bottom", fontsize=9, color=DARK, fontweight="bold")

    hpqa_labels = [c["label"] for c in hpqa_conditions]
    hpqa_vals   = [c["frac_anchored"] * 100 for c in hpqa_conditions]
    hpqa_n      = [c["n_scoreable_turns"] for c in hpqa_conditions]
    hpqa_ci     = [c.get("ci", (0, 0)) for c in hpqa_conditions]
    colors_hpqa = ["#27ae60", "#16a085", "#2c3e50"]

    ax2.bar(x, hpqa_vals, width=w, color=colors_hpqa, alpha=0.75,
            edgecolor="white", linewidth=0.8)

    for xi, (v, (lo, hi)) in enumerate(zip(hpqa_vals, hpqa_ci)):
        ax2.plot([xi, xi], [lo, hi], color=DARK, lw=1.8, zorder=5)
        ax2.plot([xi - 0.08, xi + 0.08], [lo, lo], color=DARK, lw=1.5, zorder=5)
        ax2.plot([xi - 0.08, xi + 0.08], [hi, hi], color=DARK, lw=1.5, zorder=5)

    for xi, (v, n) in enumerate(zip(hpqa_vals, hpqa_n)):
        ci_hi = hpqa_ci[xi][1]
        ax2.text(xi, max(v, ci_hi) + 1.5, f"{v:.1f}%\n(n={n} turns)",
                 ha="center", va="bottom", fontsize=8, fontweight="bold", color=DARK)

    ax2.set_xticks(x)
    ax2.set_xticklabels(hpqa_labels, fontsize=8.5)
    ax2.set_title(
        "HotpotQA (Factual Multi-hop QA)\nTiered max-turns design; same 20 questions per condition",
        fontsize=9.5, pad=6)

    if hpqa_vals[0] > 0 and hpqa_vals[2] > 0:
        ratio = hpqa_vals[2] / hpqa_vals[0]
        y_top = max(hpqa_vals) + 12
        ax2.annotate("", xy=(x[2], y_top), xytext=(x[0], y_top),
                     arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
        ax2.text(x[1], y_top + 0.8, f"{ratio:.1f}x gradient",
                 ha="center", va="bottom", fontsize=9, color=DARK, fontweight="bold")

    y_max = max(max(gsm8k_vals), max(hpqa_vals)) + 25
    ax1.set_ylim(0, y_max)

    fig.suptitle(
        "CoT anchoring increases with conversation length in both numeric math and factual QA",
        fontsize=11, fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.97, wspace=0.28)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_faith",    required=True,
                    help="hotpotqa_s1/faithfulness.jsonl (max_turns=25, long)")
    ap.add_argument("--s2_faith",    required=True,
                    help="hotpotqa_s2/faithfulness.jsonl (max_turns=10, short)")
    ap.add_argument("--s3_faith",    required=True,
                    help="hotpotqa_s3/faithfulness.jsonl (max_turns=17, medium)")
    ap.add_argument("--gsm8k_stats", required=True,
                    help="bistability_v3_combined/bistability_stats.json")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    base = "research_projects/multi_turn_cot_faithfulness/results"
    s1 = args.s1_faith if not args.s1_faith.startswith("results/") else \
         os.path.join(base, args.s1_faith[len("results/"):])
    s2 = args.s2_faith if not args.s2_faith.startswith("results/") else \
         os.path.join(base, args.s2_faith[len("results/"):])
    s3 = args.s3_faith if not args.s3_faith.startswith("results/") else \
         os.path.join(base, args.s3_faith[len("results/"):])

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading HotpotQA faithfulness files...")
    s2_data = load_and_compute(args.s2_faith, "Short (<=10 turns, s2)")
    s3_data = load_and_compute(args.s3_faith, "Medium (<=17 turns, s3)")
    s1_data = load_and_compute(args.s1_faith, "Long (<=25 turns, s1)")

    s2_vals = load_all_is_anchored_str(args.s2_faith)
    s3_vals = load_all_is_anchored_str(args.s3_faith)
    s1_vals = load_all_is_anchored_str(args.s1_faith)

    s2_data["ci"] = _bootstrap_ci(s2_vals)
    s3_data["ci"] = _bootstrap_ci(s3_vals)
    s1_data["ci"] = _bootstrap_ci(s1_vals)

    s2_data["label"] = "Short\n(<=10 turns)"
    s3_data["label"] = "Medium\n(<=17 turns)"
    s1_data["label"] = "Long\n(<=25 turns)"

    hpqa_conditions = [s2_data, s3_data, s1_data]

    with open(args.gsm8k_stats) as f:
        gsm8k_stats = json.load(f)
    gsm8k_bins = gsm8k_stats.get("anchored_by_length_bin", {})

    print("\n=== Gradient comparison ===")
    print("HotpotQA:")
    for c in hpqa_conditions:
        lo, hi = c.get("ci", (0, 0))
        print(f"  {c['label'].replace(chr(10),' ')}: {c['frac_anchored']:.1%}  "
              f"CI=[{lo:.1f}%, {hi:.1f}%]  n={c['n_scoreable_turns']} turns")

    if s2_data["frac_anchored"] > 0:
        ratio = s1_data["frac_anchored"] / s2_data["frac_anchored"]
        print(f"  Gradient ratio: {ratio:.1f}x")

    print("\nGSM8K (for comparison):")
    for k in ["short (<10)", "medium (10-20)", "long (>20)"]:
        v = gsm8k_bins.get(k, {})
        print(f"  {k}: {v.get('mean_anchored', 0):.1%}  n={v.get('n_conv', 0)} convs")

    stats_out = {
        "design": "tiered_max_turns",
        "conditions": {
            "short_max10":  {k: v for k, v in s2_data.items() if k != "ci"},
            "medium_max17": {k: v for k, v in s3_data.items() if k != "ci"},
            "long_max25":   {k: v for k, v in s1_data.items() if k != "ci"},
        },
        "gradient_ratio_long_vs_short": (
            s1_data["frac_anchored"] / s2_data["frac_anchored"]
            if s2_data["frac_anchored"] > 0 else None
        ),
    }
    stats_path = os.path.join(args.out_dir, "hotpotqa_gradient_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nStats saved: {stats_path}")

    out_fig = FIG_OUT / "gsm8k_hotpotqa_gradient.png"
    plot_side_by_side(gsm8k_bins, hpqa_conditions, out_fig)

    short_pct = s2_data["frac_anchored"] * 100
    med_pct   = s3_data["frac_anchored"] * 100
    long_pct  = s1_data["frac_anchored"] * 100
    ratio = long_pct / short_pct if short_pct > 0 else float("inf")
    print(f"\n=== HEADLINE NUMBERS FOR PAPER TEXT ===")
    print(f"  Short (<=10 turns):  {short_pct:.1f}%")
    print(f"  Medium (<=17 turns): {med_pct:.1f}%")
    print(f"  Long (<=25 turns):   {long_pct:.1f}%  (existing hotpotqa_s1 data)")
    print(f"  Gradient: {ratio:.1f}x")
    print(f"  vs GSM8K overall 18.1%, vs GSM8K long 28.4%")


if __name__ == "__main__":
    main()
