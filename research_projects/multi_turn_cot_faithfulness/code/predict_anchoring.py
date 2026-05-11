"""Exp 5: Predict per-turn anchoring from surface features.

Loads faithfulness.jsonl + trace_sharded_*.json files across all six
phases, derives features per turn (including n_shards_revealed_so_far
when traces are present), labels each turn anchored vs exploring via
is_anchored(), and fits a logistic regression with GroupKFold cross-
validation (grouping by task_id to prevent within-conversation leakage).

Outputs:
  results/anchoring_predictor/predictor_stats.json
  results/anchoring_predictor/per_turn_features.csv
  paper/figures/anchoring_predictor.png  (ROC + coefficient bars)

Run from the repo root:
  python code/predict_anchoring.py
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Reuse from the existing analysis script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_bistability_analysis import (
    is_anchored,
    get_primary_answer,
)


def build_shard_table(trace_dirs: List[str]) -> Dict:
    """Walk trace_sharded_*.json files; return dict (task_id, turn_1idx) -> n_shards_revealed_so_far."""
    table = {}
    for d in trace_dirs:
        if not os.path.isdir(d):
            continue
        for tf in sorted(glob.glob(os.path.join(d, "trace_sharded_*.json"))):
            try:
                with open(tf) as f:
                    payload = json.load(f)
            except Exception:
                continue
            tid = payload.get("task_id")
            trace = payload.get("trace", [])
            cum = 0
            n_assist = 0
            for m in trace:
                role = m.get("role")
                if role == "log":
                    c = m.get("content", {})
                    if isinstance(c, dict) and c.get("type") == "shard_revealed":
                        cum += 1
                elif role == "assistant":
                    n_assist += 1
                    table[(tid, n_assist)] = cum
    return table


def load_faith_rows(faith_paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in faith_paths:
        if not os.path.exists(p):
            print(f"  WARN: {p} not found, skipping")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def build_feature_table(rows: List[Dict[str, Any]],
                          shard_table: Dict = None):
    """Group rows by task_id, sort by turn, derive per-turn features and labels."""
    shard_table = shard_table or {}
    by_tid = defaultdict(list)
    for r in rows:
        if r.get("task_id") is None or r.get("turn") is None:
            continue
        by_tid[r["task_id"]].append(r)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda r: r["turn"])

    feature_rows = []
    for tid, conv_rows in by_tid.items():
        conv_total_turns = len(conv_rows)
        # Max shards observed for this conversation (for shard_progress feature)
        conv_max_shards = max(
            (shard_table.get((tid, int(r["turn"])), 0) for r in conv_rows),
            default=0,
        )
        prev_anchored = None
        prev_answer = None
        for r in conv_rows:
            anchored = is_anchored(r)
            if anchored is None:
                prev_answer = get_primary_answer(r)
                continue
            primary = get_primary_answer(r)
            thinking_chars = (r.get("result") or {}).get("orig_thinking_chars", 0) or 0
            if prev_answer is None:
                prior_changed = 1
            else:
                prior_changed = 0 if (primary is not None and primary == prev_answer) else 1
            n_shards = shard_table.get((tid, int(r["turn"])), None)
            shard_progress = (
                (n_shards / conv_max_shards) if (n_shards is not None and conv_max_shards > 0)
                else 0.0
            )
            feat = {
                "task_id": tid,
                "turn": int(r["turn"]),
                "turn_index": int(r["turn"]),
                "thinking_chars": int(thinking_chars),
                "log_thinking_chars": float(np.log1p(thinking_chars)),
                "prior_answer_changed": int(prior_changed),
                "prior_was_anchored": int(prev_anchored) if prev_anchored is not None else 0,
                "conv_total_turns": int(conv_total_turns),
                "n_shards_revealed": int(n_shards) if n_shards is not None else 0,
                "shard_progress": float(shard_progress),
                "is_anchored": int(anchored),
            }
            feature_rows.append(feat)
            prev_anchored = anchored
            prev_answer = primary
    return feature_rows


def cv_logistic_regression(feature_rows, feature_cols, n_splits=5, seed=42):
    X = np.array([[fr[c] for c in feature_cols] for fr in feature_rows], dtype=float)
    y = np.array([fr["is_anchored"] for fr in feature_rows], dtype=int)
    groups = np.array([fr["task_id"] for fr in feature_rows])

    n_groups = len(set(groups.tolist()))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"Not enough conversations for CV (n_groups={n_groups})")

    gkf = GroupKFold(n_splits=n_splits)
    fold_aucs = []
    fold_oof_probs = np.zeros_like(y, dtype=float)
    fold_assignments = np.full_like(y, -1)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 solver="liblinear", random_state=seed,
                                 max_iter=1000)
        clf.fit(X_tr, y[train_idx])
        probs = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y[test_idx], probs)
        except ValueError:
            auc = float("nan")
        fold_aucs.append(auc)
        fold_oof_probs[test_idx] = probs
        fold_assignments[test_idx] = fold_idx

    # Refit on full data for final coefficients
    scaler_full = StandardScaler().fit(X)
    Xs = scaler_full.transform(X)
    clf_full = LogisticRegression(C=1.0, class_weight="balanced",
                                  solver="liblinear", random_state=seed,
                                  max_iter=1000).fit(Xs, y)
    coefs = clf_full.coef_[0].tolist()
    intercept = float(clf_full.intercept_[0])

    pooled_auc = float(roc_auc_score(y, fold_oof_probs))
    return {
        "fold_aucs": [float(a) for a in fold_aucs],
        "fold_auc_mean": float(np.nanmean(fold_aucs)),
        "fold_auc_std": float(np.nanstd(fold_aucs)),
        "pooled_oof_auc": pooled_auc,
        "feature_cols": feature_cols,
        "coefficients_standardized": coefs,
        "intercept": intercept,
        "n_total": int(len(y)),
        "n_anchored": int(y.sum()),
        "frac_anchored": float(y.mean()),
        "n_conversations": n_groups,
        "n_splits_used": n_splits,
    }, y, fold_oof_probs


def make_figure(stats: Dict[str, Any], y: np.ndarray, oof_probs: np.ndarray,
                out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: ROC curve
    fpr, tpr, _ = roc_curve(y, oof_probs)
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1a7a3a", lw=2,
            label=f"OOF AUC = {stats['pooled_oof_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Anchoring predictor (GroupKFold CV, N={stats['n_total']} turns, "
                 f"{stats['n_conversations']} conv)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Right panel: standardized coefficient bars
    ax = axes[1]
    cols = stats["feature_cols"]
    coefs = stats["coefficients_standardized"]
    order = np.argsort(np.abs(coefs))[::-1]
    cols_o = [cols[i] for i in order]
    coefs_o = [coefs[i] for i in order]
    colors = ["#a31515" if c < 0 else "#1a7a3a" for c in coefs_o]
    ax.barh(range(len(cols_o)), coefs_o, color=colors)
    ax.set_yticks(range(len(cols_o)))
    ax.set_yticklabels(cols_o)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized logistic coefficient")
    ax.set_title("Feature importance (positive → predicts anchored)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Predicting anchored turns from surface features",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    base = "research_projects/multi_turn_cot_faithfulness/results"
    ap.add_argument("--faith_paths", nargs="+", default=[
        f"{base}/phase2/faithfulness.jsonl",
        f"{base}/phase3/faithfulness.jsonl",
        f"{base}/phase4/faithfulness.jsonl",
        f"{base}/phase5_s1/faithfulness.jsonl",
        f"{base}/phase5_s2/faithfulness.jsonl",
        f"{base}/phase5_s3/faithfulness.jsonl",
    ])
    ap.add_argument("--trace_dirs", nargs="+", default=[
        f"{base}/phase2",
        f"{base}/phase3",
        f"{base}/phase4",
        f"{base}/phase5_s1",
        f"{base}/phase5_s2",
        f"{base}/phase5_s3",
    ])
    ap.add_argument("--out_dir", default=f"{base}/anchoring_predictor")
    ap.add_argument("--fig_path", default="research_projects/multi_turn_cot_faithfulness/paper/figures/anchoring_predictor.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)

    print("Loading faithfulness rows...")
    rows = load_faith_rows(args.faith_paths)
    print(f"  loaded {len(rows)} rows")

    print("Loading trace files for shard progress feature...")
    shard_table = build_shard_table(args.trace_dirs)
    print(f"  loaded n_shards_revealed for {len(shard_table)} (task_id, turn) keys")

    print("Building per-turn feature table...")
    feat_rows = build_feature_table(rows, shard_table=shard_table)
    print(f"  {len(feat_rows)} labelable turns across "
          f"{len(set(fr['task_id'] for fr in feat_rows))} conversations")

    feature_cols = [
        "turn_index",
        "log_thinking_chars",
        "prior_answer_changed",
        "prior_was_anchored",
        "conv_total_turns",
        "n_shards_revealed",
        "shard_progress",
    ]
    print(f"Fitting logistic regression with features: {feature_cols}")
    stats, y, oof = cv_logistic_regression(feat_rows, feature_cols)
    print(f"  Fold AUCs: {[f'{a:.3f}' for a in stats['fold_aucs']]}")
    print(f"  Mean fold AUC: {stats['fold_auc_mean']:.3f} ± {stats['fold_auc_std']:.3f}")
    print(f"  Pooled OOF AUC: {stats['pooled_oof_auc']:.3f}")

    # Save CSV of features
    csv_path = os.path.join(args.out_dir, "per_turn_features.csv")
    with open(csv_path, "w") as f:
        cols = ["task_id", "turn", "is_anchored"] + feature_cols
        f.write(",".join(cols) + "\n")
        for fr in feat_rows:
            f.write(",".join(str(fr[c]) for c in cols) + "\n")
    print(f"  wrote {csv_path}")

    # Save JSON stats
    json_path = os.path.join(args.out_dir, "predictor_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {json_path}")

    # Figure
    make_figure(stats, y, oof, args.fig_path)
    print("DONE.")


if __name__ == "__main__":
    main()
