"""Phase 2 — Bistability analysis.

Three hypotheses:
  H1: Anchored-mode runs are longer than a geometric (i.i.d.) baseline,
      meaning the model is genuinely "stuck" rather than randomly producing
      the same answer by coincidence.

      Statistical test: parametric bootstrap of the KS statistic under the
      fitted geometric null (fixes the discrete-distribution artifact in the
      standard scipy.stats.kstest, which always gives D ≥ p_hat at the left
      boundary for integer-valued run-length data, making p<0.001 guaranteed
      by construction regardless of data — see Massey 1951).
      A chi-square GOF test is reported as a second independent test.

  H2: Fraction of turns spent in anchored mode negatively predicts
      conversation accuracy (anchored = lost in conversation).

  H3: Across conversations, there is a bimodal distribution of per-turn
      faithfulness — most turns are near 0 (anchored) or near 1 (exploring),
      not uniformly distributed.

      Extended test: chi-square test on per-conversation anchored fractions
      vs the Bernoulli(p_global) null, to verify that conversations vary more
      than pure chance would predict (genuine mode-switching).

  Confound check: fraction of anchored turns that are "self-repetitions"
      (anchored answer == answer on the immediately preceding turn). If high
      (>70%), anchored mode may reflect answer repetition under conversational
      pressure rather than post-hoc rationalization per se.

  Commitment alignment: for each conversation, correlate the first anchored
      turn with the "commitment turn" (first turn with >=2 consecutive
      identical-answer turns), to test the premature-commitment alignment
      claim systematically across all conversations.

  Length-controlled: report anchored fraction binned by conversation length
      to check for a confound from longer conversations having more anchored
      turns.

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
from scipy.stats import chi2, pointbiserialr, pearsonr

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


def get_primary_answer(row):
    """Return the numeric answer at 100% truncation (full CoT), or None."""
    tr = row.get("result", {})
    trunc_results = tr.get("truncation_results", [])
    if not trunc_results:
        return None
    full = trunc_results[-1].get("regen_answer_preview", "")
    n = extract_numeric(full)
    return n if n else None


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


# ── H1 helpers ────────────────────────────────────────────────────────────────

def _ks_stat_discrete(data, p):
    """Two-sided KS statistic between data and Geometric(p) CDF."""
    sorted_data = np.sort(np.asarray(data, dtype=int))
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n
    theory_cdf = 1.0 - (1.0 - p) ** sorted_data
    d_plus = float(np.max(ecdf - theory_cdf))
    d_minus = float(np.max(theory_cdf - np.arange(0, n) / n))
    return max(d_plus, d_minus)


def bootstrap_ks_p(observed_runs, n_bootstrap=10_000, seed=42):
    """Parametric bootstrap p-value for H1.

    Fits Geometric(p_hat) to the observed run lengths, then repeatedly
    draws samples of the same size from that null and recomputes the KS
    statistic (re-fitting p on each simulated sample, mirroring what was
    done on the real data).  The p-value is the fraction of null samples
    whose KS stat >= the observed stat.

    This corrects the well-known discrete-distribution artifact of scipy's
    kstest: when data are integers starting at 1 and the null is Geometric(p),
    the D- term equals p_hat at the left boundary, guaranteeing KS >= p_hat
    regardless of whether the data actually fit the distribution.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(observed_runs, dtype=int)
    n = len(data)
    p_hat = 1.0 / float(np.mean(data))
    obs_stat = _ks_stat_discrete(data, p_hat)

    null_stats = []
    for _ in range(n_bootstrap):
        sim = rng.geometric(p=p_hat, size=n)
        p_sim = 1.0 / float(np.mean(sim))
        null_stats.append(_ks_stat_discrete(sim, p_sim))

    p_value = float(np.mean(np.array(null_stats) >= obs_stat))
    return obs_stat, p_value, p_hat


def chisq_geometric_gof(observed_runs, p_hat):
    """Chi-square GOF for geometric run lengths; bins: 1, 2, 3, >=4.

    Degrees of freedom = (non-empty bins - 1) - 1 (for the estimated p).
    Returns (chi2_stat, p_value, obs_counts, exp_counts).
    """
    data = np.asarray(observed_runs, dtype=int)
    n = len(data)
    obs_counts = [
        int(np.sum(data == 1)),
        int(np.sum(data == 2)),
        int(np.sum(data == 3)),
        int(np.sum(data >= 4)),
    ]
    exp_probs = [
        p_hat,
        p_hat * (1 - p_hat),
        p_hat * (1 - p_hat) ** 2,
        (1 - p_hat) ** 3,
    ]
    exp_counts = [e * n for e in exp_probs]

    # Pool bins with expected < 5 into neighbours
    pooled_obs, pooled_exp = [], []
    acc_o, acc_e = 0, 0.0
    for o, e in zip(obs_counts, exp_counts):
        if e < 5:
            acc_o += o
            acc_e += e
        else:
            if acc_e > 0:
                pooled_obs.append(acc_o)
                pooled_exp.append(acc_e)
                acc_o, acc_e = 0, 0.0
            pooled_obs.append(o)
            pooled_exp.append(e)
    if acc_e > 0:
        pooled_obs.append(acc_o)
        pooled_exp.append(acc_e)

    if len(pooled_obs) < 2:
        return float("nan"), float("nan"), obs_counts, exp_counts

    stat = float(sum((o - e) ** 2 / e for o, e in zip(pooled_obs, pooled_exp)))
    df = max(1, len(pooled_obs) - 1 - 1)  # -1 for estimated p
    p_val = float(1.0 - chi2.cdf(stat, df))
    return stat, p_val, obs_counts, exp_counts


# ── H3 extended test ───────────────────────────────────────────────────────────

def h3_variance_test(conv_records):
    """Chi-square test: do conversations vary more than Bernoulli(p_global) predicts?

    Under i.i.d. Bernoulli(p), per-conversation anchored fraction has
    variance p(1-p)/n_turns.  Standardised residuals should be ~N(0,1);
    their sum-of-squares is approximately chi-square(k-1).  A significant
    result means conversations differ in anchoring rate beyond pure chance
    -- genuine mode-switching rather than independent per-turn coin flips.

    Returns (chi2_stat, df, p_value, p_global).
    """
    fracs = np.array([r["frac_anchored"] for r in conv_records])
    ns    = np.array([r["n_turns"]       for r in conv_records], dtype=float)
    if len(fracs) < 3 or np.all(ns == 0):
        return float("nan"), 0, float("nan"), float("nan")

    p_global = float(np.sum(fracs * ns) / np.sum(ns))
    if p_global <= 0 or p_global >= 1:
        return float("nan"), 0, float("nan"), p_global

    variances = p_global * (1.0 - p_global) / ns
    residuals = (fracs - p_global) / np.sqrt(variances)
    stat = float(np.sum(residuals ** 2))
    df   = len(fracs) - 1
    p_val = float(1.0 - chi2.cdf(stat, df))
    return stat, df, p_val, p_global


# ── H3 within-bin variance test ──────────────────────────────────────────────

def h3_variance_test_within_bins(conv_records):
    """Run H3 variance test separately within each conversation-length bin.

    The pooled chi-square test is confounded by length because short
    conversations almost certainly have 0 anchored turns by chance, while
    long ones accumulate many.  Running within each length stratum removes
    this artifact: if H3 holds inside the long-conversation bin, the
    between-conversation variance is not merely a length effect.

    Returns dict keyed by bin name; each value has n_conv, chi2, df, p.
    """
    bin_defs = [
        ("short_leq10",  lambda r: r["n_turns"] <= 10),
        ("medium_10_20", lambda r: 10 < r["n_turns"] <= 20),
        ("long_gt20",    lambda r: r["n_turns"] > 20),
    ]
    results = {}
    for name, predicate in bin_defs:
        records = [r for r in conv_records if predicate(r)]
        if len(records) < 3:
            results[name] = {"n_conv": len(records), "verdict": "insufficient data"}
            continue
        stat, df_val, p_val, p_global = h3_variance_test(records)
        p_str   = round(p_val,   4) if not math.isnan(p_val)   else "nan"
        chi_str = round(stat,    3) if not math.isnan(stat)     else "nan"
        pg_str  = round(p_global,3) if not math.isnan(p_global) else "nan"
        results[name] = {
            "n_conv":    len(records),
            "chi2":      chi_str,
            "df":        df_val,
            "p":         p_str,
            "p_global":  pg_str,
            "significant": (not isinstance(p_str, str) and p_str < 0.05),
        }
    return results


# ── ICC (intraclass correlation) ──────────────────────────────────────────────

def icc_from_binary_sequences(conv_records):
    """One-way ANOVA ICC for the anchored-turn indicator.

    ICC = between-conversation variance / (between + within variance).
    High ICC means the conversation (not the turn) determines whether
    a turn is anchored -- direct evidence that anchoring is a
    conversation-level mode, not random per-turn noise.
    Implemented via variance components without statsmodels.
    """
    groups = [r["anchored_seq"] for r in conv_records
              if len(r.get("anchored_seq", [])) >= 2]
    if len(groups) < 3:
        return {"icc": "nan", "verdict": "insufficient data (need ≥3 conversations)"}

    all_vals   = [v for g in groups for v in g]
    grand_mean = float(np.mean(all_vals))
    n_total    = len(all_vals)
    k          = len(groups)

    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups)
    ss_within  = sum((v - float(np.mean(g))) ** 2 for g in groups for v in g)

    df_between = max(k - 1, 1)
    df_within  = max(n_total - k, 1)

    ms_between = ss_between / df_between
    ms_within  = ss_within  / df_within
    n_bar      = n_total / k

    sigma2_b = (ms_between - ms_within) / n_bar
    sigma2_w = ms_within
    denom    = sigma2_b + sigma2_w
    icc      = float(np.clip(sigma2_b / denom, 0, 1)) if denom > 0 else 0.0

    strength = "strong" if icc > 0.3 else "moderate" if icc > 0.1 else "weak"
    return {
        "icc":            round(icc, 4),
        "ms_between":     round(ms_between, 6),
        "ms_within":      round(ms_within,  6),
        "n_conv":         k,
        "n_turns_total":  n_total,
        "verdict": (
            f"ICC={icc:.3f} ({strength} conversation-level clustering); "
            f"{icc:.0%} of anchoring variance is between conversations"
        ),
    }


# ── Confound check ─────────────────────────────────────────────────────────────

def repetition_confound_fraction(by_tid):
    """What fraction of anchored turns simply repeat the previous turn's answer?

    Returns (n_anchored, n_repetitions, fraction).
    """
    n_anchored = 0
    n_repetitions = 0

    for tid, turn_rows in by_tid.items():
        prev_answer = None
        for r in turn_rows:
            a = is_anchored(r)
            ans = get_primary_answer(r)
            if a is True and ans is not None:
                n_anchored += 1
                if prev_answer is not None and ans == prev_answer:
                    n_repetitions += 1
            if ans is not None:
                prev_answer = ans

    frac = n_repetitions / n_anchored if n_anchored > 0 else float("nan")
    return n_anchored, n_repetitions, frac


def repetition_rate_by_mode(by_tid):
    """Compare per-turn repetition rates for anchored vs exploring turns.

    If anchored turns repeat the prior answer at much higher rates than
    exploring turns (ratio ≥ 2×), anchoring is a real mode above natural
    answer inertia, not a simple repetition artifact.
    If the rates are similar, the confound is substantial.
    """
    anch_rep, anch_total = 0, 0
    expl_rep, expl_total = 0, 0

    for tid, turn_rows in by_tid.items():
        prev_answer = None
        for r in turn_rows:
            a   = is_anchored(r)
            ans = get_primary_answer(r)
            if a is True and ans is not None and prev_answer is not None:
                anch_total += 1
                if ans == prev_answer:
                    anch_rep += 1
            elif a is False and ans is not None and prev_answer is not None:
                expl_total += 1
                if ans == prev_answer:
                    expl_rep += 1
            if ans is not None:
                prev_answer = ans

    anch_rate = anch_rep / anch_total if anch_total else float("nan")
    expl_rate = expl_rep / expl_total if expl_total else float("nan")
    ratio = (anch_rate / expl_rate
             if not (math.isnan(anch_rate) or math.isnan(expl_rate) or expl_rate == 0)
             else float("nan"))

    if math.isnan(ratio):
        interp = "insufficient data"
    elif ratio >= 2.0:
        interp = (f"Anchored turns repeat at {ratio:.1f}× the rate of exploring "
                  f"({anch_rate:.0%} vs {expl_rate:.0%}) — anchoring is a real "
                  f"mode above natural answer inertia")
    else:
        interp = (f"Anchored repetition ({anch_rate:.0%}) similar to exploring "
                  f"({expl_rate:.0%}, ratio={ratio:.2f}) — confound may be "
                  f"substantial; reinterpret as answer-inertia effect")

    return {
        "anchored_repetition_rate":            round(anch_rate, 3) if not math.isnan(anch_rate) else "nan",
        "exploring_repetition_rate":           round(expl_rate, 3) if not math.isnan(expl_rate) else "nan",
        "anchored_n":                          anch_total,
        "exploring_n":                         expl_total,
        "rate_ratio_anchored_over_exploring":  round(ratio, 2) if not math.isnan(ratio) else "nan",
        "interpretation":                      interp,
    }


# ── Commitment alignment ───────────────────────────────────────────────────────

def commitment_alignment(by_tid):
    """For each conversation, return first_anchored_turn and commitment_turn.

    commitment_turn: first turn index where >=2 consecutive turns share the
    same primary answer (proxy for premature commitment onset).
    Only returns records for conversations that have >=1 anchored turn.
    """
    records = []
    for tid, turn_rows in by_tid.items():
        seq = []
        for r in turn_rows:
            a = is_anchored(r)
            ans = get_primary_answer(r)
            seq.append((r["turn"], a, ans))

        valid_turns = [(idx, a, ans) for idx, a, ans in seq
                       if a is not None and ans is not None]
        if not valid_turns:
            continue

        anchored_turns = [idx for idx, a, _ in valid_turns if a is True]
        if not anchored_turns:
            continue
        first_anchored = min(anchored_turns)

        # commitment_turn: start of the first run of >=2 identical answers
        commitment = None
        prev_ans = None
        run_start = None
        for idx, _, ans in valid_turns:
            if ans == prev_ans and commitment is None:
                commitment = run_start
            else:
                prev_ans = ans
                run_start = idx

        records.append({
            "task_id":            tid,
            "first_anchored_turn": first_anchored,
            "commitment_turn":    commitment,
        })

    return records


# ── Length-controlled anchored fraction ───────────────────────────────────────

def anchored_by_length_bin(conv_records):
    """Mean anchored fraction binned by conversation length (short/medium/long)."""
    bins = {"short (<10)": [], "medium (10-20)": [], "long (>20)": []}
    for r in conv_records:
        n = r["n_turns"]
        f = r["frac_anchored"]
        if n < 10:
            bins["short (<10)"].append(f)
        elif n <= 20:
            bins["medium (10-20)"].append(f)
        else:
            bins["long (>20)"].append(f)
    return {
        k: {"n_conv": len(v),
            "mean_anchored": round(float(np.mean(v)), 3) if v else float("nan")}
        for k, v in bins.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────

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
    ap.add_argument("--n_bootstrap", type=int, default=10_000,
                    help="Bootstrap iterations for H1 KS p-value")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load faithfulness rows ────────────────────────────────────────────────
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

    # ── Load traces for is_correct labels ────────────────────────────────────
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

    # ── Build per-conversation data ───────────────────────────────────────────
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
            "task_id":          tid,
            "n_turns":          len(valid_anchored),
            "frac_anchored":    round(frac_anchored, 3),
            "n_anchored_runs":  len(anchored_rl),
            "max_anchored_run": max(anchored_rl) if anchored_rl else 0,
            "mean_anchored_run": float(np.mean(anchored_rl)) if anchored_rl else 0.0,
            "mean_faith":       float(np.mean(valid_faith)) if valid_faith else float("nan"),
            "is_correct":       correct_by_id.get(tid),
            "anchored_seq":     valid_anchored,
        })

    df = pd.DataFrame(conv_records)
    print(f"\n{len(df)} conversations analyzed:")
    print(df[["task_id", "n_turns", "frac_anchored", "max_anchored_run",
              "is_correct"]].to_string(index=False))

    # ── Statistical tests ─────────────────────────────────────────────────────
    stats_out = {}

    # H1: Anchored run-length vs geometric — parametric bootstrap + chi-square
    if all_anchored_rl:
        mean_rl = float(np.mean(all_anchored_rl))
        p_hat   = 1.0 / mean_rl

        bs_stat, bs_p, _ = bootstrap_ks_p(
            all_anchored_rl, n_bootstrap=args.n_bootstrap)

        cs_stat, cs_p, obs_cnt, exp_cnt = chisq_geometric_gof(
            all_anchored_rl, p_hat)

        h1 = {
            "n_anchored_runs":      len(all_anchored_rl),
            "mean_run_length":      round(mean_rl, 3),
            "p_geometric_mle":      round(p_hat, 4),
            "bootstrap_ks_stat":    round(bs_stat, 4),
            "bootstrap_ks_p":       round(bs_p, 4),
            "chisq_stat":
                round(cs_stat, 4) if not math.isnan(cs_stat) else "nan",
            "chisq_p":
                round(cs_p, 4)    if not math.isnan(cs_p)    else "nan",
            "obs_counts_1_2_3_ge4": obs_cnt,
            "exp_counts_1_2_3_ge4": [round(e, 1) for e in exp_cnt],
        }
        sig = ((not math.isnan(bs_p) and bs_p < 0.05) or
               (not math.isnan(cs_p) and cs_p < 0.05))
        h1["verdict"] = (
            "anchored runs exceed geometric null "
            f"(bootstrap p={bs_p:.4f}, chisq p={cs_p:.4f})"
            if sig else
            "consistent with geometric null -- no significant excess clustering "
            f"at N={len(all_anchored_rl)} runs "
            f"(bootstrap p={bs_p:.4f}, chisq p={cs_p:.4f})"
        )
        stats_out["H1_runlength"] = h1
        print(f"\nH1  n_runs={len(all_anchored_rl)}  mean_rl={mean_rl:.2f}"
              f"  p_hat={p_hat:.3f}")
        print(f"    Bootstrap KS: stat={bs_stat:.3f}  p={bs_p:.4f}")
        print(f"    Chi-square:   stat={cs_stat:.3f}  p={cs_p:.4f}")
        print(f"    -> {h1['verdict']}")

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
        h2["verdict"] = (
            f"significant negative correlation (r={r_val:.2f}, p={p_val:.3f})"
            if p_val < 0.05 and r_val < 0 else
            f"not significant (r={r_val:.2f}, p={p_val:.3f})"
        )
        stats_out["H2_frac_anchored_vs_correct"] = h2
        print(f"\nH2  r={r_val:.3f}  p={p_val:.3f}  n={len(valid)}"
              f"  -> {h2['verdict']}")

    # H3: Mode co-existence + per-conversation variance test
    all_anchored_vals = []
    for tid, turn_rows in by_tid.items():
        for r in turn_rows:
            a = is_anchored(r)
            if a is not None:
                all_anchored_vals.append(int(a))

    if len(all_anchored_vals) >= 20:
        frac_anchored_global  = sum(all_anchored_vals) / len(all_anchored_vals)
        frac_exploring_global = 1.0 - frac_anchored_global
        both_modes_present = min(frac_anchored_global, frac_exploring_global) > 0.05

        h3_var_stat, h3_var_df, h3_var_p, _ = h3_variance_test(conv_records)

        h3 = {
            "n_obs":               len(all_anchored_vals),
            "frac_anchored":       round(frac_anchored_global, 3),
            "frac_exploring":      round(frac_exploring_global, 3),
            "both_modes_present":  both_modes_present,
            "variance_chisq_stat":
                round(h3_var_stat, 3) if not math.isnan(h3_var_stat) else "nan",
            "variance_chisq_df":   h3_var_df,
            "variance_chisq_p":
                round(h3_var_p, 4) if not math.isnan(h3_var_p) else "nan",
        }
        h3["verdict"] = (
            f"mode co-existence confirmed: anchored={frac_anchored_global:.0%}, "
            f"exploring={frac_exploring_global:.0%}; "
            f"variance test p={h3_var_p:.4f}"
            if both_modes_present else
            f"one mode dominates (anchored={frac_anchored_global:.0%})"
        )
        stats_out["H3_bimodality"] = h3
        print(f"\nH3  frac_anchored={frac_anchored_global:.2f}  "
              f"both_modes={both_modes_present}  "
              f"variance_chisq_p={h3_var_p:.4f}  -> {h3['verdict']}")

    # H3 within-bin variance test (removes length confound from pooled chi-square)
    within_bin_h3 = h3_variance_test_within_bins(conv_records)
    stats_out["H3_within_bin_variance"] = within_bin_h3
    print("\nH3 within-bin variance test (length-stratified):")
    for bin_name, info in within_bin_h3.items():
        sig = " ** SIGNIFICANT **" if info.get("significant") else ""
        print(f"  {bin_name}: n={info['n_conv']}  chi2={info.get('chi2','?')}  "
              f"p={info.get('p','?')}{sig}")

    # ICC: conversation-level clustering of anchored turns
    icc_result = icc_from_binary_sequences(conv_records)
    stats_out["icc_conversation_level"] = icc_result
    print(f"\nICC: {icc_result.get('verdict', '?')}")

    # Confound: self-repetition fraction (pooled)
    n_anch, n_rep, rep_frac = repetition_confound_fraction(by_tid)
    rep_frac_safe = round(rep_frac, 3) if not math.isnan(rep_frac) else "nan"
    stats_out["confound_repetition"] = {
        "n_anchored_turns":    n_anch,
        "n_self_repetitions":  n_rep,
        "repetition_fraction": rep_frac_safe,
        "interpretation": (
            "HIGH: anchored mode may largely reflect answer repetition "
            "under conversational pressure (context-solvability confound)"
            if (not math.isnan(rep_frac) and rep_frac > 0.70) else
            "LOW/MODERATE: most anchored turns are not simple self-repetitions; "
            "context-solvability confound is limited"
        ),
    }
    print(f"\nConfound check (pooled): {n_rep}/{n_anch} anchored turns are "
          f"self-repetitions ({rep_frac:.1%})")

    # Confound: repetition rate by mode (anchored vs exploring, the decisive comparison)
    rep_by_mode = repetition_rate_by_mode(by_tid)
    stats_out["confound_repetition_by_mode"] = rep_by_mode
    print(f"Repetition by mode: {rep_by_mode['interpretation']}")

    # Commitment alignment
    ca_records = commitment_alignment(by_tid)
    paired = [(r["first_anchored_turn"], r["commitment_turn"])
              for r in ca_records if r["commitment_turn"] is not None]
    if len(paired) >= 5:
        fa_turns = [p[0] for p in paired]
        ct_turns = [p[1] for p in paired]
        r_ca, p_ca = pearsonr(fa_turns, ct_turns)
        mean_offset = float(np.mean([fa - ct for fa, ct in paired]))
        ca_stats = {
            "n_conversations_with_commitment": len(paired),
            "pearson_r":        round(r_ca, 4),
            "pearson_p":        round(p_ca, 4),
            "mean_offset_turns": round(mean_offset, 2),
            "verdict": (
                f"anchored onset correlates with commitment onset "
                f"(r={r_ca:.2f}, p={p_ca:.3f}); "
                f"mean offset = {mean_offset:.1f} turns"
                if p_ca < 0.05 else
                f"no significant correlation between anchored and commitment onset "
                f"(r={r_ca:.2f}, p={p_ca:.3f})"
            ),
        }
    else:
        ca_stats = {
            "n_conversations_with_commitment": len(paired),
            "verdict": "insufficient paired data for correlation",
        }
    stats_out["commitment_alignment"] = ca_stats
    print(f"\nCommitment alignment: {ca_stats['verdict']}")

    # Length-controlled analysis
    len_bins = anchored_by_length_bin(conv_records)
    stats_out["anchored_by_length_bin"] = len_bins
    print("\nAnchored fraction by conversation length:")
    for bin_name, info in len_bins.items():
        print(f"  {bin_name}: n_conv={info['n_conv']}  "
              f"mean_anchored={info['mean_anchored']:.3f}")

    any_significant = (
        stats_out.get("H1_runlength", {}).get("bootstrap_ks_p", 1.0) < 0.05 or
        (not math.isnan(stats_out.get("H1_runlength", {}).get("chisq_p", float("nan")))
         and stats_out.get("H1_runlength", {}).get("chisq_p", 1.0) < 0.05) or
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

    # ── Plot 1: Faithfulness heatmap ─────────────────────────────────────────
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
    im = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Conversation (GSM8K sample ID)")
    ax.set_yticks(range(n_conv))
    ax.set_yticklabels(
        [f"{row['task_id'].split('/')[-1]}"
         for _, row in df.iterrows()],
        fontsize=7,
    )

    ax2 = ax.twinx()
    ax2.set_ylim(-0.5, n_conv - 0.5)
    ax2.set_yticks(range(n_conv))
    ax2.set_yticklabels(
        [f"{row['frac_anchored']:.0%}" for _, row in df.iterrows()],
        fontsize=7,
    )
    ax2.set_ylabel("Anchored fraction", color="#e74c3c")

    n_valid  = int(np.sum(~np.isnan(heat)))
    n_anch_h = int(np.nansum(heat))
    ax.set_title(
        f"Per-turn CoT faithfulness mode (N={n_conv} conversations, "
        f"{n_valid} turns)\n"
        f"Overall: {n_anch_h}/{n_valid} turns anchored "
        f"({n_anch_h/n_valid:.0%})\n"
        f"green = exploring (CoT causal)   red = anchored (CoT post-hoc)"
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.02)
    cbar.set_ticklabels(["exploring\n(CoT causal)", "anchored\n(CoT post-hoc)"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "heatmap_faithfulness.png"), dpi=150)
    plt.close(fig)
    print("saved heatmap_faithfulness.png")

    # ── Plot 2: Run-length distribution vs geometric null ────────────────────
    if all_anchored_rl:
        max_run    = max(all_anchored_rl)
        bins       = np.arange(0.5, max_run + 1.5)
        p_hat_plot = stats_out.get("H1_runlength", {}).get("p_geometric_mle", 0.5)
        bs_p_plot  = stats_out.get("H1_runlength", {}).get("bootstrap_ks_p", float("nan"))
        cs_p_plot  = stats_out.get("H1_runlength", {}).get("chisq_p",         float("nan"))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.hist(all_anchored_rl, bins=bins, density=True, alpha=0.75,
                color="#e74c3c",
                label=f"Observed (n={len(all_anchored_rl)} runs)")
        ks_x   = np.arange(1, max_run + 1)
        geom_y = [geometric_pmf(k, p_hat_plot) for k in ks_x]
        ax.plot(ks_x, geom_y, "k--", lw=1.5,
                label=f"Geometric null (p={p_hat_plot:.2f})")
        ax.set_xlabel("Run length (consecutive anchored turns)")
        ax.set_ylabel("Density")
        bs_str = f"{bs_p_plot:.4f}" if not math.isnan(bs_p_plot) else "n/a"
        cs_str = f"{cs_p_plot:.4f}" if not math.isnan(cs_p_plot) else "n/a"
        ax.set_title(
            f"H1: Anchored run-length vs geometric null\n"
            f"Bootstrap KS p = {bs_str}   chi2 p = {cs_str}"
        )
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        if all_exploring_rl:
            ax.hist(all_exploring_rl,
                    bins=np.arange(0.5, max(all_exploring_rl) + 1.5),
                    density=True, alpha=0.75, color="#2ecc71",
                    label=f"Exploring runs (n={len(all_exploring_rl)})")
        ax.set_xlabel("Run length (consecutive exploring turns)")
        ax.set_ylabel("Density")
        ax.set_title("Exploring run-length distribution (reference)")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "runlength_dist.png"), dpi=150)
        plt.close(fig)
        print("saved runlength_dist.png")

    # ── Plot 3: Faith distribution histogram ─────────────────────────────────
    if all_faith_values:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(all_faith_values, bins=20, density=True, alpha=0.75,
                color="#3498db", edgecolor="white")
        ax.axvline(0.5, color="k", lw=1.2, ls="--", label="midpoint (0.5)")
        ax.set_xlabel("Per-turn faithfulness score")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Distribution of per-turn faithfulness "
            f"(n={len(all_faith_values)} obs)"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "faith_distribution.png"), dpi=150)
        plt.close(fig)
        print("saved faith_distribution.png")

    # ── Plot 4: frac_anchored scatter + H3 pie ───────────────────────────────
    valid = df[df["is_correct"].notna()].copy()
    if len(valid) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        colors = ["#2ecc71" if c else "#e74c3c" for c in valid["is_correct"]]
        sizes  = valid["max_anchored_run"] * 20 + 20
        ax.scatter(valid["frac_anchored"], valid["n_turns"],
                   c=colors, s=sizes, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.5)
        for _, row in valid.iterrows():
            ax.annotate(row["task_id"].split("/")[-1],
                        (row["frac_anchored"], row["n_turns"]),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=6, alpha=0.6)
        ax.axvline(0.05, color="grey", lw=1, ls=":", alpha=0.5,
                   label="5% threshold (H3 minimum)")
        r_info = stats_out.get("H2_frac_anchored_vs_correct", {})
        ax.set_xlabel("Fraction of turns in anchored mode")
        ax.set_ylabel("Conversation length (# turns)")
        ax.set_title(
            f"Anchored fraction vs length  (green=correct, red=wrong)\n"
            f"marker size ~ max anchored run length\n"
            f"r={r_info.get('r', float('nan')):.2f}  "
            f"p={r_info.get('p', float('nan')):.3f}  n={r_info.get('n', '?')}"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1]
        h3_info = stats_out.get("H3_bimodality", {})
        fa = h3_info.get("frac_anchored", 0.13)
        fe = h3_info.get("frac_exploring", 0.87)
        ax.pie([fe, fa],
               labels=[f"Exploring ({fe:.0%})", f"Anchored ({fa:.0%})"],
               colors=["#2ecc71", "#e74c3c"], startangle=90,
               wedgeprops={"width": 0.5})
        vp = h3_info.get("variance_chisq_p", float("nan"))
        vp_str = f"{vp:.4f}" if not math.isnan(float(vp) if vp != "nan" else float("nan")) else "n/a"
        ax.set_title(
            f"H3: Both modes present (N={len(df)} conversations)\n"
            f"Variance test p = {vp_str}"
        )

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "frac_anchored_scatter.png"),
                    dpi=150)
        plt.close(fig)
        print("saved frac_anchored_scatter.png")

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
