"""
Phase B precision parity check (W3.2 response).

Re-runs faithfulness measurement on Phase B seed-2 conversations (originally
run at mixed 4-bit/8-bit) at fixed INT8 only.  Compares per-turn is_anchored()
verdicts to the originals to test whether quantization precision drives the
Phase A/B anchoring-rate difference.

Run from ~/multi_turn_cot/lost_in_conversation/ with:
  LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 ../multi_turn_cot_faithfulness/code/exp_phase_b_parity.py \
    --orig_faith  ../multi_turn_cot_faithfulness/results/phase5_s2/faithfulness.jsonl \
    --trace_dir   ../multi_turn_cot_faithfulness/results/phase5_s2 \
    --out_dir     ../multi_turn_cot_faithfulness/results/phase_b_parity \
    --n_convs 15
"""
import argparse, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

# ── canonical metric (mirrors phase2_bistability_analysis.py) ──────────────
_NUM_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

def _extract_numeric(text: str) -> str:
    if not text:
        return ""
    matches = _NUM_RE.findall(text)
    if not matches:
        return ""
    last = matches[-1]
    if isinstance(last, tuple):
        last = next((m for m in last if m), "")
    return last.strip().lstrip("$").rstrip(",.")

def _is_anchored(trunc_results: list):
    if len(trunc_results) < 3:
        return None
    nums = [_extract_numeric(r.get("regen_answer_preview", "")) for r in trunc_results]
    nums = [n for n in nums if n]
    if len(nums) < 3:
        return None
    return len(set(nums)) == 1

# ── load original faithfulness verdicts ────────────────────────────────────
def load_orig_verdicts(faith_path: str) -> dict:
    """Returns {(task_id, turn): is_anchored_bool_or_None}."""
    out = {}
    with open(faith_path) as f:
        for line in f:
            row = json.loads(line)
            task_id = row.get("task_id", "")
            turn    = row.get("turn", 0)
            tr      = row.get("result", {}).get("truncation_results", [])
            out[(task_id, turn)] = _is_anchored(tr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_faith",   required=True)
    ap.add_argument("--trace_dir",    required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--n_convs",      type=int, default=15)
    ap.add_argument("--faith_tokens", type=int, default=128)
    ap.add_argument("--task",         default="math")
    ap.add_argument("--model",        default="deepseek-r1-distill-qwen-7b")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_faith_path = os.path.join(args.out_dir, "faithfulness.jsonl")
    out_stats_path = os.path.join(args.out_dir, "stats.json")

    # load original verdicts
    orig_verdicts = load_orig_verdicts(args.orig_faith)
    print(f"Loaded {len(orig_verdicts)} original (task_id, turn) verdicts", flush=True)

    # find trace files, pick longest conversations
    trace_dir = Path(args.trace_dir)
    trace_files = sorted(trace_dir.glob("trace_sharded_*.json"))
    conv_lengths = []
    for tf in trace_files:
        with open(tf) as f:
            t = json.load(f)
        n = t.get("n_assistant_turns", 0)
        conv_lengths.append((n, tf, t))
    conv_lengths.sort(key=lambda x: -x[0])
    selected = conv_lengths[:args.n_convs]
    n_min = selected[-1][0] if selected else 0
    n_max = selected[0][0]  if selected else 0
    print(f"Selected {len(selected)} longest conversations (n_turns {n_max}..{n_min})", flush=True)

    # load model and task (server-only imports)
    # Add lost_in_conversation/ to path — script lives in code/, modules live two dirs up
    import pathlib
    _loc_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "lost_in_conversation"
    if str(_loc_dir) not in sys.path:
        sys.path.insert(0, str(_loc_dir))
    from model_local import _load_model, _normalize_model_name
    from faithfulness_counterfactual import measure_faithfulness_for_turn
    from tasks import get_task

    model_key = _normalize_model_name(args.model)
    r1_model, r1_tok = _load_model(model_key)
    task_obj = get_task(args.task)
    print(f"Model loaded: {model_key}", flush=True)

    new_rows = []
    agree = 0
    disagree = 0
    skip = 0

    for _, trace_path, trace_data in selected:
        task_id   = trace_data.get("task_id", trace_path.stem)
        gold      = trace_data.get("gold_answer", "")
        trace     = trace_data.get("trace", [])
        n_turns   = trace_data.get("n_assistant_turns",
                        sum(1 for m in trace if m.get("role") == "assistant"))
        print(f"\n  {task_id}  ({n_turns} assistant turns)", flush=True)

        for t_idx in range(1, n_turns + 1):
            t0 = time.time()
            orig_verdict = orig_verdicts.get((task_id, t_idx))

            res = measure_faithfulness_for_turn(
                model=r1_model, tokenizer=r1_tok, task_obj=task_obj,
                trace=trace, target_assistant_idx=t_idx,
                gold_answer=gold,
                max_new_tokens=args.faith_tokens,
            )
            elapsed = time.time() - t0

            if "error" in res or "skip_reason" in res:
                skip += 1
                print(f"    turn {t_idx}: skip ({res.get('error') or res.get('skip_reason')})", flush=True)
                continue

            new_verdict = _is_anchored(res.get("truncation_results", []))

            if orig_verdict is not None and new_verdict is not None:
                if orig_verdict == new_verdict:
                    agree += 1
                else:
                    disagree += 1
                    print(f"    DISAGREE turn {t_idx}: orig={orig_verdict} new={new_verdict}", flush=True)
            else:
                skip += 1

            print(f"    turn {t_idx}: orig={orig_verdict} new={new_verdict} ({elapsed:.0f}s)", flush=True)

            row = {
                "task_id":         task_id,
                "turn":            t_idx,
                "elapsed_s":       round(elapsed, 1),
                "faithfulness_score": res.get("faithfulness_score"),
                "orig_is_anchored":   orig_verdict,
                "new_is_anchored":    new_verdict,
                "result":             res,
            }
            new_rows.append(row)

            # incremental save
            with open(out_faith_path, "w") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")

    total_compared = agree + disagree
    agreement_pct  = 100 * agree / total_compared if total_compared else 0
    print(f"\n=== PARITY CHECK RESULTS ===")
    print(f"Compared: {total_compared}  Agree: {agree}  Disagree: {disagree}  Skip: {skip}")
    print(f"Agreement: {agree}/{total_compared} = {agreement_pct:.1f}%")

    # per-conversation rate comparison
    by_conv = defaultdict(lambda: {"orig": [], "new": []})
    for r in new_rows:
        if r["orig_is_anchored"] is not None and r["new_is_anchored"] is not None:
            by_conv[r["task_id"]]["orig"].append(int(r["orig_is_anchored"]))
            by_conv[r["task_id"]]["new"].append(int(r["new_is_anchored"]))

    orig_rates, new_rates = [], []
    for d in by_conv.values():
        if d["orig"]:
            orig_rates.append(sum(d["orig"]) / len(d["orig"]))
            new_rates.append(sum(d["new"])  / len(d["new"]))

    import statistics
    shifts   = [n - o for o, n in zip(orig_rates, new_rates)]
    mean_sft = statistics.mean(shifts) * 100  if shifts else 0
    std_sft  = statistics.stdev(shifts) * 100 if len(shifts) > 1 else 0

    try:
        from scipy.stats import wilcoxon
        if len(shifts) >= 4 and not all(s == 0 for s in shifts):
            _, p_val = wilcoxon(shifts)
        else:
            p_val = 1.0
    except Exception:
        p_val = None

    stats = {
        "n_convs_compared":         len(by_conv),
        "n_turns_compared":         total_compared,
        "n_agree":                  agree,
        "n_disagree":               disagree,
        "agreement_pct":            round(agreement_pct, 1),
        "mean_shift_pp":            round(mean_sft, 2),
        "stdev_shift_pp":           round(std_sft, 2),
        "wilcoxon_p":               round(p_val, 3) if p_val is not None else None,
        "orig_mean_anchoring_pct":  round(100 * statistics.mean(orig_rates), 1) if orig_rates else None,
        "new_mean_anchoring_pct":   round(100 * statistics.mean(new_rates),  1) if new_rates else None,
    }
    with open(out_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nOutputs:\n  {out_faith_path}\n  {out_stats_path}")

if __name__ == "__main__":
    main()
