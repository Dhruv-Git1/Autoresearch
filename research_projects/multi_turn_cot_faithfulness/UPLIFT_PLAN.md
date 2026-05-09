# Uplift Plan: Getting to >60% Acceptance Probability

**Current estimated probability:** 25–35% (CAISc 2026)  
**Target:** >60%  
**Constraint:** Infinite compute, infinite time  
**Reviewer:** Strict CAISc / NeurIPS Safe-GenAI standard

---

## The Honest Diagnosis First

The paper has one genuinely confirmed result (H3) and one failed result (H1). The title and framing claim "mode-switching," which requires H1. What you actually proved is "conversations differ in anchoring rate" — a real but much softer finding. Every structural change below is aimed at either (a) actually confirming H1, or (b) making the confirmed claims so strong that H1 doesn't matter. There are also three statistical artifacts that a reviewer who reads carefully will catch and that must be fixed regardless.

---

## Priority 1 — Fatal Weaknesses (Fix These or Stay Below 40%)

These are the issues that will cause an outright rejection from any reviewer who digs in. Fix all of them.

---

### Fix 1.1 — Confirm H1 with N ≥ 200 Long Conversations

**The problem:**  
H1 (run-length persistence — the core evidence for "mode-switching") is inconclusive at N=20 because you only have 25 anchored runs. A chi-square GOF test on 25 data points with 4 bins has almost no power. You need ~100 anchored runs for the distribution to be stable. Given that ~14% of turns are anchored and ~1.8 mean run length, and that long conversations contribute most anchored turns, the formula is roughly:

```
anchored_runs_needed = 100
turns_per_long_conv ≈ 30 (with max_shards=20)
anchored_turns_per_long_conv ≈ 30 × 0.20 = 6
anchored_runs_per_long_conv ≈ 6 / 1.8 = 3.3 runs
conversations_needed = 100 / 3.3 ≈ 30 long conversations
→ run at least 150 conversations total (mix of lengths)
  to get ≥30 long ones (>20 turns) = ≥100 anchored runs
```

**What to run:**  
```bash
# Use max_shards=20 to force long conversations
# Use many seeds, --exclude_dirs to avoid overlap
LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 phase2_batch_runner.py \
  --n_samples 60 --max_shards 20 --faith_tokens 128 \
  --seed 11111 \
  --exclude_dirs [all prior phase dirs] \
  --out_dir ../multi_turn_cot_faithfulness/results/phase5_long

# Repeat with seeds 22222, 33333 for 180 total conversations
```

**Why this matters for acceptance:**  
A reviewer who sees "H1 inconclusive" in the abstract will mark it weak immediately. With 100+ anchored runs, the bootstrap has actual power. If H1 clears p<0.05 with proper run-length persistence, your title becomes defensible. If it still doesn't clear, you now have a definitive sample-size-independent negative result, which is also publishable — but you'll need to reframe around H3 only.

---

### Fix 1.2 — Run the H3 Variance Test Within Length Bins

**The problem:**  
The current chi-square test (χ²=90.2, df=19, p<0.001) mixes conversations of 3 turns with conversations of 43 turns under a single Bernoulli(p=0.137) null. Short conversations almost certainly have 0 anchored turns by chance, and long conversations are expected to have many. The test confounds "conversation length varies" with "anchoring rate varies." A reviewer who checks the heatmap (which you now include) will notice that the bottom 11 conversations all have 0% anchored, are all short, and are dragging the chi-square statistic by looking "below the null." This is length effect, not mode-switching evidence.

**What to do:**  
Add a `h3_variance_test_within_bins` function to `phase2_bistability_analysis.py` that runs the variance test separately for short / medium / long bins and reports three separate chi-square statistics. This takes 20 lines of code. If H3 is significant **within the long-conversation bin** (where anchoring is high and variable), that is a far more credible result than the pooled test.

```python
def h3_variance_test_within_bins(conv_records):
    bins = {
        "short_leq10":  [r for r in conv_records if r["n_turns"] <= 10],
        "medium_10_20": [r for r in conv_records if 10 < r["n_turns"] <= 20],
        "long_gt20":    [r for r in conv_records if r["n_turns"] > 20],
    }
    results = {}
    for name, records in bins.items():
        if len(records) < 3:
            results[name] = {"n": len(records), "verdict": "insufficient data"}
            continue
        stat, df, p_val, p_global = h3_variance_test(records)
        results[name] = {
            "n_conv": len(records),
            "chi2": round(stat, 3),
            "df": df,
            "p": round(p_val, 4),
            "p_global": round(p_global, 3),
        }
    return results
```

Report these numbers in a new Table 2 in the paper. If H3 holds within the long-conversation bin with N≥30 such conversations (from Fix 1.1), your statistical argument is clean and length-independent.

**Why this matters for acceptance:**  
A reviewer who reads Section 3.3 ("H3: Both Modes Substantially Present") and then looks at Section 3.6 ("Length-Stratified Analysis") will immediately ask: "did you run H3 within bins?" If the answer is no, they will assume the H3 result is a length artifact. This fix costs nothing and removes a fatal objection.

---

### Fix 1.3 — Demolish the Repetition Confound Properly

**The problem:**  
61.4% of "anchored" turns simply repeat the prior turn's answer. The paper sets an arbitrary threshold of 70% and says "below threshold, fine." This is wrong. The threshold has no statistical or theoretical grounding. A reviewer will ask: "what is the baseline repetition rate for non-anchored turns?" If the model repeats the prior answer 50% of the time when exploring, then 61.4% repetition in anchored mode is only marginally elevated and not evidence of a distinct mode.

**What to measure:**  
Add to the analysis script:

```python
def repetition_rate_by_mode(by_tid):
    """Repetition rate separately for anchored vs exploring turns."""
    anchored_rep, anchored_total = 0, 0
    exploring_rep, exploring_total = 0, 0
    for tid, turn_rows in by_tid.items():
        prev_answer = None
        for r in turn_rows:
            a = is_anchored(r)
            ans = get_primary_answer(r)
            if a is True and ans is not None and prev_answer is not None:
                anchored_total += 1
                if ans == prev_answer:
                    anchored_rep += 1
            elif a is False and ans is not None and prev_answer is not None:
                exploring_total += 1
                if ans == prev_answer:
                    exploring_rep += 1
            if ans is not None:
                prev_answer = ans
    return {
        "anchored_rep_rate":  anchored_rep / anchored_total if anchored_total else float("nan"),
        "exploring_rep_rate": exploring_rep / exploring_total if exploring_total else float("nan"),
        "anchored_total":     anchored_total,
        "exploring_total":    exploring_total,
    }
```

You expect anchored repetition rate ≈ 61% and exploring repetition rate ≈ 15–30%. If the difference is large (≥2×), the confound is limited: anchored turns repeat at much higher rates than exploring turns, which is itself evidence of a distinct mode. If they are similar, you have a serious confound and need to restructure the metric.

**Additionally:** Run a "committed-but-exploring" ablation. Find turns where the model gives the same answer as the prior turn *but* the 5 truncation levels disagree. These are "repetitions without anchoring" — the baseline for natural answer repetition under no-faithfulness-change. Compare their rate to the anchored repetition rate. If anchored turns repeat far more than "same answer but exploring" turns, anchoring is a real phenomenon on top of natural repetition inertia.

**Why this matters:**  
Without this data, the paper can be dismissed with one sentence: "61% of anchored turns are just the model repeating itself; this is conversational inertia, not post-hoc rationalization." With it, you have a clean counter-argument.

---

### Fix 1.4 — Report Commitment-Alignment Numbers in the Paper

**The problem:**  
Section 3.5 claims to report Pearson correlation and mean offset between first_anchored_turn and commitment_turn, and says results are in `bistability_stats.json`. The key `commitment_alignment` is **absent from the JSON.** The analysis code exists and is correct, but the outputs were not saved or not routed to the right file.

**What to do:**  
1. Re-run the bistability analysis on the full N=24+ dataset — the code already has `commitment_alignment()`.  
2. Verify `commitment_alignment` appears in the output JSON.  
3. Copy the actual numbers into Section 3.5 of the paper. The paper currently says "are reported in bistability_stats.json" — replace that with "r=X, p=Y, mean offset=Z turns."  
4. If n<5 paired conversations (insufficient for correlation), report descriptive stats instead: "of N conversations with ≥1 anchored turn, M/N had a commitment turn within 2 turns of the first anchored turn."

**Why this matters:**  
The alignment between anchored mode and premature commitment is the most interesting and safety-relevant claim in the paper. Right now it is asserted qualitatively for sample 965 only. Verifying it systematically and actually reporting the correlation in the text is the difference between "anecdote" and "finding."

---

## Priority 2 — Generalizability (Required to Cross 60%)

Even with all Priority 1 fixes, a reviewer can still reject with: "interesting finding in one quantized 7B model on math tasks, not generalizable." With infinite compute, there is no excuse not to fix this.

---

### Fix 2.1 — Replicate on ≥3 Models Across Two Architecture Families

**Why:**  
A finding in one model is a curiosity. A finding across architectures is a phenomenon. The CAISc and NeurIPS Safe-GenAI audience cares about whether this affects deployed systems. o3, Claude, and Gemini Thinking are deployed systems.

**Which models to run (ranked by priority):**

| Model | Why This Model | How to Run |
|---|---|---|
| **QwQ-32B** (full precision or 4-bit) | Same training paradigm as DeepSeek-R1, bigger, more capable — tests scale within family | HuggingFace, 4-bit on your RTX 6000 Ada |
| **DeepSeek-R1-Distill-Qwen-7B at full FP16** | Tests whether 8-bit quantization is causing the anchoring artifacts (quantization confound) | Load without BitsAndBytes, needs ~14GB VRAM |
| **Llama-3.1-8B-Instruct** (no explicit CoT) | Tests whether the phenomenon requires explicit `<think>` blocks or occurs in any multi-turn model | HuggingFace, small and fast |
| **o3-mini via API** | Deployed reasoning model; if anchoring occurs in o3-mini, safety claim is directly actionable | OpenAI API, expensive but fast |
| **Claude-3.5-Haiku via API** | Tests a different training paradigm entirely | Anthropic API |

**Minimum for a strong paper:** QwQ-32B + DeepSeek FP16. This takes your single-model weakness off the table.

**Practical target:** Run N=50 conversations per model on the same GSM8K shards (using the same --exclude_dirs logic to get different samples). Compare anchored fraction and run-length distributions across models. A table like:

| Model | N conv | Anchored % | H1 bootstrap p | H3 var. p |
|---|---|---|---|---|
| DeepSeek-R1-7B (8-bit) | 150 | 14% | 0.03 | <0.001 |
| DeepSeek-R1-7B (FP16) | 50 | X% | X | X |
| QwQ-32B (4-bit) | 50 | X% | X | X |
| o3-mini | 50 | X% | X | X |

If anchoring appears in all four models, you have a general phenomenon. If it disappears in FP16 DeepSeek, you have a quantization artifact paper (publishable but different). Either way, you know.

---

### Fix 2.2 — Run H2 on HumanEval Coding Conversations

**The problem:**  
H2 (anchored fraction predicts conversation failure) is the most actionable safety hypothesis — if anchoring predicts failure, you can detect and intervene. It is currently "reserved for future work" because GSM8K correctness labels are unreliable in multi-turn settings. With infinite compute, "future work" is "this week."

**What to do:**  
The `microsoft/lost-in-conversation` benchmark includes HumanEval problems split into shards. HumanEval has clean binary outcomes (code compiles and passes tests, or doesn't). Replace the GSM8K setup with HumanEval:

1. Set up the sharded HumanEval benchmark (same repo, different problem file)
2. Run N=100 conversations with DeepSeek-R1-7B (or QwQ-32B)
3. At the end of each conversation, extract the model's code, run it against the HumanEval test suite, get a binary pass/fail
4. Compute point-biserial r(frac_anchored, is_correct)

If r < 0 and p < 0.05, H2 is confirmed: anchored mode predicts failure. This transforms the paper from "we observe a curious phenomenon" to "we have a predictive safety signal for conversation failure." That is a qualitatively different contribution.

**Expected result:** You should expect anchoring to predict failure, because anchored mode = premature commitment = wrong answer persisted. But you don't know until you test it, and a paper that empirically confirms a safety signal is orders of magnitude more impactful than one that theorizes it.

---

### Fix 2.3 — Test on a Non-Math Task Domain

**Why:**  
A reviewer will ask "does this only happen in math because math has unique right/wrong answers?" The answer is: probably not, but you don't know. Adding one additional domain costs relatively little compute.

**Best choice: Multi-turn factual QA (TriviaQA or Natural Questions split into shards)**  
- Split a 5-part factual question across 5 turns (e.g., "What year was X founded? [turn 1 context] ... [turn 2 context] ...")  
- Run faithfulness measurement the same way  
- This tests whether anchoring is a math-specific artifact or a general multi-turn reasoning failure mode

**Minimum: N=50 conversations on one non-math domain.** The GSM8K result becomes "Experiment 1" and this becomes "Experiment 2." A two-domain paper is substantially stronger than a single-domain paper.

---

## Priority 3 — Mechanistic Understanding (Will Push You to 70%+)

These experiments elevate the paper from "we measured something" to "we understand what's happening." They are not required for acceptance at a workshop but will strongly differentiate you.

---

### Fix 3.1 — Logit-Level Analysis: What Changes at the First Anchored Turn?

**The idea:**  
If anchoring is a real mode switch, something must change in the model's internal state at the turn boundary. The most accessible observable is the logit distribution over the answer tokens. Specifically:

At exploring turns: the model's token probabilities for different numeric answers should be spread across several values (high entropy → model is uncertain)  
At anchored turns: one numeric token should dominate across all 5 truncation levels (low entropy → model is committed)

**What to measure:**  
Modify `faithfulness_counterfactual.py` to extract, for each truncation level at each turn, the softmax probability of the top-5 output tokens at the first generated digit position. Compute entropy across these probabilities.

Expected finding: entropy drops sharply at the transition to anchored mode, and stays low. This is a mechanistic signature of commitment rather than post-hoc rationalization.

**Why this matters:**  
A mechanistic finding upgrades the paper from an observational study to an explanatory one. It also directly answers the "repetition confound" objection: if the model's output distribution has collapsed to one token, it's not repeating by inertia — its computation is genuinely locked.

---

### Fix 3.2 — "Restart" Intervention Experiment

**The idea:**  
The Discussion section proposes "summarise-and-restart" as a mitigation: when a conversation enters anchored mode, summarize the conversation state and start fresh. With infinite compute, you can actually test this.

**Design:**  
1. Run conversations until a conversation enters anchored mode (detected online by checking if 2 consecutive turns are anchored)  
2. At that point, summarize the conversation state (using Qwen2.5-7B as summarizer: "The user has told you: [shard 1], [shard 2], ... Your previous answer was X. Here is a summary: ...")  
3. Restart the assistant with this summary as the first user message  
4. Measure whether the model exits anchored mode and whether final accuracy improves  

**Control group:** Same conversations continued without intervention.

**Why this matters:**  
A paper that identifies a failure mode AND tests a mitigation is substantially more publishable than one that only identifies it. A positive mitigation result ("restart reduces anchored fraction from 20% to 4% and improves final accuracy by 15 pp") is a headline result that reviewers will remember.

---

### Fix 3.3 — Attention-Based Early Detection

**The idea:**  
If anchored mode has a mechanistic signature in the model's internals, it may be detectable *before* the turn completes (i.e., from early tokens of the thinking block, not from the answer). This would enable real-time intervention.

**What to measure:**  
At each turn, extract the model's attention entropy across heads in layer 16-24 (middle layers) during the first 50 tokens of generation. Compute a simple anomaly score (deviation from the running mean of attention entropy across prior turns). 

Test whether this score at turn T predicts anchored classification at turn T. If it does, you have an *early warning system* — a safety monitor that doesn't require waiting for the full generation.

**Why this matters:**  
This experiment directly addresses the safety monitoring use case. A lightweight attention-based monitor that can flag anchored mode in real time is a concrete artifact, not just a conceptual contribution.

---

## Priority 4 — Statistical and Writing Fixes (Polish, but Reviewers Notice)

---

### Fix 4.1 — Fix the Mixed-Effects Model for H3

Replace the current chi-square variance test with a proper mixed-effects logistic regression:

```
anchored_turn ~ 1 + turn_index + conversation_length + (1 | conversation_id)
```

Where `(1 | conversation_id)` is a random intercept per conversation. The intraclass correlation coefficient (ICC) from this model directly measures how much of the variance in anchoring is at the conversation level vs the turn level. An ICC of 0.15–0.30 would directly support the "conversation-level phenomenon" claim with no length confound. Use `statsmodels` MixedLM or R's `lme4`.

This is a single analysis call and makes the H3 statistical argument bulletproof.

---

### Fix 4.2 — Fix the `jin2025` Citation Key

In `references.bib`, the entry:
```bibtex
@article{jin2025,
  author = {Chen, Yanda and others},
```

The key is `jin2025` but the first author is Chen, Yanda. The key should be `chen2025`. Change the key in the bib file and update all `\citet{jin2025}` / `\citep{jin2025}` references in the paper. Minor, but reviewers who look up the reference will notice.

---

### Fix 4.3 — Retitle the Paper to Match What Was Proved

**Current title:** "Lost in Reasoning: Mode-Switching Chain-of-Thought Faithfulness During Multi-Turn Conversational Derailment"

**Problem:** "Mode-switching" implies H1 (persistent modes). H1 is currently inconclusive.

**If H1 is confirmed after Fix 1.1:** Current title is fine.  

**If H1 remains inconclusive even at N=200:** Retitle to something like:
> "Conversation-Level Variation in Chain-of-Thought Faithfulness: Evidence for Anchored and Exploring Modes in Multi-Turn Derailment"

This is not a weaker title — it is an accurate title. An accurate title that matches confirmed results will not frustrate a reviewer; an oversold title will.

---

### Fix 4.4 — Add a Power Analysis Section

Add a 5-line appendix section explaining:
- How many anchored runs are needed to achieve 80% power for H1 (answer: ~80–100 runs, based on the observed run-length distribution)
- What N conversations at max_shards=20 would give that many runs (answer: ~30 long conversations → ~150 total)
- This is an honest statement of the sample size requirement, which reviewers appreciate

---

### Fix 4.5 — Report Commitment-Alignment in-text (Link to Fix 1.4)

As noted above, the `commitment_alignment` key is missing from the JSON. Fix the code to output it, re-run, and put actual numbers in Section 3.5. Do not leave it as "reported in bistability_stats.json."

---

### Fix 4.6 — Remove the stale `bistability_final/bistability_stats.json`

The `bistability_final/` directory still contains the old invalid KS result (`ks_p: 0.0`). This directory is the "final" one by name. If a reviewer or replicator runs the code and checks this file, they will find a contradictory result. Either:
- Delete `bistability_final/` and rename `bistability_p2p3p4/` to `bistability_final/`
- Or add a `README.md` to `bistability_final/` explaining it is a historical artifact

---

## Summary Roadmap

Do these in order. Each level unlocks the next acceptance threshold.

```
Phase A (2 weeks, fixes fatal flaws → 40% floor)
├── Fix 1.2  Length-stratified H3 test         [20 lines of code + re-run]
├── Fix 1.3  Repetition confound by mode       [30 lines of code + re-run]
├── Fix 1.4  Commitment alignment numbers      [re-run analysis + update paper]
└── Fix 4.2  Fix jin2025 → chen2025 in bib    [2 lines]

Phase B (4 weeks, confirms H1 → 50%)
└── Fix 1.1  N=150–200 long conversations on DeepSeek-R1-7B
             (will either confirm H1 or definitively close it)

Phase C (6 weeks, adds generalizability → 60%)
├── Fix 2.1  QwQ-32B replication (N=50)
├── Fix 2.1  DeepSeek FP16 (quantization check, N=50)
└── Fix 2.2  HumanEval H2 test (N=100 conversations)

Phase D (8 weeks, adds mechanistic story → 70%+)
├── Fix 3.2  Restart intervention experiment
├── Fix 3.1  Logit entropy analysis
└── Fix 2.3  Non-math domain replication

Phase E (parallel with all above)
├── Fix 4.1  Mixed-effects logistic regression for H3
├── Fix 4.3  Retitle if H1 still inconclusive after Phase B
└── Fix 4.4  Power analysis appendix
```

---

## What the Paper Looks Like After All Fixes

- **N=200 conversations, ~1,500 faithfulness observations** across 3 models and 2 task domains  
- **H1 confirmed** (p<0.05, >100 anchored runs, run-length persistence demonstrated)  
- **H3 confirmed within length bins** (no length confound objection possible)  
- **H2 confirmed on HumanEval** (anchored fraction predicts conversation failure)  
- **Repetition confound explained** (anchored turns repeat at 3× the rate of exploring turns, not 1.2×)  
- **Restart intervention tested** (positive or negative, either is publishable)  
- **3 models** (H1 consistent across architectures) or (H1 is model-specific — also publishable)  
- **Mixed-effects H3 with ICC** (conversation explains 25% of variance in anchoring)  

At that point, this is a **NeurIPS main track submission**, not just a workshop paper. Acceptance probability at CAISc 2026 would be >75%. Acceptance probability at NeurIPS Safe-GenAI would be >60%.

---

*Written as a strict external reviewer critique. All numbers (N, p-values, sample sizes) are estimates based on the current dataset statistics and standard power analysis conventions.*
