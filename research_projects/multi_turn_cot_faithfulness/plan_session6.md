# Session 6 Plan — Resolving the H2 Safety Hypothesis Without Outcome Labels

**Status**: Handoff document for a fresh Claude session with NO prior context. Read this end-to-end before starting work.

**Created**: 2026-05-11 (end of Session 5)
**Owner**: Dhruv (BITS Pilani, f20221683@pilani.bits-pilani.ac.in)
**Project root**: `D:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\`

---

## 0. TL;DR for the new session

You are continuing a research paper about **Chain-of-Thought (CoT) faithfulness in multi-turn derailing conversations**. The paper is at `paper/paper.tex` (27 pages, compiles cleanly). The dataset is N=67 conversations / 1,289 turn-level observations from sharded GSM8K. The model is DeepSeek-R1-Distill-Qwen-7B.

There is one unresolved problem and **five concrete tasks** to address it. The problem is that **H2 (the safety hypothesis: anchoring rate predicts task failure) is statistically underpowered** (r=−0.242, p=0.291, N=21). The paper's safety framing currently overweights H2.

After a deep literature review at the end of Session 5, we found that **H2 was the wrong load-bearing hypothesis** and the safety claim can be made *more* rigorously via process-level / monitor-evasion arguments that don't require outcome labels. The five tasks operationalize this reframing plus three new empirical analyses, all from existing data.

**Important meta-instruction**: Do not propose your own framing. The framing below was decided after careful work. Execute the five tasks; do not relitigate strategy unless you find a real bug.

---

## 1. Project background (what this paper is about)

**Core claim**: In multi-turn conversations, CoT traces increasingly fail to causally drive answers as conversations lengthen. We call turns where the CoT is causally non-necessary "anchored" turns.

**Measurement**: For each turn, regenerate the model's answer with the `<think>...</think>` block truncated to {0%, 25%, 50%, 75%, 100%}. If all five truncations produce the same numeric answer, the turn is "anchored" (CoT is post-hoc). Otherwise "exploring" (CoT is causally active). This is Lanham et al.'s counterfactual deletion methodology, applied per-turn in a multi-turn setting.

**Dataset**: 67 conversations from Microsoft's `lost-in-conversation` benchmark — sharded GSM8K problems where shards are revealed turn-by-turn. Conversations average ~20 turns each (1,289 turns total).

**Key empirical findings (all confirmed)**:
- **Length-anchoring gradient**: short 7.2% → medium 12.8% → long 28.4% anchored (3.9× ratio)
- **H3 (between-conversation variance)**: χ²=236.9, df=66, p≈0 — anchoring clusters within conversations
- **ICC = 0.152** — 15% of anchoring variance is between-conversation
- **Repetition confound closed**: anchored 75.8% vs exploring 45.6% — answer inertia accounts for some but not all
- **Exp 1 (logprobs)**: NULL — token-level confidence is NOT diagnostic of mode
- **Exp 4 (INT4 quant robustness)**: 100% per-turn agreement INT4 vs INT8 (184/184) — quantization confound closed
- **Exp 5 (predictor)**: GroupKFold logistic regression AUC=0.710 on 7 surface features

**Findings that are NULL or inconclusive (don't overclaim)**:
- **H1 (run-length persistence)**: bootstrap p=0.521 — INCONCLUSIVE. Geometric null fits the data.
- **H2 (anchoring → failure)**: r=−0.242, p=0.291, N=21 — UNDERPOWERED. This is the problem this plan addresses.

---

## 2. The H2 problem (full context)

### 2.1 The reviewer critique (verbatim)

> "H2 is the load-bearing safety hypothesis and it remains unconfirmed. The paper's central policy implication — that CoT monitors calibrated on short conversations overestimate reliability on long ones by ~4× — depends on anchoring predicting downstream failure. H2 gives r=−0.242, p=0.291 on N=21 labelled conversations. The paper correctly diagnoses this as underpowered, not falsified. But the safety framing in the discussion (the 'monitoring blind spot' quantification, the 8.5 post-hoc turns in a 30-turn conversation) is presented with a confidence that H2 does not yet support. Showing that 28% of turns are CoT-invariant in long conversations is interesting; showing that those turns are the ones that matter for safety requires the H2 link. Without it, this is a faithfulness measurement paper, not a safety monitoring paper."

### 2.2 Why this critique is partially right and partially wrong

**Right**: The paper's Discussion currently frames the 4× monitoring miscalibration as if it directly implies safety harm. That step needs H2.

**Wrong**: The 4× ratio itself is a direct empirical observation (H3 + length gradient). The safety claim CAN be made without H2, as a **process-level monitorability failure**, which is what most CoT-monitoring papers actually argue. We just have to make that argument explicitly.

### 2.3 The literature-grounded reframing (this is the key insight — read carefully)

Three discoveries from a thorough literature search:

1. **Lanham et al. 2023 found R²=0.74 between faithfulness and accuracy — and TREATED IT AS A PROBLEM.** A faithfulness metric that strongly correlates with accuracy is suspicious: it may be measuring task difficulty rather than faithfulness. Our weak H2 correlation (r=−0.242) is therefore **a feature, not a flaw** — it suggests anchoring is orthogonal to task difficulty, i.e., measures something distinct from competence.

2. **Korbak et al. 2025 ("Chain of Thought Monitorability") defines monitorability via three archetypes: Process, Outcome, and Intervention.** Our counterfactual-deletion measure is squarely a Process+Intervention property. Outcome (H2) is supplementary. The Korbak framework explicitly supports process-only monitorability claims.

3. **Meek et al. 2025 explicitly proposes monitorability metrics without ground-truth labels** (verbosity + faithfulness). There is published precedent for the no-ground-truth approach.

**The new safety argument** (this is what the Discussion should foreground):

> Anchored turns are not just instances of CoT non-faithfulness; they are turns where the CoT-answer surface alignment is produced by means OTHER than the CoT driving the answer. By construction, an anchored turn is one where ALL truncations of the CoT yield the same conclusion — meaning it would pass any CoT-consistency monitor *maximally*. A safety monitor checking "does the reasoning support the conclusion?" would PASS every anchored turn. The 4× growth in anchoring with conversation length means this monitor-evasion regime is exactly where long-context deployment lives. The safety failure is invisibility, not outcome harm.

This claim is **directly proven by our existing data**. It does not require H2, correctness labels, or new experiments. It is also closer to what AI safety researchers actually worry about (deceptive alignment, scheming, Potemkin reasoning).

### 2.4 Within-conversation contrast is methodologically novel

The literature search found NO prior published work using anchored-vs-exploring contrast *within the same conversation* as a confound control. This is currently buried in §4.4 (the Repetition section) but should be explicitly claimed as a methodological contribution.

---

## 3. The five tasks (in priority order)

**Critical**: Do these tasks **sequentially** unless you have very strong reason to parallelize. Each builds on context from the prior one. Confirm each is working before moving to the next.

### Task 1: Monitor simulation analysis ★ HIGHEST PRIORITY

**Why it's first**: This delivers the strongest paper claim with the least effort. It turns "we measured non-faithfulness" into "we directly demonstrate that anchored turns evade CoT monitoring." That's the safety claim made empirical.

**Goal**: Explicitly compute what a generic CoT-consistency monitor would say on each turn, and show that anchored turns pass at ~100% by construction.

**Method**:
- A "CoT-consistency monitor" is operationalized as: a function that takes (CoT trace, answer) and returns a robustness score in [0,1] indicating whether perturbations of the CoT change the answer.
- We operationalize this via our existing 5-level truncation. Define `monitor_score(turn) = (# truncation levels with same answer as full-CoT) / 5`.
- An anchored turn has `monitor_score = 1.0` (5/5 agree).
- An exploring turn has `monitor_score < 1.0`.
- Compute the distribution of `monitor_score` over turns.
- Cross-tab with conversation length: short conversations have lower monitor scores (more exploring); long conversations have higher (more anchored).
- **Key insight to communicate**: as conversations lengthen, the monitor becomes MORE confident in its trust signal even though the underlying CoT-answer causal link weakens. This is the "monitor-evasion gap."

**Concrete deliverables**:
- Script: `code/monitor_simulation.py`
- Stats JSON: `results/monitor_simulation/monitor_stats.json` with keys: `score_distribution_by_length`, `mean_score_short/medium/long`, `frac_turns_passing_threshold_0.8`
- Figure: `paper/figures/monitor_simulation.png` — a 2-panel plot. Left: histogram of monitor scores split by short/medium/long. Right: mean monitor score vs conversation length bin, with confidence intervals.
- Paper integration: new subsection in §5 Discussion titled "Direct demonstration of CoT monitoring failure" (about 1/3 page), placed BEFORE the Candidate Mechanisms paragraph.

**Verification step**: An anchored turn must always have `monitor_score = 1.0`. If your script disagrees with this for any anchored turn, you have a bug — find it before continuing.

---

### Task 2: H2b — Information integration test

**Goal**: Test whether anchoring rate predicts failure to update beliefs across shards. Tests the *mechanism* of anchoring without needing correctness labels.

**Method**:
- For each turn, the `regen_answer_preview` field contains the regenerated answer at each of the 5 truncation levels. The 0%-level answer is the model's answer when given the conversation context BUT NO current-turn CoT.
- This 0%-level answer is the model's "context-only" answer — what the conversation history alone determines.
- For each conversation, extract the sequence of 0%-level answers across turns.
- Compute `answer_update_rate = (# consecutive-turn pairs where 0%-level answer changes) / (# consecutive pairs)`.
- Correlate `anchoring_rate` (conversation-level) with `answer_update_rate` (conversation-level) across N=67 conversations.

**Prediction**: high anchoring rate ↔ low answer update rate. Spearman ρ < 0, p < 0.05.

**Why this is a clean test**:
- It doesn't need correctness labels — it tests "does the model update?" not "does the model get the right answer?"
- It tests the mechanism (information integration failure) rather than the downstream consequence (wrong answer).
- It's directly relevant to safety: a model that fails to update beliefs in response to new evidence is a safety problem regardless of whether the answers are nominally correct.

**Concrete deliverables**:
- Script: `code/h2b_information_integration.py`
- Stats JSON: `results/h2b/integration_stats.json` with keys: `spearman_rho`, `spearman_p`, `per_conv_data` (list of {conv_id, anchoring_rate, answer_update_rate, n_turns})
- Figure: `paper/figures/h2b_integration.png` — scatter plot of anchoring rate vs answer update rate per conversation, with regression line.
- Paper integration: new subsection in §4 titled "Anchoring predicts failure to integrate new evidence" (about 1/2 page), placed AFTER §4.4 Repetition.

**Important parsing detail**: The `regen_answer_preview` field is a list of dicts or a string-encoded structure (verify the exact format before writing the script — read 3 lines of `results/phase2/faithfulness.jsonl` and `code/phase2_bistability_analysis.py`'s `is_anchored()` function to see how answers are extracted). The answer-extraction regex is already defined in `phase2_bistability_analysis.py` — use the SAME regex for consistency. Do not reinvent it.

---

### Task 3: Path B — Label recovery from sharded-GSM8K

**Goal**: Recover correctness labels for all 67 conversations (vs current N=21), enabling a higher-power H2 test.

**Why we attempt this even though Task 1+2 don't require it**:
- If H2 confirms with N=67: paper is stronger AND we have direct evidence for both process-level and outcome-level claims.
- If H2 stays null with N=67: this is *actually a stronger result* given the Lanham insight — confirms orthogonality to task difficulty. Frame as a positive.
- Either way, the H2 reframing in Task 4 doesn't change. Path B is bonus information.

**Method**:
- `task_id` format in our JSONL is `sharded-GSM8K/{integer_id}`. Extract the integer.
- GSM8K test set has known gold answers — load from HuggingFace `openai/gsm8k` (the `test` split, 1319 problems), or look in the `lost-in-conversation` repo for cached versions.
- Parse the model's FINAL stated answer at the LAST turn of each conversation trace. This is from the conversation trace files (NOT the faithfulness JSONL).
- Conversation traces are typically at:
  - `~/multi_turn_cot/results/day1/` (server) — Phase 1 traces
  - `~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase{2,3,4,5_s1,5_s2,5_s3}/` (server)
  - **CHECK FIRST**: Are these traces also stored locally? Look in `D:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results\phase*\` for trace files (likely named `*.json` or `*_trace.json` per conversation).
- Score each conversation: `is_correct = (extracted_final_answer == gold_answer)`.

**Concrete deliverables**:
- Script: `code/recover_correctness_labels.py`
- Labels JSON: `results/correctness_labels/labels.json` — `{conv_id: {task_id, final_answer, gold_answer, is_correct}}`
- Updated H2 analysis: rerun `phase2_bistability_analysis.py`'s H2 section with new labels.
- Paper integration: update §4 H2 section with new N. If H2 stays underpowered, foreground the orthogonality framing per Task 4.

**CAVEATS**:
- The conversation may NOT have a clean "final answer" if it derailed badly. Some conversations will be unrecoverable. Aim for "as many as possible" — even N=40 is a big improvement over N=21.
- If trace files aren't local, SSH access to 172.24.16.177 (user: vasudev_majhi_2021) is needed. Use paramiko per existing project pattern.
- Do NOT confuse `faithfulness.jsonl` rows (which have `regen_answer_preview` per turn) with conversation trace files (which have the original model responses per turn). The model's FINAL answer in a real conversation is in the TRACE, not in the faithfulness measurement output.

---

### Task 4: Discussion reframe (paper edits, NO new experiments)

**Goal**: Rewrite §5 Discussion to lead with process-level / monitor-evasion arguments rather than outcome-prediction (H2). This is the heaviest single change to the paper text.

**Concrete edits to `paper/paper.tex`**:

1. **At the start of §5 Discussion**, add a new paragraph titled "Process-level monitorability" that:
   - States the safety claim in process terms: anchored turns are monitor-evasion regimes
   - Cites Korbak et al. 2025's three monitorability archetypes (Process / Outcome / Intervention)
   - States that our metric is Process+Intervention, and that outcome-level evidence (H2) is supplementary
   - References Task 1's monitor simulation result as direct empirical demonstration

2. **In §5 (somewhere appropriate)**, add a paragraph titled "Orthogonality to task difficulty" that:
   - Cites Lanham et al. 2023's R²=0.74 finding
   - Notes that strong faithfulness-accuracy correlations are suspect — they may collapse the metric into a difficulty proxy
   - States that our weak H2 correlation (r=−0.242, p=0.291) is consistent with anchoring being orthogonal to task difficulty
   - Frames this as a feature: anchoring measures a property of conversational state, not problem hardness

3. **In §2 Related Work** (or §1 Introduction, your call), add a sentence or two claiming **within-conversation contrast as a methodological contribution**. Currently §4.4 uses this without crediting it. Sample wording: "Our analysis uses a within-conversation contrast methodology — comparing anchored vs exploring turns within the same conversation — which controls for between-conversation confounds (problem difficulty, model state, length). To our knowledge this is the first multi-turn CoT faithfulness study to use this matched design."

4. **Demote H2 in the Discussion**:
   - H2 is currently presented as load-bearing. Change framing to: "We additionally test whether anchoring correlates with outcome failure (H2). With N=21 the test is underpowered and the effect, if any, is modest (r=−0.242, p=0.291). This null is consistent with the orthogonality argument above; a higher-power test (Path B with N=67, see §X) is reported in [new section]."

5. **Update the "8.5 post-hoc turns" framing**: do not say this implies harm — say it implies invisibility. "In a 30-turn conversation, approximately 8.5 turns are anchored — these turns are precisely the ones where any CoT-consistency monitor would assign maximum trust while the underlying causal link is absent."

**Do NOT**:
- Remove any existing citations
- Remove the Limitations entry about "Interpretation of answer invariance" (this is the Hydra Effect rebuttal — keep it, it's load-bearing for a different reviewer objection)
- Remove the Candidate Mechanisms paragraph
- Change wording on chua2025 (the 59% figure) or meek2025 (the verbosity description) — these were fixed in Session 5 after errors. Re-read references.bib and the relevant paper paragraphs before any edit nearby.

---

### Task 5: Shard-order natural experiment

**Goal**: Test whether anchoring is order-dependent — i.e., whether the model "locks in" on early shards. This is an exploratory robustness check.

**Method**:
- Phase B has 3 seeds (s1, s2, s3) which use different shard orderings of the same problem set.
- For task_ids that appear across multiple seeds, compare the model's final-answer distribution.
- If reasoning drives the answer: final answers should be invariant to shard order (modulo correctness).
- If anchoring is genuinely a failure of evidence integration: final answers should differ across orderings.

**Concrete deliverables**:
- Script: `code/shard_order_experiment.py`
- Stats JSON: `results/shard_order/order_dependence_stats.json` with keys: `n_overlapping_task_ids`, `frac_consistent_across_seeds`, `frac_consistent_split_by_anchoring_rate`
- Paper integration: add as a paragraph in §4 (probably end of §4 robustness section) — small contribution, ~1/3 page

**This task is lowest priority.** If time is short, skip or defer to a future session. Tasks 1, 2, 4 are the must-haves.

---

## 4. What NOT to do (hard constraints)

These are derived from prior sessions, prior bugs, and explicit user feedback. Violating any of these will require redoing work.

- **DO NOT** use the words "bistable" or "bistability" in the paper text. H1 was never confirmed (bootstrap p=0.521). The paper used to use these words; they were removed in earlier sessions. The directory is still named `bistability_v3_combined/` — that's fine, but the paper text must not use the term.
- **DO NOT** use `scipy.stats.kstest` on the run-length distribution. It's invalid on discrete data (Massey 1951). Use the parametric bootstrap in `phase2_bistability_analysis.py`.
- **DO NOT** rerun any Phase A/B experiments. The data is complete (1,289 observations) and analyzed.
- **DO NOT** delete or overwrite any of the 12 existing figures in `paper/figures/`. Add new figures with new names (e.g., `monitor_simulation.png`, `h2b_integration.png`).
- **DO NOT** modify the chua2025 paragraph or meek2025 paragraph in paper.tex without re-reading their current text and the references.bib entries. Both were fixed for accuracy errors in Session 5:
  - chua2025: the 59% cue-identification figure CANNOT be compared to chen2025's 25-39% — different measurement protocols
  - meek2025: their "verbosity" = how much reasoning is externalized, NOT causal necessity (causal necessity is OUR contribution; previous draft misattributed it to them)
- **DO NOT** exceed 30 pages total. Target venue (CAISc 2026 / NeurIPS Safe-GenAI workshop) has page caps; we're at 27 now.
- **DO NOT** commit anything to git unless the user explicitly says "commit". Suggest a commit message when work is done; let them decide.
- **DO NOT** push to remote.
- **DO NOT** invent author names. References.bib has two entries with `author = {others}` (thought_anchors2026, conv_inertia2026) — leave them as-is; the user will fill in real names at camera-ready.
- **DO NOT** write paper claims without grounding in the actual computed stats. Every numeric claim in the paper must be traceable to a JSON output file.
- **DO NOT** delete files unnecessarily. If you need to replace a script, edit it.
- **DO NOT** add features beyond what each task specifies. No "while I'm here let me also refactor X."
- **DO NOT** add error handling, fallbacks, or validation for scenarios that can't happen. Internal scripts don't need defensive programming.
- **DO NOT** add comments explaining WHAT the code does. Only WHY when non-obvious.

---

## 5. Data locations (everything you need)

### Local files (Windows, `D:\Desktop\Autoresearch\`)

| Type | Path |
|------|------|
| Project root | `research_projects\multi_turn_cot_faithfulness\` |
| Paper TeX | `research_projects\multi_turn_cot_faithfulness\paper\paper.tex` |
| Paper bib | `research_projects\multi_turn_cot_faithfulness\paper\references.bib` |
| Paper figures | `research_projects\multi_turn_cot_faithfulness\paper\figures\` |
| Faithfulness Phase 2 | `research_projects\multi_turn_cot_faithfulness\results\phase2\faithfulness.jsonl` |
| Faithfulness Phase 3 | `research_projects\multi_turn_cot_faithfulness\results\phase3\faithfulness.jsonl` |
| Faithfulness Phase 4 | `research_projects\multi_turn_cot_faithfulness\results\phase4\faithfulness.jsonl` |
| Faithfulness Phase 5 s1 | `research_projects\multi_turn_cot_faithfulness\results\phase5_s1\faithfulness.jsonl` |
| Faithfulness Phase 5 s2 | `research_projects\multi_turn_cot_faithfulness\results\phase5_s2\faithfulness.jsonl` |
| Faithfulness Phase 5 s3 | `research_projects\multi_turn_cot_faithfulness\results\phase5_s3\faithfulness.jsonl` |
| Combined stats | `research_projects\multi_turn_cot_faithfulness\results\bistability_v3_combined\bistability_stats.json` |
| Core analysis script | `research_projects\multi_turn_cot_faithfulness\code\phase2_bistability_analysis.py` |
| Figure script (main) | `research_projects\multi_turn_cot_faithfulness\code\generate_paper_figures.py` |
| Figure script (suppl) | `research_projects\multi_turn_cot_faithfulness\code\generate_supplementary_figures.py` |
| Status doc | `research_projects\multi_turn_cot_faithfulness\status.md` |
| Project CLAUDE.md | `D:\Desktop\Autoresearch\CLAUDE.md` |
| Memory dir | `C:\Users\nisha\.claude\projects\d--Desktop-Autoresearch\memory\` |

### Server files (172.24.16.177, user: `vasudev_majhi_2021`)

| Type | Path |
|------|------|
| Phase 1 traces | `~/multi_turn_cot/results/day1/` |
| Phase A faithfulness | `~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase{2,3,4}/` |
| Phase B faithfulness | `~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s{1,2,3}/` |
| Cross-model experiments | tmux session `experiments` (running Qwen3-14B + R1-Distill-Llama-8B) |

**Server access pattern**: existing project uses paramiko via Python. See prior scripts for SSH boilerplate.

---

## 6. Pre-flight checklist (do this BEFORE starting Task 1)

1. **Read `code/phase2_bistability_analysis.py`** end-to-end. In particular understand:
   - The `is_anchored()` function and its regex for extracting numeric answers
   - The H2 computation (which conversations have correctness labels and where they come from)
   - The bootstrap implementation for H1
2. **Read one row of `results/phase2/faithfulness.jsonl`** (just `head -1`). Note every field name. In particular note the structure of `regen_answer_preview` — is it a list of dicts? Is each dict keyed by truncation_pct? Is the answer text raw or pre-extracted?
3. **Read `results/bistability_v3_combined/bistability_stats.json`** — find what's already computed. Don't recompute things; reuse.
4. **Read paper.tex §4.4 (Repetition section)** — match the writing style of any new subsections you add.
5. **Read paper.tex §5 Discussion in full** — understand current safety framing before rewriting.
6. **Read paper.tex §6 Limitations** — find the "Interpretation of answer invariance" paragraph (added Session 5). Do NOT remove it.

---

## 7. Verification & compilation

After each task:
1. Run the new script. Check the output JSON exists and looks sane.
2. View the new figure. Check it isn't empty or broken.
3. After paper edits: recompile:
   ```bash
   cd research_projects/multi_turn_cot_faithfulness/paper
   pdflatex paper.tex
   bibtex paper
   pdflatex paper.tex
   pdflatex paper.tex
   ```
   `caisc_2026.sty` is already there; do not recopy.
4. Open `paper.pdf` and verify the new section/figure rendered.
5. Update `status.md` with a new "Session 6 Update" section listing what was done.
6. Tell the user what's done. Do NOT commit unless asked.

**Windows-specific note**: When generating figures locally, prefix with `PYTHONIOENCODING=utf-8` to avoid cp1252 errors with arrow characters.

---

## 8. Reference: relevant citations already in `references.bib`

These are already cited correctly. Cite by key.

| Key | Use for |
|-----|---------|
| `lanham2023` | Counterfactual deletion methodology (our anchoring measure); R²=0.74 contrast |
| `korbak2025` | Monitorability framework — three archetypes |
| `meek2025` | Verbosity+faithfulness without ground truth labels |
| `chua2025` | DeepSeek-R1 single-turn faithfulness baseline (59% cue identification — careful: don't conflate with chen2025) |
| `chen2025` | "Reasoning Models Don't Always Say What They Think" — 25-39% verbalization |
| `turpin2023` | CoT unfaithful explanations — foundational |
| `thought_anchors2026` | Sentence-level "anchoring" — disambiguated in §2 |
| `conv_inertia2026` | Conversational inertia |
| `context_length_hurts2025` | Context length degrades reasoning (14-85%) |
| `cot_necessity2025` | Task-theoretic CoT necessity |
| `laban2025` | ICLR 2026 Best Paper — multi-turn derailment benchmark we use |
| `deepseekr1_2025` | DeepSeek-R1 paper — for RL training mechanism |
| `mechanistic_faithfulness2026` | Mechanistic evidence for faithfulness decay |
| `massey1951` | KS test invalid on discrete data |
| `reasoning_theater2026` | Boppana et al. — closest methodological precedent |

---

## 9. Expected deliverables at end of Session 6

A complete Session 6 should produce:

1. **Two new analysis scripts** with JSON outputs and PNG figures:
   - `code/monitor_simulation.py` → `results/monitor_simulation/monitor_stats.json` + `paper/figures/monitor_simulation.png`
   - `code/h2b_information_integration.py` → `results/h2b/integration_stats.json` + `paper/figures/h2b_integration.png`

2. **(Best effort)** correctness labels for as many conversations as possible:
   - `code/recover_correctness_labels.py` → `results/correctness_labels/labels.json`
   - If recovered, an updated H2 analysis with new N

3. **(Optional)** shard-order experiment if time permits:
   - `code/shard_order_experiment.py` → `results/shard_order/order_dependence_stats.json`

4. **Rewritten paper Discussion section** with:
   - "Process-level monitorability" paragraph leading the section
   - "Orthogonality to task difficulty" paragraph
   - Within-conversation contrast credited as methodological contribution
   - H2 demoted from load-bearing to supplementary
   - Monitor simulation result foregrounded as direct safety claim

5. **Updated documents**:
   - `status.md` — new "Session 6 Update (2026-05-12 or whenever)" section
   - `CLAUDE.md` — status block updated
   - `memory/project_multi_turn_cot_faithfulness.md` — updated with Session 6 state

6. **Paper compiles cleanly**, < 30 pages, all cross-references resolve.

---

## 10. Session 5 summary (what was already done, for context)

If you need to know what state the paper is in:

- Paper is 27 pages, compiles cleanly
- 22 references in references.bib (6 added in Session 5)
- New §2 paragraph: "Thought Anchors at the sentence level" — disambiguation
- New §5 paragraph: "Candidate mechanisms" — three testable hypotheses
- New Limitations entry: "Interpretation of answer invariance" — Hydra Effect rebuttal
- Laban 2025 flagged as ICLR 2026 Best Paper at first mention
- Cross-model experiments queued: Qwen3-14B (seed 44444) + R1-Distill-Llama-8B (seed 55555) — see `~/wait_and_run.sh` on server, runs when GPU has ≥30 GB free
- Chua2025 and Meek2025 wording fixed for accuracy

---

## 11. Final guidance

**On creativity**: Do not propose alternative tasks. The five above were chosen after a long exploratory conversation. Execute them. If you discover that one of them has a fundamental problem (e.g., the data doesn't support the analysis), explain the problem and stop — do not silently substitute.

**On scope**: Do exactly what each task specifies. Don't add bonus analyses. Don't refactor existing code.

**On the paper**: Do not change the core empirical claims. The H3 χ², the 3.9× gradient, the ICC = 0.152, the H1 inconclusive result — these are settled. New tasks add to the paper; they don't override.

**On the user**: They are a BITS Pilani student conducting an autonomous research run with the Autoresearch harness. They are technically capable but want minimal hand-holding. Communicate progress briefly, ask permission for destructive actions, do not lecture.

**On context recovery**: If anything in this document is unclear, read these files in order:
1. `D:\Desktop\Autoresearch\CLAUDE.md` (project overview)
2. `research_projects/multi_turn_cot_faithfulness/status.md` (most recent session state)
3. `research_projects/multi_turn_cot_faithfulness/paper/paper.tex` (the actual artifact)
4. `research_projects/multi_turn_cot_faithfulness/code/phase2_bistability_analysis.py` (the core analysis)

Begin with the pre-flight checklist in §6, then Task 1.
