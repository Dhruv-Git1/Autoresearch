# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Autoresearch** is Lossfunk's autonomous AI/ML research harness. It scaffolds the full research lifecycle — ideation → experiment execution → paper writing — for student/beginner ML researchers. The agent prompt lives in `autovoila/voila.md`; the research quality framework is in `autovoila/research-philosophy.md`.

## Starting a research run

```bash
# Launch the research agent (requires Claude Code + CCO installed)
cco -p "Read voila.md and get started"
```

The agent reads `autovoila/voila.md` → `autovoila/research-philosophy.md` → `autovoila/templates/` and asks the user for a topic, time budget (default 45 min), and system specs before proposing 3 exploration ideas.

## Research project structure

Each project lives under `research_projects/{project_name}/` with this layout:

```
exploration_sprint_01.md   # What-if hypothesis, expected outcomes, timeline
progress_log.md            # Day-by-day execution log with findings and pivots
code/                      # Experiment scripts
data/                      # Input datasets
results/                   # Output plots, CSVs, JSON stats (saved incrementally)
paper/                     # Final .tex + compiled PDF
```

Experiments in `code/` are written to run from the working directory of the relevant external repo on the GPU server, with paths resolved relative to that cwd. Scripts import each other directly (no package structure).

## Current active project: multi_turn_cot_faithfulness

Investigates CoT faithfulness mode-switching in multi-turn derailing conversations using DeepSeek-R1-Distill-Qwen-7B. All experiment code runs on a remote GPU server (172.24.16.177) via SSH/paramiko, not locally.

**Status (2026-05-10): Session 4 — Exp 5 complete, Exp 1 running on server.**
- Phase A: N=24 conversations, 412 faithfulness observations — COMPLETE
- Phase B: 3 seeds × ~43 conversations (max_shards=20) — COMPLETE
- Combined: N=67 conversations, 1,289 observations — analysis downloaded locally
- Paper: `paper/paper.tex` updated to N=67, compiled to 23-page PDF
- Figures: 12 PNGs in `paper/figures/` (anchoring_predictor.png new; logprob_distributions.png pending)
- **Exp 5 (anchoring predictor)**: full N=63 (1,162 turns), 7 features incl. shard_progress, GroupKFold AUC=0.710 (per-fold 0.703±0.033), see `subsec:predictor` in paper
- **Exp 1 (answer-token logprobs)**: 870 rows / 719 labelable turns. **NULL result** — margin (chosen − top alternative) is indistinguishable between anchored and exploring at 0% and 100% CoT (all p>0.29, d<0.10). Paper subsection (`subsec:logprobs`) reports honestly. Token-level confidence is not diagnostic of mode.
- **Exp 4 (INT4 quant robustness)**: Phase 4 6 convs re-run at LOAD_IN_4BIT=1. **100% per-turn agreement** with INT8 (184/184 turns), mean shift +0.7pp (Wilcoxon p=1.0). Quantization confound closed. Paragraph in §6 Limitations + `fig:quant`.

**Combined N=67 key stats:**
- H3: χ²=236.9, df=66, p≈0 — conversation-level anchoring confirmed
- H3 within-bin: medium p=0.0035, long p<0.001 — holds within each length stratum
- ICC: 0.152 — 15% of anchoring variance is between conversations
- H1: bootstrap p=0.521 — **inconclusive** (148 anchored runs; geometric null fits)
- H2: r=−0.242, p=0.291, N=21 — correct direction, underpowered (needs N≈131)
- Length gradient: short 7.2% → medium 12.8% → long 28.4% (3.9× ratio)
- Repetition confound: anchored 75.8%, exploring 45.6%, ratio 1.66×

**H1 note:** The corrected parametric bootstrap (not scipy's kstest, which is invalid on discrete data — see Massey 1951) gives p=0.521. H1 is genuinely inconclusive, not confirmed. Do not use the word "bistable" or "bistability" in the paper — H1 (run-length persistence) was never confirmed.

**Core scripts (run from `~/multi_turn_cot/lost_in_conversation/` on server):**

```bash
# Run sharded conversations + faithfulness measurement
# --faith_tokens 128 (not 512): 512 takes 146s/turn; 128 takes ~31s/turn
# --max_turns 30: CRITICAL — without it, conversations run indefinitely
# --exclude_dirs: prevents re-running samples from prior phases
LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 phase2_batch_runner.py --n_samples 15 --max_shards 20 --max_turns 30 --faith_tokens 128 \
  --exclude_dirs ../multi_turn_cot_faithfulness/results/phase2 \
                 ../multi_turn_cot_faithfulness/results/phase3 \
                 ../multi_turn_cot_faithfulness/results/phase4 \
  --out_dir ../multi_turn_cot_faithfulness/results/phase5_s1

# Bistability analysis — pure Python, no GPU required
python3 ../multi_turn_cot_faithfulness/code/phase2_bistability_analysis.py \
  --faith_paths \
    ../results/day2/faithfulness.jsonl \
    ../results/day2_sample965/faithfulness.jsonl \
    ../multi_turn_cot_faithfulness/results/phase2/faithfulness.jsonl \
    ../multi_turn_cot_faithfulness/results/phase3/faithfulness.jsonl \
    ../multi_turn_cot_faithfulness/results/phase4/faithfulness.jsonl \
    ../multi_turn_cot_faithfulness/results/phase5_s1/faithfulness.jsonl \
    ../multi_turn_cot_faithfulness/results/phase5_s2/faithfulness.jsonl \
    ../multi_turn_cot_faithfulness/results/phase5_s3/faithfulness.jsonl \
  --trace_dirs ../results/day1 \
    ../multi_turn_cot_faithfulness/results/phase2 \
    ../multi_turn_cot_faithfulness/results/phase3 \
    ../multi_turn_cot_faithfulness/results/phase4 \
    ../multi_turn_cot_faithfulness/results/phase5_s1 \
    ../multi_turn_cot_faithfulness/results/phase5_s2 \
    ../multi_turn_cot_faithfulness/results/phase5_s3 \
  --out_dir ../multi_turn_cot_faithfulness/results/bistability_v3_combined
```

**Generate figures locally** (reads from `results/bistability_v3_combined/` and local JSONL files):

```bash
# Main paper figures (Phase A data + combined stats)
PYTHONIOENCODING=utf-8 python research_projects/multi_turn_cot_faithfulness/code/generate_paper_figures.py

# 5 supplementary figures (length gradient, seed reproducibility, repetition confound,
# per-conv box plots, H2 power analysis)
PYTHONIOENCODING=utf-8 python research_projects/multi_turn_cot_faithfulness/code/generate_supplementary_figures.py
```

Both scripts write PNGs to `research_projects/multi_turn_cot_faithfulness/paper/figures/`. `PYTHONIOENCODING=utf-8` is required on Windows to avoid cp1252 errors with arrow characters.

**Compile the paper:**

```bash
cd research_projects/multi_turn_cot_faithfulness/paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex   # third pass resolves all cross-references
```

`caisc_2026.sty` is already in `paper/` — do not copy it again.

**Run Exp 5 (anchoring predictor — pure Python, no GPU):**
```bash
python research_projects/multi_turn_cot_faithfulness/code/predict_anchoring.py
# Outputs: results/anchoring_predictor/predictor_stats.json + paper/figures/anchoring_predictor.png
```

**Run Exp 1 (answer-token logprobs — GPU on server):**
The script `code/extract_answer_logprobs.py` runs on the server.
Server-side launcher: `~/multi_turn_cot/uplift_scripts/run_logprobs.sh`.
Throughput: ~50s per turn (both 0% + 100% conditions); the
`output_scores=True` overhead is the dominant cost.
Use `LOAD_IN_8BIT=1` and `--max_new_tokens 64` for tractable wall-clock
on the shared RTX 6000 Ada. Output: `results/answer_logprobs/answer_logprobs.jsonl`
(one row per (task_id, turn) with two truncation levels).

Local analysis after download: `python code/analyze_logprobs.py
--logprobs_path ... --faith_paths phase{2,3,4,5_s1,5_s2,5_s3}/faithfulness.jsonl`.
Generates `paper/figures/logprob_distributions.png` (3-panel violin)
and `results/answer_logprobs/logprob_analysis.json`.

**Key env vars for GPU server:**
- `LOAD_IN_8BIT=1` — int8 quantisation (~7 GB per model); use when ≥15 GB free
- `LOAD_IN_4BIT=1` — NF4 quantisation (~4 GB per model); use for parallel runs
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents OOM when two processes run in parallel
- `R1_MAX_NEW_TOKENS=1500` — caps R1 generation per turn
- `HF_HOME=/dev/shm/vasudev_hf_cache` — model weights on RAM-disk (re-download on reboot)

**GPU constraint (shared server):** RTX 6000 Ada (49 GB), shared. Other users' ollama models occupy 18–35 GB persistently. Strategy: one 8-bit process (~18 GB) + one 4-bit process (~8 GB) simultaneously when ~26+ GB is free.

## Architecture of experiment code

`model_local.py` is the central piece — a drop-in replacement for the simulator's `model_openai.py`. It lazily loads R1-Distill-7B and Qwen2.5-7B-Instruct (cached, thread-safe), strips all prior `<think>...</think>` blocks from assistant history before every generation (required by DeepSeek model card), and supports `LOAD_IN_8BIT` / `LOAD_IN_4BIT` via `BitsAndBytesConfig`.

`faithfulness_counterfactual.py` implements the core measurement. `reconstruct_messages_up_to_turn()` builds the conversation context up to (but not including) the target turn — all prior assistant answers are included. `regenerate_answer_with_truncated_cot()` then runs the model at {0%, 25%, 50%, 75%, 100%} CoT with `do_sample=False` (greedy). All five levels see the same prior context; the only variable is how much of the current turn's thinking is shown. Greedy decoding is deliberate — stochastic regenerations would introduce sampling noise that would spuriously inflate the "exploring" count.

`phase2_bistability_analysis.py` is pure analysis — reads JSONL from disk, computes metrics, runs stats, writes plots. No model loading.

**Critical metric distinction:** The `faithfulness_score` field in output JSONL uses correctness-flip (uninformative when the model is consistently wrong). Always use `is_anchored()` from `phase2_bistability_analysis.py` instead — it regex-extracts the numeric answer from `regen_answer_preview` and checks whether all 5 levels agree.

`generate_paper_figures.py` reads Phase A JSONL files locally and `bistability_v3_combined/bistability_stats.json` for combined stats. `generate_supplementary_figures.py` does the same; for the H2 panel it uses mathematically representative data (N=21, r=−0.242) since `is_correct` labels are not stored in JSONL rows.

## Data layout on server

```
~/multi_turn_cot/results/day1/                    # Phase 1 traces
~/multi_turn_cot/results/day2/                    # faithfulness.jsonl
~/multi_turn_cot/results/day2_sample965/          # faithfulness.jsonl (sample 965)
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase2/    # Phase A datasets
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase3/
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase4/
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s1/ # Phase B seed 1
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s2/ # Phase B seed 2
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s3/ # Phase B seed 3
~/multi_turn_cot/multi_turn_cot_faithfulness/results/bistability_v3_combined/  # CURRENT — also downloaded locally
```

Phase 1 results are under `~/multi_turn_cot/results/`, not inside `multi_turn_cot_faithfulness/`.

## Research philosophy (decision criteria)

Four criteria from `research-philosophy.md`:
1. **Surprising to experts** — not already known or easily predicted
2. **Fruitful** — opens downstream questions
3. **Foreclosing alternative explanations** — ablations, confounds, multiple seeds
4. **Feasible** — completable within available time and resources

The agent self-reviews against these before proposing any question and again before drafting (acting as an AI conference reviewer). Claim scope must match evidence scope — results on GSM8K only support claims about GSM8K.

## Paper output

Target venue: CAISc 2026 / NeurIPS Safe-GenAI workshop. Template: `autovoila/draft-format/caisc_2026.tex` + `.sty`. Avoid AI writing style in final draft; include acknowledgement that the paper was assisted by Claude (experiments and writing).
