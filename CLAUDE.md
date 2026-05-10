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

Experiments in `code/` are written to run from the working directory of the relevant external repo (e.g., `~/multi_turn_cot/lost_in_conversation/` on the GPU server), with paths resolved relative to that cwd. Scripts import each other directly (no package structure).

## Current active project: multi_turn_cot_faithfulness

Investigates bistable CoT faithfulness in multi-turn derailing conversations using DeepSeek-R1-Distill-Qwen-7B. All experiment code runs on a remote GPU server (172.24.16.177) via SSH/paramiko, not locally.

**Status (2026-05-10): Phase B COMPLETE. Combined N=67 analysis done. Supplementary figures generated.**
- Phase A dataset: **N=24 conversations, 412 faithfulness observations** across Phases 1–4 — COMPLETE
- Phase B (uplift): 3 seeds × ~43 conversations — **COMPLETE** (tmux session `uplift` finished)
- Combined analysis: **N=67 conversations, 1,289 obs** — `results/bistability_v3_combined/bistability_stats.json` downloaded locally
- Paper `paper/paper.tex` has N=24 stats; **needs update to N=67** (combined stats below)
- 5 new supplementary figures generated: `code/generate_supplementary_figures.py` → `paper/figures/`

**Combined N=67 key stats (use these for paper update):**
- H3: χ²=236.9, df=66, p≈0 — bistability confirmed across all 67 conversations
- H3 within-bin: medium p=0.0035, long p<0.001 — holds within each length bin
- ICC: 0.152 — 15% of anchoring variance is between conversations
- H1: bootstrap p=0.521 — still inconclusive (insufficient anchored runs at N=67)
- H2: r=−0.242, p=0.291, N=21 — correct direction, underpowered
- Length gradient: short 7.2% → medium 12.8% → long 28.4% (**3.9× ratio**)
- Repetition confound: anchored 75.8%, exploring 45.6%, ratio 1.66× (not pure inertia)

**Root-cause lesson from Phase B startup (2026-05-08):**
`simulator_sharded.py` had NO max_turns limit. Shard revelation only happens on answer attempts (not clarification questions), so a 6-shard problem could run 12+ turns per shard → 76-turn conversations. GSM8K/913 ran 76 turns (116 min faithfulness time alone). Fix: `--max_turns 30` was added to `phase2_batch_runner.py` and `simulator_sharded.py`. **Always use `--max_turns 30` for Phase B data collection.**

**Core scripts (run from `~/multi_turn_cot/lost_in_conversation/` on server):**

```bash
# Run sharded conversations + faithfulness measurement
# IMPORTANT: use --faith_tokens 128 (not 512). 512 takes 146s/turn; 128 takes ~31s/turn.
# IMPORTANT: use --max_turns 30. Without it, conversations run 26-76 turns (no termination limit).
# Use --exclude_dirs to avoid re-running already-measured conversations from prior phases.
LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 phase2_batch_runner.py --n_samples 15 --max_shards 20 --max_turns 30 --faith_tokens 128 \
  --exclude_dirs ../multi_turn_cot_faithfulness/results/phase2 \
                 ../multi_turn_cot_faithfulness/results/phase3 \
                 ../multi_turn_cot_faithfulness/results/phase4 \
  --out_dir ../multi_turn_cot_faithfulness/results/phase5_s1

# Bistability analysis — pure Python, no GPU required
# Pass ALL faith paths together to get full dataset stats
# Phase B adds phase5_s1, phase5_s2, phase5_s3 to the path list
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
  --out_dir ../multi_turn_cot_faithfulness/results/bistability_v2_full
```

**Phase B tmux session `uplift` on server — ALL COMPLETE (2026-05-10):**

| Window | Script | Status |
|---|---|---|
| phaseB_long_conv | run_phase5_s1.sh → s2 → s3 → run_analysis_full.sh | **DONE** |
| phaseC_humaneval | waits PHASEB_DONE → humaneval → HUMANEVAL_DONE | Did not complete (HumanEval not reached) |
| phaseC_fp16 | waits HUMANEVAL_DONE → deepseek_fp16 → FP16_DONE | Did not complete |
| phaseA_v2 | bistability_v2_full analysis | DONE |

Scripts in `/home/vasudev_majhi_2021/multi_turn_cot/uplift_scripts/`. Each seed: `--n_samples 15 --max_shards 20 --max_turns 30 --faith_tokens 128`.

**GPU note:** Seeds 2 and 3 ran in parallel (8-bit + 4-bit) after a watcher script fired when ollama freed ≥18 GB. Total wall-clock ~48h for all 3 seeds.

**Key env vars:**
- `LOAD_IN_8BIT=1` — load both models in int8 (~7 GB each); prefer when GPU has ~15 GB free
- `LOAD_IN_4BIT=1` — load both models in NF4 (~4 GB each); can coexist with an 8-bit process if needed
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents OOM during caching_allocator_warmup; use when running two processes in parallel
- `R1_MAX_NEW_TOKENS=1500` — cap R1 generation per turn
- `HF_HOME=/dev/shm/vasudev_hf_cache` — model weights on RAM-disk (ephemeral; re-download on reboot)
- `--faith_tokens 128` — use 128, NOT 512. 512 was the default and took 146s/turn; 128 takes ~31s/turn. The metric only needs a numeric answer, so 128 is always sufficient.
- `--max_turns 30` — CRITICAL. Without this, conversations run indefinitely (simulator terminates only when all shards revealed OR correct answer; shards reveal on answer attempts only → 12+ turns per shard).

**GPU constraint (shared server):** The RTX 6000 Ada (49 GB) is shared. Other users' ollama models can occupy 18–35 GB persistently. Strategy: run one 8-bit process (~18 GB) + one 4-bit process (~8 GB) simultaneously to fill the GPU when ~26+ GB is free. Both processes will compete for compute but combined throughput beats running serially.

**Graduate/Shelve decision rule (bistability hypotheses):**
- H1: KS test on anchored run-lengths vs geometric null, p < 0.05 → anchored mode is persistent ✅ confirmed at N=24
- H2: point-biserial r(frac\_anchored, is\_correct) < 0, p < 0.05 → anchored mode predicts failure (unresolvable: most math samples return is_correct=None when model never converges)
- H3: both anchored AND exploring modes present (neither > 95% of turns) → bistability confirmed ✅ confirmed at N=24
- Any H significant → GRADUATE ✅ **GRADUATED**

## Architecture of experiment code

`model_local.py` is the central piece — a drop-in replacement for the simulator's `model_openai.py`. It:
- Lazily loads R1-Distill-7B and Qwen2.5-7B-Instruct from HuggingFace (cached, thread-safe)
- Strips all prior `<think>...</think>` blocks from assistant history before every generation (required by DeepSeek model card to avoid context overflow)
- Supports `LOAD_IN_8BIT` / `LOAD_IN_4BIT` via `BitsAndBytesConfig`
- Exposes `generate()`, `generate_json()`, and `split_thinking()` with the same signatures as the original OpenAI shim

`faithfulness_counterfactual.py` implements Lanham-style counterfactual deletion: for each assistant turn, it truncates the `<think>` block to {0%, 25%, 50%, 75%, 100%} of its tokens, force-appends `</think>`, and regenerates the answer with `do_sample=False`.

**Known metric limitation:** The `faithfulness_score` field in output JSONL uses correctness-flip (did the correctness label change across truncation levels?). This is uninformative when the model is consistently wrong at all truncation levels — the score is 0 even if the model is genuinely anchored. Use the `is_anchored()` function in `phase2_bistability_analysis.py` instead: it regex-extracts the numeric answer from `regen_answer_preview` and checks whether all 5 levels agree. That is the correct metric for bistability analysis.

`analyze_faithfulness.py` / `phase2_bistability_analysis.py` are pure analysis scripts — they read JSONL files from disk, compute metrics, run stats, and write plots. They do not load models.

## Data layout on server

Phase 1 results are at `~/multi_turn_cot/results/` (not inside `multi_turn_cot_faithfulness/`):
```
~/multi_turn_cot/results/day1/           # 4 trace JSONs + per_turn_records.jsonl
~/multi_turn_cot/results/day2/           # faithfulness.jsonl (9 turns, 3 small samples)
~/multi_turn_cot/results/day2_sample965/ # faithfulness.jsonl (44 turns, sample 965)
~/multi_turn_cot/results/bistability_phase1_only/  # bistability plots + stats JSON
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase2/   # Phase 2 traces + faithfulness
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase3/   # Phase 3 traces + faithfulness
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase4/   # Phase 4 traces + faithfulness
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s1/ # Phase B seed 1 (COMPLETE)
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s2/ # Phase B seed 2 (COMPLETE)
~/multi_turn_cot/multi_turn_cot_faithfulness/results/phase5_s3/ # Phase B seed 3 (COMPLETE)
~/multi_turn_cot/multi_turn_cot_faithfulness/results/bistability_v2_full/  # N=24 Phase A analysis (superseded)
~/multi_turn_cot/multi_turn_cot_faithfulness/results/bistability_v3_combined/  # N=67 combined analysis (CURRENT)
```

Sentinel files (written to results/ dir on server as progress markers):
- `PHASEB_DATA_DONE_S1` — seed 1 data collection complete
- `PHASEB_DATA_DONE` — all 3 seeds complete
- `PHASEB_DONE` — Phase B analysis also done (bistability_v2_full updated)
- `HUMANEVAL_DONE` — HumanEval H2 analysis complete
- `FP16_DONE` — FP16 precision comparison complete

## Research philosophy (decision criteria)

Four criteria for rating a research question (from `research-philosophy.md`):
1. **Surprising to experts** — not already known, not easily predicted
2. **Fruitful** — opens downstream questions, not just +5% on an obscure eval
3. **Foreclosing alternative explanations** — ablations, confounds, multiple seeds
4. **Feasible** — completable within available time, resources, and skills

The agent self-reviews against these before proposing any question and again before drafting the paper (acting as an AI conference reviewer).

## Paper output

Target venue: CAISc 2026 / NeurIPS Safe-GenAI workshop. Template at `autovoila/draft-format/caisc_2026.tex` + `.sty`. Final output goes in `research_projects/{name}/paper/`. The agent is instructed to avoid AI writing style in the final draft and to include a Claude acknowledgement.

Current paper: `research_projects/multi_turn_cot_faithfulness/paper/paper.tex`. Figures (heatmap, run-length distribution, faith distribution, frac-anchored scatter) are in `paper/figures/`. To compile: copy `autovoila/draft-format/caisc_2026.sty` to `paper/` and run `pdflatex paper.tex`.
