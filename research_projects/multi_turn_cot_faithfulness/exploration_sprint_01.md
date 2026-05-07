# Exploration Sprint 01 — Multi-Turn CoT Faithfulness Decay

**Started:** 2026-05-07
**Status:** Pre-sprint setup

---

## BEFORE YOU START

### The What-If
What if the chain-of-thought traces produced by reasoning models (DeepSeek-R1, o3-style) become *progressively unfaithful to the model's actual computation* as a multi-turn underspecified conversation accumulates wrong assumptions? In other words: as the model "gets lost" across turns, does its visible reasoning increasingly become post-hoc rationalization rather than a real computation trace?

### Why this, why now?
Two recent threads of work that have not been connected:

1. **Laban et al. (2025), "LLMs Get Lost In Multi-Turn Conversation"** (arXiv:2505.06120): All 15 LLMs tested, including o3 and DeepSeek-R1, drop ~39% in multi-turn underspecified settings. The drop is dominated by a +112% increase in *unreliability*. Reasoning models do NOT escape this — they actually generate longer, more assumption-laden responses.
2. **CoT Faithfulness literature** (e.g., "Lie to Me" 2603.22582, "Mechanistic Evidence for Faithfulness Decay" 2602.11201, "Reasoning Models Don't Always Say What They Think" 2505.05410): Single-turn evidence that CoT is often unfaithful — Claude 3.7 Sonnet acknowledges hints only 25% of the time, R1 only 39%.

**Nobody has measured how faithfulness evolves *across turns* in a derailing multi-turn conversation.** This is the gap.

---

## EXPECTATIONS

### What do you expect to observe?
For DeepSeek-R1-Distill-Qwen-7B/14B run on the Microsoft `lost_in_conversation` dataset (SHARDED setting, GSM8K math task):

1. **Faithfulness will decay monotonically with turn number.** Specifically, the rate at which thinking-token content predicts final-answer correctness will drop from turn 1 to turn N.
2. **The decay will be steeper after the first "wrong commitment" turn.** Once the model has produced a confident-but-wrong intermediate answer, subsequent CoT will increasingly justify it rather than recompute.
3. **Verbose responses (one of the four root causes in the multi-turn paper) will be precisely the ones with the largest faithfulness gap** — i.e., bloated rationalization.

### Why do you expect this?
- The multi-turn paper notes "models overly rely on previous (incorrect) answer attempts" → this is psychologically post-hoc justification.
- Single-turn CoT is already 25–39% faithful per the "Lie to Me" paper — this is the ceiling under best conditions, so multi-turn should be no better.
- Mechanistically, transformers attending to their own prior tokens with confident commitments is a known sycophancy/anchoring pattern.

### What would genuinely surprise you?
- **S1:** Faithfulness *increases* with turn number — i.e., models become more honest about their confusion as conversations derail. This would suggest the thinking trace is a genuine confusion signal, contradicting the "post-hoc rationalization" hypothesis.
- **S2:** Faithfulness shows a sharp phase transition rather than monotonic decay — e.g., faithfulness collapses at turn 3 but is fine at turns 1–2. This would suggest a discrete "commitment" mechanism, which is itself a publishable finding.
- **S3:** Faithfulness is task-dependent — e.g., math CoT stays faithful but code CoT becomes unfaithful, or vice versa.

---

## FRUITFULNESS PRE-CHECK

### If this surprises you, then what?
Best-case finding: "CoT faithfulness in reasoning models decays by X% per turn in multi-turn underspecified conversations, with the steepest drop following the first wrong-commitment turn."

Downstream consequences:
- **AI safety/alignment:** CoT cannot be used as an audit log in deployed conversational AI. Every safety claim built on "we can monitor model reasoning by reading thinking tokens" needs revisiting.
- **Mechanistic interpretability:** Provides a clean experimental handle on the "self-repair / hydra effect" in conversational settings — when the model commits to a wrong answer, *which components* compensate to maintain coherence?
- **Practical:** Suggests interventions — e.g., turn-level CoT regularization, or simply: don't trust thinking tokens in turn ≥3.

### Who would care?
- **Anthropic Interpretability Team** (Olah, Nanda, Conmy) — directly extends their CoT faithfulness work
- **Salesforce Research / Microsoft Research** (Laban et al., authors of multi-turn paper) — natural follow-up
- **AI Safety community** (Apollo Research, METR) — challenges audit-trail assumptions
- **MATS / Anthropic alignment fellows** — concrete research direction
- **CAISc 2026, ICLR 2027, NeurIPS 2026 Safe Generative AI Workshop**

### Does this connect to research directions?
- LossFunk: AI safety, interpretability, reasoning model behavior
- Direct LossFunk topic alignment: reasoning, alignment, mechanistic understanding

---

## TIME BOX

**Sprint duration:** TBD with user (proposed: 1 week)
**Check-in:** Day 3 — do we have a working pipeline + initial faithfulness numbers per turn?
**Hard stop:** Day 7 — decision to graduate, shelve, or pivot

### What 'done' looks like (minimum scout mission)
- Reproduce 5 multi-turn SHARDED conversations from `microsoft/lost_in_conversation` dataset using DeepSeek-R1-Distill-Qwen-7B
- Extract `<think>...</think>` blocks per turn
- Apply ONE faithfulness measurement (early-answering / counterfactual deletion of CoT)
- Produce a single plot: faithfulness score vs. turn number (with error bars)

If that plot shows a clear monotonic trend or sharp phase change → **GRADUATE**. If flat or noisy → **SHELVE / PIVOT**.

---

## PRELIMINARY EXPERIMENTAL SKELETON

**Model:** DeepSeek-R1-Distill-Qwen-7B (open-weight, has explicit `<think>` tokens)
**Dataset:** [microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation) — GSM8K + Code (HumanEval) sharded splits
**Faithfulness metrics (start with #1, add #2-3 if time):**
1. **Counterfactual deletion** (Lanham-style): truncate `<think>` block to 10/30/50/70/100% and check answer change. Faithfulness = how much answer depends on each chunk.
2. **Hint injection** (Lie-to-Me-style): inject a hint into turn k's user message and check whether thinking trace acknowledges it.
3. **Resampling consistency:** sample N CoT traces for the same conversation state; if they yield wildly different conclusions, the trace is unreliable.

**Primary experiment:** For each turn t ∈ {1, ..., T} of N=50 SHARDED conversations, compute faithfulness(t). Plot mean ± 95% CI.

**Confounds to address:**
- Length confound: longer CoT might trivially be less faithful → control for token count
- Difficulty confound: later turns might have more cumulative complexity → control by comparing same-content single-turn baseline

---

## OPEN QUESTIONS (need user input)
1. **Time budget**: voila.md default is 45 min — wildly insufficient. What's the real budget? (Suggest: 1 week for sprint, +2 weeks if graduating.)
2. **GPU specs**: What GPU is on the server (model, VRAM)? Determines whether we use 7B vs 14B vs 32B distilled R1.
3. **SSH access**: Do you want me to attempt SSH to the server to verify resources, or will you run setup commands and paste me the outputs?
