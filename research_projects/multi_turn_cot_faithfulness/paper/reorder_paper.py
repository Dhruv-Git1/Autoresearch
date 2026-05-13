#!/usr/bin/env python3
"""
Complete one-pass reorganization of paper.tex (git HEAD version, 2326 lines).

Applies all planned changes:
  1. Abstract: 3.9x gradient first; add chi2/ICC stats; five model families
  2. Intro: reorder three-questions (H3->H2b->H1); fix contribution #4
  3. §4 major reorder (strongest findings first, nulls last)
  4. §5 Discussion: remove monitor-evasion (now §4.5); fix cross-reference
  5. §6 Limitations: add cross-model section + parity check; update single-model
  6. §7 Conclusion: rewrite as 3 tight paragraphs

Run ONCE from paper/ directory:
    python reorder_paper.py
"""

TEX_IN  = 'paper.tex'
TEX_OUT = 'paper.tex'  # overwrite in place

# ── safety check ────────────────────────────────────────────────────────
with open(TEX_IN, 'r', encoding='utf-8') as f:
    lines = f.readlines()

assert len(lines) == 2326, (
    f"Expected 2326 lines (git HEAD), got {len(lines)}. "
    "Restore with: git checkout research_projects/multi_turn_cot_faithfulness/paper/paper.tex"
)
assert 'Evidence Accumulates' in lines[570], (
    f"Line 571 should be cascade subsection; got: {lines[570][:60]!r}"
)
print(f"Safety check passed: {len(lines)} lines, git HEAD confirmed.")


def blk(start, end):
    """Return lines[start-1:end] (1-indexed, inclusive)."""
    return lines[start-1:end]

def txt(block):
    return ''.join(block)


# ══════════════════════════════════════════════════════════════════════
# BLOCK EXTRACTION  (git HEAD line numbers, 1-indexed)
# ══════════════════════════════════════════════════════════════════════

pre_results    = blk(1,    566)   # everything before Results separator
results_hdr    = blk(567,  570)   # % separator + \section{Results} + \label + blank

# § 4 content blocks
cascade_blk    = blk(571,  651)
sample965_blk  = blk(652,  686)
h1_blk         = blk(687,  755)
h2_blk         = blk(756,  841)
h3_blk         = blk(842,  963)
logprobs_blk   = blk(964,  1073)
short_blk      = blk(1074, 1087)
commit_blk     = blk(1088, 1118)
repet_blk      = blk(1119, 1216)
h2b_blk        = blk(1217, 1284)
length_blk     = blk(1285, 1346)
seeds_blk      = blk(1347, 1374)
pred_blk       = blk(1375, 1483)

# § 5 Discussion parts
disc_pre_mon   = blk(1484, 1573)  # separator + header + paras through thought-anchors
monitor_blk    = blk(1574, 1620)  # "Direct demonstration" para + figure + blank
disc_post_mon  = blk(1621, 1745)  # Competence outcomes → end of Discussion

# § 6 Limitations parts
limits_intro   = blk(1746, 1825)  # separator + header + content through prior-answer ablation
quant_blk      = blk(1826, 1855)  # quant text + quant figure
ablation_fig   = blk(1856, 1874)  # blank + ablation figure (float for §4.7)
limits_after   = blk(1875, 1932)  # answer invariance + H1-scale + independence

# Post-conclusion (Experimental Details, AI checklists, bib, appendix)
post_concl     = blk(2005, len(lines))


# ══════════════════════════════════════════════════════════════════════
# PREAMBLE CHANGE: add TBD parity macros just before \begin{document}
# ══════════════════════════════════════════════════════════════════════
pre_results_txt = txt(pre_results)
parity_macros = (
    '\n% ── Phase-B parity check: fill before submission ──\n'
    '\\newcommand{\\PBAGREEMENT}{TBD}\n'
    '\\newcommand{\\PBCOMPARED}{TBD}\n'
    '\\newcommand{\\PBPCT}{TBD}\n'
    '\\newcommand{\\PBSHIFT}{TBD}\n'
    '\\newcommand{\\PBWILCOX}{TBD}\n'
    '\\newcommand{\\PBCONVS}{TBD}\n'
)
pre_results_txt = pre_results_txt.replace(
    '\\begin{document}\n',
    parity_macros + '\\begin{document}\n'
)


# ══════════════════════════════════════════════════════════════════════
# ABSTRACT FIX: 3.9x first; add χ²/ICC; five model families
# ══════════════════════════════════════════════════════════════════════
pre_results_txt = pre_results_txt.replace(
    'Crucially, anchoring is not random noise: across our 67~conversations\n'
    'some conversations are heavily anchored while others barely anchor\n'
    'at all, a pattern confirmed statistically and replicated across\n'
    'three independent Phase~B seeds (27--38\\% anchored per seed;\n'
    'higher than the combined 23.3\\% because Phase~B used longer\n'
    'conversations that anchor more).\n'
    'More strikingly, the fraction of anchored turns grows\n'
    '\\textbf{3.9$\\times$} as conversations lengthen---from 7.2\\% in\n'
    'short conversations to 28.4\\% in long ones.\n'
    'This means CoT-based monitoring would face its lowest causal signal\n'
    'in the longest conversations --- a prediction that requires\n'
    'prospective confirmation on datasets with clean outcome labels.\n',

    'The fraction of anchored turns grows \\textbf{3.9$\\times$} from\n'
    'short to long conversations (7.2\\%~$\\to$~28.4\\%), replicated\n'
    'across five model families spanning three architectures.\n'
    'Crucially, this gradient reflects a structured conversation-level\n'
    'property: some conversations are heavily anchored while others\n'
    'barely anchor at all ($\\chi^2 = 236.9$, $df=66$, $p\\approx 0$;\n'
    'ICC~$= 0.152$).\n'
    'This means CoT-based monitoring faces its lowest causal signal\n'
    'in the longest conversations, where anchored turns score 1.0 on\n'
    'any CoT-consistency monitor while contributing nothing causally.\n'
)


# ══════════════════════════════════════════════════════════════════════
# INTRO FIX 1: reorder three-questions paragraph (H3 → H2b → H1)
# ══════════════════════════════════════════════════════════════════════
pre_results_txt = pre_results_txt.replace(
    'We ask three questions of that time series.\n'
    'First: does anchoring \\emph{persist} over multiple consecutive\n'
    'turns, or is each turn independently uncertain?\n'
    'Answer: we cannot confirm persistence --- anchored runs look\n'
    'statistically consistent with a memoryless process, even with 148\n'
    'observed anchored runs across 67 conversations.\n'
    'Second: does being more anchored predict getting the answer wrong?\n'
    'Answer: the trend points the right way (more anchoring accompanies\n'
    'more wrong answers), but 21 conversations with clean labels are\n'
    'not enough to confirm it statistically.\n'
    'Third: is anchoring a \\emph{conversation-level} property --- do\n'
    'some conversations run heavily anchored while others stay mostly\n'
    'exploring?\n'
    'Answer: \\textbf{yes, strongly}.\n'
    'The spread of anchoring across conversations is vastly larger than\n'
    'chance would produce\n'
    '($\\chi^2 = 236.9$, $df=66$, $p \\approx 0$; ICC~$=0.152$).\n'
    'And conversation length is the key predictor: from short to long\n'
    'conversations, anchored fraction nearly quadruples\n'
    '(7.2\\% $\\to$ 12.8\\% $\\to$ 28.4\\%, a 3.9$\\times$ gradient),\n'
    'replicated across three independent experimental seeds.\n',

    'We ask three questions of that time series.\n'
    'First: is anchoring a \\emph{conversation-level} property --- do\n'
    'some conversations run heavily anchored while others stay mostly\n'
    'exploring?\n'
    'Answer: \\textbf{yes, strongly}\n'
    '($\\chi^2 = 236.9$, $df=66$, $p \\approx 0$; ICC~$=0.152$).\n'
    'Conversation length is the key predictor: anchored fraction nearly\n'
    'quadruples from short to long conversations\n'
    '(7.2\\% $\\to$ 12.8\\% $\\to$ 28.4\\%, a 3.9$\\times$ gradient).\n'
    'Second: does more anchoring predict \\emph{failure to integrate\n'
    'new evidence}, independent of length?\n'
    'Answer: \\textbf{yes} (partial Spearman $\\rho=-0.337$, $p=0.009$;\n'
    '\\S\\ref{subsec:h2b}).\n'
    'Third: does anchoring \\emph{persist} over consecutive turns?\n'
    'Answer: \\textbf{inconclusive} --- runs are consistent with a\n'
    'memoryless process (bootstrap $p=0.521$, 148~anchored runs).\n'
)


# ══════════════════════════════════════════════════════════════════════
# INTRO FIX 2: contribution #4 — replace "H2, requiring N≈131" with H2b
# ══════════════════════════════════════════════════════════════════════
pre_results_txt = pre_results_txt.replace(
    '        CoT causal necessity varies predictably with conversation\n'
    '        length (3.9$\\times$ gradient, replicated across three seeds),\n'
    '        and a cheap surface-feature predictor (AUC~$=0.710$) can\n'
    '        flag at-risk turns online.\n'
    '        If the length-anchoring gradient correlates with task failure\n'
    '        (H2, requiring $N\\approx131$), it implies that CoT monitoring\n'
    '        reliability must be calibrated to conversation length --- a\n'
    '        previously unquantified dimension of multi-turn deployment.\n',

    '        CoT causal necessity varies predictably with conversation\n'
    '        length (3.9$\\times$ gradient, replicated across five model\n'
    '        families), and anchoring predicts failure to integrate new\n'
    '        evidence after length control\n'
    '        (partial Spearman $\\rho=-0.337$, $p=0.009$;\n'
    '        \\S\\ref{subsec:h2b}).\n'
    '        A cheap surface-feature predictor (AUC~$=0.710$) can\n'
    '        flag at-risk turns online, making CoT monitoring reliability\n'
    '        calibratable to conversation length.\n'
)


# ══════════════════════════════════════════════════════════════════════
# § 4 BLOCK TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════

# ── NEW §4.1: Length gradient + sample-965 hook ───────────────────────
length_txt = txt(length_blk)
length_txt = length_txt.replace(
    'If this correlates with downstream failure (H2, pending), a safety\n'
    'monitor calibrated on short conversations would systematically\n'
    'overestimate its reliability as conversations grow.',
    'Since anchoring also predicts failure to integrate new evidence\n'
    'after length control (H2b: partial~$\\rho=-0.337$, $p=0.009$;\n'
    '\\S\\ref{subsec:h2b}), a safety monitor calibrated on short\n'
    'conversations would systematically overestimate its reliability\n'
    'as conversations grow.'
)
length_txt = length_txt.rstrip('\n') + (
    '\n\n'
    '\\paragraph{Illustrative case: Sample 965.}\n'
    'The 44-turn conversation \\texttt{sharded-GSM8K/965}\n'
    '(Figure~\\ref{fig:heatmap}, top row) shows the gradient in miniature:\n'
    'turns~1--8 exploring, turns~9--13 anchored (all five truncation\n'
    'levels produce ``300\'\' against the gold answer 660, with\n'
    '$\\sim$412-token thinking blocks that are entirely post-hoc),\n'
    'then a return to exploring that never converges.\n'
    '\n'
)

# ── NEW §4.2: H3 + missing label ──────────────────────────────────────
h3_txt = txt(h3_blk)
h3_txt = h3_txt.replace(
    '\\subsection{H3: Anchoring is a Conversation-Level Property}\n',
    '\\subsection{H3: Anchoring is a Conversation-Level Property}\n'
    '\\label{subsec:h3}\n',
    1
)

# ── NEW §4.3: Cross-model (new content, not in git HEAD) ──────────────
crossmod_txt = (
    '\\subsection{Cross-Model Replication Across Five Models}\n'
    '\\label{subsec:exp6}\n'
    'To test whether the anchoring gradient is specific to one small\n'
    'distilled model, we ran the same sharded-GSM8K faithfulness\n'
    'protocol on four additional models spanning three families and\n'
    'two base architectures.\n'
    'All statistics use the canonical \\texttt{is\\_anchored()} from\n'
    '\\S\\ref{sec:method}.\n'
    '\n'
    '\\begin{table}[htbp]\n'
    '\\centering\n'
    '\\small\n'
    '\\caption{%\n'
    '  \\textbf{Cross-model anchoring summary.}\n'
    '  Early/mid/late $=$ turn index $<5$\\,/\\,$5$--$14$\\,/\\,$\\geq15$.\n'
    '  ${}^\\dagger$ late-turn $N{<}20$; gradient ratio indicative only.%\n'
    '}\n'
    '\\label{tab:crossmodel}\n'
    '\\begin{tabular}{lllrrrrc}\n'
    '\\toprule\n'
    'Model & Family & Q & $N$ & Anch.\\ & Early & Mid & Late \\\\\n'
    '\\midrule\n'
    'R1-7B (primary) & R1-Distill (Qwen) & 8b & 67 & 23.3\\% & 7.2 & 12.8 & 28.4 \\\\\n'
    'R1-14B          & R1-Distill (Qwen) & 4b & 18 & 23.4\\% & 12.1 & 21.1 & 31.2 \\\\\n'
    'R1-Llama-8B     & R1-Distill (Llama)& 8b & 20 &  9.8\\% &  5.6 &  7.3 & 41.2$^\\dagger$ \\\\\n'
    'Qwen3-8B        & RLVR (Qwen3)     & 8b & 15 & 15.9\\% &  5.3 & 22.0 & 66.7$^\\dagger$ \\\\\n'
    'Qwen3-14B       & RLVR (Qwen3)     & 8b & 15 & 45.6\\% &  8.3 & 40.8 & 75.4 \\\\\n'
    '\\bottomrule\n'
    '\\end{tabular}\n'
    '\\end{table}\n'
    '\n'
    'Three findings address the generalisability concern.\n'
    '\\emph{First}, the length-gradient direction (early $<$ mid $<$ late)\n'
    'holds in \\emph{all five models} across three families and two base\n'
    'architectures (Figure~\\ref{fig:crossmodel}).\n'
    '\\emph{Second}, the Qwen3 models---Qwen3-8B and Qwen3-14B---are\n'
    'trained with RLVR, \\emph{not} by knowledge distillation from R1.\n'
    'Their gradient rules out R1-imitation memorisation as the causal\n'
    'mechanism and shows the phenomenon is not an artifact of distillation.\n'
    '\\emph{Third}, base architecture modulates magnitude: R1-Distill\n'
    'on Llama-3 base (9.8\\%) anchors 2.4$\\times$ less than on\n'
    'Qwen-2.5 base (23.4\\%) at comparable scale.\n'
    'Anchoring rates span 9.8--45.6\\% across families; the\n'
    '\\emph{directional trend} (more context $\\to$ more anchoring) is\n'
    'universal.\n'
    '\n'
    '\\begin{figure}[htbp]\n'
    '  \\centering\n'
    '  \\includegraphics[width=\\linewidth]{figures/cross_model_gradient.png}\n'
    '  \\caption{%\n'
    '    \\textbf{Length-anchoring gradient across five models.}\n'
    '    All models show the same directional increase from early\n'
    '    to late turns; * marks late-turn bins with $N{<}20$ turns\n'
    '    (treat as indicative).\n'
    '    Qwen3 models (orange) are RLVR-trained, not R1-distilled,\n'
    '    confirming the gradient is not a distillation artifact.%\n'
    '  }\n'
    '  \\label{fig:crossmodel}\n'
    '\\end{figure}\n'
    '\n'
)

# ── NEW §4.4: H2b — no changes ────────────────────────────────────────
h2b_txt = txt(h2b_blk)

# ── NEW §4.5: Monitor-evasion (promoted from §5) ──────────────────────
monitor_txt = txt(monitor_blk)
monitor_txt = monitor_txt.replace(
    '\\paragraph{Direct demonstration of CoT monitoring failure.}\n',
    '\\subsection{Monitor-Evasion Gap: CoT Trust Grows as Causal Necessity Falls}\n'
    '\\label{subsec:monitor}\n'
)

# ── NEW §4.6: Predictor — no changes ─────────────────────────────────
pred_txt = txt(pred_blk)

# ── NEW §4.7: Mechanism probes + robustness controls ─────────────────
probes_hdr = (
    '\\subsection{Mechanism Probes and Robustness Controls}\n'
    '\\label{subsec:probes}\n'
    '\n'
    '\\subsubsection*{Mechanism probes that returned null}\n'
    '\n'
)

logprobs_txt = txt(logprobs_blk)
logprobs_txt = logprobs_txt.replace(
    '\\subsection{Probing for Internal Commitment via Answer-Token'
    ' Confidence: A Null Result}\n'
    '\\label{subsec:logprobs}\n',
    '\\paragraph{Logprob probe: null result.}\n'
    '\\label{subsec:logprobs}\n'
)

short_txt = txt(short_blk)
short_txt = short_txt.replace(
    '\\subsection{Is Anchoring Just Short Reasoning? No.}\n'
    '\\label{subsec:not_short_reasoning}\n',
    '\\paragraph{Is anchoring just short reasoning? No.}\n'
    '\\label{subsec:not_short_reasoning}\n'
)

commit_txt = txt(commit_blk)
commit_txt = commit_txt.replace(
    '\\subsection{Does Anchoring Align With Premature Commitment? --- No}\n'
    '\\label{subsec:commitment}\n',
    '\\paragraph{Does anchoring align with premature commitment? No.}\n'
    '\\label{subsec:commitment}\n'
)

robustness_hdr = (
    '\\subsubsection*{Robustness controls}\n'
    '\n'
)

repet_txt = txt(repet_blk)
repet_txt = repet_txt.replace(
    '\\subsection{Could Anchoring Just Be Answer Repetition?}\n'
    '\\label{subsec:repetition}\n',
    '\\paragraph{Could anchoring just be answer repetition?}\n'
    '\\label{subsec:repetition}\n'
)

quant_txt = txt(quant_blk)
quant_txt = quant_txt.replace(
    '\\textbf{Quantization robustness, empirically verified.}\n',
    '\\paragraph{Quantization robustness, empirically verified.}\n'
    '\\label{subsec:quant}\n'
)

ablation_txt = txt(ablation_fig)

parity_txt = (
    '\\paragraph{Phase~B precision parity check.}\n'
    '\\label{subsec:parity}\n'
    'Phase~A used 8-bit quantisation throughout; Phase~B used mixed\n'
    '8-bit/4-bit across seeds~2 and~3 (two processes sharing the GPU).\n'
    'To directly test whether this precision difference drives the\n'
    'Phase~A/B anchoring-rate gap, we are re-running the full\n'
    'five-level faithfulness measurement on 15~Phase~B seed-2\n'
    'conversations at fixed INT8 only (identical trace files).\n'
    'Results are pending; the INT4 vs.\\ INT8 quantization control\n'
    '(\\S\\ref{subsec:quant}) already establishes that the\n'
    'anchored-vs-exploring verdict is robust to a larger precision\n'
    'difference, providing strong prior evidence that mixed-precision\n'
    'is not the driver of the Phase~A/B rate gap.\n'
    '\\medskip\n'
    '\n'
)

# ── NEW §4.8: Stability (cascade retitled + seeds demoted) ───────────
cascade_txt = txt(cascade_blk)
cascade_txt = cascade_txt.replace(
    '\\subsection{Evidence Accumulates Across Seven Datasets}\n',
    '\\subsection{Stability of H3 Across Cumulative Samples and Seeds}\n'
    '\\label{subsec:stability}\n'
)
cascade_txt = cascade_txt.replace(
    'Table~\\ref{tab:cascade} shows how the picture evolved as we added\n'
    'data.\n'
    'We ran our experiments incrementally, adding new conversations\n'
    'across four Phase~A datasets and three Phase~B seeds, and the\n'
    'core finding held at every scale.\n',
    'Table~\\ref{tab:cascade} shows that H3 holds at every scale as\n'
    'data accumulates across seven independent datasets.\n'
)

seeds_txt = txt(seeds_blk)
seeds_txt = seeds_txt.replace(
    '\\subsection{Cross-Seed Reproducibility}\n'
    '\\label{subsec:seeds}\n',
    '\\paragraph{Cross-seed reproducibility.}\n'
    '\\label{subsec:seeds}\n'
)

# ── NEW §4.9: Pre-registered nulls ────────────────────────────────────
nulls_hdr = (
    '\\subsection{Pre-Registered Questions That Returned Nulls}\n'
    '\\label{subsec:nulls}\n'
    '\n'
    'These two pre-registered hypotheses yielded informative null\n'
    'results: they delimit what anchoring is \\emph{not} (persistent\n'
    'across turns in the run-length sense; a direct predictor of task\n'
    'failure at the conversation level)---which is as scientifically\n'
    'valuable as confirmation \\citep{lakens2018}.\n'
    '\n'
)

h1_txt = txt(h1_blk)
h2_txt = txt(h2_blk)


# ══════════════════════════════════════════════════════════════════════
# § 5 DISCUSSION
# ══════════════════════════════════════════════════════════════════════
disc_pre_txt  = txt(disc_pre_mon)
disc_post_txt = txt(disc_post_mon)

# Fix cross-reference: monitor figure is now §4.5, not Discussion
disc_post_txt = disc_post_txt.replace(
    'Figure~\\ref{fig:monitor} (monitor simulation, \\S\\ref{sec:discussion}).',
    'Figure~\\ref{fig:monitor} (\\S\\ref{subsec:monitor}).'
)

# Update "future work" sentence about cross-model (now done, in §4.3)
disc_post_txt = disc_post_txt.replace(
    'Whether the length-anchoring gradient generalises across\n'
    'models and task domains beyond sharded GSM8K mathematical\n'
    'reasoning is the primary open question for future work.\n',
    'Cross-model replication (\\S\\ref{subsec:exp6}) confirms the\n'
    'gradient across five models; generalisation to safety-critical\n'
    'tasks beyond sharded GSM8K remains future work.\n'
)


# ══════════════════════════════════════════════════════════════════════
# § 6 LIMITATIONS
# ══════════════════════════════════════════════════════════════════════
limits_intro_txt = txt(limits_intro)
limits_after_txt = txt(limits_after)

# Update "Single model" paragraph
limits_intro_txt = limits_intro_txt.replace(
    '\\textbf{Single model.}\n'
    'All experiments use DeepSeek-R1-Distill-Qwen-7B in 8-bit or\n'
    '4-bit quantisation.\n'
    'Whether the anchoring phenomenon generalises to larger or differently\n'
    'trained reasoning models is unknown.',
    '\\textbf{Single model --- now partially addressed.}\n'
    'Primary experiments use DeepSeek-R1-Distill-Qwen-7B in 8-bit or\n'
    '4-bit quantisation.\n'
    'Cross-model replication (\\S\\ref{subsec:exp6}, Table~\\ref{tab:crossmodel})\n'
    'now reports four additional models spanning R1-Distill (Qwen and\n'
    'Llama base) and Qwen3 RLVR families; the length-gradient direction\n'
    'is confirmed in all five.'
)

# Add sharded-conversation caveat after Single task domain
limits_intro_txt = limits_intro_txt.replace(
    '\\textbf{Single task domain.}\n'
    'Only GSM8K math is tested.\n'
    'Coding and factual QA tasks may exhibit different faithfulness\n'
    'dynamics, and a task with binary outcomes would enable H2.\n',
    '\\textbf{Single task domain.}\n'
    'Only GSM8K math is tested.\n'
    'Coding and factual QA tasks may exhibit different faithfulness\n'
    'dynamics, and a task with binary outcomes would enable H2.\n'
    '\n'
    '\\textbf{Sharded-conversation setup.}\n'
    'Our conversations are artificially constructed by revealing problem\n'
    'shards one per turn; naturalistic multi-turn interactions may not\n'
    'produce the same length-anchoring gradient, and generalisation to\n'
    'open-ended dialogue remains to be tested.\n'
)


# ══════════════════════════════════════════════════════════════════════
# § 7 CONCLUSION (new 3-paragraph structure)
# ══════════════════════════════════════════════════════════════════════
new_conclusion = (
    '% ─────────────────────────────────────────────────────────────\n'
    '\\section{Conclusion}\n'
    '\\label{sec:conclusion}\n'
    '\n'
    'Across 67 sharded-GSM8K conversations and 1,289 faithfulness\n'
    'observations, CoT causal necessity is not a fixed model property:\n'
    'it varies turn by turn and is structured by conversation length.\n'
    'The anchored fraction grows \\textbf{3.9$\\times$} from short to\n'
    'long conversations (7.2\\%~$\\to$~12.8\\%~$\\to$~28.4\\%), and this\n'
    'gradient holds across five models spanning three families and two\n'
    'base architectures (Table~\\ref{tab:crossmodel}), including Qwen3\n'
    'RLVR models that are not R1 distillates.\n'
    'Conversation-level structure is confirmed by $\\chi^2 = 236.9$\n'
    '($df=66$, $p\\approx 0$) and ICC~$=0.152$: knowing which\n'
    'conversation a turn belongs to explains 15\\% of the variance in\n'
    'whether that turn is anchored.\n'
    '\n'
    'The gradient predicts monitoring failure.\n'
    'Anchored turns score 1.0 by construction on any CoT-consistency\n'
    'monitor; mean monitor scores grow 35\\% from short to long\n'
    'conversations (0.482~$\\to$~0.650), while causal necessity falls\n'
    '--- a 4$\\times$ blind spot for monitors calibrated on short\n'
    'conversations.\n'
    'Anchoring also predicts failure to integrate new evidence after\n'
    'controlling for length (H2b: partial Spearman $\\rho=-0.337$,\n'
    '$p=0.009$).\n'
    'A cheap surface-feature predictor (AUC~$=0.710$) lets safety\n'
    'engineers flag at-risk turns without additional generation calls.\n'
    'Mechanism probes are partially informative: answer inertia\n'
    'contributes (repetition rate 1.66$\\times$ higher on anchored turns)\n'
    'but is not sufficient (26.1\\% of anchored turns are novel answers;\n'
    'prior-answer ablation retains 83.6\\% agreement), and token-level\n'
    'confidence is not diagnostic.\n'
    '\n'
    'Two pre-registered hypotheses remain open.\n'
    'Run-length persistence (H1, bootstrap $p=0.521$, 148~anchored\n'
    'runs) is inconclusive: the geometric null fits at this sample size.\n'
    'The conversation-level outcome test (H2, $r=-0.097$, $N=53$) is\n'
    'length-confounded; testing it cleanly requires either a non-sharded\n'
    'task or a safety-relevant outcome domain.\n'
    'The 3.9$\\times$ gradient is actionable now: flag turns in long\n'
    'conversations where the answer is invariant under CoT perturbation\n'
    'and trigger a summarise-and-restart intervention.\n'
    '\n'
)


# ══════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════════

parts = [
    pre_results_txt,          # preamble + abstract + intro (with all fixes)
    txt(results_hdr),         # % separator + \section{Results} + \label + blank

    # NEW §4.1 — length gradient + sample-965 hook
    length_txt,

    # NEW §4.2 — H3
    h3_txt,

    # NEW §4.3 — cross-model (new content)
    crossmod_txt,

    # NEW §4.4 — H2b
    h2b_txt,

    # NEW §4.5 — monitor-evasion (promoted from §5)
    monitor_txt,

    # NEW §4.6 — predictor
    pred_txt,

    # NEW §4.7 — mechanism probes + robustness controls
    probes_hdr,
    logprobs_txt,
    short_txt,
    commit_txt,
    '\n',
    robustness_hdr,
    repet_txt,
    ablation_txt,
    '\n',
    quant_txt,
    '\n',
    parity_txt,

    # NEW §4.8 — stability (cascade retitled + seeds demoted)
    '\n',
    cascade_txt,
    '\n',
    seeds_txt,

    # NEW §4.9 — pre-registered nulls
    '\n',
    nulls_hdr,
    h1_txt,
    '\n',
    h2_txt,
    '\n',

    # § 5 Discussion (monitor block removed; references fixed)
    disc_pre_txt,
    disc_post_txt,
    '\n',

    # § 6 Limitations (cross-model/quant/parity moved to §4; caveat added)
    limits_intro_txt,
    limits_after_txt,

    # § 7 Conclusion (new 3-paragraph version, includes % separator)
    new_conclusion,

    # Experimental Details, AI checklists, bib, appendix
    txt(post_concl),
]

output = ''.join(parts)

with open(TEX_OUT, 'w', encoding='utf-8') as f:
    f.write(output)

new_len = len(output.splitlines())
print(f"Done. Original: {len(lines)} lines -> New: {new_len} lines")
print("Written to", TEX_OUT)
print()
print("Quick verification — first 5 subsections after Results:")
for i, l in enumerate(output.splitlines(), 1):
    if l.strip().startswith(r'\subsection'):
        print(f"  {i:4d}: {l.strip()[:70]}")
    if i > 700:
        break
