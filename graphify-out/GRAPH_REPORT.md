# Graph Report - .  (2026-05-07)

## Corpus Check
- Corpus is ~1,409 words - fits in a single context window. You may not need a graph.

## Summary
- 32 nodes · 48 edges · 6 communities
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 10 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Research Philosophy and Process|Research Philosophy and Process]]
- [[_COMMUNITY_Agent Execution and Feasibility|Agent Execution and Feasibility]]
- [[_COMMUNITY_AutoVoila System Overview|AutoVoila System Overview]]
- [[_COMMUNITY_Rigor and Methodology|Rigor and Methodology]]
- [[_COMMUNITY_Research Quality Criteria|Research Quality Criteria]]
- [[_COMMUNITY_Concrete Examples and Fruitfulness|Concrete Examples and Fruitfulness]]

## God Nodes (most connected - your core abstractions)
1. `voila.md â€” Autonomous Research Agent Prompt` - 8 edges
2. `Exploration Sprint Template` - 8 edges
3. `Research Question Sharpener Template` - 8 edges
4. `AutoVoila README` - 7 edges
5. `Section 2: The Why â€” Fruitfulness` - 6 edges
6. `Research Philosophy Document` - 5 edges
7. `Great Research Question Criteria` - 5 edges
8. `Lossfunk` - 3 edges
9. `Surprisingness Criterion` - 3 edges
10. `Fruitfulness Criterion` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Section 4: The So What â€” Impact Depth` --semantically_similar_to--> `Fruitfulness Criterion`  [INFERRED] [semantically similar]
  autovoila/templates/research_question_sharpner.pdf → autovoila/research-philosophy.md
- `Research Phases (Exploration â†’ Sharpening â†’ Execution â†’ Writing)` --semantically_similar_to--> `LossFunk Research Flow`  [INFERRED] [semantically similar]
  autovoila/research-philosophy.md → autovoila/templates/exploration_sprint.pdf
- `Research Claim as Prior Intuition` --semantically_similar_to--> `What-If Curiosity Prompt`  [INFERRED] [semantically similar]
  autovoila/research-philosophy.md → autovoila/templates/exploration_sprint.pdf
- `Fruitfulness Criterion` --conceptually_related_to--> `Section 2: The Why â€” Fruitfulness`  [INFERRED]
  autovoila/research-philosophy.md → autovoila/templates/research_question_sharpner.pdf
- `AutoVoila README` --references--> `Exploration Sprint Template`  [EXTRACTED]
  autovoila/README.md → autovoila/templates/exploration_sprint.pdf

## Hyperedges (group relationships)
- **LossFunk Full Research Pipeline** — sprint_what_if, template_exploration_sprint, sprint_decision_gsp, template_rq_sharpener, philosophy_research_phases, voila_main_prompt [EXTRACTED 0.95]
- **Four Criteria for Research Question Quality** — philosophy_surprisingness, philosophy_fruitfulness, philosophy_rigor, philosophy_feasibility [EXTRACTED 1.00]
- **Research Question Sharpener Four-Section Structure** — rqs_the_what, rqs_the_why, rqs_the_how, rqs_the_so_what [EXTRACTED 1.00]

## Communities (6 total, 0 thin omitted)

### Community 0 - "Research Philosophy and Process"
Cohesion: 0.33
Nodes (7): Research Claim as Prior Intuition, Research Phases (Exploration â†’ Sharpening â†’ Execution â†’ Writing), Research Philosophy Document, LossFunk Research Flow, Time-Box Discipline, What-If Curiosity Prompt, Exploration Sprint Template

### Community 1 - "Agent Execution and Feasibility"
Cohesion: 0.33
Nodes (6): Feasibility Criterion, Codex Feedback Integration, Submittable Paper Output (LaTeX + PDF), voila.md â€” Autonomous Research Agent Prompt, System Resource Discovery, AI Conference Reviewer Self-Check

### Community 2 - "AutoVoila System Overview"
Cohesion: 0.4
Nodes (6): AutoVoila README, CCO (Claude Code Orchestrator), Claude Code, Lossfunk, Section 1: The What â€” Specific Claim, Fruitfulness Pre-Check

### Community 3 - "Rigor and Methodology"
Cohesion: 0.4
Nodes (5): Rigor and Claim Scope Criterion, Section 3: The How â€” Experimental Design, Section 4: The So What â€” Impact Depth, Graduate / Shelve / Pivot Decision, Research Question Sharpener Template

### Community 4 - "Research Quality Criteria"
Cohesion: 0.5
Nodes (4): Fruitfulness Criterion, Great Research Question Criteria, Surprisingness Criterion, Written Expectations Before Sprint

### Community 5 - "Concrete Examples and Fruitfulness"
Cohesion: 0.5
Nodes (4): Adversarial Examples (High Fruitfulness Example), Grokking (High Fruitfulness Example), Sycophancy Paper (High Fruitfulness Example), Section 2: The Why â€” Fruitfulness

## Knowledge Gaps
- **6 isolated node(s):** `Time-Box Discipline`, `Codex Feedback Integration`, `Submittable Paper Output (LaTeX + PDF)`, `Grokking (High Fruitfulness Example)`, `Adversarial Examples (High Fruitfulness Example)` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Research Question Sharpener Template` connect `Rigor and Methodology` to `Research Philosophy and Process`, `Agent Execution and Feasibility`, `AutoVoila System Overview`, `Concrete Examples and Fruitfulness`?**
  _High betweenness centrality (0.310) - this node is a cross-community bridge._
- **Why does `voila.md â€” Autonomous Research Agent Prompt` connect `Agent Execution and Feasibility` to `Research Philosophy and Process`, `AutoVoila System Overview`, `Rigor and Methodology`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Why does `Exploration Sprint Template` connect `Research Philosophy and Process` to `Agent Execution and Feasibility`, `AutoVoila System Overview`, `Rigor and Methodology`, `Research Quality Criteria`?**
  _High betweenness centrality (0.226) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Section 2: The Why â€” Fruitfulness` (e.g. with `Fruitfulness Criterion` and `Fruitfulness Pre-Check`) actually correct?**
  _`Section 2: The Why â€” Fruitfulness` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Time-Box Discipline`, `Codex Feedback Integration`, `Submittable Paper Output (LaTeX + PDF)` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._