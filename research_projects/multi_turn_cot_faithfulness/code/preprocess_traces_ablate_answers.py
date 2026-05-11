"""Preprocess sharded conversation traces to replace each assistant turn's
ANSWER text (the part after </think>) with a placeholder, while keeping
the thinking-block framing intact.

Rationale: in the standard pipeline, model_local.py strips prior <think>
blocks before generation, so the post-</think> answer text is the entire
carrier of the model's prior commitment in the visible context. Replacing
that text with a fixed placeholder is a complete information-suppression
of the model's prior commitment while holding everything else (user shards,
system prompt, turn count, conversation position) identical to the original.

Usage:
    python preprocess_traces_ablate_answers.py \\
        --src_dir  results/phase4 \\
        --dst_dir  results/ablation_no_prior_answer_traces

Behaviour per assistant message:
    Original:  "<think>{thinking}</think>\\n\\n{answer}"
    Rewritten: "<think>{thinking}</think>\\n\\n[PRIOR ANSWER REDACTED]"

Edge cases:
    - No </think> tag (rare; truncated max_tokens path): replace entire
      content with the placeholder.
    - "[response truncated at max_new_tokens]" markers in the answer: the
      content of the placeholder is the same regardless of whether the
      original answer was truncated.
"""
import argparse
import copy
import glob
import json
import os


PLACEHOLDER = "[PRIOR ANSWER REDACTED]"


def ablate_assistant_content(content: str) -> str:
    """Replace the post-</think> portion of an assistant message with the
    placeholder. If no </think> is present, replace the entire content.
    Preserves the <think>...</think> wrapper so the textual structure is
    minimally altered."""
    if not isinstance(content, str):
        return PLACEHOLDER
    if "</think>" in content:
        head, _ = content.split("</think>", 1)
        return f"{head}</think>\n\n{PLACEHOLDER}"
    return PLACEHOLDER


def ablate_trace(payload: dict) -> dict:
    """Return a deep-copied trace payload with every assistant message's
    answer text replaced by the placeholder."""
    out = copy.deepcopy(payload)
    trace = out.get("trace", [])
    for m in trace:
        if m.get("role") == "assistant":
            m["content"] = ablate_assistant_content(m.get("content", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True,
                    help="Source directory with trace_sharded_*.json files")
    ap.add_argument("--dst_dir", required=True,
                    help="Destination directory for ablated traces")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    src_files = sorted(glob.glob(os.path.join(args.src_dir, "trace_sharded_*.json")))
    print(f"found {len(src_files)} trace files in {args.src_dir}")

    n_assistant_total = 0
    for sf in src_files:
        with open(sf, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ablated = ablate_trace(payload)
        n_assistant = sum(1 for m in ablated.get("trace", []) if m.get("role") == "assistant")
        n_assistant_total += n_assistant
        dst_path = os.path.join(args.dst_dir, os.path.basename(sf))
        with open(dst_path, "w", encoding="utf-8") as fh:
            json.dump(ablated, fh, ensure_ascii=False)
        print(f"  {os.path.basename(sf)}: {n_assistant} assistant turns redacted")

    print(f"\nwrote {len(src_files)} ablated traces "
          f"({n_assistant_total} assistant turns) to {args.dst_dir}")
    print(f"placeholder used: {PLACEHOLDER!r}")


if __name__ == "__main__":
    main()
