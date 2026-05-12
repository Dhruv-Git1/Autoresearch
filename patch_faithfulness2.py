"""Patches faithfulness_counterfactual.py to support non-R1 thinking models (Qwen3 etc).
Uses line-by-line matching to avoid multi-line string escape issues.
"""
import sys

path = "/home/vasudev_majhi_2021/multi_turn_cot/lost_in_conversation/faithfulness_counterfactual.py"
with open(path) as f:
    lines = f.readlines()

# Find the line with forced_prefix inside regenerate_answer_with_truncated_cot
target = '    forced_prefix = f"{truncated_thinking}'
idx = None
for i, line in enumerate(lines):
    if line.startswith(target):
        idx = i
        break

if idx is None:
    print("ERROR: could not find forced_prefix line. Dumping context around regenerate_answer:")
    for i, line in enumerate(lines):
        if "forced_prefix" in line:
            print(f"  L{i+1}: {repr(line)}")
    sys.exit(1)

print(f"Found forced_prefix at line {idx+1}: {repr(lines[idx])}")

# Build the replacement lines (3 lines instead of 1)
indent = "    "
new_lines = [
    f"{indent}# Auto-detect whether chat template already injects <think> (R1-Distill does; Qwen3 doesn't).\n",
    f"{indent}_probe = tokenizer.apply_chat_template(\n",
    f"{indent}    [{{\"role\": \"user\", \"content\": \"x\"}}], tokenize=False, add_generation_prompt=True,\n",
    f"{indent})\n",
    f"{indent}think_open = \"<think>\" if \"<think>\" not in _probe else \"\"\n",
    f"{indent}forced_prefix = f\"{{think_open}}{{truncated_thinking}}\\n</think>\\n\\n\"\n",
]

lines[idx:idx+1] = new_lines

with open(path, "w") as f:
    f.writelines(lines)

print(f"Patched: replaced line {idx+1} with {len(new_lines)} lines.")
print("Verification — lines around change:")
start = max(0, idx - 1)
end = min(len(lines), idx + len(new_lines) + 2)
for i in range(start, end):
    print(f"  L{i+1}: {lines[i]}", end="")
