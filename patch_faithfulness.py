"""
Patches faithfulness_counterfactual.py so regenerate_answer_with_truncated_cot()
works for both R1-Distill models (which auto-inject <think>) and Qwen3-style
models (which don't). It auto-detects by probing the chat template output.
"""

path = "/home/vasudev_majhi_2021/multi_turn_cot/lost_in_conversation/faithfulness_counterfactual.py"
with open(path) as f:
    src = f.read()

old_fn = '''\
def regenerate_answer_with_truncated_cot(
    model, tokenizer, messages: List[Dict[str, str]], truncated_thinking: str, max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Apply chat template, append truncated thinking + closing </think>, then
    let the model produce the answer.

    R1-Distill's chat template inserts `<think>\\n` at the start of the assistant's
    response, so the prompt we build ends with `<think>\\n{truncated}\\n</think>\\n\\n`
    and the model continues with just the answer text.
    """
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Force-close the thinking block so the model produces the answer directly.
    forced_prefix = f"{truncated_thinking}\\n</think>\\n\\n"
    full_prompt = base_prompt + forced_prefix'''

new_fn = '''\
def regenerate_answer_with_truncated_cot(
    model, tokenizer, messages: List[Dict[str, str]], truncated_thinking: str, max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Apply chat template, append truncated thinking + closing </think>, then
    let the model produce the answer.

    R1-Distill models: chat template inserts <think>\\n automatically.
    Qwen3 and similar: chat template does NOT insert <think>, so we add it.
    Auto-detected by probing apply_chat_template on a dummy message.
    """
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Auto-detect whether the chat template already injects <think>.
    _probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
    )
    think_open = "<think>" if "<think>" not in _probe else ""
    # Force-close the thinking block so the model produces the answer directly.
    forced_prefix = f"{think_open}{truncated_thinking}\\n</think>\\n\\n"
    full_prompt = base_prompt + forced_prefix'''

assert old_fn in src, "regenerate_answer_with_truncated_cot signature not found — check file"
src = src.replace(old_fn, new_fn)

with open(path, "w") as f:
    f.write(src)

print("faithfulness_counterfactual.py patched successfully.")
print("Auto-detects <think> injection need for Qwen3/non-R1 models.")
