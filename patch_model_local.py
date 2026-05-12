"""Patches model_local.py to add Qwen3-14B and DeepSeek-R1-Distill-Llama-8B support."""
import re

path = "/home/vasudev_majhi_2021/multi_turn_cot/lost_in_conversation/model_local.py"
with open(path) as f:
    src = f.read()

# 1. Add new models to MODEL_REGISTRY
old_registry = '''\
MODEL_REGISTRY = {
    "deepseek-r1-distill-qwen-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "is_reasoning": True,
    },
    "qwen2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "is_reasoning": False,
    },
}'''

new_registry = '''\
MODEL_REGISTRY = {
    "deepseek-r1-distill-qwen-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "is_reasoning": True,
    },
    "qwen2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "is_reasoning": False,
    },
    "deepseek-r1-distill-llama-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "is_reasoning": True,
    },
    "qwen3-14b": {
        "hf_id": "Qwen/Qwen3-14B",
        "is_reasoning": True,
        "inject_thinking": True,
        # Cached in ~/.cache/huggingface/hub; set HF_HOME=~/.cache/huggingface at runtime
    },
}'''

assert old_registry in src, "MODEL_REGISTRY block not found — check model_local.py"
src = src.replace(old_registry, new_registry)

# 2. Add aliases to _normalize_model_name
old_aliases = '''\
    aliases = {
        "deepseek-r1-distill-qwen-7b": "deepseek-r1-distill-qwen-7b",
        "deepseek-r1": "deepseek-r1-distill-qwen-7b",
        "r1-distill-7b": "deepseek-r1-distill-qwen-7b",
        "r1": "deepseek-r1-distill-qwen-7b",
        "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
        "qwen2.5": "qwen2.5-7b-instruct",
        "qwen": "qwen2.5-7b-instruct",
    }'''

new_aliases = '''\
    aliases = {
        "deepseek-r1-distill-qwen-7b": "deepseek-r1-distill-qwen-7b",
        "deepseek-r1": "deepseek-r1-distill-qwen-7b",
        "r1-distill-7b": "deepseek-r1-distill-qwen-7b",
        "r1": "deepseek-r1-distill-qwen-7b",
        "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
        "qwen2.5": "qwen2.5-7b-instruct",
        "qwen": "qwen2.5-7b-instruct",
        "deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
        "r1-llama": "deepseek-r1-distill-llama-8b",
        "r1-llama-8b": "deepseek-r1-distill-llama-8b",
        "qwen3-14b": "qwen3-14b",
        "qwen3": "qwen3-14b",
    }'''

assert old_aliases in src, "aliases block not found — check model_local.py"
src = src.replace(old_aliases, new_aliases)

# 3. Add inject_thinking support in generate() — patch apply_chat_template call
old_template_call = '''\
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )'''

new_template_call = '''\
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # For models that require an explicit <think> opening (e.g. Qwen3),
    # append it directly after the generation prompt. Do NOT rstrip — the
    # template already ends with \\n after the assistant role token.
    if spec.get("inject_thinking"):
        prompt_text = prompt_text + "<think>\\n"'''

assert old_template_call in src, "apply_chat_template call not found — check model_local.py"
src = src.replace(old_template_call, new_template_call)

with open(path, "w") as f:
    f.write(src)

print("model_local.py patched successfully.")
print("Changes: +qwen3-14b, +deepseek-r1-distill-llama-8b in registry; inject_thinking support added.")
