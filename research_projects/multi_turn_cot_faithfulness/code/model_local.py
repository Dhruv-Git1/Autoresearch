"""
Drop-in replacement for model_openai.py that uses local HuggingFace models.

Routes by model name:
- "deepseek-r1-distill-qwen-7b"  -> DeepSeek-R1-Distill-Qwen-7B (assistant under test)
- "qwen2.5-7b-instruct"           -> Qwen2.5-7B-Instruct (user simulator + system verifier)

Exposes: generate(messages, model, ...), generate_json(messages, model, ...)
Same return shape as model_openai's generate() with return_metadata=True.
"""
import os
import re
import json
import time
import threading
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_HOME = os.environ.get("HF_HOME", "/dev/shm/vasudev_hf_cache")
os.environ["HF_HOME"] = HF_HOME

MODEL_REGISTRY = {
    "deepseek-r1-distill-qwen-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "is_reasoning": True,
    },
    "qwen2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "is_reasoning": False,
    },
}

_MODELS: Dict[str, Any] = {}
_TOKENIZERS: Dict[str, Any] = {}
_LOCK = threading.Lock()


def _load_model(model_name: str):
    """Lazy-load and cache an HF model on the GPU.

    Env vars:
      LOAD_IN_8BIT=1   — load both models in int8 via bitsandbytes (~7 GB each vs 14 GB)
      LOAD_IN_4BIT=1   — load both models in NF4 via bitsandbytes (~4 GB each)
    Use these when GPU is shared and VRAM is tight.
    """
    if model_name in _MODELS:
        return _MODELS[model_name], _TOKENIZERS[model_name]

    with _LOCK:
        if model_name in _MODELS:
            return _MODELS[model_name], _TOKENIZERS[model_name]

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Known: {list(MODEL_REGISTRY)}")
        spec = MODEL_REGISTRY[model_name]
        hf_id = spec["hf_id"]

        load_in_8bit = os.environ.get("LOAD_IN_8BIT", "0") == "1"
        load_in_4bit = os.environ.get("LOAD_IN_4BIT", "0") == "1"

        print(f"[model_local] loading {hf_id} "
              f"({'8-bit' if load_in_8bit else '4-bit' if load_in_4bit else 'bf16'}) ...", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=os.path.join(HF_HOME, "hub"))
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        load_kwargs = dict(
            device_map="cuda:0",
            cache_dir=os.path.join(HF_HOME, "hub"),
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        model.eval()
        print(f"[model_local] loaded {hf_id} in {time.time()-t0:.1f}s", flush=True)
        _MODELS[model_name] = model
        _TOKENIZERS[model_name] = tok
        return model, tok


def _normalize_model_name(model: str) -> str:
    """Map various spellings to our registry keys."""
    m = model.lower().strip()
    aliases = {
        "deepseek-r1-distill-qwen-7b": "deepseek-r1-distill-qwen-7b",
        "deepseek-r1": "deepseek-r1-distill-qwen-7b",
        "r1-distill-7b": "deepseek-r1-distill-qwen-7b",
        "r1": "deepseek-r1-distill-qwen-7b",
        "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
        "qwen2.5": "qwen2.5-7b-instruct",
        "qwen": "qwen2.5-7b-instruct",
    }
    return aliases.get(m, m)


def format_messages(messages, variables=None):
    """Replicates model_openai.format_messages: substitute [[KEY]] placeholders in last user msg."""
    if not variables:
        return messages
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]
    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert isinstance(v, str), f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if keys_still_in_prompt:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")
    return messages


# ---------------------------------------------------------------------------
# Public API: generate / generate_json
# ---------------------------------------------------------------------------
def generate(
    messages: List[Dict[str, str]],
    model: str = "qwen2.5-7b-instruct",
    timeout: int = 60,                # ignored locally
    max_retries: int = 3,
    temperature: float = 1.0,
    is_json: bool = False,
    return_metadata: bool = False,
    max_tokens: Optional[int] = None,
    variables: Optional[Dict[str, str]] = None,
):
    if variables is None:
        variables = {}
    norm = _normalize_model_name(model)
    spec = MODEL_REGISTRY.get(norm)
    if spec is None:
        raise ValueError(f"Unknown model {model} (normalized={norm})")

    messages = format_messages(messages, variables)
    hf_model, tok = _load_model(norm)

    # IMPORTANT: Strip <think>...</think> blocks from ALL prior assistant turns
    # before resending to ANY model.
    #   - For R1 generating turn N: DeepSeek's own model card says to drop prior
    #     thinking; otherwise context grows ~2-5K tokens per turn and hits the
    #     32K attention budget after ~5 turns.
    #   - For Qwen2.5 (user simulator + system verifier): it doesn't need
    #     R1's internal monologue and sending it just bloats its input window.
    cleaned = []
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and "</think>" in m["content"]:
            cleaned.append({"role": "assistant", "content": m["content"].split("</think>", 1)[1].strip()})
        else:
            cleaned.append(m)
    messages = cleaned

    # Apply chat template. Qwen models support add_generation_prompt.
    # For R1-Distill, the chat template auto-injects opening <think> tag.
    if is_json and not spec["is_reasoning"]:
        # Append a minimal JSON-mode hint to the last system or first user msg if not present
        sys_msg_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
        json_hint = "\nYou must respond with a single valid JSON object only, with no extra text before or after."
        if sys_msg_idx is not None:
            if "JSON" not in messages[sys_msg_idx]["content"]:
                messages[sys_msg_idx]["content"] += json_hint
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant." + json_hint}] + messages

    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    if max_tokens is None:
        max_tokens = 10000 if spec["is_reasoning"] else 1024

    # Hard cap reasoning-model output. The simulator passes 10000 by default,
    # but on GSM8K math R1's typical CoT is 1-3K tokens. A 4000 ceiling lets
    # the chain finish naturally while preventing 8-shard conversations from
    # taking 25+ minutes.
    if spec["is_reasoning"]:
        max_tokens = min(max_tokens, int(os.environ.get("R1_MAX_NEW_TOKENS", "4000")))

    inputs = tok(prompt_text, return_tensors="pt", truncation=False).to(hf_model.device)
    prompt_token_count = inputs["input_ids"].shape[1]

    # Build a robust eos_token_id list so the model actually stops at end-of-turn
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(int(tok.eos_token_id))
    for tk in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tok.convert_tokens_to_ids(tk)
            if tid is not None and tid != tok.unk_token_id and tid not in eos_ids:
                eos_ids.append(int(tid))
        except Exception:
            pass

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids if eos_ids else None,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    last_err = None
    for attempt in range(max_retries):
        try:
            with torch.inference_mode():
                out = hf_model.generate(**inputs, **gen_kwargs)
            break
        except Exception as e:
            last_err = e
            torch.cuda.empty_cache()
            time.sleep(1)
    else:
        raise RuntimeError(f"Generation failed after {max_retries} attempts: {last_err}")

    completion_ids = out[0, prompt_token_count:]
    completion_text = tok.decode(completion_ids, skip_special_tokens=False)
    # strip end-of-turn tokens
    for stop_str in ["<|im_end|>", "<|endoftext|>", tok.eos_token or ""]:
        if stop_str and completion_text.endswith(stop_str):
            completion_text = completion_text[: -len(stop_str)]
    completion_text = completion_text.strip()

    # If a reasoning model hits max_new_tokens without emitting </think>,
    # force-close the thinking block so split_thinking() returns ("...", "")
    # rather than mistakenly treating the truncated thinking as the answer.
    if spec["is_reasoning"] and "</think>" not in completion_text and len(completion_ids) >= max_tokens - 5:
        completion_text = completion_text.rstrip() + "\n</think>\n\n[response truncated at max_new_tokens]"

    completion_token_count = len(completion_ids)

    if is_json:
        # Some models wrap JSON in code fences or add prose. Try to extract.
        completion_text = _extract_json(completion_text)

    if not return_metadata:
        return completion_text

    return {
        "message": completion_text,
        "total_tokens": int(prompt_token_count + completion_token_count),
        "prompt_tokens": int(prompt_token_count),
        "prompt_tokens_cached": 0,
        "completion_tokens": int(completion_token_count),
        "total_usd": 0.0,
    }


def _sanitize_json_escapes(text: str) -> str:
    r"""Drop invalid backslash-escapes that Qwen2.5 sometimes emits.

    JSON only allows \\, \/, \b, \f, \n, \r, \t, \" and \uXXXX. Sequences like
    \$, \(, \[, \% etc. cause json.loads to fail. We strip the offending
    backslash so the literal character survives.
    """
    return re.sub(r'\\(?=[^"\\/bfnrtu])', '', text)


def generate_json(messages, model="qwen2.5-7b-instruct", **kwargs):
    """JSON-mode generate. Returns dict with parsed JSON in 'message'."""
    # Drop any caller-provided values for the keys we set explicitly to avoid
    # 'multiple values for keyword argument' from callers like system_agent.
    kwargs.pop("is_json", None)
    kwargs.pop("return_metadata", None)
    response = generate(messages, model=model, is_json=True, return_metadata=True, **kwargs)
    raw = response["message"]
    parsed = None
    last_err = None

    def _try(s):
        try:
            return json.loads(s), None
        except Exception as ex:
            return None, ex

    # 1) direct
    parsed, last_err = _try(raw)
    # 2) regex-extract first {...} block
    if parsed is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed, last_err = _try(m.group(0))
            if parsed is None:
                # 3) sanitize then retry
                parsed, last_err = _try(_sanitize_json_escapes(m.group(0)))
    # 4) sanitize the raw string and retry
    if parsed is None:
        parsed, last_err = _try(_sanitize_json_escapes(raw))
    if parsed is None:
        raise ValueError(f"Could not parse JSON from model output. Error: {last_err}\nRaw: {raw[:500]}")
    response["message"] = parsed
    return response


def _extract_json(text: str) -> str:
    """Strip common wrappers around JSON outputs (code fences, R1 thinking blocks).

    R1-Distill emits <think>...</think> before the answer; we discard the thinking
    portion when JSON-mode is requested (used only by user simulator and system verifier
    which are NOT R1; this is defense in depth).
    """
    # If a <think>...</think> block is present, keep what's after the closing tag.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Otherwise return text as-is; generate_json will do the {.+} extraction
    return text


# ---------------------------------------------------------------------------
# Helper exposed so research code can pull <think> separately
# ---------------------------------------------------------------------------
def split_thinking(response_text: str):
    """For R1-Distill outputs: return (thinking_text, answer_text).

    R1-Distill format: optional opening <think> ... </think> then the answer.
    The chat template typically injects the opening <think>, so the model's first
    output token is the thinking content. If <think> is not in the text we treat
    everything as 'answer'.
    """
    # Many chat templates insert <think> automatically; the model output begins
    # with the thinking content followed by </think>.
    if "</think>" in response_text:
        parts = response_text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer
    if "<think>" in response_text:
        # Edge case: model wrote <think> but never closed it
        thinking = response_text.split("<think>", 1)[1].strip()
        return thinking, ""
    # No thinking block at all
    return "", response_text


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-r1-distill-qwen-7b")
    parser.add_argument("--prompt", type=str, default="What is 2 + 3? Answer in one word.")
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    msgs = [{"role": "user", "content": args.prompt}]
    out = generate(msgs, model=args.model, max_tokens=args.max_tokens, return_metadata=True, temperature=0.7)
    print(json.dumps({k: (v if k != "message" else v[:1500]) for k, v in out.items()}, indent=2))

    if "deepseek-r1" in args.model:
        thinking, answer = split_thinking(out["message"])
        print("\n--- THINKING ---\n", thinking[:1000])
        print("\n--- ANSWER ---\n", answer[:1000])
