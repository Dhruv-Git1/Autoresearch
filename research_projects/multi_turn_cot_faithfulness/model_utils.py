"""
model_utils.py - Model loading and inference utilities.
Shared helpers for cross-model faithfulness experiments.
"""

import os
import re
import torch
from typing import Optional, List, Dict, Tuple

DEFAULT_GEN_CONFIG = {
    "max_new_tokens": 1500,
    "do_sample": False,
    "temperature": 1.0,
    "repetition_penalty": 1.1,
}

SUPPORTED_MODELS = {
    "r1_7b":        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1_14b":       "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "r1_llama_8b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwen3_8b":     "Qwen/Qwen3-8B",
    "qwen3_14b":    "Qwen/Qwen3-14B",
    "qwen3_32b":    "Qwen/Qwen3-32B",
    "qwq_32b":      "Qwen/QwQ-32B",
}


def get_quantization_config(bits: int = 8):
    from transformers import BitsAndBytesConfig
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model_and_tokenizer(model_key: str, bits: int = 8, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = SUPPORTED_MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_quantization_config(bits),
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_numeric_answer(text: str) -> Optional[str]:
    patterns = [
        r"####\s*([-\d,\.]+)",
        r"answer is\s*([-\d,\.]+)",
        r"=\s*([-\d,\.]+)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def build_chat_messages(history: List[Dict], system_prompt: str = "") -> List[Dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    return msgs


def get_model_memory_gb(model) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total / (1024 ** 3), 2)


# EXTEND_MODEL_UTILS
