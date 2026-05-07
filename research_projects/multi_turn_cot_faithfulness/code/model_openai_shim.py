"""Shim that replaces lost_in_conversation/model_openai.py.

The simulator imports `from model_openai import generate, generate_json`.
We delegate to model_local (HuggingFace transformers) instead of OpenAI.
"""
from model_local import generate, generate_json, format_messages, split_thinking  # noqa: F401


class OpenAI_Model:
    """Thin compatibility shim. The original module instantiates this at import."""
    def generate(self, *args, **kwargs):
        return generate(*args, **kwargs)

    def generate_json(self, *args, **kwargs):
        return generate_json(*args, **kwargs)


model = OpenAI_Model()
