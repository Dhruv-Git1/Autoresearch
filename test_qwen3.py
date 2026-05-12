import sys
sys.path = [p for p in sys.path if "/tmp" not in p]
from transformers import AutoTokenizer

snap = "/home/vasudev_majhi_2021/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18"
tok = AutoTokenizer.from_pretrained(snap)
msgs = [{"role": "user", "content": "What is 2+2?"}]
pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
think_open = "<think>"
pt_think = pt.rstrip() + think_open + "\n"
print("WITHOUT:", repr(pt[-60:]))
print("WITH:   ", repr(pt_think[-80:]))
print("TEST_OK")
