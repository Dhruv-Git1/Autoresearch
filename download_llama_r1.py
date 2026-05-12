import sys, os
sys.path = [p for p in sys.path if "/tmp" not in p]
os.environ["HF_HOME"] = "/dev/shm/vasudev_hf_cache"
from huggingface_hub import snapshot_download
sys.stderr.write("Starting download of DeepSeek-R1-Distill-Llama-8B to /dev/shm...\n")
sys.stderr.flush()
path = snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    cache_dir="/dev/shm/vasudev_hf_cache/hub",
    ignore_patterns=["*.pt"],
)
sys.stderr.write("DOWNLOAD COMPLETE: " + path + "\n")
sys.stderr.flush()
