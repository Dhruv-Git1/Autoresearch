#!/bin/bash
# Run anchoring experiment with DeepSeek-R1-Distill-Llama-8B as the assistant.
# Model is downloaded to /dev/shm/vasudev_hf_cache by download_llama_r1.py.
# Run from: ~/multi_turn_cot/lost_in_conversation/
set -e
cd ~/multi_turn_cot/lost_in_conversation

echo "=== R1-Distill-Llama-8B anchoring experiment ==="
echo "GPU state before launch:"
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader

# Check model is downloaded
DL_PATH="/dev/shm/vasudev_hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B"
if [ ! -d "$DL_PATH" ]; then
  echo "ERROR: R1-Distill-Llama-8B not yet downloaded to $DL_PATH"
  echo "Run: python3 ~/download_llama_r1.py"
  exit 1
fi
echo "Model found at $DL_PATH"

LOAD_IN_8BIT=1 \
HF_HOME=/dev/shm/vasudev_hf_cache \
R1_MAX_NEW_TOKENS=1500 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 phase2_batch_runner.py \
  --assistant_model deepseek-r1-distill-llama-8b \
  --n_samples 20 \
  --max_shards 20 \
  --max_turns 30 \
  --faith_tokens 128 \
  --seed 55555 \
  --out_dir ../multi_turn_cot_faithfulness/results/r1_llama_s1

echo "=== R1-Distill-Llama-8B experiment complete ==="
