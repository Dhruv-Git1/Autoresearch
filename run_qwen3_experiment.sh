#!/bin/bash
# Run anchoring experiment with Qwen3-14B as the assistant under test.
# Qwen3-14B is cached at ~/.cache/huggingface/hub — set HF_HOME accordingly.
# Run from: ~/multi_turn_cot/lost_in_conversation/
set -e
cd ~/multi_turn_cot/lost_in_conversation

echo "=== Qwen3-14B anchoring experiment ==="
echo "GPU state before launch:"
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader

LOAD_IN_8BIT=1 \
HF_HOME=/root/.cache/huggingface \
R1_MAX_NEW_TOKENS=1500 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 phase2_batch_runner.py \
  --assistant_model qwen3-14b \
  --n_samples 20 \
  --max_shards 20 \
  --max_turns 30 \
  --faith_tokens 128 \
  --seed 44444 \
  --out_dir ../multi_turn_cot_faithfulness/results/qwen3_14b_s1

echo "=== Qwen3-14B experiment complete ==="
