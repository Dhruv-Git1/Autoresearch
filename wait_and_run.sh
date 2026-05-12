#!/bin/bash
# Waits until GPU has enough free memory, then runs both experiments sequentially.
# Launch with: nohup bash ~/wait_and_run.sh > ~/wait_and_run.log 2>&1 &

REQUIRED_MB=30000   # need ~30GB free (Qwen3-14B@8bit ~14GB + Qwen2.5@8bit ~7GB + headroom)
POLL_SECS=60
LOG_DIR="/home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results"

echo "[$(date)] wait_and_run.sh started. Waiting for ${REQUIRED_MB}MB free GPU memory..."

while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    echo "[$(date)] GPU free: ${FREE_MB}MB"

    if [ "$FREE_MB" -ge "$REQUIRED_MB" ]; then
        echo "[$(date)] GPU has ${FREE_MB}MB free — starting experiments."
        break
    fi

    sleep $POLL_SECS
done

# ── Experiment 1: Qwen3-14B ──────────────────────────────────────────────────
echo "[$(date)] === Starting Qwen3-14B experiment ==="
bash ~/run_qwen3_experiment.sh > "$LOG_DIR/qwen3_14b_s1/run.log" 2>&1
echo "[$(date)] === Qwen3-14B experiment finished ==="

# ── Experiment 2: R1-Distill-Llama-8B ───────────────────────────────────────
echo "[$(date)] === Starting R1-Distill-Llama-8B experiment ==="
bash ~/run_llama_r1_experiment.sh > "$LOG_DIR/r1_llama_s1/run.log" 2>&1
echo "[$(date)] === R1-Distill-Llama-8B experiment finished ==="

echo "[$(date)] All experiments complete."
