#!/bin/bash
BASE=/home/vasudev_majhi_2021/multi_turn_cot
LIC=$BASE/lost_in_conversation
RES=$BASE/multi_turn_cot_faithfulness/results
CODE=$BASE/multi_turn_cot_faithfulness/code

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Waiting for Phase 3 (seed 12345) to finish..."
while pgrep -f "seed 12345" > /dev/null; do
    sleep 60
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    log "Phase 3 still running... GPU free: ${FREE}MB"
done
log "Phase 3 done."

# Bistability: P1 + P2 + P3
log "Bistability P1+P2+P3..."
python3 $CODE/phase2_bistability_analysis.py \
    --faith_paths \
        $BASE/results/day2/faithfulness.jsonl \
        $BASE/results/day2_sample965/faithfulness.jsonl \
        $RES/phase2/faithfulness.jsonl \
        $RES/phase3/faithfulness.jsonl \
    --trace_dirs $BASE/results/day1 $RES/phase2 $RES/phase3 \
    --out_dir $RES/bistability_p1p2p3
log "Bistability P1+P2+P3 done."

# Phase 4
log "Starting Phase 4 (15 samples, max_shards=10, seed=99999, faith_tokens=128)..."
mkdir -p $RES/phase4
cd $LIC
LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
    python3 phase2_batch_runner.py \
        --n_samples 15 --task math --seed 99999 --max_shards 10 \
        --faith_tokens 128 \
        --exclude_dirs $RES/phase2 $RES/phase3 \
        --out_dir $RES/phase4 \
    2>&1 | tee $RES/phase4/run.log
log "Phase 4 done."

# Final bistability
log "Final bistability analysis..."
python3 $CODE/phase2_bistability_analysis.py \
    --faith_paths \
        $BASE/results/day2/faithfulness.jsonl \
        $BASE/results/day2_sample965/faithfulness.jsonl \
        $RES/phase2/faithfulness.jsonl \
        $RES/phase3/faithfulness.jsonl \
        $RES/phase4/faithfulness.jsonl \
    --trace_dirs $BASE/results/day1 $RES/phase2 $RES/phase3 $RES/phase4 \
    --out_dir $RES/bistability_final
log "=== ALL DONE. Check $RES/bistability_final/ ==="
