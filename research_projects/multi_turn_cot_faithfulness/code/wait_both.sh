#!/bin/bash
BASE=/home/vasudev_majhi_2021/multi_turn_cot
RES=$BASE/multi_turn_cot_faithfulness/results
CODE=$BASE/multi_turn_cot_faithfulness/code

log() { echo "[$(date '+%H:%M:%S')] $*"; }

p3_running() { ps aux | grep -v grep | grep -q "seed 12345"; }
p4_running() { ps aux | grep -v grep | grep -q "seed 99999"; }
gpu_free()   { /usr/bin/nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

log "Waiting for Phase 3 (seed 12345) and Phase 4 (seed 99999) to both finish..."
while p3_running || p4_running; do
    P3=$(p3_running && echo 1 || echo 0)
    P4=$(p4_running && echo 1 || echo 0)
    log "Phase3 running=$P3  Phase4 running=$P4  GPU free: $(gpu_free)MB"
    sleep 120
done
log "Both finished."

log "Running final bistability analysis (all phases)..."
mkdir -p $RES/bistability_final
python3 $CODE/phase2_bistability_analysis.py \
    --faith_paths \
        $BASE/results/day2/faithfulness.jsonl \
        $BASE/results/day2_sample965/faithfulness.jsonl \
        $RES/phase2/faithfulness.jsonl \
        $RES/phase3/faithfulness.jsonl \
        $RES/phase4/faithfulness.jsonl \
    --trace_dirs \
        $BASE/results/day1 \
        $RES/phase2 \
        $RES/phase3 \
        $RES/phase4 \
    --out_dir $RES/bistability_final
log "=== ALL DONE. Check $RES/bistability_final/ ==="
