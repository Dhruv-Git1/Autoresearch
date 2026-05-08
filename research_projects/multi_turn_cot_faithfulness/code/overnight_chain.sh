#!/bin/bash
# Overnight experiment chain — runs phases 3 and 4 after current phase 2 finishes,
# with bistability analysis between each. All output goes to overnight_chain.log.

BASE=/home/vasudev_majhi_2021/multi_turn_cot
LIC=$BASE/lost_in_conversation
PROJ=$BASE/multi_turn_cot_faithfulness
RESULTS=$PROJ/results
CODE=$PROJ/code

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $RESULTS/overnight_chain.log; }

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

mkdir -p $RESULTS
log "=== OVERNIGHT CHAIN STARTED ==="
log "Waiting for Phase 2 run (PID 1017880) to finish..."

while kill -0 1017880 2>/dev/null; do
    FREE=$(gpu_free_mb)
    log "Still running... GPU free: ${FREE} MB"
    sleep 120
done
log "Phase 2 finished. GPU free: $(gpu_free_mb) MB"

# -----------------------------------------------------------------------
# BISTABILITY: Phase 1 + Phase 2
# -----------------------------------------------------------------------
log "--- Bistability analysis: Phase1 + Phase2 ---"
mkdir -p $RESULTS/bistability_p1p2
python3 $CODE/phase2_bistability_analysis.py \
    --faith_paths \
        $BASE/results/day2/faithfulness.jsonl \
        $BASE/results/day2_sample965/faithfulness.jsonl \
        $RESULTS/phase2/faithfulness.jsonl \
    --trace_dirs \
        $BASE/results/day1 \
        $RESULTS/phase2 \
    --out_dir $RESULTS/bistability_p1p2 \
    >> $RESULTS/overnight_chain.log 2>&1
log "Bistability P1+P2 done."

# -----------------------------------------------------------------------
# PHASE 3: 20 samples, max_shards=6, fresh seed
# Excludes Phase 1 + Phase 2 samples to maximize unique conversations.
# -----------------------------------------------------------------------
FREE=$(gpu_free_mb)
log "--- Phase 3 start. GPU free: ${FREE} MB ---"

if [ "${FREE:-0}" -lt 12000 ]; then
    log "GPU too full (${FREE} MB < 12000 MB). Waiting up to 20 min for other users to free memory..."
    for i in $(seq 1 10); do
        sleep 120
        FREE=$(gpu_free_mb)
        log "  Still waiting... GPU free: ${FREE} MB"
        [ "${FREE:-0}" -ge 12000 ] && break
    done
fi

FREE=$(gpu_free_mb)
if [ "${FREE:-0}" -ge 12000 ]; then
    log "Starting Phase 3 (20 samples, max_shards=6, seed=12345)..."
    mkdir -p $RESULTS/phase3
    cd $LIC
    LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
        python3 phase2_batch_runner.py \
            --n_samples 20 --task math --seed 12345 --max_shards 6 \
            --faith_tokens 512 \
            --exclude_dirs $RESULTS/phase2 \
            --out_dir $RESULTS/phase3 \
        > $RESULTS/phase3/run.log 2>&1
    EXIT3=$?
    if [ $EXIT3 -eq 0 ]; then
        log "Phase 3 complete."
    else
        log "Phase 3 exited with code $EXIT3. Check $RESULTS/phase3/run.log"
    fi
else
    log "GPU still too full after wait. Skipping Phase 3."
fi

# -----------------------------------------------------------------------
# BISTABILITY: Phase 1 + Phase 2 + Phase 3
# -----------------------------------------------------------------------
log "--- Bistability analysis: Phase1 + Phase2 + Phase3 ---"
mkdir -p $RESULTS/bistability_p1p2p3
python3 $CODE/phase2_bistability_analysis.py \
    --faith_paths \
        $BASE/results/day2/faithfulness.jsonl \
        $BASE/results/day2_sample965/faithfulness.jsonl \
        $RESULTS/phase2/faithfulness.jsonl \
        $RESULTS/phase3/faithfulness.jsonl \
    --trace_dirs \
        $BASE/results/day1 \
        $RESULTS/phase2 \
        $RESULTS/phase3 \
    --out_dir $RESULTS/bistability_p1p2p3 \
    >> $RESULTS/overnight_chain.log 2>&1
log "Bistability P1+P2+P3 done."

# -----------------------------------------------------------------------
# PHASE 4: 15 samples, max_shards=10 — captures longer conversations
# where bistability dynamics have more room to emerge.
# -----------------------------------------------------------------------
FREE=$(gpu_free_mb)
log "--- Phase 4 start. GPU free: ${FREE} MB ---"

if [ "${FREE:-0}" -lt 12000 ]; then
    log "Waiting for GPU..."
    for i in $(seq 1 10); do
        sleep 120
        FREE=$(gpu_free_mb)
        log "  GPU free: ${FREE} MB"
        [ "${FREE:-0}" -ge 12000 ] && break
    done
fi

FREE=$(gpu_free_mb)
if [ "${FREE:-0}" -ge 12000 ]; then
    log "Starting Phase 4 (15 samples, max_shards=10, seed=99999)..."
    mkdir -p $RESULTS/phase4
    cd $LIC
    LOAD_IN_8BIT=1 HF_HOME=/dev/shm/vasudev_hf_cache R1_MAX_NEW_TOKENS=1500 \
        python3 phase2_batch_runner.py \
            --n_samples 15 --task math --seed 99999 --max_shards 10 \
            --faith_tokens 512 \
            --exclude_dirs $RESULTS/phase2 $RESULTS/phase3 \
            --out_dir $RESULTS/phase4 \
        > $RESULTS/phase4/run.log 2>&1
    EXIT4=$?
    if [ $EXIT4 -eq 0 ]; then
        log "Phase 4 complete."
    else
        log "Phase 4 exited with code $EXIT4. Check $RESULTS/phase4/run.log"
    fi
else
    log "GPU still too full. Skipping Phase 4."
fi

# -----------------------------------------------------------------------
# FINAL BISTABILITY: all phases
# -----------------------------------------------------------------------
log "--- Final bistability analysis (all phases) ---"
mkdir -p $RESULTS/bistability_final
python3 $CODE/phase2_bistability_analysis.py \
    --faith_paths \
        $BASE/results/day2/faithfulness.jsonl \
        $BASE/results/day2_sample965/faithfulness.jsonl \
        $RESULTS/phase2/faithfulness.jsonl \
        $RESULTS/phase3/faithfulness.jsonl \
        $RESULTS/phase4/faithfulness.jsonl \
    --trace_dirs \
        $BASE/results/day1 \
        $RESULTS/phase2 \
        $RESULTS/phase3 \
        $RESULTS/phase4 \
    --out_dir $RESULTS/bistability_final \
    >> $RESULTS/overnight_chain.log 2>&1

log "=== ALL DONE. Final results in $RESULTS/bistability_final/ ==="
log "GPU status: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)"
