#!/usr/bin/env bash
# Waits for hotpotqa_s2 (short) and hotpotqa_s3 (medium) runs to finish,
# downloads results, then runs tiered analysis alongside the existing s1 (long) data.
# Run from repo root: bash research_projects/multi_turn_cot_faithfulness/code/download_and_analyze_hotpotqa.sh

set -e
SERVER=gpu-server
REMOTE_BASE="/home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results"
LOCAL_BASE="research_projects/multi_turn_cot_faithfulness/results"
ANALYSIS_OUT="${LOCAL_BASE}/bistability_hotpotqa"
GSM8K_STATS="${LOCAL_BASE}/bistability_v3_combined/bistability_stats.json"

echo "[local] Polling server for s2 (short) and s3 (medium) completion..."
until ssh "$SERVER" "grep -q 'Phase 2 complete' ~/hotpotqa_s2.log 2>/dev/null && grep -q 'Phase 2 complete' ~/hotpotqa_s3.log 2>/dev/null"; do
    S2_DONE=$(ssh "$SERVER" "ls ${REMOTE_BASE}/hotpotqa_s2/trace_sharded_*.json 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    S3_DONE=$(ssh "$SERVER" "ls ${REMOTE_BASE}/hotpotqa_s3/trace_sharded_*.json 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    echo "  ...$(date '+%H:%M') — s2: ${S2_DONE}/20 convs done | s3: ${S3_DONE}/20 convs done"
    sleep 300
done
echo "[local] Both runs complete. Downloading s2 and s3 results..."

mkdir -p "${LOCAL_BASE}/hotpotqa_s2"
mkdir -p "${LOCAL_BASE}/hotpotqa_s3"

scp "${SERVER}:${REMOTE_BASE}/hotpotqa_s2/faithfulness.jsonl" "${LOCAL_BASE}/hotpotqa_s2/"
scp "${SERVER}:${REMOTE_BASE}/hotpotqa_s2/trace_sharded_*.json" "${LOCAL_BASE}/hotpotqa_s2/" 2>/dev/null || true
scp "${SERVER}:${REMOTE_BASE}/hotpotqa_s2/summary.json" "${LOCAL_BASE}/hotpotqa_s2/" 2>/dev/null || true

scp "${SERVER}:${REMOTE_BASE}/hotpotqa_s3/faithfulness.jsonl" "${LOCAL_BASE}/hotpotqa_s3/"
scp "${SERVER}:${REMOTE_BASE}/hotpotqa_s3/trace_sharded_*.json" "${LOCAL_BASE}/hotpotqa_s3/" 2>/dev/null || true
scp "${SERVER}:${REMOTE_BASE}/hotpotqa_s3/summary.json" "${LOCAL_BASE}/hotpotqa_s3/" 2>/dev/null || true

echo "[local] Downloaded. Running tiered analysis..."

PYTHONIOENCODING=utf-8 python research_projects/multi_turn_cot_faithfulness/code/analyze_hotpotqa.py \
  --s1_faith  "${LOCAL_BASE}/hotpotqa_s1/faithfulness.jsonl" \
  --s2_faith  "${LOCAL_BASE}/hotpotqa_s2/faithfulness.jsonl" \
  --s3_faith  "${LOCAL_BASE}/hotpotqa_s3/faithfulness.jsonl" \
  --gsm8k_stats "$GSM8K_STATS" \
  --out_dir   "$ANALYSIS_OUT"

echo ""
echo "[local] Done. Key outputs:"
echo "  Figure : research_projects/multi_turn_cot_faithfulness/paper/figures/gsm8k_hotpotqa_gradient.png"
echo "  Stats  : ${ANALYSIS_OUT}/hotpotqa_gradient_stats.json"
