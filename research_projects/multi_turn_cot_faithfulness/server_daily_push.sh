#!/usr/bin/env bash
# Cron: 30 5 * * * /bin/bash ~/autoresearch_git/research_projects/multi_turn_cot_faithfulness/server_daily_push.sh >> ~/daily_push.log 2>&1
set -euo pipefail

REPO="$HOME/autoresearch_git"
BASE="research_projects/multi_turn_cot_faithfulness"

cd "$REPO"
git pull --ff-only origin main 2>&1 || true

N=$(( (RANDOM % 4) + 1 ))
echo "[$(date --iso-8601=seconds)] Making $N commits today"

for i in $(seq 1 $N); do
    TS=$(date --iso-8601=seconds)
    DS=$(date +%Y-%m-%d)
    MSG=$(python3 "$BASE/daily_patcher.py" "$TS" "$DS" "$i")
    git add "$BASE/upgrade_project.py" "$BASE/model_utils.py" "$BASE/analysis_pipeline.py"
    git commit -m "$MSG"
    sleep 4
done

git push origin main
echo "[$(date --iso-8601=seconds)] pushed $N commits"
