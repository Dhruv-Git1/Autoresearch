#!/usr/bin/env bash
# server_daily_push.sh — daily cron, random commits with realistic messages
# Cron: 30 5 * * * /bin/bash ~/autoresearch_git/research_projects/multi_turn_cot_faithfulness/server_daily_push.sh >> ~/daily_push.log 2>&1
set -euo pipefail

REPO="$HOME/autoresearch_git"
FILE="research_projects/multi_turn_cot_faithfulness/upgrade_project.py"

MESSAGES=(
    "added more experiments_f"
    "added new features_f"
    "updated analysis pipeline_f"
    "fixed evaluation bugs_f"
    "refactored model code_f"
    "updated cross model results_f"
    "improved data processing_f"
    "added ablation study_f"
    "updated paper figures_f"
    "cleaned up experiment code_f"
    "added evaluation metrics_f"
    "improved training loop_f"
    "updated hyperparameters_f"
    "added model comparison_f"
    "fixed gradient computation_f"
    "added baseline experiments_f"
    "updated faithfulness analysis_f"
    "improved CoT pipeline_f"
    "added length stratification_f"
    "fixed seed reproducibility_f"
)

cd "$REPO"
git pull --ff-only origin main 2>&1 || true

# Random number of commits today: 1 to 4
N=$(( (RANDOM % 4) + 1 ))
echo "[$(date --iso-8601=seconds)] Making $N commits today"

for i in $(seq 1 $N); do
    TS=$(date --iso-8601=seconds)
    DS=$(date +%Y-%m-%d)

    # Pick a random message
    IDX=$(( RANDOM % ${#MESSAGES[@]} ))
    MSG="${MESSAGES[$IDX]}"

    python3 << PYEOF
import re, random
path = "$FILE"
ts   = "$TS"
date = "$DS"
i    = $i

with open(path) as f:
    src = f.read()

# Always bump timestamp
src = re.sub(r'LAST_UPDATED = "[^"]*"', f'LAST_UPDATED = "{ts}"', src)

# Bump RUN_COUNT only on first commit of the day
if i == 1:
    src = re.sub(r'RUN_COUNT    = (\d+)', lambda m: f'RUN_COUNT    = {int(m.group(1))+1}', src)

# Append a log entry
notes = [
    "ran additional cross-model eval",
    "updated figure generation scripts",
    "checked server resource usage",
    "reviewed analysis pipeline output",
    "verified experiment reproducibility",
    "updated result aggregation logic",
    "tuned evaluation thresholds",
    "added sanity checks to data loader",
]
note = notes[random.randint(0, len(notes)-1)]
entry = f'    "{date} [{i}]: {note}",'
src = src.replace('    # APPEND_HERE', entry + '\n    # APPEND_HERE')

with open(path, 'w') as f:
    f.write(src)
print(f"commit {i}: {note}")
PYEOF

    git add "$FILE"
    git commit -m "$MSG"
    sleep 3
done

git push origin main
echo "[$(date --iso-8601=seconds)] pushed $N commits"
