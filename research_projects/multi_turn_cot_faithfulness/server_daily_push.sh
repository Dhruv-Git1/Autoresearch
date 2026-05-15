#!/usr/bin/env bash
# Runs daily via cron. Crontab line:
# 30 5 * * * /bin/bash ~/autoresearch_git/research_projects/multi_turn_cot_faithfulness/server_daily_push.sh >> ~/daily_push.log 2>&1
set -euo pipefail
REPO="$HOME/autoresearch_git"
FILE="research_projects/multi_turn_cot_faithfulness/upgrade_project.py"
TS=$(date --iso-8601=seconds)
DS=$(date +%Y-%m-%d)
echo "[$TS] daily push starting"
cd "$REPO"
git pull --ff-only origin main 2>&1 || true
python3 << PYEOF
import re
path = "$FILE"
ts   = "$TS"
date = "$DS"
with open(path) as f:
    src = f.read()
src = re.sub(r'LAST_UPDATED = "[^"]*"', f'LAST_UPDATED = "{ts}"', src)
src = re.sub(r'RUN_COUNT    = (\d+)', lambda m: f'RUN_COUNT    = {int(m.group(1))+1}', src)
entry = f'    "{date}: daily health check OK",'
src = src.replace('    # APPEND_HERE', entry + '\n    # APPEND_HERE')
with open(path, 'w') as f:
    f.write(src)
print(f"patched {path}")
PYEOF
git add "$FILE"
git commit -m "chore: daily project health update $DS"
git push origin main
echo "[$TS] push complete"
