#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# AI News Pipeline — Progress Monitor
# ============================================================================

show_help() {
    cat << 'HELPEOF'
AI News Pipeline — Progress Monitor

  Connects to the remote VPS over SSH and queries the SQLite database
  and log file to provide a status snapshot of the current (or most
  recent) pipeline run.

  Displays:
    - Pipeline runs and their status (pending / running / completed /
      failed) with current stage, date, and error messages.
    - Article counts per run.
    - Theme status counts per run (active, dropped, etc.).
    - Deliverable counts per theme.
    - Daily brief word counts.
    - The last 12 non-debug log entries from the pipeline log file.

  Prerequisites:
    - A local virtual environment with paramiko installed.
    - SSH access to the target VPS (password authentication used).

  Usage:
    export VPS_HOST=1.2.3.4 VPS_USER=myself VPS_SSH_PASSWORD=...
    ./deploy/monitor.sh

  Required environment variables:
    VPS_HOST           Hostname or IP address of the target VPS.
    VPS_USER           SSH username for authentication.
    VPS_SSH_PASSWORD   Password for SSH and sudo on the VPS.

  Arguments:  None.
  Exit codes: 0 on success; non-zero on any failure (set -e).
HELPEOF
    exit 0
}

for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        show_help
    fi
done

# Required environment variables:
#   VPS_HOST          — hostname or IP of the target VPS
#   VPS_USER          — SSH username
#   VPS_SSH_PASSWORD  — SSH/sudo password
# ============================================================================

: "${VPS_HOST:?VPS_HOST must be set}"
: "${VPS_USER:?VPS_USER must be set}"
: "${VPS_SSH_PASSWORD:?VPS_SSH_PASSWORD must be set}"

export VPS_HOST VPS_USER VPS_SSH_PASSWORD

VENV_PYTHON="/home/daniel/projects/ai_news_scraper/venv/bin/python3"

"${VENV_PYTHON}" << 'PYEOF'
import json, os, paramiko

HOST = os.environ["VPS_HOST"]
USER = os.environ["VPS_USER"]
PASS = os.environ["VPS_SSH_PASSWORD"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=HOST, username=USER, password=PASS, timeout=10, port=22)

# DB state
cmd = """python3 << 'INNER'
import sqlite3
c = sqlite3.connect('/opt/ai-news-pipeline/pipeline.db')
c.row_factory = sqlite3.Row

print("=== PIPELINE RUNS ===")
rows = list(c.execute('SELECT * FROM pipeline_runs ORDER BY id'))
if not rows:
    print("  (no runs yet)")
for r in rows:
    d = dict(r)
    err = d.get('error_message','') or ''
    if err:
        d['error_message'] = err[:150] + '...' if len(err) > 150 else err
    print(f"  Run {d['id']}: {d['status']} | stage={d.get('current_stage','?')} | date={d['run_date']}")
    if d.get('error_message'):
        print(f"    Error: {d['error_message']}")

print()
print("=== ARTICLES PER RUN ===")
for r in c.execute('SELECT pipeline_run_id, count(*) as cnt FROM articles GROUP BY pipeline_run_id'):
    print(f"  Run {r['pipeline_run_id']}: {r['cnt']} articles")

print()
print("=== THEMES ===")
for r in c.execute('SELECT pipeline_run_id, status, count(*) as cnt FROM themes GROUP BY pipeline_run_id, status ORDER BY pipeline_run_id, status'):
    print(f"  Run {r['pipeline_run_id']} | {r['status']}: {r['cnt']} themes")

print()
print("=== DELIVERABLES PER THEME ===")
for r in c.execute('SELECT theme_id, count(*) as cnt FROM deliverables GROUP BY theme_id ORDER BY theme_id'):
    print(f"  Theme {r['theme_id']}: {r['cnt']} deliverables")

print()
print("=== BRIEFS ===")
for r in c.execute('SELECT pipeline_run_id, word_count FROM daily_briefs ORDER BY pipeline_run_id'):
    print(f"  Run {r['pipeline_run_id']}: {r['word_count']} words")

c.close()
INNER"""
stdin, stdout, stderr = client.exec_command(cmd)
print(stdout.read().decode())

# Recent log entries (non-debug, src only)
cmd2 = """python3 -c '
import json
with open("/opt/ai-news-pipeline/logs/pipeline.log") as f:
    lines = f.readlines()
count = 0
for line in reversed(lines):
    try:
        d = json.loads(line)
        if d.get("level") != "DEBUG" and d.get("source","").startswith("src."):
            ts = d.get("timestamp","").split("T")[1][:12]
            print(f"{ts} {d.get(\"message\",\"\")}")
            count += 1
            if count >= 12:
                break
    except:
        pass
' 2>/dev/null || echo "(no log file yet)" """
stdin2, stdout2, stderr2 = client.exec_command(cmd2)
out = stdout2.read().decode().strip()
if out:
    print()
    print("=== RECENT LOG (src messages) ===")
    print(out)

client.close()
PYEOF
