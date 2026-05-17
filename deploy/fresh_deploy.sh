#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# AI News Pipeline — Fresh Deploy & Run Script
# ============================================================================
# This script assumes:
#   1. Local repo at /home/daniel/projects/ai_news_scraper is the source of truth
#   2. VPS credentials provided via environment variables
#   3. Deployment target is /opt/ai-news-pipeline/
#   4. All source changes (parallel generator, orphan recovery, 
#      configurable max_themes, increased timeout) are committed locally
# ============================================================================
#
# Required environment variables:
#   VPS_HOST          — hostname or IP of the target VPS
#   VPS_USER          — SSH username
#   VPS_SSH_PASSWORD  — SSH/sudo password
#
# Example:
#   export VPS_HOST=1.2.3.4 VPS_USER=myself VPS_SSH_PASSWORD=...
#   ./deploy/fresh_deploy.sh
# ============================================================================

: "${VPS_HOST:?VPS_HOST must be set}"
: "${VPS_USER:?VPS_USER must be set}"
: "${VPS_SSH_PASSWORD:?VPS_SSH_PASSWORD must be set}"

export VPS_HOST VPS_USER VPS_SSH_PASSWORD LOCAL_PROJECT

LOCAL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_TARGET="/opt/ai-news-pipeline"
VENV_PYTHON="${LOCAL_PROJECT}/venv/bin/python3"

echo "========================================="
echo "AI News Pipeline — Fresh Deploy & Run"
echo "========================================="

# Step 1: Rsync all source/config/prompt files to VPS
echo ""
echo "[1/4] Syncing project files to VPS..."
"${VENV_PYTHON}" << 'PYEOF'
import os, hashlib, paramiko

HOST = os.environ["VPS_HOST"]
USER = os.environ["VPS_USER"]
PASS = os.environ["VPS_SSH_PASSWORD"]
LOCAL = os.environ["LOCAL_PROJECT"]

FILES_TO_SYNC = [
    "src/__init__.py",
    "src/main.py",
    "src/config.py",
    "src/models.py",
    "src/db.py",
    "src/llm.py",
    "src/scraper.py",
    "src/analyzer.py",
    "src/generator.py",
    "src/evaluator.py",
    "src/brief.py",
    "src/emailer.py",
    "prompts/analyze.txt",
    "prompts/brief.txt",
    "prompts/evaluate_adversarial.txt",
    "prompts/evaluate_quality.txt",
    "prompts/refine.txt",
    "prompts/script_de.txt",
    "prompts/script_en.txt",
    "prompts/summary_en.txt",
    "config/database.yaml",
    "config/email.yaml",
    "config/feeds.yaml",
    "config/models.yaml",
    "config/openrouter.yaml",
    "config/pipeline.yaml",
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=HOST, username=USER, password=PASS, timeout=10, port=22)
sftp = client.open_sftp()

for local_rel in FILES_TO_SYNC:
    local_path = os.path.join(LOCAL, local_rel)
    remote_path = os.path.join("/opt/ai-news-pipeline", local_rel)
    
    with open(local_path, "rb") as f:
        local_hash = hashlib.md5(f.read()).hexdigest()
    
    sftp.put(local_path, f"/tmp/sync_{os.path.basename(local_rel)}")
    
    # Move into place with correct ownership
    cmd = f"sudo -S mv /tmp/sync_{os.path.basename(local_rel)} {remote_path} && sudo -S chown ai-news-pipeline:ai-news-pipeline {remote_path} && echo OK"
    stdin, stdout, stderr = client.exec_command(cmd)
    stdin.write(PASS + "\n")
    stdin.flush()
    out = stdout.read().decode().strip()
    err = stderr.read().decode()
    if "OK" in out:
        print(f"  SYNCED  {local_rel}")
    else:
        print(f"  FAILED  {local_rel}: {out} {err}")

sftp.close()
client.close()
PYEOF

# Step 2: Deploy systemd service file
echo ""
echo "[2/4] Deploying systemd service file..."
"${VENV_PYTHON}" << 'PYEOF'
import os, paramiko

HOST = os.environ["VPS_HOST"]
USER = os.environ["VPS_USER"]
PASS = os.environ["VPS_SSH_PASSWORD"]
LOCAL = os.environ["LOCAL_PROJECT"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=HOST, username=USER, password=PASS, timeout=10, port=22)
sftp = client.open_sftp()
sftp.put(os.path.join(LOCAL, "deploy/ai-news-pipeline.service"), "/tmp/ai-news-pipeline.service")
sftp.close()

cmd = "sudo -S cp /tmp/ai-news-pipeline.service /etc/systemd/system/ai-news-pipeline.service && sudo -S systemctl daemon-reload && echo OK"
stdin, stdout, stderr = client.exec_command(cmd)
stdin.write(PASS + "\n")
stdin.flush()
print(stdout.read().decode().strip())

# Verify TimeoutStartSec
stdin2, stdout2, stderr2 = client.exec_command("grep TimeoutStartSec /etc/systemd/system/ai-news-pipeline.service")
print("Timeout:", stdout2.read().decode().strip())

client.close()
PYEOF

# Step 3: Wipe database and logs (FRESH START)
echo ""
echo "[3/4] Wiping database and logs for clean slate..."
"${VENV_PYTHON}" << 'PYEOF'
import os, paramiko

HOST = os.environ["VPS_HOST"]
USER = os.environ["VPS_USER"]
PASS = os.environ["VPS_SSH_PASSWORD"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=HOST, username=USER, password=PASS, timeout=10, port=22)

cmd = "sudo -S rm -f /opt/ai-news-pipeline/pipeline.db /opt/ai-news-pipeline/pipeline.db-wal /opt/ai-news-pipeline/pipeline.db-shm /opt/ai-news-pipeline/logs/pipeline.log && echo 'DB and logs wiped'"
stdin, stdout, stderr = client.exec_command(cmd)
stdin.write(PASS + "\n")
stdin.flush()
print(stdout.read().decode().strip())

# Verify
stdin2, stdout2, stderr2 = client.exec_command("ls /opt/ai-news-pipeline/pipeline.db /opt/ai-news-pipeline/logs/pipeline.log 2>&1; echo 'EXIT:'$?")
out = stdout2.read().decode()
if "cannot access" in out:
    print("VERIFIED: DB and logs do not exist — clean slate confirmed")
else:
    print("WARNING: something still exists!", out)

client.close()
PYEOF

# Step 4: Start the pipeline
echo ""
echo "[4/4] Starting pipeline..."
"${VENV_PYTHON}" << 'PYEOF'
import os, paramiko, time

HOST = os.environ["VPS_HOST"]
USER = os.environ["VPS_USER"]
PASS = os.environ["VPS_SSH_PASSWORD"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=HOST, username=USER, password=PASS, timeout=10, port=22)

cmd = "sudo -S systemctl start --no-block ai-news-pipeline.service"
stdin, stdout, stderr = client.exec_command(cmd)
stdin.write(PASS + "\n")
stdin.flush()
out = stdout.read().decode()
err = stderr.read().decode()
print("Started:", out, err)

# Quick status
time.sleep(2)
stdin2, stdout2, stderr2 = client.exec_command("systemctl status ai-news-pipeline.service --no-pager 2>&1")
print(stdout2.read().decode()[:500])

client.close()
PYEOF

echo ""
echo "========================================="
echo "Pipeline started. Monitor with:"
echo "  ./monitor.sh"
echo "========================================="
