# AI News Pipeline

A persistent multi-interest news pipeline that scrapes ~50 AI-focused sites and blogs, finds the biggest themes of the day, and generates polished summaries, scripts, and a daily brief — all managed through a web UI. Runs unattended on a cheap Ubuntu VPS.

**Cool, but why?**

Because keeping up with AI news is impossible. Dozens of sites publish every day, and most of it is noise. This pipeline does the reading for you, picks out what actually matters, and hands you a finished summary — plus scripts you can literally read off a teleprompter if you run a YouTube channel. You can run multiple independent "interests" (e.g. AI news, climate tech, crypto) from the same instance, each with its own schedule, feeds, and output settings.

---

## How It Works (90 Seconds)

The pipeline runs as a **persistent background service** — a Flask web server with an in-process APScheduler. You configure everything through the web UI (or YAML files), and the scheduler kicks off pipeline runs per interest at their scheduled times.

```
Your VPS runs a persistent daemon (systemd Type=simple, Restart=always)
    │
    ├─ Flask HTTPS web server on port 8443 for admin (dashboard, CRUD interests, edit config)
    │
    ├─ APScheduler manages per-interest schedules
    │   (e.g. "AI News" at 04:00 daily, "Crypto" at 05:00 daily)
    │
    ├─ When a scheduled run triggers for an interest:
    │   ├─ Scrapes that interest's RSS feeds, pulls full article text
    │   ├─ Compares today's articles against yesterday's brief
    │   │   to find genuinely new themes (not "AI is changing everything" — again)
    │   ├─ For each theme found, generates up to three deliverables:
    │   │   • English summary
    │   │   • English YouTube script
    │   │   • Natively-written German YouTube script
    │   ├─ Runs each deliverable through quality checks and
    │   │   adversarial fact-checking, refining up to 3 times
    │   ├─ Writes a daily brief tying all themes together
    │   └─ Emails everything to you via Gmail
    │
    └─ If anything breaks, you get an alert email with exactly what failed and why.
```

At any time you can log into the web UI and click "Run Now" on any interest to trigger an immediate pipeline run.

---

## Prerequisites

You need a Linux server (or an old laptop running Linux) with:

| Thing | Minimum | Notes |
|-------|---------|-------|
| OS | Ubuntu 24.04 LTS | Other Debian-based distros should work, but untested |
| Python | 3.12 or newer | Comes with Ubuntu 24.04 by default |
| Disk | ~2 GB free | Database grows about 5 MB per day |
| Network | Outbound internet | To reach RSS feeds, OpenRouter API, and Gmail SMTP |
| Inbound ports | **8443** (HTTPS) | So you can reach the web admin UI from your browser |

You also need accounts on these services:

- **[OpenRouter](https://openrouter.ai/)** — provides API access to LLMs. You prepay credits (start with $5). This project uses DeepSeek models through OpenRouter because they're cheap (~$0.50–$1.00 per nightly run).
- **[Gmail](https://gmail.com)** — for sending the email digests. You'll generate an "App Password" (not your real password — more on that below).

---

## Installation

The whole thing installs with a single script. It takes about 2 minutes plus however long your VPS takes to `pip install`.

### Step 1: Clone the repo onto your server

```bash
git clone https://github.com/your-username/ai_news_scraper.git
cd ai_news_scraper
```

### Step 2: Set up your API keys

Before running the installer, create a `.env` file with your secrets. A template is included:

```bash
cp .env.example .env
```

Now edit `.env` and fill in real values:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

**How to get these:**

- **OpenRouter API key:** Sign up at [openrouter.ai](https://openrouter.ai/), go to Settings → Keys, create a key, and add credits.
- **Gmail App Password:** Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords). You must have 2-Step Verification enabled first. Google will give you a 16-character password — use that, not your normal Gmail password. This is way safer because the app password can only send email, not read your inbox or change settings.

### Step 3: Set an admin password for the web UI

Edit `config/server.yaml` and set a password for the web admin dashboard:

```yaml
admin_password: "your-secure-password-here"
```

### Step 4: Run the installer

```bash
sudo bash deploy/install.sh
```

The installer does everything: creates a locked-down system user, sets up a Python virtual environment, copies files into `/opt/ai-news-pipeline/`, and starts the persistent service. You'll see output like:

```
=== AI News Pipeline Installation ===
[1/8] Creating system user 'ai-news-pipeline'...
[2/8] Creating directory structure under /opt/ai-news-pipeline...
[3/8] Setting ownership to ai-news-pipeline...
[4/8] Copying project files...
[5/8] Creating Python virtual environment...
[6/8] Creating .env file from template...
[7/8] Installing systemd service and logrotate config...
[8/8] Enabling and starting the persistent service...

=== Installation complete ===
```

### Step 5: Verify it works

```bash
sudo systemctl status ai-news-pipeline.service
```

You should see `Active: active (running)`. The web UI is now available at `https://your-server-ip:8443` — your browser will warn about a self-signed certificate (that's expected, see "Self-signed certs" below).

Navigate there, log in with the admin password you set, and you'll see the dashboard.

---

## Understanding the Moving Parts

This section explains concepts the project depends on. If you already know these, skip ahead to Configuration.

### What's a virtual environment (venv) and why do we use one?

Python projects install packages (libraries) from the internet. If you install them globally, every Python project on your server shares the same versions. This breaks when Project A needs `httpx==0.27` but Project B needs `httpx==0.25`.

A **virtual environment** is a self-contained folder with its own copy of Python and its own set of installed packages. The project's venv lives at `/opt/ai-news-pipeline/venv/`. When the pipeline runs, it uses the Python inside this folder — not your system Python. This means:

- The pipeline's dependencies can't conflict with anything else on the server.
- You can delete the whole thing by removing one folder.
- You know exactly which versions are installed (pinned in `requirements.txt`).

The installer creates and populates the venv automatically. The key commands if you ever need to do it manually:

```bash
# Create a venv called "venv" in the current directory
python3 -m venv venv

# "Activate" it (makes your terminal use the venv's Python)
source venv/bin/activate

# Install packages listed in requirements.txt
pip install -r requirements.txt

# "Deactivate" when you're done
deactivate
```

The pipeline doesn't bother with `activate`/`deactivate` — it just calls the Python binary directly by its full path: `/opt/ai-news-pipeline/venv/bin/python`.

### What's systemd and why a persistent service instead of a timer?

The old version of this pipeline used a systemd timer (like cron) to run a oneshot job at 4 AM. The new version instead runs a **persistent service** that stays alive 24/7.

**systemd** is the thing that starts and stops programs on Linux. On Ubuntu, it runs as PID 1 (the first process, the one that launches everything else).

Why persistent instead of timer-driven?

- **Multi-interest scheduling**: APScheduler (an in-process Python scheduler) manages multiple interests with different schedules. You might have "AI News" at 4 AM and "Crypto" at 6 AM — APScheduler handles that natively.
- **Web UI**: A Flask server runs inside the same process, giving you a dashboard to add/remove interests, edit feeds, and trigger runs manually.
- **Self-healing**: With `Restart=always`, if the process crashes, systemd brings it back within 10 seconds. APScheduler re-evaluates missed runs on restart.

The service type is `Type=simple` (standard for daemons) with `Restart=always` and a 10-second restart delay. It's not `Type=oneshot` because it never exits intentionally.

### Self-signed certs and the browser warning

The web UI runs over HTTPS on port 8443. Since you probably don't have a real TLS certificate for your VPS, the pipeline auto-generates a **self-signed certificate** on first startup.

Your browser will show a "Your connection is not private" warning. This is expected and safe — you're connecting to your own server. Click "Advanced" → "Proceed to site" (Chrome) or "Accept the Risk and Continue" (Firefox).

The cert is stored at `/opt/ai-news-pipeline/cert.pem` and `/opt/ai-news-pipeline/key.pem`. It's valid for 10 years and auto-generated if missing.

### Terminal multiplexers: tmux and screen

If you're managing a server over SSH, your connection can drop. When it does, any program you were running in that terminal dies with it. A **terminal multiplexer** solves this by running a persistent session on the server that survives disconnects.

- **tmux** (recommended): Install with `sudo apt install tmux`. Start a session with `tmux new -s mysession`, detach with `Ctrl+B then D`, reattach with `tmux attach -t mysession`.
- **screen**: The older alternative. `screen -S mysession`, detach with `Ctrl+A then D`, reattach with `screen -r mysession`.

For this project specifically, you probably don't need a multiplexer day-to-day because the pipeline runs as a systemd service — not from your terminal. But tmux is useful when you're debugging:

```bash
# Quick tmux workflow for monitoring the pipeline
tmux new -s pipeline-watch
sudo journalctl -u ai-news-pipeline.service -f
# Ctrl+B, D to detach. Come back later:
tmux attach -t pipeline-watch
```

---

## Configuration: Every Setting Explained

All configuration lives in `/opt/ai-news-pipeline/config/` as separate, domain-specific YAML files. Each file controls one aspect of the pipeline:

| File | Purpose |
|------|---------|
| `feeds.yaml` | Default RSS/Atom feed sources (used as template for new interests) |
| `models.yaml` | LLM model assignments (strong vs weak) |
| `pipeline.yaml` | Runtime behavior (retries, timeouts, max themes) |
| `email.yaml` | SMTP settings for delivering the digest |
| `database.yaml` | SQLite database path |
| `openrouter.yaml` | OpenRouter API connection |
| `server.yaml` | Web server port, admin password, cert directory |

**Important:** In v2, feeds are managed **per-interest** through the web UI (stored in the database), not globally in `feeds.yaml`. The `feeds.yaml` file serves as a starting template for new interests you create.

### `feeds` — Default feed template

```yaml
feeds:
  news:                     # Traditional news outlets
    - name: "Ars Technica AI"
      url: "https://feeds.arstechnica.com/arstechnica/technology-lab"
    - name: "MIT Tech Review AI"
      url: "https://www.technologyreview.com/feed/"
  commentators:             # Individual bloggers/analysts
    - name: "Simon Willison"
      url: "https://simonwillison.net/atom/everything/"
    - name: "Jack Clark"
      url: "https://jack-clark.com/feed/"
```

| Field | Meaning |
|-------|---------|
| `name` | Human-readable label. Used in logs and emails. |
| `url` | Must be a valid RSS or Atom feed URL (starts with `http://` or `https://`). |

The pipeline treats `news` and `commentators` identically in scraping — the distinction is for the analyzer, which may weight commentary for "what are smart people talking about" vs. news for "what happened."

To add or remove feeds, use the web UI (navigate to an interest and edit its feeds). Changes take effect on the next scheduled run.

**Finding RSS feeds:** Many sites hide their feed links. Try appending `/feed`, `/rss`, or `/atom.xml` to the domain. You can also right-click → View Page Source and search for `application/rss+xml`. Browser extensions like "RSS Feed Reader" can auto-detect feeds on any page.

### `models` — Which LLMs to use

```yaml
models:
  strong:
    id: "deepseek/deepseek-v4-pro"    # For generation (writing)
    temperature: 0.7
  weak:
    id: "deepseek/deepseek-v4-flash"  # For evaluation (cheaper)
    temperature: 0.7
```

| Field | Meaning |
|-------|---------|
| `id` | The model identifier on OpenRouter. Format is `provider/model-name`. You can swap in any model OpenRouter supports (Claude, GPT-4o, Gemini, etc.) — just copy the ID from [openrouter.ai/models](https://openrouter.ai/models). |
| `temperature` | 0.0 = deterministic and boring, 1.0 = creative and sometimes unhinged. 0.7 is a good middle ground. |

**Why two models?** The "strong" model does the expensive work (writing summaries and scripts). The "weak" model does the cheaper work (evaluating quality, fact-checking). Strong costs more per token but writes better. Weak is fast and cheap, and "is this factual?" doesn't need a genius — just consistency checking. You can use the same model for both if you prefer.

**Cost estimate:** With DeepSeek v4 models, a full nightly run costs roughly $0.50–$1.00. With Claude 3.5 Sonnet or GPT-4o, expect $2–$5 per run. Forty runs a month (one per day) at $0.75/run = $30/month.

### `pipeline` — Runtime behavior

```yaml
pipeline:
  schedule: "04:00"                    # HH:MM, 24-hour format (default for new interests)
  timezone: "Europe/Berlin"
  max_retries: 2                       # Retry each stage up to 2 extra times
  max_refinement_rounds: 3             # How many rounds of "fix this"
  retry_backoff_seconds: 30            # Wait between retries
  article_fetch_timeout_seconds: 15    # Max wait per article fetch
  llm_request_timeout_seconds: 120     # Max wait per LLM API call
  max_themes: 10                       # Max themes per run
```

| Field | Meaning |
|-------|---------|
| `schedule` | Default start time for new interests. Each interest can have its own schedule set via the web UI. |
| `timezone` | Timezone for scheduling and log timestamps. |
| `max_retries` | Extra attempts per stage. With `max_retries: 2`, each stage gets up to 3 total attempts (1 initial + 2 retries). |
| `max_refinement_rounds` | After generating a deliverable, the evaluator checks quality. If it fails, the generator gets another shot with feedback. This controls how many times that cycle repeats before auto-approving. More rounds = higher quality but more API cost. |
| `retry_backoff_seconds` | How long to wait before retrying a failed stage. |
| `article_fetch_timeout_seconds` | Some sites are slow. This is the max time to wait for one article before giving up on it. |
| `llm_request_timeout_seconds` | Max wait for an OpenRouter API response. 120s is generous — DeepSeek usually responds in 5–15s. Bump this up if you switch to a slower model. |
| `max_themes` | Maximum number of themes the analyzer will identify per run. |

### `email` — How to send the digest

```yaml
email:
  recipient: "you@gmail.com"           # Where the digest goes
  sender: "pipeline@gmail.com"         # The Gmail account that sends it
  smtp_host: "smtp.gmail.com"
  smtp_port: 587                       # 587 = STARTTLS, 465 = SSL
  smtp_user: "pipeline@gmail.com"      # Usually same as sender
  smtp_password_env: "GMAIL_APP_PASSWORD"  # Name of the env var with the password
```

| Field | Meaning |
|-------|---------|
| `recipient` | Where you receive the digest. Can be any email address, not necessarily Gmail. |
| `sender` | The Gmail address that sends the email. Must match the account you generated the App Password for. |
| `smtp_host` | Gmail's SMTP server. Don't change this unless you're using a different email provider. |
| `smtp_port` | 587 with STARTTLS is the modern standard. Works with Gmail. |
| `smtp_user` | Usually the same as `sender`. This is the login username for SMTP authentication. |
| `smtp_password_env` | **Don't put your password here.** This is the name of the environment variable that holds the password. The installer sets this to `GMAIL_APP_PASSWORD`, which gets loaded from `/opt/ai-news-pipeline/.env`. |

### `database` — Where data lives

```yaml
database:
  path: "pipeline.db"                  # Relative to --config directory
```

The database uses SQLite with WAL mode (Write-Ahead Logging), which means the analyzer can read while the scraper writes — no locking issues. At ~5 MB/day growth, a 240 GB SSD lasts about 130 years before this becomes a storage problem.

To inspect the database directly: `sqlite3 /opt/ai-news-pipeline/data/pipeline.db`

### `openrouter` — API connection

```yaml
openrouter:
  api_key_env: "OPENROUTER_API_KEY"   # Name of the env var with the API key
  base_url: "https://openrouter.ai/api/v1"
```

| Field | Meaning |
|-------|---------|
| `api_key_env` | The environment variable name that holds your OpenRouter key. Set in `.env`. |
| `base_url` | The API endpoint. OpenRouter is OpenAI-compatible, so this follows the standard `/chat/completions` path. |

### `server` — Web admin UI

```yaml
server:
  port: 8443
  admin_password: "your-password-here"
  cert_dir: "/opt/ai-news-pipeline"
```

| Field | Meaning |
|-------|---------|
| `port` | HTTPS port for the admin web UI. Default: 8443. |
| `admin_password` | Password for HTTP Basic Auth. Required to access the dashboard. |
| `cert_dir` | Directory where the self-signed TLS certificate (`cert.pem` + `key.pem`) is stored or auto-generated. |

---

## Multi-Interest Architecture

This is the biggest change from v1. Instead of one hardcoded pipeline run, you can now manage **multiple independent "interests"** — each with its own:

- **Name** (e.g. "AI News", "Climate Tech", "Crypto")
- **Schedule** (start time + interval — e.g. every 24 hours at 04:00, or every 12 hours)
- **Feed list** (which RSS/Atom feeds to scrape)
- **Deliverable toggles** (which outputs to generate: summary, English script, German script)
- **Word count targets** (how long each deliverable should be)
- **Data-length mode** (full article, truncated to word count, or headers only)

### Managing interests through the web UI

Navigate to `https://your-server:8443` and log in. The dashboard shows all your interests, their status, next run time, and a "Run Now" button for each.

- **Create**: Click "New Interest", give it a name, set the schedule, select which deliverables to generate.
- **Edit**: Click an interest to change its settings, add/remove feeds, or adjust word counts.
- **Delete**: Remove an interest entirely (does not delete the generated content — that stays in the database).
- **Run Now**: Trigger an immediate pipeline run for a specific interest, regardless of schedule.

### Global config

The "Global Configuration" page in the web UI lets you edit `pipeline.yaml`, `models.yaml`, `email.yaml`, `database.yaml`, and `openrouter.yaml` without SSH-ing into your server.

---

## Secrets & Security

### The `.env` file

Secrets (API keys, passwords) are stored in `/opt/ai-news-pipeline/.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

This file must be readable only by the pipeline user:

```bash
sudo chmod 600 /opt/ai-news-pipeline/.env
sudo chown ai-news-pipeline:ai-news-pipeline /opt/ai-news-pipeline/.env
```

The installer does this automatically.

**Why not put secrets in the YAML config?** Because config files might get shared, committed to git, or accidentally pasted into a chat. The `.env` file is in `.gitignore` and has restrictive permissions. Keep it that way.

**Why App Passwords instead of your real Gmail password?** App Passwords are scoped — they can send email but can't read your inbox, change your password, or delete your account. If the pipeline has a bug that leaks credentials (theoretically), the damage is contained. Also, if you use 2FA (you should), your real password won't work for SMTP anyway.

**Why self-signed HTTPS instead of plain HTTP?** Even on a private VPS, sending passwords over plain HTTP is a bad habit. The self-signed cert encrypts your connection. The browser warning is cosmetic — the encryption is real.

---

## Daily Operations

### Checking if the pipeline is running

```bash
sudo systemctl status ai-news-pipeline.service
```

This shows whether the persistent daemon is alive. For scheduled runs, check the dashboard at `https://your-server:8443`.

### Running an interest manually (right now)

Either:
- Log into the web UI and click "Run Now" on the interest, OR
- SSH in and trigger via the old oneshot runner:

```bash
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python \
  /opt/ai-news-pipeline/src/main_old.py \
  --config /opt/ai-news-pipeline/config/
```

### Viewing logs

```bash
# Live tail of all pipeline output
sudo journalctl -u ai-news-pipeline.service -f

# Last 100 lines
sudo journalctl -u ai-news-pipeline.service -n 100 --no-pager

# Structured JSON log file
tail -f /opt/ai-news-pipeline/logs/pipeline.log
```

### Running the tests

```bash
cd /opt/ai-news-pipeline
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python -m pytest tests/ -v --tb=short
```

### Updating the pipeline code

```bash
cd ~/ai_news_scraper
git pull

# Copy updated files
sudo -u ai-news-pipeline cp -r src/ /opt/ai-news-pipeline/src/
sudo -u ai-news-pipeline cp -r prompts/ /opt/ai-news-pipeline/prompts/
sudo -u ai-news-pipeline cp -r templates/ /opt/ai-news-pipeline/templates/

# If requirements.txt changed
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt

# If the database schema changed (rare)
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python \
  /opt/ai-news-pipeline/src/main_old.py --init-db --config /opt/ai-news-pipeline/config/

# Restart the service to pick up code changes
sudo systemctl restart ai-news-pipeline.service
```

### Adding a new interest

The easiest way is through the web UI — click "New Interest" on the dashboard. If you prefer the CLI:

```bash
sqlite3 /opt/ai-news-pipeline/data/pipeline.db "INSERT INTO interests ..."
# Then restart the scheduler from the web UI, or restart the service:
sudo systemctl restart ai-news-pipeline.service
```

### Changing an interest's schedule

Edit the interest through the web UI (click "Edit" next to it). The scheduler picks up changes immediately — no restart needed.

---

## Troubleshooting

### "Service is active but no runs are happening"

1. Check the web UI dashboard: are there any interests configured? Do they have feeds?
2. Check the logs: `sudo journalctl -u ai-news-pipeline.service -n 100 --no-pager`
3. Look for scheduler messages about missed catch-up runs
4. Click "Run Now" on an interest in the web UI to force a run

### "Browser says connection is not private"

This is the self-signed certificate. Click "Advanced" → "Proceed to site" (Chrome) or "Accept the Risk and Continue" (Firefox). This is expected and safe for your own server.

### "Can't log into the web UI" / 401 Unauthorized

You didn't set the admin password in `config/server.yaml`, or it was auto-generated. Check:

```bash
sudo cat /opt/ai-news-pipeline/config/server.yaml | grep admin_password
```

Set a new password there, then restart the service: `sudo systemctl restart ai-news-pipeline.service`.

### "Connection refused" or HTTP errors during scraping

Some feeds block datacenter IPs or require a User-Agent. If a specific feed consistently fails, try removing it from that interest's feed list via the web UI and checking if the URL is still valid in a browser.

### "OpenRouter API error: 401 Unauthorized"

Your API key is wrong or has zero credits. Check:
```bash
sudo cat /opt/ai-news-pipeline/.env
```
Then visit [openrouter.ai/credits](https://openrouter.ai/credits) to verify your balance.

### "SMTP authentication failed"

Gmail App Password might have expired or been revoked. Generate a new one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) and update `.env`.

### "No module named 'feedparser'" or other import errors

The venv wasn't created or packages weren't installed. Re-run:
```bash
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt
```

### Restarting after a crash or reboot

The service has `Restart=always` and auto-starts on boot. If the process crashes, systemd restarts it within 10 seconds. APScheduler re-evaluates missed runs and catches up automatically.

---

## File Layout

Everything the pipeline needs lives under `/opt/ai-news-pipeline/`:

```
/opt/ai-news-pipeline/
├── .env                            # API keys (secret — chmod 600)
├── requirements.txt                # Pinned Python dependencies
├── cert.pem                        # Self-signed TLS certificate (auto-generated)
├── key.pem                         # Private key for TLS (auto-generated)
├── config/
│   ├── feeds.yaml                  # Default feed template for new interests
│   ├── models.yaml                 # LLM model assignments
│   ├── pipeline.yaml               # Runtime behavior and thresholds
│   ├── email.yaml                  # SMTP delivery settings
│   ├── database.yaml               # SQLite database path
│   ├── openrouter.yaml             # OpenRouter API connection
│   └── server.yaml                 # Web UI port, admin password, cert dir
├── src/
│   ├── main.py                     # Entry point: persistent server + scheduler
│   ├── main_old.py                 # Legacy entry point: oneshot batch mode
│   ├── pipeline.py                 # Orchestrator: per-interest pipeline execution
│   ├── config.py                   # Reads & validates YAML config files
│   ├── db.py                       # SQLite database layer (8 tables + auto-migration)
│   ├── scraper.py                  # RSS parsing + article text extraction
│   ├── analyzer.py                 # Theme identification via LLM
│   ├── generator.py                # Summary + script generation
│   ├── evaluator.py                # Quality evaluation + refinement loop
│   ├── brief.py                    # Daily brief synthesis
│   ├── emailer.py                  # SMTP dispatch (Gmail)
│   ├── llm.py                      # OpenRouter HTTP client
│   ├── scheduler.py                # APScheduler wrapper for multi-interest timing
│   ├── server.py                   # Flask HTTPS web admin UI
│   ├── certs.py                    # Self-signed TLS certificate generator
│   └── models.py                   # Data models (Pydantic)
├── prompts/                        # LLM prompt templates (plain text)
│   ├── analyze.txt
│   ├── summary_en.txt
│   ├── script_en.txt
│   ├── script_de.txt
│   ├── evaluate_quality.txt
│   ├── evaluate_adversarial.txt
│   ├── refine.txt
│   └── brief.txt
├── templates/                      # Flask HTML templates for the web UI
│   ├── base.html
│   ├── dashboard.html
│   ├── interest_editor.html
│   └── global_config.html
├── tests/
│   ├── test_*.py                   # Unit + integration tests
│   └── fixtures/                   # Sample data for tests
├── deploy/
│   ├── install.sh                  # One-shot installer
│   ├── ai-news-pipeline.service    # systemd service definition (persistent)
│   ├── fresh_deploy.sh             # Remote deploy via Paramiko
│   ├── monitor.sh                  # Remote monitoring script
│   └── logrotate.conf              # Log rotation rules
├── venv/                           # Python virtual environment (auto-created)
├── data/
│   └── pipeline.db                 # SQLite database (created at first run)
└── logs/
    └── pipeline.log                # Structured JSON log (runtime)
```

---

## Architecture (For the Curious)

### Design decisions

- **Persistent server + in-process scheduler** instead of systemd timer: enables multi-interest scheduling, web UI, and automatic catch-up on missed runs.
- **Sequential stages within a run** instead of parallel: simpler error handling, no rate-limit contention on the LLM API, and the scraper + analyzer only take ~2 minutes of the ~10-minute total runtime.
- **Per-theme parallelism:** deliverables for different themes are generated concurrently (ThreadPoolExecutor, 3 workers), and English + German scripts within a theme are generated in parallel.
- **Each stage retried up to 3 times** (1 attempt + 2 retries) with 30-second backoff: handles transient failures without getting stuck.
- **Deliverables refined up to 3 rounds, then auto-approved:** the evaluator isn't perfect, and an imperfect deliverable is better than no deliverable.
- **Two-model strategy (strong + weak):** writing needs quality, evaluation needs speed and cheapness. You can override this to use one model for both.
- **SQLite with WAL mode:** concurrent reads during writes, zero setup, negligible maintenance.
- **Secrets only in environment variables:** nothing sensitive in config files, nothing sensitive in git.
- **Self-signed TLS for the admin UI:** encrypted connections without the hassle of Let's Encrypt for a private dashboard.
- **Multi-interest design:** each interest is an independent pipeline configuration with its own feeds, schedule, and output settings — all stored in the database.
- **Database auto-migration:** upgrades from v1 (single-interest) to v2 (multi-interest) schema automatically on first run.

### Pipeline stages in order (per interest)

| Stage | File | What happens | Retryable |
|-------|------|-------------|-----------|
| Init | `pipeline.py` | Parse config, load DB, recover orphaned articles, load previous brief | Yes |
| Scrape | `scraper.py` | Fetch all feeds for this interest, extract full article text | Yes |
| Analyze | `analyzer.py` | Find 1–10 themes, classify as novel or continuation | Yes |
| Generate+Eval | `generator.py` + `evaluator.py` | For each theme: generate up to 3 deliverables, evaluate, refine up to 3 rounds. Themes processed concurrently. | Yes |
| Brief | `brief.py` | Synthesize a daily brief from approved themes | Yes |
| Email | `emailer.py` | Send one email per theme + one daily brief email | Yes |

### Email format

You receive one email per discovered theme (each containing the summary, English script, and German script), plus one "daily brief" email that stitches all themes together. On failure, you get one alert email with the stage name, error traceback, and recent log lines — enough context to debug without logging into the server.

### Web UI routes

| Route | Method | Auth | Description |
|-------|--------|------|-------------|
| `/` | GET | Yes | Dashboard with all interests, status, next run times |
| `/interest/new` | GET/POST | Yes | Create a new interest |
| `/interest/<id>/edit` | GET/POST | Yes | Edit an interest + manage its feeds |
| `/interest/<id>/delete` | POST | Yes | Delete an interest |
| `/interest/<id>/run` | POST | Yes | Trigger immediate pipeline run |
| `/global-config` | GET/POST | Yes | Edit pipeline, models, email, database, openrouter config |
| `/interest/<id>/feed/add` | POST | Yes | Add a feed to an interest |
| `/interest/<id>/feed/<fid>/edit` | POST | Yes | Edit a feed |
| `/interest/<id>/feed/<fid>/delete` | POST | Yes | Delete a feed |

---

## Costs (Rough Estimate)

| Item | Cost |
|------|------|
| VPS (1 vCPU, 1 GB RAM) | ~$5/month |
| OpenRouter API credits | ~$15–30/month (with DeepSeek models, one interest per day) |
| Gmail | Free (within sending limits) |
| **Total** | **~$20–35/month** |

You can cut costs by switching both models to an even cheaper option (like `meta-llama/llama-3.2-3b-instruct` for the weak model) or by reducing `max_refinement_rounds` to 1. Quality will drop, but the pipeline will still work.

Running multiple interests adds cost proportionally — each additional interest doubles the LLM spend (or more, if it runs as frequently).

---

## License

MIT — see [LICENSE](LICENSE).
