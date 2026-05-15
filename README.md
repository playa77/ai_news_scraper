# AI News Pipeline

Every night at 4 AM, this pipeline scrapes ~50 AI-focused news sites and blogs, finds the 1–5 biggest themes of the day, and emails you a polished daily brief along with YouTube-ready scripts (in English and German). It runs unattended on a cheap Ubuntu VPS.

**Cool, but why?**

Because keeping up with AI news is impossible. Dozens of sites publish every day, and most of it is noise. This pipeline does the reading for you, picks out what actually matters, and hands you a finished summary — plus scripts you can literally read off a teleprompter if you run a YouTube channel.

---

## How It Works (90 Seconds)

```
Your VPS wakes up at 04:00 Berlin time
    │
    ├─ Scrapes ~50 RSS feeds, pulls full article text
    │
    ├─ Compares today's articles against yesterday's brief
    │   to find genuinely new themes (not "AI is changing everything" — again)
    │
    ├─ For each theme found, generates three deliverables:
    │   • ~750-word English summary
    │   • ~1000–1500-word English YouTube script
    │   • Natively-written German YouTube script
    │
    ├─ Runs each deliverable through quality checks and
    │   adversarial fact-checking, refining up to 3 times
    │
    ├─ Writes a ~700-word daily brief tying all themes together
    │
    └─ Emails everything to you via Gmail. Done.
```

If anything breaks, you get an alert email with exactly what failed and why.

---

## Prerequisites

You need a Linux server (or an old laptop running Linux) with:

| Thing | Minimum | Notes |
|-------|---------|-------|
| OS | Ubuntu 24.04 LTS | Other Debian-based distros should work, but untested |
| Python | 3.12 or newer | Comes with Ubuntu 24.04 by default |
| Disk | ~2 GB free | Database grows about 5 MB per day |
| Network | Outbound internet | To reach RSS feeds, OpenRouter API, and Gmail SMTP |
| Inbound ports | None | No firewall holes needed |

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

### Step 3: Run the installer

```bash
sudo bash deploy/install.sh
```

The installer does everything: creates a locked-down system user, sets up a Python virtual environment, copies files into `/opt/ai-news-pipeline/`, and schedules the daily run. You'll see output like:

```
=== AI News Pipeline Installation ===
[1/8] Creating system user 'ai-news-pipeline'...
[2/8] Creating directory structure under /opt/ai-news-pipeline...
[3/8] Setting ownership to ai-news-pipeline...
[4/8] Copying project files...
[5/8] Creating Python virtual environment...
[6/8] Creating .env file from template...
[7/8] Installing systemd units and logrotate config...
[8/8] Enabling and starting systemd timer...

=== Installation complete ===
```

### Step 4: Verify it works

```bash
sudo systemctl status ai-news-pipeline.timer
```

You should see `Active: active (waiting)` and the next trigger time. If you want to run it right now instead of waiting for 4 AM:

```bash
sudo systemctl start ai-news-pipeline.service
```

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

### What's systemd and why a timer?

**systemd** is the thing that starts and stops programs on Linux. On Ubuntu, it runs as PID 1 (the first process, the one that launches everything else). It can:

- Start a program when the server boots
- Restart a program if it crashes
- Run a program on a schedule

A **systemd service** defines *what* to run (our pipeline). A **systemd timer** defines *when* (every day at 04:00 Europe/Berlin).

Our service is `Type=oneshot`, meaning it runs once and exits — like a cron job, but with better logging and failure handling.

### Why not cron?

Cron is fine, but systemd gives us:

- Logs that go to journald (viewable with `journalctl`)
- Automatic retry detection via `Persistent=true` on the timer (if the server was off at 4 AM, the job runs as soon as it boots)
- Clean environment variable handling via `EnvironmentFile`

### Do I need `loginctl enable-linger`?

**No — not for this project.** A quick explainer so you know why:

- **System services** live in `/etc/systemd/system/`. systemd itself manages them. Even though our service runs as the user `ai-news-pipeline`, it's still a system service — the `User=` directive just says "run this process as that user."
- **User services** live in `~/.config/systemd/user/`. These are tied to a user session, and they die when the user logs out — unless you run `loginctl enable-linger <username>`, which tells systemd to keep the user's service manager alive even when they're not logged in.

Since our service is installed at `/etc/systemd/system/ai-news-pipeline.service`, it's a system service. No linger needed.

### Terminal multiplexers: tmux and screen

If you're managing a server over SSH, your connection can drop. When it does, any program you were running in that terminal dies with it. A **terminal multiplexer** solves this by running a persistent session on the server that survives disconnects.

- **tmux** (recommended): Install with `sudo apt install tmux`. Start a session with `tmux new -s mysession`, detach with `Ctrl+B then D`, reattach with `tmux attach -t mysession`.
- **screen**: The older alternative. `screen -S mysession`, detach with `Ctrl+A then D`, reattach with `screen -r mysession`.

For this project specifically, you probably don't need a multiplexer day-to-day because the pipeline runs via systemd — not from your terminal. But tmux is useful when you're debugging: start a tmux session, run `journalctl -u ai-news-pipeline.service -f` to tail the logs, detach, and come back later to see what happened.

```bash
# Quick tmux workflow for monitoring a pipeline run
tmux new -s pipeline-watch
sudo systemctl start ai-news-pipeline.service
journalctl -u ai-news-pipeline.service -f
# Ctrl+B, D to detach. Come back later:
tmux attach -t pipeline-watch
```

---

## Configuration: Every Setting Explained

All configuration lives in `/opt/ai-news-pipeline/config/` as separate, domain-specific YAML files. Each file controls one aspect of the pipeline:

| File | Purpose |
|------|---------|
| `feeds.yaml` | RSS/Atom feed sources to scrape |
| `models.yaml` | LLM model assignments (strong vs weak) |
| `pipeline.yaml` | Runtime behavior (retries, timeouts, schedule) |
| `email.yaml` | SMTP settings for delivering the digest |
| `database.yaml` | SQLite database path |
| `openrouter.yaml` | OpenRouter API connection |

### `feeds` — What sources to scrape

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

To add a feed, just add another entry under the right category. To remove one, delete its block. No restart needed — feeds are read at the start of each run.

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
  schedule: "04:00"                    # HH:MM, 24-hour format
  timezone: "Europe/Berlin"
  max_retries: 2                       # Retry each stage up to 2 extra times
  max_refinement_rounds: 3             # How many rounds of "fix this"
  retry_backoff_seconds: 30            # Wait between retries
  article_fetch_timeout_seconds: 15    # Max wait per article fetch
  llm_request_timeout_seconds: 120     # Max wait per LLM API call
```

| Field | Meaning |
|-------|---------|
| `schedule` | What time the timer triggers. Note: this is informational — the actual schedule is in the systemd timer file (`OnCalendar=*-*-* 04:00 Europe/Berlin`). Changing it here doesn't change when the pipeline runs; you'd also need to edit the timer. |
| `timezone` | Timezone for log timestamps. |
| `max_retries` | Extra attempts per stage. With `max_retries: 2`, each stage gets up to 3 total attempts (1 initial + 2 retries). |
| `max_refinement_rounds` | After generating a deliverable, the evaluator checks quality. If it fails, the generator gets another shot with feedback. This controls how many times that cycle repeats before auto-approving. More rounds = higher quality but more API cost. |
| `retry_backoff_seconds` | How long to wait before retrying a failed stage. |
| `article_fetch_timeout_seconds` | Some sites are slow. This is the max time to wait for one article before giving up on it. |
| `llm_request_timeout_seconds` | Max wait for an OpenRouter API response. 120s is generous — DeepSeek usually responds in 5–15s. Bump this up if you switch to a slower model. |

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
  path: "pipeline.db"                  # Relative to the --config directory
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

---

## Daily Operations

### Checking if it ran last night

```bash
# See timer status and next run time
sudo systemctl status ai-news-pipeline.timer

# See when it last triggered
sudo systemctl list-timers ai-news-pipeline.timer

# Read the last run's log
sudo journalctl -u ai-news-pipeline.service --since "yesterday" --no-pager
```

### Running it manually (right now)

```bash
sudo systemctl start ai-news-pipeline.service
```

Watch the logs live in another terminal (or tmux pane):

```bash
sudo journalctl -u ai-news-pipeline.service -f
```

### Running it without systemd (for debugging)

```bash
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python \
  /opt/ai-news-pipeline/src/main.py \
  --config /opt/ai-news-pipeline/config/
```

### Viewing the persistent log file

In addition to journald, the pipeline writes structured JSON logs to `/opt/ai-news-pipeline/logs/pipeline.log`:

```bash
tail -f /opt/ai-news-pipeline/logs/pipeline.log
```

### Running the tests

```bash
cd /opt/ai-news-pipeline
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python -m pytest tests/ -v --tb=short
```

### Adding or removing feeds

Edit `/opt/ai-news-pipeline/config/feeds.yaml`. Changes take effect on the next run — no restart or reload needed.

### Updating the pipeline code

```bash
cd ~/ai_news_scraper
git pull

# Copy updated files
sudo -u ai-news-pipeline cp -r src/ /opt/ai-news-pipeline/src/
sudo -u ai-news-pipeline cp -r prompts/ /opt/ai-news-pipeline/prompts/

# If requirements.txt changed
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt

# If the database schema changed (rare)
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python \
  /opt/ai-news-pipeline/src/main.py --init-db --config /opt/ai-news-pipeline/config/
```

---

## Troubleshooting

### "Timer is active but nothing happens at 4 AM"

1. Check if the server was even on: `uptime`
2. systemd's `Persistent=true` means it should catch up, but verify: `sudo systemctl list-timers ai-news-pipeline.timer`
3. Try running manually: `sudo systemctl start ai-news-pipeline.service`
4. Check the logs: `sudo journalctl -u ai-news-pipeline.service -n 100 --no-pager`

### "Connection refused" or HTTP errors during scraping

Some feeds block datacenter IPs or require a User-Agent. If a specific feed consistently fails, try removing it from `feeds.yaml` and checking if the URL is still valid in a browser.

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

### Changing the schedule

The schedule lives in the systemd timer, not in the config files. Edit `/etc/systemd/system/ai-news-pipeline.timer`:

```ini
[Timer]
OnCalendar=*-*-* 06:30 Europe/Berlin   # Change the time here
Persistent=true
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ai-news-pipeline.timer
```

`OnCalendar` format is `DayOfWeek Year-Month-Day Hour:Minute:Second Timezone`. `*-*-*` means "every day." See `man systemd.timer` for the full spec.

---

## File Layout

Everything the pipeline needs lives under `/opt/ai-news-pipeline/`:

```
/opt/ai-news-pipeline/
├── .env                            # API keys (secret — chmod 600)
├── requirements.txt                # Pinned Python dependencies
├── config/
│   ├── feeds.yaml                  # News and commentator feed sources
│   ├── models.yaml                 # LLM model assignments
│   ├── pipeline.yaml               # Runtime behavior and thresholds
│   ├── email.yaml                  # SMTP delivery settings
│   ├── database.yaml               # SQLite database path
│   └── openrouter.yaml             # OpenRouter API connection
├── src/
│   ├── main.py                     # Entry point: parses args, runs the show
│   ├── config.py                   # Reads & validates config files
│   ├── db.py                       # SQLite database layer (7 tables)
│   ├── scraper.py                  # RSS parsing + article text extraction
│   ├── analyzer.py                 # Theme identification via LLM
│   ├── generator.py                # Summary + script generation
│   ├── evaluator.py                # Quality evaluation + refinement loop
│   ├── brief.py                    # Daily brief synthesis
│   ├── emailer.py                  # SMTP dispatch (Gmail)
│   ├── llm.py                      # OpenRouter HTTP client
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
├── tests/
│   ├── test_*.py                   # Unit + integration tests
│   └── fixtures/                   # Sample data for tests
├── deploy/
│   ├── install.sh                  # One-shot installer
│   ├── ai-news-pipeline.service    # systemd service definition
│   ├── ai-news-pipeline.timer      # systemd timer definition
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

- **Sequential stages** instead of parallel: simpler error handling, no rate-limit contention on the LLM API, and the scraper + analyzer only take ~2 minutes of the ~10-minute total runtime.
- **Each stage retried 3 times** (1 attempt + 2 retries) with 30-second backoff: handles transient failures without getting stuck.
- **Deliverables refined up to 3 rounds, then auto-approved:** the evaluator isn't perfect, and an imperfect deliverable is better than no deliverable.
- **Two-model strategy (strong + weak):** writing needs quality, evaluation needs speed and cheapness. You can override this to use one model for both.
- **SQLite with WAL mode:** concurrent reads during writes, zero setup, negligible maintenance.
- **Secrets only in environment variables:** nothing sensitive in config files, nothing sensitive in git.
- **Failure alert email** includes the stage name, error traceback, and recent log lines: enough context to debug without logging into the server.

### Pipeline stages in order

| Stage | File | What happens | Retryable |
|-------|------|-------------|-----------|
| Init | `main.py` | Parse config, init DB, load previous brief | Yes |
| Scrape | `scraper.py` | Fetch all feeds, extract full article text | Yes |
| Analyze | `analyzer.py` | Find 1–5 themes, classify as novel or continuation | Yes |
| Generate+Eval | `generator.py` + `evaluator.py` | For each theme: generate 3 deliverables, evaluate, refine up to 3 rounds | Yes |
| Brief | `brief.py` | Synthesize a daily brief from approved themes | Yes |
| Email | `emailer.py` | Send one email per theme + one daily brief email | Yes |

### Email format

You receive one email per discovered theme (each containing the summary, English script, and German script), plus one "daily brief" email that stitches all themes together. On failure, you get one alert email.

---

## Costs (Rough Estimate)

| Item | Cost |
|------|------|
| VPS (1 vCPU, 1 GB RAM) | ~$5/month |
| OpenRouter API credits | ~$15–30/month (with DeepSeek models) |
| Gmail | Free (within sending limits) |
| **Total** | **~$20–35/month** |

You can cut costs by switching both models to an even cheaper option (like `meta-llama/llama-3.2-3b-instruct` for the weak model) or by reducing `max_refinement_rounds` to 1. Quality will drop, but the pipeline will still work.

---

## License

MIT — see [LICENSE](LICENSE).
