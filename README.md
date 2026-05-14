# AI News Pipeline

Automated nightly batch pipeline that scrapes 50 AI-focused RSS feeds, identifies novel themes, and generates polished deliverables (English summaries, English/German YouTube scripts) — all dispatched via email at 04:00 Europe/Berlin.

## Overview

The pipeline runs on an Ubuntu 24.04 VPS, triggered by a systemd timer. Each night it:

1. **Scrapes** 30 news + 20 commentator RSS feeds, extracting full article content (with paywall fallback)
2. **Analyzes** articles against the previous day's brief to surface 1–5 novel or evolving themes
3. **Generates** per theme: ~750-word English summary, ~1000–1500-word English YouTube script, and a natively-written German YouTube script
4. **Evaluates** all deliverables via LLM-based quality + adversarial fact-checking, with up to 3 refinement rounds
5. **Produces** a ~700-word daily brief synthesizing all approved themes
6. **Emails** one daily brief + one email per theme (with all three deliverables) to the configured recipient

On failure, it sends an alert email with the stage name, error traceback, and recent log output.

## Architecture

```
┌─────────────────────────────────┐
│         SYSTEMD TIMER            │
│    (04:00 Europe/Berlin daily)    │
└───────────────┬──────────────────┘
                │ triggers
                ▼
┌─────────────────────────────────────────────────────┐
│              PIPELINE ORCHESTRATOR                    │
│              (main.py — sequential stages)            │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────┐  ┌──────┐│
│  │  SCRAPE  │→│ ANALYZE  │→│ GEN+EVAL per  │→│ BRIEF │→│EMAIL ││
│  │  STAGE   │  │  STAGE   │  │   THEME       │  │ STAGE │  │STAGE ││
│  └──────────┘  └──────────┘  └──────────────┘  └───────┘  └──────┘│
│       │              │              │               │          │    │
│       ▼              ▼              ▼               ▼          ▼    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      SQLITE DATABASE (WAL)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Failure at any stage after retries → FAILURE ALERT EMAIL         │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- Sequential stage execution (simpler error handling, no rate-limit contention)
- Each stage retried up to 2 times (3 total attempts) with 30s backoff
- Deliverables refined up to 3 rounds, then auto-approved
- All LLM calls via OpenRouter (`deepseek/deepseek-v4-pro` for generation, `deepseek/deepseek-v4-flash` for evaluation)
- Gmail SMTP with App Password for email delivery
- Secrets loaded from environment variables only

## Requirements

- **OS:** Ubuntu 24.04 LTS
- **Python:** 3.12+
- **Disk:** ~2 GB (database grows ~5 MB/day)
- **Network:** Outbound HTTPS access (RSS feeds, OpenRouter API, Gmail SMTP)
- **No inbound ports** required

### Dependencies

```
feedparser==6.0.11
trafilatura==1.12.0
httpx==0.27.2
pyyaml==6.0.2
python-dotenv==1.0.1
pydantic==2.9.2
```

## Quick Start

### 1. Clone and set up the service user

```bash
sudo useradd --system --home-dir /opt/ai-news-pipeline --shell /usr/sbin/nologin ai-news-pipeline
sudo mkdir -p /opt/ai-news-pipeline/{config,data,logs,src,prompts,tests,tests/fixtures,deploy,venv}
sudo cp -r src/ prompts/ tests/ requirements.txt /opt/ai-news-pipeline/
sudo chown -R ai-news-pipeline:ai-news-pipeline /opt/ai-news-pipeline
```

### 2. Create virtual environment

```bash
sudo -u ai-news-pipeline python3 -m venv /opt/ai-news-pipeline/venv
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt
```

### 3. Configure secrets

Create `/opt/ai-news-pipeline/.env` (chmod 600):

```bash
sudo -u ai-news-pipeline cat > /opt/ai-news-pipeline/.env << 'EOF'
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
EOF
sudo chmod 600 /opt/ai-news-pipeline/.env
```

### 4. Configure feeds

Copy and edit `config/feeds.yaml` with your actual feed URLs. See `config/feeds.yaml` for the schema.

### 5. Initialize the database

```bash
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --init-db --config /opt/ai-news-pipeline/config/feeds.yaml
```

### 6. Install systemd timer

```bash
sudo cp deploy/ai-news-pipeline.service /etc/systemd/system/
sudo cp deploy/ai-news-pipeline.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-news-pipeline.timer
```

### 7. Install logrotate (optional)

```bash
sudo cp deploy/logrotate.conf /etc/logrotate.d/ai-news-pipeline
```

## Configuration

### feeds.yaml

```yaml
feeds:
  news:
    - name: "Ars Technica AI"
      url: "https://feeds.arstechnica.com/arstechnica/technology-lab"
  commentators:
    - name: "Simon Willison"
      url: "https://simonwillison.net/atom/everything/"

models:
  strong:
    id: "deepseek/deepseek-v4-pro"
    temperature: 0.7
  weak:
    id: "deepseek/deepseek-v4-flash"
    temperature: 0.7

pipeline:
  schedule: "04:00"
  timezone: "Europe/Berlin"
  max_retries: 2
  max_refinement_rounds: 3

email:
  recipient: "recipient@gmail.com"
  sender: "sender@gmail.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  smtp_user: "sender@gmail.com"
  smtp_password_env: "GMAIL_APP_PASSWORD"

database:
  path: "/opt/ai-news-pipeline/data/pipeline.db"

openrouter:
  api_key_env: "OPENROUTER_API_KEY"
  base_url: "https://openrouter.ai/api/v1"
```

## Directory Layout

```
/opt/ai-news-pipeline/
├── config/
│   └── feeds.yaml                 # Feed list and pipeline config
├── data/
│   └── pipeline.db                # SQLite database (runtime)
├── logs/
│   └── pipeline.log               # Structured JSON logs (runtime)
├── src/
│   ├── main.py                    # Orchestrator entry point
│   ├── config.py                  # Config loading and validation
│   ├── db.py                      # Database layer
│   ├── llm.py                     # OpenRouter LLM client
│   ├── scraper.py                 # RSS scraping + article extraction
│   ├── analyzer.py                # Theme identification
│   ├── generator.py               # Deliverable generation
│   ├── evaluator.py               # Quality + adversarial evaluation
│   ├── brief.py                   # Daily brief generation
│   ├── emailer.py                 # Email dispatch
│   └── models.py                  # Dataclass/type definitions
├── prompts/
│   ├── analyze.txt
│   ├── summary_en.txt
│   ├── script_en.txt
│   ├── script_de.txt
│   ├── evaluate_quality.txt
│   ├── evaluate_adversarial.txt
│   ├── refine.txt
│   └── brief.txt
├── tests/
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_llm.py
│   ├── test_scraper.py
│   ├── test_analyzer.py
│   ├── test_generator.py
│   ├── test_evaluator.py
│   ├── test_brief.py
│   ├── test_emailer.py
│   ├── test_integration.py
│   └── fixtures/
├── deploy/
│   ├── ai-news-pipeline.service
│   ├── ai-news-pipeline.timer
│   └── logrotate.conf
├── .env.example
├── requirements.txt
└── README.md
```

## Usage

### Manual run

```bash
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --config /opt/ai-news-pipeline/config/feeds.yaml
```

Or via systemd:

```bash
sudo systemctl start ai-news-pipeline.service
sudo journalctl -u ai-news-pipeline.service -f
```

### Check timer status

```bash
sudo systemctl list-timers ai-news-pipeline.timer
```

### View logs

```bash
# Journald
journalctl -u ai-news-pipeline.service

# Log file
tail -f /opt/ai-news-pipeline/logs/pipeline.log
```

### Inspect database

```bash
sqlite3 /opt/ai-news-pipeline/data/pipeline.db
```

### Run tests

```bash
cd /opt/ai-news-pipeline
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python -m pytest tests/ -v --tb=short
```

### Managing feeds

Add or remove feeds by editing `/opt/ai-news-pipeline/config/feeds.yaml`. No restart needed — config is read at pipeline start.

### Updating

```bash
# Copy updated source and prompts
sudo -u ai-news-pipeline cp -r src/ /opt/ai-news-pipeline/src/
sudo -u ai-news-pipeline cp -r prompts/ /opt/ai-news-pipeline/prompts/

# If dependencies changed
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt

# If schema changed
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --init-db --config /opt/ai-news-pipeline/config/feeds.yaml
```

## License

MIT — see [LICENSE](LICENSE).
