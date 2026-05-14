# Design Document

## 1. Overview & Goals

A nightly batch pipeline running on an Ubuntu 24.04 VPS that scrapes 50 AI-focused RSS feeds, distills new and updated themes from the content, generates three deliverables per theme (English summary, English YouTube script, natively-written German YouTube script), subjects them to quality evaluation and adversarial review with a refinement loop, produces a daily brief, and dispatches results via email. The system operates autonomously at 04:00 Europe/Berlin time, retries failures, and alerts the operator on unrecoverable errors.

**Primary Goals:**
- Fully automated nightly execution with zero manual intervention on success.
- High-quality, style-consistent deliverables enforced by LLM-based evaluation and adversarial review.
- Deterministic novelty detection by comparing against the previous day's brief.
- Clear failure visibility via alert emails with diagnostic logs.

---

## 2. Requirements Summary

| # | Requirement | Source |
|---|-------------|--------|
| R1 | Scrape 30 news + 20 commentator RSS feeds nightly at 04:00 Europe/Berlin | Operator brief |
| R2 | Extract full article content; fallback to RSS excerpt + paywall notice | Confirmed answer, Round 2 |
| R3 | Store all data in SQLite on the VPS | Confirmed answer, Round 2 |
| R4 | Compare new items against previous day's daily brief to identify 1–5 units of meaning (novel themes + meaningful updates to existing themes) | Confirmed answer, Round 2 |
| R5 | Per unit: generate ~750-word English summary | Operator brief |
| R6 | Per unit: generate ~1000–1500-word English YouTube script (Wes Roth / Karpathy style) | Operator brief |
| R7 | Per unit: generate ~1000–1500-word German YouTube script, natively written, not translated | Operator brief |
| R8 | Quality evaluator + adversarial fact-checker/bias-hunter review all deliverables | Operator brief |
| R9 | Refinement loop: up to 3 rounds, then accept as-is | Confirmed answer, Round 1 |
| R10 | Retry each pipeline stage up to 2 times on failure (3 total attempts) | Operator brief |
| R11 | Generate ~700-word English daily brief from all approved units | Operator brief |
| R12 | Email: 1 daily brief email + 1 email per unit (summary + both scripts) to recipient@gmail.com | Operator brief |
| R13 | Failure alert email with detailed logs on unrecoverable failure | Confirmed answer, Round 2 |
| R14 | All LLM calls via OpenRouter; strong role = deepseek/deepseek-v4-pro @ 0.7; weak role = deepseek/deepseek-v4-flash @ 0.7 | Confirmed answer, Round 2 |
| R15 | Feed list managed via static YAML config file on server | Confirmed answer, Round 2 |
| R16 | Email delivery via Gmail App Password SMTP | Confirmed answer, Round 2 |
| R17 | Runs on Ubuntu 24.04 VPS | Operator brief |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEMD TIMER                                │
│                   (04:00 Europe/Berlin daily)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ triggers
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PIPELINE ORCHESTRATOR                           │
│                  (main.py — sequential stages)                      │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────┐  ┌──────┐│
│  │  SCRAPE  │→│ ANALYZE  │→│ GEN+EVAL per  │→│ BRIEF │→│EMAIL ││
│  │  STAGE   │  │  STAGE   │  │   THEME       │  │ STAGE │  │STAGE ││
│  └──────────┘  └──────────┘  └──────────────┘  └───────┘  └──────┘│
│       │              │              │               │          │    │
│       ▼              ▼              ▼               ▼          ▼    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                        SQLITE DATABASE                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Failure at any stage after retries ──→ FAILURE ALERT EMAIL        │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    OPENROUTER API   │
                │  (all LLM calls)   │
                └─────────────────────┘
```

**Stage execution model:** Sequential. Each stage must complete before the next begins. Within the Generate+Evaluate stage, themes are processed sequentially (theme 1 fully completes before theme 2 begins). [COMPLEXITY JUSTIFICATION: Parallel theme processing would add async orchestration, rate-limit handling, and partial-failure complexity. Since this runs once nightly with no latency SLA, sequential is simpler and more debuggable.]

**Retry model:** Each stage wrapper catches all exceptions. On failure, the stage is retried up to 2 times (3 total attempts) with a 30-second backoff. If all attempts fail, the orchestrator sends a failure alert email and halts.

**Refinement loop model:** Internal to the Generate+Evaluate stage per theme. Separate from the retry mechanism. The refinement loop addresses quality deficiencies, not runtime errors.

---

## 4. Component Responsibilities

### 4.1 Pipeline Orchestrator (`main.py`)
- Entry point invoked by systemd timer.
- Executes stages in order: Scrape → Analyze → Generate+Evaluate (per theme) → Brief → Email.
- Wraps each stage in retry logic (max 2 retries, 30s backoff).
- On unrecoverable failure: invokes emailer to send failure alert with stage name, error traceback, and last 100 log lines.
- Creates a `pipeline_runs` record at start; updates status on completion or failure.
- Configures structured logging to both file and stdout.

### 4.2 Scraper (`scraper.py`)
- Reads feed list from YAML config.
- For each feed: fetches RSS via `feedparser`, filters items with `published_at` after the last successful scrape timestamp.
- Deduplicates across feeds by normalized URL.
- For each new item: attempts full-article extraction via `trafilatura`. If extraction fails or returns paywalled content, falls back to RSS excerpt and marks `content_status = 'excerpt_paywall'` or `'excerpt_only'`.
- Stores all articles in SQLite with `pipeline_run_id` reference.
- Marks scrape stage complete in `pipeline_runs`.

### 4.3 Analyzer (`analyzer.py`)
- Retrieves all new articles from the current pipeline run.
- Retrieves the previous day's daily brief from `daily_briefs` table (if exists; skip comparison on first run).
- Calls LLM (strong model) with a structured prompt to identify 1–5 units of meaning, classifying each as `novel` or `continuation`, with a title, description, and list of source article IDs.
- Parses LLM output into structured theme records.
- Stores themes in SQLite.

### 4.4 Generator (`generator.py`)
- For each theme, generates three deliverables by calling the LLM (strong model) with distinct prompts:
  1. **English Summary** (~750 words): Input = theme description + full text of source articles.
  2. **English Script** (~1000–1500 words): Input = theme description + source articles + English summary. Style guide embedded in prompt.
  3. **German Script** (~1000–1500 words): Input = theme description + source articles + English summary. **Not** given the English script as input, to enforce native generation. German style prompt written to produce equivalent tone and quality independently.
- Stores each deliverable version in SQLite.

### 4.5 Evaluator (`evaluator.py`)
- Two evaluation passes per theme per refinement round:
  1. **Quality Evaluator** (weak model): Assesses style adherence, word count compliance, completeness, and prose quality for all three deliverables. Returns pass/fail per deliverable + aggregated feedback.
  2. **Adversarial Fact-Checker / Bias Hunter** (weak model): Checks factual claims against source articles. Identifies unsupported assertions, hallucinations, and ideological bias. Returns pass/fail + specific feedback.
- If both evaluators pass all deliverables → theme approved.
- If any evaluator fails any deliverable → refinement round triggered.
- Refinement: Generator receives the existing deliverables + all evaluator feedback and produces revised versions. New versions stored with incremented version number.
- After 3 refinement rounds, theme is auto-approved regardless of evaluation result.
- [PROPOSED DESIGN DECISION: Evaluator uses weak model. Rationale: evaluation is criteria-matching against a rubric, not creative synthesis. If evaluation quality proves insufficient, this can be upgraded to strong model via config change without architecture changes.]

### 4.6 Brief Generator (`brief.py`)
- Retrieves all approved themes and their final English summaries from the current run.
- Calls LLM (strong model) to synthesize a ~700-word English daily brief.
- Stores brief in `daily_briefs` table. This brief is retrieved by tomorrow's Analyze stage for novelty comparison.

### 4.7 Emailer (`emailer.py`)
- Sends emails via Gmail SMTP (smtp.gmail.com:587, STARTTLS) using App Password.
- On pipeline success:
  - 1 email: Daily Brief (subject: "AI Daily Brief — {date}").
  - N emails (1 per theme): Subject "AI Theme: {theme title} — {date}", body contains English summary + English script + German script, clearly sectioned.
- On pipeline failure:
  - 1 email: Failure alert with stage name, error message, traceback, and last 100 lines of log output.
- All emails are plain text with clear delimiter sections between deliverables.

### 4.8 LLM Client (`llm.py`)
- Thin wrapper around OpenRouter's chat completions API.
- Accepts: model ID, temperature, system prompt, user prompt.
- Returns: raw text response.
- Handles: API errors, rate limits (exponential backoff up to 60s), token counting for logging.
- Model assignments read from config file.

### 4.9 Database Layer (`db.py`)
- All SQLite operations centralized here.
- Schema initialization (idempotent).
- CRUD operations for all tables.
- Connection management via context manager (single WAL-mode connection per pipeline run).

---

## 5. Data Model

```sql
CREATE TABLE pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT    NOT NULL,           -- ISO 8601 date (Europe/Berlin)
    started_at      TEXT    NOT NULL,           -- ISO 8601 datetime
    completed_at    TEXT,                       -- ISO 8601 datetime, NULL if failed
    status          TEXT    NOT NULL DEFAULT 'running',  -- running, completed, failed
    current_stage   TEXT,                       -- scrape, analyze, generate_evaluate, brief, email
    error_message   TEXT
);

CREATE TABLE feeds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    category        TEXT    NOT NULL            -- 'news' or 'commentator'
);

CREATE TABLE articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id         INTEGER NOT NULL REFERENCES feeds(id),
    url             TEXT    NOT NULL UNIQUE,
    title           TEXT    NOT NULL,
    author          TEXT,
    published_at    TEXT    NOT NULL,           -- ISO 8601 datetime from RSS
    scraped_at      TEXT    NOT NULL,           -- ISO 8601 datetime
    rss_excerpt     TEXT,
    full_content    TEXT,
    content_status  TEXT    NOT NULL DEFAULT 'full',  -- full, excerpt_paywall, excerpt_only
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id)
);

CREATE TABLE themes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    source_article_ids TEXT NOT NULL,           -- JSON array of article IDs
    novelty_type    TEXT    NOT NULL,           -- 'novel' or 'continuation'
    order_index     INTEGER NOT NULL,           -- 1-based ordering
    status          TEXT    NOT NULL DEFAULT 'pending'  -- pending, approved, auto_approved
);

CREATE TABLE deliverables (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id        INTEGER NOT NULL REFERENCES themes(id),
    deliverable_type TEXT   NOT NULL,           -- 'summary_en', 'script_en', 'script_de'
    content         TEXT    NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL
);

CREATE TABLE evaluation_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id        INTEGER NOT NULL REFERENCES themes(id),
    round_number    INTEGER NOT NULL DEFAULT 1,
    quality_passed  TEXT,                       -- 'pass' or 'fail' (NULL if not yet run)
    quality_feedback TEXT,
    adversarial_passed TEXT,                    -- 'pass' or 'fail'
    adversarial_feedback TEXT,
    overall_passed  TEXT    NOT NULL,           -- 'pass' or 'fail'
    evaluated_at    TEXT    NOT NULL
);

CREATE TABLE daily_briefs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    content         TEXT    NOT NULL,
    word_count      INTEGER NOT NULL,
    created_at      TEXT    NOT NULL
);
```

**Key design decisions:**
- `source_article_ids` stored as JSON array rather than junction table — [COMPLEXITY JUSTIFICATION: simpler for a single-user batch system; no cross-querying needed].
- SQLite WAL mode enabled for safe concurrent reads (e.g., inspecting DB while pipeline runs).
- `deliverables` keeps all versions for auditability; the application selects the latest version per type per theme.

---

## 6. API & Interface Design

This system has no external API. All interfaces are internal between components.

### 6.1 LLM Client Interface

```
llm.complete(model_id: str, temperature: float, system_prompt: str, user_prompt: str) -> str
```

### 6.2 Stage Interface

Each stage module exposes a single function:

```
scraper.run(pipeline_run_id: int, db: Database, config: Config) -> None
analyzer.run(pipeline_run_id: int, db: Database, config: Config) -> None
generator.run(pipeline_run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None
evaluator.run(pipeline_run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None
brief.run(pipeline_run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None
emailer.run(pipeline_run_id: int, db: Database, config: Config) -> None
```

`generator.run` and `evaluator.run` are called as a paired unit per theme within the orchestrator's generate-evaluate loop.

### 6.3 Config Interface

```
config = Config.from_yaml(path: str) -> Config
config.feeds.news -> list[FeedDef]
config.feeds.commentators -> list[FeedDef]
config.models.strong -> ModelDef  # id, temperature
config.models.weak -> ModelDef    # id, temperature
config.pipeline.schedule -> str    # "04:00"
config.pipeline.timezone -> str   # "Europe/Berlin"
config.pipeline.max_retries -> int
config.pipeline.max_refinement_rounds -> int
config.email.recipient -> str
config.email.sender -> str
config.email.smtp_host -> str
config.email.smtp_port -> int
config.email.smtp_user -> str
config.email.smtp_password_env -> str  # env var name holding app password
config.db.path -> str
config.openrouter.api_key_env -> str   # env var name holding API key
```

### 6.4 YAML Config File Schema

```yaml
feeds:
  news:
    - name: "Feed Name"
      url: "https://example.com/rss"
  commentators:
    - name: "Commentator Name"
      url: "https://example.com/atom"

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

---

## 7. Security Architecture

| Concern | Measure |
|---------|---------|
| API key storage | OpenRouter API key and Gmail App Password stored in environment variables, never in config file or code. Env var names referenced in config. |
| Config file | Contains no secrets. World-readable permissions acceptable. |
| SQLite database | File permissions 0600, owned by pipeline service user. |
| Network egress | Only to OpenRouter API (HTTPS), RSS feed URLs (HTTPS preferred), and Gmail SMTP (STARTTLS). No ingress ports required. |
| Service user | Pipeline runs as a dedicated system user (`ai-news-pipeline`), not root. |
| Dependency audit | `pip audit` run as part of CI/install. Only well-maintained packages used. |

---

## 8. Infrastructure & Deployment

### 8.1 Server Requirements
- Ubuntu 24.04 LTS VPS
- Python 3.12+
- ~2 GB disk for database growth (conservative estimate: ~5 MB/day)
- Outbound internet access

### 8.2 Directory Layout

```
/opt/ai-news-pipeline/
├── config/
│   └── feeds.yaml
├── data/
│   └── pipeline.db
├── logs/
│   └── pipeline.log
├── src/
│   ├── main.py
│   ├── scraper.py
│   ├── analyzer.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── brief.py
│   ├── emailer.py
│   ├── db.py
│   ├── llm.py
│   └── models.py
├── prompts/
│   ├── analyze.txt
│   ├── summary_en.txt
│   ├── script_en.txt
│   ├── script_de.txt
│   ├── evaluate_quality.txt
│   ├── evaluate_adversarial.txt
│   ├── refine.txt
│   └── brief.txt
├── venv/
├── requirements.txt
└── deploy/
    ├── ai-news-pipeline.service
    └── ai-news-pipeline.timer
```

### 8.3 systemd Units

**Service unit** (`ai-news-pipeline.service`):
- `Type=oneshot`
- `ExecStart=/opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py`
- `EnvironmentFile=/opt/ai-news-pipeline/.env` (contains `OPENROUTER_API_KEY` and `GMAIL_APP_PASSWORD`)
- `User=ai-news-pipeline`
- `StandardOutput=journal`
- `StandardError=journal`

**Timer unit** (`ai-news-pipeline.timer`):
- `OnCalendar=*-*-* 04:00 Europe/Berlin`
- `Persistent=true` (catches up if server was down at trigger time)

### 8.4 Installation Steps
1. Create service user.
2. Clone/copy project to `/opt/ai-news-pipeline/`.
3. Create `.env` file with API keys.
4. Create and populate `config/feeds.yaml`.
5. Create Python venv, install requirements.
6. Run `python src/main.py --init-db` to initialize SQLite schema.
7. Copy systemd units, `systemctl daemon-reload`, `systemctl enable --now ai-news-pipeline.timer`.

---

## 9. Operational Model

### 9.1 Logging
- Structured logs (JSON format) written to `/opt/ai-news-pipeline/logs/pipeline.log` and stdout (journald).
- Log rotation via `logrotate` (daily, keep 30 days).
- Each log entry includes: timestamp, pipeline_run_id, stage, theme_id (if applicable), level, message.
- LLM call logging: model, prompt token count, response token count, latency.

### 9.2 Monitoring
- No external monitoring infrastructure. Reliance on:
  - systemd journal for process-level monitoring (`systemctl status`).
  - Failure alert emails for operator notification.
  - Log file for manual inspection.

### 9.3 Manual Operations
- **Add/remove feeds:** Edit `config/feeds.yaml`, no restart needed (config read at pipeline start).
- **Re-run manually:** `systemctl start ai-news-pipeline.service`.
- **Inspect database:** `sqlite3 /opt/ai-news-pipeline/data/pipeline.db`.
- **View logs:** `journalctl -u ai-news-pipeline.service` or `tail -f /opt/ai-news-pipeline/logs/pipeline.log`.

### 9.4 Data Retention
- No automated pruning. [ASSUMPTION: Operator will manage disk manually. At ~5 MB/day, disk is not an immediate concern.]
- All article content, deliverable versions, and evaluation rounds retained indefinitely for auditability.

---

## 10. Key Design Decisions

| Decision | Rationale | Traceability |
|----------|-----------|--------------|
| Sequential stage execution | Simpler error handling, no rate-limit contention, no latency SLA | [COMPLEXITY JUSTIFICATION] |
| Sequential theme processing within Generate+Evaluate | Avoids OpenRouter rate limits; simpler state management | [COMPLEXITY JUSTIFICATION] |
| German script generated from source + summary, NOT from English script | Enforces native generation per operator requirement | R7 |
| Evaluator uses weak model | Evaluation is criteria-matching, not creative work; cost-efficient; upgradable via config | [PROPOSED DESIGN DECISION] |
| systemd timer over cron | Native Ubuntu integration, `Persistent=true` for catch-up, better logging | [PROPOSED DESIGN DECISION] |
| YAML config format | Human-editable, supports comments, standard in Python ecosystem | Confirmed answer, Round 2 |
| `trafilatura` for article extraction | Best-in-class open-source extractor, handles metadata, graceful fallback | [PROPOSED DESIGN DECISION] |
| SQLite WAL mode | Allows concurrent reads during pipeline execution | [PROPOSED DESIGN DECISION] |
| All deliverable versions retained | Audit trail; enables debugging quality issues | [PROPOSED DESIGN DECISION] |
| Environment variables for secrets | Standard 12-factor practice; no secrets in config or code | Security requirement |
| 30-second retry backoff | Reasonable wait for transient API/network issues without excessive delay | [PROPOSED DESIGN DECISION] |
| Auto-approve after 3 refinement rounds | Prevents infinite loops; diminishing returns on further refinement | Confirmed answer, Round 1 |

---

## 11. Open Questions & Assumptions

| # | Item | Status | Type |
|---|------|--------|------|
| 1 | Gmail sender address assumed same as recipient (recipient@gmail.com) | [ASSUMPTION] | If sender is different, config supports it |
| 2 | Model names `deepseek/deepseek-v4-pro` and `deepseek/deepseek-v4-flash` are valid OpenRouter identifiers | [ASSUMPTION] | Verify at implementation time; config allows change |
| 3 | `trafilatura` can extract content from most feed source domains | [ASSUMPTION] | Paywall/JS-heavy sites will gracefully fall back to excerpt |
| 4 | OpenRouter rate limits are sufficient for sequential nightly calls (~15–25 calls per run) | [ASSUMPTION] | If hit, backoff logic in LLM client handles it |
| 5 | No need for automated database pruning | [ASSUMPTION] | Disk growth is modest; operator manages manually |
| 6 | Email content is plain text with section delimiters (not HTML or attachments) | [PROPOSED DESIGN DECISION] | Simpler; no rendering issues. If HTML preferred, emailer module can be updated. |
| 7 | First pipeline run has no previous daily brief, so all themes are treated as novel | [ASSUMPTION] | Analyzer handles missing brief gracefully |
| 8 | RSS feeds return `published_at` timestamps in a parseable format | [ASSUMPTION] | `feedparser` normalizes most date formats |

---
