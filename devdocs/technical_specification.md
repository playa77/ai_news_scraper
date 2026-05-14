# Technical Specification

## 1. Implementation Scope

### In Scope
- Complete pipeline: scrape → analyze → generate+evaluate → brief → email.
- SQLite schema initialization and all data access operations.
- LLM integration via OpenRouter API with retry and backoff.
- Full-article extraction with paywall fallback.
- Evaluation + adversarial review with up to 3 refinement rounds.
- Gmail SMTP email delivery (success emails + failure alerts).
- systemd service and timer units.
- Structured JSON logging to file and stdout.
- Unit and integration test suites.

### Out of Scope
- Web UI, REST API, or any inbound interface.
- Multi-user support or authentication system.
- Audio/video generation from scripts.
- Feed discovery or auto-subscription.
- Database pruning or archival automation.
- CI/CD pipeline (manual deployment specified).

---

## 2. Project Structure

```
/opt/ai-news-pipeline/
├── config/
│   └── feeds.yaml                    # Feed list and all configuration
├── data/
│   └── pipeline.db                   # SQLite database (created at runtime)
├── logs/
│   └── pipeline.log                  # Rotated log file (created at runtime)
├── src/
│   ├── __init__.py
│   ├── main.py                       # Orchestrator entry point
│   ├── config.py                     # Config loading and validation
│   ├── db.py                         # Database layer
│   ├── llm.py                        # OpenRouter LLM client
│   ├── scraper.py                    # RSS scraping + article extraction
│   ├── analyzer.py                   # Theme identification
│   ├── generator.py                  # Deliverable generation
│   ├── evaluator.py                  # Quality + adversarial evaluation
│   ├── brief.py                      # Daily brief generation
│   ├── emailer.py                    # Email dispatch
│   └── models.py                     # Dataclass/type definitions
├── prompts/
│   ├── analyze.txt                   # Analyzer system + user prompt template
│   ├── summary_en.txt                # English summary prompt template
│   ├── script_en.txt                 # English script prompt template
│   ├── script_de.txt                 # German script prompt template
│   ├── evaluate_quality.txt          # Quality evaluator prompt template
│   ├── evaluate_adversarial.txt      # Adversarial evaluator prompt template
│   ├── refine.txt                    # Refinement prompt template
│   └── brief.txt                     # Daily brief prompt template
├── tests/
│   ├── __init__.py
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
│       ├── sample_rss.xml
│       ├── sample_articles.json
│       └── sample_themes.json
├── deploy/
│   ├── ai-news-pipeline.service      # systemd service unit
│   ├── ai-news-pipeline.timer        # systemd timer unit
│   └── logrotate.conf                # Log rotation configuration
├── .env.example                      # Template for environment variables
├── requirements.txt                  # Python dependencies
└── README.md                         # Operator manual
```

---

## 3. Dependencies

### requirements.txt

```
feedparser==6.0.11
trafilatura==1.12.0
httpx==0.27.2
pyyaml==6.0.2
python-dotenv==1.0.1
pydantic==2.9.2
```

### Justification

| Package | Purpose | Alternative Considered |
|---------|---------|----------------------|
| `feedparser` | RSS/Atom feed parsing — industry standard, handles malformed feeds | `rss-parser` (less mature) |
| `trafilatura` | Full-article HTML extraction — best recall/precision in benchmarks | `newspaper3k` (unmaintained), `readability-lxml` (lower quality) |
| `httpx` | HTTP client for OpenRouter API and article fetching — async-capable, timeout control | `requests` (sync only), `aiohttp` (overkill) |
| `pyyaml` | YAML config parsing | Built-in `json` (no comments support) |
| `python-dotenv` | Load `.env` file for local development | Manual `os.environ` (error-prone) |
| `pydantic` | Config validation and dataclass definitions — type safety, clear error messages | `dataclasses` (no validation), `attrs` (less ecosystem) |

### Standard Library Usage (no install needed)
- `sqlite3` — database access
- `smtplib` + `email.mime` — email delivery
- `logging` + `json` — structured logging
- `datetime`, `zoneinfo` — time handling
- `pathlib` — file paths
- `re` — URL normalization
- `traceback` — error formatting for alert emails

### No OpenRouter SDK
The OpenRouter API is OpenAI-compatible. Direct `httpx` calls to `/chat/completions` are simpler than adding an SDK dependency. [COMPLEXITY JUSTIFICATION: avoids SDK version lock-in; the API surface is one endpoint.]

---

## 4. Configuration

### 4.1 YAML Config File (`config/feeds.yaml`)

Complete schema with all fields:

```yaml
feeds:
  news:
    - name: "Ars Technica AI"
      url: "https://feeds.arstechnica.com/arstechnica/technology-lab"
    - name: "MIT Tech Review AI"
      url: "https://www.technologyreview.com/feed/"
  commentators:
    - name: "Simon Willison"
      url: "https://simonwillison.net/atom/everything/"
    - name: "Jack Clark"
      url: "https://jack-clark.com/feed/"

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
  retry_backoff_seconds: 30
  article_fetch_timeout_seconds: 15
  llm_request_timeout_seconds: 120

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

### 4.2 Environment Variables (`.env` file, loaded by `python-dotenv`)

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

### 4.3 Config Loading (`src/config.py`)

```python
# Pseudocode — type signatures only

class ModelDef:
    id: str
    temperature: float

class FeedDef:
    name: str
    url: str

class FeedsConfig:
    news: list[FeedDef]
    commentators: list[FeedDef]

class PipelineConfig:
    schedule: str
    timezone: str
    max_retries: int
    max_refinement_rounds: int
    retry_backoff_seconds: int
    article_fetch_timeout_seconds: int
    llm_request_timeout_seconds: int

class EmailConfig:
    recipient: str
    sender: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password_env: str
    # Resolved at load time from os.environ[smtp_password_env]

class DatabaseConfig:
    path: str

class OpenRouterConfig:
    api_key_env: str
    base_url: str
    # Resolved at load time from os.environ[api_key_env]

class Config:
    feeds: FeedsConfig
    models: ModelsConfig  # contains strong: ModelDef, weak: ModelDef
    pipeline: PipelineConfig
    email: EmailConfig
    database: DatabaseConfig
    openrouter: OpenRouterConfig

    @staticmethod
    def from_yaml(path: str) -> Config:
        # Load YAML, validate with pydantic, resolve env vars
        # Raises ConfigError on missing file, invalid YAML, or missing env vars
```

Validation rules enforced by pydantic:
- `feeds.news` + `feeds.commentators` must contain ≥1 feed each.
- Feed URLs must be valid HTTP/HTTPS URLs.
- `models.strong.temperature` and `models.weak.temperature` must be in [0.0, 2.0].
- `models.strong.id` and `models.weak.id` must be non-empty strings.
- `pipeline.max_retries` must be ≥0.
- `pipeline.max_refinement_rounds` must be ≥1.
- `email.smtp_port` must be 1–65535.
- `database.path` must be a valid file path (parent directory must exist or be creatable).
- Environment variables referenced by `smtp_password_env` and `api_key_env` must be set; `ConfigError` raised if missing.

---

## 5. Data Layer

### 5.1 Schema DDL

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    completed_at    TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',
    current_stage   TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS feeds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    category        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id         INTEGER NOT NULL REFERENCES feeds(id),
    url             TEXT    NOT NULL UNIQUE,
    title           TEXT    NOT NULL,
    author          TEXT,
    published_at    TEXT    NOT NULL,
    scraped_at      TEXT    NOT NULL,
    rss_excerpt     TEXT,
    full_content    TEXT,
    content_status  TEXT    NOT NULL DEFAULT 'full',
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS themes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    source_article_ids TEXT NOT NULL,
    novelty_type    TEXT    NOT NULL,
    order_index     INTEGER NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS deliverables (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id        INTEGER NOT NULL REFERENCES themes(id),
    deliverable_type TEXT   NOT NULL,
    content         TEXT    NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id        INTEGER NOT NULL REFERENCES themes(id),
    round_number    INTEGER NOT NULL DEFAULT 1,
    quality_passed  TEXT,
    quality_feedback TEXT,
    adversarial_passed TEXT,
    adversarial_feedback TEXT,
    overall_passed  TEXT    NOT NULL,
    evaluated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_briefs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    content         TEXT    NOT NULL,
    word_count      INTEGER NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_pipeline_run ON articles(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_themes_pipeline_run ON themes(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_deliverables_theme_type ON deliverables(theme_id, deliverable_type);
CREATE INDEX IF NOT EXISTS idx_evaluation_rounds_theme ON evaluation_rounds(theme_id);
CREATE INDEX IF NOT EXISTS idx_daily_briefs_pipeline_run ON daily_briefs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date ON pipeline_runs(run_date);
```

### 5.2 Database Module Interface (`src/db.py`)

```python
class Database:
    def __init__(self, db_path: str) -> None:
        # Open connection, enable WAL, set pragmas

    def initialize_schema(self) -> None:
        # Execute all CREATE TABLE/INDEX IF NOT EXISTS statements

    def create_pipeline_run(self, run_date: str, started_at: str) -> int:
        # INSERT into pipeline_runs, return id

    def update_pipeline_run(self, run_id: int, status: str, completed_at: str = None,
                           current_stage: str = None, error_message: str = None) -> None:
        # UPDATE pipeline_runs

    def get_last_successful_run(self) -> Optional[dict]:
        # SELECT from pipeline_runs WHERE status='completed' ORDER BY id DESC LIMIT 1

    def upsert_feed(self, url: str, name: str, category: str) -> int:
        # INSERT OR IGNORE into feeds, return id

    def article_exists(self, url: str) -> bool:
        # SELECT 1 FROM articles WHERE url = ?

    def insert_article(self, feed_id: int, url: str, title: str, author: str,
                       published_at: str, scraped_at: str, rss_excerpt: str,
                       full_content: str, content_status: str,
                       pipeline_run_id: int) -> int:
        # INSERT into articles, return id

    def get_articles_for_run(self, pipeline_run_id: int) -> list[dict]:
        # SELECT * FROM articles WHERE pipeline_run_id = ?

    def insert_theme(self, pipeline_run_id: int, title: str, description: str,
                     source_article_ids: list[int], novelty_type: str,
                     order_index: int) -> int:
        # INSERT into themes, return id

    def get_themes_for_run(self, pipeline_run_id: int) -> list[dict]:
        # SELECT * FROM themes WHERE pipeline_run_id = ? ORDER BY order_index

    def update_theme_status(self, theme_id: int, status: str) -> None:
        # UPDATE themes SET status = ? WHERE id = ?

    def insert_deliverable(self, theme_id: int, deliverable_type: str,
                           content: str, version: int) -> int:
        # INSERT into deliverables, return id

    def get_latest_deliverables(self, theme_id: int) -> dict[str, dict]:
        # Returns {deliverable_type: {content, version}} for the highest version per type

    def get_deliverable_history(self, theme_id: int, deliverable_type: str) -> list[dict]:
        # Returns all versions ordered by version ASC

    def insert_evaluation_round(self, theme_id: int, round_number: int,
                                quality_passed: str, quality_feedback: str,
                                adversarial_passed: str, adversarial_feedback: str,
                                overall_passed: str) -> int:
        # INSERT into evaluation_rounds, return id

    def get_latest_evaluation(self, theme_id: int) -> Optional[dict]:
        # SELECT * FROM evaluation_rounds WHERE theme_id = ? ORDER BY round_number DESC LIMIT 1

    def insert_daily_brief(self, pipeline_run_id: int, content: str,
                           word_count: int) -> int:
        # INSERT into daily_briefs, return id

    def get_previous_daily_brief(self, current_run_date: str) -> Optional[dict]:
        # SELECT * FROM daily_briefs JOIN pipeline_runs
        # WHERE pipeline_runs.run_date < ? AND pipeline_runs.status = 'completed'
        # ORDER BY pipeline_runs.run_date DESC LIMIT 1

    def close(self) -> None:
        # Close connection
```

### 5.3 Data Flow

```
Scrape Stage:
  feeds.yaml → feeds table (upsert)
  RSS items → articles table (insert if url not exists)

Analyze Stage:
  articles[pipeline_run_id] + daily_briefs[previous] → themes table

Generate+Evaluate Stage (per theme):
  themes[id] + articles[source_article_ids] → deliverables table (version 1)
  deliverables → evaluation_rounds table
  If refinement: deliverables table (version N+1) → evaluation_rounds table

Brief Stage:
  themes[approved] + deliverables[latest] → daily_briefs table

Email Stage:
  daily_briefs + themes + deliverables → email dispatch (no DB write)
```

---

## 6. API Specification

### 6.1 OpenRouter Chat Completions

**Endpoint:** `POST {base_url}/chat/completions`

**Headers:**
```
Authorization: Bearer {api_key}
Content-Type: application/json
HTTP-Referer: https://github.com/ai-news-pipeline
X-Title: AI News Pipeline
```

**Request Body:**
```json
{
  "model": "deepseek/deepseek-v4-pro",
  "temperature": 0.7,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response (success):**
```json
{
  "id": "...",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  }
}
```

**Error handling:**
- HTTP 429 (rate limit): Retry with exponential backoff (base 10s, max 60s, up to 5 retries within the stage attempt).
- HTTP 5xx (server error): Retry with exponential backoff (base 5s, max 60s, up to 3 retries within the stage attempt).
- HTTP 4xx (client error, non-429): Do not retry. Log error, raise `LLMClientError`.
- Network timeout (configurable, default 120s): Retry up to 3 times within the stage attempt.
- Non-JSON or missing `choices[0].message.content`: Raise `LLMClientError`.

### 6.2 LLM Client Module (`src/llm.py`)

```python
class LLMClientError(Exception):
    pass

class LLMClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 120) -> None:
        # Initialize httpx.Client with base_url, default headers, timeout

    def complete(self, model_id: str, temperature: float,
                 system_prompt: str, user_prompt: str) -> str:
        # Build request, send, handle retries for 429/5xx/timeout
        # Log model, token counts, latency
        # Return choices[0].message.content
        # Raise LLMClientError on unrecoverable failure

    def close(self) -> None:
        # Close httpx.Client
```

---

## 7. AuthN/AuthZ

This system has no inbound authentication or authorization requirements. It is a single-user batch pipeline with no API surface.

**Secrets access control:**
- `OPENROUTER_API_KEY` and `GMAIL_APP_PASSWORD` are read from environment variables at process start.
- The `.env` file is owned by the `ai-news-pipeline` user with `0600` permissions.
- The systemd service loads the `.env` file via `EnvironmentFile=` directive.
- Secrets are never logged, never written to the database, and never included in email content.

---

## 8. Module Specifications

### 8.1 Orchestrator (`src/main.py`)

```python
def main() -> None:
    # 1. Parse CLI args (--init-db flag, --config path)
    # 2. Load config from YAML
    # 3. Initialize database (if --init-db) or connect
    # 4. Create pipeline_runs record
    # 5. Execute stages sequentially with retry wrapper:
    #    a. scraper.run()
    #    b. analyzer.run()
    #    c. For each theme: generator.run() then evaluator.run()
    #    d. brief.run()
    #    e. emailer.run()
    # 6. Update pipeline_runs status to 'completed'
    # 7. On any stage failure after retries:
    #    - Update pipeline_runs status to 'failed', error_message
    #    - Send failure alert email
    #    - Exit with code 1

def retry_wrapper(stage_name: str, stage_fn: Callable, max_retries: int,
                  backoff: int, run_id: int, db: Database, config: Config,
                  **kwargs) -> None:
    # Execute stage_fn
    # On exception: log, sleep(backoff * attempt), retry
    # After max_retries: raise StageFailedError(stage_name, last_exception)
```

### 8.2 Scraper (`src/scraper.py`)

```python
def run(run_id: int, db: Database, config: Config) -> None:
    # 1. Update pipeline_runs.current_stage = 'scrape'
    # 2. For each feed in config.feeds.news + config.feeds.commentators:
    #    a. db.upsert_feed(url, name, category)
    #    b. Fetch RSS via feedparser.parse(url)
    #    c. Get last successful run timestamp from db.get_last_successful_run()
    #    d. Filter entries where published_at > last_run_timestamp
    #       (If no previous run, take all entries from last 24 hours)
    #    e. For each new entry:
    #       i.   Normalize URL (strip query params for dedup: re.sub(r'\?.*$', '', url))
    #       ii.  Check db.article_exists(normalized_url) — skip if exists
    #       iii. Store rss_excerpt from entry.summary
    #       iv.  Attempt full extraction:
    #            - Fetch HTML via httpx.get(entry.link, timeout=config.pipeline.article_fetch_timeout_seconds)
    #            - Extract via trafilatura.extract(html, include_comments=False, favor_precision=True)
    #            - If result is non-empty and > 200 chars: content_status = 'full'
    #            - Else if result is non-empty but ≤ 200 chars: content_status = 'excerpt_only', full_content = rss_excerpt
    #            - Else (None/empty): content_status = 'excerpt_only', full_content = rss_excerpt
    #       v.  If httpx.HTTPStatusError (402/403): content_status = 'excerpt_paywall', full_content = rss_excerpt
    #       vi. db.insert_article(...)
    # 3. Log: total feeds, total new articles, count per content_status
```

**URL normalization rule:** Strip fragment (`#...`) and tracking query parameters. Keep semantic query params. Simplified approach: strip all query params. [PROPOSED DESIGN DECISION: Over-deduplication is preferable to duplicate articles. If two URLs with different query params point to the same article, we want one record.]

### 8.3 Analyzer (`src/analyzer.py`)

```python
def run(run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None:
    # 1. Update pipeline_runs.current_stage = 'analyze'
    # 2. Fetch articles for this run: db.get_articles_for_run(run_id)
    # 3. Fetch previous daily brief: db.get_previous_daily_brief(run_date)
    # 4. Build user prompt:
    #    - If previous brief exists: include it with instruction to identify novel themes
    #      and meaningful updates to existing themes
    #    - If no previous brief: include instruction to identify the most significant themes
    #    - Include all article titles, sources, and full_content (or rss_excerpt if no full_content)
    # 5. Call llm_client.complete(strong model, analyze prompt)
    # 6. Parse LLM response as JSON array of themes:
    #    Expected structure:
    #    [
    #      {
    #        "title": "...",
    #        "description": "...",
    #        "novelty_type": "novel" | "continuation",
    #        "source_article_indices": [0, 3, 7]
    #      }
    #    ]
    # 7. Validate: 1–5 themes, each has required fields, indices are valid
    # 8. Map source_article_indices back to article IDs
    # 9. For each theme: db.insert_theme(...)
    # 10. Log: number of themes, novelty_type distribution
```

**Prompt template structure** (`prompts/analyze.txt`):
- System prompt: Role definition (expert AI news analyst), output format specification (JSON array), constraints (1–5 themes, classify as novel or continuation, reference source articles by index).
- User prompt: Previous daily brief (if exists), article list with index, title, source, and content.

**Parsing robustness:** If LLM returns markdown-wrapped JSON (```json ... ```), strip the wrapping. If JSON parsing fails, log the raw response and raise `AnalysisParseError` (triggers stage retry).

### 8.4 Generator (`src/generator.py`)

```python
def run(run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None:
    # Called once per theme by the orchestrator
    # 1. Get theme details from db
    # 2. Get source articles by IDs from theme.source_article_ids
    # 3. Generate three deliverables sequentially:
    #    a. summary_en: llm_client.complete(strong model, summary_en prompt, articles content)
    #    b. script_en: llm_client.complete(strong model, script_en prompt, articles + summary_en)
    #    c. script_de: llm_client.complete(strong model, script_de prompt, articles + summary_en)
    #       NOTE: script_de does NOT receive script_en as input
    # 4. For each deliverable: db.insert_deliverable(theme_id, type, content, version=1)
    # 5. Log: theme title, deliverable types generated, word counts

def refine(run_id: int, db: Database, config: Config, llm_client: LLMClient,
           theme_id: int, evaluation_feedback: str) -> None:
    # 1. Get current deliverables for theme (latest version)
    # 2. Get source articles
    # 3. For each deliverable type:
    #    llm_client.complete(strong model, refine prompt,
    #                        current_content + evaluation_feedback + source_articles)
    # 4. db.insert_deliverable(theme_id, type, new_content, version=prev_version+1)
    # 5. Log: theme title, refinement round, new version numbers
```

**Prompt template structures:**

`prompts/summary_en.txt`:
- System: Role (technical writer), output constraints (~750 words, English, factual, structured with headings).
- User: Theme description, source article texts.

`prompts/script_en.txt`:
- System: Role (YouTube scriptwriter in Wes Roth / Andrej Karpathy style), style guide (conversational, enthusiastic, uses analogies, explains technical concepts accessibly, hook opening, clear structure), output constraints (~1000–1500 words, English).
- User: Theme description, source article texts, English summary.

`prompts/script_de.txt`:
- System: Role (German YouTube scriptwriter, same style as English but natively German), style guide (same principles, German idioms and phrasing, NOT a translation), output constraints (~1000–1500 words, German).
- User: Theme description, source article texts, English summary. Explicitly: "Schreibe das Skript auf Deutsch. Es muss ein nativ geschriebenes Skript sein, keine Übersetzung des englischen Skripts."

`prompts/refine.txt`:
- System: Role (revising scriptwriter), instruction to address specific feedback while maintaining original style and constraints.
- User: Current deliverable content, evaluation feedback, source articles.

### 8.5 Evaluator (`src/evaluator.py`)

```python
def run(run_id: int, db: Database, config: Config, llm_client: LLMClient,
        theme_id: int) -> str:
    # Returns: 'approved' or 'needs_refinement'
    # 1. Get latest deliverables for theme
    # 2. Get source articles for theme
    # 3. Get current refinement round number from db.get_latest_evaluation(theme_id)
    # 4. Quality evaluation:
    #    llm_client.complete(weak model, evaluate_quality prompt,
    #                        deliverables + source_articles)
    #    Parse response as JSON: {"summary_en": {"pass": bool, "feedback": "..."}, ...}
    # 5. Adversarial evaluation:
    #    llm_client.complete(weak model, evaluate_adversarial prompt,
    #                        deliverables + source_articles)
    #    Parse response as JSON: {"pass": bool, "feedback": "...", "issues": [...]}
    # 6. Determine overall: pass if BOTH evaluators pass ALL deliverables
    # 7. db.insert_evaluation_round(theme_id, round_number, quality results,
    #                               adversarial results, overall)
    # 8. If overall_passed == 'pass':
    #       db.update_theme_status(theme_id, 'approved')
    #       return 'approved'
    #    Else if round_number >= config.pipeline.max_refinement_rounds:
    #       db.update_theme_status(theme_id, 'auto_approved')
    #       return 'approved'  # treat as approved for downstream
    #    Else:
    #       return 'needs_refinement'
```

**Prompt template structures:**

`prompts/evaluate_quality.txt`:
- System: Role (senior editor), evaluation rubric (style adherence, word count compliance, completeness, prose quality, structural coherence), output format (JSON per deliverable type with pass/fail and specific feedback).
- User: All three deliverables, theme description.

`prompts/evaluate_adversarial.txt`:
- System: Role (fact-checker and bias analyst), evaluation rubric (factual accuracy against sources, unsupported claims, hallucinations, ideological bias, omissions of key information), output format (JSON with pass/fail, feedback, and list of specific issues with references).
- User: All three deliverables, source article texts.

**Parsing robustness:** Same JSON extraction logic as analyzer. If parsing fails, treat as `needs_refinement` with feedback "Evaluation response could not be parsed; please review."

### 8.6 Brief Generator (`src/brief.py`)

```python
def run(run_id: int, db: Database, config: Config, llm_client: LLMClient) -> None:
    # 1. Update pipeline_runs.current_stage = 'brief'
    # 2. Get all approved/auto_approved themes for this run
    # 3. For each theme, get latest summary_en deliverable
    # 4. Call llm_client.complete(strong model, brief prompt, all summaries)
    # 5. Compute word count
    # 6. db.insert_daily_brief(run_id, content, word_count)
    # 7. Log: brief word count, number of themes included
```

`prompts/brief.txt`:
- System: Role (executive briefing writer), output constraints (~700 words, English, concise overview of each theme with key takeaways, professional tone).
- User: All approved theme summaries with titles.

### 8.7 Emailer (`src/emailer.py`)

```python
def run(run_id: int, db: Database, config: Config) -> None:
    # 1. Update pipeline_runs.current_stage = 'email'
    # 2. Get daily brief for this run
    # 3. Send brief email:
    #    Subject: "AI Daily Brief — {run_date}"
    #    Body: brief content
    # 4. For each approved/auto_approved theme:
    #    a. Get latest deliverables (summary_en, script_en, script_de)
    #    b. Send theme email:
    #       Subject: "AI Theme: {theme_title} — {run_date}"
    #       Body:
    #         "=== ENGLISH SUMMARY ===\n{summary_en}\n\n
    #          === ENGLISH SCRIPT ===\n{script_en}\n\n
    #          === GERMAN SCRIPT ===\n{script_de}"
    # 5. Log: number of emails sent

def send_failure_alert(config: Config, stage_name: str, error_message: str,
                       traceback_str: str, log_tail: str) -> None:
    # Send single email:
    # Subject: "AI Pipeline FAILURE — {stage_name} — {date}"
    # Body:
    #   "Stage: {stage_name}\n
    #    Error: {error_message}\n\n
    #    Traceback:\n{traceback_str}\n\n
    #    Recent logs:\n{log_tail}"

def _send_email(config: Config, subject: str, body: str) -> None:
    # Internal: connect to smtp.gmail.com:587, STARTTLS, login with app password,
    # send MIME text/plain message from sender to recipient, quit.
    # On SMTP error: log and raise EmailError
```

**Log tail retrieval:** The orchestrator reads the last 100 lines from the current log file before calling `send_failure_alert`.

---

## 9. Background Jobs

### 9.1 Scheduled Job

| Attribute | Value |
|-----------|-------|
| Name | `ai-news-pipeline.timer` |
| Schedule | `04:00 Europe/Berlin` daily |
| Type | systemd timer → oneshot service |
| Catch-up | Yes (`Persistent=true`) — if server was down at 04:00, runs on next boot |
| Timeout | `TimeoutStartSec=1800` (30 minutes) — if pipeline exceeds this, systemd kills it |
| On failure | systemd logs the failure; pipeline sends alert email before exiting |

### 9.2 No Other Background Jobs

No cron jobs, no daemons, no watchers. The pipeline is purely triggered by the systemd timer or manual invocation.

---

## 10. Observability

### 10.1 Structured Logging

Every log entry is a JSON object written to both stdout and the log file:

```json
{
  "timestamp": "2025-01-15T04:00:01.234+01:00",
  "level": "INFO",
  "pipeline_run_id": 42,
  "stage": "scrape",
  "theme_id": null,
  "message": "Fetched 15 new articles from feed 'Ars Technica AI'",
  "extra": {
    "feed_name": "Ars Technica AI",
    "article_count": 15,
    "content_status_counts": {"full": 12, "excerpt_paywall": 2, "excerpt_only": 1}
  }
}
```

### 10.2 Key Log Events

| Stage | Event | Level |
|-------|-------|-------|
| Orchestrator | Pipeline started | INFO |
| Orchestrator | Stage started | INFO |
| Orchestrator | Stage completed | INFO |
| Orchestrator | Stage failed (attempt N) | WARNING |
| Orchestrator | Stage failed (all retries) | ERROR |
| Orchestrator | Pipeline completed | INFO |
| Scraper | Feed fetched | INFO |
| Scraper | Article extracted (with content_status) | INFO |
| Scraper | Article extraction failed, using excerpt | WARNING |
| Scraper | Paywall detected | INFO |
| Analyzer | Themes identified | INFO |
| Generator | Deliverable generated (with word count) | INFO |
| Evaluator | Evaluation round result | INFO |
| Evaluator | Refinement needed | INFO |
| Evaluator | Theme approved / auto-approved | INFO |
| Brief | Daily brief generated (with word count) | INFO |
| Email | Email sent (with subject) | INFO |
| LLM Client | API call (model, tokens, latency) | INFO |
| LLM Client | API retry (status code, attempt) | WARNING |
| LLM Client | API failure | ERROR |

### 10.3 Log Rotation

`deploy/logrotate.conf`:
```
/opt/ai-news-pipeline/logs/pipeline.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

Installed by copying to `/etc/logrotate.d/ai-news-pipeline`.

---

## 11. Error Handling

### 11.1 Error Hierarchy

```
PipelineError (base)
├── ConfigError              # Invalid config, missing env vars
├── StageFailedError         # Stage exhausted all retries
│   └── .stage_name: str
│   └── .cause: Exception
├── LLMClientError           # Unrecoverable API error
├── AnalysisParseError       # LLM response could not be parsed
├── EmailError               # SMTP failure
└── DatabaseError            # SQLite operational error
```

### 11.2 Retry Strategy

| Error Type | Retries | Backoff | Scope |
|------------|---------|---------|-------|
| LLM 429/5xx/timeout | 3–5 (within stage attempt) | Exponential: 10s base, 60s cap | `llm.py` internal |
| Stage any exception | 2 (3 total attempts) | Fixed: 30s | `main.py` retry_wrapper |
| Email SMTP transient | 2 (3 total attempts) | Fixed: 10s | `emailer.py` internal |

### 11.3 Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| Single feed unreachable | Log warning, skip feed, continue with other feeds |
| Article extraction fails | Fall back to RSS excerpt, set content_status accordingly |
| LLM response unparseable (analyzer) | Raise `AnalysisParseError`, trigger stage retry |
| LLM response unparseable (evaluator) | Treat as `needs_refinement`, log warning |
| Evaluation never passes | Auto-approve after 3 refinement rounds |
| Email send fails on success emails | Retry 3 times; if all fail, log error but do NOT fail the pipeline (data is in DB) |
| Email send fails on failure alert | Log error to stdout/journald as last resort |
| No new articles found | Log info, skip generate/evaluate/brief stages, send "no new articles" brief email |
| No previous daily brief (first run) | Analyzer treats all content as novel, no comparison |

### 11.4 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Pipeline completed successfully |
| 1 | Pipeline failed (any stage) |
| 2 | Configuration error (cannot start) |

---

## 12. Testing Strategy

### 12.1 Unit Tests

Each module has a corresponding `test_*.py` file. Tests use `pytest` with the following fixtures:

| Module | Test Focus | Mocking Strategy |
|--------|-----------|-----------------|
| `test_config.py` | YAML parsing, validation, missing env vars | Temp YAML files, clean env |
| `test_db.py` | All CRUD operations, schema initialization | In-memory SQLite (`:memory:`) |
| `test_llm.py` | Request building, retry logic, error handling | `httpx` mock via `respx` or `unittest.mock` |
| `test_scraper.py` | RSS parsing, article extraction, dedup, paywall fallback | Mock `feedparser` and `trafilatura`, sample RSS XML |
| `test_analyzer.py` | Prompt building, response parsing, validation | Mock LLM client, sample JSON responses |
| `test_generator.py` | Prompt building, version incrementing, script_de isolation | Mock LLM client, in-memory DB |
| `test_evaluator.py` | Pass/fail logic, refinement trigger, auto-approve at round 3 | Mock LLM client, in-memory DB |
| `test_brief.py` | Prompt building, word count | Mock LLM client, in-memory DB |
| `test_emailer.py` | Email formatting, SMTP interaction | Mock `smtplib.SMTP` |

### 12.2 Integration Tests

`test_integration.py` — End-to-end pipeline test with all external dependencies mocked:

1. Mock RSS feeds returning sample data.
2. Mock OpenRouter API returning predetermined responses.
3. Mock SMTP.
4. Run full pipeline via `main.main()`.
5. Assert: DB contains expected records, correct number of emails sent, pipeline status = 'completed'.

### 12.3 Test Fixtures

`tests/fixtures/sample_rss.xml` — Valid RSS 2.0 feed with 5 entries, varying dates.
`tests/fixtures/sample_articles.json` — 10 pre-built article records for DB seeding.
`tests/fixtures/sample_themes.json` — 3 theme definitions with source article references.

### 12.4 Running Tests

```bash
cd /opt/ai-news-pipeline
python -m pytest tests/ -v --tb=short
```

No external services required. All tests run offline with mocks.

---

## 13. Build/Run/Deploy

### 13.1 Initial Setup

```bash
# 1. Create service user
sudo useradd --system --home-dir /opt/ai-news-pipeline --shell /usr/sbin/nologin ai-news-pipeline

# 2. Create directory structure
sudo mkdir -p /opt/ai-news-pipeline/{config,data,logs,src,prompts,tests,tests/fixtures,deploy,venv}
sudo chown -R ai-news-pipeline:ai-news-pipeline /opt/ai-news-pipeline

# 3. Copy project files (as ai-news-pipeline user)
sudo -u ai-news-pipeline cp -r src/ /opt/ai-news-pipeline/src/
sudo -u ai-news-pipeline cp -r prompts/ /opt/ai-news-pipeline/prompts/
sudo -u ai-news-pipeline cp -r tests/ /opt/ai-news-pipeline/tests/
sudo -u ai-news-pipeline cp requirements.txt /opt/ai-news-pipeline/

# 4. Create virtual environment and install dependencies
sudo -u ai-news-pipeline python3 -m venv /opt/ai-news-pipeline/venv
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt

# 5. Create .env file
sudo -u ai-news-pipeline cat > /opt/ai-news-pipeline/.env << 'EOF'
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
EOF
sudo chmod 600 /opt/ai-news-pipeline/.env

# 6. Create config file
sudo -u ai-news-pipeline cp config/feeds.yaml /opt/ai-news-pipeline/config/feeds.yaml
# Edit feeds.yaml with actual feed URLs

# 7. Initialize database
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --init-db --config /opt/ai-news-pipeline/config/feeds.yaml

# 8. Install systemd units
sudo cp deploy/ai-news-pipeline.service /etc/systemd/system/
sudo cp deploy/ai-news-pipeline.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-news-pipeline.timer
sudo systemctl start ai-news-pipeline.timer

# 9. Install logrotate
sudo cp deploy/logrotate.conf /etc/logrotate.d/ai-news-pipeline

# 10. Verify timer
sudo systemctl list-timers ai-news-pipeline.timer
```

### 13.2 Manual Run

```bash
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --config /opt/ai-news-pipeline/config/feeds.yaml
```

Or via systemd:

```bash
sudo systemctl start ai-news-pipeline.service
sudo journalctl -u ai-news-pipeline.service -f
```

### 13.3 Update Deployment

```bash
# 1. Copy updated source files
sudo -u ai-news-pipeline cp -r src/ /opt/ai-news-pipeline/src/
sudo -u ai-news-pipeline cp -r prompts/ /opt/ai-news-pipeline/prompts/

# 2. If dependencies changed:
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/pip install -r /opt/ai-news-pipeline/requirements.txt

# 3. If schema changed:
sudo -u ai-news-pipeline /opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --init-db --config /opt/ai-news-pipeline/config/feeds.yaml

# 4. Test manually before next scheduled run:
sudo systemctl start ai-news-pipeline.service
```

### 13.4 systemd Unit Files

**deploy/ai-news-pipeline.service:**
```ini
[Unit]
Description=AI News Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ai-news-pipeline
Group=ai-news-pipeline
EnvironmentFile=/opt/ai-news-pipeline/.env
ExecStart=/opt/ai-news-pipeline/venv/bin/python /opt/ai-news-pipeline/src/main.py --config /opt/ai-news-pipeline/config/feeds.yaml
TimeoutStartSec=1800
StandardOutput=journal
StandardError=journal
```

**deploy/ai-news-pipeline.timer:**
```ini
[Unit]
Description=AI News Pipeline Daily Timer

[Timer]
OnCalendar=*-*-* 04:00 Europe/Berlin
Persistent=true

[Install]
WantedBy=timers.target
```

---

## 14. Acceptance Criteria

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC1 | Pipeline runs end-to-end at 04:00 Europe/Berlin via systemd timer | `systemctl list-timers` shows next run; check journald after 04:00 |
| AC2 | All 50 feeds are scraped; new articles stored in SQLite | Query `articles` table after run; count matches feed entries |
| AC3 | Full-article extraction attempted for every new article; paywalled articles marked `excerpt_paywall` | Query `articles.content_status` distribution |
| AC4 | Duplicate articles (same normalized URL) are not inserted twice | Insert same URL twice; second insert is skipped |
| AC5 | Analyzer identifies 1–5 themes per run, classified as `novel` or `continuation` | Query `themes` table after run |
| AC6 | Three deliverables generated per theme: `summary_en`, `script_en`, `script_de` | Query `deliverables` table; 3 records per theme |
| AC7 | German script is generated without English script as input (only source articles + English summary) | Code review of `generator.py`; verify `script_de` prompt does not include `script_en` content |
| AC8 | Evaluation runs quality + adversarial checks; refinement loop executes up to 3 rounds | Query `evaluation_rounds` table; verify round_number ≤ 3 |
| AC9 | Theme auto-approved after 3 refinement rounds regardless of evaluation result | Integration test with mock evaluator always returning `fail`; verify `status='auto_approved'` |
| AC10 | Daily brief generated (~700 words) and stored in `daily_briefs` | Query table; verify word count |
| AC11 | Previous day's daily brief used for novelty comparison | Log message confirms brief loaded; `continuation` themes appear when appropriate |
| AC12 | Success emails sent: 1 brief + N theme emails to recipient@gmail.com | Check Gmail inbox after run |
| AC13 | Failure alert email sent with stage name, error, traceback, and log tail on unrecoverable failure | Force a failure (e.g., invalid API key); check inbox |
| AC14 | Pipeline exits with code 0 on success, 1 on failure | `echo $?` after manual run |
| AC15 | All secrets loaded from environment variables, not present in config file or logs | Grep config file and log file for API key patterns |
| AC16 | SQLite database uses WAL mode | `sqlite3 data/pipeline.db 'PRAGMA journal_mode;'` returns `wal` |
| AC17 | Log file contains structured JSON entries for all stages | Inspect `pipeline.log` after run |
| AC18 | Pipeline completes within 30 minutes for a typical run (50 feeds, 3–5 themes) | Measure wall-clock time |

---

## 15. Final Consistency Checklist

| # | Check | Status |
|---|-------|--------|
| 1 | Doc 1 data model matches Doc 2 DDL exactly | ✅ All tables, columns, types, and constraints identical |
| 2 | Doc 1 component responsibilities match Doc 2 module specifications | ✅ Each component in Doc 1 §4 has a corresponding module spec in Doc 2 §8 |
| 3 | Model assignments consistent (strong/weak) across both docs | ✅ Generator/Analyzer/Brief use strong; Evaluator uses weak |
| 4 | Refinement round ceiling (3) consistent | ✅ Doc 1 §2 R9, Doc 1 §10 table, Doc 2 §8.5, Doc 2 §4 config |
| 5 | Retry count (2 retries = 3 attempts) consistent | ✅ Doc 1 §3, Doc 2 §11.2, Doc 2 §4 config |
| 6 | Email recipients and method consistent | ✅ Gmail SMTP with App Password in both docs |
| 7 | German script generation isolation (no English script input) consistent | ✅ Doc 1 §4.4, Doc 2 §8.4 |
| 8 | SQLite as sole data store consistent | ✅ No other store mentioned in either doc |
| 9 | No TBDs or placeholders in either doc | ✅ All sections complete |
| 10 | No invented third-party services | ✅ Only OpenRouter (specified), Gmail SMTP (specified), trafilatura/httpx (libraries) |
| 11 | All assumptions from Doc 1 carried forward or resolved in Doc 2 | ✅ Assumptions addressed in module specs and config |
| 12 | Config schema in Doc 1 §6.4 matches Doc 2 §4.1 | ✅ All fields present and consistent |
| 13 | Error handling in Doc 2 covers all failure modes identified in Doc 1 §11 risks | ✅ Paywall fallback, LLM parse errors, rate limits, email failure all addressed |
| 14 | Acceptance criteria traceable to requirements in Doc 1 §2 | ✅ Each AC maps to one or more R# requirements |
| 15 | No contradictions between documents | ✅ Reviewed; none found |

---
