# Implementation Roadmap

## Phase 1: Foundation & Infrastructure
**Goal:** Establish the environment, configuration, and data layer.
1. **Project Scaffolding:** Create the directory structure, `requirements.txt`, and `.env.example`.
2. **Configuration (`config.py`, `models.py`):** Implement Pydantic models and YAML loading. Validate env var resolution.
3. **Database Layer (`db.py`):** Implement schema DDL initialization and all CRUD methods. Use in-memory SQLite for initial testing.
4. **Infrastructure Setup:** Create the `ai-news-pipeline` user, directory structure, `.env` file, and set permissions.

## Phase 2: External Integrations
**Goal:** Implement and isolate all third-party API interactions.
1. **LLM Client (`llm.py`):** Implement the OpenRouter HTTP wrapper, retry logic (429/5xx/timeout), and error hierarchy.
2. **Emailer (`emailer.py`):** Implement Gmail SMTP logic, success formatting, and failure alert formatting. Test with App Password.

## Phase 3: Pipeline Stages (Sequential Build)
**Goal:** Build each stage from input to output, mocking downstream dependencies.
1. **Scraper (`scraper.py`):** Implement `feedparser` + `trafilatura` logic, URL normalization, and paywall fallback. Verify DB writes.
2. **Analyzer (`analyzer.py`):** Implement prompt building (with/without previous brief) and JSON response parsing. Verify theme DB writes.
3. **Generator (`generator.py`):** Implement the three prompt templates and generation logic. Enforce German script isolation (no English script input). Verify deliverable DB writes.
4. **Evaluator (`evaluator.py`):** Implement quality and adversarial prompt logic, JSON parsing, and the 3-round refinement loop trigger. Verify evaluation DB writes and status updates.
5. **Brief Generator (`brief.py`):** Implement brief prompt and generation. Verify daily brief DB writes.

## Phase 4: Orchestration & Observability
**Goal:** Wire the stages together and establish logging.
1. **Orchestrator (`main.py`):** Implement the sequential runner, stage retry wrapper, and failure alert trigger.
2. **Structured Logging:** Implement JSON logging to stdout and file, ensuring `pipeline_run_id` and `stage` are attached to all entries.
3. **Log Rotation:** Deploy `logrotate.conf`.

## Phase 5: Testing & Deployment
**Goal:** Verify the system works end-to-end before relying on the timer.
1. **Unit Tests:** Complete the `tests/` suite using mocked LLM and SMTP.
2. **Integration Test:** Run the full pipeline with mocked external APIs to verify DB state and email output.
3. **Manual Dry Run:** Execute `main.py` manually on the VPS against the real OpenRouter API and Gmail SMTP with 2-3 feeds.
4. **Systemd Activation:** Deploy `.service` and `.timer` units. Enable the timer.
5. **First Automated Run:** Monitor the 04:00 run via `journalctl` and verify the final email delivery.

---
