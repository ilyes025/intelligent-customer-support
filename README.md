# Intelligent Support Agent API

A production-style FastAPI service implementing an **intelligent support
agent**, built on top of a FastAPI + async SQLAlchemy + Alembic + JWT/Docker
template. It answers user support requests by retrieving grounded content
from a knowledge base, scoring its own confidence, and routing the request
to one of: an automatic answer, an answer pending human validation, or an
escalation to a human agent.

> This started from a generic FastAPI template (auth, async DB, Docker,
> migrations already in place) and adds a full support-agent domain on top
> of it, following the same router → service → model layering.

This README has two parts:
- **Part 1 — Business Context**: the problem, why this approach, and who uses it.
- **Part 2 — Technical Documentation**: architecture, API, configuration, and setup.

---

# Part 1 — Business Context

## The problem

Every company selling online faces a flood of repetitive support questions —
"where's my order," "how do I reset my password," "why was I charged twice."
Handling these manually is:

- **Slow for customers** — waiting in a queue for a known, simple answer
- **Expensive for the business** — staff retyping the same answers repeatedly
- **Inconsistent** — different agents phrase or even get answers wrong

The obvious fix — "let an AI chatbot answer everything" — creates a worse
problem: a generic LLM will confidently invent an answer it doesn't actually
know. A wrong answer about a refund policy or account security is worse than
no answer at all. This project exists to solve that trade-off: **automate
the repetitive stuff safely, and know when to hand off.**

## Why this approach

- **Grounding (RAG)** — the AI only ever answers from the company's actual
  help articles, never from general knowledge. It cannot invent a policy
  that doesn't exist.
- **Confidence-based escalation** — the system continuously asks "how sure
  am I, and do I have a real source?" If either answer is no, it stops and
  routes to a human instead of guessing.
- **Staff stay in control** — nothing uncertain reaches a customer without a
  human review, and staff expand what the AI knows simply by adding new
  articles — no engineering work required.

## Worked example: Northwind Outfitters

A mid-size e-commerce business selling hiking and camping gear, with a
4-person support team. Its knowledge base is organized into the same
categories built into this project:

| Category | Typical customer questions |
|---|---|
| Shipping | "Where's my package?" / "It says delivered but I never got it" |
| Billing | "Why was I charged twice?" / "I want a refund" |
| Account | "How do I update my email?" / "How do I delete my account?" |
| Technical | "Your checkout page is throwing an error" / "I forgot my password" |

For a team of 4, these categories can be ~80% of daily ticket volume — and
almost none of them need human judgment. With this system:

- A 2 AM "where's my order" question gets an instant, accurate answer.
- A genuinely unusual case (damaged item, dispute, angry complaint) is
  escalated immediately — the system recognizes it has no good answer.
- Staff spend their time only on the ~20% of tickets that need a human.
- New edge cases are handled by adding a help article — no code changes.

**Business need, in one sentence:** let the AI safely absorb the repetitive,
low-risk majority of support questions, while guaranteeing that anything
uncertain or high-stakes still reaches a human.

## Use cases and users

| # | Use case | Primary user(s) | Description |
|---|---|---|---|
| 1 | Account sign-up & login | Regular user | Create an account and log in to get an access token (60 min validity). All accounts start as non-staff. |
| 2 | Staff promotion & login | Admin, Staff | Admin promotes a user to staff directly in the database (no public endpoint). Staff logs in via the same endpoint and the token carries staff access. |
| 3 | Automated answer (high confidence) | Regular user | A strong knowledge-base match produces a grounded answer that is auto-sent — no human involved. |
| 4 | Escalation to a human | Regular user → Staff | No KB match, or the message is urgent → a support ticket is created immediately regardless of confidence. |
| 5 | Pending validation (medium confidence) | Regular user → Staff | A plausible but uncertain answer (confidence 0.40–0.75) is held for staff to confirm before being trusted. |
| 6 | Ticket review & resolution | Staff | Staff list and resolve tickets created by escalation or pending-validation cases. |
| 7 | Knowledge-base expansion | Staff | Staff add new articles; embeddings are computed lazily on first matching query. |
| 8 | System diagnostics | Admin / Developer | Technical checks (embedding connectivity, cached embeddings, staff dashboard) — not customer-facing. |

**By role:**
- **Regular user (customer):** signs up, logs in, asks questions → gets an instant answer, a pending-confirmation answer, or an escalation.
- **Staff (support agent):** reviews tickets, resolves what the AI can't answer, grows the knowledge base.
- **Admin:** promotes users to staff (a deliberate, database-level action).
- **Developer/Ops:** runs diagnostics to confirm the pipeline is healthy.

---

# Part 2 — Technical Documentation

## Why this design

**Theme 1 — Reliability & response validation**
The agent never freely generates text unsupervised. Answers are *composed
from retrieved, trust-scored knowledge-base entries* (see
`app/services/agent.py::_compose_answer`), then a **faithfulness check**
(`app/services/faithfulness.py`) verifies the generated answer is actually
supported by that context — an LLM instruction to "only use the context" is
not a guarantee, so this checks it after the fact. A weighted **confidence
engine** (`app/services/confidence.py`) combines KB relevance, source
trust/verification status, intent-classification confidence, and agreement
with the user's own history into a single score. Confidence + faithfulness
together drive routing (`app/services/routing_policy.py`):

| Confidence | Decision |
|---|---|
| ≥ 0.75 (grounded source, not urgent, and faithfulness check passes) | `auto_send` |
| 0.40 – 0.75 | `pending_validation` (sent, but flagged; a staff member must approve/replace it) |
| < 0.40, or no grounded source at all, or urgent | `escalated` (a `SupportTicket` is opened for a human) |

A generated answer that fails the faithfulness check is downgraded one
notch regardless of its confidence score (`auto_send` → `pending_validation`
→ `escalated`) — see [Metrics](#metrics) below for exactly how both scores
are calculated. Every decision is written to `AgentDecisionLog` for
audit/explainability.

**Theme 2 — Context understanding & routing**
`app/services/context.py` builds a `UserContext` from (a) this app's own
conversation history and (b) an external profile/history lookup
(JSONPlaceholder, standing in for a CRM/order system). `app/services/classifier.py`
classifies intent (billing / shipping / technical / account / general) and
urgency with a fast local keyword scorer, optionally blended with a Hugging
Face zero-shot model if configured. If the request is too ambiguous to
classify confidently, the agent asks a clarifying question instead of
guessing (`app/services/agent.py`, clarification gate).

**Theme 3 — Data source access & filtering**
`app/services/knowledge_base.py` retrieves the most relevant KB entries.
Three retrieval modes are available, selected by `KB_RETRIEVAL_MODE`:

- `tfidf` — the original dependency-free TF-IDF + cosine-similarity
  implementation (pure Python, no scikit-learn/scipy)
- `bm25` — BM25Okapi lexical ranking (`rank_bm25`), generally stronger on
  short FAQ-style text than TF-IDF
- `hybrid` (**default**) — BM25 (lexical) fused with dense semantic search
  (local `sentence-transformers` embeddings) via Reciprocal Rank Fusion;
  automatically degrades to BM25-only if embeddings are unavailable, so it
  is always safe to leave enabled

In every mode, entries **below a relevance threshold are dropped entirely**
(source filtering) rather than returned as noisy matches, so the agent can
never treat weak overlap as a real source.
`app/services/external_sources.py` wraps the external API with a timeout,
an in-memory LRU cache (reusing the template's `LRUCache`), and a small
circuit breaker so a flaky upstream degrades to "no data from this source"
instead of crashing the pipeline.

## Architecture

```
Client
  │  POST /support/chat  (JWT auth)
  ▼
app/api/support.py
  ▼
app/services/agent.py  (orchestrator)
  ├─ app/services/context.py        → internal history + external profile/history
  ├─ app/services/classifier.py     → intent + urgency (local, +optional HF)
  ├─ app/services/knowledge_base.py → hybrid (BM25 + dense/RRF) retrieval, source-filtered
  ├─ app/services/confidence.py     → weighted confidence score
  ├─ app/services/faithfulness.py   → NLI/lexical faithfulness check on the generated answer
  └─ app/services/routing_policy.py → auto_send | pending_validation | escalated
                                       | (clarification handled earlier;
                                       faithfulness can downgrade one notch)
  ▼
Persists: Message, AgentDecisionLog, SupportTicket (if any)
```

### Data model additions

- `Conversation`, `Message` — chat history
- `KnowledgeBaseEntry` — curated FAQ/doc content, each with `trust_score`
  and `verified`; only verified, high-trust entries can back an
  auto-sent answer
- `SupportTicket` — created for `pending_validation` / `escalated` decisions
- `AgentDecisionLog` — full audit trail of every routing decision, including
  `intent_confidence`, `confidence_score`, `faithfulness_score`, and which
  `retrieval_mode` (`tfidf` | `bm25` | `hybrid`) produced the answer
- `KnowledgeBaseEntry` gains `embedding` — a JSON-encoded vector, lazily
  computed and cached the first time hybrid/dense retrieval needs it
- `User` gains `is_staff`, `is_active`, `external_user_id`

## API

All endpoints below are new; the template's existing `/auth/*` and
`/health` endpoints are unchanged (see below for those).

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/support/chat` | user | Send a message, get a routed response |
| GET | `/support/conversations/{id}` | user (owner) | Get conversation history |
| GET | `/support/kb` | user | List/search knowledge-base entries |
| POST | `/support/kb` | staff | Add a knowledge-base entry |
| GET | `/support/tickets` | staff | List validation/escalation queue (`?status=`) |
| POST | `/support/tickets/{id}/resolve` | staff | Approve/reject/resolve a ticket |
| GET | `/support/decision-logs` | staff | Full audit trail (`?conversation_id=`) |
| GET | `/support/metrics/summary` | staff | Aggregated live metrics + latest offline eval results (see [Metrics](#metrics)) |

### Example: chat

```bash
curl -X POST http://localhost:8000/support/chat \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```

```json
{
  "conversation_id": 1,
  "message_id": 2,
  "answer": "Go to the login page and click 'Forgot password'...",
  "intent": "technical",
  "intent_confidence": 0.95,
  "confidence_score": 0.942,
  "decision": "auto_send",
  "sources": [
    {"type": "knowledge_base", "reference": "kb:5:Resetting your password", "relevance": 0.36, "trust_score": 0.95}
  ],
  "ticket_id": null,
  "faithfulness_score": 0.88
}
```

A vague message (`"help"`) returns `decision: "clarification_requested"`
instead of a guessed answer. A message with no matching, verified KB entry
returns `decision: "escalated"` and opens a `SupportTicket`.

### Staff / validation flow

Give a user staff rights (`is_staff=True` on the `User` row — there's no
public "become staff" endpoint by design), then:

```bash
# See what's waiting on a human
curl http://localhost:8000/support/tickets?status=pending_validation \
  -H "Authorization: Bearer $STAFF_TOKEN"

# Approve, reject, or resolve with a final answer
curl -X POST http://localhost:8000/support/tickets/3/resolve \
  -H "Authorization: Bearer $STAFF_TOKEN" -H "Content-Type: application/json" \
  -d '{"action": "resolve", "final_answer": "Handled manually."}'
```

## Configuration

New environment variables (all optional, sensible defaults provided):

| Variable | Description | Default |
|---|---|---|
| `EXTERNAL_API_BASE_URL` | Mock support/CRM API base URL | `https://jsonplaceholder.typicode.com` |
| `EXTERNAL_API_TIMEOUT_SECONDS` | Timeout per external call | `3.0` |
| `EXTERNAL_API_CACHE_SIZE` | In-memory LRU cache size | `256` |
| `EXTERNAL_API_FAILURE_THRESHOLD` | Consecutive failures before the circuit opens | `3` |
| `EXTERNAL_API_COOLDOWN_SECONDS` | Circuit-breaker cooldown | `30.0` |
| `HF_API_TOKEN` | Hugging Face token; leave unset to disable **hosted** HF calls (classification, NLI, generation) — local embeddings still work without it | `""` |
| `HF_ZERO_SHOT_MODEL` | Zero-shot classification model (hosted) | `facebook/bart-large-mnli` |
| `HF_API_TIMEOUT_SECONDS` | Timeout for hosted HF calls | `5.0` |
| `HF_GENERATION_MODEL` | Hosted model used to compose RAG answers / internal drafts | `Qwen/Qwen2.5-7B-Instruct` |
| `HF_GENERATION_PROVIDER` | Optional: pin a specific inference provider (e.g. `novita`, `together`) | `""` (auto-select) |
| `HF_GENERATION_MAX_TOKENS` | Max tokens for a generated answer | `300` |
| `HF_GENERATION_TEMPERATURE` | Sampling temperature for generation | `0.3` |
| `HF_EMBEDDING_MODEL` | **Local** sentence-transformers model for the dense half of hybrid retrieval — runs on-box, no `HF_API_TOKEN` needed | `sentence-transformers/all-MiniLM-L6-v2` |
| `HF_NLI_MODEL` | Hosted model used for the faithfulness entailment check | `facebook/bart-large-mnli` |
| `KB_RETRIEVAL_MODE` | Retrieval strategy: `tfidf` \| `bm25` \| `hybrid` (hybrid falls back to BM25-only if embeddings are unavailable) | `hybrid` |
| `KB_RRF_K` | Reciprocal Rank Fusion constant used to combine BM25 + dense rankings in hybrid mode | `60` |
| `KB_MIN_RELEVANCE_SCORE` | Minimum relevance (0–1, any mode) to keep a KB match | `0.12` |
| `KB_TOP_K` | Max KB entries retrieved per query | `3` |
| `FAITHFULNESS_THRESHOLD` | Minimum faithfulness score to accept a generated answer as-is; below this the decision is downgraded one notch | `0.55` |
| `CONFIDENCE_AUTO_THRESHOLD` | Score required to auto-send | `0.75` |
| `CONFIDENCE_VALIDATION_THRESHOLD` | Score required to queue for validation (below → escalate) | `0.40` |
| `INTENT_CLARIFICATION_THRESHOLD` | Intent confidence below which short messages trigger a clarifying question | `0.35` |

The knowledge base is seeded automatically on first startup from
`app/db/seed_data.py` (10 curated FAQ entries across billing, shipping,
technical, account and general categories) if the table is empty.

## Metrics

Every routing decision the agent makes is scored and logged, so quality can
be measured both live (in production) and offline (against a golden test
set). This section explains exactly how each number is calculated.

### Confidence score (`app/services/confidence.py`)

A single 0–1 score, computed fresh for every message, as a **weighted sum
of four signals**:

| Signal | Weight | What it measures |
|---|---|---|
| `kb_relevance` | 0.35 | Retrieval score of the top-ranked KB match (relevance × 2.5, capped at 1.0 — rewards even a moderate match) |
| `kb_trust` | 0.25 | The matched entry's `trust_score` if `verified=True`; otherwise `trust_score × 0.5` (unverified content is penalized) |
| `intent_confidence` | 0.25 | How confident the intent classifier is that it understood the *type* of request |
| `source_agreement` | 0.15 | `1.0` normally; drops to `0.3` if a conflict is detected — currently: the user has 3+ recent history items, treated as a sign of a recurring/unresolved issue rather than a fresh one-off |

```
score = 0.35·min(kb_relevance×2.5, 1.0) + 0.25·kb_trust
        + 0.25·intent_confidence + 0.15·source_agreement
```

**Hard cap:** if no KB entry was retrieved at all, the score is capped at
`0.3` regardless of the other signals — there is nothing verifiable to
back the answer. The final score, and each signal that fed it, is stored
per-request for audit (`ConfidenceBreakdown`).

### Faithfulness score (`app/services/faithfulness.py`)

Computed only when a RAG answer is actually generated (`auto_send` /
`pending_validation` paths). Checks whether the generated text is genuinely
supported by the retrieved KB context it was supposed to be grounded in —
an instruction to the LLM ("only answer from this context") is not a
guarantee, so this verifies it after the fact:

- **When Hugging Face is enabled:** an NLI entailment score (0–1) between
  the KB context (premise) and the generated answer (hypothesis), via
  `HF_NLI_MODEL`.
- **Fallback (HF disabled or the call fails):** a lexical-coverage
  heuristic — the fraction of the answer's meaningful (non-stopword)
  tokens that also appear in the context.

A score below `FAITHFULNESS_THRESHOLD` (default `0.55`) is *not faithful*
and downgrades the routing decision one notch: `auto_send` →
`pending_validation` → `escalated`. This runs after `routing_policy.decide`
so a confidently-retrieved answer can still be caught if the LLM drifted
from its source.

### Retrieval relevance (`app/services/knowledge_base.py`)

Depends on `KB_RETRIEVAL_MODE`:
- `tfidf` — cosine similarity between TF-IDF vectors of the query and each entry (0–1)
- `bm25` — BM25Okapi score, normalized by the top score in that ranking (0–1)
- `hybrid` — BM25 and dense-embedding cosine-similarity rankings are fused
  with **Reciprocal Rank Fusion**: `score(doc) = Σ 1 / (KB_RRF_K + rank + 1)`
  across both rankings, then normalized by the highest fused score (0–1)

Entries scoring below `KB_MIN_RELEVANCE_SCORE` are dropped before they ever
reach the confidence engine.

### Intent confidence (`app/services/classifier.py`)

- **Local keyword scorer:** `confidence = min(0.95, 0.4 + best_keyword_hit_ratio × 3)`
- **If Hugging Face zero-shot is enabled and agrees** with the local
  result: the two confidences are averaged and boosted (+0.1, capped at
  `0.97`).
- **If they disagree:** the more confident of the two "wins," but its
  confidence is capped at `0.6` — disagreement itself is treated as
  uncertainty, so the system never reports high confidence when its two
  signals contradict each other.

### Live production metrics — `GET /support/metrics/summary` (staff-only)

Computed on demand from `AgentDecisionLog` / `SupportTicket` (`app/services/metrics.py`):

| Metric | How it's calculated |
|---|---|
| `decision_distribution` | Count of logged decisions, grouped by `decision` |
| `auto_send_rate`, `escalation_rate` | `count(decision) / total_decisions` |
| `avg_confidence_by_intent` | Average `confidence_score`, grouped by `intent` |
| `avg_faithfulness` | Average `faithfulness_score` across all logs where one was computed |
| `faithfulness_over_time` | Chronological list of `(timestamp, faithfulness_score)` pairs, most recent 100 |
| `tickets.resolution_rate` | `count(status='resolved') / total_tickets` |

An empty decision log returns zeroed stats rather than dividing by zero.
This is what powers the staff dashboard at `/static/dashboard.html`.

### Offline evaluation — `eval/run_eval.py`

Runs a golden Q&A dataset (`eval/golden_dataset.json`) through the real
pipeline against a disposable in-memory database, and reports RAGAS-style
metrics, written to `eval/results/latest.json`:

| Metric | How it's calculated |
|---|---|
| `context_recall` | Did retrieval surface the expected KB entry at all, for cases that have one? |
| `answer_relevancy` | Keyword-coverage proxy: fraction of expected keywords/phrases present in the generated answer (partial credit for partial phrase overlap) |
| `faithfulness` | The same live faithfulness score, aggregated across the golden set |
| `decision_distribution` | How many golden-set queries ended up in each decision bucket |
| `clarification_accuracy` / `no_match_escalation_accuracy` | Sanity checks that the deliberately "hard" golden-set cases (a vague message, an out-of-domain one) actually get `clarification_requested` / `escalated` |

Run it with:
```bash
python -m eval.run_eval
```
The staff `/support/metrics/summary` endpoint includes the most recent
`eval/results/latest.json` alongside live stats, so long as the file
exists; a missing file simply omits that section.

## Running it

Same workflow as the base template (Docker or local); see below. The
knowledge base seeds itself on first run, so you can try `/support/chat`
immediately after signing up a user.

```bash
docker-compose -f docker-compose.dev.yml up --build
# then:
curl -X POST http://localhost:8000/auth/signup -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demopass123"}'
curl -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demopass123"}'
# use the returned access_token for /support/chat as shown above
```

## Testing

```bash
pytest -q
```

`tests/test_support.py` covers: the confidence engine (source-agreement
penalties, trust-weighting, no-source cap), the routing policy (all four
decisions), the intent classifier, and end-to-end `/support/chat` flows for
`auto_send`, `clarification_requested`, and `escalated`, plus staff-only
ticket/KB/audit endpoints. `tests/test_advanced_rag.py` additionally covers
BM25/hybrid retrieval (including graceful fallback to BM25-only when HF is
disabled), relevance-threshold filtering across all retrieval modes, the
faithfulness check (grounded vs. hallucinated vs. empty answers), that
`/support/chat` returns a `faithfulness_score`, and the `/support/metrics/summary`
endpoint (staff-only access, response shape). All external calls are
avoided or explicitly mocked, so the suite is fast and network-free.

## Notable fixes made to the base template while building this

- The base template defined a `lifespan()` context manager (for DB cleanup
  on shutdown) but never actually passed it to `FastAPI(...)`, so it was
  dead code — startup/shutdown hooks silently never ran. Fixed by passing
  `lifespan=lifespan` to the `FastAPI()` constructor (this is also what
  now runs the knowledge-base seeding on startup).
- `async_sessionmaker` didn't set `expire_on_commit=False`, which causes a
  `MissingGreenlet` error the moment code reads an attribute off an ORM
  object right after `commit()` in an async context — a common async
  SQLAlchemy pitfall. Fixed at the session-factory level.
- `passlib==1.7.4`'s bcrypt backend runs an internal self-test
  (`detect_wrap_bug`) that's broken against `bcrypt>=4.1`: it hashes an
  oversized canned string, and newer bcrypt raises instead of silently
  truncating secrets over 72 bytes, so *every* password hash fails,
  regardless of the user's actual password length
  ([upstream issue](https://github.com/pyca/bcrypt/issues/684)). Pinning
  `bcrypt<4.1` "fixes" it locally but breaks the Alpine Docker build,
  since `bcrypt<4.1` has no musllinux wheel and Alpine lacks a Rust
  toolchain to build it from source. The durable fix was to drop passlib
  for password hashing entirely and call `bcrypt` directly
  (`app/core/security.py`) — fewer moving parts, and compatible with
  current bcrypt releases on any platform.
- Knowledge-base retrieval intentionally avoids scikit-learn: its `scipy`
  dependency has no musllinux wheels, which would force the Alpine image
  to compile a full scientific-Python stack from source. The `tfidf` mode
  in `app/services/knowledge_base.py` implements TF-IDF + cosine similarity
  in ~60 lines of pure Python instead; `bm25` uses the pure-Python
  `rank_bm25` package for the same reason. The dense half of `hybrid` mode
  runs `sentence-transformers` **locally** rather than calling HF's hosted
  embedding API, since HF has been progressively dropping free-tier
  serverless hosting for embedding models — running locally removes that
  dependency entirely and adds no network round-trip to the retrieval path.

## Known limitations / next steps

- Confidence/urgency/intent/faithfulness are heuristic; the weights and
  thresholds in `app/services/confidence.py`, `app/services/routing_policy.py`,
  and `app/services/faithfulness.py` are a reasonable starting point and are
  meant to be tuned against real labeled data (the offline eval harness in
  `eval/run_eval.py` exists for exactly this).
- mypy flags `Column[int]` vs `int` mismatches throughout the ORM layer —
  this is a pre-existing pattern from the base template's use of classic
  `Column(...)` declarations rather than SQLAlchemy 2.0 `Mapped[...]`
  typing; fixing it project-wide would be a separate refactor.
- `answer_relevancy` in the offline eval is a keyword-coverage proxy, not a
  semantic or LLM-judge score; swapping in an embedding-similarity or
  LLM-judge metric would give a more faithful relevancy signal once budget
  allows.
- The `source_agreement` conflict heuristic in `confidence.py` is currently
  a simple threshold (3+ recent history items = possible conflict); a
  proper contradiction-detection model would be more precise.

---

## Base template documentation (auth, Docker, migrations)

## Features

- **Modern Python**: Type hints, async/await syntax, and the latest FastAPI features
- **JWT Authentication**: Complete authentication system with access and refresh tokens
- **SQLAlchemy with Async**: Fully async database operations using SQLAlchemy 2.0+
- **Alembic Migrations**: Database schema migrations with Alembic
- **Role-based Access Control**: User roles with different permission levels (active, staff, superuser)
- **Docker Support**: Ready-to-use Docker and Docker Compose configurations
- **Developer-friendly**: Auto-reload, debugging, and development tools
- **Production-ready**: Configuration for deployment in production environments

## Project Structure

```
.
├── alembic/                 # Database migrations
├── app/                     # Main application package
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality (config, security)
│   ├── db/                  # Database session and base
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   └── utils/               # Utility functions
├── docker-compose.yml       # Docker Compose for production
├── docker-compose.dev.yml   # Docker Compose for development
├── Dockerfile               # Docker configuration
├── alambic.ini              # Alembic configuration
├── main.py                  # Application entry point
├── pyproject.toml           # Project dependencies and metadata
├── start.sh                 # Production startup script
└── start-dev.sh             # Development startup script
```

## Requirements

- Python 3.11+
- Docker (optional)

## Installation

### Using Docker (recommended)

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd fastapi-template
   ```

2. Start the application with Docker Compose:
   ```bash
   # For development
   docker-compose -f docker-compose.dev.yml up --build

   # For production
   docker-compose up --build
   ```

3. The API will be available at http://localhost:8000

### Local Development

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd fastapi-template
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up environment variables (create a `.env` file):
   ```
   DEBUG=true
   SECRET_KEY=your-secret-key
   DB_ENGINE=sqlite  # or postgresql
   # For PostgreSQL, add these:
   # DB_USER=postgres
   # DB_PASSWORD=password
   # DB_HOST=localhost
   # DB_PORT=5432
   # DB_NAME=app
   ```

5. Run migrations:
   ```bash
   alembic upgrade head
   ```

6. Start the application:
   ```bash
   uvicorn main:app --reload
   ```

7. The API will be available at http://localhost:8000

## API Documentation

Once the application is running, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /auth/signup` - Register a new user
- `POST /auth/login` - Authenticate and get tokens
- `POST /auth/token/refresh` - Refresh access token
- `POST /auth/logout` - Logout user
- `GET /auth/me` - Get current user information

### System

- `GET /health` - Health check endpoint

## Configuration

The application is configured through environment variables which can be set in a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `SECRET_KEY` | JWT secret key | `supersecretkey` |
| `ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiration time | `60` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token expiration time | `7` |
| `CORS_ORIGINS` | CORS allowed origins | `["*"]` |
| `DB_ENGINE` | Database engine | `sqlite` |
| `DB_USER` | Database user | `""` |
| `DB_PASSWORD` | Database password | `""` |
| `DB_HOST` | Database host | `""` |
| `DB_PORT` | Database port | `""` |
| `DB_NAME` | Database name | `app.db` |

## Development

### Running Tests

```bash
pytest
```

### Code Quality Tools

The project uses several tools to ensure code quality:

- **Black**: Code formatter
- **isort**: Import sorter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality checks

To set up pre-commit hooks:

```bash
pre-commit install
```

## Database

The template supports SQLite for development and PostgreSQL for production. The default is SQLite.

### Migrations

To create a new migration after changing models:

```bash
alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
alembic upgrade head
```

## Docker

The project includes Docker configurations for both development and production:

- `docker-compose.yml`: Production setup
- `docker-compose.dev.yml`: Development setup with hot-reload

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b ft/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin ft/my-feature`