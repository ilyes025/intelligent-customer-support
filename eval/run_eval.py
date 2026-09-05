"""
Offline RAG evaluation harness.

Runs the golden Q&A dataset (golden_dataset.json) through the real
retrieval + generation + faithfulness pipeline against a disposable
in-memory SQLite database seeded with the same KB content used in
production, and reports RAGAS-style metrics:

  - context_recall      : was the expected KB entry actually retrieved?
  - answer_relevancy    : does the generated answer cover the expected
                           keywords for that question? (lexical proxy -
                           swap for an embedding-similarity or LLM-judge
                           score if/when HF is enabled and budget allows)
  - faithfulness        : the same NLI/lexical-overlap score used live in
                           agent.py, reported here in aggregate
  - decision_distribution: how many queries ended up auto_send /
                           pending_validation / escalated / clarification
  - clarification_accuracy / no_match_escalation_accuracy: sanity checks
                           on the two deliberately "hard" queries in the
                           golden set (a vague message, and an
                           out-of-domain one)

Usage:
    python -m eval.run_eval
    # or, from the project root:
    python eval/run_eval.py

Results are printed to stdout and written to eval/results/latest.json,
which is what the metrics dashboard (GET /support/metrics/dashboard)
reads to show retrieval/faithfulness quality alongside the live
production decision stats.

This intentionally does NOT go through the HTTP API or JWT auth - it
calls the service layer directly so it stays fast and can run in CI
without a running server.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models.support import KnowledgeBaseEntry
from app.models.user import User
from app.services import agent
from app.services import knowledge_base as kb_service

EVAL_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


async def _build_scratch_db() -> AsyncSession:
    """
    A disposable in-memory SQLite database, schema created directly from
    the ORM metadata (no Alembic needed for eval), seeded with the same
    KB_SEED_ENTRIES used in production plus one throwaway user.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    db = session_factory()

    from app.db.seed_data import KB_SEED_ENTRIES

    for entry_data in KB_SEED_ENTRIES:
        db.add(KnowledgeBaseEntry(**entry_data))

    eval_user = User(
        username="eval_runner",
        hashed_password="not_a_real_hash",
        is_staff=False,
        is_active=True,
    )
    db.add(eval_user)
    await db.commit()
    await db.refresh(eval_user)

    db.info["eval_user_id"] = eval_user.id
    return db


def _keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_tokens = _tokenize(answer)
    hits = 0
    for phrase in expected_keywords:
        phrase_tokens = _tokenize(phrase)
        if phrase_tokens and phrase_tokens.issubset(answer_tokens):
            hits += 1
        elif phrase_tokens & answer_tokens:
            hits += 0.5  # partial credit for partial phrase overlap
    return min(hits / len(expected_keywords), 1.0)


async def _evaluate_retrieval(db: AsyncSession, case: dict) -> bool:
    """context_recall for a single case: did retrieval surface the
    expected KB entry at all (any retrieval mode currently configured)?"""
    if not case.get("expected_kb_title"):
        return True  # nothing to recall for clarification/no-match cases
    matches = await kb_service.search_knowledge_base(db, query=case["query"], category=None)
    return any(m.entry.title == case["expected_kb_title"] for m in matches)


async def run_eval() -> dict:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text())
    db = await _build_scratch_db()
    user_id = db.info["eval_user_id"]

    # agent.handle_message expects a User ORM object, not just an id
    user = await db.get(User, user_id)

    per_case_results = []
    started = time.monotonic()

    for case in dataset:
        retrieval_hit = await _evaluate_retrieval(db, case)

        response = await agent.handle_message(
            db, user=user, message_text=case["query"], conversation_id=None
        )

        relevancy = _keyword_coverage(response.answer, case.get("expected_answer_keywords", []))

        per_case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "decision": response.decision,
                "context_recall_hit": retrieval_hit,
                "faithfulness_score": response.faithfulness_score,
                "answer_relevancy": relevancy,
                "expect_clarification": case.get("expect_clarification", False),
                "expect_no_match": case.get("expect_no_match", False),
            }
        )

    elapsed = time.monotonic() - started

    # --- Aggregate metrics --------------------------------------------------
    n = len(per_case_results)
    context_recall = sum(1 for r in per_case_results if r["context_recall_hit"]) / n

    faith_scores = [r["faithfulness_score"] for r in per_case_results if r["faithfulness_score"] is not None]
    avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else None

    relevancy_scores = [r["answer_relevancy"] for r in per_case_results]
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None

    decision_counts: Dict[str, int] = {}
    for r in per_case_results:
        decision_counts[r["decision"]] = decision_counts.get(r["decision"], 0) + 1

    clarification_cases = [r for r in per_case_results if r["expect_clarification"]]
    clarification_accuracy = (
        sum(1 for r in clarification_cases if r["decision"] == "clarification_requested")
        / len(clarification_cases)
        if clarification_cases
        else None
    )

    no_match_cases = [r for r in per_case_results if r["expect_no_match"]]
    no_match_escalation_accuracy = (
        sum(1 for r in no_match_cases if r["decision"] == "escalated") / len(no_match_cases)
        if no_match_cases
        else None
    )

    results = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": n,
        "elapsed_seconds": round(elapsed, 3),
        "metrics": {
            "context_recall": round(context_recall, 3),
            "avg_faithfulness": round(avg_faithfulness, 3) if avg_faithfulness is not None else None,
            "avg_answer_relevancy": round(avg_relevancy, 3) if avg_relevancy is not None else None,
            "clarification_accuracy": clarification_accuracy,
            "no_match_escalation_accuracy": no_match_escalation_accuracy,
            "decision_distribution": decision_counts,
        },
        "cases": per_case_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(results, indent=2))

    return results


def _print_summary(results: dict) -> None:
    m = results["metrics"]
    print(f"\nRan {results['n_cases']} cases in {results['elapsed_seconds']}s\n")
    print(f"  context_recall               : {m['context_recall']:.0%}")
    print(f"  avg_faithfulness             : {m['avg_faithfulness']}")
    print(f"  avg_answer_relevancy         : {m['avg_answer_relevancy']}")
    print(f"  clarification_accuracy       : {m['clarification_accuracy']}")
    print(f"  no_match_escalation_accuracy : {m['no_match_escalation_accuracy']}")
    print(f"  decision_distribution        : {m['decision_distribution']}")
    print(f"\nFull results written to {RESULTS_DIR / 'latest.json'}\n")


if __name__ == "__main__":
    results = asyncio.run(run_eval())
    _print_summary(results)
