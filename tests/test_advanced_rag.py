"""
Tests for the hybrid-retrieval, faithfulness-checking, and metrics-dashboard
additions on top of the base support-agent pipeline.

Like test_support.py, Hugging Face is never actually called here (no
HF_API_TOKEN configured in tests), so:
  - "hybrid" retrieval exercises its BM25-only fallback path (dense
    retrieval returns [] because embed_text() is disabled).
  - faithfulness checking exercises its lexical-overlap fallback path
    (nli_entailment() is disabled).
Both fallback paths are real, shipped code - not test-only stubs - so this
still tests production behavior, just the "HF unavailable" branch of it.
"""

from typing import Any, cast

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_password_hash
from app.db.seed import seed_knowledge_base
from app.models.user import User
from app.services import faithfulness as faithfulness_engine
from app.services import knowledge_base as kb_service
from app.models.support import KnowledgeBaseEntry


@pytest_asyncio.fixture(autouse=True)
async def seeded_kb(session: AsyncSession) -> None:
    await seed_knowledge_base(session)


@pytest_asyncio.fixture
async def staff_user(session: AsyncSession) -> User:
    result = await session.execute(select(User).where(User.username == "rag_staff_user"))
    user = result.scalar_one_or_none()
    if user:
        return cast(User, user)
    user = User(
        username="rag_staff_user",
        hashed_password=get_password_hash("pw123456"),
        is_staff=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return cast(User, user)


@pytest_asyncio.fixture
async def staff_token(staff_user: User) -> str:
    token = create_access_token("rag_staff_user", staff_user.id)
    return f"Bearer {token}"


# --------------------------------------------------------------------------
# BM25 / hybrid retrieval
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bm25_retrieval_finds_relevant_entry(session: AsyncSession) -> None:
    results = await kb_service.search_knowledge_base(
        session, query="reset my password", retrieval_mode="bm25"
    )
    assert results
    assert any("password" in r.entry.title.lower() for r in results)
    # BM25 scores are normalized into 0..1 by search_knowledge_base
    assert all(0.0 <= r.relevance <= 1.0 for r in results)


@pytest.mark.asyncio
async def test_hybrid_retrieval_falls_back_to_bm25_when_hf_disabled(
    session: AsyncSession,
) -> None:
    """
    With no HF_API_TOKEN configured, dense retrieval returns [] and hybrid
    search must degrade to the BM25 ranking rather than returning nothing.
    """
    hybrid_results = await kb_service.search_knowledge_base(
        session, query="reset my password", retrieval_mode="hybrid"
    )
    bm25_results = await kb_service.search_knowledge_base(
        session, query="reset my password", retrieval_mode="bm25"
    )
    assert [r.entry.id for r in hybrid_results] == [r.entry.id for r in bm25_results]


@pytest.mark.asyncio
async def test_retrieval_modes_all_drop_irrelevant_entries(session: AsyncSession) -> None:
    for mode in ("tfidf", "bm25", "hybrid"):
        results = await kb_service.search_knowledge_base(
            session, query="completely unrelated cryptocurrency NFT query", retrieval_mode=mode
        )
        assert results == [], f"retrieval_mode={mode} should have found nothing relevant"


# --------------------------------------------------------------------------
# Faithfulness checking
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_faithfulness_high_when_answer_grounded_in_context() -> None:
    context = (
        "You can request a refund from your Order History page within 30 "
        "days of purchase. Refunds are processed within 5-7 business days."
    )
    answer = "You can request a refund from your Order History page within 30 days of purchase."
    result = await faithfulness_engine.check_faithfulness(answer, context)
    assert result.method == "lexical_overlap"  # HF disabled in tests
    assert result.score >= 0.8
    assert result.is_faithful


@pytest.mark.asyncio
async def test_faithfulness_low_when_answer_invents_unsupported_content() -> None:
    context = (
        "You can request a refund from your Order History page within 30 "
        "days of purchase."
    )
    answer = "We offer unlimited lifetime refunds and free international shipping upgrades."
    result = await faithfulness_engine.check_faithfulness(answer, context)
    assert not result.is_faithful
    assert result.score < faithfulness_engine.settings.FAITHFULNESS_THRESHOLD


@pytest.mark.asyncio
async def test_faithfulness_empty_answer_is_not_faithful() -> None:
    result = await faithfulness_engine.check_faithfulness("", "some context")
    assert not result.is_faithful
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_chat_response_includes_faithfulness_score(
    async_client: AsyncClient, session: AsyncSession
) -> None:
    user = User(username="rag_chat_user", hashed_password=get_password_hash("pw123456"))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token = f"Bearer {create_access_token('rag_chat_user', user.id)}"

    response = await async_client.post(
        "/support/chat",
        json={"message": "How do I reset my password? I forgot my login."},
        headers={"Authorization": token},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "auto_send"
    # Raw KB text is used verbatim (no HF configured), so it's faithful to
    # itself by construction - this exercises the wiring, not the score itself.
    assert data["faithfulness_score"] is not None
    assert data["faithfulness_score"] >= faithfulness_engine.settings.FAITHFULNESS_THRESHOLD


# --------------------------------------------------------------------------
# Metrics dashboard endpoint
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_summary_requires_staff(
    async_client: AsyncClient, session: AsyncSession
) -> None:
    user = User(username="rag_nonstaff_user", hashed_password=get_password_hash("pw123456"))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token = f"Bearer {create_access_token('rag_nonstaff_user', user.id)}"

    response = await async_client.get(
        "/support/metrics/summary", headers={"Authorization": token}
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_metrics_summary_shape(
    async_client: AsyncClient, staff_token: str, session: AsyncSession
) -> None:
    # Generate at least one decision log to aggregate over.
    user = User(username="rag_metrics_user", hashed_password=get_password_hash("pw123456"))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token = f"Bearer {create_access_token('rag_metrics_user', user.id)}"
    await async_client.post(
        "/support/chat",
        json={"message": "How do I reset my password? I forgot my login."},
        headers={"Authorization": token},
    )

    response = await async_client.get(
        "/support/metrics/summary", headers={"Authorization": staff_token}
    )
    assert response.status_code == 200
    data = response.json()
    assert "live" in data
    assert "eval" in data
    assert data["live"]["total_decisions"] >= 1
    assert "decision_distribution" in data["live"]
    assert "avg_confidence_by_intent" in data["live"]
    assert "tickets" in data["live"]
