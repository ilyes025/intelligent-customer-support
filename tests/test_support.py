"""
Tests for the intelligent support agent module.

External calls (JSONPlaceholder, Hugging Face) are never hit in these
tests: Hugging Face is disabled by default (no HF_API_TOKEN configured),
and JSONPlaceholder is only called when a user has ``external_user_id``
set, which none of the test users do, except in
``test_context_enrichment_from_external_source`` where it is explicitly
monkeypatched. This keeps the suite fast, deterministic and network-free.
"""

from typing import Any, Dict, List, cast

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_password_hash
from app.db.seed import seed_knowledge_base
from app.models.user import User
from app.services import classifier, confidence as confidence_engine, routing_policy
from app.services.knowledge_base import RetrievedEntry


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture(autouse=True)
async def seeded_kb(session: AsyncSession) -> None:
    await seed_knowledge_base(session)


@pytest_asyncio.fixture
async def regular_user(session: AsyncSession) -> User:
    result = await session.execute(select(User).where(User.username == "support_user"))
    user = result.scalar_one_or_none()
    if user:
        return cast(User, user)
    user = User(username="support_user", hashed_password=get_password_hash("pw123456"))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return cast(User, user)


@pytest_asyncio.fixture
async def staff_user(session: AsyncSession) -> User:
    result = await session.execute(select(User).where(User.username == "staff_user"))
    user = result.scalar_one_or_none()
    if user:
        return cast(User, user)
    user = User(
        username="staff_user",
        hashed_password=get_password_hash("pw123456"),
        is_staff=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return cast(User, user)


@pytest_asyncio.fixture
async def user_token(regular_user: User) -> str:
    token = create_access_token("support_user", regular_user.id)
    return f"Bearer {token}"


@pytest_asyncio.fixture
async def staff_token(staff_user: User) -> str:
    token = create_access_token("staff_user", staff_user.id)
    return f"Bearer {token}"


# --------------------------------------------------------------------------
# Unit tests: confidence engine
# --------------------------------------------------------------------------


class _FakeEntry:
    def __init__(self, trust_score: float, verified: bool) -> None:
        self.trust_score = trust_score
        self.verified = verified
        self.id = 1
        self.title = "fake"


def test_confidence_no_source_is_capped() -> None:
    breakdown = confidence_engine.compute_confidence(
        top_kb=None, intent_confidence=0.95, user_history=[], intent="billing"
    )
    assert breakdown.score <= 0.3


def test_confidence_high_when_strong_verified_match() -> None:
    entry = RetrievedEntry(entry=cast(Any, _FakeEntry(trust_score=0.95, verified=True)), relevance=0.8)
    breakdown = confidence_engine.compute_confidence(
        top_kb=entry, intent_confidence=0.9, user_history=[], intent="billing"
    )
    assert breakdown.score >= 0.75


def test_confidence_lower_for_unverified_source() -> None:
    verified = RetrievedEntry(entry=cast(Any, _FakeEntry(trust_score=0.9, verified=True)), relevance=0.5)
    unverified = RetrievedEntry(entry=cast(Any, _FakeEntry(trust_score=0.9, verified=False)), relevance=0.5)
    verified_score = confidence_engine.compute_confidence(
        top_kb=verified, intent_confidence=0.8, user_history=[], intent="billing"
    ).score
    unverified_score = confidence_engine.compute_confidence(
        top_kb=unverified, intent_confidence=0.8, user_history=[], intent="billing"
    ).score
    assert unverified_score < verified_score


def test_confidence_drops_on_conflicting_history() -> None:
    entry = RetrievedEntry(entry=cast(Any, _FakeEntry(trust_score=0.9, verified=True)), relevance=0.6)
    no_conflict = confidence_engine.compute_confidence(
        top_kb=entry, intent_confidence=0.8, user_history=[], intent="billing"
    )
    with_conflict = confidence_engine.compute_confidence(
        top_kb=entry,
        intent_confidence=0.8,
        user_history=[{"id": i} for i in range(5)],
        intent="billing",
    )
    assert with_conflict.score < no_conflict.score
    assert with_conflict.conflict_detected is True


# --------------------------------------------------------------------------
# Unit tests: routing policy
# --------------------------------------------------------------------------


def test_routing_auto_send_above_threshold() -> None:
    decision = routing_policy.decide(confidence_score=0.9, urgency="normal", has_grounded_source=True)
    assert decision.decision == "auto_send"


def test_routing_pending_validation_mid_range() -> None:
    decision = routing_policy.decide(confidence_score=0.5, urgency="normal", has_grounded_source=True)
    assert decision.decision == "pending_validation"


def test_routing_escalates_without_grounded_source() -> None:
    decision = routing_policy.decide(confidence_score=0.95, urgency="normal", has_grounded_source=False)
    assert decision.decision == "escalated"


def test_routing_escalates_urgent_even_if_confident() -> None:
    decision = routing_policy.decide(confidence_score=0.95, urgency="urgent", has_grounded_source=True)
    assert decision.decision == "escalated"
    assert decision.priority == "urgent"


# --------------------------------------------------------------------------
# Unit tests: classifier
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_intent_billing_keywords() -> None:
    result = await classifier.classify_intent("I was charged twice for my order, can I get a refund?")
    assert result.intent == "billing"
    assert 0.0 < result.confidence <= 1.0


def test_is_ambiguous_short_low_confidence() -> None:
    assert classifier.is_ambiguous("help", intent_confidence=0.2, threshold=0.35) is True


def test_is_ambiguous_false_when_confident() -> None:
    assert classifier.is_ambiguous("refund", intent_confidence=0.9, threshold=0.35) is False


# --------------------------------------------------------------------------
# Integration tests: /support/chat
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_requires_auth(async_client: AsyncClient) -> None:
    response = await async_client.post("/support/chat", json={"message": "hello"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_auto_send_on_strong_kb_match(async_client: AsyncClient, user_token: str) -> None:
    response = await async_client.post(
        "/support/chat",
        json={"message": "How do I reset my password? I forgot my login."},
        headers={"Authorization": user_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "auto_send"
    assert data["confidence_score"] >= 0.75
    assert data["intent"] == "technical"
    assert "password" in data["answer"].lower()
    assert data["ticket_id"] is None
    assert any(s["type"] == "knowledge_base" for s in data["sources"])


@pytest.mark.asyncio
async def test_chat_clarification_for_vague_message(async_client: AsyncClient, user_token: str) -> None:
    response = await async_client.post(
        "/support/chat", json={"message": "help"}, headers={"Authorization": user_token}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "clarification_requested"
    assert data["confidence_score"] == 0.0
    assert data["ticket_id"] is None


@pytest.mark.asyncio
async def test_chat_escalates_when_no_kb_match(async_client: AsyncClient, user_token: str) -> None:
    response = await async_client.post(
        "/support/chat",
        json={"message": "zzyzx quombat flibber nonsense unrelated gibberish text here"},
        headers={"Authorization": user_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "escalated"
    assert data["ticket_id"] is not None


@pytest.mark.asyncio
async def test_chat_persists_conversation_and_can_be_fetched(
    async_client: AsyncClient, user_token: str
) -> None:
    first = await async_client.post(
        "/support/chat",
        json={"message": "How can I track my delivery?"},
        headers={"Authorization": user_token},
    )
    conversation_id = first.json()["conversation_id"]

    second = await async_client.post(
        "/support/chat",
        json={"message": "It still hasn't arrived, what should I do?", "conversation_id": conversation_id},
        headers={"Authorization": user_token},
    )
    assert second.json()["conversation_id"] == conversation_id

    history = await async_client.get(
        f"/support/conversations/{conversation_id}", headers={"Authorization": user_token}
    )
    assert history.status_code == 200
    messages = history.json()["messages"]
    # 2 user turns + 2 agent turns
    assert len(messages) == 4


# --------------------------------------------------------------------------
# Integration tests: staff-only endpoints
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_staff_cannot_list_tickets(async_client: AsyncClient, user_token: str) -> None:
    response = await async_client.get("/support/tickets", headers={"Authorization": user_token})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_staff_can_list_and_resolve_tickets(
    async_client: AsyncClient, user_token: str, staff_token: str
) -> None:
    # Generate an escalated ticket first.
    await async_client.post(
        "/support/chat",
        json={"message": "asdkjh qweoiu zznonsense gibberish escalation trigger text"},
        headers={"Authorization": user_token},
    )

    tickets_resp = await async_client.get(
        "/support/tickets", params={"status": "escalated"}, headers={"Authorization": staff_token}
    )
    assert tickets_resp.status_code == 200
    tickets = tickets_resp.json()
    assert len(tickets) >= 1
    ticket_id = tickets[0]["id"]

    resolve_resp = await async_client.post(
        f"/support/tickets/{ticket_id}/resolve",
        json={"action": "resolve", "final_answer": "Handled manually by a human agent."},
        headers={"Authorization": staff_token},
    )
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["status"] == "resolved"


@pytest.mark.asyncio
async def test_staff_can_create_and_list_kb_entries(async_client: AsyncClient, staff_token: str) -> None:
    create_resp = await async_client.post(
        "/support/kb",
        json={
            "title": "Test entry",
            "content": "This is a test knowledge base entry about warranty claims.",
            "category": "billing",
            "trust_score": 0.8,
            "verified": True,
        },
        headers={"Authorization": staff_token},
    )
    assert create_resp.status_code == 201

    list_resp = await async_client.get("/support/kb", headers={"Authorization": staff_token})
    assert list_resp.status_code == 200
    assert any(e["title"] == "Test entry" for e in list_resp.json())


@pytest.mark.asyncio
async def test_non_staff_cannot_create_kb_entry(async_client: AsyncClient, user_token: str) -> None:
    response = await async_client.post(
        "/support/kb",
        json={"title": "x", "content": "y", "category": "general"},
        headers={"Authorization": user_token},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_staff_can_view_decision_logs(async_client: AsyncClient, user_token: str, staff_token: str) -> None:
    await async_client.post(
        "/support/chat",
        json={"message": "How do I get a refund for my order?"},
        headers={"Authorization": user_token},
    )
    logs_resp = await async_client.get("/support/decision-logs", headers={"Authorization": staff_token})
    assert logs_resp.status_code == 200
    assert len(logs_resp.json()) >= 1


# --------------------------------------------------------------------------
# Context enrichment (external source), fully mocked - no real network call
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_enrichment_from_external_source(
    async_client: AsyncClient, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = User(
        username="linked_user",
        hashed_password=get_password_hash("pw123456"),
        external_user_id=1,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token = f"Bearer {create_access_token('linked_user', user.id)}"

    async def fake_profile(external_user_id: int) -> Dict[str, Any]:
        return {"id": external_user_id, "name": "Jane Doe", "email": "jane@example.com"}

    async def fake_history(external_user_id: int) -> List[Dict[str, Any]]:
        return [{"id": 1, "title": "past thread"}]

    monkeypatch.setattr("app.services.context.external_sources.get_user_profile", fake_profile)
    monkeypatch.setattr("app.services.context.external_sources.get_user_history", fake_history)

    response = await async_client.post(
        "/support/chat",
        json={"message": "How do I track my shipping package delivery?"},
        headers={"Authorization": token},
    )
    assert response.status_code == 200
    data = response.json()
    assert any(s["type"] == "user_profile" for s in data["sources"])
    assert any(s["type"] == "order_history" for s in data["sources"])
