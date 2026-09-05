"""
Support agent orchestrator.

Ties together context building, intent classification, knowledge-base
retrieval, confidence scoring and routing into a single pipeline, and
persists the full result (message, decision log, ticket) for audit.

See the module docstrings in app/services/{context,classifier,
knowledge_base,confidence,routing_policy}.py for the reasoning behind each
stage.
"""

import json
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.support import (
    AgentDecisionLog,
    Conversation,
    Message,
    SupportTicket,
)
from app.models.user import User
from app.core.config import settings
from app.schemas.support import ChatResponse, SourceRef
from app.services import classifier, confidence as confidence_engine
from app.services import faithfulness as faithfulness_engine
from app.services import huggingface_client
from app.services import knowledge_base as kb_service
from app.services import routing_policy
from app.services.context import build_context

CLARIFICATION_TEMPLATE = (
    "I want to make sure I help you correctly - could you give me a bit "
    "more detail about your issue? For example, mention the order, "
    "feature, or error message involved."
)

ESCALATION_TEMPLATE = (
    "I wasn't able to find a confirmed answer for this, so I've forwarded "
    "your request to a member of our support team. They will follow up "
    "with you shortly."
)

PENDING_VALIDATION_PREFIX = (
    "Here is what I believe is the answer to your question. A support "
    "agent will double-check this shortly, but in the meantime:\n\n"
)


async def _get_or_create_conversation(
    db: AsyncSession, user: User, conversation_id: Optional[int]
) -> Conversation:
    if conversation_id is not None:
        result = await db.execute(
            select(Conversation).filter(
                Conversation.id == conversation_id, Conversation.user_id == user.id
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation

    conversation = Conversation(user_id=user.id)
    db.add(conversation)
    await db.flush()
    return conversation


def _raw_kb_answer(entries: List[kb_service.RetrievedEntry]) -> str:
    """Concatenate retrieved KB entries verbatim, with no LLM involved."""
    parts = [entry.entry.content.strip() for entry in entries]
    return "\n\n".join(parts)


def _build_rag_prompt(question: str, entries: List[kb_service.RetrievedEntry]) -> str:
    context_block = "\n\n".join(
        f"Source: {e.entry.title}\n{e.entry.content.strip()}" for e in entries
    )
    return (
        "You are a customer support assistant. Answer the user's question "
        "using ONLY the information in the context below. Do not invent "
        "facts, prices, dates, or policies that are not present in the "
        "context. If the context does not fully answer the question, say "
        "so plainly rather than guessing.\n\n"
        f"Context:\n{context_block}\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )


async def _compose_answer(question: str, entries: List[kb_service.RetrievedEntry]) -> str:
    """
    Compose the answer that is actually sent to the user.

    This is only ever called on the ``auto_send`` / ``pending_validation``
    paths, i.e. when a grounded KB source exists. It tries a RAG pass -
    the retrieved entries are given to the LLM as context so it can phrase
    a direct answer - but always falls back to the raw KB text verbatim if
    Hugging Face is disabled or the call fails, so the response is never
    blocked on, or silently degraded by, an external API (Theme 1: minimize
    hallucinations, never a hard dependency on HF).
    """
    fallback = _raw_kb_answer(entries)
    if not entries:
        return fallback

    prompt = _build_rag_prompt(question, entries)
    generated = await huggingface_client.generate_reply(prompt)
    return generated or fallback


def _build_draft_prompt(question: str) -> str:
    return (
        "You are drafting an INTERNAL, UNVERIFIED note for a human support "
        "agent - this text will never be sent to the customer directly. "
        "No knowledge-base article matched their question, so treat any "
        "specifics (prices, dates, policies, order status) as unknown and "
        "say so rather than inventing them. Suggest a short, cautious "
        "starting point the human agent can verify and build on.\n\n"
        f"Customer question: {question}\n\n"
        "Draft note for the human agent:"
    )


async def _generate_draft_answer(question: str) -> Optional[str]:
    """
    Best-effort, ungrounded LLM draft used ONLY as ``SupportTicket.proposed_answer``
    for a human reviewer on the escalation path (no KB match at all).

    This intentionally never reaches ``ChatResponse.answer`` - the user
    still gets ``ESCALATION_TEMPLATE``. ``routing_policy.decide`` always
    escalates when there is no grounded source, and that invariant is not
    touched here; this only gives the human reviewer a head start.
    """
    if not settings.HF_ENABLED:
        return None
    prompt = _build_draft_prompt(question)
    return await huggingface_client.generate_reply(prompt)


async def handle_message(
    db: AsyncSession, user: User, message_text: str, conversation_id: Optional[int]
) -> ChatResponse:
    conversation = await _get_or_create_conversation(db, user, conversation_id)

    user_message = Message(conversation_id=conversation.id, role="user", content=message_text)
    db.add(user_message)
    await db.flush()

    context = await build_context(db, user, conversation)
    intent_result = await classifier.classify_intent(message_text)

    sources: List[SourceRef] = []

    # --- Clarification gate -------------------------------------------------
    if classifier.is_ambiguous(
        message_text, intent_result.confidence, threshold=_clarification_threshold()
    ):
        answer = CLARIFICATION_TEMPLATE
        decision = "clarification_requested"
        agent_message = Message(conversation_id=conversation.id, role="agent", content=answer)
        db.add(agent_message)
        await db.flush()

        log = AgentDecisionLog(
            conversation_id=conversation.id,
            message_id=user_message.id,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            confidence_score=0.0,
            decision=decision,
            sources_used=json.dumps([]),
        )
        db.add(log)
        await db.commit()

        return ChatResponse(
            conversation_id=conversation.id,
            message_id=agent_message.id,
            answer=answer,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            confidence_score=0.0,
            decision=decision,
            sources=[],
            ticket_id=None,
        )

    # --- Retrieval ------------------------------------------------------------
    kb_matches = await kb_service.search_knowledge_base(
        db, query=message_text, category=intent_result.intent
    )
    if not kb_matches:
        # Widen the search across all categories before giving up, in case
        # the classifier mis-labelled the category but the KB still has a
        # relevant, cross-category answer.
        kb_matches = await kb_service.search_knowledge_base(db, query=message_text, category=None)

    top_kb = kb_matches[0] if kb_matches else None

    # --- Confidence -------------------------------------------------------
    breakdown = confidence_engine.compute_confidence(
        top_kb=top_kb,
        intent_confidence=intent_result.confidence,
        user_history=context.external_history,
        intent=intent_result.intent,
    )

    routing = routing_policy.decide(
        confidence_score=breakdown.score,
        urgency=intent_result.urgency,
        has_grounded_source=top_kb is not None,
    )

    for match in kb_matches:
        sources.append(
            SourceRef(
                type="knowledge_base",
                reference=f"kb:{match.entry.id}:{match.entry.title}",
                relevance=match.relevance,
                trust_score=match.entry.trust_score,
            )
        )
    if context.external_profile:
        sources.append(SourceRef(type="user_profile", reference="jsonplaceholder:user"))
    if context.external_history:
        sources.append(
            SourceRef(type="order_history", reference=f"jsonplaceholder:{len(context.external_history)}_items")
        )

    # --- Compose response text based on decision --------------------------
    # Compute the RAG-composed answer once up front (whenever we have any
    # KB grounding at all), then run a faithfulness check against it before
    # committing to a final decision. A generated answer that isn't well
    # supported by its own retrieved context downgrades the decision one
    # notch (auto_send -> pending_validation -> escalated) - this is the
    # "verify, don't just prompt for it" check on top of routing_policy's
    # confidence-only decision.
    final_decision = routing.decision
    faithfulness_score: Optional[float] = None
    composed_answer: Optional[str] = None

    if kb_matches:
        composed_answer = await _compose_answer(message_text, kb_matches)
        context_text = "\n\n".join(m.entry.content.strip() for m in kb_matches)
        result = await faithfulness_engine.check_faithfulness(composed_answer, context_text)
        faithfulness_score = result.score

        if not result.is_faithful and final_decision == "auto_send":
            final_decision = "pending_validation"
        elif not result.is_faithful and final_decision == "pending_validation":
            final_decision = "escalated"

    ticket_id: Optional[int] = None
    if final_decision == "auto_send":
        answer = composed_answer or ESCALATION_TEMPLATE
    elif final_decision == "pending_validation":
        answer = PENDING_VALIDATION_PREFIX + (composed_answer or "")
        reason = f"Low-to-medium confidence ({breakdown.score}) for intent '{intent_result.intent}'."
        if faithfulness_score is not None and faithfulness_score < settings.FAITHFULNESS_THRESHOLD:
            reason += f" Downgraded after a low faithfulness score ({faithfulness_score:.2f})."
        ticket = SupportTicket(
            conversation_id=conversation.id,
            status="pending_validation",
            priority=routing.priority,
            reason=reason,
            proposed_answer=answer,
            confidence_score=breakdown.score,
        )
        db.add(ticket)
        await db.flush()
        ticket_id = ticket.id
    else:  # escalated
        answer = ESCALATION_TEMPLATE
        if top_kb is None:
            reason = "No verified knowledge-base source found."
        elif faithfulness_score is not None and faithfulness_score < settings.FAITHFULNESS_THRESHOLD:
            reason = f"Generated answer failed the faithfulness check ({faithfulness_score:.2f})."
        else:
            reason = f"Confidence too low ({breakdown.score}) or urgent request."

        if kb_matches:
            # Some (weak, or unfaithfully-phrased) grounding exists - reuse
            # the already-composed answer as the human reviewer's starting draft.
            proposed_answer: Optional[str] = composed_answer
        else:
            # No grounding at all - at best an unverified LLM draft for the
            # human reviewer; never shown to the user (see _generate_draft_answer).
            proposed_answer = await _generate_draft_answer(message_text)
        ticket = SupportTicket(
            conversation_id=conversation.id,
            status="escalated",
            priority=routing.priority,
            reason=reason,
            proposed_answer=proposed_answer,
            confidence_score=breakdown.score,
        )
        db.add(ticket)
        await db.flush()
        ticket_id = ticket.id

    agent_message = Message(conversation_id=conversation.id, role="agent", content=answer)
    db.add(agent_message)
    await db.flush()

    log = AgentDecisionLog(
        conversation_id=conversation.id,
        message_id=user_message.id,
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        confidence_score=breakdown.score,
        decision=final_decision,
        sources_used=json.dumps([s.reference for s in sources]),
        faithfulness_score=faithfulness_score,
        retrieval_mode=settings.KB_RETRIEVAL_MODE,
    )
    db.add(log)

    await db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=agent_message.id,
        answer=answer,
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        confidence_score=breakdown.score,
        decision=final_decision,
        sources=sources,
        ticket_id=ticket_id,
        faithfulness_score=faithfulness_score,
    )


def _clarification_threshold() -> float:
    return settings.INTENT_CLARIFICATION_THRESHOLD
