"""
Support agent routes.

Auth: reuses the template's existing JWT auth (AuthUserDep). Staff-only
endpoints additionally require ``is_staff=True`` (StaffUserDep).
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from app.api.deps import AuthUserDep, DBSessionDep, StaffUserDep
from app.models.support import AgentDecisionLog, Conversation, KnowledgeBaseEntry, SupportTicket
from app.schemas.support import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    DecisionLogOut,
    KBEntryCreate,
    KBEntryOut,
    TicketOut,
    TicketResolve,
)
from app.services.agent import handle_message
from app.services.metrics import build_metrics_summary
from app.services.tickets import resolve_ticket

router = APIRouter()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: DBSessionDep, user: AuthUserDep) -> ChatResponse:
    """Send a message to the support agent and get a routed response."""
    return await handle_message(
        db, user=user, message_text=payload.message, conversation_id=payload.conversation_id
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: int, db: DBSessionDep, user: AuthUserDep
) -> Conversation:
    result = await db.execute(
        select(Conversation).filter(
            Conversation.id == conversation_id, Conversation.user_id == user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    # Explicitly load messages (lazy loading is not awaited-safe with async ORM).
    await db.refresh(conversation, attribute_names=["messages"])
    return conversation


# --------------------------------------------------------------------------
# Knowledge base (staff-managed)
# --------------------------------------------------------------------------


@router.get("/kb", response_model=List[KBEntryOut])
async def list_kb_entries(
    db: DBSessionDep,
    user: AuthUserDep,
    category: Optional[str] = Query(default=None),
) -> List[KnowledgeBaseEntry]:
    stmt = select(KnowledgeBaseEntry)
    if category:
        stmt = stmt.filter(KnowledgeBaseEntry.category == category)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.post("/kb", response_model=KBEntryOut, status_code=status.HTTP_201_CREATED)
async def create_kb_entry(
    payload: KBEntryCreate, db: DBSessionDep, staff: StaffUserDep
) -> KnowledgeBaseEntry:
    entry = KnowledgeBaseEntry(**payload.model_dump())
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


# --------------------------------------------------------------------------
# Tickets (staff-managed - internal validation & escalation queue)
# --------------------------------------------------------------------------


@router.get("/tickets", response_model=List[TicketOut])
async def list_tickets(
    db: DBSessionDep,
    staff: StaffUserDep,
    status_filter: Optional[str] = Query(default=None, alias="status"),
) -> List[SupportTicket]:
    stmt = select(SupportTicket)
    if status_filter:
        stmt = stmt.filter(SupportTicket.status == status_filter)
    stmt = stmt.order_by(SupportTicket.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.post("/tickets/{ticket_id}/resolve", response_model=TicketOut)
async def resolve_ticket_endpoint(
    ticket_id: int, payload: TicketResolve, db: DBSessionDep, staff: StaffUserDep
) -> SupportTicket:
    result = await db.execute(select(SupportTicket).filter(SupportTicket.id == ticket_id))
    ticket = result.scalar_one_or_none()
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return await resolve_ticket(db, ticket, payload, staff_user_id=staff.id)


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------


@router.get("/decision-logs", response_model=List[DecisionLogOut])
async def list_decision_logs(
    db: DBSessionDep,
    staff: StaffUserDep,
    conversation_id: Optional[int] = Query(default=None),
) -> List[AgentDecisionLog]:
    stmt = select(AgentDecisionLog)
    if conversation_id:
        stmt = stmt.filter(AgentDecisionLog.conversation_id == conversation_id)
    stmt = stmt.order_by(AgentDecisionLog.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


# --------------------------------------------------------------------------
# Metrics (staff dashboard)
# --------------------------------------------------------------------------


@router.get("/metrics/summary")
async def metrics_summary(db: DBSessionDep, staff: StaffUserDep) -> dict:
    """
    Aggregated live production stats (decision distribution, auto-send /
    escalation rates, avg confidence by intent, avg faithfulness, ticket
    resolution rate) plus the most recent offline eval-harness run, if one
    has been run (see eval/run_eval.py). Consumed by the static dashboard
    at /static/dashboard.html.
    """
    return await build_metrics_summary(db)
