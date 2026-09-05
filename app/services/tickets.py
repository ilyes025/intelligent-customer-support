"""
Support ticket resolution service - the human-in-the-loop half of Theme 1.

Pending-validation tickets hold an answer the agent already sent to the
user provisionally; staff can ``approve`` it (confirms correctness),
``reject`` it (marks it wrong so the KB/thresholds can be reviewed), or
directly ``resolve`` it with a final answer of their own, which is then
appended to the conversation as the authoritative agent message.
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.support import Message, SupportTicket
from app.schemas.support import TicketResolve


async def resolve_ticket(
    db: AsyncSession, ticket: SupportTicket, payload: TicketResolve, staff_user_id: int
) -> SupportTicket:
    ticket.resolved_at = datetime.utcnow()
    ticket.resolution_note = payload.resolution_note
    ticket.assigned_to = staff_user_id

    if payload.action == "approve":
        ticket.status = "resolved"
    elif payload.action == "reject":
        ticket.status = "resolved"
        if payload.final_answer:
            _append_agent_message(db, ticket, payload.final_answer)
    else:  # "resolve" - staff supplies the authoritative answer
        ticket.status = "resolved"
        final_answer = payload.final_answer or ticket.proposed_answer
        if final_answer:
            _append_agent_message(db, ticket, final_answer)

    await db.commit()
    await db.refresh(ticket)
    return ticket


def _append_agent_message(db: AsyncSession, ticket: SupportTicket, content: str) -> None:
    message = Message(
        conversation_id=ticket.conversation_id,
        role="agent",
        content=f"[Staff-reviewed answer] {content}",
    )
    db.add(message)
