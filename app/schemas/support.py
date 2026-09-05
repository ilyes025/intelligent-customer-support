"""
Pydantic schemas for the intelligent support agent domain.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """A message sent by a user to the support agent."""

    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[int] = None


class SourceRef(BaseModel):
    """A single source that contributed to the agent's answer."""

    type: str  # "knowledge_base" | "order_history" | "user_profile"
    reference: str
    relevance: Optional[float] = None
    trust_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response returned by the support agent for a single chat turn."""

    conversation_id: int
    message_id: int
    answer: str
    intent: str
    intent_confidence: float
    confidence_score: float
    decision: str  # auto_send | pending_validation | escalated | clarification_requested
    sources: List[SourceRef] = []
    ticket_id: Optional[int] = None
    faithfulness_score: Optional[float] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationOut(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    messages: List[MessageOut] = []

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------


class KBEntryCreate(BaseModel):
    title: str
    content: str
    category: str
    tags: Optional[str] = None
    source: str = "internal"
    trust_score: float = Field(0.5, ge=0.0, le=1.0)
    verified: bool = False


class KBEntryOut(BaseModel):
    id: int
    title: str
    content: str
    category: str
    tags: Optional[str] = None
    source: str
    trust_score: float
    verified: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# Tickets
# --------------------------------------------------------------------------


class TicketOut(BaseModel):
    id: int
    conversation_id: int
    status: str
    priority: str
    reason: str
    proposed_answer: Optional[str] = None
    confidence_score: Optional[float] = None
    assigned_to: Optional[int] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TicketResolve(BaseModel):
    action: str = Field(..., pattern="^(approve|reject|resolve)$")
    resolution_note: Optional[str] = None
    final_answer: Optional[str] = None


# --------------------------------------------------------------------------
# Decision log (audit)
# --------------------------------------------------------------------------


class DecisionLogOut(BaseModel):
    id: int
    conversation_id: int
    message_id: Optional[int] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    confidence_score: float
    decision: str
    sources_used: Optional[str] = None
    faithfulness_score: Optional[float] = None
    retrieval_mode: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
