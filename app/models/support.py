"""
SQLAlchemy models for the intelligent support agent domain.

Table overview
--------------
- Conversation: groups the turns of one support session for a user.
- Message: a single user/agent turn inside a conversation.
- KnowledgeBaseEntry: curated, trust-scored source material. Only entries
  with ``verified=True`` and a sufficiently high ``trust_score`` may back
  an auto-sent answer (see app.services.confidence).
- SupportTicket: created whenever a request is escalated to a human agent
  or queued for internal validation.
- AgentDecisionLog: full audit trail of every routing decision the agent
  made, for explainability, QA and later fine-tuning of thresholds.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    messages = relationship(
        "Message", back_populates="conversation", order_by="Message.created_at"
    )
    tickets = relationship("SupportTicket", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False, index=True
    )
    role = Column(String, nullable=False)  # "user" | "agent"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class KnowledgeBaseEntry(Base):
    __tablename__ = "knowledge_base_entries"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False, index=True)  # billing/shipping/...
    tags = Column(String, nullable=True)  # comma-separated
    source = Column(String, nullable=False, default="internal")
    trust_score = Column(Float, nullable=False, default=0.5)  # 0..1
    verified = Column(Boolean, nullable=False, default=False)
    # JSON-encoded embedding vector (list[float]), lazily computed on first
    # hybrid/dense retrieval and cached here so we never re-embed the same
    # entry twice. NULL until a dense search actually needs it, or if HF
    # embeddings are disabled (dense retrieval degrades to BM25/TF-IDF only
    # in that case - see app/services/knowledge_base.py).
    embedding = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False, index=True
    )
    status = Column(String, nullable=False, default="open", index=True)
    # open | pending_validation | escalated | resolved
    priority = Column(String, nullable=False, default="normal")
    # low | normal | high | urgent
    reason = Column(String, nullable=False)
    proposed_answer = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_note = Column(Text, nullable=True)

    conversation = relationship("Conversation", back_populates="tickets")


class AgentDecisionLog(Base):
    __tablename__ = "agent_decision_logs"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False, index=True
    )
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    intent = Column(String, nullable=True)
    intent_confidence = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    # auto_send | pending_validation | escalated | clarification_requested
    sources_used = Column(Text, nullable=True)  # JSON-encoded list
    # NLI-based faithfulness score (0..1) of the generated answer against
    # its retrieved context, or NULL when there was no generated answer to
    # check (clarification/escalation-with-no-grounding). See
    # app/services/faithfulness.py. A low score can downgrade the routing
    # decision after the fact - `decision` reflects the FINAL decision,
    # post-downgrade.
    faithfulness_score = Column(Float, nullable=True)
    retrieval_mode = Column(String, nullable=True)  # tfidf | bm25 | hybrid
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
