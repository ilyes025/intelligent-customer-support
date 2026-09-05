"""
Context builder (Theme 2: "analyze user context: profile, history,
environment").

Combines:
- internal conversation history (this app's own DB - reliable, always
  available)
- external profile/history (JSONPlaceholder - best-effort, degrades to
  empty on failure, see app/services/external_sources.py)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.support import Conversation, Message
from app.models.user import User
from app.services import external_sources


@dataclass
class UserContext:
    internal_history: List[Message] = field(default_factory=list)
    external_profile: Optional[Dict[str, Any]] = None
    external_history: List[Dict[str, Any]] = field(default_factory=list)


async def build_context(db: AsyncSession, user: User, conversation: Conversation) -> UserContext:
    result = await db.execute(
        select(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    internal_history = list(result.scalars().all())

    external_profile = None
    external_history: List[Dict[str, Any]] = []
    if user.external_user_id:
        external_profile = await external_sources.get_user_profile(user.external_user_id)
        external_history = await external_sources.get_user_history(user.external_user_id)

    return UserContext(
        internal_history=internal_history,
        external_profile=external_profile,
        external_history=external_history,
    )
