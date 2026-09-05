"""
Startup seeding: populates the knowledge base with curated FAQ entries the
first time the app starts against an empty table. This keeps the demo
self-contained (no manual data-entry step needed to try the API) while
staying a no-op on subsequent restarts.
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.seed_data import KB_SEED_ENTRIES
from app.models.support import KnowledgeBaseEntry


async def seed_knowledge_base(db: AsyncSession) -> None:
    result = await db.execute(select(func.count()).select_from(KnowledgeBaseEntry))
    count = result.scalar_one()
    if count and count > 0:
        return

    for entry_data in KB_SEED_ENTRIES:
        db.add(KnowledgeBaseEntry(**entry_data))
    await db.commit()
