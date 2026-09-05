"""add hybrid retrieval embedding cache and faithfulness audit columns

Revision ID: f1a2b3c4d5e6
Revises: bd3ac5e4056a
Create Date: 2026-09-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'bd3ac5e4056a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Dense-retrieval embedding cache (JSON-encoded vector), lazily populated
    # the first time hybrid search embeds a given KB entry. NULL for every
    # existing row until then - no backfill required.
    op.add_column(
        'knowledge_base_entries',
        sa.Column('embedding', sa.Text(), nullable=True),
    )

    # Audit trail additions: what the faithfulness check scored the
    # generated answer, and which retrieval mode produced the sources used.
    op.add_column(
        'agent_decision_logs',
        sa.Column('faithfulness_score', sa.Float(), nullable=True),
    )
    op.add_column(
        'agent_decision_logs',
        sa.Column('retrieval_mode', sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('agent_decision_logs', 'retrieval_mode')
    op.drop_column('agent_decision_logs', 'faithfulness_score')
    op.drop_column('knowledge_base_entries', 'embedding')
