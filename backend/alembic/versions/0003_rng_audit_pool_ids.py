"""rng_audit stores the whole pool

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-14 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rows written before this revision get an empty pool and cannot be replayed. Adding a column
    # with a default does not fire the append-only UPDATE trigger.
    op.add_column(
        'rng_audit',
        sa.Column('pool_ids', postgresql.ARRAY(sa.Integer()), nullable=False, server_default='{}'),
    )
    op.alter_column('rng_audit', 'pool_ids', server_default=None)


def downgrade() -> None:
    op.drop_column('rng_audit', 'pool_ids')
