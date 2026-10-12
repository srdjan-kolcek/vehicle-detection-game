"""one open round per city

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'uq_rounds_city_id_open', 'rounds', ['city_id'], unique=True,
        postgresql_where=sa.text("status IN ('BETTING','LOCKED','PLAYBACK')"),
    )


def downgrade() -> None:
    op.drop_index(
        'uq_rounds_city_id_open', table_name='rounds',
        postgresql_where=sa.text("status IN ('BETTING','LOCKED','PLAYBACK')"),
    )
