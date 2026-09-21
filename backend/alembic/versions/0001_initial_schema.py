"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-09-21 16:18:05.497845
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('app_config',
    sa.Column('key', sa.String(length=64), nullable=False),
    sa.Column('value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.PrimaryKeyConstraint('key', name=op.f('pk_app_config'))
    )
    op.create_table('cities',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('slug', sa.String(length=64), nullable=False),
    sa.Column('name_key', sa.String(length=128), nullable=False),
    sa.Column('lat', sa.Float(), nullable=False),
    sa.Column('lng', sa.Float(), nullable=False),
    sa.Column('theme_key', sa.String(length=64), nullable=False),
    sa.Column('active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('line_l', sa.Integer(), nullable=False),
    sa.Column('min_stake', sa.BigInteger(), nullable=False),
    sa.Column('max_stake', sa.BigInteger(), nullable=False),
    sa.CheckConstraint('min_stake > 0 AND max_stake >= min_stake', name=op.f('ck_cities_stake_range_valid')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_cities')),
    sa.UniqueConstraint('slug', name=op.f('uq_cities_slug'))
    )
    op.create_table('players',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('username', sa.String(length=64), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=True),
    sa.Column('locale', sa.String(length=8), server_default='en', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_players')),
    sa.UniqueConstraint('username', name=op.f('uq_players_username'))
    )
    op.create_table('clips',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('city_id', sa.Integer(), nullable=False),
    sa.Column('angle_id', sa.String(length=64), nullable=False),
    sa.Column('file', sa.String(length=512), nullable=False),
    sa.Column('timeline_file', sa.String(length=512), nullable=False),
    sa.Column('tracks_file', sa.String(length=512), nullable=False),
    sa.Column('duration_s', sa.Float(), nullable=False),
    sa.Column('count', sa.Integer(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('timeline_sha256', sa.String(length=64), nullable=False),
    sa.Column('tracks_sha256', sa.String(length=64), nullable=False),
    sa.Column('signature', sa.String(length=128), nullable=False),
    sa.Column('source_dataset', sa.String(length=128), nullable=False),
    sa.Column('licence', sa.String(length=128), nullable=False),
    sa.Column('weights_sha256', sa.String(length=64), nullable=False),
    sa.Column('pipeline_version', sa.String(length=64), nullable=False),
    sa.Column('verified', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.CheckConstraint('count >= 0', name=op.f('ck_clips_count_non_negative')),
    sa.ForeignKeyConstraint(['city_id'], ['cities.id'], name=op.f('fk_clips_city_id_cities')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_clips')),
    sa.UniqueConstraint('city_id', 'file', name='uq_clips_city_id_file')
    )
    op.create_index('ix_clips_city_id_verified', 'clips', ['city_id', 'verified'], unique=False)
    op.create_table('payout_tables',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('city_id', sa.Integer(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('p_under', sa.Numeric(precision=10, scale=8), nullable=False),
    sa.Column('p_exact', sa.Numeric(precision=10, scale=8), nullable=False),
    sa.Column('p_over', sa.Numeric(precision=10, scale=8), nullable=False),
    sa.Column('m_under', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('m_exact', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('m_over', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('rtp_under', sa.Numeric(precision=10, scale=8), nullable=False),
    sa.Column('rtp_exact', sa.Numeric(precision=10, scale=8), nullable=False),
    sa.Column('rtp_over', sa.Numeric(precision=10, scale=8), nullable=False),
    sa.Column('active', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.ForeignKeyConstraint(['city_id'], ['cities.id'], name=op.f('fk_payout_tables_city_id_cities')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_payout_tables')),
    sa.UniqueConstraint('city_id', 'version', name='uq_payout_tables_city_id_version')
    )
    op.create_index('uq_payout_tables_city_id_active', 'payout_tables', ['city_id'], unique=True, postgresql_where=sa.text('active'))
    op.create_table('wallet_accounts',
    sa.Column('player_id', sa.UUID(), nullable=False),
    sa.Column('balance', sa.BigInteger(), server_default='0', nullable=False),
    sa.CheckConstraint('balance >= 0', name=op.f('ck_wallet_accounts_balance_non_negative')),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], name=op.f('fk_wallet_accounts_player_id_players')),
    sa.PrimaryKeyConstraint('player_id', name=op.f('pk_wallet_accounts'))
    )
    op.create_table('rounds',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('city_id', sa.Integer(), nullable=False),
    sa.Column('clip_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('payout_table_id', sa.Integer(), nullable=False),
    sa.Column('betting_opens_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('locks_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('playback_starts_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('settled_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('result_count', sa.Integer(), nullable=True),
    sa.Column('result_outcome', sa.String(length=8), nullable=True),
    sa.CheckConstraint("result_outcome IS NULL OR result_outcome IN ('under','exact','over')", name=op.f('ck_rounds_result_outcome_valid')),
    sa.CheckConstraint("status IN ('BETTING','LOCKED','PLAYBACK','SETTLED','VOIDED')", name=op.f('ck_rounds_status_valid')),
    sa.ForeignKeyConstraint(['city_id'], ['cities.id'], name=op.f('fk_rounds_city_id_cities')),
    sa.ForeignKeyConstraint(['clip_id'], ['clips.id'], name=op.f('fk_rounds_clip_id_clips')),
    sa.ForeignKeyConstraint(['payout_table_id'], ['payout_tables.id'], name=op.f('fk_rounds_payout_table_id_payout_tables')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_rounds'))
    )
    op.create_index('ix_rounds_city_id_status', 'rounds', ['city_id', 'status'], unique=False)
    op.create_table('bets',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('round_id', sa.BigInteger(), nullable=False),
    sa.Column('player_id', sa.UUID(), nullable=False),
    sa.Column('choice', sa.String(length=8), nullable=False),
    sa.Column('stake', sa.BigInteger(), nullable=False),
    sa.Column('multiplier', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('status', sa.String(length=16), server_default='open', nullable=False),
    sa.Column('payout', sa.BigInteger(), server_default='0', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("choice IN ('under','exact','over')", name=op.f('ck_bets_choice_valid')),
    sa.CheckConstraint("status IN ('open','won','lost','refunded')", name=op.f('ck_bets_status_valid')),
    sa.CheckConstraint('payout >= 0', name=op.f('ck_bets_payout_non_negative')),
    sa.CheckConstraint('stake > 0', name=op.f('ck_bets_stake_positive')),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], name=op.f('fk_bets_player_id_players')),
    sa.ForeignKeyConstraint(['round_id'], ['rounds.id'], name=op.f('fk_bets_round_id_rounds')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_bets')),
    sa.UniqueConstraint('round_id', 'player_id', name='uq_bets_round_id_player_id')
    )
    op.create_index('ix_bets_player_id_created_at', 'bets', ['player_id', 'created_at'], unique=False)
    op.create_index('ix_bets_round_id', 'bets', ['round_id'], unique=False)
    op.create_table('rng_audit',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('round_id', sa.BigInteger(), nullable=False),
    sa.Column('city_id', sa.Integer(), nullable=False),
    sa.Column('pool_size', sa.Integer(), nullable=False),
    sa.Column('cooldown_ids', postgresql.ARRAY(sa.Integer()), nullable=False),
    sa.Column('chosen_clip_id', sa.Integer(), nullable=False),
    sa.Column('raw_random', sa.String(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['chosen_clip_id'], ['clips.id'], name=op.f('fk_rng_audit_chosen_clip_id_clips')),
    sa.ForeignKeyConstraint(['city_id'], ['cities.id'], name=op.f('fk_rng_audit_city_id_cities')),
    sa.ForeignKeyConstraint(['round_id'], ['rounds.id'], name=op.f('fk_rng_audit_round_id_rounds')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_rng_audit'))
    )
    op.create_index('ix_rng_audit_round_id', 'rng_audit', ['round_id'], unique=False)
    op.create_table('ledger_entries',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('player_id', sa.UUID(), nullable=False),
    sa.Column('bet_id', sa.UUID(), nullable=True),
    sa.Column('type', sa.String(length=16), nullable=False),
    sa.Column('amount', sa.BigInteger(), nullable=False),
    sa.Column('balance_after', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("type IN ('bet','payout','refund','topup')", name=op.f('ck_ledger_entries_type_valid')),
    sa.CheckConstraint('balance_after >= 0', name=op.f('ck_ledger_entries_balance_after_non_negative')),
    sa.ForeignKeyConstraint(['bet_id'], ['bets.id'], name=op.f('fk_ledger_entries_bet_id_bets')),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], name=op.f('fk_ledger_entries_player_id_players')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ledger_entries'))
    )
    op.create_index('ix_ledger_entries_player_id_created_at', 'ledger_entries', ['player_id', 'created_at'], unique=False)
    op.create_index('uq_ledger_entries_bet_id_type', 'ledger_entries', ['bet_id', 'type'], unique=True, postgresql_where=sa.text('bet_id IS NOT NULL'))

    # Ledger and RNG audit rows are append-only.
    op.execute(
        """
        CREATE FUNCTION reject_modification() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    for table in ('ledger_entries', 'rng_audit'):
        op.execute(
            f"""
            CREATE TRIGGER {table}_append_only
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION reject_modification()
            """
        )


def downgrade() -> None:
    op.execute('DROP TRIGGER rng_audit_append_only ON rng_audit')
    op.execute('DROP TRIGGER ledger_entries_append_only ON ledger_entries')
    op.execute('DROP FUNCTION reject_modification()')
    op.drop_index('uq_ledger_entries_bet_id_type', table_name='ledger_entries', postgresql_where=sa.text('bet_id IS NOT NULL'))
    op.drop_index('ix_ledger_entries_player_id_created_at', table_name='ledger_entries')
    op.drop_table('ledger_entries')
    op.drop_index('ix_rng_audit_round_id', table_name='rng_audit')
    op.drop_table('rng_audit')
    op.drop_index('ix_bets_round_id', table_name='bets')
    op.drop_index('ix_bets_player_id_created_at', table_name='bets')
    op.drop_table('bets')
    op.drop_index('ix_rounds_city_id_status', table_name='rounds')
    op.drop_table('rounds')
    op.drop_table('wallet_accounts')
    op.drop_index('uq_payout_tables_city_id_active', table_name='payout_tables', postgresql_where=sa.text('active'))
    op.drop_table('payout_tables')
    op.drop_index('ix_clips_city_id_verified', table_name='clips')
    op.drop_table('clips')
    op.drop_table('players')
    op.drop_table('cities')
    op.drop_table('app_config')
