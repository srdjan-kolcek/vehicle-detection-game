import uuid

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.models import Base
from tests.conftest import BACKEND_DIR

EXPECTED_TABLES = {
    "players",
    "wallet_accounts",
    "ledger_entries",
    "cities",
    "clips",
    "payout_tables",
    "rounds",
    "bets",
    "rng_audit",
    "app_config",
}


def _seed_round(conn) -> tuple[int, uuid.UUID]:
    """Insert one city, clip, payout table, round and player; return (round_id, player_id)."""
    player_id = uuid.uuid4()
    conn.execute(
        text("INSERT INTO players (id, username) VALUES (:id, :u)"),
        {"id": player_id, "u": f"p-{player_id}"},
    )
    city_id = conn.execute(
        text(
            "INSERT INTO cities (slug, name_key, lat, lng, theme_key, line_l, min_stake, max_stake) "
            "VALUES ('las_vegas', 'city.las_vegas', 36.1, -115.1, 'vegas', 12, 100, 10000) RETURNING id"
        )
    ).scalar_one()
    clip_id = conn.execute(
        text(
            "INSERT INTO clips (city_id, angle_id, file, timeline_file, tracks_file, duration_s, count, "
            "sha256, timeline_sha256, tracks_sha256, signature, source_dataset, licence, "
            "weights_sha256, pipeline_version) "
            "VALUES (:c, 'a1', 'f.mp4', 't.json', 'k.json', 30, 12, 'x', 'x', 'x', 's', 'test', 'test', 'x', '0') "
            "RETURNING id"
        ),
        {"c": city_id},
    ).scalar_one()
    table_id = conn.execute(
        text(
            "INSERT INTO payout_tables (city_id, version, p_under, p_exact, p_over, m_under, m_exact, "
            "m_over, rtp_under, rtp_exact, rtp_over, active) "
            "VALUES (:c, 1, .44, .12, .44, 2.11, 7.75, 2.11, .93, .93, .93, true) RETURNING id"
        ),
        {"c": city_id},
    ).scalar_one()
    round_id = conn.execute(
        text(
            "INSERT INTO rounds (city_id, clip_id, status, payout_table_id, betting_opens_at, locks_at, "
            "playback_starts_at) VALUES (:c, :clip, 'BETTING', :t, now(), now(), now()) RETURNING id"
        ),
        {"c": city_id, "clip": clip_id, "t": table_id},
    ).scalar_one()
    return round_id, player_id


def _insert_bet(conn, round_id: int, player_id: uuid.UUID, choice: str = "under") -> uuid.UUID:
    bet_id = uuid.uuid4()
    conn.execute(
        text(
            "INSERT INTO bets (id, round_id, player_id, choice, stake, multiplier) "
            "VALUES (:id, :r, :p, :choice, 100, 2.11)"
        ),
        {"id": bet_id, "r": round_id, "p": player_id, "choice": choice},
    )
    return bet_id


def test_all_tables_created(migrated_engine):
    assert EXPECTED_TABLES <= set(inspect(migrated_engine).get_table_names())


def test_models_match_migrations(migrated_engine):
    with migrated_engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(ctx, Base.metadata) == []


def test_downgrade_removes_everything(migrated_engine):
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.downgrade(cfg, "base")
    assert EXPECTED_TABLES.isdisjoint(inspect(migrated_engine).get_table_names())


def test_one_bet_per_player_per_round(migrated_engine):
    with migrated_engine.begin() as conn:
        round_id, player_id = _seed_round(conn)
        _insert_bet(conn, round_id, player_id, "under")
    with pytest.raises(IntegrityError, match="uq_bets_round_id_player_id"):
        with migrated_engine.begin() as conn:
            _insert_bet(conn, round_id, player_id, "over")


def test_bet_choice_and_stake_are_validated(migrated_engine):
    with migrated_engine.begin() as conn:
        round_id, player_id = _seed_round(conn)
    with pytest.raises(IntegrityError, match="choice_valid"):
        with migrated_engine.begin() as conn:
            _insert_bet(conn, round_id, player_id, "draw")


def test_only_one_active_payout_table_per_city(migrated_engine):
    with migrated_engine.begin() as conn:
        _seed_round(conn)
    with pytest.raises(IntegrityError, match="uq_payout_tables_city_id_active"):
        with migrated_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO payout_tables (city_id, version, p_under, p_exact, p_over, m_under, "
                    "m_exact, m_over, rtp_under, rtp_exact, rtp_over, active) "
                    "SELECT city_id, 2, .4, .2, .4, 2.3, 4.6, 2.3, .93, .93, .93, true FROM payout_tables"
                )
            )


def test_ledger_is_idempotent_per_bet_and_type(migrated_engine):
    with migrated_engine.begin() as conn:
        round_id, player_id = _seed_round(conn)
        bet_id = _insert_bet(conn, round_id, player_id)
        conn.execute(
            text(
                "INSERT INTO ledger_entries (player_id, bet_id, type, amount, balance_after) "
                "VALUES (:p, :b, 'bet', -100, 900)"
            ),
            {"p": player_id, "b": bet_id},
        )
    with pytest.raises(IntegrityError, match="uq_ledger_entries_bet_id_type"):
        with migrated_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO ledger_entries (player_id, bet_id, type, amount, balance_after) "
                    "VALUES (:p, :b, 'bet', -100, 800)"
                ),
                {"p": player_id, "b": bet_id},
            )


def test_wallet_balance_cannot_go_negative(migrated_engine):
    with migrated_engine.begin() as conn:
        _, player_id = _seed_round(conn)
        conn.execute(
            text("INSERT INTO wallet_accounts (player_id, balance) VALUES (:p, 100)"),
            {"p": player_id},
        )
    with pytest.raises(IntegrityError, match="balance_non_negative"):
        with migrated_engine.begin() as conn:
            conn.execute(text("UPDATE wallet_accounts SET balance = -1"))


@pytest.mark.parametrize("statement", ["UPDATE ledger_entries SET amount = 0", "DELETE FROM ledger_entries"])
def test_ledger_is_append_only(migrated_engine, statement):
    with migrated_engine.begin() as conn:
        _, player_id = _seed_round(conn)
        conn.execute(
            text(
                "INSERT INTO ledger_entries (player_id, type, amount, balance_after) "
                "VALUES (:p, 'topup', 100000, 100000)"
            ),
            {"p": player_id},
        )
    with pytest.raises(DBAPIError, match="append-only"):
        with migrated_engine.begin() as conn:
            conn.execute(text(statement))


@pytest.mark.parametrize("statement", ["UPDATE rng_audit SET pool_size = 0", "DELETE FROM rng_audit"])
def test_rng_audit_is_append_only(migrated_engine, statement):
    with migrated_engine.begin() as conn:
        round_id, _ = _seed_round(conn)
        conn.execute(
            text(
                "INSERT INTO rng_audit (round_id, city_id, pool_size, cooldown_ids, chosen_clip_id, raw_random) "
                "SELECT id, city_id, 35, ARRAY[]::integer[], clip_id, '42' FROM rounds WHERE id = :r"
            ),
            {"r": round_id},
        )
    with pytest.raises(DBAPIError, match="append-only"):
        with migrated_engine.begin() as conn:
            conn.execute(text(statement))
