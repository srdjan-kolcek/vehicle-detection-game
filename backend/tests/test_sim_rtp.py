from decimal import Decimal

import pytest

from app.payout import TargetRtp, build_table
from app.sim.rtp import CitySetup, gate, simulate


def _setup(counts: list[int], line: int, stake: int = 100) -> CitySetup:
    row = build_table(counts, line, TargetRtp())
    return CitySetup("test", list(range(1, len(counts) + 1)), counts, line,
                     {b: row[f"m_{b}"] for b in ("under", "exact", "over")}, stake)


SMALL = _setup([1, 2, 3], 2)  # K = 0: every round is an independent draw
DEMO = _setup([17, 19, 20, 22, 22, 24, 25, 25, 26, 31, 41, 57], 24)  # Belgrade demo pool, K = 3


def test_exact_rtp_uses_the_floored_payout():
    assert DEMO.multipliers == {"under": Decimal("2.16"), "exact": Decimal("9.6"), "over": Decimal("1.8")}
    assert _setup([1, 2, 3], 2, stake=1).multipliers["under"] == Decimal("2.7")
    small = simulate(_setup([1, 2, 3], 2, stake=1), 10, "x")
    assert small.exact_rtp["under"] == 2 / 3  # floor(1 x 2.70) = 2 minor units


def test_same_seed_same_result():
    assert simulate(DEMO, 5000, "a") == simulate(DEMO, 5000, "a")
    assert simulate(DEMO, 5000, "a") != simulate(DEMO, 5000, "b")


def test_fixed_bets_match_the_exact_rtp():
    report = simulate(DEMO, 200_000, "fixed")
    for bet in ("under", "exact", "over"):
        assert abs(report.strategies[bet].rtp - report.exact_rtp[bet]) < 0.01
    assert report.clip_share_deviation < 0.05


def test_independent_draws_pass_the_gate():
    report = simulate(SMALL, 200_000, "small")
    assert report.best_round_rtp == pytest.approx(max(report.exact_rtp.values()))  # nothing to learn from history
    assert gate(report, TargetRtp(), 0.01) == []


def test_a_player_who_tracks_the_cooldown_beats_the_demo_tables():
    """With K = 3 of 12 clips, the last results tell which clips cannot come up next."""
    report = simulate(DEMO, 50_000, "informed")
    assert report.strategies["informed"].rtp > 1.0
    assert report.best_round_rtp > 1.1
    problems = gate(report, TargetRtp(), 0.01)
    assert any("tracks the cooldown" in p for p in problems)
    assert any("some rounds" in p for p in problems)


def test_gate_flags_a_fixed_bet_off_its_target():
    report = simulate(SMALL, 20_000, "off")
    problems = gate(report, TargetRtp(under_over=0.80, exact=0.80), 0.01)
    assert any(p.startswith("under RTP") for p in problems)


def test_lucky_sessions_are_reported():
    report = simulate(SMALL, 20_000, "sessions", session_rounds=100)
    assert report.strategies["exact"].session_rtp_p99 > report.strategies["exact"].rtp
