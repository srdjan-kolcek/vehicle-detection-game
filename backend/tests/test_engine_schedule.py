from datetime import datetime, timedelta, timezone

import pytest

from app.payout import outcome_for
from app.engine.schedule import PhaseDurations, schedule_from

START = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _s(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


def test_default_cycle_is_15_5_30_5():
    schedule = schedule_from(START, PhaseDurations())
    assert schedule.betting_opens_at == START
    assert schedule.locks_at == _s(15)
    assert schedule.playback_starts_at == _s(20)
    assert schedule.settles_at == _s(50)
    assert schedule.ends_at == _s(55)


def test_config_overrides_only_the_given_phases():
    durations = PhaseDurations.from_config({"betting": 20, "settle": 2.5})
    assert durations == PhaseDurations(betting=20, locked=5, playback=30, settle=2.5)
    assert schedule_from(START, durations).ends_at == _s(57.5)


@pytest.mark.parametrize(
    "value",
    [[15, 5, 30, 5], {"betting": 0}, {"locked": -1}, {"playback": "30"}, {"settle": True}, {"warmup": 5}],
)
def test_bad_config_is_rejected(value):
    with pytest.raises(ValueError):
        PhaseDurations.from_config(value)


def test_config_round_trips():
    durations = PhaseDurations(betting=10, locked=3, playback=30, settle=4)
    assert PhaseDurations.from_config(durations.as_config()) == durations


@pytest.mark.parametrize(("count", "outcome"), [(0, "under"), (10, "under"), (11, "exact"), (12, "over"), (40, "over")])
def test_outcome_against_line_11(count, outcome):
    assert outcome_for(count, 11) == outcome
