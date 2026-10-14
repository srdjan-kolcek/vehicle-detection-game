"""Phase durations and round deadlines (doc 03, round lifecycle).

All deadlines of a round are fixed when it opens, so a slow step never pushes the later ones back.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

PHASE_DURATIONS_KEY = "phase_durations"


@dataclass(frozen=True)
class PhaseDurations:
    betting: float = 15
    locked: float = 5
    playback: float = 30
    settle: float = 5

    @classmethod
    def from_config(cls, value: object) -> "PhaseDurations":
        """Build from the app_config value; missing keys keep their defaults. Raises ValueError."""
        if not isinstance(value, dict):
            raise ValueError(f"{PHASE_DURATIONS_KEY} must be an object, got {value!r}")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown phases in {PHASE_DURATIONS_KEY}: {sorted(unknown)}")
        for phase, seconds in value.items():
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
                raise ValueError(f"{PHASE_DURATIONS_KEY}.{phase} must be a positive number, got {seconds!r}")
        return cls(**value)

    def as_config(self) -> dict[str, float]:
        return {"betting": self.betting, "locked": self.locked, "playback": self.playback, "settle": self.settle}


@dataclass(frozen=True)
class Schedule:
    betting_opens_at: datetime
    locks_at: datetime
    playback_starts_at: datetime
    settles_at: datetime
    ends_at: datetime  # end of the settle window; the next round opens here


def schedule_from(betting_opens_at: datetime, durations: PhaseDurations) -> Schedule:
    locks_at = betting_opens_at + timedelta(seconds=durations.betting)
    playback_starts_at = locks_at + timedelta(seconds=durations.locked)
    return Schedule(betting_opens_at, locks_at, playback_starts_at, *_after_playback(playback_starts_at, durations))


def schedule_of(round_, durations: PhaseDurations) -> Schedule:
    """Deadlines of an existing round. Only the last two are derived (the table stores the first three)."""
    return Schedule(
        round_.betting_opens_at,
        round_.locks_at,
        round_.playback_starts_at,
        *_after_playback(round_.playback_starts_at, durations),
    )


def _after_playback(playback_starts_at: datetime, durations: PhaseDurations) -> tuple[datetime, datetime]:
    settles_at = playback_starts_at + timedelta(seconds=durations.playback)
    return settles_at, settles_at + timedelta(seconds=durations.settle)
