"""Target return to player and payout tables built from a city's pool (doc 03, payout & RTP math).

The targets live in app_config under "target_rtp", one for Under and Over and one for Exact. To change
how much players get back, run `python -m app.payout --rtp 0.80 --rtp-exact 0.75`: it stores the new
targets and adds a new active payout table for every active city. Open rounds keep the table they
were opened with; the change applies from the next round.

Same rules as the pipeline's tables step: p = share of the pool for that bet type, multiplier =
floor to 0.01 of r / p, at least 1.01 and at most 50, and the effective RTP p * m must land in
[r - 1%, r]. The multiplier is computed with exact fractions, so no float rounding creeps in.
"""
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

BETS = ("under", "exact", "over")
TARGET_RTP_KEY = "target_rtp"
MIN_MULTIPLIER = Decimal("1.01")
MAX_MULTIPLIER = Decimal("50.00")
RTP_BAND = Decimal("0.01")  # the effective RTP may sit at most this far below the target
P_STEP = Decimal("0.00000001")  # payout_tables p_* and rtp_* are NUMERIC(10, 8)


@dataclass(frozen=True)
class TargetRtp:
    under_over: float = 0.90
    exact: float = 0.80

    def for_bet(self, bet: str) -> float:
        return self.exact if bet == "exact" else self.under_over

    @classmethod
    def from_config(cls, value: object) -> "TargetRtp":
        """Build from the app_config value; missing keys keep their defaults. Raises ValueError."""
        if not isinstance(value, dict):
            raise ValueError(f"{TARGET_RTP_KEY} must be an object, got {value!r}")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys in {TARGET_RTP_KEY}: {sorted(unknown)}")
        targets = cls(**value)
        targets.check()
        return targets

    def check(self) -> None:
        for name, rtp in self.as_config().items():
            if isinstance(rtp, bool) or not isinstance(rtp, (int, float)) or not 0.5 <= rtp < 1:
                raise ValueError(f"{TARGET_RTP_KEY}.{name} must be at least 0.5 and below 1, got {rtp!r}")

    def as_config(self) -> dict[str, float]:
        return {"under_over": self.under_over, "exact": self.exact}


def outcome_for(count: int, line: int) -> str:
    if count < line:
        return "under"
    if count == line:
        return "exact"
    return "over"


def multiplier(k: int, n: int, rtp: float) -> Decimal:
    """floor to 0.01 of r / p for p = k / n, clamped to [1.01, 50]."""
    cents = math.floor(Fraction(str(rtp)) * n / k * 100)
    return min(max(Decimal(cents) / 100, MIN_MULTIPLIER), MAX_MULTIPLIER)


def build_table(counts: list[int], line: int, targets: TargetRtp) -> dict[str, Decimal]:
    """p_*, m_* and rtp_* for a pool of clip counts, ready for payout_tables. Raises ValueError."""
    n = len(counts)
    if n == 0:
        raise ValueError("the pool is empty")
    sizes = dict.fromkeys(BETS, 0)
    for count in counts:
        sizes[outcome_for(count, line)] += 1
    for bet, k in sizes.items():
        if k == 0:
            raise ValueError(f"no clips for {bet} at line {line}, so that bet could never win")
    row = {}
    for bet, k in sizes.items():
        rtp = Decimal(str(targets.for_bet(bet)))
        m = multiplier(k, n, float(rtp))
        effective = Decimal(k) * m / Decimal(n)
        if not rtp - RTP_BAND <= effective <= rtp:
            raise ValueError(f"{bet} RTP {effective:.4f} is outside [{rtp - RTP_BAND}, {rtp}]")
        row.update({
            f"p_{bet}": (Decimal(k) / Decimal(n)).quantize(P_STEP),
            f"m_{bet}": m,
            f"rtp_{bet}": effective.quantize(P_STEP),
        })
    return row
