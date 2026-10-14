"""RTP simulation (doc 03, doc 08).

Plays many rounds of one city with the production draw (cooldown included) and the production payout
rounding, and measures what players get back for every 1 credit staked under these strategies:
- under / exact / over: the same bet every round;
- informed: a player who remembers the last rounds, so knows which clips are in the cooldown and
  cannot come up, and each round bets the choice with the highest expected return given that.
  If the cooldown makes some rounds pay more than usual, this strategy finds it.

Per strategy it reports the overall RTP, the spread of a single bet, and the RTP of lucky players
(the 99th percentile over sessions of a fixed number of rounds). It also reports how often each clip
was drawn and the best expected return any single round offered.

The random source is seeded so a run can be repeated exactly; the production CSPRNG is tested
separately (tests/test_rng_draw.py).
"""
import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal

from app.engine.settlement import payout_for
from app.payout import BETS, TargetRtp, outcome_for
from app.rng.draw import cooldown_size, draw_clip

STRATEGIES = (*BETS, "informed")


@dataclass(frozen=True)
class CitySetup:
    slug: str
    clip_ids: list[int]
    counts: list[int]  # same order as clip_ids
    line: int
    multipliers: dict[str, Decimal]
    stake: int


@dataclass(frozen=True)
class StrategyStats:
    rtp: float
    bet_sd: float  # standard deviation of one bet's return per credit staked
    session_rtp_p99: float


@dataclass(frozen=True)
class CityReport:
    slug: str
    rounds: int
    seed: str
    exact_rtp: dict[str, float]  # p * payout / stake per bet type, no simulation involved
    strategies: dict[str, StrategyStats]
    best_round_rtp: float  # highest expected return any round offered to a player who tracks the cooldown
    clip_share_deviation: float  # largest relative gap between a clip's draw share and 1 / pool size


def exact_rtp(setup: CitySetup) -> dict[str, float]:
    n = len(setup.counts)
    outcomes = Counter(outcome_for(c, setup.line) for c in setup.counts)
    return {b: outcomes[b] / n * payout_for(setup.stake, setup.multipliers[b]) / setup.stake for b in BETS}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def simulate(setup: CitySetup, rounds: int, seed: str, session_rounds: int = 1000) -> CityReport:
    rng = random.Random(seed)
    outcome = {cid: outcome_for(c, setup.line) for cid, c in zip(setup.clip_ids, setup.counts)}
    pool_by_outcome = Counter(outcome.values())
    # Return per credit staked when the bet wins, after flooring to whole minor units.
    win = {b: payout_for(setup.stake, setup.multipliers[b]) / setup.stake for b in BETS}
    recent: deque[int] = deque(maxlen=cooldown_size(len(setup.clip_ids)))

    returned = dict.fromkeys(STRATEGIES, 0.0)
    squares = dict.fromkeys(STRATEGIES, 0.0)
    session_returned = dict.fromkeys(STRATEGIES, 0.0)
    session_rtps: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    drawn: Counter[int] = Counter()
    best_round = 0.0

    for i in range(1, rounds + 1):
        draw = draw_clip(setup.clip_ids, recent, rng.getrandbits)
        # What the informed player knows before betting: the clips that cannot come up this round.
        left = pool_by_outcome.copy()
        left.subtract(outcome[c] for c in draw.cooldown_ids)
        candidates = len(setup.clip_ids) - len(draw.cooldown_ids)
        expected = {b: left[b] / candidates * win[b] for b in BETS}
        informed = max(BETS, key=expected.__getitem__)
        best_round = max(best_round, expected[informed])

        result = outcome[draw.clip_id]
        for strategy in STRATEGIES:
            bet = informed if strategy == "informed" else strategy
            r = win[bet] if bet == result else 0.0
            returned[strategy] += r
            squares[strategy] += r * r
            session_returned[strategy] += r
        if i % session_rounds == 0:
            for strategy in STRATEGIES:
                session_rtps[strategy].append(session_returned[strategy] / session_rounds)
                session_returned[strategy] = 0.0
        drawn[draw.clip_id] += 1
        recent.appendleft(draw.clip_id)

    strategies = {}
    for s in STRATEGIES:
        mean = returned[s] / rounds
        strategies[s] = StrategyStats(
            rtp=mean,
            bet_sd=math.sqrt(max(squares[s] / rounds - mean * mean, 0.0)),
            session_rtp_p99=_percentile(session_rtps[s], 0.99),
        )
    uniform = rounds / len(setup.clip_ids)
    deviation = max(abs(drawn[c] - uniform) / uniform for c in setup.clip_ids)
    return CityReport(setup.slug, rounds, seed, exact_rtp(setup), strategies, best_round, deviation)


def gate(report: CityReport, targets: TargetRtp, tolerance: float) -> list[str]:
    """Why the city fails; empty when it passes.

    - every fixed bet returns its target RTP within the tolerance, both ways;
    - the informed player gets back no more than the higher target plus the tolerance, so following
      the cooldown cannot beat the house;
    - no single round offers an expected return above that ceiling.
    """
    problems = []
    for bet in BETS:
        target, measured = targets.for_bet(bet), report.strategies[bet].rtp
        if abs(measured - target) > tolerance:
            problems.append(f"{bet} RTP {measured:.2%} is not within {tolerance:.1%} of the target {target:.0%}")
    ceiling = max(targets.under_over, targets.exact) + tolerance
    if report.strategies["informed"].rtp > ceiling:
        problems.append(f"a player who tracks the cooldown gets back {report.strategies['informed'].rtp:.2%}, "
                        f"above {ceiling:.1%}")
    if report.best_round_rtp > ceiling:
        problems.append(f"some rounds offer an expected return of {report.best_round_rtp:.2%}, above {ceiling:.1%}")
    return problems
