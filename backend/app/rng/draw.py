"""Secure clip draw (doc 03, RNG).

A 64-bit value from the OS CSPRNG picks one clip uniformly among the city's pool minus the
cooldown window. Values in the uneven tail above the largest multiple of n are rejected, so
"raw mod n" has no modulo bias. The accepted raw value is kept for the audit log: anyone can
recompute the result as raw mod n over the candidates sorted by clip id.
"""
import math
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass

RAW_BITS = 64
COOLDOWN_SHARE = 0.30
COOLDOWN_MAX = 100

# Returns a uniformly random non-negative integer with the given number of bits.
RandomBits = Callable[[int], int]


@dataclass(frozen=True)
class Draw:
    clip_id: int
    raw_random: int
    pool_size: int
    cooldown_ids: list[int]


class EmptyPoolError(Exception):
    """The city has no drawable clips."""


def cooldown_size(pool_size: int) -> int:
    """K = min(100, floor(30% of the pool)); 3 for a pool of 11 or 12."""
    return min(COOLDOWN_MAX, math.floor(pool_size * COOLDOWN_SHARE))


def uniform_index(n: int, randbits: RandomBits = secrets.randbits) -> tuple[int, int]:
    """Uniform index in [0, n) by rejection sampling. Returns (index, accepted raw value)."""
    if n <= 0:
        raise ValueError("n must be positive")
    limit = (2**RAW_BITS // n) * n
    while True:
        raw = randbits(RAW_BITS)
        if raw < limit:
            return raw % n, raw


def draw_clip(
    pool_ids: Sequence[int],
    recent_ids: Sequence[int],
    randbits: RandomBits = secrets.randbits,
) -> Draw:
    """Draw one clip from the pool, skipping the cooldown window.

    recent_ids are the clip ids of the city's latest rounds, newest first; the window is the
    first K distinct ones that are still in the pool.
    """
    pool = sorted(set(pool_ids))
    if not pool:
        raise EmptyPoolError
    k = cooldown_size(len(pool))
    in_pool = set(pool)
    cooldown: list[int] = []
    for clip_id in recent_ids:
        if len(cooldown) >= k:
            break
        if clip_id in in_pool and clip_id not in cooldown:
            cooldown.append(clip_id)
    candidates = [c for c in pool if c not in cooldown]
    index, raw = uniform_index(len(candidates), randbits)
    return Draw(clip_id=candidates[index], raw_random=raw, pool_size=len(pool), cooldown_ids=cooldown)
