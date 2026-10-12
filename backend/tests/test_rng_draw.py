import random
from collections import Counter

import pytest

from app.rng.draw import RAW_BITS, EmptyPoolError, cooldown_size, draw_clip, uniform_index


@pytest.mark.parametrize(
    ("pool_size", "k"), [(0, 0), (1, 0), (3, 0), (4, 1), (11, 3), (12, 3), (100, 30), (333, 99), (334, 100), (1000, 100)]
)
def test_cooldown_size_is_30_percent_floored_and_capped(pool_size, k):
    assert cooldown_size(pool_size) == k


def test_uniform_index_rejects_the_biased_tail():
    n = 12
    limit = (2**RAW_BITS // n) * n
    values = iter([2**RAW_BITS - 1, limit, limit - 1])
    calls = []

    def fake_randbits(bits):
        calls.append(bits)
        return next(values)

    index, raw = uniform_index(n, fake_randbits)
    assert raw == limit - 1  # the first two values were in the tail and thrown away
    assert index == (limit - 1) % n
    assert calls == [RAW_BITS, RAW_BITS, RAW_BITS]


def test_uniform_index_accepts_the_first_value_below_the_limit():
    assert uniform_index(12, lambda bits: 25) == (1, 25)


@pytest.mark.parametrize("n", [0, -1])
def test_uniform_index_needs_a_positive_n(n):
    with pytest.raises(ValueError):
        uniform_index(n)


def test_uniform_index_with_the_real_csprng_stays_in_range():
    for n in (1, 2, 7, 12):
        for _ in range(1000):
            index, raw = uniform_index(n)
            assert 0 <= index < n
            assert 0 <= raw < 2**RAW_BITS
            assert raw % n == index


def test_draw_skips_the_cooldown_window():
    pool = list(range(1, 13))  # 12 clips, K = 3
    for _ in range(2000):
        draw = draw_clip(pool, [5, 9, 2, 7])
        assert draw.cooldown_ids == [5, 9, 2]  # only the newest K count
        assert draw.clip_id not in {5, 9, 2}
        assert draw.pool_size == 12


def test_draw_maps_raw_value_onto_sorted_candidates():
    """An auditor recomputes the clip as raw mod n over the candidates sorted by id."""
    draw = draw_clip([40, 10, 30, 20, 50], [30], lambda bits: 7)  # K = 1
    candidates = [10, 20, 40, 50]
    assert draw.cooldown_ids == [30]
    assert draw.clip_id == candidates[7 % 4] == 50
    assert draw.raw_random == 7


def test_cooldown_ignores_duplicates_and_clips_no_longer_in_the_pool():
    pool = list(range(1, 13))
    draw = draw_clip(pool, [99, 4, 4, 6, 8, 10], lambda bits: 0)
    assert draw.cooldown_ids == [4, 6, 8]


def test_small_pool_has_no_cooldown():
    draw = draw_clip([1, 2, 3], [1, 2, 3], lambda bits: 0)
    assert draw.cooldown_ids == []
    assert draw.clip_id == 1


def test_empty_pool_raises():
    with pytest.raises(EmptyPoolError):
        draw_clip([], [])


def test_consecutive_draws_never_repeat_within_the_window():
    pool = list(range(1, 13))
    history: list[int] = []
    for _ in range(5000):
        draw = draw_clip(pool, list(reversed(history[-3:])))
        assert draw.clip_id not in history[-3:]
        history.append(draw.clip_id)


def test_draws_are_uniform_without_cooldown():
    """Chi-square over 120k draws of 12 clips. Seeded source, so the test is deterministic."""
    rng = random.Random(20261012)
    n, draws = 12, 120_000
    counts = Counter(uniform_index(n, rng.getrandbits)[0] for _ in range(draws))
    expected = draws / n
    chi2 = sum((counts[i] - expected) ** 2 / expected for i in range(n))
    assert chi2 < 31.26  # 11 degrees of freedom, p = 0.001
