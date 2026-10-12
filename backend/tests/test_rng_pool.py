import pytest
from sqlalchemy import text

from app.media.signing import clip_record, clip_signature_valid, sign_record
from app.models import City, Clip
from app.rng.draw import EmptyPoolError
from app.rng.pool import draw_for_city, load_pool, recent_clip_ids

KEY = "test-key"


def _clip(**overrides) -> Clip:
    values = dict(
        angle_id="MVI_39031",
        file="clips/MVI_39031-00001.mp4",
        timeline_file="timelines/MVI_39031-00001.json",
        tracks_file="tracks/MVI_39031-00001.json",
        duration_s=30.0,
        count=11,
        sha256="a" * 64,
        timeline_sha256="b" * 64,
        tracks_sha256="c" * 64,
        source_dataset="UA-DETRAC",
        licence="UA-DETRAC research",
        weights_sha256="d" * 64,
        pipeline_version="0.4.0",
        signature="",
    )
    values.update(overrides)
    return Clip(**values)


def test_signature_matches_the_pipeline():
    """Reference value computed with pipeline/sign.py for the same record and key."""
    clip = _clip()
    assert sign_record(clip_record(clip, "las_vegas"), KEY) == (
        "211d53b8bd8d356c97ffa6186edbfe5f41ccbf3db4bec1a017e8c6137a218f78"
    )


def test_signature_check():
    clip = _clip()
    clip.signature = sign_record(clip_record(clip, "las_vegas"), KEY)
    assert clip_signature_valid(clip, "las_vegas", KEY)
    assert not clip_signature_valid(clip, "tokyo", KEY)  # signed for another city
    assert not clip_signature_valid(clip, "las_vegas", "other-key")
    assert not clip_signature_valid(clip, "las_vegas", "")  # no key: nothing passes
    clip.count = 12  # outcome changed after signing
    assert not clip_signature_valid(clip, "las_vegas", KEY)


def test_unsigned_clip_is_invalid():
    assert not clip_signature_valid(_clip(signature=""), "las_vegas", KEY)


# --- database (needs TEST_DATABASE_URL) ---

async def _city(session, slug: str = "las_vegas") -> City:
    city = City(slug=slug, name_key=f"city.{slug}", lat=0, lng=0, theme_key="vegas", line_l=11,
                min_stake=100, max_stake=10000)
    session.add(city)
    await session.flush()
    return city


async def _add_clip(session, city: City, n: int, *, verified=True, signed=True, key=KEY) -> Clip:
    clip = _clip(city_id=city.id, file=f"clips/c{n}.mp4", verified=verified)
    if signed:
        clip.signature = sign_record(clip_record(clip, city.slug), key)
    session.add(clip)
    await session.flush()
    return clip


async def _add_round(session, city: City, clip: Clip, status: str = "SETTLED") -> None:
    table_id = await session.scalar(text(
        "INSERT INTO payout_tables (city_id, version, p_under, p_exact, p_over, m_under, m_exact, m_over, "
        "rtp_under, rtp_exact, rtp_over) VALUES (:c, (SELECT count(*) + 1 FROM payout_tables), "
        ".5, .1, .4, 1.8, 8, 2.25, .9, .8, .9) RETURNING id"), {"c": city.id})
    await session.execute(text(
        "INSERT INTO rounds (city_id, clip_id, status, payout_table_id, betting_opens_at, locks_at, "
        "playback_starts_at) VALUES (:c, :clip, :s, :t, now(), now(), now())"),
        {"c": city.id, "clip": clip.id, "s": status, "t": table_id})


@pytest.mark.asyncio
async def test_pool_keeps_only_verified_and_correctly_signed_clips(session):
    city = await _city(session)
    good = await _add_clip(session, city, 1)
    await _add_clip(session, city, 2, verified=False)
    await _add_clip(session, city, 3, signed=False)
    await _add_clip(session, city, 4, key="wrong-key")
    other_city = await _city(session, "tokyo")
    await _add_clip(session, other_city, 5)
    pool = await load_pool(session, city, KEY)
    assert [c.id for c in pool] == [good.id]


@pytest.mark.asyncio
async def test_recent_clip_ids_are_newest_first_and_include_voided_rounds(session):
    city = await _city(session)
    clips = [await _add_clip(session, city, n) for n in range(4)]
    for clip, status in zip(clips, ["SETTLED", "VOIDED", "SETTLED", "BETTING"]):
        await _add_round(session, city, clip, status)
    assert await recent_clip_ids(session, city.id, 3) == [clips[3].id, clips[2].id, clips[1].id]
    assert await recent_clip_ids(session, city.id, 0) == []


@pytest.mark.asyncio
async def test_draw_for_city_respects_the_cooldown(session):
    city = await _city(session)
    clips = [await _add_clip(session, city, n) for n in range(12)]  # K = 3
    for clip in clips[:5]:
        await _add_round(session, city, clip)
    blocked = {clips[4].id, clips[3].id, clips[2].id}
    for _ in range(200):
        draw = await draw_for_city(session, city, KEY)
        assert draw.clip_id not in blocked
        assert sorted(draw.cooldown_ids) == sorted(blocked)
        assert draw.pool_size == 12


@pytest.mark.asyncio
async def test_draw_for_city_with_no_signed_clips_raises(session):
    city = await _city(session)
    await _add_clip(session, city, 1, signed=False)
    with pytest.raises(EmptyPoolError):
        await draw_for_city(session, city, KEY)
