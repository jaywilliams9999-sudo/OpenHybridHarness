import pytest

from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.errors import VramExhausted
from ohh_gpu.registry import ModelSpec
from ohh_gpu.scheduler import VramScheduler

# An 8GB card: coder (4700) + painter (7000) cannot coexist under a 768 headroom.
SMALL_CARD_MB = 8192
# A 12GB card: coder + painter also cannot coexist (7588 free, 6820 usable),
# but one eviction is enough — which is what makes LRU *order* observable.
BIG_CARD_MB = 12288


def build(registry, clock, total_mb) -> VramScheduler:
    gpu = FakeGpuBackend(total_mb=total_mb)
    return VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)


@pytest.fixture
def small_scheduler(registry, clock) -> VramScheduler:
    return build(registry, clock, SMALL_CARD_MB)


async def test_evicts_to_make_room(small_scheduler):
    await small_scheduler.load("coder")
    result = await small_scheduler.load("painter")
    assert [e.model for e in result.evicted] == ["coder"]
    assert result.evicted[0].freed_mb == 4700
    assert result.evicted[0].reason == "evicted"
    assert small_scheduler.is_resident("painter")
    assert not small_scheduler.is_resident("coder")


async def test_evicts_least_recently_used_first(registry, clock):
    scheduler = build(registry, clock, BIG_CARD_MB)
    await scheduler.load("tiny")     # 500, loaded first
    await scheduler.load("coder")    # 4700, loaded second
    await scheduler.load("tiny")     # touched, so coder is now the LRU

    # painter needs 7000; usable free is 12288-5200-768 = 6320.
    # Evicting coder alone yields 6320+4700 = 11020, so tiny survives.
    result = await scheduler.load("painter")
    assert [e.model for e in result.evicted] == ["coder"]
    assert scheduler.is_resident("tiny")


async def test_untouched_lru_order_evicts_the_oldest_first(registry, clock):
    scheduler = build(registry, clock, BIG_CARD_MB)
    await scheduler.load("tiny")     # 500, oldest
    await scheduler.load("coder")    # 4700

    # Evicting tiny alone yields 6320+500 = 6820 -> still short, so coder follows.
    result = await scheduler.load("painter")
    assert [e.model for e in result.evicted] == ["tiny", "coder"]


async def test_stops_evicting_once_the_request_fits(registry, clock):
    scheduler = build(registry, clock, BIG_CARD_MB)
    await scheduler.load("coder")    # 4700, oldest
    await scheduler.load("tiny")     # 500

    # Evicting coder alone yields 6320+4700 = 11020 >= 7000, so tiny is spared.
    result = await scheduler.load("painter")
    assert [e.model for e in result.evicted] == ["coder"]


async def test_pinned_models_are_never_evicted(small_scheduler):
    await small_scheduler.load("coder", pin=True)
    with pytest.raises(VramExhausted) as excinfo:
        await small_scheduler.load("painter")
    assert small_scheduler.is_resident("coder")
    assert excinfo.value.detail["requested_mb"] == 7000
    assert "pinned" in excinfo.value.remedy


async def test_eviction_emits_unloaded_events(registry, clock):
    from ohh_gpu.events import EventBus

    gpu = FakeGpuBackend(total_mb=SMALL_CARD_MB)
    events = EventBus()
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )
    with events.subscribe() as stream:
        await scheduler.load("coder")
        await scheduler.load("painter")
        drained = stream.drain()
    unloaded = [e for e in drained if e.kind == "model.unloaded"]
    assert [e.body["model"] for e in unloaded] == ["coder"]
    assert unloaded[0].body["reason"] == "evicted"


async def test_an_external_gpu_consumer_is_named_instead_of_phantom_pins(
    registry, clock
):
    # The machine may run games. When something outside the GPU service holds the
    # card, nothing is resident -- and the old remedy still said "every
    # remaining resident model is pinned. Unpin one", with `pinned: []`
    # printed right beside it. That misdirects on the branch's most likely
    # real-hardware failure.
    gpu = FakeGpuBackend(total_mb=BIG_CARD_MB)
    # Allocate straight on the fake card, bypassing the VramScheduler entirely:
    # this is exactly what an external CUDA process looks like to the probe.
    await gpu.load(
        ModelSpec(name="a-game", backend="fake", vram_mb=9000, role="other"),
        progress=lambda s, t: None,
    )
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)

    with pytest.raises(VramExhausted) as excinfo:
        await scheduler.load("painter")

    remedy = excinfo.value.remedy
    assert "another process on this machine" in remedy.lower()
    assert "pinned" not in remedy.lower()
    assert "proceed without this model" in remedy
    assert excinfo.value.detail["pinned"] == []
    assert scheduler.status().resident == []


async def test_the_pinned_remedy_still_applies_when_models_are_resident(small_scheduler):
    # The other side of that branch: with a pinned model actually holding the
    # VRAM, the actionable instruction really is "unpin one".
    await small_scheduler.load("coder", pin=True)
    with pytest.raises(VramExhausted) as excinfo:
        await small_scheduler.load("painter")
    assert "pinned (coder)" in excinfo.value.remedy
    assert "another process" not in excinfo.value.remedy.lower()


async def test_vram_exhausted_always_carries_a_freed_list(small_scheduler, registry, clock):
    # `freed` used to be present on one raise site and absent on another, so
    # the client had to branch on key existence to read it.
    await small_scheduler.load("coder", pin=True)
    with pytest.raises(VramExhausted) as excinfo:
        await small_scheduler.load("painter")
    assert excinfo.value.to_dict()["freed"] == []

    tiny_card = build(registry, clock, 4096)
    with pytest.raises(VramExhausted) as too_big:
        await tiny_card.load("painter")
    assert too_big.value.to_dict()["freed"] == []


async def test_nothing_evictable_raises_with_the_attempted_allocation(registry, clock):
    scheduler = build(registry, clock, SMALL_CARD_MB)
    await scheduler.load("coder", pin=True)
    await scheduler.load("tiny", pin=True)
    with pytest.raises(VramExhausted) as excinfo:
        await scheduler.load("painter")
    detail = excinfo.value.detail
    assert detail["requested_mb"] == 7000
    assert detail["usable_free_mb"] == SMALL_CARD_MB - 4700 - 500 - 768
    assert sorted(detail["pinned"]) == ["coder", "tiny"]
