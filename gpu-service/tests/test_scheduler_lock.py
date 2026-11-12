import asyncio

import pytest

from ohh_gpu.backends.base import ProgressFn
from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.errors import BackendOom, VramExhausted
from ohh_gpu.registry import ModelSpec
from ohh_gpu.scheduler import VramScheduler


class SlowGpu(FakeGpuBackend):
    """Records the interleaving of concurrent loads to prove serialization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeline: list[str] = []

    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        self.timeline.append(f"enter:{spec.name}")
        await asyncio.sleep(0.01)
        await super().load(spec, progress)
        self.timeline.append(f"exit:{spec.name}")


class OomOnceGpu(FakeGpuBackend):
    """Raises BackendOom on the first load of a named model, then succeeds."""

    def __init__(self, oom_for: str, **kwargs):
        super().__init__(**kwargs)
        self._oom_for = oom_for
        self._raised = False

    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        if spec.name == self._oom_for and not self._raised:
            self._raised = True
            # Record the attempt before raising, so load_calls counts attempts.
            self.load_calls.append(spec.name)
            raise BackendOom(
                "Simulated transient CUDA OOM.",
                remedy="Free VRAM and retry.",
                model=spec.name,
                requested_mb=spec.vram_mb,
            )
        await super().load(spec, progress)


class AlwaysOomGpu(FakeGpuBackend):
    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        # Record the attempt before raising, so load_calls counts attempts.
        self.load_calls.append(spec.name)
        raise BackendOom(
            "Simulated persistent CUDA OOM.",
            remedy="Free VRAM and retry.",
            model=spec.name,
            requested_mb=spec.vram_mb,
        )


class AlwaysOomForGpu(FakeGpuBackend):
    """Raises BackendOom on every load of a named model; other models load
    normally, so a model can be loaded and pinned before the OOM target
    is attempted."""

    def __init__(self, oom_for: str, **kwargs):
        super().__init__(**kwargs)
        self._oom_for = oom_for

    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        if spec.name == self._oom_for:
            # Record the attempt before raising, so load_calls counts attempts.
            self.load_calls.append(spec.name)
            raise BackendOom(
                "Simulated persistent CUDA OOM.",
                remedy="Free VRAM and retry.",
                model=spec.name,
                requested_mb=spec.vram_mb,
            )
        await super().load(spec, progress)


async def test_concurrent_loads_are_serialized(registry, clock):
    gpu = SlowGpu(total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    await asyncio.gather(scheduler.load("coder"), scheduler.load("tiny"))
    # No load may begin before the previous one exits.
    assert gpu.timeline in (
        ["enter:coder", "exit:coder", "enter:tiny", "exit:tiny"],
        ["enter:tiny", "exit:tiny", "enter:coder", "exit:coder"],
    )


async def test_queue_depth_reflects_waiters(registry, clock):
    gpu = SlowGpu(total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    first = asyncio.create_task(scheduler.load("coder"))
    await asyncio.sleep(0)  # let `first` take the lock
    second = asyncio.create_task(scheduler.load("tiny"))
    await asyncio.sleep(0)
    assert scheduler.status().busy is True
    assert scheduler.status().queue_depth == 1
    await asyncio.gather(first, second)
    assert scheduler.status().busy is False
    assert scheduler.status().queue_depth == 0


async def test_oom_unloads_unpinned_and_retries_once(registry, clock):
    gpu = OomOnceGpu(oom_for="painter", total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    await scheduler.load("coder")
    await scheduler.load("tiny")

    result = await scheduler.load("painter")
    assert scheduler.is_resident("painter")
    assert sorted(e.model for e in result.evicted) == ["coder", "tiny"]
    assert all(e.reason == "evicted" for e in result.evicted)
    # Observed cost must be measured across the attempt that succeeded, not
    # across the evictions that preceded it — otherwise this reads 6500.
    assert result.vram_mb == 7000
    assert scheduler.observed_mb("painter") == 7000


async def test_oom_after_retry_raises_vram_exhausted_with_the_allocation(
    registry, clock
):
    gpu = AlwaysOomGpu(total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    with pytest.raises(VramExhausted) as excinfo:
        await scheduler.load("painter")
    detail = excinfo.value.detail
    assert detail["model"] == "painter"
    assert detail["requested_mb"] == 7000
    assert "without" in excinfo.value.remedy.lower()


async def test_oom_with_nothing_to_free_does_not_retry(registry, clock):
    gpu = AlwaysOomGpu(total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    with pytest.raises(VramExhausted):
        await scheduler.load("painter")
    assert gpu.load_calls.count("painter") == 1


async def test_lock_is_released_after_a_failed_load(registry, clock):
    gpu = AlwaysOomGpu(total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    with pytest.raises(VramExhausted):
        await scheduler.load("painter")
    assert scheduler.status().busy is False
    assert scheduler.status().queue_depth == 0


async def test_both_oom_raises_carry_a_freed_list(registry, clock):
    # `freed` used to be present on the post-panic raise and absent on the
    # pre-panic one, so a client reading "what did the GPU service free for me?"
    # had to branch on whether the key existed. Empty list, never absent.
    nothing_to_free = AlwaysOomGpu(total_mb=12288)
    scheduler = VramScheduler(
        registry=registry,
        probe=nothing_to_free,
        backends={"fake": nothing_to_free},
        clock=clock,
    )
    with pytest.raises(VramExhausted) as before_panic:
        await scheduler.load("painter")
    assert before_panic.value.to_dict()["freed"] == []

    # "tiny" (500MB) fits alongside a resident unpinned "coder", so nothing is
    # planned for eviction and the OOM recovery is what frees it -- reaching
    # the second raise, the one that has something to report.
    with_a_victim = AlwaysOomForGpu(oom_for="tiny", total_mb=12288)
    retried = VramScheduler(
        registry=registry,
        probe=with_a_victim,
        backends={"fake": with_a_victim},
        clock=clock,
    )
    await retried.load("coder")
    with pytest.raises(VramExhausted) as after_panic:
        await retried.load("tiny")
    assert after_panic.value.to_dict()["freed"] == ["coder"]


async def test_panic_evict_never_unloads_a_pinned_model(registry, clock):
    # "tiny" (500MB) fits the usable free VRAM even with "coder" (4700MB)
    # pinned and resident, so _plan_evictions finds nothing to do and the
    # load reaches _load's OOM-recovery path rather than failing earlier
    # in the fit check. That is the realistic condition under which a
    # panic-evict runs: the probe said it would fit, and the device
    # disagreed.
    gpu = AlwaysOomForGpu(oom_for="tiny", total_mb=12288)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    await scheduler.load("coder", pin=True)

    with pytest.raises(VramExhausted):
        await scheduler.load("tiny")

    # The panic-evict found nothing unpinned to free, so it must not have
    # touched the pinned model, and must not have retried the load.
    assert scheduler.is_resident("coder")
    assert gpu.load_calls.count("tiny") == 1
