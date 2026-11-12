"""`/status` tells the truth about the card: who holds it, how long models
have sat idle, and the last swaps -- so a page opened late still has history."""

import asyncio
from datetime import datetime, timezone

from tests.conftest import CODER, PAINTER, TINY
from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry
from ohh_gpu.scheduler import VramScheduler


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, /, **body) -> None:
        self.events.append((kind, body))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]


def build(gpu=None, events="recorder", total_mb=24576):
    gpu = gpu or FakeGpuBackend(total_mb=total_mb)
    clock = ManualClock()
    rec = Recorder() if events == "recorder" else events
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768, idle_unload_s=0),
        models={m.name: m for m in (CODER, PAINTER, TINY)},
    )
    scheduler = VramScheduler(
        registry=registry,
        probe=gpu,
        backends={"fake": gpu},
        events=rec,
        clock=clock,
        wall_clock=lambda: datetime(2026, 10, 22, tzinfo=timezone.utc),
    )
    return scheduler, gpu, clock, rec


class GatedGpu(FakeGpuBackend):
    """A backend whose load waits until the test lets it finish."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.release = asyncio.Event()

    async def load(self, spec, progress) -> None:
        await self.release.wait()
        await super().load(spec, progress)


async def test_status_names_the_busy_model():
    gpu = GatedGpu(total_mb=24576)
    scheduler, _, _, _ = build(gpu=gpu)
    task = asyncio.create_task(scheduler.load("coder"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert scheduler.status().busy_model == "coder"
    gpu.release.set()
    await task
    assert scheduler.status().busy_model is None


async def test_lock_emits_busy_then_idle():
    scheduler, _, _, rec = build()
    await scheduler.load("coder")
    kinds = rec.kinds()
    assert ("gpu.busy", {"model": "coder"}) in rec.events
    assert kinds.index("gpu.busy") < kinds.index("model.loaded")
    assert kinds.index("model.loaded") < kinds.index("gpu.idle")


async def test_resident_reports_idle_seconds():
    scheduler, _, clock, _ = build()
    await scheduler.load("coder")
    clock.now = 125
    entry = scheduler.status().resident[0]
    assert entry["idle_s"] == 125
    assert isinstance(entry["load_ms"], int)


async def test_eviction_names_the_model_it_made_room_for():
    scheduler, _, _, rec = build(total_mb=PAINTER.vram_mb + 1000)
    await scheduler.load("coder")
    await scheduler.load("painter")
    unloaded = [b for k, b in rec.events if k == "model.unloaded"]
    assert unloaded[0]["model"] == "coder"
    assert unloaded[0]["reason"] == "evicted"
    assert unloaded[0]["for_model"] == "painter"

    await scheduler.unload("painter")
    last = [b for k, b in rec.events if k == "model.unloaded"][-1]
    assert last["model"] == "painter"
    assert "for_model" not in last


async def test_recent_keeps_the_last_twenty_model_events():
    scheduler, _, _, _ = build()
    for _ in range(13):
        await scheduler.load("tiny")
        await scheduler.unload("tiny")
    recent = scheduler.status().recent
    assert len(recent) == 20
    assert all({"event", "at", "model"} <= set(e) for e in recent)
    assert recent[-1]["event"] == "model.unloaded"
    assert [e["event"] for e in recent[:2]] == ["model.loaded", "model.unloaded"]


async def test_recent_survives_without_an_events_bus():
    scheduler, _, _, _ = build(events=None)
    await scheduler.load("coder")
    assert [e["event"] for e in scheduler.status().recent] == ["model.loaded"]
