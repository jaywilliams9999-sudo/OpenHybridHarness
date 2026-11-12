"""Idle eviction is the VramScheduler's job, not a backend's.

Ollama keeps its own idle timer. Left running, it unloads a model the VramScheduler
still books as resident, and `_plan_evictions` then counts that VRAM twice:
once in `usable_free_mb()`, which already includes it, and again as the ghost's
`vram_mb`. One subsystem owns residency, so the backend is told to
keep models forever and the VramScheduler sweeps idle ones itself -- keeping the book
and the card in step while still handing the card back to a game.
"""

import json

import httpx

from tests.conftest import CODER, PAINTER, TINY
from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.backends.ollama import OllamaBackend
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry, load_registry
from ohh_gpu.scheduler import VramScheduler

IDLE_S = 1800


class ManualClock:
    """A clock that moves only when told to, so idle time is explicit."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def build(idle_s: int = IDLE_S) -> tuple[VramScheduler, FakeGpuBackend, ManualClock]:
    gpu = FakeGpuBackend(total_mb=24576)
    clock = ManualClock()
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(
            host="t", port=1, vram_headroom_mb=768, idle_unload_s=idle_s
        ),
        models={m.name: m for m in (CODER, PAINTER, TINY)},
    )
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    return scheduler, gpu, clock


async def test_sweep_unloads_only_models_idle_past_the_limit():
    scheduler, gpu, clock = build()
    await scheduler.load("coder")
    clock.now = 1000
    await scheduler.load("painter")

    clock.now = IDLE_S + 1  # coder idle 1801s, painter idle 801s
    records = await scheduler.sweep_idle()

    assert [(r.model, r.reason) for r in records] == [("coder", "idle")]
    assert records[0].freed_mb == CODER.vram_mb
    assert not scheduler.is_resident("coder")
    assert scheduler.is_resident("painter")
    assert gpu.free_mb() == 24576 - PAINTER.vram_mb


async def test_sweep_spares_pinned_models():
    scheduler, _, clock = build()
    await scheduler.load("coder", pin=True)
    clock.now = IDLE_S * 10
    assert await scheduler.sweep_idle() == []
    assert scheduler.is_resident("coder")


async def test_use_resets_the_idle_timer():
    scheduler, _, clock = build()
    await scheduler.load("coder")
    clock.now = IDLE_S - 1
    await scheduler.load("coder")  # already resident: a touch, not a load
    clock.now = IDLE_S + 1
    assert await scheduler.sweep_idle() == []


async def test_zero_disables_the_sweep():
    scheduler, _, clock = build(idle_s=0)
    await scheduler.load("coder")
    clock.now = 10**9
    assert await scheduler.sweep_idle() == []
    assert scheduler.is_resident("coder")


def test_registry_reads_idle_unload_s(tmp_path):
    path = tmp_path / "g.toml"
    path.write_text("[gpu_service]\nidle_unload_s = 600\n", encoding="utf-8")
    assert load_registry(path).gpu_service.idle_unload_s == 600


def test_registry_idle_default_is_thirty_minutes(tmp_path):
    path = tmp_path / "g.toml"
    path.write_text("[gpu_service]\n", encoding="utf-8")
    assert load_registry(path).gpu_service.idle_unload_s == 1800


async def test_ollama_default_never_expires_a_model_on_its_own():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    backend = OllamaBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await backend.load(CODER, progress=lambda s, t: None)
    assert seen[0]["keep_alive"] == -1


async def test_the_app_runs_the_sweep_while_serving(ops, events, artifacts):
    import asyncio

    from ohh_gpu.api import create_app

    scheduler, _, clock = build()
    await scheduler.load("coder")
    clock.now = IDLE_S + 1
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token="t",
        version="0.1.0",
        idle_sweep_interval_s=0.01,
    )
    async with app.router.lifespan_context(app):
        for _ in range(100):
            if not scheduler.is_resident("coder"):
                break
            await asyncio.sleep(0.01)
    assert not scheduler.is_resident("coder")


async def test_a_failing_sweep_does_not_stop_the_sweeper(ops, events, artifacts):
    import asyncio

    from ohh_gpu.api import create_app

    scheduler, _, _ = build()
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ollama hiccup")
        return []

    scheduler.sweep_idle = flaky
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token="t",
        version="0.1.0",
        idle_sweep_interval_s=0.01,
    )
    async with app.router.lifespan_context(app):
        for _ in range(100):
            if calls >= 2:
                break
            await asyncio.sleep(0.01)
    assert calls >= 2
