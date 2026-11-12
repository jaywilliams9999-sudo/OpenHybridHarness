"""Startup reconciliation.

A backend can outlive the GPU service, so a cold VramScheduler can face a card that is
already partly occupied. The invariant these tests protect: after `reconcile`,
the card holds only what the GPU service put there.
"""

import json

import httpx
import pytest

from tests.conftest import CODER, PAINTER, TEST_TOKEN
from ohh_gpu.backends.base import EnumerableBackend
from ohh_gpu.backends.ollama import OllamaBackend
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry, ModelSpec
from ohh_gpu.scheduler import VramScheduler


class FakeEnumerable:
    """A backend that reports a fixed set of names and records releases."""

    def __init__(self, names: list[str], total_mb: int = 12288) -> None:
        self._names = list(names)
        self._total_mb = total_mb
        self.released: list[str] = []
        self._allocated: dict[str, int] = {}

    # VramProbe
    def name(self) -> str:
        return "FakeEnumerable"

    def total_mb(self) -> int:
        return self._total_mb

    def free_mb(self) -> int:
        return self._total_mb - sum(self._allocated.values())

    # Backend
    async def load(self, spec: ModelSpec, progress) -> None:
        self._allocated[spec.name] = spec.vram_mb
        if spec.name not in self._names:
            self._names.append(spec.name)

    async def unload(self, spec: ModelSpec) -> None:
        self._allocated.pop(spec.name, None)

    # EnumerableBackend
    async def resident_names(self) -> list[str]:
        return list(self._names)

    async def release_by_name(self, name: str) -> None:
        self.released.append(name)
        self._names.remove(name)
        self._allocated.pop(name, None)


def scheduler_with(backend, models) -> VramScheduler:
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="test", port=7311, vram_headroom_mb=768),
        models={m.name: m for m in models},
    )
    return VramScheduler(
        registry=registry, probe=backend, backends={"fake": backend}, events=None
    )


def test_the_protocol_is_satisfied_by_the_real_backend():
    # A runtime_checkable Protocol only checks method presence, which is
    # exactly the thing that silently breaks when a method is renamed.
    assert isinstance(OllamaBackend(), EnumerableBackend)


async def test_an_orphan_is_released():
    backend = FakeEnumerable(names=["llama3.1:8b"])
    scheduler = scheduler_with(backend, [CODER])

    released = await scheduler.reconcile()

    assert released == ["llama3.1:8b"]
    assert backend.released == ["llama3.1:8b"]


async def test_a_model_the_scheduler_knows_about_is_left_alone():
    backend = FakeEnumerable(names=[])
    scheduler = scheduler_with(backend, [CODER])
    await scheduler.load(CODER.name)

    released = await scheduler.reconcile()

    # It is resident *and* accounted for; releasing it would evict a live model
    # on every restart of nothing in particular.
    assert released == []
    assert backend.released == []
    assert scheduler.is_resident(CODER.name)


async def test_orphans_go_but_known_models_stay():
    backend = FakeEnumerable(names=["stranger"])
    scheduler = scheduler_with(backend, [CODER])
    await scheduler.load(CODER.name)

    released = await scheduler.reconcile()

    assert released == ["stranger"]
    assert scheduler.is_resident(CODER.name)


async def test_an_empty_card_reconciles_to_nothing():
    backend = FakeEnumerable(names=[])
    scheduler = scheduler_with(backend, [CODER])
    assert await scheduler.reconcile() == []


async def test_a_backend_that_cannot_enumerate_is_skipped(registry, gpu):
    # FakeGpuBackend has no resident_names; reconcile must not crash on it.
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=None
    )
    assert await scheduler.reconcile() == []


async def test_each_backend_is_visited_once_even_when_registered_twice():
    backend = FakeEnumerable(names=["orphan"])
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="test", port=7311, vram_headroom_mb=768),
        models={CODER.name: CODER},
    )
    # The same instance under two keys -- as ollama/diffusers would be if one
    # object served both. Releasing twice would log a phantom second eviction.
    scheduler = VramScheduler(
        registry=registry,
        probe=backend,
        backends={"a": backend, "b": backend},
        events=None,
    )
    released = await scheduler.reconcile()
    assert released == ["orphan"]


async def test_reconcile_emits_vram_changed_only_when_it_freed_something(events):
    quiet = FakeEnumerable(names=[])
    scheduler = VramScheduler(
        registry=ModelRegistry(
            gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768),
            models={CODER.name: CODER},
        ),
        probe=quiet,
        backends={"fake": quiet},
        events=events,
    )
    with events.subscribe() as subscription:
        await scheduler.reconcile()
        # The lock still announces itself; what must stay silent is the card.
        assert [e.kind for e in subscription.drain()] == ["gpu.busy", "gpu.idle"]

    busy = FakeEnumerable(names=["orphan"])
    scheduler = VramScheduler(
        registry=ModelRegistry(
            gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768),
            models={CODER.name: CODER},
        ),
        probe=busy,
        backends={"fake": busy},
        events=events,
    )
    with events.subscribe() as subscription:
        await scheduler.reconcile()
        kinds = [e.kind for e in subscription.drain()]
    assert "vram.changed" in kinds


# -- the real backend's two halves --------------------------------------


def stub(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_resident_names_reads_api_ps():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"model": "llama3.1:8b", "size_vram": 6000},
                    {"name": "qwen3:8b"},
                ]
            },
        )

    assert await OllamaBackend(client=stub(handler)).resident_names() == [
        "llama3.1:8b",
        "qwen3:8b",
    ]


async def test_resident_names_is_empty_when_ollama_is_down():
    # A GPU service that refused to boot because Ollama blinked would be worse than
    # one that boots believing the card is empty -- which is the status quo.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await OllamaBackend(client=stub(handler)).resident_names() == []


async def test_release_by_name_needs_no_registry_entry():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    await OllamaBackend(client=stub(handler)).release_by_name("some:model")

    assert seen[0]["model"] == "some:model"
    assert seen[0]["keep_alive"] == 0
    # No window on an eviction: it would reload what we are evicting.
    assert "options" not in seen[0]


# -- through the app ----------------------------------------------------


async def test_the_app_reconciles_before_serving(ops, events, artifacts):
    from ohh_gpu.api import create_app

    backend = FakeEnumerable(names=["left-behind"])
    scheduler = scheduler_with(backend, [CODER, PAINTER])
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )

    # httpx's ASGITransport does not run lifespan events at all, so entering
    # the lifespan context is the only way to exercise startup wiring. uvicorn
    # does run it in production -- this asserts the hook is connected, which a
    # request-level test silently cannot.
    async with app.router.lifespan_context(app):
        assert backend.released == ["left-behind"]


async def test_a_failing_reconcile_does_not_block_startup(ops, events, artifacts):
    from ohh_gpu.api import create_app

    class Exploding(FakeEnumerable):
        async def resident_names(self) -> list[str]:
            raise RuntimeError("ollama is on fire")

    backend = Exploding(names=[])
    scheduler = scheduler_with(backend, [CODER])
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )

    # Startup must complete despite the fault. Refusing to start over a stale
    # model would turn a recoverable annoyance into an outage.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gpu-service",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        ) as client:
            response = await client.get("/health")
    assert response.status_code == 200


async def test_release_by_name_waits_until_ollama_lets_go():
    """Startup fits models to the card right after reconcile releases orphans.

    ``keep_alive=0`` is acknowledged before the device is freed, so returning
    at once left `fit_to_card` reading the orphan's VRAM as other programs'
    and marking every model unavailable (2026-11-11, RTX 5060 Laptop).
    """
    paths: list[str] = []
    resident = [True, True, False]

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/ps":
            still_there = resident.pop(0) if resident else False
            return httpx.Response(
                200, json={"models": [{"model": "some:model"}] if still_there else []}
            )
        return httpx.Response(200, json={"done": True})

    await OllamaBackend(
        client=stub(handler), unload_poll_interval_s=0
    ).release_by_name("some:model")

    assert paths[0] == "/api/generate"
    assert paths.count("/api/ps") == 3, paths


async def test_shutdown_unloads_what_the_gpu_service_booked(ops, events, artifacts):
    """Ctrl+C must leave the card free.

    Ollama outlives this process and holds what it loaded (keep_alive=-1), so
    without this the next start finds the card full.
    """
    from ohh_gpu.api import create_app

    backend = FakeEnumerable(names=[])
    scheduler = scheduler_with(backend, [CODER])
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )

    async with app.router.lifespan_context(app):
        await scheduler.load(CODER.name, pin=True)
        assert scheduler.is_resident(CODER.name)
        assert backend.free_mb() < backend.total_mb()

    assert not scheduler.is_resident(CODER.name)
    assert backend.free_mb() == backend.total_mb()


async def test_a_failing_shutdown_unload_does_not_raise(ops, events, artifacts):
    from ohh_gpu.api import create_app

    class Stuck(FakeEnumerable):
        async def unload(self, spec: ModelSpec) -> None:
            raise RuntimeError("ollama is gone")

    backend = Stuck(names=[])
    scheduler = scheduler_with(backend, [CODER])
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )

    async with app.router.lifespan_context(app):
        await scheduler.load(CODER.name)
