import pytest

from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry, ModelSpec

CODER = ModelSpec(name="coder", backend="fake", vram_mb=4700, role="code")
PAINTER = ModelSpec(name="painter", backend="fake", vram_mb=7000, role="image")
TINY = ModelSpec(name="tiny", backend="fake", vram_mb=500, role="code")


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(
        gpu_service=GpuServiceSettings(host="test", port=7311, vram_headroom_mb=768),
        models={m.name: m for m in (CODER, PAINTER, TINY)},
    )


@pytest.fixture
def gpu() -> FakeGpuBackend:
    return FakeGpuBackend(total_mb=12288)


class FakeClock:
    """Monotonic clock advanced by hand, so LRU ordering is deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


import asyncio
from itertools import count

import httpx

from ohh_gpu.api import create_app
from ohh_gpu.artifacts import ArtifactStore
from ohh_gpu.events import EventBus
from ohh_gpu.ops import OpRegistry
from ohh_gpu.scheduler import VramScheduler

TEST_TOKEN = "test-token"


@pytest.fixture
def events() -> EventBus:
    return EventBus()


@pytest.fixture
def ops() -> OpRegistry:
    counter = count(1)
    return OpRegistry(id_factory=lambda: f"op{next(counter)}")


@pytest.fixture
def scheduler(registry, gpu, events, clock) -> VramScheduler:
    return VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )


@pytest.fixture
def artifacts(tmp_path) -> ArtifactStore:
    counter = count(1)
    return ArtifactStore(
        root=tmp_path / "artifacts", id_factory=lambda: f"{next(counter):032x}"
    )


@pytest.fixture
def app(scheduler, ops, events, artifacts):
    return create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as client:
        yield client


async def settle(app) -> None:
    """Await every background op the app has spawned."""
    while app.state.tasks:
        await asyncio.gather(*list(app.state.tasks), return_exceptions=True)
