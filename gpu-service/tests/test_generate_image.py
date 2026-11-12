import asyncio
import json

import httpx
import pytest

from ohh_gpu.api import create_app
from ohh_gpu.backends.base import ImageRequest
from ohh_gpu.backends.fake import FAKE_PNG, FakeGpuBackend
from ohh_gpu.errors import BackendFailure, VramExhausted
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry, ModelSpec
from ohh_gpu.scheduler import VramScheduler
from tests.conftest import TEST_TOKEN, settle

SMALL_CARD_MB = 8192


@pytest.fixture
def small_scheduler(registry, events, clock) -> VramScheduler:
    gpu = FakeGpuBackend(total_mb=SMALL_CARD_MB)
    return VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )


class _NoImageBackend:
    """Implements Backend (load/unload) but not ImageBackend — no
    generate_image — to exercise a misconfigured registry entry."""

    def __init__(self) -> None:
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []

    async def load(self, spec, progress) -> None:
        self.load_calls.append(spec.name)

    async def unload(self, spec) -> None:
        self.unload_calls.append(spec.name)


# -- VramScheduler level ------------------------------------------------------


async def test_generate_auto_loads_the_image_model(scheduler, gpu):
    assert not scheduler.is_resident("painter")
    result = await scheduler.generate_image(ImageRequest(prompt="a lantern"))
    assert result.model == "painter"
    assert result.png == FAKE_PNG
    assert scheduler.is_resident("painter")
    assert gpu.load_calls == ["painter"]


async def test_generate_evicts_to_make_room_and_reports_it(small_scheduler):
    await small_scheduler.load("coder")
    result = await small_scheduler.generate_image(ImageRequest(prompt="a lantern"))
    assert [e.model for e in result.evicted] == ["coder"]
    assert result.evicted[0].reason == "evicted"


async def test_generate_reuses_a_resident_model_without_evicting(scheduler, gpu):
    await scheduler.load("painter")
    result = await scheduler.generate_image(ImageRequest(prompt="a lantern"))
    assert result.evicted == []
    assert gpu.load_calls == ["painter"]


async def test_generate_honours_the_step_count(scheduler):
    result = await scheduler.generate_image(ImageRequest(prompt="x", steps=8))
    assert result.steps == 8


async def test_generate_defaults_steps_from_the_registry(registry, gpu, events, clock):
    # 6, not the hard-coded fallback of 4, so the registry is provably consulted.
    painter = ModelSpec(
        name="painter", backend="fake", vram_mb=7000, role="image", steps=6
    )
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=7311, vram_headroom_mb=768),
        models={**registry.models, "painter": painter},
    )
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )
    assert (await scheduler.generate_image(ImageRequest(prompt="x"))).steps == 6


async def test_generate_falls_back_to_four_steps_when_unspecified(scheduler):
    # conftest's painter has no `steps`, so the VramScheduler's own default applies.
    assert (await scheduler.generate_image(ImageRequest(prompt="x"))).steps == 4


async def test_generate_defaults_size_from_the_registry(registry, gpu, events, clock):
    # Image size is generation policy, so it lives beside `steps` in the
    # registry, not as a 1024x1024 default in the HTTP layer -- which was
    # wrong for sdxl-turbo, the only image model the GPU service actually has.
    painter = ModelSpec(
        name="painter",
        backend="fake",
        vram_mb=7000,
        role="image",
        width=512,
        height=512,
    )
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=7311, vram_headroom_mb=768),
        models={**registry.models, "painter": painter},
    )
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )
    result = await scheduler.generate_image(ImageRequest(prompt="x"))
    assert (result.width, result.height) == (512, 512)


async def test_an_explicit_size_beats_the_registry(registry, gpu, events, clock):
    painter = ModelSpec(
        name="painter",
        backend="fake",
        vram_mb=7000,
        role="image",
        width=512,
        height=512,
    )
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=7311, vram_headroom_mb=768),
        models={**registry.models, "painter": painter},
    )
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )
    result = await scheduler.generate_image(
        ImageRequest(prompt="x", width=768, height=1024)
    )
    assert (result.width, result.height) == (768, 1024)


async def test_size_falls_back_to_1024_when_the_registry_is_silent(scheduler):
    # conftest's painter declares no size, so the VramScheduler's own last resort
    # applies -- and it is the VramScheduler's, not api.py's.
    result = await scheduler.generate_image(ImageRequest(prompt="x"))
    assert (result.width, result.height) == (1024, 1024)


async def test_the_resolved_size_reaches_the_backend(registry, gpu, events, clock):
    # The point of resolving in the VramScheduler is that no backend re-derives it.
    # Capture what the backend was actually handed.
    seen: list[ImageRequest] = []
    original = gpu.generate_image

    async def spy(spec, req, progress):
        seen.append(req)
        return await original(spec, req, progress)

    gpu.generate_image = spy
    painter = ModelSpec(
        name="painter", backend="fake", vram_mb=7000, role="image", width=512, height=512
    )
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=7311, vram_headroom_mb=768),
        models={**registry.models, "painter": painter},
    )
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )
    await scheduler.generate_image(ImageRequest(prompt="x"))
    assert (seen[0].width, seen[0].height) == (512, 512)


def test_the_repo_registry_sizes_sdxl_turbo_at_512():
    # The shipped models.toml, not a fixture: sdxl-turbo is a 512x512 model
    # and rendering it at 1024 costs VRAM and time for a worse image. Located
    # from this file rather than via default_registry_path(), which an
    # exported OHH_MODELS would redirect.
    from pathlib import Path

    from ohh_gpu.registry import load_registry

    repo_root = Path(__file__).resolve().parents[2]
    spec = load_registry(repo_root / "models.toml").require("sdxl-turbo")
    assert (spec.width, spec.height) == (512, 512)


async def test_progress_total_matches_the_resolved_step_count(scheduler):
    # The VramScheduler resolves `steps` once and threads it into `req` so the
    # backend's own report(step, total) never re-derives a possibly
    # different fallback. Pin the two together so they can't drift apart.
    seen: list[tuple[int, int]] = []
    result = await scheduler.generate_image(
        ImageRequest(prompt="x"),
        progress=lambda step, total: seen.append((step, total)),
    )
    assert result.steps == 4
    assert seen  # the backend did report progress
    assert all(total == result.steps for _, total in seen)


async def test_backend_capability_is_checked_before_any_eviction_or_load(
    registry, gpu, events, clock
):
    # A registry entry tagged role="image" but pointing at a backend with no
    # generate_image (e.g. a future OllamaBackend) must fail cleanly, without
    # evicting a resident model or loading anything first.
    text_backend = _NoImageBackend()
    misconfigured = ModelSpec(name="painter", backend="text", vram_mb=7000, role="image")
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=7311, vram_headroom_mb=768),
        models={**registry.models, "painter": misconfigured},
    )
    scheduler = VramScheduler(
        registry=registry,
        probe=gpu,
        backends={"fake": gpu, "text": text_backend},
        events=events,
        clock=clock,
    )
    await scheduler.load("coder")

    with pytest.raises(BackendFailure):
        await scheduler.generate_image(ImageRequest(prompt="x"))

    assert scheduler.is_resident("coder")
    assert text_backend.load_calls == []


async def test_generate_surfaces_vram_exhaustion_with_a_remedy(registry, events, clock):
    gpu = FakeGpuBackend(total_mb=SMALL_CARD_MB)
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events, clock=clock
    )
    await scheduler.load("coder", pin=True)
    with pytest.raises(VramExhausted) as excinfo:
        await scheduler.generate_image(ImageRequest(prompt="x"))
    assert "pinned" in excinfo.value.remedy


# -- HTTP level --------------------------------------------------------


async def test_route_returns_202_with_an_op_id(client):
    response = await client.post("/generate/image", json={"prompt": "a lantern"})
    assert response.status_code == 202
    assert response.json() == {"op_id": "op1"}


async def test_op_result_carries_the_artifact_url_and_evictions(client, app):
    await client.post("/generate/image", json={"prompt": "a lantern", "steps": 4})
    await settle(app)
    result = (await client.get("/ops/op1")).json()["result"]
    assert result["model"] == "painter"
    assert result["steps"] == 4
    assert result["evicted"] == []
    assert result["artifact_url"].startswith("/artifacts/")
    assert result["artifact_url"].endswith(".png")


async def test_op_result_carries_evictions_over_http(small_scheduler, ops, events, artifacts):
    # The VramScheduler-level eviction test (test_generate_evicts_to_make_room_and_
    # reports_it) never touches the client. The requirement is that the
    # *caller* learns an unload happened, so the asdict-through-JSON leg
    # -- the part that actually matters -- needs its own coverage.
    app = create_app(
        scheduler=small_scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as client:
        await client.post("/load", json={"model": "coder"})
        await settle(app)
        await client.post("/generate/image", json={"prompt": "a lantern"})
        await settle(app)
        result = (await client.get("/ops/op2")).json()["result"]

    assert result["evicted"] == [
        {"model": "coder", "freed_mb": 4700, "reason": "evicted"}
    ]


async def test_artifact_meta_records_the_resolved_size_not_the_unset_request(
    client, app, artifacts
):
    # ImageBody no longer defaults width/height, so an unspecified size
    # arrives as None. The sidecar must record what was actually rendered.
    await client.post("/generate/image", json={"prompt": "a lantern"})
    await settle(app)
    artifact_id = (await client.get("/ops/op1")).json()["result"]["artifact_id"]
    meta = artifacts.meta(artifact_id)
    assert meta["width"] == 1024
    assert meta["height"] == 1024
    assert meta["steps"] == 4


async def test_an_out_of_range_size_is_still_rejected(client):
    # Dropping the default must not drop the bounds on explicit values.
    response = await client.post(
        "/generate/image", json={"prompt": "x", "width": 4096}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "BAD_REQUEST"
    assert "width" in response.json()["remedy"]


async def test_artifact_is_downloadable(client, app):
    await client.post("/generate/image", json={"prompt": "a lantern"})
    await settle(app)
    url = (await client.get("/ops/op1")).json()["result"]["artifact_url"]
    response = await client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == FAKE_PNG


async def test_unknown_artifact_is_404(client):
    response = await client.get(f"/artifacts/{'0' * 32}.png")
    assert response.status_code == 404
    # Flat, exactly like a GpuServiceError's 500: one envelope for every failure.
    assert response.json()["code"] == "UNKNOWN_ARTIFACT"
    assert "remedy" in response.json()


async def test_diffusion_steps_are_emitted_as_op_progress(client, app, events):
    # Finite stream, same reason as test_events_streams_sse_frames: httpx's
    # ASGITransport runs the app to completion, so the bus must be closed to end it.
    async def act_then_close():
        while events.subscriber_count == 0:
            await asyncio.sleep(0)
        await client.post("/generate/image", json={"prompt": "a lantern", "steps": 2})
        await settle(app)
        events.close()

    actor = asyncio.create_task(act_then_close())
    response = await asyncio.wait_for(client.get("/events"), timeout=10)
    await actor
    frames = response.text.splitlines()

    # SSE frames are "event: <kind>", "data: <json>", "" — pair them by walking
    # the lines, not by slicing, so a blank line never shifts the alignment.
    parsed: list[tuple[str, dict]] = []
    kind: str | None = None
    for line in frames:
        if line.startswith("event: "):
            kind = line.removeprefix("event: ")
        elif line.startswith("data: ") and kind is not None:
            parsed.append((kind, json.loads(line.removeprefix("data: "))))
            kind = None

    progress = [body for k, body in parsed if k == "op.progress"]
    assert [p["step"] for p in progress] == [1, 2]
    assert all(p["total"] == 2 for p in progress)
    assert all(p["kind"] == "image" for p in progress)
    assert all(p["op_id"] == "op1" for p in progress)
