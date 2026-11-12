import pytest

from ohh_gpu.backends.base import ImageRequest
from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.errors import BackendFailure, BackendOom
from ohh_gpu.registry import ModelSpec

CODER = ModelSpec(name="coder", backend="fake", vram_mb=4700, role="code")
IMAGE = ModelSpec(name="painter", backend="fake", vram_mb=7000, role="image")


def test_starts_fully_free():
    gpu = FakeGpuBackend(total_mb=12288)
    assert gpu.total_mb() == 12288
    assert gpu.free_mb() == 12288
    assert gpu.name() == "FakeGpu"


async def test_load_consumes_vram():
    gpu = FakeGpuBackend(total_mb=12288)
    await gpu.load(CODER, progress=lambda step, total: None)
    assert gpu.free_mb() == 12288 - 4700


async def test_unload_returns_vram():
    gpu = FakeGpuBackend(total_mb=12288)
    await gpu.load(CODER, progress=lambda s, t: None)
    await gpu.unload(CODER)
    assert gpu.free_mb() == 12288


async def test_observed_cost_overrides_the_hint():
    gpu = FakeGpuBackend(total_mb=12288, observed_cost={"coder": 5200})
    await gpu.load(CODER, progress=lambda s, t: None)
    assert gpu.free_mb() == 12288 - 5200


async def test_load_beyond_capacity_raises_oom():
    gpu = FakeGpuBackend(total_mb=8192)
    await gpu.load(CODER, progress=lambda s, t: None)
    with pytest.raises(BackendOom) as excinfo:
        await gpu.load(IMAGE, progress=lambda s, t: None)
    assert excinfo.value.detail["requested_mb"] == 7000
    assert excinfo.value.detail["free_mb"] == 8192 - 4700


async def test_load_reports_monotonic_progress_to_completion():
    gpu = FakeGpuBackend(total_mb=12288)
    seen: list[tuple[int, int]] = []
    await gpu.load(CODER, progress=lambda step, total: seen.append((step, total)))
    assert seen[-1] == (100, 100)
    assert [step for step, _ in seen] == sorted(step for step, _ in seen)


async def test_generate_image_returns_png_bytes_and_steps():
    gpu = FakeGpuBackend(total_mb=12288)
    await gpu.load(IMAGE, progress=lambda s, t: None)
    seen: list[tuple[int, int]] = []
    data = await gpu.generate_image(
        IMAGE,
        ImageRequest(prompt="a lantern", steps=4),
        progress=lambda step, total: seen.append((step, total)),
    )
    assert data.startswith(b"\x89PNG")
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


async def test_generate_image_requires_the_model_resident():
    # BackendFailure, matching DiffusersBackend for the identical condition.
    # The fake previously raised BackendOom here; a fake that disagrees with
    # the real backend on error taxonomy is worse than no fake at all,
    # because BackendOom is the code that triggers the VramScheduler's
    # panic-evict-and-retry -- a recovery that cannot fix "never loaded".
    gpu = FakeGpuBackend(total_mb=12288)
    with pytest.raises(BackendFailure) as excinfo:
        await gpu.generate_image(
            IMAGE, ImageRequest(prompt="x"), progress=lambda s, t: None
        )
    assert excinfo.value.code == "BACKEND_FAILURE"


async def test_the_fake_and_the_real_backend_agree_on_the_not_resident_code():
    # Pins the two taxonomies together so they cannot drift apart again. The
    # real backend's residency check runs before it ever touches torch, so
    # this works on a machine with neither torch nor diffusers installed.
    from ohh_gpu.backends.diffusers import DiffusersBackend

    codes = set()
    for backend in (FakeGpuBackend(total_mb=12288), DiffusersBackend()):
        with pytest.raises(BackendFailure) as excinfo:
            await backend.generate_image(
                IMAGE, ImageRequest(prompt="x"), progress=lambda s, t: None
            )
        codes.add(excinfo.value.code)
    assert codes == {"BACKEND_FAILURE"}
