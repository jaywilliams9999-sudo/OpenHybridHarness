import pytest

from ohh_gpu.backends.fake import FakeGpuBackend
from ohh_gpu.errors import UnknownModel, VramExhausted
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry, ModelSpec
from ohh_gpu.scheduler import VramScheduler


def build(registry, gpu, clock) -> VramScheduler:
    return VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)


def test_usable_free_excludes_headroom(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    assert scheduler.usable_free_mb() == 12288 - 768


async def test_load_loads_when_it_fits(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    result = await scheduler.load("coder")
    assert result.model == "coder"
    assert result.vram_mb == 4700
    assert result.evicted == []
    assert result.already_resident is False
    assert gpu.load_calls == ["coder"]


async def test_load_is_idempotent_for_a_resident_model(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    await scheduler.load("coder")
    result = await scheduler.load("coder")
    assert result.already_resident is True
    assert gpu.load_calls == ["coder"]


async def test_load_records_observed_cost_over_the_hint(registry, clock):
    gpu = FakeGpuBackend(total_mb=12288, observed_cost={"coder": 5210})
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    result = await scheduler.load("coder")
    assert result.vram_mb == 5210
    assert scheduler.observed_mb("coder") == 5210


def test_observed_mb_falls_back_to_the_registry_hint(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    assert scheduler.observed_mb("painter") == 7000


async def test_unload_frees_and_reports(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    await scheduler.load("coder")
    freed = await scheduler.unload("coder")
    assert freed == 4700
    assert gpu.unload_calls == ["coder"]
    assert scheduler.status().resident == []


async def test_unload_of_absent_model_frees_nothing(registry, gpu, clock):
    # Registered but not resident: 0 is the truthful answer, and the name
    # check below must not turn this into an error.
    scheduler = build(registry, gpu, clock)
    assert await scheduler.unload("coder") == 0


async def test_unload_of_an_unknown_model_raises_instead_of_claiming_success(
    registry, gpu, clock
):
    # load validates through registry.require; unload went straight to the
    # residency dict, so a typo'd name returned 0 and the caller concluded
    # VRAM had been freed. Same validation, same error, either way in.
    scheduler = build(registry, gpu, clock)
    with pytest.raises(UnknownModel) as excinfo:
        await scheduler.unload("qwen2.5-codr:7b")
    assert excinfo.value.detail["model"] == "qwen2.5-codr:7b"
    assert "coder" in excinfo.value.remedy
    assert gpu.unload_calls == []


async def test_unload_validation_does_not_re_enter_the_gpu_lock(registry, gpu, clock):
    # _unload_locked runs *inside* the single GPU lock; an accidental
    # re-acquisition there hangs forever rather than failing. Prove the lock
    # is released afterwards by doing more GPU work.
    import asyncio

    scheduler = build(registry, gpu, clock)
    await scheduler.load("coder")
    with pytest.raises(UnknownModel):
        await asyncio.wait_for(scheduler.unload("nonesuch"), timeout=5)
    assert await asyncio.wait_for(scheduler.unload("coder"), timeout=5) == 4700


async def test_unknown_model_raises(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    with pytest.raises(UnknownModel):
        await scheduler.load("nonesuch")


async def test_status_reports_vram_and_residency(registry, gpu, clock):
    scheduler = build(registry, gpu, clock)
    await scheduler.load("coder", pin=True)
    snapshot = scheduler.status()
    assert snapshot.vram["total_mb"] == 12288
    assert snapshot.vram["free_mb"] == 12288 - 4700
    assert snapshot.vram["headroom_mb"] == 768
    (entry,) = snapshot.resident
    assert {k: entry[k] for k in ("model", "vram_mb", "pinned", "role")} == {
        "model": "coder", "vram_mb": 4700, "pinned": True, "role": "code"
    }
    assert entry["load_ms"] == 0 and entry["idle_s"] >= 0
    assert snapshot.busy is False
    assert snapshot.queue_depth == 0


async def test_model_too_large_for_the_card_raises_vram_exhausted(registry, clock):
    gpu = FakeGpuBackend(total_mb=4096)
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, clock=clock)
    with pytest.raises(VramExhausted) as excinfo:
        await scheduler.load("painter")
    assert excinfo.value.detail["requested_mb"] == 7000
    assert "do not retry" in excinfo.value.remedy


# -- window-aware fit planning ------------------------------------------

WINDOWED = ModelSpec(
    name="windowed",
    backend="fake",
    vram_mb=7581,
    role="code",
    num_ctx=65536,
    kv_rates={"f16": 134, "q8_0": 67, "q4_0": 34},
    measured_kv_cache="q4_0",
)


def _scheduler(gpu, events, *models):
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768),
        models={m.name: m for m in models},
    )
    return VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events
    )


def test_the_hint_is_the_recorded_figure_at_the_recorded_window(gpu, events):
    scheduler = _scheduler(gpu, events, WINDOWED)
    assert scheduler.observed_mb("windowed") == 7581
    assert scheduler.observed_mb("windowed", 65536) == 7581


def test_the_hint_rebases_onto_a_larger_window(gpu, events):
    scheduler = _scheduler(gpu, events, WINDOWED)
    # +65536 tokens at 34 MB/1k = +2228 MB.
    assert scheduler.observed_mb("windowed", 131072) == 7581 + 2228


def test_the_hint_rebases_onto_a_smaller_window(gpu, events):
    scheduler = _scheduler(gpu, events, WINDOWED)
    # This is the case that failed in production: a small window planned
    # against a large-window figure refused a load that would have fit.
    assert scheduler.observed_mb("windowed", 16384) == 7581 - round(34 * 49152 / 1000)


def test_a_model_without_a_kv_rate_keeps_the_flat_figure(gpu, events):
    flat = ModelSpec(
        name="flat", backend="fake", vram_mb=5000, role="code", num_ctx=8192
    )
    scheduler = _scheduler(gpu, events, flat)
    # Unchanged behaviour for every entry that declares no rate.
    assert scheduler.observed_mb("flat", 131072) == 5000


def test_the_hint_never_goes_below_one_mb(gpu, events):
    scheduler = _scheduler(gpu, events, WINDOWED)
    # A window small enough to imply negative VRAM is nonsense, but the planner
    # must not hand a negative requirement to the eviction arithmetic.
    assert scheduler.observed_mb("windowed", 1) >= 1


async def test_a_measurement_is_keyed_to_the_window_it_was_taken_at(gpu, events):
    scheduler = _scheduler(gpu, events, WINDOWED)
    await scheduler.load("windowed")

    measured = scheduler.observed_mb("windowed", 65536)
    # The fake books spec.vram_mb, so the measurement equals the hint here;
    # what matters is that a *different* window does not read this entry.
    assert measured == 7581
    assert scheduler.observed_mb("windowed", 131072) == 7581 + 2228


# -- fits from the card ---------------------------------------------------

from ohh_gpu.backends.base import ChatRequest  # noqa: E402
from ohh_gpu.errors import ModelUnavailable  # noqa: E402
from ohh_gpu.fit import Fit  # noqa: E402

FITTED = ModelSpec(
    name="fitted", backend="fake", vram_mb=500, role="code", num_ctx=40960
)


def _fitted_scheduler(gpu, fit):
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768),
        models={"fitted": FITTED},
    )
    return VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, fits={"fitted": fit}
    )


def _req(num_ctx=None):
    return ChatRequest(messages=[{"role": "user", "content": "hi"}], num_ctx=num_ctx)


async def test_generate_text_on_an_unavailable_model_raises_with_the_fit_reason(gpu):
    reason = "needs more memory than this card has free"
    scheduler = _fitted_scheduler(
        gpu, Fit("fitted", False, None, None, reason)
    )
    with pytest.raises(ModelUnavailable) as excinfo:
        await scheduler.generate_text(_req(), model="fitted")
    assert str(excinfo.value) == f"fitted isn't available on this card: {reason}"
    assert excinfo.value.detail["model"] == "fitted"
    assert gpu.load_calls == []
    with pytest.raises(ModelUnavailable):
        await scheduler.load("fitted")


async def test_generate_text_defaults_to_the_fitted_window(gpu):
    scheduler = _fitted_scheduler(gpu, Fit("fitted", True, 16384, 40960, "x"))
    result = await scheduler.generate_text(_req(), model="fitted")
    assert result.num_ctx == 16384
    assert gpu.chat_calls[-1].num_ctx == 16384


async def test_a_smaller_explicit_request_still_wins(gpu):
    scheduler = _fitted_scheduler(gpu, Fit("fitted", True, 16384, 40960, "x"))
    result = await scheduler.generate_text(_req(8192), model="fitted")
    assert result.num_ctx == 8192


async def test_a_request_above_the_fitted_window_is_clamped(gpu):
    scheduler = _fitted_scheduler(gpu, Fit("fitted", True, 16384, 40960, "x"))
    result = await scheduler.generate_text(_req(32768), model="fitted")
    assert result.num_ctx == 16384
    assert gpu.chat_calls[-1].num_ctx == 16384


async def test_load_uses_the_fitted_window_and_books_its_hint(gpu):
    llama = ModelSpec(
        name="llama", backend="fake", vram_mb=9809, role="code", num_ctx=131072,
        kv_rates={"f16": 134, "q8_0": 67, "q4_0": 34}, measured_kv_cache="q4_0",
    )
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768),
        models={"llama": llama},
    )
    # 8 GB: the 131072-window hint (9809) could never fit; the fitted one does.
    gpu = FakeGpuBackend(total_mb=8192, observed_cost={"llama": 6397})
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu},
        fits={"llama": Fit("llama", True, 30720, 131072, "x")},
    )
    # 9809 - 34 * (131072 - 30720) / 1000 = 6397
    assert scheduler.observed_mb("llama", 30720) == 6397
    await scheduler.load("llama")
    assert gpu.loaded_specs[-1].num_ctx == 30720


async def test_generate_image_on_an_unavailable_model_raises_before_any_backend_call(gpu):
    painter = ModelSpec(name="painter", backend="fake", vram_mb=7000, role="image")
    registry = ModelRegistry(
        gpu_service=GpuServiceSettings(host="t", port=1, vram_headroom_mb=768),
        models={"painter": painter},
    )
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu},
        fits={"painter": Fit("painter", False, None, None, "needs more memory than this card has free")},
    )
    from ohh_gpu.backends.base import ImageRequest

    with pytest.raises(ModelUnavailable):
        await scheduler.generate_image(ImageRequest(prompt="x"), model="painter")
    assert gpu.load_calls == []
