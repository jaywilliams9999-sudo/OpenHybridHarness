import sys
import types

import pytest

from ohh_gpu.errors import BackendFailure, BackendOom
from ohh_gpu.registry import ModelSpec

PAINTER = ModelSpec(
    name="sdxl-turbo",
    backend="diffusers",
    vram_mb=7000,
    role="image",
    repo="stabilityai/sdxl-turbo",
    dtype="fp16",
    steps=4,
)


class StubImage:
    def save(self, buffer, format):  # noqa: A002 - mirrors PIL's signature
        buffer.write(b"\x89PNG\r\n\x1a\nstub")


class StubPipeline:
    def __init__(self, repo, **kwargs):
        self.repo = repo
        self.kwargs = kwargs
        self.moved_to = None
        self.calls: list[dict] = []

    def to(self, device):
        self.moved_to = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        total = kwargs["num_inference_steps"]
        for step in range(total):
            if callback:
                callback(self, step, 0.0, {})
        return types.SimpleNamespace(images=[StubImage()])


@pytest.fixture
def stub_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    empty_cache_calls: list[None] = []
    torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: empty_cache_calls.append(None),
        empty_cache_calls=empty_cache_calls,
        mem_get_info=lambda index=0: (4 * 1024**3, 12 * 1024**3),
        get_device_name=lambda index=0: "NVIDIA GeForce RTX 3060",
        is_available=lambda: True,
    )

    class OutOfMemoryError(RuntimeError):
        pass

    class StubGenerator:
        def __init__(self, device=None):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    torch.OutOfMemoryError = OutOfMemoryError
    torch.Generator = StubGenerator
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def stub_diffusers(monkeypatch):
    module = types.ModuleType("diffusers")
    created: list[StubPipeline] = []

    class AutoPipelineForText2Image:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            pipeline = StubPipeline(repo, **kwargs)
            created.append(pipeline)
            return pipeline

    module.AutoPipelineForText2Image = AutoPipelineForText2Image
    module.created = created
    monkeypatch.setitem(sys.modules, "diffusers", module)
    return module


async def test_load_builds_the_pipeline_from_the_registry_recipe(
    stub_torch, stub_diffusers
):
    from ohh_gpu.backends.diffusers import DiffusersBackend

    backend = DiffusersBackend()
    await backend.load(PAINTER, progress=lambda s, t: None)

    pipeline = stub_diffusers.created[0]
    assert pipeline.repo == "stabilityai/sdxl-turbo"
    assert pipeline.kwargs["dtype"] == "float16"
    assert pipeline.moved_to == "cuda"


async def test_fp32_dtype_is_honoured(stub_torch, stub_diffusers):
    from ohh_gpu.backends.diffusers import DiffusersBackend

    spec = ModelSpec(
        name="big", backend="diffusers", vram_mb=1, role="image", repo="r", dtype="fp32"
    )
    backend = DiffusersBackend()
    await backend.load(spec, progress=lambda s, t: None)
    assert stub_diffusers.created[0].kwargs["dtype"] == "float32"


async def test_an_unknown_dtype_is_refused_rather_than_silently_fp16(
    stub_torch, stub_diffusers
):
    # DTYPES.get(..., "float16") used to swallow a typo: `dtype = "fp61"`
    # loaded in fp16 with no signal at all, so a registry that says fp32 and
    # runs fp16 looks identical to one that works.
    from ohh_gpu.backends.diffusers import DiffusersBackend

    spec = ModelSpec(
        name="typo", backend="diffusers", vram_mb=1, role="image", repo="r", dtype="fp61"
    )
    with pytest.raises(BackendFailure) as excinfo:
        await DiffusersBackend().load(spec, progress=lambda s, t: None)
    error = excinfo.value
    assert "fp61" in error.message
    # The remedy has to name the keys that would work.
    for valid in ("fp16", "fp32", "bf16"):
        assert valid in error.remedy
    assert stub_diffusers.created == []


async def test_an_absent_dtype_still_means_fp16(stub_torch, stub_diffusers):
    from ohh_gpu.backends.diffusers import DiffusersBackend

    spec = ModelSpec(name="plain", backend="diffusers", vram_mb=1, role="image", repo="r")
    await DiffusersBackend().load(spec, progress=lambda s, t: None)
    assert stub_diffusers.created[0].kwargs["dtype"] == "float16"


async def test_missing_repo_fails_before_touching_the_gpu(stub_torch, stub_diffusers):
    from ohh_gpu.backends.diffusers import DiffusersBackend

    spec = ModelSpec(name="norepo", backend="diffusers", vram_mb=1, role="image")
    with pytest.raises(BackendFailure) as excinfo:
        await DiffusersBackend().load(spec, progress=lambda s, t: None)
    assert "repo" in excinfo.value.remedy
    assert stub_diffusers.created == []


async def test_generate_reports_each_step(stub_torch, stub_diffusers):
    from ohh_gpu.backends.base import ImageRequest
    from ohh_gpu.backends.diffusers import DiffusersBackend

    backend = DiffusersBackend()
    await backend.load(PAINTER, progress=lambda s, t: None)
    seen: list[tuple[int, int]] = []
    png = await backend.generate_image(
        PAINTER,
        ImageRequest(prompt="a lantern", steps=4),
        progress=lambda step, total: seen.append((step, total)),
    )
    assert png.startswith(b"\x89PNG")
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


async def test_generate_passes_prompt_size_and_seed(stub_torch, stub_diffusers):
    from ohh_gpu.backends.base import ImageRequest
    from ohh_gpu.backends.diffusers import DiffusersBackend

    backend = DiffusersBackend()
    await backend.load(PAINTER, progress=lambda s, t: None)
    await backend.generate_image(
        PAINTER,
        ImageRequest(prompt="a lantern", negative="blur", steps=2, width=512, height=768, seed=7),
        progress=lambda s, t: None,
    )
    call = stub_diffusers.created[0].calls[0]
    assert call["prompt"] == "a lantern"
    assert call["negative_prompt"] == "blur"
    assert call["num_inference_steps"] == 2
    assert call["width"] == 512
    assert call["height"] == 768
    assert call["generator"] is not None
    assert call["generator"].seed == 7


async def test_generate_without_a_seed_passes_no_generator(stub_torch, stub_diffusers):
    from ohh_gpu.backends.base import ImageRequest
    from ohh_gpu.backends.diffusers import DiffusersBackend

    backend = DiffusersBackend()
    await backend.load(PAINTER, progress=lambda s, t: None)
    await backend.generate_image(
        PAINTER,
        ImageRequest(prompt="a lantern", steps=2),
        progress=lambda s, t: None,
    )
    call = stub_diffusers.created[0].calls[0]
    assert call["generator"] is None


async def test_generate_without_load_raises_backend_failure(stub_torch, stub_diffusers):
    from ohh_gpu.backends.base import ImageRequest
    from ohh_gpu.backends.diffusers import DiffusersBackend

    with pytest.raises(BackendFailure):
        await DiffusersBackend().generate_image(
            PAINTER, ImageRequest(prompt="x"), progress=lambda s, t: None
        )


async def test_torch_oom_becomes_backend_oom(stub_torch, stub_diffusers, monkeypatch):
    from ohh_gpu.backends.diffusers import DiffusersBackend

    def explode(repo, **kwargs):
        raise stub_torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(
        sys.modules["diffusers"].AutoPipelineForText2Image, "from_pretrained", explode
    )
    with pytest.raises(BackendOom):
        await DiffusersBackend().load(PAINTER, progress=lambda s, t: None)


async def test_torch_cuda_oom_becomes_backend_oom_when_no_top_level_alias(
    stub_diffusers, monkeypatch
):
    """torch.OutOfMemoryError is the recent name; older/other builds only
    have torch.cuda.OutOfMemoryError. Catching the wrong one turns a
    recoverable OOM into an AttributeError raised while handling it, which
    silently disables the VramScheduler's panic-evict-and-retry recovery."""
    from ohh_gpu.backends.diffusers import DiffusersBackend

    class CudaOutOfMemoryError(RuntimeError):
        pass

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        mem_get_info=lambda index=0: (4 * 1024**3, 12 * 1024**3),
        get_device_name=lambda index=0: "NVIDIA GeForce RTX 3060",
        is_available=lambda: True,
        OutOfMemoryError=CudaOutOfMemoryError,
    )
    # Deliberately no torch.OutOfMemoryError at module scope.
    monkeypatch.setitem(sys.modules, "torch", torch)

    def explode(repo, **kwargs):
        raise CudaOutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(
        sys.modules["diffusers"].AutoPipelineForText2Image, "from_pretrained", explode
    )
    with pytest.raises(BackendOom):
        await DiffusersBackend().load(PAINTER, progress=lambda s, t: None)


async def test_unload_frees_the_pipeline_and_empties_the_cuda_cache(
    stub_torch, stub_diffusers
):
    from ohh_gpu.backends.diffusers import DiffusersBackend

    backend = DiffusersBackend()
    await backend.load(PAINTER, progress=lambda s, t: None)
    assert PAINTER.name in backend._pipelines

    await backend.unload(PAINTER)

    assert PAINTER.name not in backend._pipelines
    assert len(stub_torch.cuda.empty_cache_calls) == 1

    # Unloading an already-unloaded model is a no-op: nothing to free, so no
    # second empty_cache call.
    await backend.unload(PAINTER)
    assert len(stub_torch.cuda.empty_cache_calls) == 1




async def test_variant_selects_which_weight_files_are_fetched(
    stub_torch, stub_diffusers
):
    """`dtype` and `variant` are different knobs.

    dtype is what the weights are held as once loaded; variant is which files
    come down from the hub. With only dtype set, sdxl-turbo pulls the fp32
    weights and casts them -- measured at 13.01GB of cache for a model whose
    fp16 files are roughly half that.
    """
    from ohh_gpu.backends.diffusers import DiffusersBackend

    spec = ModelSpec(
        name="painter",
        backend="diffusers",
        vram_mb=1,
        role="image",
        repo="stabilityai/sdxl-turbo",
        dtype="fp16",
        variant="fp16",
    )
    await DiffusersBackend().load(spec, progress=lambda s, t: None)

    kwargs = stub_diffusers.created[0].kwargs
    assert kwargs["dtype"] == "float16"
    assert kwargs["variant"] == "fp16"


async def test_variant_is_omitted_when_unset(stub_torch, stub_diffusers):
    """Opt-in, never inferred from dtype.

    Not every repo publishes an fp16 variant, so passing variant= on the
    strength of dtype alone turns a working load into a 404.
    """
    from ohh_gpu.backends.diffusers import DiffusersBackend

    await DiffusersBackend().load(PAINTER, progress=lambda s, t: None)
    assert "variant" not in stub_diffusers.created[0].kwargs


def test_registry_reads_the_variant_key(tmp_path):
    from ohh_gpu.registry import load_registry

    path = tmp_path / "models.toml"
    path.write_text(
        """
[gpu_service]
host = "h"
port = 1
vram_headroom_mb = 1

[models."p"]
backend = "diffusers"
vram_mb = 1
variant = "fp16"
""",
        encoding="utf-8",
    )
    assert load_registry(path).models["p"].variant == "fp16"
