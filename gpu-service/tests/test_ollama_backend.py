import json

import httpx
import pytest

from ohh_gpu.backends.ollama import OllamaBackend
from ohh_gpu.errors import BackendFailure, BackendOom
from ohh_gpu.registry import ModelSpec

CODER = ModelSpec(name="qwen2.5-coder:7b", backend="ollama", vram_mb=4700, role="code")


def stub(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_load_issues_an_empty_generate_with_keep_alive():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.url.path == "/api/generate"
        return httpx.Response(200, json={"done": True})

    backend = OllamaBackend(client=stub(handler), keep_alive="30m")
    await backend.load(CODER, progress=lambda s, t: None)

    assert seen == [
        {"model": "qwen2.5-coder:7b", "prompt": "", "stream": False, "keep_alive": "30m"}
    ]


async def test_load_reports_progress_to_completion():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    seen: list[tuple[int, int]] = []
    backend = OllamaBackend(client=stub(handler))
    await backend.load(CODER, progress=lambda step, total: seen.append((step, total)))
    assert seen[0] == (0, 100)
    assert seen[-1] == (100, 100)


async def test_unload_sets_keep_alive_to_zero():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    backend = OllamaBackend(client=stub(handler))
    await backend.unload(CODER)
    assert seen[0]["keep_alive"] == 0


async def test_out_of_memory_response_becomes_backend_oom():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model requires more system memory"})

    backend = OllamaBackend(client=stub(handler))
    with pytest.raises(BackendOom) as excinfo:
        await backend.load(CODER, progress=lambda s, t: None)
    assert excinfo.value.detail["model"] == "qwen2.5-coder:7b"


async def test_other_errors_become_backend_failure_with_a_remedy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen2.5-coder:7b' not found"})

    backend = OllamaBackend(client=stub(handler))
    with pytest.raises(BackendFailure) as excinfo:
        await backend.load(CODER, progress=lambda s, t: None)
    assert "ollama pull" in excinfo.value.remedy


async def test_connection_refused_becomes_backend_failure_not_a_traceback():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    backend = OllamaBackend(client=stub(handler))
    with pytest.raises(BackendFailure) as excinfo:
        await backend.load(CODER, progress=lambda s, t: None)
    assert "Ollama" in excinfo.value.message
    assert "without" in excinfo.value.remedy.lower()


async def test_unload_waits_until_ollama_reports_the_model_gone():
    """keep_alive=0 is an instruction, not a completion.

    Ollama acknowledges the unload and frees the device afterwards, so a
    VramScheduler that probes VRAM the moment unload() returns reads a card that
    has not given the memory back. Observed on the real 3060: unloading a
    4871MB model emitted vram.changed with byte-identical figures to the
    reading before it -- the event claimed to free 4.8GB and simultaneously
    reported no additional free memory.
    """
    paths: list[str] = []
    resident = [True, True, False]

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/ps":
            still_there = resident.pop(0) if resident else False
            models = [{"model": CODER.name}] if still_there else []
            return httpx.Response(200, json={"models": models})
        return httpx.Response(200, json={"done": True})

    backend = OllamaBackend(client=stub(handler), unload_poll_interval_s=0)
    await backend.unload(CODER)

    assert paths[0] == "/api/generate"
    assert paths.count("/api/ps") == 3, paths


async def test_unload_returns_when_ollama_never_releases_the_model():
    """Bounded, and never raises.

    The VramScheduler evicts in order to make room, and has already dropped the
    residency record by the time unload runs. Raising here would fail the
    load that the eviction existed to serve, so a stuck Ollama degrades to
    the old behaviour rather than breaking the scheduler.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"model": CODER.name}]})
        return httpx.Response(200, json={"done": True})

    backend = OllamaBackend(
        client=stub(handler), unload_poll_interval_s=0, unload_max_polls=4
    )
    await backend.unload(CODER)  # must not raise, must not hang


async def test_unload_tolerates_a_failing_ps_probe():
    """A unload must not fail because the progress check did."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"done": True})

    backend = OllamaBackend(client=stub(handler), unload_poll_interval_s=0)
    await backend.unload(CODER)
