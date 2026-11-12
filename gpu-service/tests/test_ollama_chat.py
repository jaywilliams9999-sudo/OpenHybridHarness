"""The Ollama backend's text turn.

Ollama streams newline-delimited JSON from ``/api/chat`` rather than SSE, so
these tests feed it exactly that: one JSON object per line, the last carrying
``done`` and the token counts.
"""

import json

import httpx
import pytest

from ohh_gpu.backends.base import ChatRequest
from ohh_gpu.backends.ollama import OllamaBackend
from ohh_gpu.errors import BackendFailure, BackendOom
from ohh_gpu.registry import ModelSpec

LLAMA = ModelSpec(
    name="llama3.1:8b", backend="ollama", vram_mb=9692, role="code", num_ctx=131072
)


def ndjson(*frames: dict) -> bytes:
    return "".join(json.dumps(f) + "\n" for f in frames).encode()


def streaming(body: bytes, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(status, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def capturing(body: bytes, seen: list[dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


DONE = {
    "message": {"content": ""},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 51662,
    "eval_count": 12,
}


async def test_tokens_stream_and_the_whole_reply_comes_back():
    body = ndjson(
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"message": {"content": " there"}},
        DONE,
    )
    tokens: list[str] = []
    backend = OllamaBackend(client=streaming(body))

    result = await backend.chat(
        LLAMA, ChatRequest(messages=[{"role": "user", "content": "hi"}]), tokens.append
    )

    assert tokens == ["Hel", "lo", " there"]
    # The authoritative text is assembled here, not by the caller: the event bus
    # that carries tokens onward drops frames under backpressure.
    assert result.content == "Hello there"
    assert result.prompt_tokens == 51662
    assert result.completion_tokens == 12
    assert result.done_reason == "stop"


async def test_tool_calls_survive_the_trip():
    body = ndjson(
        {"message": {"content": "checking"}},
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "gpu_status", "arguments": {}}},
                ],
            }
        },
        DONE,
    )
    backend = OllamaBackend(client=streaming(body))
    result = await backend.chat(
        LLAMA, ChatRequest(messages=[{"role": "user", "content": "go"}]), lambda _: None
    )

    assert result.tool_calls == [{"function": {"name": "gpu_status", "arguments": {}}}]


async def test_tool_calls_on_the_final_frame_are_not_missed():
    # Ollama may attach tool_calls to the frame that also carries `done`.
    body = ndjson(
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "echo", "arguments": {"a": 1}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 1,
        }
    )
    backend = OllamaBackend(client=streaming(body))
    result = await backend.chat(
        LLAMA, ChatRequest(messages=[]), lambda _: None
    )
    assert len(result.tool_calls) == 1


async def test_num_ctx_and_sampling_reach_ollama_as_options():
    seen: list[dict] = []
    backend = OllamaBackend(client=capturing(ndjson(DONE), seen), keep_alive="30m")

    await backend.chat(
        LLAMA,
        ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            num_ctx=131072,
            temperature=0.2,
            max_tokens=500,
        ),
        lambda _: None,
    )

    assert seen[0]["options"] == {
        "num_ctx": 131072,
        "temperature": 0.2,
        "num_predict": 500,
    }
    assert seen[0]["stream"] is True
    # Residency belongs to the VramScheduler: answering a prompt must not evict the
    # model that answered it.
    assert seen[0]["keep_alive"] == "30m"


async def test_no_options_key_when_nothing_was_asked_for():
    seen: list[dict] = []
    backend = OllamaBackend(client=capturing(ndjson(DONE), seen))
    await backend.chat(LLAMA, ChatRequest(messages=[]), lambda _: None)
    assert "options" not in seen[0]
    assert "tools" not in seen[0]


async def test_tools_are_forwarded_only_when_present():
    seen: list[dict] = []
    tools = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]
    backend = OllamaBackend(client=capturing(ndjson(DONE), seen))
    await backend.chat(LLAMA, ChatRequest(messages=[], tools=tools), lambda _: None)
    assert seen[0]["tools"] == tools


async def test_a_blank_or_malformed_line_does_not_fail_the_turn():
    body = (
        b"\n"
        + b"not json\n"
        + json.dumps({"message": {"content": "ok"}}).encode()
        + b"\n"
        + json.dumps(DONE).encode()
        + b"\n"
    )
    backend = OllamaBackend(client=streaming(body))
    result = await backend.chat(LLAMA, ChatRequest(messages=[]), lambda _: None)
    assert result.content == "ok"


async def test_a_missing_done_frame_leaves_counts_at_zero_not_wrong():
    # A truncated stream is reported as zero tokens rather than a guess: the
    # Ledger would otherwise record a fabricated count as evidence.
    body = ndjson({"message": {"content": "partial"}})
    backend = OllamaBackend(client=streaming(body))
    result = await backend.chat(LLAMA, ChatRequest(messages=[]), lambda _: None)
    assert result.content == "partial"
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)


async def test_an_oom_mid_turn_is_a_backend_oom():
    backend = OllamaBackend(
        client=streaming(
            json.dumps({"error": "CUDA out of memory"}).encode(), status=500
        )
    )
    with pytest.raises(BackendOom):
        await backend.chat(LLAMA, ChatRequest(messages=[]), lambda _: None)


async def test_a_model_without_tool_support_is_a_backend_failure():
    backend = OllamaBackend(
        client=streaming(
            json.dumps({"error": "llama3.2 does not support tools"}).encode(),
            status=400,
        )
    )
    with pytest.raises(BackendFailure) as caught:
        await backend.chat(LLAMA, ChatRequest(messages=[]), lambda _: None)
    assert "tools" in caught.value.to_dict()["remedy"]


async def test_an_unreachable_ollama_names_the_base_url():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = OllamaBackend(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="http://127.0.0.1:11434",
    )
    with pytest.raises(BackendFailure) as caught:
        await backend.chat(LLAMA, ChatRequest(messages=[]), lambda _: None)
    assert caught.value.to_dict()["base_url"] == "http://127.0.0.1:11434"


# -- the load window ----------------------------------------------------


async def test_load_applies_the_models_context_window():
    """The window is part of what the VramScheduler measures, not a chat-time detail.

    Without this the VramScheduler books the model at Ollama's own default window,
    then the first chat supplies the real one and Ollama reloads the runner at
    up to twice the size -- so the booked figure describes a model that was
    never resident.
    """
    seen: list[dict] = []
    backend = OllamaBackend(client=capturing(ndjson(DONE), seen), keep_alive="30m")

    await backend.load(LLAMA, progress=lambda s, t: None)

    assert seen[0]["options"] == {"num_ctx": 131072}
    assert seen[0]["keep_alive"] == "30m"


async def test_load_sends_no_options_for_a_model_with_no_window():
    seen: list[dict] = []
    imageish = ModelSpec(name="m", backend="ollama", vram_mb=100, role="code")
    backend = OllamaBackend(client=capturing(ndjson(DONE), seen))

    await backend.load(imageish, progress=lambda s, t: None)

    assert "options" not in seen[0]


async def test_unload_still_carries_no_options():
    # keep_alive=0 needs no window; sending one would reload the model we are
    # trying to evict.
    seen: list[dict] = []
    backend = OllamaBackend(
        client=capturing(ndjson(DONE), seen), unload_max_polls=0
    )
    await backend.unload(LLAMA)
    assert "options" not in seen[0]
    assert seen[0]["keep_alive"] == 0
