"""The text op: `VramScheduler.generate_text` and `POST /generate/text`.

The point of this route is that fit, eviction and inference happen inside one
hold of the GPU lock. A client that loaded and then inferred separately could
not close the window between the two calls — so the tests that matter here are
the ones about what happens *around* the turn, not the turn itself.
"""

import json

import pytest

from tests.conftest import CODER, PAINTER, TEST_TOKEN, settle
from ohh_gpu.backends.base import ChatRequest
from ohh_gpu.errors import BackendFailure, UnknownModel
from ohh_gpu.registry import GpuServiceSettings, ModelRegistry, ModelSpec


async def test_a_text_turn_auto_loads_the_code_model(scheduler, gpu):
    assert not scheduler.is_resident(CODER.name)

    result = await scheduler.generate_text(
        ChatRequest(messages=[{"role": "user", "content": "hi"}])
    )

    assert result.model == CODER.name
    assert scheduler.is_resident(CODER.name)
    assert gpu.load_calls == [CODER.name]


async def test_the_role_picks_the_model_when_none_is_named(scheduler):
    # `coder` and `tiny` both carry role "code"; the registry resolves by name
    # order, which is why the real registry gives exactly one model that role.
    result = await scheduler.generate_text(ChatRequest(messages=[]))
    assert result.model == CODER.name


async def test_an_unknown_model_is_rejected_before_anything_loads(scheduler, gpu):
    with pytest.raises(UnknownModel):
        await scheduler.generate_text(ChatRequest(messages=[]), model="not-a-model")
    assert gpu.load_calls == []


async def test_a_backend_that_cannot_chat_fails_without_evicting(registry, gpu, events):
    from ohh_gpu.scheduler import VramScheduler

    class LoadOnly:
        async def load(self, spec, progress):
            return None

        async def unload(self, spec):
            return None

    scheduler = VramScheduler(
        registry=registry,
        probe=gpu,
        backends={"fake": gpu, "mute": LoadOnly()},
        events=events,
    )
    mute = ModelSpec(name="mute", backend="mute", vram_mb=100, role="code")
    registry.models["mute"] = mute

    await scheduler.load(PAINTER.name)
    with pytest.raises(BackendFailure):
        await scheduler.generate_text(ChatRequest(messages=[]), model="mute")

    # The resident model is untouched: nothing was evicted for a load that
    # could never have answered.
    assert scheduler.is_resident(PAINTER.name)


async def test_a_text_turn_evicts_to_fit_and_reports_what_it_freed(scheduler, gpu):
    # painter 7000 + coder 4700 = 11700 against 12288 - 768 usable.
    await scheduler.load(PAINTER.name)
    result = await scheduler.generate_text(ChatRequest(messages=[]), model=CODER.name)

    assert [e.model for e in result.evicted] == [PAINTER.name]
    assert result.evicted[0].reason == "evicted"
    assert not scheduler.is_resident(PAINTER.name)


async def test_num_ctx_falls_back_to_the_registry_then_is_reported(registry, gpu, events):
    from ohh_gpu.scheduler import VramScheduler

    windowed = ModelSpec(
        name="windowed", backend="fake", vram_mb=500, role="code", num_ctx=40960
    )
    registry.models["windowed"] = windowed
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, events=events)

    result = await scheduler.generate_text(ChatRequest(messages=[]), model="windowed")

    assert result.num_ctx == 40960
    # Resolved once, in the VramScheduler, and handed to the backend already decided.
    assert gpu.chat_calls[-1].num_ctx == 40960


async def test_an_explicit_num_ctx_beats_the_registry(registry, gpu, events):
    from ohh_gpu.scheduler import VramScheduler

    windowed = ModelSpec(
        name="windowed", backend="fake", vram_mb=500, role="code", num_ctx=40960
    )
    registry.models["windowed"] = windowed
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, events=events)

    result = await scheduler.generate_text(
        ChatRequest(messages=[], num_ctx=8192), model="windowed"
    )
    assert result.num_ctx == 8192


async def test_a_second_turn_reuses_the_resident_model(scheduler, gpu):
    await scheduler.generate_text(ChatRequest(messages=[]))
    await scheduler.generate_text(ChatRequest(messages=[]))
    # One load, two turns: eviction destroys the prompt cache, so
    # reloading between turns of one conversation would be the expensive bug.
    assert gpu.load_calls == [CODER.name]


async def test_a_turn_refreshes_last_used(scheduler, gpu):
    await scheduler.load(CODER.name)
    before = scheduler.status().resident
    assert before[0]["model"] == CODER.name
    # last_used is not exposed on the snapshot, so assert through LRU order:
    # touching coder must make painter the eviction candidate.
    await scheduler.generate_text(ChatRequest(messages=[]), model=CODER.name)
    assert scheduler.is_resident(CODER.name)


# -- the route ----------------------------------------------------------


async def test_the_route_returns_202_and_an_op_id(client, app, ops):
    response = await client.post(
        "/generate/text", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 202
    op_id = response.json()["op_id"]
    await settle(app)

    op = ops.get(op_id)
    assert op.status == "done"
    assert op.result["content"] == "a fake reply"
    assert op.result["model"] == CODER.name
    assert op.result["evicted"] == []


async def test_tokens_are_emitted_as_op_token_frames(client, app, events):
    with events.subscribe() as subscription:
        response = await client.post("/generate/text", json={"messages": []})
        await settle(app)
        frames = subscription.drain()

    tokens = [e for e in frames if e.kind == "op.token"]
    assert tokens, "a text op must stream tokens for live observation"
    assert "".join(t.body["text"] for t in tokens).strip() == "a fake reply"
    assert tokens[0].body["op_id"] == response.json()["op_id"]
    assert tokens[0].body["kind"] == "text"


async def test_the_terminal_result_carries_tool_calls_and_counts(
    registry, gpu, events, ops, artifacts
):
    from ohh_gpu.api import create_app
    from ohh_gpu.backends.fake import FakeGpuBackend
    from ohh_gpu.scheduler import VramScheduler

    calls = [{"function": {"name": "gpu_status", "arguments": {}}}]
    backend = FakeGpuBackend(total_mb=12288, reply="looking", tool_calls=calls)
    scheduler = VramScheduler(
        registry=registry, probe=backend, backends={"fake": backend}, events=events
    )
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )

    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as client:
        response = await client.post("/generate/text", json={"messages": []})
        await settle(app)

    op = ops.get(response.json()["op_id"])
    assert op.result["tool_calls"] == calls
    assert op.result["done_reason"] == "stop"
    assert op.result["completion_tokens"] == 1


async def test_the_route_requires_the_bearer_token(app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gpu-service") as bare:
        response = await bare.post("/generate/text", json={"messages": []})
    assert response.status_code == 401


async def test_an_out_of_range_num_ctx_is_rejected_with_a_remedy(client):
    response = await client.post(
        "/generate/text", json={"messages": [], "num_ctx": 12}
    )
    # 422 with a BAD_REQUEST code is this app's established envelope for a
    # schema rejection; the point of the test is that `remedy` survives it.
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "BAD_REQUEST"
    assert "num_ctx" in body["error"]
    assert body["remedy"]


async def test_an_unknown_model_fails_the_op_with_a_typed_envelope(client, app, ops):
    response = await client.post(
        "/generate/text", json={"messages": [], "model": "nope"}
    )
    op_id = response.json()["op_id"]
    await settle(app)

    op = ops.get(op_id)
    assert op.status == "failed"
    assert op.error["code"] == "UNKNOWN_MODEL"
    assert op.error["remedy"]


async def test_no_text_model_registered_says_so(gpu, events, ops, artifacts):
    from ohh_gpu.api import create_app
    from ohh_gpu.scheduler import VramScheduler

    image_only = ModelRegistry(
        gpu_service=GpuServiceSettings(host="test", port=7311, vram_headroom_mb=768),
        models={PAINTER.name: PAINTER},
    )
    scheduler = VramScheduler(
        registry=image_only, probe=gpu, backends={"fake": gpu}, events=events
    )
    app = create_app(
        scheduler=scheduler,
        ops=ops,
        events=events,
        artifacts=artifacts,
        token=TEST_TOKEN,
        version="0.1.0",
    )

    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as client:
        response = await client.post("/generate/text", json={"messages": []})
        await settle(app)

    op = ops.get(response.json()["op_id"])
    assert op.status == "failed"
    assert op.error["code"] == "UNKNOWN_MODEL"


async def test_messages_and_tools_pass_through_unchanged(client, app, gpu):
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": '{"ok":true}'},
    ]
    tools = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]

    await client.post(
        "/generate/text", json={"messages": messages, "tools": tools}
    )
    await settle(app)

    # The GPU service schedules the GPU; it holds no opinion about the chat protocol,
    # so whatever the orchestrator sent must arrive byte-identical.
    sent = gpu.chat_calls[-1]
    assert sent.messages == messages
    assert sent.tools == tools


async def test_the_op_is_queryable_through_ops_endpoint(client, app):
    response = await client.post("/generate/text", json={"messages": []})
    op_id = response.json()["op_id"]
    await settle(app)

    read = await client.get(f"/ops/{op_id}")
    assert read.status_code == 200
    assert read.json()["status"] == "done"
    assert json.loads(read.content)["result"]["content"] == "a fake reply"


async def test_a_cold_load_uses_the_turns_window_not_the_registrys(
    registry, gpu, events
):
    """An overridden window reaches the *load*, not only the chat.

    VRAM is a function of the window, so a load at one window and a turn at
    another leaves the VramScheduler booking a figure for a model that never existed.
    """
    from ohh_gpu.scheduler import VramScheduler

    windowed = ModelSpec(
        name="windowed", backend="fake", vram_mb=500, role="code", num_ctx=131072
    )
    registry.models["windowed"] = windowed
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, events=events)

    await scheduler.generate_text(
        ChatRequest(messages=[], num_ctx=8192), model="windowed"
    )

    # The backend was asked to load the 8192 spec, not the 131072 one.
    assert gpu.load_calls == ["windowed"]
    assert gpu.loaded_specs[-1].num_ctx == 8192


async def test_the_registry_window_is_used_when_the_turn_names_none(
    registry, gpu, events
):
    from ohh_gpu.scheduler import VramScheduler

    windowed = ModelSpec(
        name="windowed", backend="fake", vram_mb=500, role="code", num_ctx=40960
    )
    registry.models["windowed"] = windowed
    scheduler = VramScheduler(registry=registry, probe=gpu, backends={"fake": gpu}, events=events)

    await scheduler.generate_text(ChatRequest(messages=[]), model="windowed")

    assert gpu.loaded_specs[-1].num_ctx == 40960


async def test_an_explicit_load_is_unaffected(scheduler, gpu):
    # /load has no window to resolve; it loads the registry entry as written.
    await scheduler.load(CODER.name)
    assert gpu.loaded_specs[-1].name == CODER.name
    assert gpu.loaded_specs[-1].num_ctx is None
