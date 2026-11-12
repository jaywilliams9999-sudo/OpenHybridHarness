import asyncio
import json

import pytest

from tests.conftest import settle


async def test_health_reports_the_card(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["gpu"] == "FakeGpu"
    assert body["vram_total_mb"] == 12288
    assert body["vram_free_mb"] == 12288
    assert body["version"] == "0.1.0"


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/status",
        "/ops/op1",
        "/events",
        # The artifact route serves file bytes off disk. If auth ever stops
        # covering it, the leak is the generated images themselves.
        f"/artifacts/{'0' * 32}.png",
    ],
)
async def test_every_route_requires_a_bearer_token(app, path):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gpu-service") as bare:
        response = await bare.get(path)
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize(
    "path,body",
    [
        ("/load", {"model": "coder"}),
        ("/unload", {"model": "coder"}),
        ("/generate/image", {"prompt": "a lantern"}),
    ],
)
async def test_the_mutating_post_routes_also_require_a_bearer_token(app, path, body):
    # These are the state-mutating, GPU-consuming routes. They pass the
    # global auth dependency today, but that is exactly the thing a future
    # refactor (e.g. moving auth onto router-level includes) could silently
    # break for POST while the GET-route tests above kept passing.
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gpu-service") as bare:
        response = await bare.post(path, json=body)
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


async def test_a_wrong_token_is_rejected(app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": "Bearer wrong"},
    ) as bare:
        response = await bare.get("/health")
    assert response.status_code == 401


async def test_a_non_ascii_bearer_value_is_rejected_not_crashed(app):
    # secrets.compare_digest raises TypeError on two `str` when either holds a
    # character above U+007F, and Starlette latin-1-decodes headers -- so one
    # 0x80+ byte from an unauthenticated caller used to 500 every route in the
    # GPU service, /health included. It must be an ordinary 401.
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        # Raw bytes on the wire, which is the only way such a header can
        # arrive: httpx refuses to ascii-encode it for us, and a real client
        # (or an attacker) is under no such constraint.
        headers={"Authorization": b"Bearer \xff\xe9"},
    ) as bare:
        response = await bare.get("/health")
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


async def test_a_non_ascii_bearer_value_is_rejected_on_every_route(app):
    # Auth is an app-level dependency, so the crash was never route-specific.
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": b"Bearer \xff"},
    ) as bare:
        for path in ("/health", "/status", "/ops/op1"):
            assert (await bare.get(path)).status_code == 401
        assert (await bare.post("/load", json={"model": "coder"})).status_code == 401


async def test_status_reports_vram_and_residency(client, app):
    await client.post("/load", json={"model": "coder"})
    await settle(app)
    body = (await client.get("/status")).json()
    assert body["vram"]["free_mb"] == 12288 - 4700
    assert body["resident"] == [
        {
            "model": "coder",
            "vram_mb": 4700,
            "pinned": False,
            "role": "code",
            "idle_s": body["resident"][0]["idle_s"],
            "load_ms": body["resident"][0]["load_ms"],
        }
    ]
    assert body["busy"] is False
    assert body["queue_depth"] == 0


async def test_load_returns_202_with_an_op_id_immediately(client, app, ops):
    response = await client.post("/load", json={"model": "coder"})
    assert response.status_code == 202
    op_id = response.json()["op_id"]
    assert op_id == "op1"

    # The 202 contract is the whole reason a cold load isn't a synchronous
    # HTTP call: the op must still be running at the moment the response
    # comes back, before the background task has been settled. Read the
    # registry directly (no further await, no extra ASGI round trip) so
    # nothing between the POST returning and this assertion gives the
    # spawned task a chance to run ahead of us.
    op = ops.get(op_id)
    assert op is not None
    assert op.status == "running"

    await settle(app)
    assert ops.get(op_id).status == "done"


async def test_op_reports_the_load_result_with_evictions(client, app):
    # 12288 total: coder (4700) leaves 6820 usable, short of painter's 7000,
    # so the second load must evict — and must say so in the op result.
    await client.post("/load", json={"model": "coder"})
    await settle(app)
    await client.post("/load", json={"model": "painter"})
    await settle(app)

    body = (await client.get("/ops/op2")).json()
    assert body["status"] == "done"
    assert body["result"]["model"] == "painter"
    assert body["result"]["vram_mb"] == 7000
    assert body["result"]["evicted"] == [
        {"model": "coder", "freed_mb": 4700, "reason": "evicted"}
    ]


async def test_op_reports_failure_with_code_and_remedy(client, app):
    await client.post("/load", json={"model": "nonesuch"})
    await settle(app)
    body = (await client.get("/ops/op1")).json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "UNKNOWN_MODEL"
    assert "remedy" in body["error"]


async def test_unknown_op_id_is_404(client):
    assert (await client.get("/ops/nope")).status_code == 404


async def test_a_gpu_service_error_from_a_synchronous_route_carries_its_remedy(client):
    # /unload is the one synchronous route that can raise a real GpuServiceError:
    # _unload_locked validates the name against the registry, so an unknown
    # model raises UnknownModel rather than reporting a successful no-op.
    # This drives the app-level GpuServiceError -> 500 handler end to end, with no
    # monkeypatching between the route and the error.
    response = await client.post("/unload", json={"model": "nonesuch"})

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "UNKNOWN_MODEL"
    assert "remedy" in body
    # The remedy must name the models that *do* exist, so the caller's next
    # attempt is a correction rather than another guess.
    assert "coder" in body["remedy"]


async def test_unloading_an_unknown_model_does_not_report_success(client):
    # The failure this replaces: `{"freed_mb": 0}` with a 200, from which a
    # caller that typo'd a name concludes VRAM was freed.
    response = await client.post("/unload", json={"model": "qwen2.5-codr:7b"})
    assert response.status_code == 500
    assert response.json()["code"] == "UNKNOWN_MODEL"


async def test_validation_failures_carry_a_code_and_a_remedy(client):
    # The path an agent hits by guessing a parameter range. FastAPI's raw
    # 422 body has neither a code nor a remedy.
    response = await client.post(
        "/generate/image", json={"prompt": "a lantern", "steps": 500}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "BAD_REQUEST"
    assert "steps" in body["error"]
    assert "steps" in body["remedy"]
    # The raw pydantic errors survive, so the caller can read the bound it broke.
    assert [item["loc"] for item in body["fields"]] == [["body", "steps"]]
    assert body["fields"][0]["type"] == "less_than_equal"


async def test_a_valid_request_body_is_not_treated_as_a_validation_failure(client):
    # Guard the handler against over-reach: the 422 envelope must not appear
    # on a request the model spec accepts.
    response = await client.post("/generate/image", json={"prompt": "a lantern"})
    assert response.status_code == 202


async def test_a_non_dict_detail_keeps_the_shape_http_clients_expect(client):
    # Starlette raises its own HTTPException with a plain-string detail for an
    # unrouted path. The unwrapping handler must leave that alone rather than
    # inventing a code/remedy the GPU service never wrote.
    response = await client.get("/no-such-route")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


async def test_a_malformed_body_still_gets_the_gpu_service_envelope(client):
    # Not a field-level failure: the body is not an object at all.
    response = await client.post("/generate/image", json=[1, 2, 3])
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "BAD_REQUEST"
    assert body["remedy"]
    assert body["fields"]


FAILURE_ROUTES = ("401", "404 unknown op", "404 unknown artifact", "422", "500")


async def test_every_failure_route_speaks_one_error_envelope(app, client):
    """The whole error contract, walked once.

    Governing rule: an error reaching a model must be an actionable
    instruction. That only holds if the client can find `remedy` without
    knowing which failure it hit -- so every failure route must put `code`,
    `error` and `remedy` at the *same depth*. Two envelopes (top-level for
    GpuServiceError, nested under `detail` for HTTPException) and a 422 with
    neither field would each have been invisible to a per-route test.
    """
    import httpx

    responses: dict[str, httpx.Response] = {}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gpu-service",
        headers={"Authorization": "Bearer not-the-token"},
    ) as bare:
        responses["401"] = await bare.get("/health")

    responses["404 unknown op"] = await client.get("/ops/nope")
    responses["404 unknown artifact"] = await client.get(f"/artifacts/{'0' * 32}.png")
    responses["422"] = await client.post(
        "/generate/image", json={"prompt": "x", "steps": 500}
    )
    responses["500"] = await client.post("/unload", json={"model": "nonesuch"})

    assert set(responses) == set(FAILURE_ROUTES)
    assert [r.status_code for r in responses.values()] == [401, 404, 404, 422, 500]

    for label, response in responses.items():
        body = response.json()
        assert isinstance(body, dict), label
        # Same depth: top level, on every one of them.
        for key in ("code", "error", "remedy"):
            assert key in body, f"{label} has no top-level {key!r}"
            assert isinstance(body[key], str), f"{label}'s {key!r} is not a string"
            assert body[key].strip(), f"{label}'s {key!r} is empty"
        # And nothing hides the same fields one level down, which is the
        # shape that forced the client to unwrap two envelopes.
        assert not isinstance(body.get("detail"), dict), label

    codes = {label: response.json()["code"] for label, response in responses.items()}
    assert codes == {
        "401": "UNAUTHORIZED",
        "404 unknown op": "UNKNOWN_OP",
        "404 unknown artifact": "UNKNOWN_ARTIFACT",
        "422": "BAD_REQUEST",
        "500": "UNKNOWN_MODEL",
    }


async def test_unload_is_synchronous_and_reports_freed_mb(client, app):
    await client.post("/load", json={"model": "coder"})
    await settle(app)
    response = await client.post("/unload", json={"model": "coder"})
    assert response.status_code == 200
    assert response.json() == {"freed_mb": 4700}


async def test_events_streams_sse_frames(client, app, events):
    # httpx's ASGITransport runs the app to completion before returning, so the
    # stream must be finite: a background task acts while the generator is
    # subscribed, then closes the bus to end it. Reading the stream concurrently
    # instead would deadlock (verified — it hangs rather than failing).
    async def act_then_close():
        while events.subscriber_count == 0:
            await asyncio.sleep(0)
        await client.post("/load", json={"model": "coder"})
        await settle(app)
        events.close()

    actor = asyncio.create_task(act_then_close())
    response = await asyncio.wait_for(client.get("/events"), timeout=10)
    await actor

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = response.text.splitlines()
    kinds = [ln.removeprefix("event: ") for ln in frames if ln.startswith("event: ")]
    assert "model.loading" in kinds
    assert "model.loaded" in kinds
    assert "op.done" in kinds

    data_lines = [ln for ln in frames if ln.startswith("data: ")]
    assert json.loads(data_lines[0].removeprefix("data: "))["model"] == "coder"


async def test_closing_the_bus_releases_the_subscriber(client, app, events):
    async def close_soon():
        while events.subscriber_count == 0:
            await asyncio.sleep(0)
        events.close()

    closer = asyncio.create_task(close_soon())
    await asyncio.wait_for(client.get("/events"), timeout=10)
    await closer
    assert events.subscriber_count == 0


async def test_events_streams_op_failed_with_the_error_envelope(client, app, events):
    """op.failed is emitted at api.py:165 and :174 and was asserted nowhere.

    Confirmed live against the real GPU service before this test existed -- an
    unknown model produced `op.failed {op_id, code: UNKNOWN_MODEL, error,
    remedy}`. This pins the documented shape, so the one event kind
    a healthy run never exercises stops being unguarded.
    """
    async def act_then_close():
        while events.subscriber_count == 0:
            await asyncio.sleep(0)
        await client.post("/load", json={"model": "no-such-model"})
        await settle(app)
        events.close()

    actor = asyncio.create_task(act_then_close())
    response = await asyncio.wait_for(client.get("/events"), timeout=10)
    await actor

    frames = response.text.splitlines()
    kinds = [ln.removeprefix("event: ") for ln in frames if ln.startswith("event: ")]
    payloads = [
        json.loads(ln.removeprefix("data: "))
        for ln in frames
        if ln.startswith("data: ")
    ]
    assert "op.failed" in kinds, kinds

    failed = next(body for kind, body in zip(kinds, payloads) if kind == "op.failed")
    assert failed["op_id"]
    assert failed["code"] == "UNKNOWN_MODEL"
    assert failed["error"]


async def test_status_json_carries_the_card_truth(client, app):
    await client.post("/load", json={"model": "coder"})
    await settle(app)
    body = (await client.get("/status")).json()
    assert body["busy_model"] is None
    assert body["resident"][0]["idle_s"] >= 0
    assert isinstance(body["resident"][0]["load_ms"], int)
    assert [e["event"] for e in body["recent"]] == ["model.loaded"]


async def test_status_lists_model_fits(registry, gpu, events, ops, artifacts):
    import httpx

    from ohh_gpu.api import create_app
    from ohh_gpu.fit import Fit
    from ohh_gpu.scheduler import VramScheduler
    from tests.conftest import TEST_TOKEN

    fits = {
        "coder": Fit("coder", True, None, None, None),
        "painter": Fit("painter", False, None, None, "needs more memory than this card has free"),
    }
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events,
        fits=fits, kv_cache_type="q8_0",
    )
    app = create_app(
        scheduler=scheduler, ops=ops, events=events, artifacts=artifacts,
        token=TEST_TOKEN, version="0.1.0",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gpu-service",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as c:
        body = (await c.get("/status")).json()
    assert body["kv_cache_type"] == "q8_0"
    by_name = {m["name"]: m for m in body["models"]}
    assert by_name["coder"] == {
        "name": "coder", "role": "code", "available": True,
        "num_ctx": None, "reduced_from": None, "reason": None,
    }
    assert by_name["painter"]["available"] is False
    assert by_name["painter"]["reason"] == "needs more memory than this card has free"


async def test_a_load_op_for_an_unavailable_model_fails_with_the_fit_message(
    registry, gpu, events, ops, artifacts
):
    import httpx

    from ohh_gpu.api import create_app
    from ohh_gpu.fit import Fit
    from ohh_gpu.scheduler import VramScheduler
    from tests.conftest import TEST_TOKEN, settle

    reason = "needs more memory than this card has free"
    scheduler = VramScheduler(
        registry=registry, probe=gpu, backends={"fake": gpu}, events=events,
        fits={"painter": Fit("painter", False, None, None, reason)},
    )
    app = create_app(
        scheduler=scheduler, ops=ops, events=events, artifacts=artifacts,
        token=TEST_TOKEN, version="0.1.0",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gpu-service",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    ) as c:
        op_id = (await c.post("/load", json={"model": "painter"})).json()["op_id"]
        await settle(app)
        op = (await c.get(f"/ops/{op_id}")).json()
    assert op["status"] == "failed"
    assert op["error"]["code"] == "MODEL_UNAVAILABLE"
    assert op["error"]["error"] == f"painter isn't available on this card: {reason}"
