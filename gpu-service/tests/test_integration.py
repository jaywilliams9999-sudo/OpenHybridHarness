"""One manual run against the real GPU.

Excluded from the default suite. Run on the GPU machine with the GPU service already
serving:  python -m pytest tests/test_integration.py -m integration -v
"""

import os

import httpx
import pytest

BASE_URL = os.environ.get("OHH_GPU_URL", "http://127.0.0.1:7311")
TOKEN = os.environ.get("OHH_GPU_TOKEN", "")
CODER = os.environ.get("OHH_INTEGRATION_CODER", "qwen2.5-coder:7b")

# Without the skip, an unset token sends `Authorization: Bearer ` to a live
# GPU service, which answers 401 -- and the first thing every test here does with
# that body is read ["op_id"], so the operator's diagnostic is
# `KeyError: 'op_id'` rather than "you did not export the token". Expressed
# as skipif rather than a module-level pytest.skip so the two tests still
# collect (and still show as deselected) in the default run.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TOKEN,
        reason="OHH_GPU_TOKEN is not set; export the GPU service's bearer token "
        "to run the integration suite against the real GPU.",
    ),
]


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=300.0,
    ) as client:
        yield client


async def poll(client: httpx.AsyncClient, op_id: str) -> dict:
    import asyncio

    for _ in range(600):
        body = (await client.get(f"/ops/{op_id}")).json()
        if body["status"] != "running":
            return body
        await asyncio.sleep(1)
    raise AssertionError(f"op {op_id} never finished")


async def test_health_reports_the_real_card(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert "NVIDIA" in body["gpu"]
    assert body["vram_total_mb"] > 0


async def test_load_the_coder_then_evict_it_for_the_painter(client):
    """The eviction path on real hardware.

    This is the only place the VramScheduler's eviction decision can ever be observed
    end to end against a real card, so the eviction itself is the assertion --
    not a side note. Without it the test passed identically when no eviction
    happened at all, which is precisely the outcome that would mean the
    scheduler is broken.
    """
    loaded = await poll(
        client, (await client.post("/load", json={"model": CODER})).json()["op_id"]
    )
    assert loaded["status"] == "done"
    assert loaded["result"]["model"] == CODER

    rendered = await poll(
        client,
        (
            await client.post(
                "/generate/image",
                json={"prompt": "a brass lantern on black", "steps": 4},
            )
        ).json()["op_id"],
    )
    assert rendered["status"] == "done"
    painter = rendered["result"]["model"]

    evicted = rendered["result"]["evicted"]
    assert evicted, (
        f"the painter loaded without evicting anything; {CODER} was loaded "
        f"first and both cannot fit, so either the card grew or the VramScheduler "
        f"stopped evicting"
    )
    assert [record["model"] for record in evicted] == [CODER]
    assert evicted[0]["reason"] == "evicted"
    assert evicted[0]["freed_mb"] > 0

    image = await client.get(rendered["result"]["artifact_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    # A plausible PNG: the signature, the IHDR chunk that must follow it, the
    # terminating IEND, and a size no real render could fall under.
    assert image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.content[12:16] == b"IHDR"
    assert image.content.rstrip().endswith(b"IEND\xaeB`\x82")
    assert len(image.content) > 1024

    # And the other side of the eviction: the painter holds the card now.
    status = (await client.get("/status")).json()
    resident = {row["model"]: row for row in status["resident"]}
    assert painter in resident
    assert CODER not in resident
    assert resident[painter]["vram_mb"] > 0

    # Record what the run actually cost, so models.toml hints can be corrected.
    print("observed:", rendered["result"], status)
