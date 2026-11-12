import asyncio
import json
import threading

import pytest

from ohh_gpu.events import Event, EventBus, sse_format


def test_subscriber_receives_emitted_events():
    bus = EventBus()
    with bus.subscribe() as stream:
        bus.emit("model.loaded", model="coder", vram_mb=4700, load_ms=120)
        events = stream.drain()
    assert len(events) == 1
    assert events[0].kind == "model.loaded"
    assert events[0].body == {"model": "coder", "vram_mb": 4700, "load_ms": 120}


def test_events_emitted_before_subscribing_are_not_replayed():
    bus = EventBus()
    bus.emit("vram.changed", used_mb=0, free_mb=12288)
    with bus.subscribe() as stream:
        assert stream.drain() == []


def test_every_subscriber_receives_a_fan_out():
    bus = EventBus()
    with bus.subscribe() as a, bus.subscribe() as b:
        bus.emit("op.progress", op_id="x", kind="image", step=1, total=4)
        assert len(a.drain()) == 1
        assert len(b.drain()) == 1


def test_unsubscribed_stream_stops_receiving():
    bus = EventBus()
    with bus.subscribe() as stream:
        pass
    bus.emit("vram.changed", used_mb=1, free_mb=2)
    assert bus.subscriber_count == 0


def test_full_queue_drops_the_oldest_event():
    bus = EventBus(maxsize=2)
    with bus.subscribe() as stream:
        bus.emit("op.progress", step=1)
        bus.emit("op.progress", step=2)
        bus.emit("op.progress", step=3)
        events = stream.drain()
    assert [e.body["step"] for e in events] == [2, 3]
    assert bus.dropped == 1


async def test_async_iteration_yields_events_as_they_arrive():
    bus = EventBus()
    received: list[Event] = []

    async def consume(stream):
        async for event in stream:
            received.append(event)
            if len(received) == 2:
                return

    with bus.subscribe() as stream:
        task = asyncio.create_task(consume(stream))
        await asyncio.sleep(0)
        bus.emit("model.loading", model="coder", pct=50)
        bus.emit("model.loaded", model="coder", vram_mb=4700, load_ms=9)
        await asyncio.wait_for(task, timeout=1)

    assert [e.kind for e in received] == ["model.loading", "model.loaded"]


async def test_close_ends_open_iterations():
    bus = EventBus()
    with bus.subscribe() as stream:
        collected: list[Event] = []

        async def consume():
            async for event in stream:
                collected.append(event)
            return "iteration ended"

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        bus.emit("op.progress", step=1)
        bus.close()
        assert await asyncio.wait_for(task, timeout=1) == "iteration ended"

    assert [e.body["step"] for e in collected] == [1]


async def test_emit_from_a_worker_thread_reaches_a_blocked_subscriber():
    """Regression guard for the thread-safe emit() fix -- platform-dependent.

    On a SelectorEventLoop (the Linux CI default), a naive put_nowait()-only
    emit() fails this test for real: nothing else is scheduled while
    `asyncio.to_thread(worker)` is awaited, so the loop never wakes to
    deliver the event, `delivered.wait(timeout=5)` returns False, and
    `assert woke is True` below fails outright -- not a hang.

    On Windows ProactorEventLoop (this dev box) the loop polls on a bounded
    timeout regardless, so the naive path also wakes and this test passes
    even without the fix -- measured ~16.9ms to deliver via the naive path
    versus ~0.1ms via call_soon_threadsafe. Here this test is a smoke test,
    not a guard; a Linux CI run is the real signal for this behavior.
    """
    bus = EventBus()
    delivered = threading.Event()
    with bus.subscribe() as stream:

        async def consume():
            async for event in stream:
                delivered.set()
                return event

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let the consumer block on queue.get()

        def worker() -> bool:
            bus.emit("model.loading", model="coder", pct=50)
            # Returns False if the loop never woke to deliver it.
            return delivered.wait(timeout=5)

        woke = await asyncio.to_thread(worker)
        event = await asyncio.wait_for(task, timeout=5)

    assert woke is True
    assert event.kind == "model.loading"
    assert event.body == {"model": "coder", "pct": 50}


async def test_subscription_close_unblocks_a_pending_anext():
    bus = EventBus()
    stream = bus.subscribe()
    task = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)  # let the task block on queue.get()
    stream.close()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(task, timeout=1)


def test_close_sentinel_is_not_drained_as_an_event():
    bus = EventBus()
    with bus.subscribe() as stream:
        bus.emit("op.progress", step=1)
        bus.close()
        events = stream.drain()
    assert [e.kind for e in events] == ["op.progress"]


def test_sse_format_renders_named_events_with_json_data():
    frame = sse_format(Event(kind="op.done", body={"op_id": "abc", "artifact_url": "/a/1.png"}))
    assert frame.startswith("event: op.done\n")
    assert frame.endswith("\n\n")
    data_line = [ln for ln in frame.splitlines() if ln.startswith("data: ")][0]
    assert json.loads(data_line.removeprefix("data: ")) == {
        "op_id": "abc",
        "artifact_url": "/a/1.png",
    }
