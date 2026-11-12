"""SSE fan-out for load progress, diffusion steps, and evictions.

``emit`` is synchronous so the VramScheduler — and backends running in worker
threads — can report without awaiting. Subscribers get bounded queues; a
stalled HTTP client drops events rather than stalling the GPU.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    kind: str
    body: dict[str, Any]


_CLOSED = object()
"""Sentinel pushed by EventBus.close() to end every subscriber's iteration."""


class Subscription:
    def __init__(self, bus: "EventBus", queue: asyncio.Queue[Event]) -> None:
        self._bus = bus
        self._queue = queue

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        # Wake anyone parked in `await self._queue.get()` on this stream.
        # EventBus.close() only reaches queues still registered with the
        # bus, so closing (or exiting) a single Subscription must push its
        # own sentinel too -- otherwise a concurrent __anext__() on this
        # queue hangs forever once _unsubscribe() removes it from the bus's
        # fan-out list and nothing else will ever wake it.
        try:
            self._queue.put_nowait(_CLOSED)
        except asyncio.QueueFull:
            self._queue.get_nowait()
            self._queue.put_nowait(_CLOSED)
        self._bus._unsubscribe(self._queue)

    def drain(self) -> list[Event]:
        """Everything queued right now, without awaiting. For tests."""
        events: list[Event] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return events
            if item is not _CLOSED:
                events.append(item)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Event:
        item = await self._queue.get()
        if item is _CLOSED:
            raise StopAsyncIteration
        return item


class EventBus:
    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self.dropped = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> Subscription:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(queue)
        if self._loop is None:
            # Captured once, from whichever thread first subscribes: that is
            # the loop emit() must marshal onto when it is later called from
            # a worker thread. Subscribing outside a running loop (plain
            # sync tests) leaves this None, and emit() then just delivers
            # directly -- there is no loop to wake.
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        return Subscription(self, queue)

    def _unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def emit(self, kind: str, /, **body: Any) -> None:
        # `kind` is positional-only on purpose. The op.progress event
        # is {op_id, kind, step, total} -- a body field literally named `kind` --
        # so `emit("op.progress", kind="image", ...)` would otherwise bind `kind`
        # twice and raise TypeError on every diffusion step.
        event = Event(kind=kind, body=body)
        loop = self._loop
        if loop is not None:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not loop:
                # Called from a worker thread (e.g. a diffusion step
                # callback under asyncio.to_thread): Queue.put_nowait()'s
                # wakeup goes through call_soon(), which is not
                # thread-safe and never touches the loop's self-pipe, so a
                # subscriber blocked in `await queue.get()` would sit until
                # the loop happened to wake for some unrelated reason.
                # call_soon_threadsafe() wakes the selector immediately and
                # is itself non-blocking, so emit() still never blocks the
                # calling thread.
                try:
                    loop.call_soon_threadsafe(self._deliver, event)
                except RuntimeError:
                    # Loop already closed -- nothing to notify. Every other
                    # drop path on this bus increments `dropped`; do the
                    # same here so a closed-loop drop is observable rather
                    # than silently swallowed.
                    self.dropped += 1
                return
        self._deliver(event)

    def _deliver(self, event: Event) -> None:
        for queue in self._subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Two emit() calls can race between the full-check above and
                # this put -- check-then-evict-then-put is not atomic -- so
                # this is reachable whenever callers emit concurrently on a
                # full queue, not only while shutting down.
                self.dropped += 1

    def close(self) -> None:
        """End every open subscription.

        Called on GPU service shutdown so uvicorn stops cleanly instead of waiting
        on SSE streams that would otherwise never finish.
        """
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(_CLOSED)
            except asyncio.QueueFull:
                queue.get_nowait()
                queue.put_nowait(_CLOSED)


def sse_format(event: Event) -> str:
    return f"event: {event.kind}\ndata: {json.dumps(event.body)}\n\n"
