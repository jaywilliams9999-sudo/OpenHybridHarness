"""Backend seams.

Two protocols, deliberately separate:

``VramProbe``   answers "how much VRAM exists and how much is free".
``Backend``     answers "put this model on the device / take it off".

They are split because the VramScheduler must measure real VRAM while executing
through Ollama, and must also run with neither present under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ..registry import ModelSpec

ProgressFn = Callable[[int, int], None]
"""Called as ``progress(step, total)``. Synchronous, so backends running in
worker threads can report without touching the event loop."""


@dataclass(frozen=True)
class ImageRequest:
    """One render request.

    ``steps``, ``width`` and ``height`` are ``None`` when the caller did not
    ask for a particular value. The VramScheduler resolves each of them against the
    model's registry entry and rewrites the request before any backend sees
    it, so generation policy lives in exactly one place.
    """

    prompt: str
    negative: str | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None


TokenFn = Callable[[str], None]
"""Called with each token as it arrives. Advisory only — see ``ChatResult``."""


@dataclass(frozen=True)
class ChatRequest:
    """One text turn.

    ``messages`` is already in the backend's wire shape; the orchestrator's Gateway
    owns the translation from its own transcript, because that mapping differs
    per provider and the GPU service must not grow a second opinion about it.

    ``num_ctx`` is ``None`` when the caller did not ask, and the VramScheduler fills it
    from the model's registry entry — the same request-beats-registry-beats-
    default chain ``ImageRequest`` uses for size, for the same reason: the
    context window is generation policy, and it belongs in one place.
    """

    messages: list[dict[str, object]]
    tools: list[dict[str, object]] | None = None
    num_ctx: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ChatResult:
    """The authoritative outcome of a text turn.

    ``content`` is the **whole** reply, not the tail of a stream. Tokens are
    also emitted on the event bus as they arrive, but that bus drops events
    when a subscriber falls behind (``EventBus._deliver``), so a client that
    assembled the reply from those frames could silently lose text. The stream
    is for watching; this is for reading.
    """

    content: str
    tool_calls: list[dict[str, object]]
    prompt_tokens: int
    completion_tokens: int
    done_reason: str


@runtime_checkable
class VramProbe(Protocol):
    def name(self) -> str: ...
    def total_mb(self) -> int: ...
    def free_mb(self) -> int: ...


@runtime_checkable
class Backend(Protocol):
    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None: ...
    async def unload(self, spec: ModelSpec) -> None: ...


@runtime_checkable
class ImageBackend(Backend, Protocol):
    async def generate_image(
        self, spec: ModelSpec, req: ImageRequest, progress: ProgressFn
    ) -> bytes: ...


@runtime_checkable
class TextBackend(Backend, Protocol):
    async def chat(
        self, spec: ModelSpec, req: ChatRequest, on_token: TokenFn
    ) -> ChatResult: ...


@runtime_checkable
class EnumerableBackend(Backend, Protocol):
    """A backend that can report, and release, what it holds by name.

    Needed because a backend can outlive the GPU service. Ollama is a separate
    long-running service with its own idle timer, so a GPU service restart leaves it
    holding models the new VramScheduler has no record of -- observed 2026-10-15: a
    freshly started GPU service reported `resident: []` against 6.0 GB of VRAM that
    Ollama was holding for another 26 minutes, which left too little for the
    image model and no mechanism to recover but waiting.

    ``resident_names`` is deliberately name-based rather than spec-based: an
    orphan need not be in the registry at all.
    """

    async def resident_names(self) -> list[str]: ...
    async def release_by_name(self, name: str) -> None: ...
