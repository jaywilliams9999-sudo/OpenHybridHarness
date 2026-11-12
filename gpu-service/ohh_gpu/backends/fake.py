"""FakeGpuBackend — simulated VRAM, load latency, and OOM.

Carries half the testing strategy. Every eviction rule, headroom calculation,
and scheduling decision in the VramScheduler becomes a fast deterministic unit
test that runs without a GPU.
"""

from __future__ import annotations

import asyncio

from ..errors import BackendFailure, BackendOom
from ..registry import ModelSpec
from .base import ChatRequest, ChatResult, ImageRequest, ProgressFn, TokenFn

# A PNG signature plus a stub body. Tests assert on the magic bytes; nothing
# in the GPU service decodes artifact contents.
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-gpu-artifact"


class FakeGpuBackend:
    """Implements VramProbe, ImageBackend and TextBackend over one fake card."""

    def __init__(
        self,
        total_mb: int = 12288,
        load_ms: int = 0,
        observed_cost: dict[str, int] | None = None,
        png: bytes = FAKE_PNG,
        reply: str = "a fake reply",
        tool_calls: list[dict] | None = None,
    ) -> None:
        self._total_mb = total_mb
        self._load_ms = load_ms
        self._observed_cost = observed_cost or {}
        self._png = png
        self._reply = reply
        self._tool_calls = tool_calls or []
        self._allocated: dict[str, int] = {}
        self.load_calls: list[str] = []
        # The full model spec each load saw, not just the name: the context window
        # rides on the spec and is what the VRAM measurement describes.
        self.loaded_specs: list[ModelSpec] = []
        self.unload_calls: list[str] = []
        self.chat_calls: list[ChatRequest] = []

    # -- VramProbe ----------------------------------------------------
    def name(self) -> str:
        return "FakeGpu"

    def total_mb(self) -> int:
        return self._total_mb

    def free_mb(self) -> int:
        return self._total_mb - sum(self._allocated.values())

    # -- Backend ------------------------------------------------------
    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        self.load_calls.append(spec.name)
        self.loaded_specs.append(spec)
        cost = self._observed_cost.get(spec.name, spec.vram_mb)
        if cost > self.free_mb():
            raise BackendOom(
                f"Simulated CUDA OOM loading {spec.name}.",
                remedy="Free VRAM by unloading a resident model, then retry.",
                model=spec.name,
                requested_mb=cost,
                free_mb=self.free_mb(),
            )
        for pct in (25, 50, 75, 100):
            if self._load_ms:
                await asyncio.sleep(self._load_ms / 4000)
            progress(pct, 100)
        self._allocated[spec.name] = cost

    async def unload(self, spec: ModelSpec) -> None:
        self.unload_calls.append(spec.name)
        self._allocated.pop(spec.name, None)

    # -- ImageBackend -------------------------------------------------
    async def generate_image(
        self, spec: ModelSpec, req: ImageRequest, progress: ProgressFn
    ) -> bytes:
        if spec.name not in self._allocated:
            # BackendFailure, not BackendOom, and deliberately the same code
            # DiffusersBackend raises for this exact condition. A fake that
            # disagrees with the real backend on error taxonomy trains the
            # VramScheduler's recovery paths against a taxonomy that does not exist:
            # BackendOom triggers panic-evict-and-retry, which cannot help
            # when the model simply was not loaded.
            raise BackendFailure(
                f"{spec.name} is not resident.",
                remedy="Load the model before generating.",
                model=spec.name,
            )
        total = req.steps or spec.steps or 4
        for step in range(1, total + 1):
            progress(step, total)
        return self._png

    # -- TextBackend --------------------------------------------------
    async def chat(
        self, spec: ModelSpec, req: ChatRequest, on_token: TokenFn
    ) -> ChatResult:
        if spec.name not in self._allocated:
            # Same taxonomy as generate_image, for the same reason: a fake that
            # raises BackendOom here would train the VramScheduler's panic-evict path
            # against a condition eviction cannot fix.
            raise BackendFailure(
                f"{spec.name} is not resident.",
                remedy="Load the model before chatting.",
                model=spec.name,
            )
        self.chat_calls.append(req)
        for chunk in self._reply.split(" "):
            on_token(chunk + " ")
        return ChatResult(
            content=self._reply,
            tool_calls=list(self._tool_calls),
            prompt_tokens=len(req.messages) * 10,
            completion_tokens=len(self._reply.split()),
            done_reason="stop",
        )
