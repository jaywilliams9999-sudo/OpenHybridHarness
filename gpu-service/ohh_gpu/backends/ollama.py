"""Ollama backend — HTTP to the Ollama instance on the same host as the GPU service.

Ollama has no explicit load call: a generate request with an empty prompt
loads the model, and ``keep_alive`` controls residency. The GPU service sets
``keep_alive=-1`` -- never expire -- and manages residency itself, so the
VramScheduler, not Ollama's idle timer, remains the source of truth. A finite value
(the former "30m") lets Ollama unload a model the VramScheduler still books, and the
VramScheduler's own idle sweep (`VramScheduler.sweep_idle`) is what returns the card.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from ..errors import BackendFailure, BackendOom
from ..registry import ModelSpec
from .base import ChatRequest, ChatResult, ProgressFn, TokenFn

OOM_MARKERS = ("out of memory", "more system memory", "insufficient memory", "cuda oom")


class OllamaBackend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str | int = -1,
        client: httpx.AsyncClient | None = None,
        unload_poll_interval_s: float = 0.2,
        unload_max_polls: int = 50,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._keep_alive = keep_alive
        self._client = client or httpx.AsyncClient(timeout=300.0)
        self._unload_poll_interval_s = unload_poll_interval_s
        self._unload_max_polls = unload_max_polls

    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        """Load at the model's own context window, not Ollama's default.

        This matters for accounting, not convenience. The VramScheduler books a model
        at the VRAM delta it measures across this call, and the KV cache -- a
        measured 34 MB per 1k tokens on q4_0 -- is most of that
        delta at a large window. Loading without `num_ctx` makes Ollama pick
        its own default (0.40.4: "4k/32k/256k based on VRAM"), the VramScheduler books
        that smaller figure, and then the first chat request supplies the real
        window and Ollama silently reloads the runner twice the size. The
        VramScheduler would be under-booking by up to 2x against a card it is
        otherwise measuring exactly -- the same overcommit NVML was adopted to
        prevent, reached from a third direction.
        """
        progress(0, 100)
        options = {"num_ctx": spec.num_ctx} if spec.num_ctx is not None else None
        await self._generate(spec, keep_alive=self._keep_alive, options=options)
        progress(100, 100)

    async def unload(self, spec: ModelSpec) -> None:
        await self._generate(spec, keep_alive=0)
        await self._await_release(spec)

    async def chat(
        self, spec: ModelSpec, req: ChatRequest, on_token: TokenFn
    ) -> ChatResult:
        """One text turn against ``/api/chat``, streamed.

        ``stream=True`` even though the result is returned whole: a 103k-token
        prefill can take two and a half minutes on this card, and a
        non-streaming request would hold the socket silent for all of it with no
        way to show progress. The tokens go to ``on_token`` as they arrive and
        the assembled text comes back in the result, so the caller never has to
        reassemble a stream it may have dropped frames from.

        ``keep_alive`` is the backend's own long value, not 0: residency is the
        VramScheduler's decision, and a chat request must not quietly evict the model
        it just used.
        """
        options: dict[str, object] = {}
        if req.num_ctx is not None:
            options["num_ctx"] = req.num_ctx
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.max_tokens is not None:
            options["num_predict"] = req.max_tokens

        payload: dict[str, object] = {
            "model": spec.name,
            "messages": req.messages,
            "stream": True,
            "keep_alive": self._keep_alive,
        }
        if req.tools:
            payload["tools"] = req.tools
        if options:
            payload["options"] = options

        content: list[str] = []
        tool_calls: list[dict[str, object]] = []
        prompt_tokens = 0
        completion_tokens = 0
        done_reason = "stop"

        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._raise_chat_failure(spec, response)

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except ValueError:
                        # Ollama emits newline-delimited JSON; a partial line is
                        # not worth failing a turn that is otherwise succeeding.
                        continue

                    message = frame.get("message") or {}
                    chunk = message.get("content") or ""
                    if chunk:
                        content.append(chunk)
                        on_token(chunk)

                    # Tool calls arrive whole rather than as deltas, and may
                    # appear on any frame, including the final one.
                    for call in message.get("tool_calls") or []:
                        tool_calls.append(call)

                    if frame.get("done"):
                        prompt_tokens = int(frame.get("prompt_eval_count") or 0)
                        completion_tokens = int(frame.get("eval_count") or 0)
                        done_reason = str(frame.get("done_reason") or "stop")
        except httpx.HTTPError as error:
            raise BackendFailure(
                f"Ollama at {self._base_url} failed mid-turn.",
                remedy=(
                    "The local model host dropped the connection. Retry, or "
                    "proceed without local inference and record the gap."
                ),
                model=spec.name,
                base_url=self._base_url,
            ) from error

        return ChatResult(
            content="".join(content),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            done_reason=done_reason,
        )

    def _raise_chat_failure(self, spec: ModelSpec, response: httpx.Response) -> None:
        detail = self._error_text(response)
        if any(marker in detail.lower() for marker in OOM_MARKERS):
            raise BackendOom(
                f"Ollama ran out of memory answering with {spec.name}: {detail}",
                remedy="Free VRAM by unloading a resident model, then retry.",
                model=spec.name,
                requested_mb=spec.vram_mb,
            )
        raise BackendFailure(
            f"Ollama refused the turn for {spec.name}: {detail}",
            remedy=(
                "Check the model name and that it supports the requested "
                "features; a model without tool support rejects a tools array."
            ),
            model=spec.name,
            status=response.status_code,
        )

    async def resident_names(self) -> list[str]:
        """What Ollama is holding right now, by name.

        Non-raising: this feeds startup reconciliation, and a GPU service that
        refused to boot because Ollama was briefly unreachable would be worse
        than one that boots believing the card is empty — which is what it
        already believes without this call.
        """
        try:
            response = await self._client.get(f"{self._base_url}/api/ps")
            models = response.json().get("models") or []
        except Exception:  # noqa: BLE001 - see docstring
            return []
        names = []
        for entry in models:
            name = entry.get("model") or entry.get("name")
            if name:
                names.append(str(name))
        return names

    async def release_by_name(self, name: str) -> None:
        """Evict by name, for a model that may not be in the registry.

        Waits for the release the way `unload` does: startup fits models to
        the card straight after this, and an orphan still on the device then
        reads as another program's memory.
        """
        spec = ModelSpec(name=name, backend="ollama", vram_mb=0, role="unknown")
        await self._generate(spec, keep_alive=0)
        await self._await_release(spec)

    async def _await_release(self, spec: ModelSpec) -> None:
        """Block until Ollama stops reporting the model resident.

        ``keep_alive=0`` is acknowledged before the device is freed, so
        returning the moment the request completes leaves the VramScheduler probing
        VRAM that has not come back yet -- measured on the real 3060, a
        unload emitted `vram.changed` with figures byte-identical to the
        reading before it.

        Bounded and non-raising. The VramScheduler evicts in order to make room and
        has already dropped the residency record by the time this runs, so
        raising would fail the very load the eviction served. A stuck Ollama
        therefore degrades to the previous behaviour instead.
        """
        for _ in range(self._unload_max_polls):
            if not await self._is_resident(spec):
                return
            await asyncio.sleep(self._unload_poll_interval_s)

    async def _is_resident(self, spec: ModelSpec) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/ps")
            models = response.json().get("models") or []
        except Exception:  # noqa: BLE001 - a failed check must not fail the unload
            return False
        return any(
            entry.get("model") == spec.name or entry.get("name") == spec.name
            for entry in models
        )

    async def _generate(
        self,
        spec: ModelSpec,
        keep_alive: str | int,
        options: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "model": spec.name,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options
        try:
            response = await self._client.post(
                f"{self._base_url}/api/generate", json=payload
            )
        except httpx.HTTPError as error:
            raise BackendFailure(
                f"Ollama at {self._base_url} is unreachable.",
                remedy=(
                    "The local model host is down. Proceed without local inference "
                    "and record the gap in your report."
                ),
                model=spec.name,
                base_url=self._base_url,
            ) from error

        if response.status_code == 200:
            return

        detail = self._error_text(response)
        if any(marker in detail.lower() for marker in OOM_MARKERS):
            raise BackendOom(
                f"Ollama refused to load {spec.name}: {detail}",
                remedy="Free VRAM by unloading a resident model, then retry.",
                model=spec.name,
                requested_mb=spec.vram_mb,
            )
        raise BackendFailure(
            f"Ollama refused to load {spec.name}: {detail}",
            remedy=(
                f"Confirm the model exists on the GPU service host with "
                f"`ollama pull {spec.name}`, then retry."
            ),
            model=spec.name,
            status=response.status_code,
        )

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            return str(response.json().get("error", response.text))
        except ValueError:
            return response.text
