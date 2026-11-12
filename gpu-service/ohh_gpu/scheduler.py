"""The VramScheduler — single source of truth for VRAM.

Holds residency, decides what fits, chooses what to evict, and serializes
every GPU operation behind one lock. Contains no I/O: measurement and
execution are injected, which is what makes the whole scheduler unit-testable
without a GPU.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable

from .backends.base import (
    Backend,
    ChatRequest,
    ChatResult,
    EnumerableBackend,
    ImageBackend,
    ImageRequest,
    ProgressFn,
    TextBackend,
    TokenFn,
    VramProbe,
)
from .errors import (
    BackendFailure,
    BackendOom,
    ModelUnavailable,
    UnknownModel,
    VramExhausted,
)
from .fit import Fit
from .registry import ModelRegistry, ModelSpec


@dataclass
class Residency:
    spec: ModelSpec
    vram_mb: int
    pinned: bool
    last_used: float
    load_ms: int | None = None


@dataclass(frozen=True)
class EvictionRecord:
    model: str
    freed_mb: int
    reason: str  # "explicit" | "evicted" | "idle"


@dataclass(frozen=True)
class LoadResult:
    model: str
    vram_mb: int
    load_ms: int
    evicted: list[EvictionRecord] = field(default_factory=list)
    already_resident: bool = False


@dataclass(frozen=True)
class ImageResult:
    model: str
    png: bytes
    steps: int
    width: int
    height: int
    evicted: list[EvictionRecord] = field(default_factory=list)


@dataclass(frozen=True)
class TextResult:
    model: str
    content: str
    tool_calls: list[dict]
    prompt_tokens: int
    completion_tokens: int
    done_reason: str
    num_ctx: int | None
    evicted: list[EvictionRecord] = field(default_factory=list)


@dataclass(frozen=True)
class StatusSnapshot:
    vram: dict[str, int]
    resident: list[dict]
    busy: bool
    queue_depth: int
    busy_model: str | None = None
    recent: list[dict] = field(default_factory=list)


DEFAULT_IMAGE_SIZE = 1024
"""Last-resort render size, used only when neither the request nor the model's
registry entry says otherwise."""


def _noop_progress(step: int, total: int) -> None:
    return None


def _noop_token(text: str) -> None:
    return None


class VramScheduler:
    def __init__(
        self,
        registry: ModelRegistry,
        probe: VramProbe,
        backends: dict[str, Backend],
        events=None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        fits: dict[str, Fit] | None = None,
        kv_cache_type: str = "f16",
    ) -> None:
        # Per-model verdicts for this card (see fit.py). None means "not
        # computed": every model is treated as available at its registry window.
        self.fits = fits
        self.kv_cache_type = kv_cache_type
        self._registry = registry
        self._probe = probe
        self._backends = backends
        self._events = events
        self._clock = clock
        self._wall_clock = wall_clock
        self._resident: dict[str, Residency] = {}
        # Keyed by (model name, context window): see `observed_mb`.
        self._observed: dict[tuple[str, int | None], int] = {}
        self._busy = False
        self._busy_model: str | None = None
        self._queued = 0
        # The last swaps, for a page that opens after they happened. Kept
        # here, not in the event bus, so it exists when `events` is None.
        self._recent: deque[dict] = deque(maxlen=20)
        self._lock = asyncio.Lock()

    # -- accounting ---------------------------------------------------
    @property
    def gpu_name(self) -> str:
        return self._probe.name()

    @property
    def headroom_mb(self) -> int:
        return self._registry.gpu_service.vram_headroom_mb

    def usable_free_mb(self) -> int:
        return max(0, self._probe.free_mb() - self.headroom_mb)

    def observed_mb(self, name: str, num_ctx: int | None = None) -> int:
        """What loading ``name`` at ``num_ctx`` is expected to cost.

        Keyed by the **pair**, not the name. A model's footprint is a function
        of its weights and its context window -- the KV cache is a measured
        34 MB per 1000 tokens on q4_0, which is nearly half the
        footprint at 131072 -- so a measurement taken at one window describes a
        different model than the one about to be loaded at another.

        Observed 2026-10-15: a turn requested at num_ctx=16384 was planned
        against 9809 MB, the figure measured at 131072 in a previous run, and
        the VramScheduler refused a load that needed roughly 5800. Keying by name
        alone made a correct measurement into a wrong prediction.
        """
        spec = self._registry.require(name)
        window = num_ctx if num_ctx is not None else spec.num_ctx
        key = (name, window)
        if key in self._observed:
            return self._observed[key]
        return self._hint_mb(spec, window)

    @staticmethod
    def _hint_mb(spec: ModelSpec, window: int | None) -> int:
        """Re-base the registry's measurement onto the requested window.

        Falls back to the flat figure when the entry carries no KV rate, which
        keeps every model that does not declare one behaving exactly as before.
        """
        if (
            spec.kv_rates is None
            or spec.measured_kv_cache not in spec.kv_rates
            or spec.num_ctx is None
            or window is None
            or window == spec.num_ctx
        ):
            return spec.vram_mb
        delta = spec.kv_rates[spec.measured_kv_cache] * (window - spec.num_ctx) / 1000
        return max(1, int(round(spec.vram_mb + delta)))

    def booked_resident_mb(self) -> int:
        return sum(r.vram_mb for r in self._resident.values())

    def model_fits(self) -> list[dict]:
        """One row per registered model, for /status."""
        rows = []
        for name in sorted(self._registry.models):
            spec = self._registry.models[name]
            fit = self.fits.get(name) if self.fits else None
            rows.append(
                {
                    "name": name,
                    "role": spec.role,
                    "available": fit.available if fit else True,
                    "num_ctx": fit.num_ctx if fit else spec.num_ctx,
                    "reduced_from": fit.reduced_from if fit else None,
                    "reason": fit.reason if fit else None,
                }
            )
        return rows

    def _require_available(self, name: str) -> None:
        fit = self.fits.get(name) if self.fits else None
        if fit is not None and not fit.available:
            raise ModelUnavailable(
                f"{name} isn't available on this card: {fit.reason}",
                remedy="Use a smaller model, or free memory on the GPU and "
                "restart the GPU service.",
                model=name,
            )

    def is_resident(self, name: str) -> bool:
        return name in self._resident

    def status(self) -> StatusSnapshot:
        return StatusSnapshot(
            vram={
                "total_mb": self._probe.total_mb(),
                "free_mb": self._probe.free_mb(),
                "headroom_mb": self.headroom_mb,
                "usable_free_mb": self.usable_free_mb(),
            },
            resident=[
                {
                    "model": r.spec.name,
                    "vram_mb": r.vram_mb,
                    "pinned": r.pinned,
                    "role": r.spec.role,
                    "idle_s": int(self._clock() - r.last_used),
                    "load_ms": r.load_ms,
                }
                for r in sorted(self._resident.values(), key=lambda r: r.spec.name)
            ],
            busy=self._busy,
            queue_depth=self._queued,
            busy_model=self._busy_model,
            recent=list(self._recent),
        )

    def _emit(self, kind: str, /, **body) -> None:
        if kind in ("model.loaded", "model.unloaded"):
            self._recent.append(
                {"event": kind, "at": self._wall_clock().isoformat(), **body}
            )
        if self._events is not None:
            self._events.emit(kind, **body)

    def _used_mb(self) -> int:
        return self._probe.total_mb() - self._probe.free_mb()

    def _backend_for(self, spec: ModelSpec) -> Backend:
        backend = self._backends.get(spec.backend)
        if backend is None:
            raise UnknownModel(
                f"No backend registered for {spec.backend!r}.",
                remedy=f"Register a backend named {spec.backend!r} on the GPU service.",
                model=spec.name,
                backend=spec.backend,
            )
        return backend

    # -- locking --------------------------------------------------------
    @asynccontextmanager
    async def _gpu(self, model: str | None = None):
        """The single GPU lock. One card, one queue, one operation at a time.

        ``model`` names who holds it, for `/status`; reconcile and the idle
        sweep act on no particular model and pass nothing.
        """
        self._queued += 1
        try:
            await self._lock.acquire()
        finally:
            self._queued -= 1
        self._busy = True
        self._busy_model = model
        self._emit("gpu.busy", model=model)
        try:
            yield
        finally:
            self._busy = False
            self._busy_model = None
            self._lock.release()
            self._emit("gpu.idle")

    # -- reconciliation -----------------------------------------------
    async def reconcile(self) -> list[str]:
        """Return the card to a state the VramScheduler can describe.

        A backend can outlive the GPU service. Ollama is a separate service with its
        own idle timer, so a restart leaves it holding models this VramScheduler has
        no record of. Measured 2026-10-15: a freshly started GPU service reported
        `resident: []` while Ollama held 6.0 GB for another 26 minutes, leaving
        4836 MB usable against the 7087 MB image model -- the harness was
        unusable and nothing but the clock could fix it.

        **Orphans are released, not adopted.** Adopting looks tempting -- the
        VRAM is real and NVML can see it -- but the VramScheduler would have to book a
        figure for it, and it does not know the context window the orphan was
        loaded at. Booking the wrong number is the bug this is adjacent to, not
        a fix for it. Releasing gives up a warm model and buys an invariant: a
        cold GPU service means a card holding only what the GPU service put there.

        Returns the names released, for the operator log.
        """
        released: list[str] = []
        async with self._gpu():
            for backend in dict.fromkeys(self._backends.values()):
                if not isinstance(backend, EnumerableBackend):
                    continue
                for name in await backend.resident_names():
                    if name in self._resident:
                        continue
                    await backend.release_by_name(name)
                    released.append(name)
        if released:
            self._emit(
                "vram.changed",
                used_mb=self._used_mb(),
                free_mb=self._probe.free_mb(),
            )
        return released

    # -- operations ---------------------------------------------------
    async def load(
        self, name: str, pin: bool = False, progress: ProgressFn = _noop_progress
    ) -> LoadResult:
        async with self._gpu(name):
            return await self._load_locked(name, pin=pin, progress=progress)

    async def _load_locked(
        self,
        name: str,
        pin: bool = False,
        progress: ProgressFn = _noop_progress,
        spec: ModelSpec | None = None,
    ) -> LoadResult:
        """``spec`` overrides the registry entry for this load only.

        Used by ``generate_text`` to load at the window the turn will actually
        use. The registry's ``num_ctx`` is a default, and a request may override
        it -- but the VRAM a model costs is a function of that window, so the
        load and the measurement have to agree with the request or the VramScheduler
        books a figure describing a model that was never loaded.
        """
        spec = spec if spec is not None else self._registry.require(name)
        self._require_available(name)
        fit = self.fits.get(name) if self.fits else None
        if (
            fit is not None
            and fit.num_ctx is not None
            and spec.num_ctx is not None
            and fit.num_ctx < spec.num_ctx
        ):
            # An explicit load agrees with generate_text: the fitted window,
            # not the registry's, is what this card can hold.
            spec = replace(spec, num_ctx=fit.num_ctx)
        if name in self._resident:
            residency = self._resident[name]
            residency.last_used = self._clock()
            residency.pinned = residency.pinned or pin
            return LoadResult(
                model=name,
                vram_mb=residency.vram_mb,
                load_ms=0,
                already_resident=True,
            )

        need_mb = self.observed_mb(name, spec.num_ctx)
        capacity = self._probe.total_mb() - self.headroom_mb
        if need_mb > capacity:
            raise VramExhausted(
                f"{name} needs {need_mb}MB but the card's usable capacity is {capacity}MB.",
                remedy=(
                    f"This model cannot fit on this GPU at all. Choose a smaller or "
                    f"quantized model; do not retry {name}."
                ),
                model=name,
                requested_mb=need_mb,
                capacity_mb=capacity,
                freed=[],
            )

        evicted = await self._make_room(need_mb, protect=name)
        return await self._load(spec, pin=pin, evicted=evicted, progress=progress)

    async def generate_image(
        self,
        req: ImageRequest,
        model: str | None = None,
        progress: ProgressFn = _noop_progress,
    ) -> ImageResult:
        spec = (
            self._registry.require(model)
            if model
            else self._registry.first_with_role("image")
        )
        if spec is None:
            raise UnknownModel(
                "No model with role 'image' is registered.",
                remedy="Add an image model to models.toml, or proceed without "
                "generated assets and note the gap in your report.",
                role="image",
            )

        # Validate the backend can even generate images before touching
        # residency: a misconfigured registry entry should fail cleanly, not
        # evict a resident model and pull a cold load only to raise anyway.
        backend = self._backend_for(spec)
        if not isinstance(backend, ImageBackend):
            raise BackendFailure(
                f"Backend {spec.backend!r} cannot generate images.",
                remedy="Register an image-capable backend for this model.",
                model=spec.name,
                backend=spec.backend,
            )

        # Resolve generation policy once, here, and thread it through `req` so
        # no backend re-derives it independently — a second copy of these
        # fallback chains (in a future DiffusersBackend, say) could drift from
        # the values reported in ImageResult and the op.progress `total`.
        # Request beats registry beats a last-resort default; sdxl-turbo is a
        # 512x512 model, so the default is never right for every model and the
        # per-model answer belongs in the registry, not in the HTTP layer.
        steps = req.steps or spec.steps or 4
        width = req.width or spec.width or DEFAULT_IMAGE_SIZE
        height = req.height or spec.height or DEFAULT_IMAGE_SIZE
        req = replace(req, steps=steps, width=width, height=height)

        async with self._gpu(spec.name):
            evicted: list[EvictionRecord] = []
            if not self.is_resident(spec.name):
                loaded = await self._load_locked(spec.name)
                evicted = list(loaded.evicted)

            def report(step: int, total: int) -> None:
                progress(step, total)

            png = await backend.generate_image(spec, req, report)
            self._resident[spec.name].last_used = self._clock()
            return ImageResult(
                model=spec.name,
                png=png,
                steps=steps,
                width=width,
                height=height,
                evicted=evicted,
            )

    async def generate_text(
        self,
        req: ChatRequest,
        model: str | None = None,
        on_token: TokenFn = _noop_token,
    ) -> TextResult:
        """One text turn, reserving the model through the same path as a render.

        Fit, eviction and inference happen inside **one** hold of the GPU lock.
        That is the whole reason this route exists rather than letting the
        orchestrator load and then talk to Ollama itself: between a separate load
        and a separate inference call, nothing stops the VramScheduler evicting the
        model the caller just reserved. A client cannot close that race from
        outside; holding the lock across both closes it by construction.
        """
        spec = (
            self._registry.require(model)
            if model
            else self._registry.first_with_role("code")
        )
        if spec is None:
            raise UnknownModel(
                "No model with role 'code' is registered.",
                remedy="Add a text model to models.toml, or use the "
                "anthropic provider.",
                role="code",
            )

        # Checked before residency is touched, for the same reason
        # generate_image does: a misconfigured entry should fail cleanly rather
        # than evict a resident model to load one that cannot answer anyway.
        backend = self._backend_for(spec)
        if not isinstance(backend, TextBackend):
            raise BackendFailure(
                f"Backend {spec.backend!r} cannot answer a text turn.",
                remedy="Register a text-capable backend for this model.",
                model=spec.name,
                backend=spec.backend,
            )

        # Request beats registry beats the model's own default, resolved once
        # here so no backend re-derives it. The window is also what the VRAM
        # hint was measured at, so overriding one without the other is how a
        # fit calculation silently stops describing the load it plans.
        self._require_available(spec.name)
        num_ctx = req.num_ctx or spec.num_ctx
        fit = self.fits.get(spec.name) if self.fits else None
        if fit is not None and fit.num_ctx is not None:
            # The fitted window is the default and the ceiling.
            num_ctx = min(num_ctx, fit.num_ctx) if num_ctx else fit.num_ctx
        req = replace(req, num_ctx=num_ctx)
        # Carried on the spec too, so a cold load uses this window and the
        # VramScheduler's measured VRAM describes the model the turn actually ran on.
        spec = replace(spec, num_ctx=num_ctx)

        async with self._gpu(spec.name):
            evicted: list[EvictionRecord] = []
            if not self.is_resident(spec.name):
                loaded = await self._load_locked(spec.name, spec=spec)
                evicted = list(loaded.evicted)

            result: ChatResult = await backend.chat(spec, req, on_token)
            self._resident[spec.name].last_used = self._clock()
            return TextResult(
                model=spec.name,
                content=result.content,
                tool_calls=result.tool_calls,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                done_reason=result.done_reason,
                num_ctx=num_ctx,
                evicted=evicted,
            )

    async def _attempt_load(
        self, backend: Backend, spec: ModelSpec, report: ProgressFn
    ) -> tuple[int, int]:
        """One load attempt; returns (VRAM actually consumed, load time in ms).

        Both figures are measured across just this attempt, not across any
        eviction or retry around it — a `free_before`/`started` captured at
        the top of `_load` would count the evictions' effect on VRAM and
        elapsed time against the model that merely happened to load next.
        """
        free_before = self._probe.free_mb()
        started = time.perf_counter()
        await backend.load(spec, report)
        load_ms = int((time.perf_counter() - started) * 1000)
        observed = max(1, free_before - self._probe.free_mb())
        return observed, load_ms

    async def _load(
        self,
        spec: ModelSpec,
        pin: bool,
        evicted: list[EvictionRecord],
        progress: ProgressFn,
    ) -> LoadResult:
        backend = self._backend_for(spec)

        def report(step: int, total: int) -> None:
            progress(step, total)
            self._emit("model.loading", model=spec.name, pct=int(step * 100 / total))

        try:
            observed, load_ms = await self._attempt_load(backend, spec, report)
        except BackendOom as first:
            panic = await self._unload_all_unpinned(protect=spec.name)
            evicted = [*evicted, *panic]
            remedy = (
                f"The GPU cannot host {spec.name} right now. Proceed without "
                f"it and record the gap in your report."
            )
            if not panic:
                raise VramExhausted(
                    f"CUDA OOM loading {spec.name} with nothing left to free.",
                    remedy=remedy,
                    model=spec.name,
                    requested_mb=self.observed_mb(spec.name, spec.num_ctx),
                    # Always present, even when empty: the client should not
                    # have to branch on whether the key exists to learn what
                    # the GPU service freed on its behalf.
                    freed=[],
                    free_mb=self._probe.free_mb(),
                ) from first
            try:
                observed, load_ms = await self._attempt_load(backend, spec, report)
            except BackendOom as second:
                raise VramExhausted(
                    f"CUDA OOM loading {spec.name} even after freeing every "
                    f"unpinned model.",
                    remedy=remedy,
                    model=spec.name,
                    requested_mb=self.observed_mb(spec.name, spec.num_ctx),
                    freed=[e.model for e in panic],
                    free_mb=self._probe.free_mb(),
                ) from second
        self._observed[(spec.name, spec.num_ctx)] = observed
        self._resident[spec.name] = Residency(
            spec=spec, vram_mb=observed, pinned=pin, last_used=self._clock(), load_ms=load_ms
        )
        self._emit("model.loaded", model=spec.name, vram_mb=observed, load_ms=load_ms)
        self._emit(
            "vram.changed", used_mb=self._used_mb(), free_mb=self._probe.free_mb()
        )
        return LoadResult(
            model=spec.name, vram_mb=observed, load_ms=load_ms, evicted=evicted
        )

    async def unload(self, name: str, reason: str = "explicit") -> int:
        async with self._gpu(name):
            return await self._unload_locked(name, reason=reason)

    async def _unload_locked(
        self, name: str, reason: str = "explicit", for_model: str | None = None
    ) -> int:
        # Validate against the registry the way load does. Without this a
        # typo'd name returns freed_mb=0 -- indistinguishable from "registered
        # but not resident" -- and the caller concludes VRAM was freed.
        self._registry.require(name)
        residency = self._resident.get(name)
        if residency is None:
            return 0
        backend = self._backend_for(residency.spec)
        await backend.unload(residency.spec)
        del self._resident[name]
        extra = {"for_model": for_model} if for_model is not None else {}
        self._emit(
            "model.unloaded",
            model=name,
            freed_mb=residency.vram_mb,
            reason=reason,
            **extra,
        )
        self._emit(
            "vram.changed", used_mb=self._used_mb(), free_mb=self._probe.free_mb()
        )
        return residency.vram_mb

    async def unload_all(self, reason: str = "shutdown") -> list[str]:
        """Unload everything this GPU service booked, pinned or not.

        For shutdown: Ollama outlives this process and keeps what it loaded,
        so a GPU service that exits without this leaves the card full for the
        next start. One model failing to unload does not stop the rest.
        """
        unloaded: list[str] = []
        async with self._gpu():
            for name in sorted(self._resident):
                try:
                    await self._unload_locked(name, reason=reason)
                    unloaded.append(name)
                except Exception as error:  # noqa: BLE001 - free what we can
                    print(f"unload {name} on shutdown failed: {error}", flush=True)
        return unloaded

    async def sweep_idle(self) -> list[EvictionRecord]:
        """Unload unpinned models unused for the registry's `idle_unload_s`.

        This is the only idle timer on the card. Backends are told to keep
        models forever (Ollama: `keep_alive=-1`) because a backend that expires
        a model on its own leaves the VramScheduler booking VRAM that is already free,
        and `_plan_evictions` then counts it twice. The card still goes back to
        a game after half an hour -- the VramScheduler just does the handing back.
        """
        limit = self._registry.gpu_service.idle_unload_s
        if limit <= 0:
            return []
        async with self._gpu():
            now = self._clock()
            names = [
                r.spec.name
                for r in self._resident.values()
                if not r.pinned and now - r.last_used >= limit
            ]
            records: list[EvictionRecord] = []
            for name in names:
                freed = await self._unload_locked(name, reason="idle")
                records.append(EvictionRecord(model=name, freed_mb=freed, reason="idle"))
            return records

    def _plan_evictions(self, need_mb: int, protect: str) -> list[str]:
        """Least-recently-used unpinned models, in the order they should go."""
        if need_mb <= self.usable_free_mb():
            return []

        candidates = sorted(
            (
                r
                for r in self._resident.values()
                if not r.pinned and r.spec.name != protect
            ),
            key=lambda r: r.last_used,
        )

        plan: list[str] = []
        projected = self.usable_free_mb()
        for residency in candidates:
            plan.append(residency.spec.name)
            projected += residency.vram_mb
            if projected >= need_mb:
                return plan

        pinned = sorted(r.spec.name for r in self._resident.values() if r.pinned)
        raise VramExhausted(
            f"Cannot free {need_mb}MB; only {projected}MB reachable by eviction.",
            remedy=self._exhaustion_remedy(pinned),
            requested_mb=need_mb,
            usable_free_mb=self.usable_free_mb(),
            reachable_mb=projected,
            pinned=pinned,
            freed=[],
        )

    def _exhaustion_remedy(self, pinned: list[str]) -> str:
        """What the caller should actually do, which depends on who holds the VRAM.

        This is a machine with one card that may also run games. When nothing is
        resident the GPU service is not the reason the card is full, and telling the
        caller to unpin a model -- with an empty `pinned` list right beside it
        -- misdirects them on the most likely real-hardware failure.
        """
        if not self._resident:
            return (
                "The GPU service holds no models, so it has nothing to evict: another "
                "process on this machine (a game, another CUDA job) is holding "
                "the card's VRAM. Close it and retry, or proceed without this "
                "model and record the gap in your report."
            )
        if pinned:
            return (
                f"Every remaining resident model is pinned "
                f"({', '.join(pinned)}). Unpin one, or proceed without this "
                f"model and note the gap."
            )
        return (
            "Freeing every unpinned model still leaves too little VRAM, so "
            "another process on this machine is holding the rest. Close it and "
            "retry, or proceed without this model and record the gap in your "
            "report."
        )

    async def _unload_each(
        self, names: list[str], for_model: str
    ) -> list[EvictionRecord]:
        records: list[EvictionRecord] = []
        for name in names:
            freed = await self._unload_locked(
                name, reason="evicted", for_model=for_model
            )
            records.append(EvictionRecord(model=name, freed_mb=freed, reason="evicted"))
        return records

    async def _make_room(self, need_mb: int, protect: str) -> list[EvictionRecord]:
        return await self._unload_each(self._plan_evictions(need_mb, protect), protect)

    async def _unload_all_unpinned(self, protect: str) -> list[EvictionRecord]:
        names = [
            r.spec.name
            for r in self._resident.values()
            if not r.pinned and r.spec.name != protect
        ]
        return await self._unload_each(names, protect)
