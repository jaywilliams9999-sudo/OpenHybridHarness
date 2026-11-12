"""HTTP surface. Holds no scheduling policy — every decision lives in the VramScheduler.

Long operations return 202 with an op_id and run as background tasks: a cold
SDXL load is 20-60s, and a synchronous call there means an HTTP timeout tuned
to the slowest cold start with no progress visible while it happens.
"""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .artifacts import ArtifactStore
from .backends.base import ChatRequest, ImageRequest
from .errors import GpuServiceError, ModelUnavailable, UnknownArtifact
from .events import EventBus, sse_format
from .ops import Op, OpRegistry
from .scheduler import VramScheduler


class LoadBody(BaseModel):
    model: str
    pin: bool = False


class UnloadBody(BaseModel):
    model: str


class ChatBody(BaseModel):
    """One text turn.

    `messages` and `tools` pass through to the backend unchanged: the orchestrator's
    Gateway owns the translation from its own transcript, and a second opinion
    about that mapping living here is how the two drift apart. They are typed
    as loose dicts for the same reason -- the GPU service schedules the GPU, it does
    not validate a chat protocol it has no stake in.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    # Unset means "ask the VramScheduler", which resolves the window from the model's
    # registry entry. The bound is the largest window any model here declares.
    num_ctx: int | None = Field(default=None, ge=256, le=1_048_576)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131_072)
    model: str | None = None


class ImageBody(BaseModel):
    prompt: str
    negative: str | None = None
    steps: int | None = Field(default=None, ge=1, le=150)
    # Unset means "ask the VramScheduler", which resolves the model's own size from
    # the registry. Bounds still apply to values the caller does supply.
    width: int | None = Field(default=None, ge=64, le=2048)
    height: int | None = Field(default=None, ge=64, le=2048)
    seed: int | None = None
    model: str | None = None


def _auth_dependency(token: str) -> Callable[..., Awaitable[None]]:
    # Compare bytes, never str. `secrets.compare_digest` on two `str` raises
    # TypeError for any character above U+007F, and Starlette latin-1-decodes
    # request headers, so a single 0x80+ byte in the Authorization header
    # reaches this comparison and turns an unauthenticated request into a 500
    # on every route. `surrogateescape` round-trips whatever latin-1 produced
    # without raising on its way back to bytes.
    expected_bytes = f"Bearer {token}".encode()

    async def verify(authorization: str | None = Header(default=None)) -> None:
        if authorization is None or not secrets.compare_digest(
            authorization.encode("utf-8", "surrogateescape"), expected_bytes
        ):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "UNAUTHORIZED",
                    "error": "Missing or invalid bearer token.",
                    "remedy": "Send Authorization: Bearer $OHH_GPU_TOKEN.",
                },
            )

    return verify


def create_app(
    scheduler: VramScheduler,
    ops: OpRegistry,
    events: EventBus,
    artifacts: ArtifactStore,
    token: str,
    version: str,
    idle_sweep_interval_s: float = 60.0,
    after_reconcile: Callable[[], None] | None = None,
) -> FastAPI:
    auth = Depends(_auth_dependency(token))

    async def sweep_forever() -> None:
        # The VramScheduler owns the only idle timer on the card (see sweep_idle).
        # One bad sweep -- Ollama briefly unreachable -- must not end the
        # loop, or idle models would hold the card until the next restart.
        while True:
            await asyncio.sleep(idle_sweep_interval_s)
            try:
                for record in await scheduler.sweep_idle():
                    print(f"idle: unloaded {record.model} ({record.freed_mb}MB)", flush=True)
            except Exception as error:  # noqa: BLE001 - keep sweeping
                print(f"idle sweep failed: {error}", flush=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Before serving anything: release models a previous GPU service left behind.
        # Ollama outlives this process and keeps whatever it holds (the GPU service
        # loads with keep_alive=-1), so without this a restart leaves the card partly occupied by
        # something the VramScheduler cannot account for or evict.
        try:
            released = await scheduler.reconcile()
            if released:
                print(f"reconciled: released {', '.join(released)}", flush=True)
        except Exception as error:  # noqa: BLE001 - never block startup
            print(f"reconcile skipped: {error}", flush=True)
        if after_reconcile is not None:
            try:
                after_reconcile()
            except Exception as error:  # noqa: BLE001 - never block startup
                print(f"model fit skipped: {error}", flush=True)
        sweeper = asyncio.create_task(sweep_forever())
        try:
            yield
        finally:
            sweeper.cancel()
            # Ctrl+C leaves the card free: Ollama would otherwise keep what
            # this process loaded, and the next start would find it full.
            try:
                for name in await scheduler.unload_all():
                    print(f"shutdown: unloaded {name}", flush=True)
            except Exception as error:  # noqa: BLE001 - never block shutdown
                print(f"shutdown unload failed: {error}", flush=True)
            # Without this, uvicorn waits on SSE generators that never finish.
            events.close()

    app = FastAPI(
        title="OpenHybridHarness GPU service", version=version, dependencies=[auth], lifespan=lifespan
    )
    app.state.tasks = set()
    app.state.scheduler = scheduler
    app.state.ops = ops
    app.state.events = events
    app.state.artifacts = artifacts

    def spawn(coro: Awaitable[None]) -> None:
        task = asyncio.create_task(coro)
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)

    app.state.spawn = spawn

    @app.exception_handler(GpuServiceError)
    async def gpu_service_error_handler(_request: Request, error: GpuServiceError) -> JSONResponse:
        # Every route, not just the background-op path: a GpuServiceError reaching
        # the client without its remedy is the one failure mode the caller
        # cannot act on.
        status = 409 if isinstance(error, ModelUnavailable) else 500
        return JSONResponse(status_code=status, content=error.to_dict())

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        # One envelope, one depth. FastAPI's default handler wraps `detail` in
        # {"detail": ...}, so a 401/404 raised here put `remedy` one level
        # deeper than a GpuServiceError's 500 did -- and `remedy` is the single
        # field the error rule exists to deliver. Unwrap our dict details to
        # the top level; leave a plain-string detail (Starlette's own 404/405)
        # in the shape every HTTP client already expects.
        detail = error.detail
        content = detail if isinstance(detail, dict) else {"detail": detail}
        return JSONResponse(
            status_code=error.status_code,
            content=jsonable_encoder(content),
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # An agent guessing a parameter range lands here. FastAPI's raw
        # {"detail": [...pydantic...]} has no code and no remedy at all, which
        # violates the error rule outright on the one path a caller is most
        # likely to hit by accident.
        raw = error.errors()
        fields = sorted(
            {".".join(str(part) for part in item["loc"][1:]) or "body" for item in raw}
        )
        return JSONResponse(
            status_code=422,
            content={
                "code": "BAD_REQUEST",
                "error": f"Request rejected: {', '.join(fields)} failed validation.",
                "remedy": (
                    f"Correct {', '.join(fields)} to satisfy the constraints listed "
                    f"under `fields`, then retry the request. Retrying it "
                    f"unchanged will fail identically."
                ),
                "fields": jsonable_encoder(raw),
            },
        )

    async def run_op(op: Op, work: Awaitable[dict[str, Any]]) -> None:
        try:
            result = await work
        except GpuServiceError as error:
            ops.fail(op.id, error.to_dict())
            events.emit("op.failed", op_id=op.id, **error.to_dict())
        except Exception as error:  # noqa: BLE001 - never leak a traceback outward
            payload = {
                "code": "BACKEND_FAILURE",
                "error": str(error),
                "remedy": "The GPU service hit an unexpected fault. Proceed without this "
                "operation and record the gap in your report.",
            }
            ops.fail(op.id, payload)
            events.emit("op.failed", op_id=op.id, **payload)
        else:
            ops.finish(op.id, result)
            events.emit("op.done", op_id=op.id, **result)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        snapshot = scheduler.status()
        return {
            "status": "ok",
            "gpu": scheduler.gpu_name,
            "vram_total_mb": snapshot.vram["total_mb"],
            "vram_free_mb": snapshot.vram["free_mb"],
            "version": version,
        }

    @app.get("/status")
    async def status() -> dict[str, Any]:
        body = asdict(scheduler.status())
        body["kv_cache_type"] = scheduler.kv_cache_type
        body["models"] = scheduler.model_fits()
        return body

    @app.post("/load", status_code=202)
    async def load(body: LoadBody) -> dict[str, str]:
        op = ops.create("load")

        async def work() -> dict[str, Any]:
            result = await scheduler.load(body.model, pin=body.pin)
            return {
                "model": result.model,
                "vram_mb": result.vram_mb,
                "load_ms": result.load_ms,
                "already_resident": result.already_resident,
                "evicted": [asdict(e) for e in result.evicted],
            }

        spawn(run_op(op, work()))
        return {"op_id": op.id}

    @app.post("/unload")
    async def unload(body: UnloadBody) -> dict[str, int]:
        freed = await scheduler.unload(body.model)
        return {"freed_mb": freed}

    @app.get("/ops/{op_id}")
    async def read_op(op_id: str) -> dict[str, Any]:
        op = ops.get(op_id)
        if op is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "UNKNOWN_OP",
                    "error": f"No operation {op_id!r}.",
                    "remedy": "Re-issue the request to get a fresh op_id.",
                },
            )
        return asdict(op)

    @app.get("/events")
    async def stream_events() -> StreamingResponse:
        async def generator():
            with events.subscribe() as subscription:
                async for event in subscription:
                    yield sse_format(event)

        return StreamingResponse(generator(), media_type="text/event-stream")

    @app.post("/generate/image", status_code=202)
    async def generate_image(body: ImageBody) -> dict[str, str]:
        op = ops.create("generate_image")

        def report(step: int, total: int) -> None:
            events.emit(
                "op.progress", op_id=op.id, kind="image", step=step, total=total
            )

        async def work() -> dict[str, Any]:
            result = await scheduler.generate_image(
                ImageRequest(
                    prompt=body.prompt,
                    negative=body.negative,
                    steps=body.steps,
                    width=body.width,
                    height=body.height,
                    seed=body.seed,
                ),
                model=body.model,
                progress=report,
            )
            artifact_id = artifacts.put_png(
                result.png,
                meta={
                    "prompt": body.prompt,
                    "negative": body.negative,
                    # Resolved values, not the request's: an unspecified size
                    # is answered by the registry, and recording `null` here
                    # would lose the one record of what was actually rendered.
                    "steps": result.steps,
                    "width": result.width,
                    "height": result.height,
                    "seed": body.seed,
                    "model": result.model,
                },
            )
            return {
                "model": result.model,
                "steps": result.steps,
                "artifact_id": artifact_id,
                "artifact_url": artifacts.url(artifact_id),
                "evicted": [asdict(e) for e in result.evicted],
            }

        spawn(run_op(op, work()))
        return {"op_id": op.id}

    @app.post("/generate/text", status_code=202)
    async def generate_text(body: ChatBody) -> dict[str, str]:
        """202 and an op_id, like every other long operation.

        A cold load plus a 103k-token prefill is minutes of work (measured on this card), so
        a synchronous call here would need an HTTP timeout tuned to the slowest
        possible prefill with nothing visible while it ran.

        Tokens are emitted as `op.token` frames for live observation. They are
        **advisory**: the event bus drops frames when a subscriber falls behind,
        so the authoritative reply is the op's terminal `content`. A client that
        rebuilt the text from these frames could lose words silently.
        """
        op = ops.create("generate_text")

        def on_token(text: str) -> None:
            events.emit("op.token", op_id=op.id, kind="text", text=text)

        async def work() -> dict[str, Any]:
            result = await scheduler.generate_text(
                ChatRequest(
                    messages=body.messages,
                    tools=body.tools,
                    num_ctx=body.num_ctx,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                ),
                model=body.model,
                on_token=on_token,
            )
            return {
                "model": result.model,
                "content": result.content,
                "tool_calls": result.tool_calls,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "done_reason": result.done_reason,
                "num_ctx": result.num_ctx,
                "evicted": [asdict(e) for e in result.evicted],
            }

        spawn(run_op(op, work()))
        return {"op_id": op.id}

    @app.get("/artifacts/{artifact_id}.png")
    async def read_artifact(artifact_id: str) -> FileResponse:
        try:
            path = artifacts.path(artifact_id)
        except UnknownArtifact as error:
            raise HTTPException(status_code=404, detail=error.to_dict()) from error
        return FileResponse(path, media_type="image/png")

    return app
