"""Diffusers backend.

``torch`` and ``diffusers`` are imported inside method bodies, not at module
scope, so this module imports cleanly on a machine without the GPU extras, where
neither is installed. Only the integration test needs the real libraries.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

from ..errors import BackendFailure, BackendOom
from ..registry import ModelSpec
from .base import ImageRequest, ProgressFn

DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - no GPU extras installed
        raise BackendFailure(
            "torch is not installed on this GPU service host.",
            remedy="Install the GPU extra: pip install -e '.[gpu]'.",
        ) from error
    return torch


def _oom_error(torch: Any) -> type[BaseException]:
    """torch.OutOfMemoryError is recent; torch.cuda.OutOfMemoryError is the
    long-standing name. Catching the wrong one turns a recoverable OOM into an
    AttributeError raised while handling it. ``getattr``'s default argument is
    evaluated eagerly, so this cannot be a one-liner without risking exactly
    that AttributeError when only the newer name exists.
    """
    if hasattr(torch, "OutOfMemoryError"):
        return torch.OutOfMemoryError
    return torch.cuda.OutOfMemoryError


class DiffusersBackend:
    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._pipelines: dict[str, Any] = {}

    async def load(self, spec: ModelSpec, progress: ProgressFn) -> None:
        if not spec.repo:
            raise BackendFailure(
                f"{spec.name} has no `repo` in the registry.",
                remedy="Add a `repo` key naming the Hugging Face model id.",
                model=spec.name,
            )
        # Before touching torch: an unrecognised dtype used to fall through
        # DTYPES.get()'s default and silently run fp16, so `dtype = "fp61"`
        # produced a working load and no signal that the registry was wrong.
        dtype_key = spec.dtype or "fp16"
        if dtype_key not in DTYPES:
            raise BackendFailure(
                f"{spec.name} requests an unknown dtype {dtype_key!r}.",
                remedy=(
                    f"Set dtype in the registry to one of: "
                    f"{', '.join(sorted(DTYPES))}."
                ),
                model=spec.name,
                dtype=dtype_key,
            )
        if spec.name in self._pipelines:
            return

        torch = _torch()
        from diffusers import AutoPipelineForText2Image

        dtype = getattr(torch, DTYPES[dtype_key])
        progress(0, 100)

        def build() -> Any:
            # `dtype` is what the weights are held as; `variant` is which files
            # are fetched. Separate knobs, and variant is only sent when the
            # registry asks for it -- not every repo publishes an fp16 variant,
            # and an unsolicited variant= turns a working load into a 404.
            kwargs: dict[str, Any] = {"dtype": dtype}
            if spec.variant:
                kwargs["variant"] = spec.variant
            pipeline = AutoPipelineForText2Image.from_pretrained(spec.repo, **kwargs)
            return pipeline.to(self._device)

        try:
            self._pipelines[spec.name] = await asyncio.to_thread(build)
        except _oom_error(torch) as error:
            raise BackendOom(
                f"CUDA out of memory loading {spec.name}.",
                remedy="Free VRAM by unloading a resident model, then retry.",
                model=spec.name,
                requested_mb=spec.vram_mb,
            ) from error
        except Exception as error:  # noqa: BLE001
            raise BackendFailure(
                f"Could not load {spec.name} from {spec.repo!r}: {error}",
                remedy="Verify the repo id and that the host can reach Hugging Face.",
                model=spec.name,
            ) from error
        progress(100, 100)

    async def unload(self, spec: ModelSpec) -> None:
        if self._pipelines.pop(spec.name, None) is not None:
            _torch().cuda.empty_cache()

    async def generate_image(
        self, spec: ModelSpec, req: ImageRequest, progress: ProgressFn
    ) -> bytes:
        pipeline = self._pipelines.get(spec.name)
        if pipeline is None:
            raise BackendFailure(
                f"{spec.name} is not resident.",
                remedy="Load the model before generating.",
                model=spec.name,
            )

        torch = _torch()
        steps = req.steps or spec.steps or 4

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(req.seed)

        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            progress(step_index + 1, steps)
            return callback_kwargs

        def run() -> bytes:
            output = pipeline(
                prompt=req.prompt,
                negative_prompt=req.negative,
                num_inference_steps=steps,
                width=req.width,
                height=req.height,
                generator=generator,
                callback_on_step_end=on_step_end,
            )
            buffer = io.BytesIO()
            output.images[0].save(buffer, format="PNG")
            data = buffer.getvalue()
            # Hand the render's activation pool back to the card.
            #
            # Without this the caching allocator keeps every block the render
            # touched, so the model's footprint silently becomes its *peak*
            # rather than its weights. The VramScheduler books the weights, so the
            # difference is memory nothing can see and nothing can free --
            # which is how a 1536x864 render booked at 6988 MB left the card
            # at 12059 MiB used and drove usable_free_mb to 0.
            #
            # The cost is real but small: the next render re-allocates instead
            # of reusing cached blocks. That is the right trade for a daemon
            # that shares one card between an image model and a coder.
            del output
            torch.cuda.empty_cache()
            return data

        try:
            return await asyncio.to_thread(run)
        except _oom_error(torch) as error:
            # Reclaim before reporting: an OOM leaves the pool at high-water
            # mark, and a caller who unloads a model and retries should not
            # be defeated by the failed attempt's own leftovers.
            torch.cuda.empty_cache()
            raise BackendOom(
                f"CUDA out of memory rendering with {spec.name}.",
                remedy="Reduce width/height or unload another model, then retry.",
                model=spec.name,
                requested_mb=spec.vram_mb,
            ) from error
