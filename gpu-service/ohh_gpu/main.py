"""Entry point. Wires the real backends and probe; run on the GPU machine only."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from . import __version__
from .api import create_app
from .artifacts import ArtifactStore
from .backends.diffusers import DiffusersBackend
from .backends.ollama import OllamaBackend
from .dotenv import load_dotenv
from .errors import NvmlUnavailable
from .fit import fit_registry, kv_cache_type
from .events import EventBus
from .registry import default_registry_path, load_registry
from .ops import OpRegistry
from .scheduler import VramScheduler
from .vram import NvmlVramProbe, settled_free_mb


def build_app() -> FastAPI:
    # The repo-root .env fills in what the shell left unset.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    token = os.environ.get("OHH_GPU_TOKEN")
    if not token:
        raise SystemExit(
            "OHH_GPU_TOKEN is not set. The GPU service loads and executes models on "
            "request and must never run unauthenticated, even on a LAN."
        )

    try:
        probe = NvmlVramProbe()
    except NvmlUnavailable as error:
        raise SystemExit(
            f"{error} "
            "The VramScheduler cannot schedule a GPU it cannot measure: a probe "
            "blind to other processes reports a card emptier than it is, so the "
            "GPU service would decline to evict and let the device overcommit."
        ) from error

    registry = load_registry(default_registry_path())
    events = EventBus()
    scheduler = VramScheduler(
        registry=registry,
        probe=probe,
        backends={
            "ollama": OllamaBackend(
                base_url=os.environ.get(
                    "OHH_OLLAMA_URL", "http://127.0.0.1:11434"
                )
            ),
            "diffusers": DiffusersBackend(),
        },
        events=events,
    )
    kv_type = kv_cache_type(os.environ)
    scheduler.kv_cache_type = kv_type

    def fit_to_card() -> None:
        # Runs after reconcile(), so whatever the card still holds belongs to
        # other programs (games, the desktop), not to a model we could evict.
        total = probe.total_mb()
        # Settled, not one reading: a release frees the device a beat after
        # Ollama acknowledges it, and a reading mid-release would count the
        # orphan as another program's memory.
        free = settled_free_mb(probe)
        others_mb = max(0, total - free - scheduler.booked_resident_mb())
        scheduler.fits = fits = fit_registry(registry, total, others_mb, kv_type)
        for fit in fits.values():
            if fit.reason:
                state = "reduced" if fit.available else "unavailable"
                print(f"{fit.model}: {state} - {fit.reason}", flush=True)
        print(f"KV cache assumed: {kv_type}", flush=True)

    artifacts = ArtifactStore(
        root=Path(os.environ.get("OHH_GPU_ARTIFACTS", "./artifacts"))
    )
    return create_app(
        scheduler=scheduler,
        ops=OpRegistry(),
        events=events,
        artifacts=artifacts,
        token=token,
        version=__version__,
        after_reconcile=fit_to_card,
    )


def main() -> None:
    registry = load_registry(default_registry_path())
    uvicorn.run(
        build_app(),
        host=registry.gpu_service.host,
        port=registry.gpu_service.port,
        log_level=os.environ.get("OHH_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
