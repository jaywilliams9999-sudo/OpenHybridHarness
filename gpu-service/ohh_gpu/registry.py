"""Model registry: parses models.toml into typed specs.

``vram_mb`` here is a planning hint used before a model's first real load.
The VramScheduler records observed cost afterwards and prefers it. Config
bootstraps; reality corrects.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InvalidRegistry, UnknownModel

DEFAULT_HEADROOM_MB = 768
# How long an unpinned model may sit unused before the VramScheduler hands its VRAM
# back. The machine may also run games; 0 disables the sweep.
DEFAULT_IDLE_UNLOAD_S = 1800
DEFAULT_PORT = 7311


@dataclass(frozen=True)
class GpuServiceSettings:
    host: str
    port: int
    vram_headroom_mb: int
    idle_unload_s: int = DEFAULT_IDLE_UNLOAD_S


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    vram_mb: int
    role: str
    repo: str | None = None
    dtype: str | None = None
    variant: str | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    # Context window for text models. A weights-level maximum exists per model
    # (qwen3 caps at 40960, llama3.1 declares 131072) and VRAM cost scales with
    # it at a measured 34 MB per 1k tokens on q4_0 KV -- so the window and the
    # vram_mb hint beside it are two readings of one decision.
    num_ctx: int | None = None
    # MB of KV cache per 1000 context tokens, by KV cache type ("f16",
    # "q8_0", "q4_0"). With it, `vram_mb` becomes a reading at a stated window
    # and cache type that can be re-based to another: a model's footprint is a
    # function of (weights, window, cache type), and planning a load against a
    # figure measured under different ones is how the VramScheduler mis-books.
    kv_rates: dict[str, int] | None = None
    # The KV cache type `vram_mb` was measured under.
    measured_kv_cache: str | None = None


@dataclass(frozen=True)
class ModelRegistry:
    gpu_service: GpuServiceSettings
    models: dict[str, ModelSpec]

    def require(self, name: str) -> ModelSpec:
        spec = self.models.get(name)
        if spec is None:
            known = ", ".join(sorted(self.models)) or "(none)"
            raise UnknownModel(
                f"No model named {name!r} in the registry.",
                remedy=f"Use one of the registered models: {known}.",
                model=name,
            )
        return spec

    def first_with_role(self, role: str) -> ModelSpec | None:
        for name in sorted(self.models):
            if self.models[name].role == role:
                return self.models[name]
        return None


def default_registry_path() -> Path:
    override = os.environ.get("OHH_MODELS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "models.toml"


REQUIRED_MODEL_KEYS = ("backend", "vram_mb")


def _kv_rates(entry: dict[str, Any]) -> dict[str, int] | None:
    rates = {
        kv: int(entry[f"kv_mb_per_1k_{kv}"])
        for kv in ("f16", "q8_0", "q4_0")
        if f"kv_mb_per_1k_{kv}" in entry
    }
    return rates or None


def _model_spec(name: str, entry: dict[str, Any]) -> ModelSpec:
    """Turn one ``[models."..."]`` table into a spec, or say exactly what is wrong.

    A bare ``KeyError`` here is the operator's first-run experience, and a
    traceback reading ``KeyError: 'backend'`` does not say which of several
    model tables is short a key.
    """
    for key in REQUIRED_MODEL_KEYS:
        if key not in entry:
            raise InvalidRegistry(
                f"Model {name!r} in the registry has no {key!r} key.",
                remedy=(
                    f'Add {key} = ... to the [models."{name}"] table in '
                    f"models.toml; every model needs both "
                    f"{' and '.join(REQUIRED_MODEL_KEYS)}."
                ),
                model=name,
                missing_key=key,
            )
    try:
        vram_mb = int(entry["vram_mb"])
    except (TypeError, ValueError) as error:
        raise InvalidRegistry(
            f"Model {name!r} has a non-numeric vram_mb: {entry['vram_mb']!r}.",
            remedy=(
                f'Set vram_mb in the [models."{name}"] table to an integer '
                f"number of megabytes."
            ),
            model=name,
            missing_key="vram_mb",
        ) from error
    return ModelSpec(
        name=name,
        backend=entry["backend"],
        vram_mb=vram_mb,
        role=entry.get("role", "general"),
        repo=entry.get("repo"),
        dtype=entry.get("dtype"),
        variant=entry.get("variant"),
        steps=entry.get("steps"),
        width=entry.get("width"),
        height=entry.get("height"),
        num_ctx=entry.get("num_ctx"),
        kv_rates=_kv_rates(entry),
        measured_kv_cache=entry.get("measured_kv_cache"),
    )


def load_registry(path: Path) -> ModelRegistry:
    raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    service_raw = raw.get("gpu_service", {})
    gpu_service = GpuServiceSettings(
        host=service_raw.get("host", "127.0.0.1"),
        port=int(service_raw.get("port", DEFAULT_PORT)),
        vram_headroom_mb=int(service_raw.get("vram_headroom_mb", DEFAULT_HEADROOM_MB)),
        idle_unload_s=int(service_raw.get("idle_unload_s", DEFAULT_IDLE_UNLOAD_S)),
    )
    models = {
        name: _model_spec(name, entry) for name, entry in raw.get("models", {}).items()
    }
    return ModelRegistry(gpu_service=gpu_service, models=models)
