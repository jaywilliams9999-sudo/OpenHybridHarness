"""Fit each registered model to the card this machine actually has.

The registry's numbers were measured on one 12 GB card. On a smaller card, or
with a heavier KV cache type, the same model needs a smaller context window --
or cannot run at all. This module turns (card size, what else is using it, KV
cache type) into a per-model verdict, so the GPU service refuses cleanly up
front instead of crashing out of memory mid-load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .registry import ModelRegistry, ModelSpec

KV_TYPES = ("f16", "q8_0", "q4_0")
MIN_WINDOW = 4096
OTHERS_FLOOR_MB = 1024
UNAVAILABLE_REASON = "needs more memory than this card has free"


@dataclass(frozen=True)
class Fit:
    model: str
    available: bool
    num_ctx: int | None  # effective window; None for non-text models
    reduced_from: int | None  # registry window when it was reduced, else None
    reason: str | None  # plain words when unavailable or reduced


def kv_cache_type(env: Mapping[str, str]) -> str:
    """The KV cache type Ollama will run with, as far as we can tell."""
    ours = env.get("OHH_KV_CACHE_TYPE")
    if ours in KV_TYPES:
        return ours
    theirs = env.get("OLLAMA_KV_CACHE_TYPE")
    if theirs in KV_TYPES and env.get("OLLAMA_FLASH_ATTENTION") in ("1", "true"):
        return theirs
    return "f16"


def _fit_one(spec: ModelSpec, budget_mb: int, kv_type: str) -> Fit:
    unavailable = Fit(spec.name, False, None, None, UNAVAILABLE_REASON)
    rates = spec.kv_rates
    if (
        rates is None
        or spec.num_ctx is None
        or spec.measured_kv_cache not in rates
        or kv_type not in rates
    ):
        if spec.vram_mb <= budget_mb:
            return Fit(spec.name, True, spec.num_ctx, None, None)
        return unavailable
    weights = spec.vram_mb - rates[spec.measured_kv_cache] * spec.num_ctx / 1000
    window = int((budget_mb - weights) / rates[kv_type] * 1000)
    window = min(window, spec.num_ctx)
    window = window // 1024 * 1024
    if window < MIN_WINDOW:
        return unavailable
    if window >= spec.num_ctx:
        return Fit(spec.name, True, spec.num_ctx, None, None)
    return Fit(
        spec.name,
        True,
        window,
        spec.num_ctx,
        f"context reduced to {window // 1024}k to fit this card",
    )


def fit_registry(
    registry: ModelRegistry, total_mb: int, others_mb: int, kv_type: str
) -> dict[str, Fit]:
    budget = (
        total_mb
        - max(others_mb, OTHERS_FLOOR_MB)
        - registry.gpu_service.vram_headroom_mb
    )
    return {
        name: _fit_one(spec, budget, kv_type)
        for name, spec in registry.models.items()
    }
