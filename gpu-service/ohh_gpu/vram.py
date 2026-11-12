"""Device-wide VRAM measurement.

``torch.cuda.mem_get_info()`` answers "how much can *this CUDA context* still
allocate?" Under Windows' WDDM driver model the OS hands each process a budget
it can page against, so another process's resident weights are never
subtracted from the answer. Measured on the real 3060: Ollama loaded 4871MB
and the torch reading did not move by a single byte, which booked every Ollama
model at the ``max(1, ...)`` floor in the VramScheduler and stopped it ever evicting.

NVML asks the driver what is actually on the device -- the same question
``nvidia-smi`` asks, and the right one here, since the GPU service's own
Ollama is by construction a separate process.

``pynvml`` is imported inside a function so this module stays importable on
a machine without the GPU extras, where it is not installed.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from .errors import NvmlUnavailable

_REMEDY = (
    "Install the GPU extra on the GPU service host (pip install -e '.[gpu]') and "
    "confirm `nvidia-smi` runs. NVML itself ships with the NVIDIA driver."
)


def _pynvml() -> Any:
    try:
        import pynvml
    except ImportError as error:
        raise NvmlUnavailable(
            "The nvidia-ml-py bindings are not importable.", remedy=_REMEDY
        ) from error
    return pynvml


class NvmlVramProbe:
    """Reads device-wide VRAM every call.

    Never cached, for the same reason the probe exists at all: this machine
    may run games, and the GPU service's Ollama is a separate process whose residency
    changes independently of anything the VramScheduler did.
    """

    def __init__(self, device_index: int = 0) -> None:
        nvml = _pynvml()
        try:
            nvml.nvmlInit()
            self._handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as error:  # noqa: BLE001 - any NVML failure is fatal here
            raise NvmlUnavailable(
                f"NVML would not initialise: {error}", remedy=_REMEDY
            ) from error
        self._nvml = nvml
        self._index = device_index

    def name(self) -> str:
        raw = self._nvml.nvmlDeviceGetName(self._handle)
        return raw.decode() if isinstance(raw, bytes) else raw

    def total_mb(self) -> int:
        return self._memory().total // (1024 * 1024)

    def free_mb(self) -> int:
        return self._memory().free // (1024 * 1024)

    def _memory(self) -> Any:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle)


class _FreeProbe(Protocol):
    def free_mb(self) -> int: ...


def settled_free_mb(
    probe: _FreeProbe,
    interval_s: float = 0.2,
    timeout_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Free VRAM once two consecutive readings agree, bounded by ``timeout_s``.

    Ollama acknowledges a release before the device is freed, so one reading
    taken just after can still count a departing model. Bounded and
    non-raising: a card that never settles (a game loading) gets the last
    reading, which is what a single read would have given anyway.
    """
    last = probe.free_mb()
    for _ in range(max(0, int(timeout_s / interval_s))):
        sleep(interval_s)
        current = probe.free_mb()
        if current == last:
            return current
        last = current
    return last
