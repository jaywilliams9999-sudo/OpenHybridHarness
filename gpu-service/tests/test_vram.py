"""The VramScheduler's VRAM probe.

These tests exist because `torch.cuda.mem_get_info()` — the original probe —
answers "how much can *this CUDA context* still allocate?", not "what is on
the device?". Measured on the real 3060: Ollama loaded 4871MB and the torch
reading did not move by a single byte, so every Ollama model was booked at the
`max(1, ...)` floor and the VramScheduler never evicted anything. NVML asks the
driver, which is the question that actually matters here.
"""

import sys
import types

import pytest

from ohh_gpu.errors import NvmlUnavailable


def _stub_pynvml(monkeypatch, *, total, free, name="NVIDIA GeForce RTX 3060",
                 init_error=None, absent=False):
    if absent:
        monkeypatch.setitem(sys.modules, "pynvml", None)
        return None

    pynvml = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    def nvmlInit():
        if init_error:
            raise NVMLError(init_error)

    pynvml.NVMLError = NVMLError
    pynvml.nvmlInit = nvmlInit
    pynvml.nvmlShutdown = lambda: None
    pynvml.nvmlDeviceGetHandleByIndex = lambda index: f"handle-{index}"
    pynvml.nvmlDeviceGetName = lambda handle: name
    pynvml.reads = []

    def nvmlDeviceGetMemoryInfo(handle):
        value = free[len(pynvml.reads)] if isinstance(free, list) else free
        pynvml.reads.append(handle)
        return types.SimpleNamespace(total=total, free=value, used=total - value)

    pynvml.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    return pynvml


def test_nvml_probe_reports_the_whole_device(monkeypatch):
    _stub_pynvml(monkeypatch, total=12288 * 1024**2, free=5845 * 1024**2)
    from ohh_gpu.vram import NvmlVramProbe

    probe = NvmlVramProbe()
    assert probe.total_mb() == 12288
    assert probe.free_mb() == 5845
    assert probe.name() == "NVIDIA GeForce RTX 3060"


def test_nvml_probe_rereads_memory_on_every_call(monkeypatch):
    """Never cached: the machine may run games and a separate Ollama process."""
    stub = _stub_pynvml(
        monkeypatch,
        total=12288 * 1024**2,
        free=[10716 * 1024**2, 5845 * 1024**2],
    )
    from ohh_gpu.vram import NvmlVramProbe

    probe = NvmlVramProbe()
    assert probe.free_mb() == 10716
    assert probe.free_mb() == 5845
    assert len(stub.reads) == 2


def test_nvml_probe_decodes_a_bytes_device_name(monkeypatch):
    """nvidia-ml-py returned bytes before 12.x and str after."""
    _stub_pynvml(
        monkeypatch,
        total=12288 * 1024**2,
        free=5845 * 1024**2,
        name=b"NVIDIA GeForce RTX 3060",
    )
    from ohh_gpu.vram import NvmlVramProbe

    assert NvmlVramProbe().name() == "NVIDIA GeForce RTX 3060"


def test_nvml_probe_refuses_to_construct_when_nvml_will_not_initialise(monkeypatch):
    _stub_pynvml(
        monkeypatch, total=1, free=1, init_error="Driver Not Loaded"
    )
    from ohh_gpu.vram import NvmlVramProbe

    with pytest.raises(NvmlUnavailable) as caught:
        NvmlVramProbe()
    assert "Driver Not Loaded" in str(caught.value)
    assert caught.value.remedy


def test_nvml_probe_refuses_to_construct_when_the_bindings_are_missing(monkeypatch):
    _stub_pynvml(monkeypatch, total=0, free=0, absent=True)
    from ohh_gpu.vram import NvmlVramProbe

    with pytest.raises(NvmlUnavailable) as caught:
        NvmlVramProbe()
    assert caught.value.remedy


MODELS_TOML = """
[gpu_service]
host = "example"
port = 7311
vram_headroom_mb = 768

[models."qwen2.5-coder:7b"]
backend = "ollama"
vram_mb = 4700
role = "code"
"""


def test_build_app_refuses_to_start_when_the_gpu_cannot_be_measured(
    monkeypatch, tmp_path
):
    """The policy: no probe, no GPU service.

    A probe blind to other processes reports a card emptier than it is, so the
    VramScheduler declines to evict and the device overcommits under load. Refusing to
    start is the louder failure, and matches how main.py already treats a
    missing OHH_GPU_TOKEN.
    """
    path = tmp_path / "models.toml"
    path.write_text(MODELS_TOML, encoding="utf-8")
    monkeypatch.setenv("OHH_GPU_TOKEN", "a-token")
    monkeypatch.setenv("OHH_MODELS", str(path))
    _stub_pynvml(monkeypatch, total=0, free=0, absent=True)

    from ohh_gpu.main import build_app

    with pytest.raises(SystemExit) as caught:
        build_app()
    assert "measure" in str(caught.value).lower()


# -- settling -----------------------------------------------------------


class Readings:
    """A probe whose free figure follows a script, then holds its last value."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)
        self.reads = 0

    def free_mb(self) -> int:
        self.reads += 1
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


def test_settled_free_mb_waits_for_two_agreeing_readings():
    # A release frees the device a beat after Ollama acknowledges it.
    from ohh_gpu.vram import settled_free_mb

    probe = Readings([2000, 5000, 7000, 7000])
    sleeps: list[float] = []
    assert settled_free_mb(probe, interval_s=0.1, timeout_s=2.0, sleep=sleeps.append) == 7000
    assert probe.reads == 4
    assert sleeps == [0.1, 0.1, 0.1]


def test_settled_free_mb_is_bounded_and_returns_the_last_reading():
    from ohh_gpu.vram import settled_free_mb

    climbing = iter(range(1000, 100000, 100))

    class Never:
        def free_mb(self) -> int:
            return next(climbing)

    sleeps: list[float] = []
    result = settled_free_mb(Never(), interval_s=0.5, timeout_s=2.0, sleep=sleeps.append)
    assert len(sleeps) == 4  # 2.0 s of 0.5 s waits, then give up
    assert result == 1400
