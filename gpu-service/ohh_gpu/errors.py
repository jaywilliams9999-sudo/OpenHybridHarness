"""Typed errors.

Governing rule: errors that reach a model must be actionable
instructions, not stack traces. Every error therefore carries a ``remedy``
telling the caller what to do instead of merely what went wrong.
"""

from typing import Any, ClassVar


class GpuServiceError(Exception):
    code: ClassVar[str] = "GPU_SERVICE_ERROR"

    def __init__(self, message: str, remedy: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "error": self.message,
            "remedy": self.remedy,
            **self.detail,
        }


class UnknownModel(GpuServiceError):
    code = "UNKNOWN_MODEL"


class VramExhausted(GpuServiceError):
    code = "VRAM_EXHAUSTED"


class BackendOom(GpuServiceError):
    """Raised by a backend when the device refused an allocation.

    The VramScheduler catches this, frees what it can, and retries once before
    surfacing VramExhausted.
    """

    code = "BACKEND_OOM"


class BackendFailure(GpuServiceError):
    code = "BACKEND_FAILURE"


class UnknownArtifact(GpuServiceError):
    code = "UNKNOWN_ARTIFACT"


class NvmlUnavailable(GpuServiceError):
    """The VramScheduler cannot measure the GPU, so it must not schedule one.

    Raised at startup only. A probe that cannot observe other processes'
    allocations reports a card that is emptier than it is, and the VramScheduler
    then declines to evict and lets the device overcommit -- which is worse
    than refusing to start, because it fails silently under load.
    """

    code = "NVML_UNAVAILABLE"


class InvalidRegistry(GpuServiceError):
    """A malformed models.toml. Operator-facing, raised at startup.

    Distinct from UnknownModel: the model was *named*, but its table cannot
    be turned into a ModelSpec. The remedy names the file, the model and the
    key, because this is the first-run experience.
    """

    code = "INVALID_REGISTRY"


class ModelUnavailable(GpuServiceError):
    """The model does not fit on this card at all (see fit.py)."""

    code = "MODEL_UNAVAILABLE"
