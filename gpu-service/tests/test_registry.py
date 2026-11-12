from pathlib import Path

import pytest

from ohh_gpu.errors import InvalidRegistry, UnknownModel
from ohh_gpu.registry import load_registry

MODELS_TOML = """
[gpu_service]
host             = "gpu.local"
port             = 7311
vram_headroom_mb = 768

[models."qwen2.5-coder:7b"]
backend = "ollama"
vram_mb = 4700
role    = "code"

[models."sdxl-turbo"]
backend = "diffusers"
repo    = "stabilityai/sdxl-turbo"
vram_mb = 7000
role    = "image"
dtype   = "fp16"
steps   = 4
width   = 512
height  = 512
"""


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(MODELS_TOML, encoding="utf-8")
    return path


def test_loads_gpu_service_settings(registry_path: Path):
    registry = load_registry(registry_path)
    assert registry.gpu_service.host == "gpu.local"
    assert registry.gpu_service.port == 7311
    assert registry.gpu_service.vram_headroom_mb == 768


def test_loads_ollama_model(registry_path: Path):
    spec = load_registry(registry_path).require("qwen2.5-coder:7b")
    assert spec.name == "qwen2.5-coder:7b"
    assert spec.backend == "ollama"
    assert spec.vram_mb == 4700
    assert spec.role == "code"
    assert spec.repo is None


def test_loads_diffusers_model_with_recipe(registry_path: Path):
    spec = load_registry(registry_path).require("sdxl-turbo")
    assert spec.backend == "diffusers"
    assert spec.repo == "stabilityai/sdxl-turbo"
    assert spec.dtype == "fp16"
    assert spec.steps == 4
    assert spec.width == 512
    assert spec.height == 512


def test_size_is_optional_and_left_to_the_scheduler(registry_path: Path):
    # The coder declares no size; the VramScheduler supplies the fallback, so the
    # model spec must say "unset" rather than inventing a number here.
    spec = load_registry(registry_path).require("qwen2.5-coder:7b")
    assert spec.width is None
    assert spec.height is None


def test_unknown_model_raises_with_actionable_remedy(registry_path: Path):
    registry = load_registry(registry_path)
    with pytest.raises(UnknownModel) as excinfo:
        registry.require("llama-99b")
    error = excinfo.value
    assert error.code == "UNKNOWN_MODEL"
    assert "qwen2.5-coder:7b" in error.remedy
    assert error.to_dict()["code"] == "UNKNOWN_MODEL"
    assert error.to_dict()["model"] == "llama-99b"


@pytest.mark.parametrize("missing", ["backend", "vram_mb"])
def test_a_model_table_missing_a_required_key_names_it(tmp_path: Path, missing: str):
    # This is the operator's first-run experience. A bare `KeyError: 'backend'`
    # does not say which of several model tables is short a key.
    keys = {"backend": '"ollama"', "vram_mb": "4700"}
    del keys[missing]
    body = "\n".join(f"{k} = {v}" for k, v in keys.items())
    path = tmp_path / "g.toml"
    path.write_text(f'[models."broken:7b"]\n{body}\n', encoding="utf-8")

    with pytest.raises(InvalidRegistry) as excinfo:
        load_registry(path)
    error = excinfo.value
    assert error.code == "INVALID_REGISTRY"
    assert "broken:7b" in error.message
    assert missing in error.message
    assert missing in error.remedy
    assert "models.toml" in error.remedy
    assert error.detail == {"model": "broken:7b", "missing_key": missing}


def test_a_non_numeric_vram_hint_is_reported_not_raised_as_a_value_error(
    tmp_path: Path,
):
    path = tmp_path / "g.toml"
    path.write_text(
        '[models."broken:7b"]\nbackend = "ollama"\nvram_mb = "lots"\n', encoding="utf-8"
    )
    with pytest.raises(InvalidRegistry) as excinfo:
        load_registry(path)
    assert "broken:7b" in excinfo.value.message
    assert "integer" in excinfo.value.remedy


def test_missing_headroom_defaults_to_768(tmp_path: Path):
    path = tmp_path / "g.toml"
    path.write_text('[gpu_service]\nhost = "h"\nport = 7311\n', encoding="utf-8")
    assert load_registry(path).gpu_service.vram_headroom_mb == 768


def test_defaults_bind_loopback_on_7311(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text("[gpu_service]\n", encoding="utf-8")
    reg = load_registry(path)
    assert reg.gpu_service.host == "127.0.0.1"
    assert reg.gpu_service.port == 7311


def test_the_shipped_registry_declares_kv_rates():
    from ohh_gpu.registry import default_registry_path

    spec = load_registry(default_registry_path()).require("llama3.1:8b")
    assert spec.kv_rates == {"f16": 134, "q8_0": 67, "q4_0": 34}
    assert spec.measured_kv_cache == "q4_0"
