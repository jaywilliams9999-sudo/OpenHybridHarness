from ohh_gpu.fit import fit_registry, kv_cache_type
from ohh_gpu.registry import default_registry_path, load_registry

REG = load_registry(default_registry_path())
LLAMA = "llama3.1:8b"
UNAVAILABLE = "needs more memory than this card has free"


def test_12gb_q4_keeps_full_window():
    f = fit_registry(REG, 12288, 900, "q4_0")[LLAMA]
    assert f.available and f.num_ctx == 131072 and f.reduced_from is None
    assert f.reason is None


def test_12gb_f16_reduces_window():
    f = fit_registry(REG, 12288, 900, "f16")[LLAMA]
    assert f.available and f.num_ctx == 37888 and f.reduced_from == 131072
    assert f.reason == "context reduced to 37k to fit this card"


def test_8gb_q4_reduces_and_drops_image():
    fits = fit_registry(REG, 8192, 600, "q4_0")
    assert fits[LLAMA].available and fits[LLAMA].num_ctx == 30720
    assert not fits["sdxl-turbo"].available
    assert fits["sdxl-turbo"].num_ctx is None


def test_others_floor_applies():
    assert fit_registry(REG, 8192, 200, "q4_0") == fit_registry(REG, 8192, 1024, "q4_0")


def test_too_small_card_marks_text_model_unavailable():
    f = fit_registry(REG, 6144, 900, "f16")[LLAMA]
    assert f.available is False and f.num_ctx is None
    assert f.reason == UNAVAILABLE


def test_kv_cache_type_resolution():
    assert kv_cache_type({}) == "f16"
    assert kv_cache_type({"OHH_KV_CACHE_TYPE": "q8_0"}) == "q8_0"
    assert kv_cache_type({"OHH_KV_CACHE_TYPE": "bogus"}) == "f16"
    assert kv_cache_type({"OLLAMA_KV_CACHE_TYPE": "q4_0"}) == "f16"
    assert kv_cache_type(
        {"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "1"}
    ) == "q4_0"
    assert kv_cache_type(
        {"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "true"}
    ) == "q4_0"
    assert kv_cache_type(
        {"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "0"}
    ) == "f16"
    assert kv_cache_type(
        {
            "OHH_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_KV_CACHE_TYPE": "q4_0",
            "OLLAMA_FLASH_ATTENTION": "1",
        }
    ) == "q8_0"
