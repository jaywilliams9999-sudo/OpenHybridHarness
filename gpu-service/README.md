# GPU service

The GPU service. Runs on the machine with the NVIDIA GPU; the orchestrator
dials it over HTTP+SSE on 127.0.0.1:7311.

## Install

Tests only (no GPU libraries):

    cd gpu-service && pip install -e ".[dev]"

Serving (GPU libraries):

    cd gpu-service && pip install -e ".[gpu,dev]"

## Preflight (run once, before the first real serve)

`models.toml`'s `vram_mb` entries are planning hints, not measurements —
nobody has run this against the real card yet. Before the first real serve,
confirm this machine's actual capabilities:

    python --version
    nvidia-smi --query-gpu=name,memory.total --format=csv
    ollama --version
    ollama list
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Then pull the code model if it is absent:

    ollama pull qwen2.5-coder:7b

If the card's actual VRAM differs from what `models.toml` assumes, correct
`vram_mb` there. After the first real image generation, replace the
`sdxl-turbo` and `qwen2.5-coder:7b` `vram_mb` hints with the observed figures
printed by the integration test (see Test, below) — config bootstraps,
reality corrects.

## Run

    export OHH_GPU_TOKEN=<a long random string>
    python -m ohh_gpu.main

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `OHH_GPU_TOKEN` | *(required)* | Bearer token for every route |
| `OHH_MODELS` | `../models.toml` | Model registry path |
| `OHH_GPU_ARTIFACTS` | `./artifacts` | Where generated PNGs are written |
| `OHH_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama on this machine |
| `OHH_LOG_LEVEL` | `info` | uvicorn log level |

## Test

Full unit suite (needs no GPU):

    cd gpu-service && python -m pytest

Integration run, on the GPU machine, GPU service already serving:

    python -m pytest tests/test_integration.py -m integration -v
