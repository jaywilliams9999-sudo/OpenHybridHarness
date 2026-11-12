# OpenHybridHarness

An agent harness for a single NVIDIA machine. It runs its own agent loop, routes
work across cloud models and locally-hosted ones, and treats the GPU as a
scheduled resource rather than something you hope is free.

Two processes on one machine:

```
┌────────────────────────────┐             ┌──────────────────────────┐
│  ORCHESTRATOR  TypeScript  │  HTTP+SSE   │  GPU SERVICE  Python     │
│                            │◄───────────►│                          │
│  turns, tools, the Ledger, │  127.0.0.1  │  VRAM scheduler: load,   │
│  budgets, the Hub          │    :7311    │  unload, render, report  │
└────────────────────────────┘             └──────────────────────────┘
```

## Quickstart

Start the GPU service (the machine with the NVIDIA GPU):

```bash
cd gpu-service
export OHH_GPU_TOKEN=<a long random string>
python -m ohh_gpu.main
```

Then run an agent against it:

```bash
cd orchestrator
export OHH_GPU_TOKEN=<the same token>
node src/main.ts "a brass agent on black, high contrast"
```

You get the resolved config, the GPU service's health, each turn, each tool call
with its risk tier and duration, live render progress over SSE, and the rows left
in the Ledger. The PNG lands in `orchestrator/artifacts/`.

## Tests

```bash
cd gpu-service  && python -m pytest    # no GPU required
cd orchestrator && npm test            # node --test, then tsc --noEmit
```

Both suites are green on a machine with no GPU. Anything needing real hardware
is tagged and excluded by default: `pytest -m integration` for the GPU service,
`OHH_INTEGRATION=1 npm run test:integration` for the orchestrator.

## Repository

```
models.toml          model registry + the GPU service's address; read by both sides
gpu-service/         Python. The VRAM scheduler and its backends.   -> gpu-service/README.md
orchestrator/        TypeScript. Agents, routing, budgets, Hub.     -> orchestrator/README.md
docs/ARCHITECTURE.md how it fits together, with diagrams
```

## Where to read next

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — components, the op
  protocol, the eviction decision, the Ledger schema, and the measurements that
  changed the design
