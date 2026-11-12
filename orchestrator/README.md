# Orchestrator

The orchestrator: agents, routing, budgets, the Ledger and the Hub. Dials the
GPU service over HTTP+SSE on `127.0.0.1:7311`.

## Run it

```bash
export OHH_GPU_TOKEN=<the token the GPU service was started with>
node src/main.ts "a brass agent on black, high contrast"
```

Prints the config it resolved, the GPU service's health, each turn, each tool call
with its risk tier and duration, live render progress over SSE, and the rows it
left in the Ledger. The PNG lands in `artifacts/`.

The GPU service must be serving. From `gpu-service/`:

```bash
export OHH_GPU_TOKEN=<a long random string>
python -m ohh_gpu.main
```

## Run the tree

`lead` coordinates; it hands one task at a time to another agent (`site`,
`scribe`) with a slice of its budget and reads the report back.

```bash
node src/main.ts --agent lead "make a hero image for a landing page and publish it"
```

The Anthropic key goes in `.env` at the repo root as `OHH_ANTHROPIC_API_KEY` (gitignored).
Never export `ANTHROPIC_API_KEY`: Claude Code reads that name too and will bill
it instead of a subscription.

## Where each agent runs

Each agent file's `model` line decides where its runs go. There is no separate
provider setting.

- A name from `models.toml` (e.g. `llama3.1:8b`) runs on the local GPU.
- A `claude-*` name (e.g. `claude-sonnet-5-5`) runs on Claude when
  `OHH_ANTHROPIC_API_KEY` is set. Without a key it runs locally instead, and
  the run is recorded as `local-no-key`.
- No `model` line runs locally on the registry's `role = "code"` model.
- Any other name stops the agent from loading, with an error naming its file.

As shipped, `lead` plans on Claude (`claude-sonnet-5-5`) and `site` and
`scribe` run locally. With no key the orchestrator runs in local-only mode:
every run uses a real local model. A Claude run that reaches the spending
limit finishes on the local model; the Ledger records that run as
`local-fallback`, and other runs are unaffected.

The startup line says which mode is in effect: `mode: routed (Claude + local)`,
`mode: local-only (no Anthropic key configured)`, or `mode: scripted`.
`OHH_PROVIDER=scripted` swaps every model for a fixed three-turn script; it is
for tests and demos only.

When a child reaches a `gated` call (`publish`), the child stops and so does `lead`.
Answer it once and the whole tree resumes:

```bash
node src/main.ts --pending              # what is waiting, and on what
node src/main.ts --approve <id>         # or --deny <id>
```

Published files land in `orchestrator/published/` (gitignored). The ledger section
prints each Run in the tree and a tree total.

## The Hub

A page on this machine for the same things: speak a goal, watch the board, answer
what needs you, see what the GPU service is holding.

```bash
npm run hub     # prints  hub  http://127.0.0.1:6680/  and the mode
```

Open the printed address. The port is `OHH_HUB_PORT` (default `6680`). It
listens on `127.0.0.1` only, so it is local by design: nothing else on the
network can reach it, and the GPU service's token never leaves the process. The
Anthropic key is the same `OHH_ANTHROPIC_API_KEY` in `.env` at the repo root.

- **The card**, across the top of every page: the models loaded, the GPU memory
  that is not a loaded model (labelled "not models": mostly the GPU service's own
  runtime overhead, plus any other program using the GPU), what is free, the
  safety headroom, and the last load or unload.
- **The goal box.** Type a goal and press Start. The budget starts at $1.50;
  change it, up to $10.
- **The board**: *Working*, *Needs you* and *Done*. A `gated` call (`publish`)
  waits under *Needs you* with Approve and Deny. A failed goal, or one stopped
  with nothing to decide (its budget spent), is under *Done* with Run again: a
  fresh goal with the same agent, text and budget. Click a card for its story.
- **Models** (`/models`): every model in the registry, with Load and Unload.
  Before a load it says which loaded models it will probably unload, and asks;
  after it, what it actually unloaded, or why it failed.

The terminal commands above keep working alongside it, and answer the same
Ledger: approve in one place, see it in the other.

## Test

```bash
npm test               # node --test && tsc --noEmit   (needs no GPU)
npm run test:integration   # needs OHH_INTEGRATION=1 and a live GPU service
```

`tsc --noEmit` is part of the gate rather than a build step. Node 24 runs
TypeScript by **stripping** types, which does not check them — `node foo.ts`
will happily run code with a type error in it. Without the gate the annotations
would be decoration.

## Configuration

`models.toml` at the repo root is the single source of truth for the GPU service's
address; the orchestrator reads it rather than keeping a second copy.

| Variable | Default | Meaning |
|---|---|---|
| `OHH_GPU_TOKEN` | *(required)* | Bearer token; must match the GPU service's |
| `OHH_GPU_URL` | from the registry | Overrides host and port entirely |
| `OHH_ANTHROPIC_API_KEY` | unset | Enables Claude for `claude-*` agents; unset means local-only (put it in the repo-root `.env`) |
| `OHH_PROVIDER` | unset | `scripted` for the fixed test/demo script; anything else is an error |
| `OHH_LEDGER` | `orchestrator/ledger.db` | sqlite file |
| `OHH_ARTIFACTS` | `orchestrator/artifacts` | Where fetched artifacts land |
| `OHH_HUB_PORT` | `6680` | Port for `npm run hub` (127.0.0.1 only) |
| `OHH_RUNNING_DIR` | `orchestrator/running` | Where running goals record their owner |
| `OHH_MAX_TURNS` | `20` | The Loop's hard stop |
| `OHH_INTEGRATION` | unset | `1` opts into the hardware tests |

## Layout

```
src/config/toml.ts     strict TOML subset; refuses what it cannot parse
src/config/config.ts   registry + env -> Config
src/ledger/ledger.ts   node:sqlite, six tables
src/gpu/http.ts      node:http request helper  (see "Why not fetch")
src/gpu/sse.ts       byte stream -> SSE frames
src/gpu/client.ts    the op protocol; the only wire-aware module
src/gateway/types.ts   TurnEvent — the provider contract
src/gateway/            the providers: anthropic, ollama, scripted, failover
src/agents/            agent definitions (agents/*.toml) and the factory
src/tools/             registry, gpu_status, gpu_generate_image
src/loop/loop.ts       one Run: turn, dispatch, repeat
src/run/               the shared runner: a goal, a retry, a decision
src/hub/               the Hub's server, board, goal detail and model list
src/gpu/watch.ts     follows the GPU service's event stream, reports the link
src/hub.ts             `npm run hub`
hub/                   the Hub's pages, scripts and style (no build step)
src/main.ts
```

## Two things that will look odd

**Why not `fetch`.** The GPU client uses `node:http` so one `send` helper serves
both the plain-HTTP GPU service and the HTTPS provider APIs, and so the SSE
reader and the error envelope are shared. See `src/gpu/http.ts`.

**Why the client polls as well as streaming.** `GpuClient` subscribes to
`/events` *before* posting, then races the terminal SSE event against a 2s poll
of `/ops/{id}`. The GPU service's own tests assert that events emitted before a
subscriber exists are never replayed, and that a full queue drops the oldest —
so `op.done` can simply never arrive. The poll is the backstop; the stream is
the mechanism. `test/gpu-client.test.ts` drops the terminal event on purpose
and asserts the op still completes.
