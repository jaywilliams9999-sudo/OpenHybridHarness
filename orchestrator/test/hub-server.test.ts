/**
 * The Hub's HTTP server: the page, the read routes, the SSE relay, and the
 * checks on every request that could spend money.
 *
 * A temp Ledger, a fake GPU service and a recording runner; the server listens on
 * port 0 and is spoken to over `node:http`.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { request as httpRequest, type IncomingHttpHeaders, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger, type Ledger } from "../src/ledger/ledger.ts";
import { GpuError, type GpuClient, type Status } from "../src/gpu/client.ts";
import { RunRefused, type RunnerOptions } from "../src/run/runner.ts";
import { createHubServer, type HubDeps, type HubEvent } from "../src/hub/server.ts";
import { createHubRunner } from "../src/hub/drive.ts";
import type { Config } from "../src/config/config.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";
import { createRegistry } from "../src/tools/registry.ts";

const SECRET = "s3cret";

const REGISTRY = `
[gpu_service]
host = "127.0.0.1"
port = 6666

[models.flux]
backend = "diffusers"
role = "image"
vram_mb = 12000
`;

const STATUS: Status = {
  vram: { total_mb: 24000, free_mb: 24000, usable_free_mb: 23000 },
  resident: [],
  busy: false,
  queue_depth: 0,
};

interface Recorded {
  goals: Array<{ goal: string; budgetUsd: number }>;
  decisions: Array<{ id: string; d: string }>;
  agains: string[];
  loads: string[];
  events: HubEvent[];
}

interface Hub {
  dir: string;
  ledger: Ledger;
  port: number;
  artifactsDir: string;
  recorded: Recorded;
  emit(e: HubEvent): void;
  close(): Promise<void>;
}

async function startHub(
  opts: {
    gpuDown?: boolean;
    /** The GPU service never answers health or status: a sleeping GPU machine. */
    gpuHangs?: boolean;
    /** A load or unload the GPU service refuses. */
    gpuRefuses?: boolean;
    runner?: HubDeps["runner"];
  } = {},
): Promise<Hub> {
  const dir = mkdtempSync(join(tmpdir(), "ohh-hub-server-"));
  const ledger = openLedger(join(dir, "ledger.db"));
  const staticDir = join(dir, "static");
  const artifactsDir = join(dir, "artifacts");
  mkdirSync(staticDir);
  mkdirSync(artifactsDir);
  writeFileSync(join(staticDir, "index.html"), "<p>secret={{HUB_SECRET}}</p>");
  const registryPath = join(dir, "models.toml");
  writeFileSync(registryPath, REGISTRY);

  const recorded: Recorded = { goals: [], decisions: [], agains: [], loads: [], events: [] };
  const down = () => Promise.reject(new GpuError("GPU service unreachable"));
  const never = () => new Promise<never>(() => {});
  const refused = () => Promise.reject(new GpuError("not enough VRAM\nstack"));
  const gpu: HubDeps["gpu"] = {
    health: opts.gpuHangs
      ? never
      : opts.gpuDown
        ? down
        : async () => ({ status: "ok", gpu: "x", vram_total_mb: 1, vram_free_mb: 1, version: "1" }),
    status: opts.gpuHangs ? never : opts.gpuDown ? down : async () => STATUS,
    load: async (model) => {
      recorded.loads.push(model);
      if (opts.gpuRefuses) return refused();
      return { model, vram_mb: 1, load_ms: 1, evicted: [{ model: "qwen", freed_mb: 5000, reason: "lru" }] };
    },
    unload: async () => (opts.gpuRefuses ? refused() : { freed_mb: 1 }),
  };

  const runner: HubDeps["runner"] = opts.runner ?? {
    startGoal: async (req) => {
      recorded.goals.push(req);
    },
    // Records the decision as the real adapter does, synchronously.
    decide: (id, d) => {
      ledger.decideApproval(id, d);
      recorded.decisions.push({ id, d });
      return Promise.resolve();
    },
    again: async (rootId) => {
      recorded.agains.push(rootId);
    },
  };

  const listeners = new Set<(e: HubEvent) => void>();
  const emit = (e: HubEvent): void => {
    recorded.events.push(e);
    listeners.forEach((l) => l(e));
  };
  const server: Server = createHubServer({
    ledger,
    gpu,
    runner,
    registryPath,
    staticDir,
    artifactsDir,
    secret: SECRET,
    runningDir: join(dir, "running"),
    gpuTimeoutMs: 100,
    emit,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    dir,
    ledger,
    port,
    artifactsDir,
    recorded,
    emit,
    close: () =>
      new Promise<void>((done) => {
        server.closeAllConnections();
        server.close(() => {
          ledger.close();
          rmSync(dir, { recursive: true, force: true });
          done();
        });
      }),
  };
}

interface Reply {
  status: number;
  type: string | undefined;
  headers: IncomingHttpHeaders;
  body: Buffer;
  json(): any;
}

function call(
  hub: Hub,
  method: string,
  path: string,
  init: { headers?: Record<string, string>; body?: unknown } = {},
): Promise<Reply> {
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      { host: "127.0.0.1", port: hub.port, method, path, headers: init.headers },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () => {
          const body = Buffer.concat(chunks);
          resolve({
            status: res.statusCode ?? 0,
            type: res.headers["content-type"],
            headers: res.headers,
            body,
            json: () => JSON.parse(body.toString("utf8")),
          });
        });
      },
    );
    req.on("error", reject);
    if (init.body !== undefined) req.write(JSON.stringify(init.body));
    req.end();
  });
}

/** A POST the Hub's own page would make. */
function post(hub: Hub, path: string, body: unknown = {}, headers: Record<string, string> = {}) {
  return call(hub, "POST", path, {
    headers: {
      "content-type": "application/json",
      "x-hub-secret": SECRET,
      origin: `http://127.0.0.1:${hub.port}`,
      ...headers,
    },
    body,
  });
}

const root = (l: Ledger, goal = "make a logo", budgetUsd = 1.5) =>
  l.createRun({ agent: "lead", goal, parentId: null, budgetUsd });

test("the page carries the secret", async () => {
  const hub = await startHub();
  try {
    const page = await call(hub, "GET", "/");
    assert.equal(page.status, 200);
    assert.match(page.type ?? "", /text\/html/);
    assert.equal(page.body.toString(), `<p>secret=${SECRET}</p>`);
  } finally {
    await hub.close();
  }
});

test("the pages refuse to be framed", async () => {
  const hub = await startHub();
  try {
    writeFileSync(join(hub.dir, "static", "models.html"), "<p>models</p>");
    for (const path of ["/", "/models"]) {
      const page = await call(hub, "GET", path);
      assert.equal(page.status, 200, path);
      assert.equal(page.headers["x-frame-options"], "DENY", path);
      assert.equal(page.headers["content-security-policy"], "frame-ancestors 'none'", path);
    }
  } finally {
    await hub.close();
  }
});

test("a request naming a foreign Host is 403", async () => {
  const hub = await startHub();
  try {
    const page = await call(hub, "GET", "/", { headers: { host: `evil.example:${hub.port}` } });
    assert.equal(page.status, 403);
    assert.doesNotMatch(page.body.toString(), new RegExp(SECRET));
  } finally {
    await hub.close();
  }
});

test("a POST without the secret is 403 and starts nothing", async () => {
  const hub = await startHub();
  try {
    const wrong = await post(hub, "/api/goals", { goal: "x" }, { "x-hub-secret": "nope" });
    assert.equal(wrong.status, 403);
    const none = await call(hub, "POST", "/api/goals", {
      headers: { "content-type": "application/json", origin: `http://127.0.0.1:${hub.port}` },
      body: { goal: "x" },
    });
    assert.equal(none.status, 403);
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("a POST from a foreign Origin is 403", async () => {
  const hub = await startHub();
  try {
    const reply = await post(hub, "/api/goals", { goal: "x" }, { origin: "http://evil.example" });
    assert.equal(reply.status, 403);
    const otherPort = await post(hub, "/api/goals", { goal: "x" }, { origin: `http://127.0.0.1:${hub.port + 1}` });
    assert.equal(otherPort.status, 403);
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("a POST with no Origin is 403", async () => {
  const hub = await startHub();
  try {
    const reply = await call(hub, "POST", "/api/goals", {
      headers: { "content-type": "application/json", "x-hub-secret": SECRET },
      body: { goal: "x" },
    });
    assert.equal(reply.status, 403);
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("localhost is the Hub's own origin too", async () => {
  const hub = await startHub();
  try {
    const reply = await post(hub, "/api/goals", { goal: "x" }, { origin: `http://localhost:${hub.port}` });
    assert.equal(reply.status, 202);
  } finally {
    await hub.close();
  }
});

test("a POST that is not JSON is refused", async () => {
  const hub = await startHub();
  try {
    const form = await post(hub, "/api/goals", { goal: "x" }, { "content-type": "text/plain" });
    assert.equal(form.status, 415);
    const broken = await call(hub, "POST", "/api/goals", {
      headers: {
        "content-type": "application/json",
        "x-hub-secret": SECRET,
        origin: `http://127.0.0.1:${hub.port}`,
      },
    });
    // An empty body is not a goal.
    assert.equal(broken.status, 400);
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("a body over 16 KiB is 413", async () => {
  const hub = await startHub();
  try {
    const reply = await post(hub, "/api/goals", { goal: "x", pad: "y".repeat(17 * 1024) });
    assert.equal(reply.status, 413);
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("a goal with budget 150 is 400", async () => {
  const hub = await startHub();
  try {
    for (const budget of [150, 0, -1, "1.5", null]) {
      const reply = await post(hub, "/api/goals", { goal: "x", budget_usd: budget });
      assert.equal(reply.status, 400, `budget ${JSON.stringify(budget)}`);
      assert.equal(typeof reply.json().error, "string");
    }
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("an empty goal is 400", async () => {
  const hub = await startHub();
  try {
    for (const goal of ["", "   ", 7, "g".repeat(2001)]) {
      const reply = await post(hub, "/api/goals", { goal });
      assert.equal(reply.status, 400);
    }
    assert.deepEqual(hub.recorded.goals, []);
  } finally {
    await hub.close();
  }
});

test("a valid goal is 202 and reaches the runner with budget 1.5", async () => {
  const hub = await startHub();
  try {
    const reply = await post(hub, "/api/goals", { goal: "  make a logo  " });
    assert.equal(reply.status, 202);
    assert.deepEqual(reply.json(), { accepted: true });
    const explicit = await post(hub, "/api/goals", { goal: "two", budget_usd: 10 });
    assert.equal(explicit.status, 202);
    assert.deepEqual(hub.recorded.goals, [
      { goal: "make a logo", budgetUsd: 1.5 },
      { goal: "two", budgetUsd: 10 },
    ]);
  } finally {
    await hub.close();
  }
});

test("a second decision on the same approval is 409", async () => {
  const hub = await startHub();
  try {
    const runId = root(hub.ledger);
    const id = hub.ledger.requestApproval({ runId, tool: "publish", args: {} });
    const first = await post(hub, `/api/approvals/${id}`, { decision: "approved" });
    assert.equal(first.status, 202);
    const second = await post(hub, `/api/approvals/${id}`, { decision: "denied" });
    assert.equal(second.status, 409);
    assert.deepEqual(hub.recorded.decisions, [{ id, d: "approved" }]);
  } finally {
    await hub.close();
  }
});

test("a decision must be approved or denied", async () => {
  const hub = await startHub();
  try {
    const runId = root(hub.ledger);
    const id = hub.ledger.requestApproval({ runId, tool: "publish", args: {} });
    const reply = await post(hub, `/api/approvals/${id}`, { decision: "maybe" });
    assert.equal(reply.status, 400);
    assert.deepEqual(hub.recorded.decisions, []);
  } finally {
    await hub.close();
  }
});

test("an unknown approval is 404", async () => {
  const hub = await startHub();
  try {
    const reply = await post(hub, "/api/approvals/nope", { decision: "approved" });
    assert.equal(reply.status, 404);
    assert.deepEqual(hub.recorded.decisions, []);
  } finally {
    await hub.close();
  }
});

test("artifacts are served by id only", async () => {
  const hub = await startHub();
  try {
    const runId = root(hub.ledger);
    const file = join(hub.artifactsDir, "a.png");
    writeFileSync(file, Buffer.from([0x89, 0x50, 0x4e, 0x47]));
    const id = hub.ledger.recordArtifact({ runId, kind: "image", path: file, meta: {} });
    // A row pointing outside the artifacts directory -- a published copy.
    const outside = join(hub.dir, "published.png");
    writeFileSync(outside, "x");
    const outsideId = hub.ledger.recordArtifact({ runId, kind: "image", path: outside, meta: {} });

    const known = await call(hub, "GET", `/artifacts/${id}`);
    assert.equal(known.status, 200);
    assert.equal(known.type, "image/png");
    assert.deepEqual([...known.body], [0x89, 0x50, 0x4e, 0x47]);

    assert.equal((await call(hub, "GET", "/artifacts/unknown")).status, 404);
    assert.equal((await call(hub, "GET", "/artifacts/..%2F..%2Fledger.db")).status, 404);
    assert.equal((await call(hub, "GET", `/artifacts/${outsideId}`)).status, 404);
  } finally {
    await hub.close();
  }
});

test("an artifact is never served as active content", async () => {
  const hub = await startHub();
  try {
    const runId = root(hub.ledger);
    const png = join(hub.artifactsDir, "a.png");
    writeFileSync(png, "png");
    const pngId = hub.ledger.recordArtifact({ runId, kind: "image", path: png, meta: {} });
    const html = join(hub.artifactsDir, "x.html");
    writeFileSync(html, "<script>alert(1)</script>");
    const htmlId = hub.ledger.recordArtifact({ runId, kind: "file", path: html, meta: {} });

    const image = await call(hub, "GET", `/artifacts/${pngId}`);
    assert.equal(image.type, "image/png");
    assert.equal(image.headers["x-content-type-options"], "nosniff");
    assert.equal(image.headers["content-security-policy"], "sandbox; default-src 'none'");

    const page = await call(hub, "GET", `/artifacts/${htmlId}`);
    assert.equal(page.status, 200);
    assert.equal(page.type, "application/octet-stream");
    assert.equal(page.headers["content-disposition"], "attachment");
    assert.equal(page.headers["x-content-type-options"], "nosniff");
    assert.equal(page.headers["content-security-policy"], "sandbox; default-src 'none'");
  } finally {
    await hub.close();
  }
});

test("malformed percent-encoding is 404, not a fault", async () => {
  const hub = await startHub();
  try {
    assert.equal((await call(hub, "GET", "/artifacts/%E0")).status, 404);
    assert.equal((await call(hub, "GET", "/api/goals/%E0")).status, 404);
    assert.equal((await call(hub, "GET", "/%E0")).status, 404);
    assert.equal((await post(hub, "/api/approvals/%E0", { decision: "approved" })).status, 404);
    assert.equal((await post(hub, "/api/models/%E0/load")).status, 404);
  } finally {
    await hub.close();
  }
});

test("the card says unreachable when the GPU service is down", async () => {
  const hub = await startHub({ gpuDown: true });
  try {
    const reply = await call(hub, "GET", "/api/card");
    assert.equal(reply.status, 200);
    assert.deepEqual(reply.json(), { reachable: false, error: "GPU service unreachable" });
  } finally {
    await hub.close();
  }
});

test("the card says unreachable when the GPU service never answers", async () => {
  const hub = await startHub({ gpuHangs: true });
  try {
    const card = (await call(hub, "GET", "/api/card")).json();
    assert.equal(card.reachable, false);
    assert.match(card.error, /did not answer/);
    // The model list does not wait on it either, and knows nothing loaded.
    const models = (await call(hub, "GET", "/api/models")).json();
    assert.equal(models[0].loaded, null);
  } finally {
    await hub.close();
  }
});

test("the model list says loaded or not when the GPU service answers", async () => {
  const hub = await startHub();
  try {
    const models = (await call(hub, "GET", "/api/models")).json();
    assert.equal(models[0].loaded, false);
  } finally {
    await hub.close();
  }
});

test("the card carries health and status when the GPU service is up", async () => {
  const hub = await startHub();
  try {
    const card = (await call(hub, "GET", "/api/card")).json();
    assert.equal(card.reachable, true);
    assert.equal(card.health.status, "ok");
    assert.deepEqual(card.status, STATUS);
  } finally {
    await hub.close();
  }
});

test("the board still renders when the GPU service is down", async () => {
  const hub = await startHub({ gpuDown: true });
  try {
    const id = root(hub.ledger);
    hub.ledger.setRunStatus(id, "blocked");
    const reply = await call(hub, "GET", "/api/board");
    assert.equal(reply.status, 200);
    const board = reply.json();
    assert.deepEqual(Object.keys(board).sort(), ["done", "needs_you", "working"]);
    assert.equal(board.done[0].id, id);

    const models = (await call(hub, "GET", "/api/models")).json();
    assert.equal(models[0].name, "flux");
    // Not "not loaded": with no reading, nobody knows.
    assert.equal(models[0].loaded, null);
  } finally {
    await hub.close();
  }
});

test("the board sweeps a running root nobody owns", async () => {
  const hub = await startHub();
  try {
    const id = root(hub.ledger);
    const board = (await call(hub, "GET", "/api/board")).json();
    assert.equal(board.done[0].id, id);
    assert.equal(board.done[0].reason, "interrupted");
  } finally {
    await hub.close();
  }
});

test("a goal's story is served, and an unknown one is 404", async () => {
  const hub = await startHub();
  try {
    const id = root(hub.ledger);
    const story = await call(hub, "GET", `/api/goals/${id}`);
    assert.equal(story.status, 200);
    assert.equal(story.json().card.id, id);
    assert.equal((await call(hub, "GET", "/api/goals/nope")).status, 404);
  } finally {
    await hub.close();
  }
});

test("events reach an SSE subscriber", async () => {
  const hub = await startHub();
  try {
    const frame = await new Promise<{ type: string | undefined; text: string }>((resolve, reject) => {
      const req = httpRequest(
        { host: "127.0.0.1", port: hub.port, path: "/api/events" },
        (res) => {
          let text = "";
          res.setEncoding("utf8");
          res.on("data", (chunk: string) => {
            text += chunk;
            if (text.includes("\n\n")) {
              req.destroy();
              resolve({ type: res.headers["content-type"], text });
            }
          });
          // Headers are out, so the server has subscribed.
          hub.emit({ source: "gpu", event: "model.loaded", model: "flux" });
        },
      );
      req.on("error", reject);
      req.end();
    });
    assert.match(frame.type ?? "", /text\/event-stream/);
    assert.equal(
      frame.text,
      `event: model.loaded\ndata: ${JSON.stringify({ source: "gpu", event: "model.loaded", model: "flux" })}\n\n`,
    );
  } finally {
    await hub.close();
  }
});

test("again is 409 for a done goal and 202 for a failed one", async () => {
  const hub = await startHub();
  try {
    const done = root(hub.ledger);
    hub.ledger.setRunStatus(done, "done");
    const failed = root(hub.ledger, "try again", 2);
    hub.ledger.setRunStatus(failed, "failed");

    assert.equal((await post(hub, `/api/goals/${done}/again`)).status, 409);
    assert.equal((await post(hub, "/api/goals/nope/again")).status, 404);
    const reply = await post(hub, `/api/goals/${failed}/again`);
    assert.equal(reply.status, 202);
    assert.deepEqual(hub.recorded.agains, [failed]);
  } finally {
    await hub.close();
  }
});

test("again starts over a goal stopped blocked with nothing to decide", async () => {
  const hub = await startHub();
  try {
    const stopped = root(hub.ledger, "spent", 0.1);
    hub.ledger.setRunStatus(stopped, "blocked");
    const reply = await post(hub, `/api/goals/${stopped}/again`);
    assert.equal(reply.status, 202);
    assert.deepEqual(hub.recorded.agains, [stopped]);
  } finally {
    await hub.close();
  }
});

test("again is 409 for a goal waiting on an approval in its tree", async () => {
  const hub = await startHub();
  try {
    const waiting = root(hub.ledger);
    const kid = hub.ledger.createRun({ agent: "site", goal: "sub", parentId: waiting, budgetUsd: 0.5 });
    hub.ledger.requestApproval({ runId: kid, tool: "publish", args: {} });
    hub.ledger.setRunStatus(kid, "blocked");
    hub.ledger.setRunStatus(waiting, "blocked");
    const reply = await post(hub, `/api/goals/${waiting}/again`);
    assert.equal(reply.status, 409);
    assert.match(reply.json().error, /approval/);
    assert.deepEqual(hub.recorded.agains, []);
  } finally {
    await hub.close();
  }
});

/** The first hub event named `name`, once the background work settles. */
async function outcome(hub: Hub, name: string): Promise<HubEvent> {
  for (let i = 0; i < 100; i++) {
    const found = hub.recorded.events.find((e) => e.event === name);
    if (found) return found;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`no ${name} event`);
}

test("a load's outcome reaches the page, with what it unloaded", async () => {
  const hub = await startHub();
  try {
    assert.equal((await post(hub, "/api/models/flux/load")).status, 202);
    assert.deepEqual(await outcome(hub, "model.outcome"), {
      source: "hub",
      event: "model.outcome",
      model: "flux",
      action: "load",
      ok: true,
      evicted: [{ model: "qwen", freed_mb: 5000, reason: "lru" }],
    });
  } finally {
    await hub.close();
  }
});

test("a refused load or unload reaches the page as an error", async () => {
  const hub = await startHub({ gpuRefuses: true });
  try {
    assert.equal((await post(hub, "/api/models/flux/load")).status, 202);
    assert.deepEqual(await outcome(hub, "model.outcome"), {
      source: "hub",
      event: "model.outcome",
      model: "flux",
      action: "load",
      ok: false,
      error: "not enough VRAM",
    });
    hub.recorded.events.length = 0;
    assert.equal((await post(hub, "/api/models/flux/unload")).status, 202);
    const unload = await outcome(hub, "model.outcome");
    assert.equal(unload.action, "unload");
    assert.equal(unload.ok, false);
  } finally {
    await hub.close();
  }
});

test("loading a model the registry does not know is 404", async () => {
  const hub = await startHub();
  try {
    assert.equal((await post(hub, "/api/models/nope/load")).status, 404);
    assert.equal((await post(hub, "/api/models/nope/unload")).status, 404);
    assert.deepEqual(hub.recorded.loads, []);
    assert.equal((await post(hub, "/api/models/flux/load")).status, 202);
    assert.deepEqual(hub.recorded.loads, ["flux"]);
  } finally {
    await hub.close();
  }
});

test("only named page assets are served", async () => {
  const hub = await startHub();
  try {
    writeFileSync(join(hub.dir, "static", "board.js"), "x()");
    const js = await call(hub, "GET", "/board.js");
    assert.equal(js.status, 200);
    assert.match(js.type ?? "", /javascript/);
    assert.equal((await call(hub, "GET", "/missing.js")).status, 404);
    assert.equal((await call(hub, "GET", "/..%2Fmodels.toml")).status, 404);
    // No models.html in this staticDir.
    assert.equal((await call(hub, "GET", "/models")).status, 503);
  } finally {
    await hub.close();
  }
});

test("a refusal from the runner is a 409 given before any work starts", async () => {
  const hub = await startHub({
    runner: {
      startGoal: async () => undefined,
      decide: () => {
        throw new RunRefused("already running", "");
      },
      again: () => {
        throw new RunRefused("already running", "");
      },
    },
  });
  try {
    const runId = root(hub.ledger);
    const id = hub.ledger.requestApproval({ runId, tool: "publish", args: {} });
    const decided = await post(hub, `/api/approvals/${id}`, { decision: "approved" });
    assert.equal(decided.status, 409);
    assert.deepEqual(decided.json(), { error: "already running" });

    const failed = root(hub.ledger);
    hub.ledger.setRunStatus(failed, "failed");
    const again = await post(hub, `/api/goals/${failed}/again`);
    assert.equal(again.status, 409);
    assert.deepEqual(again.json(), { error: "already running" });
  } finally {
    await hub.close();
  }
});

// ---- the runner adapter: one drive per tree inside the Hub process ----

/** A provider whose first turn waits until `release` is called. */
function held(): { provider: Provider; release: () => void } {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  const provider: Provider = {
    name: "scripted",
    model: "claude-opus-5-5",
    async *chat(): AsyncIterable<TurnEvent> {
      await gate;
      yield { kind: "text", text: "done" };
      yield { kind: "end", reason: "end_turn" };
    },
  };
  return { provider, release };
}

function adapterSetup() {
  const dir = mkdtempSync(join(tmpdir(), "ohh-hub-drive-"));
  const ledger = openLedger(join(dir, "ledger.db"));
  const runningDir = join(dir, "running");
  const gate = held();
  const config = {
    runningDir,
    budgetUsd: 2,
    maxTurns: 6,
  } as Config;
  const opts: RunnerOptions = {
    config,
    gpu: {} as GpuClient,
    ledger,
    harness: "HARNESS",
    output: () => ({}),
    factory: () => ({
      agent: (name) => ({
        agent: { name, role: name, tools: [], limits: {} },
        system: name,
        placement: "local",
        deps: { provider: gate.provider, registry: createRegistry([]), ledger, maxTurns: 6 },
      }),
    }),
  };
  let settled = 0;
  const runner = createHubRunner(opts, () => settled++);
  return {
    ledger,
    runner,
    release: gate.release,
    settled: () => settled,
    cleanup: () => {
      ledger.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

test("again on a root already being driven is refused synchronously", async () => {
  const t = adapterSetup();
  try {
    const failed = root(t.ledger, "retry me", 0.75);
    t.ledger.setRunStatus(failed, "failed");

    const first = t.runner.again(failed);
    assert.throws(() => t.runner.again(failed), /already running/);
    t.release();
    await first;

    // The new goal carries the old one's text and budget.
    const fresh = t.ledger.listRootRuns().find((p) => p.id !== failed)!;
    assert.equal(fresh.goal, "retry me");
    assert.equal(fresh.budget_usd, 0.75);
    assert.equal(fresh.status, "done");
    assert.ok(t.settled() >= 1);

    // Free again once it settles.
    const second = t.runner.again(failed);
    await second;
  } finally {
    t.cleanup();
  }
});

test("again keeps the old goal's agent", async () => {
  const t = adapterSetup();
  try {
    const failed = t.ledger.createRun({ agent: "site", goal: "a page", parentId: null, budgetUsd: 1 });
    t.ledger.setRunStatus(failed, "failed");
    const work = t.runner.again(failed);
    t.release();
    await work;
    const fresh = t.ledger.listRootRuns().find((p) => p.id !== failed)!;
    assert.equal(fresh.agent, "site");
  } finally {
    t.cleanup();
  }
});

test("again refuses a budget over the Hub's limit, and a goal still waiting", () => {
  const t = adapterSetup();
  try {
    const big = root(t.ledger, "costly", 12);
    t.ledger.setRunStatus(big, "failed");
    assert.throws(() => t.runner.again(big), (e: Error) => {
      assert.ok(e instanceof RunRefused);
      assert.equal(
        `${e.head} ${e.detail}`.trim(),
        "budget $12.00 is over the Hub's $10 limit; start it again from the goal box",
      );
      return true;
    });

    const waiting = root(t.ledger);
    t.ledger.requestApproval({ runId: waiting, tool: "publish", args: {} });
    t.ledger.setRunStatus(waiting, "blocked");
    assert.throws(() => t.runner.again(waiting), /approval/);
    assert.equal(t.ledger.listRootRuns().length, 2);
  } finally {
    t.cleanup();
  }
});

test("a decision on a tree the Hub is driving is refused and left pending", async () => {
  const t = adapterSetup();
  try {
    let rootId: string | undefined;
    const running = t.runner.startGoal({ goal: "draw", budgetUsd: 1.5 });
    // The root exists once the first turn is under way.
    while ((rootId = t.ledger.listRootRuns()[0]?.id) === undefined) {
      await new Promise((r) => setTimeout(r, 5));
    }
    const approval = t.ledger.requestApproval({ runId: rootId, tool: "publish", args: {} });

    assert.throws(() => t.runner.decide(approval, "approved"), /already running/);
    assert.equal(t.ledger.readApproval(approval)?.status, "pending");

    t.release();
    await running;
  } finally {
    t.cleanup();
  }
});

test("an unknown approval is refused synchronously by the adapter", () => {
  const t = adapterSetup();
  try {
    assert.throws(() => t.runner.decide("nope", "approved"), (e: Error) => {
      assert.ok(e instanceof RunRefused);
      assert.equal(e.head, "unknown approval");
      return true;
    });
  } finally {
    t.cleanup();
  }
});
