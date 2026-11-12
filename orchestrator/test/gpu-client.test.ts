import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";

import { createGpuClient, GpuError } from "../src/gpu/client.ts";

const PNG = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  Buffer.from("IHDRxxxxxxxxxxxxIEND"),
]);

interface StubOptions {
  /** Emit progress but never the terminal event — the dropped-event case. */
  dropTerminal?: boolean;
  /** Finish the op as failed rather than done. */
  failOp?: boolean;
  /** Reject every request with 401. */
  unauthorized?: boolean;
}

interface Stub {
  url: string;
  close(): Promise<void>;
  /** How many times /ops/{id} was polled — proves which path terminated it. */
  polls: number;
  subscribers: number;
}

function startStubGpu(options: StubOptions = {}): Promise<Stub> {
  const streams = new Set<import("node:http").ServerResponse>();
  const ops = new Map<string, { status: string; result?: unknown }>();
  const stub = { polls: 0, subscribers: 0 } as Stub;

  function push(event: string, data: unknown): void {
    const frame = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    for (const stream of streams) stream.write(frame);
  }

  const server: Server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://stub");

    if (options.unauthorized) {
      res.writeHead(401, { "content-type": "application/json" });
      res.end(JSON.stringify({ code: "UNAUTHORIZED", error: "bad token" }));
      return;
    }

    if (url.pathname === "/health") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          status: "ok",
          gpu: "NVIDIA GeForce RTX 3060",
          vram_total_mb: 12288,
          vram_free_mb: 10400,
          version: "0.1.0",
        }),
      );
      return;
    }

    if (url.pathname === "/status") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          vram: { total_mb: 12288, free_mb: 10400, usable_free_mb: 9632 },
          resident: [],
          busy: false,
          queue_depth: 0,
        }),
      );
      return;
    }

    if (url.pathname === "/events") {
      stub.subscribers++;
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      // writeHead only stages headers; Node sends them with the first write.
      // fetch() waits on headers, so without this the subscription never opens.
      // FastAPI's StreamingResponse flushes immediately, so the real GPU service is fine.
      res.flushHeaders();
      streams.add(res);
      req.on("close", () => streams.delete(res));
      return;
    }

    if (url.pathname === "/generate/image" && req.method === "POST") {
      const opId = `op-${ops.size + 1}`;
      ops.set(opId, { status: "running" });
      res.writeHead(202, { "content-type": "application/json" });
      res.end(JSON.stringify({ op_id: opId }));

      setTimeout(() => {
        push("op.progress", { op_id: opId, kind: "image", step: 1, total: 4 });
        push("op.progress", { op_id: opId, kind: "image", step: 4, total: 4 });

        if (options.failOp) {
          const payload = {
            code: "VRAM_EXHAUSTED",
            error: "Cannot free 7515MB",
            remedy: "Unload a resident model, then retry.",
          };
          ops.set(opId, { status: "failed", result: payload });
          push("op.failed", { op_id: opId, ...payload });
          return;
        }

        const result = {
          model: "sdxl-turbo",
          steps: 4,
          artifact_id: "abc",
          artifact_url: "/artifacts/abc.png",
          evicted: [
            { model: "qwen2.5-coder:7b", freed_mb: 4870, reason: "evicted" },
          ],
        };
        ops.set(opId, { status: "done", result });
        // The whole reason the client reconciles: this event may never arrive.
        if (!options.dropTerminal) push("op.done", { op_id: opId, ...result });
      }, 10);
      return;
    }

    if (url.pathname.startsWith("/ops/")) {
      stub.polls++;
      const op = ops.get(url.pathname.slice("/ops/".length));
      res.writeHead(op ? 200 : 404, { "content-type": "application/json" });
      res.end(JSON.stringify(op ?? { code: "UNKNOWN_OP" }));
      return;
    }

    if (url.pathname === "/artifacts/abc.png") {
      res.writeHead(200, { "content-type": "image/png" });
      res.end(PNG);
      return;
    }

    res.writeHead(404);
    res.end();
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      stub.url = `http://127.0.0.1:${port}`;
      stub.close = () =>
        new Promise((done) => {
          for (const stream of streams) stream.end();
          // server.close() waits for every connection, and fetch keeps its
          // sockets alive, so without this the process never exits.
          server.closeAllConnections();
          server.close(() => done());
        });
      resolve(stub);
    });
  });
}

function client(stub: Stub, token = "test-token") {
  return createGpuClient({
    baseUrl: stub.url,
    token,
    pollIntervalMs: 40,
    settleMs: 20,
  });
}

test("health reports the card", async () => {
  const stub = await startStubGpu();
  const health = await client(stub).health();
  assert.equal(health.status, "ok");
  assert.match(health.gpu, /NVIDIA/);
  await stub.close();
});

test("generateImage returns the result and surfaces progress", async () => {
  const stub = await startStubGpu();
  const progress: Array<{ step: number; total: number }> = [];
  const gpu = createGpuClient({
    baseUrl: stub.url,
    token: "t",
    pollIntervalMs: 40,
    settleMs: 20,
    onProgress: (event) =>
      progress.push({ step: event.step ?? 0, total: event.total ?? 0 }),
  });

  const result = await gpu.generateImage({ prompt: "a brass agent", steps: 4 });

  assert.equal(result.model, "sdxl-turbo");
  assert.equal(result.artifact_url, "/artifacts/abc.png");
  assert.equal(result.evicted[0]?.model, "qwen2.5-coder:7b");
  assert.equal(result.evicted[0]?.freed_mb, 4870);
  assert.ok(progress.length >= 1, "no progress events surfaced");
  await stub.close();
});

test("a dropped terminal event is recovered by the reconciliation poll", async () => {
  // The GPU service's own tests assert that a full queue drops the oldest event, so
  // op.done can simply never arrive. Without the backstop the client waits
  // forever on a stream that will never mention its op.
  const stub = await startStubGpu({ dropTerminal: true });
  const result = await client(stub).generateImage({ prompt: "x" });

  assert.equal(result.artifact_url, "/artifacts/abc.png");
  assert.ok(stub.polls > 0, "the poll never ran, so it cannot have terminated");
  await stub.close();
});

test("op.failed surfaces the GPU service's error envelope", async () => {
  const stub = await startStubGpu({ failOp: true });
  await assert.rejects(
    () => client(stub).generateImage({ prompt: "x" }),
    (error: unknown) => {
      assert.ok(error instanceof GpuError);
      assert.equal(error.code, "VRAM_EXHAUSTED");
      assert.match(error.message, /Cannot free/);
      assert.match(error.remedy ?? "", /Unload/);
      return true;
    },
  );
  await stub.close();
});

test("a 401 is a hard error naming the token variable, not a retry", async () => {
  const stub = await startStubGpu({ unauthorized: true });
  await assert.rejects(
    () => client(stub, "wrong").health(),
    (error: unknown) => {
      assert.ok(error instanceof GpuError);
      assert.equal(error.status, 401);
      assert.match(error.message, /OHH_GPU_TOKEN/);
      return true;
    },
  );
  await stub.close();
});

test("fetchArtifact returns the bytes and touches no disk", async () => {
  // GpuClient owns the wire and nothing else, which is what lets it be
  // tested against a stub with no filesystem and no database in play.
  const stub = await startStubGpu();
  const bytes = await client(stub).fetchArtifact("/artifacts/abc.png");
  assert.ok(bytes.length > 8);
  assert.deepEqual(Array.from(bytes.slice(0, 8)), [
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  ]);
  await stub.close();
});

test("the event subscription opens before the command is posted", async () => {
  // Subscribe-then-POST is the ordering the race depends on.
  const stub = await startStubGpu();
  await client(stub).generateImage({ prompt: "x" });
  assert.ok(stub.subscribers >= 1, "never subscribed to /events");
  await stub.close();
});

test("an unreachable GPU service is a named error, not a raw socket dump", async () => {
  // The most common failure in practice: the GPU service simply is not running. A
  // bare ECONNREFUSED object tells the operator nothing about what to start.
  const gpu = createGpuClient({ baseUrl: "http://127.0.0.1:1", token: "t" });
  await assert.rejects(
    () => gpu.health(),
    (error: unknown) => {
      assert.ok(error instanceof GpuError, "not a GpuError");
      assert.match(error.message, /127\.0\.0\.1:1/, "does not name the target");
      assert.ok(error.remedy, "no remedy offered");
      return true;
    },
  );
});

test("an unreachable GPU service fails the same way mid-op", async () => {
  // runOp opens /events first, so the failure surfaces from a different call
  // path than health() and has to be wrapped there too.
  const gpu = createGpuClient({ baseUrl: "http://127.0.0.1:1", token: "t" });
  await assert.rejects(
    () => gpu.generateImage({ prompt: "x" }),
    (error: unknown) => {
      assert.ok(error instanceof GpuError, "not a GpuError");
      assert.match(error.message, /127\.0\.0\.1:1/);
      return true;
    },
  );
});
