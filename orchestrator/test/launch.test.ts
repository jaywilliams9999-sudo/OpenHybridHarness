import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { launch, LaunchError, type SpawnFn } from "../src/launch.ts";

function makeRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "ohh-launch-"));
  mkdirSync(join(root, "gpu-service"));
  writeFileSync(join(root, ".env"), "OHH_GPU_TOKEN=tok\n");
  writeFileSync(join(root, "models.toml"), '[gpu_service]\nhost = "127.0.0.1"\nport = 7311\n');
  return root;
}

/** A clean base environment, so a developer's exported OHH_* variables never leak in. */
const CLEAN: NodeJS.ProcessEnv = { PATH: process.env["PATH"] ?? "" };

function harness(opts: { health?: number[]; ollamaDown?: boolean; exitCode?: number; spawnError?: Error; hubFails?: boolean } = {}) {
  const calls: string[] = [];
  const spawned: Array<{ command: string; args: string[]; cwd?: string | undefined; env?: NodeJS.ProcessEnv | undefined }> = [];
  const hubEnv: Array<NodeJS.ProcessEnv | undefined> = [];
  let healthN = 0;
  const health = opts.health ?? [200];
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    const u = String(url);
    if (u.endsWith("/api/version")) {
      calls.push("ollama");
      if (opts.ollamaDown) throw new TypeError("fetch failed");
      return new Response("{}", { status: 200 });
    }
    calls.push("/health");
    assert.deepEqual(init?.headers, { authorization: "Bearer tok" });
    const status = health[Math.min(healthN++, health.length - 1)]!;
    return new Response("{}", { status });
  }) as typeof fetch;
  const spawnFn: SpawnFn = (command, args, options) => {
    calls.push("spawn");
    spawned.push({ command, args, cwd: options.cwd, env: options.env });
    return {
      exitCode: opts.exitCode ?? null,
      once: (_event, listener) => {
        if (opts.spawnError) setTimeout(() => listener(opts.spawnError!), 0);
      },
      kill: () => {
        calls.push("kill");
        return true;
      },
    };
  };
  const startHub = async (env?: NodeJS.ProcessEnv) => {
    hubEnv.push(env);
    calls.push("startHub");
    if (opts.hubFails) throw new Error("port in use");
    return {
      close: async () => {
        calls.push("hubClose");
      },
    };
  };
  return { calls, fetchFn, spawnFn, startHub, spawned, hubEnv };
}

test("launch refuses with a plain message when Ollama is unreachable", async () => {
  const h = harness({ ollamaDown: true });
  await assert.rejects(
    launch({ repoRoot: makeRoot(), env: CLEAN, fetch: h.fetchFn, spawn: h.spawnFn, startHub: h.startHub, log: () => {} }),
    (e: unknown) => {
      assert.ok(e instanceof LaunchError);
      assert.match(e.message, /^Ollama isn't reachable at http:\/\/127\.0\.0\.1:11434/);
      assert.equal(
        e.message,
        "Ollama isn't reachable at http://127.0.0.1:11434. Install it from https://ollama.com and start it (`ollama serve`), then run this again.",
      );
      return true;
    },
  );
  assert.ok(!h.calls.includes("spawn"));
});

test("launch starts the GPU service, waits for health, then the hub", async () => {
  const h = harness({ health: [503, 503, 200] });
  const lines: string[] = [];
  const handle = await launch({
    repoRoot: makeRoot(),
    env: CLEAN,
    fetch: h.fetchFn,
    spawn: h.spawnFn,
    startHub: h.startHub,
    healthIntervalMs: 1,
    log: (l) => lines.push(l),
  });
  assert.deepEqual(h.calls, ["ollama", "spawn", "/health", "/health", "/health", "startHub"]);
  assert.deepEqual(h.spawned[0]!.args, ["-m", "ohh_gpu.main"]);
  assert.equal(h.spawned[0]!.env?.["OHH_GPU_TOKEN"], "tok");
  assert.match(lines.join("\n"), /OpenHybridHarness is running at http:\/\/127\.0\.0\.1:6680/);
  await handle.stop();
});

test("launch reports a GPU service that never becomes healthy", async () => {
  const h = harness({ health: [503] });
  await assert.rejects(
    launch({
      repoRoot: makeRoot(),
    env: CLEAN,
      fetch: h.fetchFn,
      spawn: h.spawnFn,
      startHub: h.startHub,
      healthTimeoutMs: 30,
      healthIntervalMs: 5,
      log: () => {},
    }),
    (e: unknown) => {
      assert.ok(e instanceof LaunchError);
      assert.equal(e.message, "The GPU service didn't become healthy within 60s. Its output is above.");
      return true;
    },
  );
  assert.ok(h.calls.includes("kill"));
  assert.ok(!h.calls.includes("startHub"));
});

test("stop closes the hub before killing the GPU service", async () => {
  const h = harness();
  const handle = await launch({
    repoRoot: makeRoot(),
    env: CLEAN,
    fetch: h.fetchFn,
    spawn: h.spawnFn,
    startHub: h.startHub,
    log: () => {},
  });
  await handle.stop();
  assert.deepEqual(h.calls.slice(-2), ["hubClose", "kill"]);
});

test("a hub that fails to start kills the GPU service and rethrows", async () => {
  const h = harness({ hubFails: true });
  await assert.rejects(
    launch({ repoRoot: makeRoot(), env: CLEAN, fetch: h.fetchFn, spawn: h.spawnFn, startHub: h.startHub, log: () => {} }),
    /port in use/,
  );
  assert.deepEqual(h.calls.slice(-2), ["startHub", "kill"]);
});

test("a GPU service that exits early is reported without waiting out the budget", async () => {
  const h = harness({ health: [503], exitCode: 1 });
  await assert.rejects(
    launch({ repoRoot: makeRoot(), env: CLEAN, fetch: h.fetchFn, spawn: h.spawnFn, startHub: h.startHub, log: () => {} }),
    (e: unknown) => {
      assert.ok(e instanceof LaunchError);
      assert.match(e.message, /^The GPU service exited with code 1 before it was healthy/);
      return true;
    },
  );
  assert.ok(h.calls.includes("kill"));
  assert.ok(!h.calls.includes("startHub"));
});

test("a Python that cannot be started becomes a plain LaunchError", async () => {
  const h = harness({ health: [503], spawnError: new Error("spawn python ENOENT") });
  await assert.rejects(
    launch({
      repoRoot: makeRoot(),
      env: CLEAN,
      fetch: h.fetchFn,
      spawn: h.spawnFn,
      startHub: h.startHub,
      healthIntervalMs: 5,
      log: () => {},
    }),
    (e: unknown) => {
      assert.ok(e instanceof LaunchError);
      assert.match(e.message, /^Couldn't start the GPU service with .+: spawn python ENOENT\. Run scripts\/install\.ps1 first\.$/);
      return true;
    },
  );
  assert.ok(!h.calls.includes("startHub"));
});

test("OHH_OLLAMA_URL is honored and reaches the GPU service", async () => {
  const urls: string[] = [];
  const h = harness();
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    urls.push(String(url));
    return h.fetchFn(url, init);
  }) as typeof fetch;
  const handle = await launch({
    repoRoot: makeRoot(),
    env: { ...CLEAN, OHH_OLLAMA_URL: "http://127.0.0.1:11999" },
    fetch: fetchFn,
    spawn: h.spawnFn,
    startHub: h.startHub,
    log: () => {},
  });
  assert.equal(urls[0], "http://127.0.0.1:11999/api/version");
  assert.equal(h.spawned[0]!.env?.["OHH_OLLAMA_URL"], "http://127.0.0.1:11999");
  await handle.stop();
});

test("the provider key is not passed to the GPU service, but the hub keeps it", async () => {
  const h = harness();
  const handle = await launch({
    repoRoot: makeRoot(),
    env: { ...CLEAN, OHH_ANTHROPIC_API_KEY: "sk-secret" },
    fetch: h.fetchFn,
    spawn: h.spawnFn,
    startHub: h.startHub,
    log: () => {},
  });
  assert.equal(h.spawned[0]!.env?.["OHH_ANTHROPIC_API_KEY"], undefined);
  assert.equal(h.hubEnv[0]?.["OHH_ANTHROPIC_API_KEY"], "sk-secret");
  await handle.stop();
});

test("a signal during startup kills the GPU service", async () => {
  const h = harness({ health: [503] });
  const exit = process.exit;
  let exited: number | undefined;
  (process as { exit: unknown }).exit = (code?: number) => {
    exited = code;
  };
  try {
    const p = launch({
      repoRoot: makeRoot(),
      env: CLEAN,
      fetch: h.fetchFn,
      spawn: h.spawnFn,
      startHub: h.startHub,
      handleSignals: true,
      healthTimeoutMs: 200,
      healthIntervalMs: 20,
      log: () => {},
    });
    p.catch(() => {});
    await new Promise((r) => setTimeout(r, 60));
    process.emit("SIGTERM");
    await new Promise((r) => setTimeout(r, 20));
    assert.ok(h.calls.includes("kill"));
    assert.equal(exited, 0);
    await p.catch(() => {});
    assert.equal(process.listenerCount("SIGTERM"), 0);
  } finally {
    (process as { exit: unknown }).exit = exit;
  }
});
