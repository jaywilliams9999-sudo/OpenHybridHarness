import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  loadConfig,
  ConfigError,
  defaultModelsPath,
  describeMode,
} from "../src/config/config.ts";

const REGISTRY = `
[gpu_service]
host             = "127.0.0.1"
port             = 7311
vram_headroom_mb = 768

[models."sdxl-turbo"]
backend = "diffusers"
vram_mb = 7515
`;

function withRegistry(body = REGISTRY): string {
  const dir = mkdtempSync(join(tmpdir(), "ohh-cfg-"));
  const path = join(dir, "models.toml");
  writeFileSync(path, body, "utf8");
  return path;
}

test("the gpu url is built from the registry, not duplicated in env", () => {
  const config = loadConfig({ OHH_GPU_TOKEN: "t" }, withRegistry());
  assert.equal(config.gpuUrl, "http://127.0.0.1:7311");
});

test("OHH_GPU_URL overrides the registry entirely", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_GPU_URL: "http://127.0.0.1:7777" },
    withRegistry(),
  );
  assert.equal(config.gpuUrl, "http://127.0.0.1:7777");
});

test("a trailing slash on the override is normalised away", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_GPU_URL: "http://127.0.0.1:7777/" },
    withRegistry(),
  );
  assert.equal(config.gpuUrl, "http://127.0.0.1:7777");
});

test("a missing token is fatal, and the message names the variable", () => {
  assert.throws(
    () => loadConfig({}, withRegistry()),
    (error: unknown) => {
      assert.ok(error instanceof ConfigError);
      assert.match(error.message, /OHH_GPU_TOKEN/);
      return true;
    },
  );
});

test("the override alone is enough — no registry required", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_GPU_URL: "http://127.0.0.1:7777" },
    join(tmpdir(), "definitely-not-here", "models.toml"),
  );
  assert.equal(config.gpuUrl, "http://127.0.0.1:7777");
});

test("a missing registry with no override is fatal and names the path", () => {
  const missing = join(tmpdir(), "definitely-not-here", "models.toml");
  assert.throws(
    () => loadConfig({ OHH_GPU_TOKEN: "t" }, missing),
    (error: unknown) => {
      assert.ok(error instanceof ConfigError);
      assert.match(error.message, /registry/);
      return true;
    },
  );
});

test("a registry missing [gpu_service] host is fatal rather than producing a half url", () => {
  const path = withRegistry(`[gpu_service]\nport = 7311\n`);
  assert.throws(() => loadConfig({ OHH_GPU_TOKEN: "t" }, path), ConfigError);
});

test("maxTurns defaults, and an env value overrides it", () => {
  assert.equal(
    loadConfig({ OHH_GPU_TOKEN: "t" }, withRegistry()).maxTurns,
    20,
  );
  assert.equal(
    loadConfig(
      { OHH_GPU_TOKEN: "t", OHH_MAX_TURNS: "3" },
      withRegistry(),
    ).maxTurns,
    3,
  );
});

test("a non-numeric maxTurns is rejected rather than becoming NaN", () => {
  assert.throws(
    () =>
      loadConfig(
        { OHH_GPU_TOKEN: "t", OHH_MAX_TURNS: "lots" },
        withRegistry(),
      ),
    ConfigError,
  );
});

// ANTHROPIC_API_KEY is also the variable Claude Code reads, and it takes
// precedence over a subscription login. A key exported for the orchestrator silently
// moved every Claude Code window on the machine onto metered API billing
// (2026-10-18). The orchestrator's key therefore has a name nothing else reads.
test("the orchestrator reads its key from OHH_ANTHROPIC_API_KEY", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_ANTHROPIC_API_KEY: "sk-ohh" },
    withRegistry(),
  );
  assert.equal(config.mode, "routed");
  assert.equal(config.anthropic?.apiKey, "sk-ohh");
});

test("a bare ANTHROPIC_API_KEY is ignored, not borrowed", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", ANTHROPIC_API_KEY: "sk-someone-elses" },
    withRegistry(),
  );
  assert.equal(config.mode, "routed");
  assert.equal(config.anthropic, undefined);
});

test("no key and no OHH_PROVIDER means routed mode with no anthropic config", () => {
  const config = loadConfig({ OHH_GPU_TOKEN: "t" }, withRegistry());
  assert.equal(config.mode, "routed");
  assert.equal(config.anthropic, undefined);
  assert.ok(config.ollama, "the local provider config is always present");
  assert.equal(describeMode(config), "mode: local-only (no Anthropic key configured)");
});

test("a key means routed mode with anthropic config", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_ANTHROPIC_API_KEY: "sk-ohh" },
    withRegistry(),
  );
  assert.equal(config.mode, "routed");
  assert.equal(config.anthropic?.apiKey, "sk-ohh");
  assert.ok(config.ollama);
  assert.equal(describeMode(config), "mode: routed (Claude + local)");
});

test("OHH_PROVIDER=scripted selects scripted even with a key", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_PROVIDER: "scripted", OHH_ANTHROPIC_API_KEY: "sk-ohh" },
    withRegistry(),
  );
  assert.equal(config.mode, "scripted");
  assert.equal(config.anthropic?.apiKey, "sk-ohh");
  assert.equal(describeMode(config), "mode: scripted");
});

test("OHH_PROVIDER=anthropic is rejected", () => {
  assert.throws(
    () =>
      loadConfig(
        { OHH_GPU_TOKEN: "t", OHH_PROVIDER: "anthropic", OHH_ANTHROPIC_API_KEY: "sk-ohh" },
        withRegistry(),
      ),
    (error: unknown) =>
      error instanceof ConfigError &&
      error.message === 'OHH_PROVIDER must be scripted or unset; got "anthropic".',
  );
});

test("localModels lists the registry's model names", () => {
  const config = loadConfig(
    { OHH_GPU_TOKEN: "t" },
    withRegistry(
      REGISTRY +
        `
[models."llama3.1:8b"]
backend = "ollama"
role = "code"
`,
    ),
  );
  assert.deepEqual([...config.localModels].sort(), ["llama3.1:8b", "sdxl-turbo"]);

  const overridden = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_GPU_URL: "http://127.0.0.1:7777" },
    join(tmpdir(), "definitely-not-here", "models.toml"),
  );
  assert.equal(overridden.localModels.size, 0);
});

test("publishedDir defaults to <orchestrator>/published and honours OHH_PUBLISHED", () => {
  const dflt = loadConfig({ OHH_GPU_TOKEN: "t" }, withRegistry());
  assert.equal(dflt.publishedDir, join(import.meta.dirname, "..", "published"));
  const custom = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_PUBLISHED: "/somewhere/out" },
    withRegistry(),
  );
  assert.equal(custom.publishedDir, "/somewhere/out");
});

test("runningDir sits beside the Ledger and honours OHH_RUNNING_DIR", () => {
  const dflt = loadConfig({ OHH_GPU_TOKEN: "t" }, withRegistry());
  assert.equal(dflt.runningDir, join(import.meta.dirname, "..", "running"));
  const moved = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_LEDGER: join("/data", "ledger.db") },
    withRegistry(),
  );
  assert.equal(moved.runningDir, join("/data", "running"));
  const custom = loadConfig(
    { OHH_GPU_TOKEN: "t", OHH_RUNNING_DIR: "/somewhere/running" },
    withRegistry(),
  );
  assert.equal(custom.runningDir, "/somewhere/running");
});

test("OHH_MODELS names the registry, as it does for the GPU service", () => {
  const path = withRegistry(
    `[gpu_service]
host = "10.0.0.5"
port = 7400

[models."only-here:1b"]
backend = "ollama"
`,
  );
  // No registry argument: the path comes from the environment.
  const config = loadConfig({ OHH_GPU_TOKEN: "t", OHH_MODELS: path });
  assert.equal(config.gpuUrl, "http://10.0.0.5:7400");
  assert.deepEqual([...config.localModels], ["only-here:1b"]);

  assert.equal(defaultModelsPath({ OHH_MODELS: path }), path);
  assert.equal(
    defaultModelsPath({}),
    join(import.meta.dirname, "..", "..", "models.toml"),
  );
});
