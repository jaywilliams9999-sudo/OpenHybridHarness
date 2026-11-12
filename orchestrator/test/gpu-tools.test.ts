import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger } from "../src/ledger/ledger.ts";
import { createGpuTools } from "../src/tools/gpu-tools.ts";
import type { GpuClient } from "../src/gpu/client.ts";

const BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4]);

function fakeGpu(overrides: Partial<GpuClient> = {}): GpuClient {
  return {
    health: async () => ({
      status: "ok",
      gpu: "FakeGpu",
      vram_total_mb: 12288,
      vram_free_mb: 10000,
      version: "0.1.0",
    }),
    status: async () => ({
      vram: { total_mb: 12288, free_mb: 10000, usable_free_mb: 9232 },
      resident: [],
      busy: false,
      queue_depth: 0,
    }),
    load: async () => ({ model: "m", vram_mb: 1, load_ms: 1 }),
    unload: async () => ({ freed_mb: 1 }),
    generateImage: async () => ({
      model: "sdxl-turbo",
      steps: 4,
      artifact_id: "deadbeef",
      artifact_url: "/artifacts/deadbeef.png",
      evicted: [
        { model: "qwen2.5-coder:7b", freed_mb: 4870, reason: "evicted" },
      ],
    }),
    fetchArtifact: async () => BYTES,
    ...overrides,
  } as GpuClient;
}

function harness() {
  const ledger = openLedger(":memory:");
  const artifactsDir = mkdtempSync(join(tmpdir(), "ohh-art-"));
  const tools = createGpuTools({
    gpu: fakeGpu(),
    ledger,
    artifactsDir,
  });
  const runId = ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });
  return { ledger, artifactsDir, tools, runId };
}

test("gpu_status is the auto tier — free and reversible", () => {
  const { ledger, tools } = harness();
  const status = tools.find((t) => t.name === "gpu_status");
  assert.ok(status);
  assert.equal(status.tier, "auto");
  ledger.close();
});

test("gpu_generate_image is budgeted — it costs GPU time", () => {
  const { ledger, tools } = harness();
  const image = tools.find((t) => t.name === "gpu_generate_image");
  assert.ok(image);
  assert.equal(image.tier, "budgeted");
  ledger.close();
});

test("the rendered bytes land on disk and in the Ledger", async () => {
  const { ledger, artifactsDir, tools, runId } = harness();
  const image = tools.find((t) => t.name === "gpu_generate_image");
  assert.ok(image);

  const result = (await image.handler(
    { prompt: "a brass agent on black" },
    { runId },
  )) as { path: string; bytes: number };

  const expected = join(artifactsDir, "deadbeef.png");
  assert.equal(result.path, expected);
  assert.deepEqual(Array.from(readFileSync(expected)), Array.from(BYTES));

  const artifacts = ledger.listArtifacts(runId);
  assert.equal(artifacts.length, 1);
  assert.equal(artifacts[0]?.path, expected);
  assert.equal(JSON.parse(artifacts[0]?.meta_json ?? "{}").bytes, BYTES.length);
  ledger.close();
});

test("the eviction records reach the tool result, not just the log", async () => {
  // This is the difference between a model that
  // commands the GPU and one that merely uses it.
  const { ledger, tools, runId } = harness();
  const image = tools.find((t) => t.name === "gpu_generate_image");
  assert.ok(image);

  const result = (await image.handler({ prompt: "x" }, { runId })) as {
    evicted: Array<{ model: string; freed_mb: number }>;
  };

  assert.equal(result.evicted.length, 1);
  assert.equal(result.evicted[0]?.model, "qwen2.5-coder:7b");
  assert.equal(result.evicted[0]?.freed_mb, 4870);
  ledger.close();
});

test("a missing prompt is refused before the GPU is touched", async () => {
  const { ledger, tools, runId } = harness();
  const image = tools.find((t) => t.name === "gpu_generate_image");
  assert.ok(image);
  await assert.rejects(() => image.handler({}, { runId }), /prompt/);
  ledger.close();
});

test("optional numeric args are forwarded; junk is dropped", async () => {
  const ledger = openLedger(":memory:");
  const artifactsDir = mkdtempSync(join(tmpdir(), "ohh-art-"));
  let seen: Record<string, unknown> | undefined;
  const tools = createGpuTools({
    gpu: fakeGpu({
      generateImage: async (request) => {
        seen = request as Record<string, unknown>;
        return {
          model: "sdxl-turbo",
          steps: 4,
          artifact_id: "deadbeef",
          artifact_url: "/artifacts/deadbeef.png",
          evicted: [],
        };
      },
    }),
    ledger,
    artifactsDir,
  });
  const runId = ledger.createRun({
    agent: "s",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });

  const image = tools.find((t) => t.name === "gpu_generate_image");
  await image?.handler(
    { prompt: "p", steps: 30, width: "wide", height: 576, seed: 7 },
    { runId },
  );

  assert.equal(seen?.["seed"], 7);
  assert.equal(seen?.["height"], 576);
  assert.ok(!("width" in (seen ?? {})), "a non-numeric width was forwarded");
  // Step count is registry policy, not a request parameter. A model that asks
  // for 30 steps on a model distilled for 4 is asking for 13x the render time
  // and a worse image, so the request is dropped rather than honoured.
  assert.ok(!("steps" in (seen ?? {})), "steps reached the GPU service");
  ledger.close();
});

test("the image schema does not offer a step count", () => {
  const ledger = openLedger(":memory:");
  const artifactsDir = mkdtempSync(join(tmpdir(), "ohh-art-"));
  const tools = createGpuTools({
    gpu: fakeGpu({}),
    ledger,
    artifactsDir,
  });

  const image = tools.find((t) => t.name === "gpu_generate_image");
  const schema = image?.schema as {
    properties: Record<string, unknown>;
    additionalProperties: boolean;
  };

  // Dropping it from the handler is not enough on its own: a schema that
  // still advertises `steps` invites the call, and the model has no way to
  // learn that the argument is being quietly discarded.
  assert.ok(!("steps" in schema.properties), "schema still advertises steps");
  assert.equal(schema.additionalProperties, false);
  assert.ok("width" in schema.properties);
  assert.ok("seed" in schema.properties);
  ledger.close();
});
