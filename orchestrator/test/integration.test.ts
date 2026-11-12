/**
 * The one run against real hardware.
 *
 * Excluded from the default run, exactly as the GPU service's `-m integration` marker
 * excludes its own: `node --test` on a machine with no GPU service
 * running must stay green. Opt in with:
 *
 *   OHH_INTEGRATION=1 OHH_GPU_TOKEN=... node --test test/integration.test.ts
 *
 * It makes "the whole thread works on real hardware" provable instead of
 * anecdotal.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadConfig } from "../src/config/config.ts";
import { openLedger } from "../src/ledger/ledger.ts";
import { createGpuClient, GpuError } from "../src/gpu/client.ts";
import { createRegistry } from "../src/tools/registry.ts";
import { createGpuTools } from "../src/tools/gpu-tools.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { runLoop } from "../src/loop/loop.ts";
import type { GpuEvent } from "../src/gpu/client.ts";

const enabled = process.env["OHH_INTEGRATION"] === "1";
const options = { skip: enabled ? false : "set OHH_INTEGRATION=1 to run" };

test("the real GPU service reports the real card", options, async () => {
  const config = loadConfig();
  const gpu = createGpuClient({
    baseUrl: config.gpuUrl,
    token: config.gpuToken,
  });

  const health = await gpu.health();
  assert.equal(health.status, "ok");
  assert.match(health.gpu, /NVIDIA/);
  assert.ok(health.vram_total_mb > 0);
});

test(
  "a Run renders on the GPU and lands in the Ledger",
  { ...options, timeout: 300_000 },
  async () => {
    const config = loadConfig();
    const ledger = openLedger(":memory:");
    const artifactsDir = mkdtempSync(join(tmpdir(), "ohh-integration-"));

    const progress: GpuEvent[] = [];
    const gpu = createGpuClient({
      baseUrl: config.gpuUrl,
      token: config.gpuToken,
      onProgress: (event) => progress.push(event),
    });

    const registry = createRegistry(
      createGpuTools({ gpu, ledger, artifactsDir }),
    );

    const provider = createScriptedProvider({
      turns: [
        [
          { kind: "tool_request", id: "t1", name: "gpu_status", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
        [
          {
            kind: "tool_request",
            id: "t2",
            name: "gpu_generate_image",
            input: { prompt: "a brass agent on black", steps: 4 },
          },
          { kind: "end", reason: "tool_use" },
        ],
        [
          { kind: "text", text: "done" },
          { kind: "end", reason: "end_turn" },
        ],
      ],
    });

    const result = await runLoop(
      { provider, registry, ledger, maxTurns: 10 },
      { agent: "integration", goal: "render one asset", budgetUsd: 1 },
    );

    assert.equal(result.status, "done", "the run did not complete");
    assert.equal(result.turns, 3);

    // Both tools ran, and their tiers were recorded.
    const calls = ledger.listToolCalls(result.runId);
    assert.deepEqual(
      calls.map((c) => [c.tool, c.tier]),
      [
        ["gpu_status", "auto"],
        ["gpu_generate_image", "budgeted"],
      ],
    );

    // A real PNG, on this machine's disk, fetched across the network.
    const artifacts = ledger.listArtifacts(result.runId);
    assert.equal(artifacts.length, 1);
    const bytes = readFileSync(artifacts[0]!.path);
    assert.ok(bytes.length > 1024, "suspiciously small render");
    assert.deepEqual(Array.from(bytes.subarray(0, 8)), [
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ]);
    assert.equal(bytes.subarray(12, 16).toString("ascii"), "IHDR");
    assert.ok(
      bytes.subarray(-8).equals(
        Buffer.from([0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82]),
      ),
      "no IEND terminator — the download was truncated",
    );

    // The SSE path carried something. If this is empty the op still succeeded,
    // but only because the reconciliation poll covered for a silent stream —
    // which is worth knowing rather than passing over.
    assert.ok(
      progress.length > 0,
      "no events arrived over SSE; the op terminated via the poll backstop alone",
    );

    ledger.close();
  },
);

test(
  "a real GPU service-side failure arrives as a typed error with its remedy",
  options,
  async () => {
    // Costs no VRAM: an unknown model is refused by the GPU service before any
    // load. Until now only a stub proved the op.failed path; this proves the
    // real envelope (code/error/remedy) survives the wire and the race.
    const config = loadConfig();
    const gpu = createGpuClient({
      baseUrl: config.gpuUrl,
      token: config.gpuToken,
    });

    await assert.rejects(
      () => gpu.load("no-such-model-exists"),
      (error: unknown) => {
        assert.ok(error instanceof GpuError, `got ${String(error)}`);
        assert.equal(error.code, "UNKNOWN_MODEL");
        assert.ok(error.remedy, "the GPU service's remedy did not survive the wire");
        return true;
      },
    );
  },
);
