/**
 * The showcase goal, live, with no Anthropic key: every agent -- the Lead
 * that dispatches included -- runs on the local model through the GPU service.
 *
 * Skipped unless opted in, like the other integration test:
 *
 *   cd orchestrator; $env:OHH_INTEGRATION=1
 *   node --test --test-timeout=600000 test/showcase.integration.test.ts
 *
 * The key is stripped from this test's environment whatever the shell holds.
 * If no GPU service is answering, one is started with `launch()` (Hub stubbed
 * out) and stopped afterwards.
 *
 * What "passed" means depends on the card, and the card is asked rather than
 * assumed. With an image model that fits: the root ends `done`, an image
 * artifact exists in the tree, and a `publish` call (auto-approved here)
 * succeeded. Without one: the root ends `done`, the "image skipped" notice is
 * on the root, a dispatched child finished, and no `site` run was started.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadConfig } from "../src/config/config.ts";
import { readDotEnv } from "../src/env.ts";
import { launch, type LaunchHandle } from "../src/launch.ts";
import { openLedger, type Ledger, type RunRow } from "../src/ledger/ledger.ts";
import { createGpuClient, imageAvailable } from "../src/gpu/client.ts";
import {
  HARNESS,
  decideApproval,
  startGoal,
  type RunnerOptions,
} from "../src/run/runner.ts";

const enabled = process.env["OHH_INTEGRATION"] === "1";
const options = {
  skip: enabled ? false : "set OHH_INTEGRATION=1 to run",
  timeout: 600_000,
};

const GOAL = "Build a landing page for a neighborhood bakery with a hero image";
const REPO = join(import.meta.dirname, "..", "..");

function tree(ledger: Ledger, rootId: string): RunRow[] {
  const out: RunRow[] = [];
  const walk = (id: string) => {
    const run = ledger.getRun(id);
    if (!run) return;
    out.push(run);
    for (const child of ledger.listChildren(id)) walk(child.id);
  };
  walk(rootId);
  return out;
}

const clip = (s: string, n = 300) => (s.length > n ? `${s.slice(0, n)}…` : s);

async function gpuUp(url: string, token: string): Promise<boolean> {
  try {
    const res = await fetch(`${url}/health`, {
      headers: { authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(3000),
    });
    return res.status === 200;
  } catch {
    return false;
  }
}

test("the showcase goal completes local-only", options, async () => {
  const scratch = mkdtempSync(join(tmpdir(), "ohh-showcase-"));
  const env: NodeJS.ProcessEnv = {
    ...readDotEnv(join(REPO, ".env")),
    ...process.env,
    OHH_LEDGER: join(scratch, "ledger.db"),
    OHH_ARTIFACTS: join(scratch, "artifacts"),
    OHH_PUBLISHED: join(scratch, "published"),
  };
  // Local-only mode is the point of this test.
  delete env["OHH_ANTHROPIC_API_KEY"];
  delete env["OHH_PROVIDER"];

  const config = loadConfig(env);
  assert.equal(config.anthropic, undefined);

  let launched: LaunchHandle | undefined;
  if (!(await gpuUp(config.gpuUrl, config.gpuToken))) {
    launched = await launch({
      repoRoot: REPO,
      env,
      startHub: async () => ({ close: async () => {} }),
      log: () => {},
      healthTimeoutMs: 120_000,
    });
  }

  const ledger = openLedger(config.ledgerPath);
  const gpu = createGpuClient({ baseUrl: config.gpuUrl, token: config.gpuToken });
  try {
    const images = imageAvailable(await gpu.status());
    console.log(`  card: image model ${images ? "available" : "unavailable"}`);

    const opts: RunnerOptions = {
      config,
      gpu,
      ledger,
      harness: HARNESS,
      output: (depth) => {
        const pad = "  ".repeat(depth + 1);
        return {
          onEvent: (event) => {
            if (event.kind === "text") console.log(`${pad}${clip(event.text.trim(), 400)}`);
            if (event.kind === "tool_request") {
              console.log(`${pad}→ ${event.name} ${clip(JSON.stringify(event.input), 400)}`);
            }
          },
          onToolCall: ({ tool, isError, ms }) =>
            console.log(`${pad}${isError ? "✗" : "✓"} ${tool} ${ms}ms`),
        };
      },
      onChildCreated: (depth, child) =>
        console.log(`${"  ".repeat(depth)}⤷ ${child.agent} ${child.id.slice(0, 8)}`),
    };

    const started = Date.now();
    let rootId: string | undefined;
    let { status } = await startGoal(opts, {
      agent: "lead",
      goal: GOAL,
      onCreated: (id) => (rootId = id),
    });
    assert.ok(rootId, "no root run was created");

    // The operator, auto-approving `publish` (the only gated tool).
    for (let rounds = 0; status === "blocked" && rounds < 5; rounds++) {
      const pending = tree(ledger, rootId)
        .flatMap((run) => ledger.listApprovals(run.id))
        .filter((a) => a.status === "pending" && a.tool === "publish");
      if (pending.length === 0) break;
      console.log(`  approving ${pending[0]!.tool} ${pending[0]!.args_json}`);
      ({ status } = (await decideApproval(opts, pending[0]!.id, "approved")).result);
    }
    console.log(`  finished in ${((Date.now() - started) / 1000).toFixed(1)}s`);

    const runs = tree(ledger, rootId);
    for (const run of runs) {
      console.log(`  run ${run.agent} ${run.status} turns=${run.turns} placement=${run.placement}`);
      for (const call of ledger.listToolCalls(run.id)) {
        console.log(`    ${call.tool} ${clip(call.args_json, 200)} -> ${clip(call.result_json, 300)}`);
      }
    }

    const root = ledger.getRun(rootId)!;
    assert.equal(root.agent, "lead");
    assert.equal(root.placement, "local-no-key", "the Lead did not run local-only");
    assert.equal(root.status, "done", `root ended ${root.status}`);

    if (images) {
      const artifacts = runs.flatMap((run) => ledger.listArtifacts(run.id));
      assert.ok(
        artifacts.some((a) => a.kind === "image"),
        "no image artifact in the tree",
      );
      const published = runs
        .flatMap((run) => ledger.listToolCalls(run.id))
        .filter((call) => call.tool === "publish")
        .filter((call) => {
          const result = JSON.parse(call.result_json) as { error?: unknown } | null;
          return result !== null && result.error === undefined;
        });
      assert.ok(published.length > 0, "no publish call succeeded");
    } else {
      const notices = ledger
        .listMessages(rootId)
        .filter((m) => m.kind === "notice")
        // Bodies are stored JSON-encoded.
        .map((m) => JSON.parse(m.body) as unknown);
      assert.ok(
        notices.includes("image skipped: model doesn't fit this card"),
        `no image-skipped notice; got ${JSON.stringify(notices)}`,
      );
      // The Lead must have driven `dispatch`: a root that ends `done` having
      // called no tool proves nothing about local tool use.
      assert.ok(
        runs.some((run) => run.parent_id === rootId && run.status === "done"),
        "the Lead dispatched no child that finished",
      );
      assert.deepEqual(
        runs.filter((run) => run.agent === "site").map((run) => run.id),
        [],
        "a site run was started on a card that cannot render",
      );
    }
  } finally {
    ledger.close();
    if (launched) {
      // Leave the card as it was found: Ollama keeps a model resident until
      // told otherwise, and stopping the GPU service does not tell it.
      try {
        for (const { model } of (await gpu.status()).resident) await gpu.unload(model);
      } catch (error) {
        console.log(`  unload on the way out failed: ${String(error)}`);
      }
      await launched.stop();
    }
  }
});
