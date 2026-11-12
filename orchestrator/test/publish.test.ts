import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { openLedger } from "../src/ledger/ledger.ts";
import { createPublishTool } from "../src/tools/publish.ts";

function harness() {
  const ledger = openLedger(":memory:");
  const dir = mkdtempSync(join(tmpdir(), "ohh-pub-"));
  const publishedDir = join(dir, "published");
  const make = () =>
    ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1 });
  const runId = make();
  const otherRunId = make();
  const tool = createPublishTool({ ledger, publishedDir });
  return { ledger, dir, publishedDir, tool, runId, otherRunId };
}

function artifact(h: ReturnType<typeof harness>, runId: string, name: string) {
  const path = join(h.dir, name);
  writeFileSync(path, Buffer.from([1, 2, 3, 255, 0, 9]));
  h.ledger.recordArtifact({ runId, kind: "image", path, meta: {} });
  return path;
}

test("publish is a gated tool that requires artifact_path", () => {
  const { tool } = harness();
  assert.equal(tool.name, "publish");
  assert.equal(tool.tier, "gated");
  const schema = tool.schema as Record<string, unknown>;
  assert.deepEqual(schema["required"], ["artifact_path"]);
  assert.equal(schema["additionalProperties"], false);
  assert.match(tool.description, /approval/i);
});

test("validate accepts an artifact recorded for this Run", () => {
  const h = harness();
  const path = artifact(h, h.runId, "a.png");
  assert.equal(
    h.tool.validate?.({ artifact_path: path }, { runId: h.runId }),
    undefined,
  );
});

test("validate rejects an unrecorded path", () => {
  const h = harness();
  const path = join(h.dir, "never-recorded.png");
  writeFileSync(path, "x");
  const msg = h.tool.validate?.({ artifact_path: path }, { runId: h.runId });
  assert.equal(typeof msg, "string");
});

test("validate rejects another Run's artifact", () => {
  const h = harness();
  const path = artifact(h, h.otherRunId, "b.png");
  const msg = h.tool.validate?.({ artifact_path: path }, { runId: h.runId });
  assert.equal(typeof msg, "string");
});

test("validate rejects a missing or non-string artifact_path", () => {
  const h = harness();
  assert.equal(typeof h.tool.validate?.({}, { runId: h.runId }), "string");
  assert.equal(
    typeof h.tool.validate?.({ artifact_path: 3 }, { runId: h.runId }),
    "string",
  );
});

test("the handler copies the file into publishedDir byte for byte", async () => {
  const h = harness();
  const path = artifact(h, h.runId, "c.png");
  const result = (await h.tool.handler(
    { artifact_path: path },
    { runId: h.runId },
  )) as { published: string };
  const dest = join(h.publishedDir, basename(path));
  assert.equal(result.published, dest);
  assert.ok(existsSync(dest));
  assert.deepEqual(readFileSync(dest), readFileSync(path));
});
