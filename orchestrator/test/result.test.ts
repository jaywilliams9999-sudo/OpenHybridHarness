/**
 * A root's result: the `messages` row a goal leaves for the operator, and the
 * sweep that gives an interrupted goal one.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger, type Ledger } from "../src/ledger/ledger.ts";
import { claimRoot, releaseRoot } from "../src/run/owners.ts";
import {
  OPERATOR,
  readResult,
  recordResult,
  summaryFromCheckpoint,
  sweepInterrupted,
} from "../src/run/result.ts";

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "ohh-result-"));
  const ledger = openLedger(join(dir, "ledger.db"));
  return {
    ledger,
    runningDir: join(dir, "running"),
    cleanup: () => {
      ledger.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

function spend(ledger: Ledger, runId: string, usd: number): void {
  ledger.recordCost({
    runId,
    provider: "scripted",
    model: "claude-opus-5-5",
    usage: { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 },
    usd,
  });
}

function tree(ledger: Ledger) {
  const root = ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1.5 });
  const child = ledger.createRun({ agent: "site", goal: "c", parentId: root, budgetUsd: 0.5 });
  const grandchild = ledger.createRun({ agent: "site", goal: "gc", parentId: child, budgetUsd: 0.1 });
  return { root, child, grandchild };
}

test("recordResult writes one result message to the operator with tree spend", () => {
  const { ledger, cleanup } = setup();
  try {
    const { root, child } = tree(ledger);
    spend(ledger, root, 0.1);
    spend(ledger, child, 0.25);

    recordResult(ledger, root, { status: "done", summary: "shipped" });

    const results = ledger.listMessages(root).filter((m) => m.kind === "result");
    assert.equal(results.length, 1);
    assert.equal(results[0]!.from_agent, "lead");
    assert.equal(results[0]!.to_agent, OPERATOR);
    const body = JSON.parse(results[0]!.body);
    assert.equal(body.status, "done");
    assert.equal(body.reason, null);
    assert.equal(body.summary, "shipped");
    assert.ok(Math.abs(body.spent_usd - 0.35) < 1e-9);
  } finally {
    cleanup();
  }
});

test("readResult returns the newest", () => {
  const { ledger, cleanup } = setup();
  try {
    const { root } = tree(ledger);
    assert.equal(readResult(ledger, root), undefined);
    recordResult(ledger, root, { status: "failed", reason: "interrupted", summary: "" });
    recordResult(ledger, root, { status: "done", summary: "second" });

    const result = readResult(ledger, root);
    assert.equal(result?.status, "done");
    assert.equal(result?.summary, "second");
    assert.equal(result?.reason, null);
  } finally {
    cleanup();
  }
});

test("summaryFromCheckpoint reads the last assistant text", () => {
  const { ledger, cleanup } = setup();
  try {
    const { root } = tree(ledger);
    assert.equal(summaryFromCheckpoint(ledger, root), undefined);

    ledger.saveCheckpoint(
      root,
      JSON.stringify({
        version: 1,
        agent: "lead",
        goal: "g",
        budgetUsd: 1.5,
        turns: 2,
        spentUsd: 0,
        transcript: [
          { role: "user", text: "g" },
          { role: "assistant", events: [{ kind: "text", text: "first" }] },
          { role: "tool_results", results: [] },
          {
            role: "assistant",
            events: [
              { kind: "thinking", text: "hmm" },
              { kind: "text", text: "waiting on " },
              { kind: "text", text: "the site" },
              { kind: "end", reason: "tool_use" },
            ],
          },
          { role: "tool_results", results: [] },
        ],
      }),
    );
    assert.equal(summaryFromCheckpoint(ledger, root), "waiting on the site");

    ledger.saveCheckpoint(root, "not json");
    assert.equal(summaryFromCheckpoint(ledger, root), undefined);
  } finally {
    cleanup();
  }
});

test("sweepInterrupted fails an unowned running root and its running children", () => {
  const { ledger, runningDir, cleanup } = setup();
  try {
    const { root, child, grandchild } = tree(ledger);
    ledger.setRunStatus(child, "done");
    spend(ledger, grandchild, 0.05);
    // Claimed by a process that has since died.
    claimRoot(runningDir, root);
    releaseRoot(runningDir, root);

    assert.deepEqual(sweepInterrupted(ledger, runningDir), [root]);

    assert.equal(ledger.getRun(root)?.status, "failed");
    assert.equal(ledger.getRun(child)?.status, "done");
    assert.equal(ledger.getRun(grandchild)?.status, "failed");
    const result = readResult(ledger, root);
    assert.equal(result?.status, "failed");
    assert.equal(result?.reason, "interrupted");
    assert.equal(result?.summary, "");
    assert.ok(Math.abs((result?.spent_usd ?? 0) - 0.05) < 1e-9);

    // A second sweep finds nothing left to fail.
    assert.deepEqual(sweepInterrupted(ledger, runningDir), []);
  } finally {
    cleanup();
  }
});

test("sweepInterrupted leaves a blocked root and an owned running root alone", () => {
  const { ledger, runningDir, cleanup } = setup();
  try {
    const blocked = tree(ledger);
    ledger.setRunStatus(blocked.root, "blocked");
    const owned = tree(ledger);
    claimRoot(runningDir, owned.root);

    assert.deepEqual(sweepInterrupted(ledger, runningDir), []);

    assert.equal(ledger.getRun(blocked.root)?.status, "blocked");
    assert.equal(ledger.getRun(blocked.child)?.status, "running");
    assert.equal(ledger.getRun(owned.root)?.status, "running");
    assert.equal(readResult(ledger, blocked.root), undefined);
    assert.equal(readResult(ledger, owned.root), undefined);
  } finally {
    cleanup();
  }
});
