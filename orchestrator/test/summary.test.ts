/**
 * The end-of-run Ledger summary speaks for the whole tree.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { printLedgerSummary } from "../src/cli/summary.ts";

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

test("a child's artifact and evictions appear under it, and the tree total is last", () => {
  const ledger = openLedger(":memory:");
  const root = ledger.createRun({ agent: "lead", goal: "ship", parentId: null, budgetUsd: 1.5 });
  const child = ledger.createRun({ agent: "site", goal: "render", parentId: root, budgetUsd: 0.5 });
  ledger.setRunStatus(child, "done");
  ledger.recordArtifact({
    runId: child,
    kind: "image",
    path: "artifacts/hero.png",
    meta: { bytes: 1234, evicted: [{ model: "qwen", freed_mb: 9000 }] },
  });
  ledger.recordCost({
    runId: child,
    provider: "anthropic",
    model: "claude-opus-5-5",
    usage: { in: 1, cacheWrite: 0, cacheRead: 0, out: 1 },
    usd: 0.25,
  });

  const lines: string[] = [];
  printLedgerSummary(
    ledger,
    { runId: root, status: "done", turns: 2, text: "", thinking: "", spentUsd: 0 },
    (line) => lines.push(plain(line)),
  );

  const at = lines.findIndex((l) => l.includes(`⤷ site ${child.slice(0, 8)} done`));
  assert.ok(at > 0, lines.join("\n"));
  // Indented one level under the child line, as the root's own lines are.
  assert.equal(lines[at + 1], "    art     artifacts/hero.png 1234 bytes");
  assert.equal(lines[at + 2], "    evicted qwen freed 9000MB");
  assert.equal(lines.at(-1), "  tree    $0.2500");
});

test("a blocked run prints the --approve command for each pending approval", () => {
  const ledger = openLedger(":memory:");
  const root = ledger.createRun({ agent: "lead", goal: "ship", parentId: null, budgetUsd: 1.5 });
  const child = ledger.createRun({ agent: "site", goal: "render", parentId: root, budgetUsd: 0.5 });
  const own = ledger.requestApproval({ runId: root, tool: "publish", args: {} });
  const childs = ledger.requestApproval({ runId: child, tool: "publish", args: {} });

  const lines: string[] = [];
  printLedgerSummary(
    ledger,
    { runId: root, status: "blocked", reason: "approval", turns: 1, text: "", thinking: "", spentUsd: 0 },
    (line) => lines.push(plain(line)),
  );

  assert.ok(lines.some((l) => l.trim() === `--approve ${own}`), lines.join("\n"));
  assert.ok(lines.some((l) => l.trim() === `--approve ${childs}`), lines.join("\n"));
  assert.ok(!lines.some((l) => l.includes("approve:")), lines.join("\n"));
});
