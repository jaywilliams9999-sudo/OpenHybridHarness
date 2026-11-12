import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger, ranAs } from "../src/ledger/ledger.ts";

const require_ = createRequire(import.meta.url);

// All six tables are created, but only runs, tool_calls and artifacts get
// accessors. The empty three are asserted to EXIST so that a later change
// adding a writer does not also have to add a table.

test("all six tables exist", () => {
  const ledger = openLedger(":memory:");
  const names = ledger.tableNames();
  for (const table of [
    "runs",
    "messages",
    "tool_calls",
    "costs",
    "artifacts",
    "approvals",
  ]) {
    assert.ok(names.includes(table), `missing table ${table}`);
  }
  ledger.close();
});

test("the costs table carries the four token buckets and usage_json", () => {
  // A cache read costs
  // ~0.1x a fresh input token, so collapsing the buckets overstates a
  // cache-heavy turn ~10x — and the budget spends that number.
  const ledger = openLedger(":memory:");
  const columns = ledger.columnNames("costs");
  for (const column of [
    "in_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "out_tokens",
    "usd",
    "usage_json",
  ]) {
    assert.ok(columns.includes(column), `costs is missing ${column}`);
  }
  ledger.close();
});

test("a run round-trips", () => {
  const ledger = openLedger(":memory:");
  const id = ledger.createRun({
    agent: "site",
    goal: "render an agent",
    parentId: null,
    budgetUsd: 2,
  });

  const run = ledger.getRun(id);
  assert.ok(run);
  assert.equal(run.agent, "site");
  assert.equal(run.goal, "render an agent");
  assert.equal(run.status, "running");
  assert.equal(run.spent_usd, 0);
  assert.equal(run.turns, 0);
  ledger.close();
});

test("status and turn count advance", () => {
  const ledger = openLedger(":memory:");
  const id = ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });

  ledger.bumpTurn(id);
  ledger.bumpTurn(id);
  ledger.setRunStatus(id, "done");

  const run = ledger.getRun(id);
  assert.ok(run);
  assert.equal(run.turns, 2);
  assert.equal(run.status, "done");
  ledger.close();
});

test("a tool call records its risk tier", () => {
  // Tiers are recorded on every call, so the data exists from the first turn
  // for the Loop to enforce against.
  const ledger = openLedger(":memory:");
  const runId = ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });

  ledger.recordToolCall({
    runId,
    tool: "gpu_generate_image",
    args: { prompt: "a brass agent" },
    result: { artifact_id: "abc" },
    tier: "budgeted",
    ms: 1234,
  });

  const calls = ledger.listToolCalls(runId);
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.tool, "gpu_generate_image");
  assert.equal(calls[0]?.tier, "budgeted");
  assert.equal(calls[0]?.ms, 1234);
  assert.deepEqual(JSON.parse(calls[0]?.args_json ?? "{}"), {
    prompt: "a brass agent",
  });
  ledger.close();
});

test("an artifact records its path and meta", () => {
  const ledger = openLedger(":memory:");
  const runId = ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });

  ledger.recordArtifact({
    runId,
    kind: "image",
    path: "orchestrator/artifacts/abc.png",
    meta: { bytes: 4096, evicted: ["qwen2.5-coder:7b"] },
  });

  const artifacts = ledger.listArtifacts(runId);
  assert.equal(artifacts.length, 1);
  assert.equal(artifacts[0]?.path, "orchestrator/artifacts/abc.png");
  assert.equal(JSON.parse(artifacts[0]?.meta_json ?? "{}").bytes, 4096);
  ledger.close();
});

test("a tool call against an unknown run is refused by the foreign key", () => {
  // Without enforcement on, sqlite accepts the orphan silently and the Ledger
  // stops being a faithful record of what happened.
  const ledger = openLedger(":memory:");
  assert.throws(() =>
    ledger.recordToolCall({
      runId: "no-such-run",
      tool: "gpu_status",
      args: {},
      result: {},
      tier: "auto",
      ms: 1,
    }),
  );
  ledger.close();
});

test("messages round-trip with their body as JSON", () => {
  const ledger = openLedger(":memory:");
  const id = ledger.createRun({
    agent: "lead",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });
  ledger.recordMessage({
    runId: id,
    from: "lead",
    to: "site",
    kind: "dispatch",
    body: { goal: "g" },
  });

  const messages = ledger.listMessages(id);
  assert.equal(messages.length, 1);
  assert.equal(messages[0]?.kind, "dispatch");
  assert.equal(messages[0]?.from_agent, "lead");
  assert.equal(messages[0]?.to_agent, "site");
  assert.equal(JSON.parse(messages[0]?.body ?? "{}").goal, "g");
  ledger.close();
});

test("tree spend sums a Run and all its descendants", () => {
  const ledger = openLedger(":memory:");
  const spend = (runId: string, usd: number) =>
    ledger.recordCost({
      runId,
      provider: "anthropic",
      model: "m",
      usage: { in: 1, cacheWrite: 0, cacheRead: 0, out: 1 },
      usd,
    });
  const root = ledger.createRun({
    agent: "lead",
    goal: "r",
    parentId: null,
    budgetUsd: 5,
  });
  const child = ledger.createRun({
    agent: "site",
    goal: "c",
    parentId: root,
    budgetUsd: 2,
  });
  const grandchild = ledger.createRun({
    agent: "site",
    goal: "gc",
    parentId: child,
    budgetUsd: 1,
  });
  spend(root, 0.1);
  spend(child, 0.2);
  spend(grandchild, 0.3);

  assert.ok(Math.abs(ledger.treeSpentUsd(root) - 0.6) < 1e-9);
  assert.ok(Math.abs(ledger.treeSpentUsd(child) - 0.5) < 1e-9);
  ledger.close();
});

test("listChildren returns only direct children", () => {
  const ledger = openLedger(":memory:");
  const root = ledger.createRun({
    agent: "lead",
    goal: "r",
    parentId: null,
    budgetUsd: 5,
  });
  const child = ledger.createRun({
    agent: "site",
    goal: "c",
    parentId: root,
    budgetUsd: 2,
  });
  ledger.createRun({
    agent: "site",
    goal: "gc",
    parentId: child,
    budgetUsd: 1,
  });

  const children = ledger.listChildren(root);
  assert.equal(children.length, 1);
  assert.equal(children[0]?.id, child);
  ledger.close();
});

test("messages come back in insertion order even within one millisecond", () => {
  // created_at has millisecond resolution and ids are random UUIDs, so only
  // rowid keeps a dispatch ahead of its report.
  const ledger = openLedger(":memory:");
  const id = ledger.createRun({
    agent: "lead",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });
  for (let i = 0; i < 20; i++) {
    ledger.recordMessage({
      runId: id,
      from: "lead",
      to: "site",
      kind: `m${i}`,
      body: {},
    });
  }
  assert.deepEqual(
    ledger.listMessages(id).map((m) => m.kind),
    Array.from({ length: 20 }, (_, i) => `m${i}`),
  );
  ledger.close();
});

test("two connections on one file see each other's writes", () => {
  // The Loop and the Hub are two writers on one ledger.db. Writes here are
  // synchronous, so this checks visibility across connections, not contention.
  const dir = mkdtempSync(join(tmpdir(), "ohh-ledger-"));
  const path = join(dir, "ledger.db");
  const first = openLedger(path);
  const second = openLedger(path);
  try {
    const ids: string[] = [];
    for (let i = 0; i < 50; i++) {
      for (const ledger of [first, second]) {
        ids.push(
          ledger.createRun({
            agent: "lead",
            goal: `g${i}`,
            parentId: null,
            budgetUsd: 1,
          }),
        );
      }
    }
    assert.equal(ids.length, 100);
    for (const id of ids) {
      assert.ok(first.getRun(id), `first cannot read ${id}`);
      assert.ok(second.getRun(id), `second cannot read ${id}`);
    }
    assert.equal(first.listRootRuns().length, 100);
  } finally {
    first.close();
    second.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the ledger runs in WAL mode", () => {
  const dir = mkdtempSync(join(tmpdir(), "ohh-ledger-"));
  const path = join(dir, "ledger.db");
  const ledger = openLedger(path);
  const { DatabaseSync } = require_("node:sqlite") as typeof import("node:sqlite");
  const raw = new DatabaseSync(path);
  try {
    const row = raw.prepare("PRAGMA journal_mode").get() as {
      journal_mode: string;
    };
    assert.equal(row.journal_mode, "wal");
  } finally {
    raw.close();
    ledger.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("listRootRuns returns only roots, newest first", () => {
  const ledger = openLedger(":memory:");
  const a = ledger.createRun({
    agent: "lead",
    goal: "a",
    parentId: null,
    budgetUsd: 1,
  });
  const b = ledger.createRun({
    agent: "lead",
    goal: "b",
    parentId: null,
    budgetUsd: 1,
  });
  ledger.createRun({
    agent: "site",
    goal: "child of a",
    parentId: a,
    budgetUsd: 1,
  });

  assert.deepEqual(
    ledger.listRootRuns().map((run) => run.id),
    [b, a],
  );
  ledger.close();
});

test("getArtifact finds by id and returns undefined for an unknown id", () => {
  const ledger = openLedger(":memory:");
  const runId = ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });
  const id = ledger.recordArtifact({
    runId,
    kind: "image",
    path: "orchestrator/artifacts/abc.png",
    meta: null,
  });

  assert.equal(ledger.getArtifact(id)?.path, "orchestrator/artifacts/abc.png");
  assert.equal(ledger.getArtifact("no-such-artifact"), undefined);
  ledger.close();
});

test("opening a ledger that holds only unrelated tables creates the runs schema", () => {
  const dir = mkdtempSync(join(tmpdir(), "ohh-ledger-"));
  try {
    const path = join(dir, "ledger.db");
    const { DatabaseSync } = require_("node:sqlite");
    const db = new DatabaseSync(path);
    db.exec("CREATE TABLE legacy_tasks (id TEXT PRIMARY KEY, owner TEXT NOT NULL)");
    db.exec("INSERT INTO legacy_tasks VALUES ('old', 'someone')");
    db.close();
    const ledger = openLedger(path);
    assert.deepEqual(ledger.listRootRuns(), []);
    const id = ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1.5 });
    assert.equal(ledger.getRun(id)?.agent, "lead");
    ledger.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("createRun stores placement", () => {
  const ledger = openLedger(":memory:");
  const claude = ledger.createRun({
    agent: "lead",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
    placement: "claude",
  });
  const plain = ledger.createRun({ agent: "site", goal: "g", parentId: claude, budgetUsd: 1 });

  assert.equal(ledger.getRun(claude)?.placement, "claude");
  // Unstated is local: the default an agent with no model line gets.
  assert.equal(ledger.getRun(plain)?.placement, "local");
  ledger.close();
});

test("ranAs reports local-fallback when a claude run recorded an ollama cost", () => {
  const ledger = openLedger(":memory:");
  const id = ledger.createRun({
    agent: "lead",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
    placement: "claude",
  });
  const usage = { in: 1, cacheWrite: 0, cacheRead: 0, out: 1 };
  ledger.recordCost({ runId: id, provider: "anthropic", model: "claude-sonnet-5-5", usage, usd: 0.01 });
  ledger.recordCost({ runId: id, provider: "ollama", model: "llama3.1:8b", usage, usd: 0 });

  assert.equal(ranAs(ledger, ledger.getRun(id)!), "local-fallback");
  ledger.close();
});

test("ranAs reports placement otherwise", () => {
  const ledger = openLedger(":memory:");
  const usage = { in: 1, cacheWrite: 0, cacheRead: 0, out: 1 };
  const claude = ledger.createRun({
    agent: "lead", goal: "g", parentId: null, budgetUsd: 1, placement: "claude",
  });
  ledger.recordCost({ runId: claude, provider: "anthropic", model: "claude-sonnet-5-5", usage, usd: 0.01 });
  const noKey = ledger.createRun({
    agent: "lead", goal: "g", parentId: null, budgetUsd: 1, placement: "local-no-key",
  });
  ledger.recordCost({ runId: noKey, provider: "ollama", model: "llama3.1:8b", usage, usd: 0 });
  const local = ledger.createRun({
    agent: "site", goal: "g", parentId: null, budgetUsd: 1, placement: "local",
  });

  assert.equal(ranAs(ledger, ledger.getRun(claude)!), "claude");
  assert.equal(ranAs(ledger, ledger.getRun(noKey)!), "local-no-key");
  assert.equal(ranAs(ledger, ledger.getRun(local)!), "local");
  ledger.close();
});

test("an older ledger without the placement column gains it on open", () => {
  const dir = mkdtempSync(join(tmpdir(), "ohh-ledger-"));
  try {
    const path = join(dir, "ledger.db");
    const { DatabaseSync } = require_("node:sqlite");
    const db = new DatabaseSync(path);
    db.exec(`CREATE TABLE runs (
      id TEXT PRIMARY KEY, agent TEXT NOT NULL, goal TEXT NOT NULL,
      parent_id TEXT REFERENCES runs(id),
      status TEXT NOT NULL CHECK (status IN ('running','blocked','done','failed')),
      budget_usd REAL NOT NULL, spent_usd REAL NOT NULL DEFAULT 0,
      turns INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL, checkpoint_blob TEXT)`);
    db.exec(`INSERT INTO runs (id, agent, goal, status, budget_usd, created_at, updated_at)
             VALUES ('old', 'site', 'g', 'done', 1, 't', 't')`);
    db.close();

    const ledger = openLedger(path);
    assert.equal(ledger.getRun("old")?.placement, "local");
    const id = ledger.createRun({
      agent: "lead", goal: "g", parentId: null, budgetUsd: 1, placement: "claude",
    });
    assert.equal(ledger.getRun(id)?.placement, "claude");
    ledger.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
