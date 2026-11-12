/**
 * Children report up, and a budget bounds the whole tree.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { resumeRun, runLoop } from "../src/loop/loop.ts";
import { readReport } from "../src/loop/report.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";

function provider(turns: TurnEvent[][]): Provider {
  return { ...createScriptedProvider({ turns }), model: "claude-opus-5-5" };
}

const FINISH: TurnEvent[] = [
  { kind: "text", text: "made it" },
  { kind: "end", reason: "end_turn" },
];

function root(ledger: ReturnType<typeof openLedger>): string {
  return ledger.createRun({
    agent: "lead",
    goal: "root",
    parentId: null,
    budgetUsd: 1,
  });
}

test("a child that finishes writes a report to its parent", async () => {
  const ledger = openLedger(":memory:");
  const parent = root(ledger);

  const result = await runLoop(
    {
      provider: provider([FINISH]),
      registry: createRegistry([]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "build", budgetUsd: 1, parentId: parent },
  );

  assert.equal(result.status, "done");
  assert.deepEqual(readReport(ledger, result.runId), {
    status: "done",
    summary: "made it",
    artifacts: [],
    spent_usd: ledger.getRun(result.runId)?.spent_usd,
    gaps: [],
  });
  const messages = ledger.listMessages(result.runId);
  assert.equal(messages.length, 1);
  assert.equal(messages[0]?.kind, "report");
  assert.equal(messages[0]?.from_agent, "site");
  assert.equal(messages[0]?.to_agent, "lead");
  ledger.close();
});

test("a root Run writes no report", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: provider([FINISH]),
      registry: createRegistry([]),
      ledger,
      maxTurns: 5,
    },
    { agent: "lead", goal: "g", budgetUsd: 1 },
  );
  assert.equal(result.status, "done");
  assert.equal(ledger.listMessages(result.runId).length, 0);
  assert.equal(readReport(ledger, result.runId), undefined);
  ledger.close();
});

test("a denied gated call is listed in gaps", async () => {
  const ledger = openLedger(":memory:");
  const parent = root(ledger);
  const publish: ToolDefinition = {
    name: "publish",
    description: "publish",
    schema: { type: "object" },
    tier: "gated",
    handler: async () => ({ ok: true }),
  };
  const registry = createRegistry([publish]);

  const first = await runLoop(
    {
      provider: provider([
        [
          { kind: "tool_request", id: "p1", name: "publish", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
      ]),
      registry,
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 1, parentId: parent },
  );
  assert.equal(first.status, "blocked");
  // Blocked is not terminal: no report yet.
  assert.equal(readReport(ledger, first.runId), undefined);

  ledger.decideApproval(ledger.listApprovals(first.runId)[0]!.id, "denied");
  const second = await resumeRun(
    { provider: provider([FINISH]), registry, ledger, maxTurns: 5 },
    first.runId,
  );

  assert.equal(second.status, "done");
  const report = readReport(ledger, first.runId);
  assert.equal(report?.status, "done");
  assert.deepEqual(report?.gaps, ["publish denied"]);
  ledger.close();
});

test("a child over budget reports failed with a budget gap", async () => {
  const ledger = openLedger(":memory:");
  const parent = root(ledger);
  const noop: ToolDefinition = {
    name: "look",
    description: "look",
    schema: { type: "object" },
    tier: "auto",
    handler: async () => ({}),
  };

  const result = await runLoop(
    {
      provider: provider([
        [
          { kind: "usage", in: 1000, cacheWrite: 0, cacheRead: 0, out: 1000 },
          { kind: "tool_request", id: "l1", name: "look", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
        FINISH,
      ]),
      registry: createRegistry([noop]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 0.000001, parentId: parent },
  );

  // A child cannot be resumed by anyone, so budget exhaustion is a failure
  // the parent hears about rather than a `blocked` nobody is watching.
  assert.equal(result.status, "failed");
  assert.equal(ledger.getRun(result.runId)?.status, "failed");
  const report = readReport(ledger, result.runId);
  assert.equal(report?.status, "failed");
  assert.deepEqual(report?.gaps, ["budget"]);
  ledger.close();
});

test("the budget check counts descendants' spend", async () => {
  const ledger = openLedger(":memory:");
  const rootId = root(ledger);
  const child = ledger.createRun({
    agent: "site",
    goal: "c",
    parentId: rootId,
    budgetUsd: 1,
  });
  ledger.recordCost({
    runId: child,
    provider: "p",
    model: "m",
    usage: { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 },
    usd: 1.0,
  });
  ledger.saveCheckpoint(
    rootId,
    JSON.stringify({
      version: 1,
      agent: "lead",
      goal: "g",
      budgetUsd: 1,
      transcript: [{ role: "user", text: "g" }],
      turns: 0,
      spentUsd: 0,
    }),
  );

  let called = false;
  const spy: Provider = {
    ...provider([FINISH]),
    chat: async function* () {
      called = true;
      yield* [] as TurnEvent[];
    },
  };
  const result = await resumeRun(
    { provider: spy, registry: createRegistry([]), ledger, maxTurns: 5 },
    rootId,
  );

  assert.equal(result.status, "blocked");
  assert.equal(result.reason, "budget");
  assert.equal(called, false);
  ledger.close();
});

test("a child that ends done on a stop reason other than end_turn says so in gaps", async () => {
  for (const reason of ["max_tokens", "pause_turn"] as const) {
    const ledger = openLedger(":memory:");
    const parent = root(ledger);
    const result = await runLoop(
      {
        provider: provider([
          [
            { kind: "text", text: "half an answ" },
            { kind: "end", reason },
          ],
        ]),
        registry: createRegistry([]),
        ledger,
        maxTurns: 5,
      },
      { agent: "site", goal: "g", budgetUsd: 1, parentId: parent },
    );
    assert.equal(result.status, "done");
    assert.deepEqual(readReport(ledger, result.runId)?.gaps, [reason]);
    ledger.close();
  }
});

test("a child's report is written before its terminal status", async () => {
  const ledger = openLedger(":memory:");
  const parent = root(ledger);
  const order: string[] = [];
  const watched = new Proxy(ledger, {
    get(target, key, receiver) {
      const value = Reflect.get(target, key, receiver);
      if (key === "setRunStatus") {
        return (id: string, status: string) => {
          order.push(`status:${status}`);
          return (value as (id: string, status: string) => unknown)(id, status);
        };
      }
      if (key === "recordMessage") {
        return (message: { kind: string }) => {
          order.push(`message:${message.kind}`);
          return (value as (m: unknown) => unknown)(message);
        };
      }
      return value;
    },
  });

  await runLoop(
    { provider: provider([FINISH]), registry: createRegistry([]), ledger: watched, maxTurns: 5 },
    { agent: "site", goal: "g", budgetUsd: 1, parentId: parent },
  );

  // A reader that sees `done` must find the report already there.
  assert.deepEqual(order, ["message:report", "status:done"]);
  ledger.close();
});
