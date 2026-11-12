import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { runLoop } from "../src/loop/loop.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";

function run(ledger: ReturnType<typeof openLedger>): string {
  return ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 2,
  });
}

test("recordCost writes the row and bumps spent_usd together", () => {
  const ledger = openLedger(":memory:");
  const id = run(ledger);

  ledger.recordCost({
    runId: id,
    provider: "anthropic",
    model: "claude-opus-5-5",
    usage: { in: 100, cacheWrite: 200, cacheRead: 300, out: 400 },
    usd: 0.25,
  });

  const rows = ledger.listCosts(id);
  assert.equal(rows.length, 1);
  assert.equal(rows[0]?.provider, "anthropic");
  assert.equal(rows[0]?.model, "claude-opus-5-5");
  assert.equal(rows[0]?.in_tokens, 100);
  assert.equal(rows[0]?.cache_write_tokens, 200);
  assert.equal(rows[0]?.cache_read_tokens, 300);
  assert.equal(rows[0]?.out_tokens, 400);
  assert.equal(ledger.getRun(id)?.spent_usd, 0.25);
  ledger.close();
});

test("usage_json keeps the raw counts so a wrong price table can be re-derived", () => {
  const ledger = openLedger(":memory:");
  const id = run(ledger);
  const usage = { in: 1, cacheWrite: 2, cacheRead: 3, out: 4 };
  ledger.recordCost({
    runId: id,
    provider: "anthropic",
    model: "m",
    usage,
    usd: 0,
  });
  assert.deepEqual(JSON.parse(ledger.listCosts(id)[0]?.usage_json ?? "{}"), usage);
  ledger.close();
});

test("spend accumulates across turns", () => {
  const ledger = openLedger(":memory:");
  const id = run(ledger);
  for (const usd of [0.1, 0.2, 0.05]) {
    ledger.recordCost({
      runId: id,
      provider: "anthropic",
      model: "m",
      usage: { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 },
      usd,
    });
  }
  assert.equal(ledger.listCosts(id).length, 3);
  assert.ok(Math.abs((ledger.getRun(id)?.spent_usd ?? 0) - 0.35) < 1e-9);
  ledger.close();
});

test("a cost row for an unknown run is rejected, not silently orphaned", () => {
  const ledger = openLedger(":memory:");
  assert.throws(() =>
    ledger.recordCost({
      runId: "not-a-run",
      provider: "anthropic",
      model: "m",
      usage: { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 },
      usd: 1,
    }),
  );
  // The failed insert must not have moved anything.
  assert.equal(ledger.listCosts("not-a-run").length, 0);
  ledger.close();
});

// -- through the loop ---------------------------------------------------

/** A provider that emits usage, so the loop has something to price. */
function pricedProvider(turns: TurnEvent[][], model: string): Provider {
  const inner = createScriptedProvider({ turns });
  return { ...inner, model };
}

test("the loop prices each turn and totals it on the result", async () => {
  const ledger = openLedger(":memory:");
  const provider = pricedProvider(
    [
      [
        { kind: "text", text: "one" },
        { kind: "usage", in: 1_000, cacheWrite: 0, cacheRead: 0, out: 1_000 },
        { kind: "end", reason: "end_turn" },
      ],
    ],
    "claude-opus-5-5",
  );

  const result = await runLoop(
    { provider, registry: createRegistry([]), ledger, maxTurns: 5 },
    { agent: "site", goal: "g", budgetUsd: 2 },
  );

  // 1000 in at $4/MTok + 1000 out at $20/MTok.
  const expected = (1_000 * 4 + 1_000 * 20) / 1_000_000;
  assert.ok(Math.abs(result.spentUsd - expected) < 1e-12, `${result.spentUsd}`);
  assert.equal(ledger.listCosts(result.runId).length, 1);
  assert.ok(
    Math.abs((ledger.getRun(result.runId)?.spent_usd ?? 0) - expected) < 1e-12,
  );
  ledger.close();
});

test("a refused turn is still priced — refusals cost money", async () => {
  const ledger = openLedger(":memory:");
  const provider = pricedProvider(
    [
      [
        { kind: "usage", in: 500, cacheWrite: 0, cacheRead: 0, out: 10 },
        { kind: "end", reason: "refusal" },
      ],
    ],
    "claude-opus-5-5",
  );

  const result = await runLoop(
    { provider, registry: createRegistry([]), ledger, maxTurns: 5 },
    { agent: "site", goal: "g", budgetUsd: 2 },
  );

  assert.equal(result.status, "failed");
  assert.equal(result.reason, "refusal");
  assert.ok(result.spentUsd > 0, "a refusal that billed tokens must be recorded");
  assert.equal(ledger.listCosts(result.runId).length, 1);
  ledger.close();
});

test("a turn cap trip records every turn it paid for", async () => {
  const ledger = openLedger(":memory:");
  const tool: ToolDefinition = {
    name: "echo",
    description: "echo",
    schema: { type: "object" },
    tier: "auto",
    handler: async () => ({ ok: true }),
  };
  const toolTurn: TurnEvent[] = [
    { kind: "tool_request", id: "t", name: "echo", input: {} },
    { kind: "usage", in: 100, cacheWrite: 0, cacheRead: 0, out: 10 },
    { kind: "end", reason: "tool_use" },
  ];
  const provider = pricedProvider(
    [toolTurn, toolTurn, toolTurn],
    "claude-opus-5-5",
  );

  const result = await runLoop(
    { provider, registry: createRegistry([tool]), ledger, maxTurns: 2 },
    { agent: "site", goal: "g", budgetUsd: 2 },
  );

  assert.equal(result.reason, "max_turns");
  assert.equal(ledger.listCosts(result.runId).length, 2);
  ledger.close();
});

test("a provider that reports no usage writes no cost row", async () => {
  // The scripted double has no spend. A zero-dollar row would be noise.
  const ledger = openLedger(":memory:");
  const provider = createScriptedProvider({
    turns: [[{ kind: "text", text: "done" }, { kind: "end", reason: "end_turn" }]],
  });

  const result = await runLoop(
    { provider, registry: createRegistry([]), ledger, maxTurns: 5 },
    { agent: "site", goal: "g", budgetUsd: 2 },
  );

  assert.equal(result.spentUsd, 0);
  assert.equal(ledger.listCosts(result.runId).length, 0);
  ledger.close();
});

test("onCost reports the per-turn and cumulative figures", async () => {
  const ledger = openLedger(":memory:");
  const turn: TurnEvent[] = [
    { kind: "tool_request", id: "t", name: "echo", input: {} },
    { kind: "usage", in: 1_000, cacheWrite: 0, cacheRead: 0, out: 0 },
    { kind: "end", reason: "tool_use" },
  ];
  const provider = pricedProvider([turn, turn], "claude-opus-5-5");
  const seen: Array<{ usd: number; cumulativeUsd: number }> = [];

  await runLoop(
    {
      provider,
      registry: createRegistry([
        {
          name: "echo",
          description: "e",
          schema: {},
          tier: "auto",
          handler: async () => ({}),
        },
      ]),
      ledger,
      maxTurns: 2,
      onCost: (cost) => seen.push(cost),
    },
    { agent: "site", goal: "g", budgetUsd: 2 },
  );

  assert.equal(seen.length, 2);
  assert.ok(Math.abs((seen[0]?.usd ?? 0) - 0.004) < 1e-12);
  assert.ok(Math.abs((seen[1]?.cumulativeUsd ?? 0) - 0.008) < 1e-12);
  ledger.close();
});
