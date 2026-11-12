/**
 * The approval queue.
 *
 * A `gated` call sets the Run to `blocked` and writes an `approvals` row. On
 * approval the agent resumes from its checkpoint; on denial the tool returns a
 * denial result the agent must handle.
 *
 * The thing these tests mostly protect is that **nothing runs before the
 * human answers** — these tools occupy a GPU and write files.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { parseCheckpoint, resumeRun, runLoop } from "../src/loop/loop.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";

function provider(turns: TurnEvent[][]): Provider {
  return { ...createScriptedProvider({ turns }), model: "claude-opus-5-5" };
}

/** Records every invocation, so "did it run?" is answerable. */
function spyTool(
  name: string,
  tier: ToolDefinition["tier"],
  calls: string[],
): ToolDefinition {
  return {
    name,
    description: name,
    schema: { type: "object" },
    tier,
    handler: async (args) => {
      calls.push(name);
      return { ran: name, args };
    },
  };
}

const ASK_GATED: TurnEvent[] = [
  { kind: "tool_request", id: "g1", name: "buy_domain", input: { name: "x.dev" } },
  { kind: "end", reason: "tool_use" },
];

const FINISH: TurnEvent[] = [
  { kind: "text", text: "all done" },
  { kind: "end", reason: "end_turn" },
];

// -- blocking -----------------------------------------------------------

test("a gated call blocks the Run and writes an approvals row", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];

  const result = await runLoop(
    {
      provider: provider([ASK_GATED, FINISH]),
      registry: createRegistry([spyTool("buy_domain", "gated", calls)]),
      ledger,
      maxTurns: 5,
    },
    { agent: "lead", goal: "get a domain", budgetUsd: 2 },
  );

  assert.equal(result.status, "blocked");
  assert.equal(result.reason, "approval");
  // The whole point: it did not run.
  assert.deepEqual(calls, []);

  const approvals = ledger.listApprovals(result.runId);
  assert.equal(approvals.length, 1);
  assert.equal(approvals[0]?.tool, "buy_domain");
  assert.equal(approvals[0]?.status, "pending");
  assert.deepEqual(JSON.parse(approvals[0]?.args_json ?? "null"), {
    name: "x.dev",
  });
  assert.equal(approvals[0]?.decided_at, null);
  ledger.close();
});

test("auto and budgeted tiers are not gated", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];

  const result = await runLoop(
    {
      provider: provider([
        [
          { kind: "tool_request", id: "a", name: "look", input: {} },
          { kind: "tool_request", id: "b", name: "render", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
        FINISH,
      ]),
      registry: createRegistry([
        spyTool("look", "auto", calls),
        spyTool("render", "budgeted", calls),
      ]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 2 },
  );

  assert.equal(result.status, "done");
  assert.deepEqual(calls.sort(), ["look", "render"]);
  assert.equal(ledger.listApprovals(result.runId).length, 0);
  ledger.close();
});

test("one gated call holds the whole turn, including its ungated siblings", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];

  const result = await runLoop(
    {
      provider: provider([
        [
          { kind: "tool_request", id: "a", name: "look", input: {} },
          { kind: "tool_request", id: "g", name: "buy_domain", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
        FINISH,
      ]),
      registry: createRegistry([
        spyTool("look", "auto", calls),
        spyTool("buy_domain", "gated", calls),
      ]),
      ledger,
      maxTurns: 5,
    },
    { agent: "lead", goal: "g", budgetUsd: 2 },
  );

  assert.equal(result.status, "blocked");
  // Running the ungated one would half-apply a turn nobody approved, and would
  // also move the resume point past a side effect.
  assert.deepEqual(calls, []);

  // Both calls are carried so the turn can be executed whole on resume; only
  // the gated one has an approval behind it.
  const checkpoint = parseCheckpoint(ledger.readCheckpoint(result.runId) ?? "");
  assert.equal(checkpoint.pending?.length, 2);
  assert.equal(
    checkpoint.pending?.filter((call) => call.approvalId !== undefined).length,
    1,
  );
  assert.equal(ledger.listApprovals(result.runId).length, 1);
  ledger.close();
});

// -- deciding -----------------------------------------------------------

test("approval lets the held turn run and the Run finish", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const registry = createRegistry([spyTool("buy_domain", "gated", calls)]);

  const first = await runLoop(
    {
      provider: provider([ASK_GATED, FINISH]),
      registry,
      ledger,
      maxTurns: 5,
    },
    { agent: "lead", goal: "g", budgetUsd: 2 },
  );
  assert.equal(first.status, "blocked");

  const approval = ledger.listApprovals(first.runId)[0];
  assert.ok(approval);
  assert.equal(ledger.decideApproval(approval.id, "approved"), true);

  const second = await resumeRun(
    { provider: provider([FINISH]), registry, ledger, maxTurns: 5 },
    first.runId,
  );

  assert.equal(second.status, "done");
  assert.deepEqual(calls, ["buy_domain"]);
  assert.equal(second.text, "all done");
  ledger.close();
});

test("denial returns a result the model handles, rather than failing the Run", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const registry = createRegistry([spyTool("buy_domain", "gated", calls)]);

  const first = await runLoop(
    { provider: provider([ASK_GATED]), registry, ledger, maxTurns: 5 },
    { agent: "lead", goal: "g", budgetUsd: 2 },
  );
  const approval = ledger.listApprovals(first.runId)[0];
  assert.ok(approval);
  ledger.decideApproval(approval.id, "denied");

  const second = await resumeRun(
    { provider: provider([FINISH]), registry, ledger, maxTurns: 5 },
    first.runId,
  );

  // A denial is a result, not a failure. The Run carries on.
  assert.equal(second.status, "done");
  assert.deepEqual(calls, [], "a denied tool must not run");

  const recorded = ledger.listToolCalls(first.runId);
  assert.equal(recorded.length, 1);
  const payload = JSON.parse(recorded[0]?.result_json ?? "{}") as {
    denied?: boolean;
  };
  assert.equal(payload.denied, true);
  ledger.close();
});

test("a denied gated call does not stop its ungated siblings on resume", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const registry = createRegistry([
    spyTool("look", "auto", calls),
    spyTool("buy_domain", "gated", calls),
  ]);

  const first = await runLoop(
    {
      provider: provider([
        [
          { kind: "tool_request", id: "a", name: "look", input: {} },
          { kind: "tool_request", id: "g", name: "buy_domain", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
      ]),
      registry,
      ledger,
      maxTurns: 5,
    },
    { agent: "lead", goal: "g", budgetUsd: 2 },
  );
  ledger.decideApproval(ledger.listApprovals(first.runId)[0]!.id, "denied");

  await resumeRun(
    { provider: provider([FINISH]), registry, ledger, maxTurns: 5 },
    first.runId,
  );

  // The turn is executed whole: the refused call returns a denial, the rest run.
  assert.deepEqual(calls, ["look"]);
  ledger.close();
});

test("an undecided approval blocks again rather than assuming consent", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const registry = createRegistry([spyTool("buy_domain", "gated", calls)]);

  const first = await runLoop(
    { provider: provider([ASK_GATED]), registry, ledger, maxTurns: 5 },
    { agent: "lead", goal: "g", budgetUsd: 2 },
  );

  // Resume without deciding anything.
  const second = await resumeRun(
    { provider: provider([FINISH]), registry, ledger, maxTurns: 5 },
    first.runId,
  );

  // Silence is not consent. An operator who has not answered is not one who
  // said yes, and the approval design exists so nobody can confuse the two.
  assert.equal(second.status, "blocked");
  assert.equal(second.reason, "approval");
  assert.deepEqual(calls, []);
  // Still resumable: the pending set survives the failed resume.
  const checkpoint = parseCheckpoint(ledger.readCheckpoint(first.runId) ?? "");
  assert.equal(checkpoint.pending?.length, 1);
  ledger.close();
});

test("a second decision cannot flip one the agent already acted on", () => {
  const ledger = openLedger(":memory:");
  const runId = ledger.createRun({
    agent: "lead",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });
  const id = ledger.requestApproval({ runId, tool: "buy_domain", args: {} });

  assert.equal(ledger.decideApproval(id, "approved"), true);
  // Guarded in SQL, not read-then-write: two operators deciding at once must
  // not have the later one silently overwrite the decision already acted on.
  assert.equal(ledger.decideApproval(id, "denied"), false);
  assert.equal(ledger.readApproval(id)?.status, "approved");
  assert.ok(ledger.readApproval(id)?.decided_at);
  ledger.close();
});

test("deciding an unknown approval reports failure rather than throwing", () => {
  const ledger = openLedger(":memory:");
  assert.equal(ledger.decideApproval("no-such-id", "approved"), false);
  assert.equal(ledger.readApproval("no-such-id"), undefined);
  ledger.close();
});

test("the turn count is not spent twice by the approval round trip", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const registry = createRegistry([spyTool("buy_domain", "gated", calls)]);

  const first = await runLoop(
    { provider: provider([ASK_GATED]), registry, ledger, maxTurns: 5 },
    { agent: "lead", goal: "g", budgetUsd: 2 },
  );
  assert.equal(first.turns, 1);

  ledger.decideApproval(ledger.listApprovals(first.runId)[0]!.id, "approved");
  const second = await resumeRun(
    { provider: provider([FINISH]), registry, ledger, maxTurns: 5 },
    first.runId,
  );

  // Settling the held turn is not a new model turn; only the follow-up is.
  assert.equal(second.turns, 2);
  ledger.close();
});
