/**
 * A parent waits on a child that is waiting on a human.
 *
 * A dispatched child that hits a `gated` call blocks; its parent must block
 * behind it *without* re-running the dispatch on resume, and receive the
 * child's report as the tool result once the child has finished. The same
 * `pending` mechanism that carries approvals carries the wait.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger, type Ledger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import {
  ChildBlocked,
  parseCheckpoint,
  resumeRun,
  runLoop,
  type LoopDeps,
} from "../src/loop/loop.ts";
import { writeReport } from "../src/loop/report.ts";
import type {
  ToolResultMessage,
  TurnEvent,
  TurnRequest,
} from "../src/gateway/types.ts";

/**
 * Stands in for the real dispatch tool: creates a child row and says the child
 * is waiting on a human. Counting invocations is the point -- a resume that
 * calls this again would dispatch a second child.
 */
function fakeDispatch(
  ledger: Ledger,
  seen: { calls: number; childId?: string },
): ToolDefinition {
  return {
    name: "fake_dispatch",
    description: "fake_dispatch",
    schema: { type: "object" },
    tier: "budgeted",
    exclusive: true,
    handler: async (_args, ctx) => {
      seen.calls++;
      const childId = ledger.createRun({
        agent: "site",
        goal: "render it",
        parentId: ctx.runId,
        budgetUsd: 1,
      });
      ledger.setRunStatus(childId, "blocked");
      seen.childId = childId;
      throw new ChildBlocked(childId);
    },
  };
}

function spyTool(
  name: string,
  tier: ToolDefinition["tier"],
  calls: string[],
  extra: Partial<ToolDefinition> = {},
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
    ...extra,
  };
}

const ASK_DISPATCH: TurnEvent[] = [
  { kind: "tool_request", id: "d1", name: "fake_dispatch", input: { goal: "x" } },
  { kind: "end", reason: "tool_use" },
];

const FINISH: TurnEvent[] = [
  { kind: "text", text: "all done" },
  { kind: "end", reason: "end_turn" },
];

function harness(
  ledger: Ledger,
  turns: TurnEvent[][],
  tools: ToolDefinition[],
): { deps: LoopDeps; requests: TurnRequest[] } {
  const requests: TurnRequest[] = [];
  const provider = {
    ...createScriptedProvider({ turns, onRequest: (r) => requests.push(r) }),
    model: "claude-opus-5-5",
  };
  return {
    deps: { provider, registry: createRegistry(tools), ledger, maxTurns: 5 },
    requests,
  };
}

function lastToolResults(requests: TurnRequest[]): ToolResultMessage[] {
  const transcript = requests[requests.length - 1]?.transcript ?? [];
  const last = transcript[transcript.length - 1];
  assert.equal(last?.role, "tool_results");
  return last.role === "tool_results" ? last.results : [];
}

const SPEC = { agent: "lead", goal: "ship the site", budgetUsd: 2 };

test("a child waiting on a human blocks the parent with reason child", async () => {
  const ledger = openLedger(":memory:");
  const seen: { calls: number; childId?: string } = { calls: 0 };
  const { deps } = harness(ledger, [ASK_DISPATCH, FINISH], [
    fakeDispatch(ledger, seen),
  ]);

  const result = await runLoop(deps, SPEC);

  assert.equal(result.status, "blocked");
  assert.equal(result.reason, "child");
  assert.equal(seen.calls, 1);
  const checkpoint = parseCheckpoint(ledger.readCheckpoint(result.runId) ?? "");
  assert.equal(checkpoint.pending?.[0]?.childRunId, seen.childId);
  // Nothing is answered yet, so nothing joins the transcript.
  assert.equal(
    checkpoint.transcript.filter((e) => e.role === "tool_results").length,
    0,
  );
  ledger.close();
});

test("resuming while the child is still blocked re-blocks and runs nothing", async () => {
  const ledger = openLedger(":memory:");
  const seen: { calls: number; childId?: string } = { calls: 0 };
  const { deps, requests } = harness(ledger, [ASK_DISPATCH, FINISH], [
    fakeDispatch(ledger, seen),
  ]);

  const first = await runLoop(deps, SPEC);
  const before = ledger.readCheckpoint(first.runId);
  const again = await resumeRun(deps, first.runId);

  assert.equal(again.status, "blocked");
  assert.equal(again.reason, "child");
  assert.equal(seen.calls, 1);
  assert.equal(requests.length, 1);
  assert.equal(ledger.readCheckpoint(first.runId), before);
  ledger.close();
});

test("once the child has reported the parent receives the report as the tool result", async () => {
  const ledger = openLedger(":memory:");
  const seen: { calls: number; childId?: string } = { calls: 0 };
  const { deps, requests } = harness(ledger, [ASK_DISPATCH, FINISH], [
    fakeDispatch(ledger, seen),
  ]);

  const first = await runLoop(deps, SPEC);
  const childId = seen.childId ?? "";
  ledger.setRunStatus(childId, "done");
  writeReport(ledger, childId, { status: "done", text: "rendered" });

  const done = await resumeRun(deps, first.runId);

  assert.equal(done.status, "done");
  assert.equal(seen.calls, 1);
  const results = lastToolResults(requests);
  assert.equal(results.length, 1);
  assert.equal(results[0]?.toolUseId, "d1");
  assert.equal((results[0]?.result as { summary?: string }).summary, "rendered");
  assert.equal(results[0]?.isError, undefined);
  // Ruling R1: the call is recorded once, when the report settles it.
  const recorded = ledger
    .listToolCalls(first.runId)
    .filter((c) => c.tool === "fake_dispatch");
  assert.equal(recorded.length, 1);
  ledger.close();
});

test("a failed child's report comes back as an error result", async () => {
  const ledger = openLedger(":memory:");
  const seen: { calls: number; childId?: string } = { calls: 0 };
  const { deps, requests } = harness(ledger, [ASK_DISPATCH, FINISH], [
    fakeDispatch(ledger, seen),
  ]);

  const first = await runLoop(deps, SPEC);
  const childId = seen.childId ?? "";
  ledger.setRunStatus(childId, "failed");
  writeReport(ledger, childId, { status: "failed", text: "", reason: "budget" });

  const done = await resumeRun(deps, first.runId);

  assert.equal(done.status, "done");
  const results = lastToolResults(requests);
  assert.equal(results[0]?.isError, true);
  assert.equal((results[0]?.result as { status?: string }).status, "failed");
  ledger.close();
});

test("a second request beside an exclusive one is answered with an error and never runs", async () => {
  const ledger = openLedger(":memory:");
  const seen: { calls: number; childId?: string } = { calls: 0 };
  const calls: string[] = [];
  const { deps, requests } = harness(
    ledger,
    [
      [
        { kind: "tool_request", id: "d1", name: "fake_dispatch", input: {} },
        { kind: "tool_request", id: "s1", name: "gpu_status", input: {} },
        { kind: "end", reason: "tool_use" },
      ],
      FINISH,
    ],
    [fakeDispatch(ledger, seen), spyTool("gpu_status", "auto", calls)],
  );

  const first = await runLoop(deps, SPEC);
  assert.equal(first.status, "blocked");
  assert.deepEqual(calls, []);

  const checkpoint = parseCheckpoint(ledger.readCheckpoint(first.runId) ?? "");
  const stored = checkpoint.pending?.find((c) => c.id === "s1");
  assert.equal(stored?.isError, true);
  assert.match((stored?.result as { error: string }).error, /runs alone/);

  const childId = seen.childId ?? "";
  ledger.setRunStatus(childId, "done");
  writeReport(ledger, childId, { status: "done", text: "rendered" });
  const done = await resumeRun(deps, first.runId);

  assert.equal(done.status, "done");
  assert.deepEqual(calls, []);
  const results = lastToolResults(requests);
  assert.deepEqual(
    results.map((r) => r.toolUseId),
    ["d1", "s1"],
  );
  const refused = results.find((r) => r.toolUseId === "s1");
  assert.equal(refused?.isError, true);
  assert.match((refused?.result as { error: string }).error, /runs alone/);
  // Recorded once, when it was refused -- not again when the turn settled.
  assert.equal(
    ledger.listToolCalls(first.runId).filter((c) => c.tool === "gpu_status")
      .length,
    1,
  );
  ledger.close();
});

test("an exclusive turn answers a gated sibling without writing an approval", async () => {
  const ledger = openLedger(":memory:");
  const seen: { calls: number; childId?: string } = { calls: 0 };
  const calls: string[] = [];
  const { deps } = harness(
    ledger,
    [
      [
        { kind: "tool_request", id: "d1", name: "fake_dispatch", input: {} },
        { kind: "tool_request", id: "g1", name: "buy_domain", input: {} },
        { kind: "end", reason: "tool_use" },
      ],
      FINISH,
    ],
    [fakeDispatch(ledger, seen), spyTool("buy_domain", "gated", calls)],
  );

  const first = await runLoop(deps, SPEC);

  assert.equal(first.status, "blocked");
  assert.equal(first.reason, "child");
  assert.equal(seen.calls, 1);
  assert.deepEqual(calls, []);
  assert.equal(ledger.listApprovals(first.runId).length, 0);
  ledger.close();
});

test("a gated call that fails validation writes no approval and does not block", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const { deps, requests } = harness(
    ledger,
    [
      [
        { kind: "tool_request", id: "g1", name: "publish", input: { path: "x" } },
        { kind: "end", reason: "tool_use" },
      ],
      FINISH,
    ],
    [
      spyTool("publish", "gated", calls, {
        validate: () => "not an artifact of this Run",
      }),
    ],
  );

  const result = await runLoop(deps, SPEC);

  assert.equal(result.status, "done");
  assert.equal(ledger.listApprovals(result.runId).length, 0);
  assert.deepEqual(calls, []);
  const results = lastToolResults(requests);
  assert.equal(results[0]?.isError, true);
  assert.equal(
    (results[0]?.result as { error: string }).error,
    "not an artifact of this Run",
  );
  assert.equal(ledger.listToolCalls(result.runId).length, 1);
  ledger.close();
});

test("a failed validation beside a valid gated call blocks only on the valid one", async () => {
  const ledger = openLedger(":memory:");
  const calls: string[] = [];
  const { deps, requests } = harness(
    ledger,
    [
      [
        { kind: "tool_request", id: "bad", name: "publish", input: { ok: false } },
        { kind: "tool_request", id: "good", name: "publish", input: { ok: true } },
        { kind: "end", reason: "tool_use" },
      ],
      FINISH,
    ],
    [
      spyTool("publish", "gated", calls, {
        validate: (args) => (args.ok ? undefined : "not an artifact of this Run"),
      }),
    ],
  );

  const first = await runLoop(deps, SPEC);
  assert.equal(first.status, "blocked");
  assert.equal(first.reason, "approval");
  const approvals = ledger.listApprovals(first.runId);
  assert.equal(approvals.length, 1);

  ledger.decideApproval(approvals[0]?.id ?? "", "approved");
  const done = await resumeRun(deps, first.runId);

  assert.equal(done.status, "done");
  assert.deepEqual(calls, ["publish"]);
  const results = lastToolResults(requests);
  assert.equal(results.find((r) => r.toolUseId === "bad")?.isError, true);
  assert.equal(results.find((r) => r.toolUseId === "good")?.isError, undefined);
  // The refusal was recorded when produced; the approved call when it ran.
  assert.equal(ledger.listToolCalls(first.runId).length, 2);
  ledger.close();
});
