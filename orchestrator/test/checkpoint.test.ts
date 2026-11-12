/**
 * Checkpoints and resume.
 *
 * The invariant worth protecting is append-only history: a thinking block's
 * signature binds it to the exact conversation prefix that produced it, so a
 * resume that rewrote any earlier turn would 400 on its first request and miss
 * the prompt cache on every one after.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import {
  CheckpointError,
  parseCheckpoint,
  resumeRun,
  runLoop,
  type Checkpoint,
} from "../src/loop/loop.ts";
import type { Provider, TranscriptEntry, TurnEvent } from "../src/gateway/types.ts";

function priced(turns: TurnEvent[][], model = "claude-opus-5-5"): Provider {
  return { ...createScriptedProvider({ turns }), model };
}

const echo: ToolDefinition = {
  name: "echo",
  description: "echo",
  schema: { type: "object" },
  tier: "auto",
  handler: async () => ({ ok: true }),
};

const TOOL_TURN: TurnEvent[] = [
  { kind: "tool_request", id: "t", name: "echo", input: {} },
  { kind: "usage", in: 0, cacheWrite: 0, cacheRead: 0, out: 1_000 },
  { kind: "end", reason: "tool_use" },
];

function readCheckpoint(
  ledger: ReturnType<typeof openLedger>,
  runId: string,
): Checkpoint {
  const blob = ledger.readCheckpoint(runId);
  assert.ok(blob, "expected a checkpoint");
  return parseCheckpoint(blob);
}

// -- writing ------------------------------------------------------------

test("a completed Run leaves a checkpoint describing its final state", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: priced([
        [
          { kind: "text", text: "done" },
          { kind: "usage", in: 10, cacheWrite: 0, cacheRead: 0, out: 5 },
          { kind: "end", reason: "end_turn" },
        ],
      ]),
      registry: createRegistry([]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "ship it", budgetUsd: 1, system: "be brief" },
  );

  const checkpoint = readCheckpoint(ledger, result.runId);
  assert.equal(checkpoint.version, 1);
  assert.equal(checkpoint.agent, "site");
  assert.equal(checkpoint.goal, "ship it");
  assert.equal(checkpoint.system, "be brief");
  assert.equal(checkpoint.budgetUsd, 1);
  assert.equal(checkpoint.turns, 1);
  assert.ok(checkpoint.spentUsd > 0);
  ledger.close();
});

test("a budget-blocked Run is checkpointed before its status is set", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: priced([TOOL_TURN, TOOL_TURN, TOOL_TURN]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.03 },
  );

  assert.equal(result.status, "blocked");
  // A budget stop promises a resumable half-finished task. A `blocked` status with no
  // checkpoint behind it would be exactly the lie that promise exists to avoid.
  const checkpoint = readCheckpoint(ledger, result.runId);
  assert.equal(checkpoint.turns, result.turns);
  assert.equal(checkpoint.spentUsd, result.spentUsd);
  ledger.close();
});

test("the checkpoint is taken after tool results, not before", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: priced([TOOL_TURN, TOOL_TURN]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.025 },
  );

  // A boundary *inside* a cycle would make a resume re-dispatch a tool that
  // already ran -- and these tools write files and occupy a GPU.
  const roles = readCheckpoint(ledger, result.runId).transcript.map(
    (entry) => entry.role,
  );
  assert.equal(roles.at(-1), "tool_results");
  ledger.close();
});

// -- reading ------------------------------------------------------------

test("resume continues from the stored turn count, respecting maxTurns", async () => {
  const ledger = openLedger(":memory:");
  const first = await runLoop(
    {
      provider: priced([TOOL_TURN, TOOL_TURN, TOOL_TURN]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.03 },
  );
  assert.equal(first.status, "blocked");
  const turnsBefore = first.turns;

  // Resume with a bigger budget: the work continues rather than restarting.
  ledger.saveCheckpoint(
    first.runId,
    JSON.stringify({
      ...readCheckpoint(ledger, first.runId),
      budgetUsd: 100,
    }),
  );

  const second = await resumeRun(
    {
      provider: priced([
        [
          { kind: "text", text: "finished" },
          { kind: "usage", in: 1, cacheWrite: 0, cacheRead: 0, out: 1 },
          { kind: "end", reason: "end_turn" },
        ],
      ]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    first.runId,
  );

  assert.equal(second.status, "done");
  assert.equal(second.turns, turnsBefore + 1);
  // Spend accumulates across the resume rather than resetting.
  assert.ok(second.spentUsd > first.spentUsd);
  ledger.close();
});

test("the restored transcript is appended to, never rewritten", async () => {
  const ledger = openLedger(":memory:");
  const first = await runLoop(
    {
      provider: priced([TOOL_TURN, TOOL_TURN]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.025 },
  );

  const before = readCheckpoint(ledger, first.runId).transcript;
  ledger.saveCheckpoint(
    first.runId,
    JSON.stringify({ ...readCheckpoint(ledger, first.runId), budgetUsd: 100 }),
  );

  let seen: readonly TranscriptEntry[] = [];
  await resumeRun(
    {
      provider: {
        name: "scripted",
        model: "claude-opus-5-5",
        chat(request) {
          seen = request.transcript;
          return {
            async *[Symbol.asyncIterator]() {
              await Promise.resolve();
              yield { kind: "end", reason: "end_turn" } as TurnEvent;
            },
          };
        },
      },
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    first.runId,
  );

  // Byte-for-byte prefix equality. Anything else invalidates every thinking
  // signature downstream of the edit.
  assert.deepEqual(seen.slice(0, before.length), before);
  ledger.close();
});

test("a thinking block's opaque payload survives the round trip", async () => {
  const ledger = openLedger(":memory:");
  const opaque = { type: "thinking", thinking: "hm", signature: "sig-xyz" };
  const first = await runLoop(
    {
      provider: priced([
        [
          { kind: "thinking", text: "hm", opaque },
          { kind: "tool_request", id: "t", name: "echo", input: {} },
          { kind: "usage", in: 0, cacheWrite: 0, cacheRead: 0, out: 1_000 },
          { kind: "end", reason: "tool_use" },
        ],
        TOOL_TURN,
      ]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.025 },
  );

  const restored = readCheckpoint(ledger, first.runId).transcript;
  const assistant = restored.find((entry) => entry.role === "assistant");
  assert.ok(assistant?.role === "assistant");
  const thinking = assistant.events.find((event) => event.kind === "thinking");
  // The signature is the whole reason `opaque` exists; JSON must not drop it.
  assert.deepEqual(
    thinking?.kind === "thinking" ? thinking.opaque : undefined,
    opaque,
  );
  ledger.close();
});

// -- refusing to guess --------------------------------------------------

test("resuming a Run with no checkpoint names the problem", async () => {
  const ledger = openLedger(":memory:");
  const runId = ledger.createRun({
    agent: "site",
    goal: "g",
    parentId: null,
    budgetUsd: 1,
  });
  await assert.rejects(
    () =>
      resumeRun(
        {
          provider: priced([]),
          registry: createRegistry([]),
          ledger,
          maxTurns: 5,
        },
        runId,
      ),
    (error: unknown) => {
      assert.ok(error instanceof CheckpointError);
      assert.match(error.message, /no checkpoint/);
      return true;
    },
  );
  ledger.close();
});

test("an unknown checkpoint version is refused, not misread", () => {
  // The Ledger has no migration runner, so a blob from a future Loop must be
  // recognisable as unreadable rather than parsed into a transcript that no
  // longer means what it says.
  assert.throws(() => parseCheckpoint(JSON.stringify({ version: 2 })), {
    name: "CheckpointError",
  });
  assert.throws(() => parseCheckpoint("not json"), { name: "CheckpointError" });
  assert.throws(() => parseCheckpoint(JSON.stringify({ version: 1 })), {
    name: "CheckpointError",
  });
});

test("a resumed Run is marked running again", async () => {
  const ledger = openLedger(":memory:");
  const first = await runLoop(
    {
      provider: priced([TOOL_TURN, TOOL_TURN]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.025 },
  );
  assert.equal(ledger.getRun(first.runId)?.status, "blocked");

  ledger.saveCheckpoint(
    first.runId,
    JSON.stringify({ ...readCheckpoint(ledger, first.runId), budgetUsd: 100 }),
  );
  const second = await resumeRun(
    {
      provider: priced([
        [{ kind: "text", text: "ok" }, { kind: "end", reason: "end_turn" }],
      ]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    first.runId,
  );
  assert.equal(second.status, "done");
  assert.equal(ledger.getRun(first.runId)?.status, "done");
  ledger.close();
});
