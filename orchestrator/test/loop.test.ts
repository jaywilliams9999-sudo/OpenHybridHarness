import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { runLoop } from "../src/loop/loop.ts";
import type { TurnEvent, TurnRequest } from "../src/gateway/types.ts";

function echoTool(overrides: Partial<ToolDefinition> = {}): ToolDefinition {
  return {
    name: "echo",
    description: "echo the args back",
    schema: { type: "object" },
    tier: "auto",
    handler: async (args) => ({ echoed: args }),
    ...overrides,
  } as ToolDefinition;
}

function harness(turns: TurnEvent[][], tools: ToolDefinition[] = [echoTool()]) {
  const ledger = openLedger(":memory:");
  const requests: TurnRequest[] = [];
  const provider = createScriptedProvider({
    turns,
    onRequest: (request) => requests.push(request),
  });
  return {
    ledger,
    requests,
    provider,
    registry: createRegistry(tools),
    run: (maxTurns = 10) =>
      runLoop(
        { provider, registry: createRegistry(tools), ledger, maxTurns },
        { agent: "site", goal: "ship the landing page", budgetUsd: 2 },
      ),
  };
}

test("a plain text turn completes the run", async () => {
  const h = harness([
    [
      { kind: "text", text: "all done" },
      { kind: "end", reason: "end_turn" },
    ],
  ]);
  const result = await h.run();

  assert.equal(result.status, "done");
  assert.equal(result.text, "all done");
  assert.equal(result.turns, 1);
  assert.equal(h.ledger.getRun(result.runId)?.status, "done");
  h.ledger.close();
});

test("a tool request is dispatched, recorded with its tier, and fed back", async () => {
  const h = harness([
    [
      { kind: "tool_request", id: "tu1", name: "echo", input: { a: 1 } },
      { kind: "end", reason: "tool_use" },
    ],
    [
      { kind: "text", text: "thanks" },
      { kind: "end", reason: "end_turn" },
    ],
  ]);
  const result = await h.run();

  assert.equal(result.status, "done");
  assert.equal(result.turns, 2);

  const calls = h.ledger.listToolCalls(result.runId);
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.tool, "echo");
  assert.equal(calls[0]?.tier, "auto");
  assert.deepEqual(JSON.parse(calls[0]?.result_json ?? "{}"), {
    echoed: { a: 1 },
  });

  // The result must reach the model, or the next turn is blind.
  const second = h.requests[1];
  assert.ok(second, "no second turn was requested");
  const last = second.transcript.at(-1);
  assert.equal(last?.role, "tool_results");
  h.ledger.close();
});

test("parallel tool requests in one turn are all dispatched", async () => {
  // The Claude API contract: one assistant message may carry several tool_use
  // blocks, and all results go back together. Honoured from the start so the
  // real adapters do not have to retrofit it.
  const h = harness([
    [
      { kind: "tool_request", id: "tu1", name: "echo", input: { n: 1 } },
      { kind: "tool_request", id: "tu2", name: "echo", input: { n: 2 } },
      { kind: "end", reason: "tool_use" },
    ],
    [{ kind: "end", reason: "end_turn" }],
  ]);
  const result = await h.run();

  assert.equal(h.ledger.listToolCalls(result.runId).length, 2);
  const second = h.requests[1];
  const last = second?.transcript.at(-1);
  assert.equal(last?.role, "tool_results");
  if (last?.role === "tool_results") assert.equal(last.results.length, 2);
  h.ledger.close();
});

test("a refusal fails the run without throwing", async () => {
  // stop_reason: "refusal" arrives as HTTP 200. It is an outcome, not a fault.
  const h = harness([[{ kind: "end", reason: "refusal" }]]);
  const result = await h.run();

  assert.equal(result.status, "failed");
  assert.equal(result.reason, "refusal");
  assert.equal(h.ledger.getRun(result.runId)?.status, "failed");
  h.ledger.close();
});

test("a throwing tool is reported back to the model, not fatal", async () => {
  const h = harness(
    [
      [
        { kind: "tool_request", id: "tu1", name: "boom", input: {} },
        { kind: "end", reason: "tool_use" },
      ],
      [
        { kind: "text", text: "recovered" },
        { kind: "end", reason: "end_turn" },
      ],
    ],
    [
      echoTool({
        name: "boom",
        handler: async () => {
          throw new Error("the gpu is down");
        },
      }),
    ],
  );
  const result = await h.run();

  assert.equal(result.status, "done");
  assert.equal(result.text, "recovered");

  const calls = h.ledger.listToolCalls(result.runId);
  assert.match(calls[0]?.result_json ?? "", /the gpu is down/);

  const last = h.requests[1]?.transcript.at(-1);
  if (last?.role === "tool_results") {
    assert.equal(last.results[0]?.isError, true);
  } else {
    assert.fail("the error never reached the model");
  }
  h.ledger.close();
});

test("an unknown tool is an error result, not a crash", async () => {
  const h = harness([
    [
      { kind: "tool_request", id: "tu1", name: "nope", input: {} },
      { kind: "end", reason: "tool_use" },
    ],
    [{ kind: "end", reason: "end_turn" }],
  ]);
  const result = await h.run();
  assert.equal(result.status, "done");
  assert.match(
    h.ledger.listToolCalls(result.runId)[0]?.result_json ?? "",
    /nope/,
  );
  h.ledger.close();
});

test("the turn cap stops a provider that never finishes", async () => {
  // This is marked failed rather than blocked: blocked promises that a resume
  // will pick the work back up, and a status that lies is worse than one that
  // is blunt.
  const looping: TurnEvent[][] = Array.from({ length: 10 }, () => [
    { kind: "tool_request", id: "tu", name: "echo", input: {} },
    { kind: "end", reason: "tool_use" },
  ]);
  const h = harness(looping);
  const result = await h.run(3);

  assert.equal(result.turns, 3);
  assert.equal(result.status, "failed");
  assert.equal(result.reason, "max_turns");
  h.ledger.close();
});

test("thinking text is kept separate from the answer", async () => {
  const h = harness([
    [
      { kind: "thinking", text: "weighing options" },
      { kind: "text", text: "the answer" },
      { kind: "end", reason: "end_turn" },
    ],
  ]);
  const result = await h.run();
  assert.equal(result.text, "the answer");
  assert.equal(result.thinking, "weighing options");
  h.ledger.close();
});

test("arguments that were not JSON are refused back to the model, not run", async () => {
  // The ollama provider keeps unparseable arguments as `{ _raw }`; running a
  // handler on them would fail on whichever field it checks first, telling
  // the model nothing about what actually went wrong.
  let ran = false;
  const h = harness(
    [
      [
        { kind: "tool_request", id: "tu1", name: "echo", input: { _raw: '{"a": "b"c"}' } },
        { kind: "end", reason: "tool_use" },
      ],
      [{ kind: "end", reason: "end_turn" }],
    ],
    [echoTool({ handler: async () => ((ran = true), {}) })],
  );
  const result = await h.run();
  assert.equal(ran, false);
  const last = h.requests[1]?.transcript.at(-1);
  if (last?.role !== "tool_results") assert.fail("the error never reached the model");
  assert.equal(last.results[0]?.isError, true);
  assert.match(JSON.stringify(last.results[0]?.result), /not valid JSON/);
  assert.equal(result.status, "done");
  h.ledger.close();
});

test("a gated call with arguments that were not JSON is refused before asking the operator", async () => {
  const h = harness(
    [
      [
        { kind: "tool_request", id: "tu1", name: "echo", input: { _raw: "{" } },
        { kind: "end", reason: "tool_use" },
      ],
      [{ kind: "end", reason: "end_turn" }],
    ],
    [echoTool({ tier: "gated" })],
  );
  const result = await h.run();
  assert.equal(result.status, "done");
  assert.equal(h.ledger.listApprovals(result.runId).length, 0);
  h.ledger.close();
});
