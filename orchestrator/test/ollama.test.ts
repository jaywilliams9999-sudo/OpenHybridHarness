/**
 * The `ollama` provider.
 *
 * It is tested against a fake `GpuClient` rather than a fake GPU service, because
 * the thing worth pinning is the translation: Ollama has no content blocks, no
 * call ids, and no cache accounting, and every one of those gaps has to be
 * bridged somewhere. Here is that somewhere.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  assistantMessage,
  createOllamaProvider,
  coerceToSchema,
  normalizeToolCalls,
  stopReasonFor,
  toOllamaMessages,
  toOllamaTools,
} from "../src/gateway/ollama.ts";
import type { GpuClient, TextRequest, TextResult } from "../src/gpu/client.ts";
import type { TranscriptEntry, TurnEvent } from "../src/gateway/types.ts";
import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { runLoop } from "../src/loop/loop.ts";

function textResult(overrides: Partial<TextResult> = {}): TextResult {
  return {
    model: "llama3.1:8b",
    content: "done",
    tool_calls: [],
    prompt_tokens: 100,
    completion_tokens: 10,
    done_reason: "stop",
    num_ctx: 131072,
    ...overrides,
  };
}

/** Records what the provider asked the GPU service for, and replays canned results. */
function fakeGpu(results: TextResult[]) {
  const sent: TextRequest[] = [];
  let turn = 0;
  const client = {
    generateText: async (req: TextRequest) => {
      sent.push(req);
      return results[turn++] ?? textResult();
    },
  } as unknown as GpuClient;
  return { client, sent };
}

async function collect(
  iterable: AsyncIterable<TurnEvent>,
): Promise<TurnEvent[]> {
  const events: TurnEvent[] = [];
  for await (const event of iterable) events.push(event);
  return events;
}

// -- message translation ------------------------------------------------

test("a user turn becomes a user message", () => {
  assert.deepEqual(toOllamaMessages([{ role: "user", text: "hi" }]), [
    { role: "user", content: "hi" },
  ]);
});

test("the system prompt is prepended as a system message", () => {
  const messages = toOllamaMessages([{ role: "user", text: "hi" }], "be brief");
  assert.deepEqual(messages[0], { role: "system", content: "be brief" });
  assert.equal(messages.length, 2);
});

test("text events flatten into one content string", () => {
  // Ollama has no block model, so N streamed deltas become one string. There
  // is no signature to preserve here, which is why flattening is safe.
  assert.deepEqual(
    assistantMessage([
      { kind: "text", text: "a" },
      { kind: "text", text: "b" },
      { kind: "end", reason: "end_turn" },
    ]),
    { role: "assistant", content: "ab" },
  );
});

test("thinking is dropped rather than folded into the content", () => {
  // Concatenating reasoning into `content` would teach the model that its own
  // scratchpad is part of its answer.
  assert.deepEqual(
    assistantMessage([
      { kind: "thinking", text: "weighing it up" },
      { kind: "text", text: "the answer" },
    ]),
    { role: "assistant", content: "the answer" },
  );
});

test("tool requests become Ollama's tool_calls shape", () => {
  assert.deepEqual(
    assistantMessage([
      { kind: "tool_request", id: "local-1-0", name: "echo", input: { a: 1 } },
    ]),
    {
      role: "assistant",
      content: "",
      tool_calls: [{ function: { name: "echo", arguments: { a: 1 } } }],
    },
  );
});

test("tool results become one tool message each, keyed by tool_name", () => {
  // Ollama's tool role has no id field, so the name is the only link back to
  // the call. Anthropic takes one message holding many blocks; this takes many.
  const messages = toOllamaMessages([
    {
      role: "tool_results",
      results: [
        { toolUseId: "local-1-0", name: "gpu_status", result: { free: 10 } },
        { toolUseId: "local-1-1", name: "echo", result: "plain" },
      ],
    },
  ]);

  assert.deepEqual(messages, [
    { role: "tool", tool_name: "gpu_status", content: '{"free":10}' },
    { role: "tool", tool_name: "echo", content: "plain" },
  ]);
});

test("an assistant turn with nothing to say is omitted", () => {
  const messages = toOllamaMessages([
    { role: "user", text: "hi" },
    { role: "assistant", events: [{ kind: "end", reason: "end_turn" }] },
  ]);
  assert.equal(messages.length, 1);
});

test("a full transcript round-trips in order", () => {
  const transcript: TranscriptEntry[] = [
    { role: "user", text: "render it" },
    {
      role: "assistant",
      events: [
        { kind: "text", text: "checking" },
        { kind: "tool_request", id: "local-1-0", name: "gpu_status", input: {} },
        { kind: "end", reason: "tool_use" },
      ],
    },
    {
      role: "tool_results",
      results: [{ toolUseId: "local-1-0", name: "gpu_status", result: { ok: 1 } }],
    },
  ];

  assert.deepEqual(
    toOllamaMessages(transcript).map((m) => m["role"]),
    ["user", "assistant", "tool"],
  );
});

test("tool schemas are renamed to parameters", () => {
  const schema = { type: "object", properties: {} };
  assert.deepEqual(
    toOllamaTools([{ name: "echo", description: "echoes", schema }]),
    [
      {
        type: "function",
        function: { name: "echo", description: "echoes", parameters: schema },
      },
    ],
  );
});

// -- stop reasons -------------------------------------------------------

test("tool calls present means tool_use, whatever done_reason says", () => {
  // Ollama reports done_reason "stop" whether the model finished talking or
  // finished asking; the Loop branches on tool_use to dispatch at all.
  assert.equal(
    stopReasonFor(
      textResult({
        done_reason: "stop",
        tool_calls: [{ function: { name: "echo", arguments: {} } }],
      }),
    ),
    "tool_use",
  );
});

test("done_reason length maps to max_tokens", () => {
  assert.equal(stopReasonFor(textResult({ done_reason: "length" })), "max_tokens");
});

test("an ordinary stop is end_turn", () => {
  assert.equal(stopReasonFor(textResult()), "end_turn");
});

// -- the provider -------------------------------------------------------

test("a turn yields text, usage and end", async () => {
  const { client } = fakeGpu([textResult({ content: "hello" })]);
  const provider = createOllamaProvider({ gpu: client });

  const events = await collect(
    provider.chat({ transcript: [{ role: "user", text: "hi" }], tools: [] }),
  );

  assert.deepEqual(events, [
    { kind: "text", text: "hello" },
    { kind: "usage", in: 100, cacheWrite: 0, cacheRead: 0, out: 10 },
    { kind: "end", reason: "end_turn" },
  ]);
});

test("empty content yields no text event", async () => {
  const { client } = fakeGpu([
    textResult({
      content: "",
      tool_calls: [{ function: { name: "echo", arguments: {} } }],
    }),
  ]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(provider.chat({ transcript: [], tools: [] }));
  assert.equal(events.some((e) => e.kind === "text"), false);
});

test("call ids are synthesized and unique across turns", async () => {
  // Ollama supplies no ids; the Loop keys tool results by one.
  const twoCalls = textResult({
    content: "",
    tool_calls: [
      { function: { name: "a", arguments: {} } },
      { function: { name: "b", arguments: {} } },
    ],
  });
  const { client } = fakeGpu([twoCalls, twoCalls]);
  const provider = createOllamaProvider({ gpu: client });

  const first = await collect(provider.chat({ transcript: [], tools: [] }));
  const second = await collect(provider.chat({ transcript: [], tools: [] }));

  const ids = [...first, ...second]
    .filter((e) => e.kind === "tool_request")
    .map((e) => (e.kind === "tool_request" ? e.id : ""));

  assert.deepEqual(ids, ["local-1-0", "local-1-1", "local-2-0", "local-2-1"]);
  assert.equal(new Set(ids).size, 4);
});

test("a nameless tool call is skipped rather than dispatched", async () => {
  const { client } = fakeGpu([
    textResult({
      content: "",
      tool_calls: [{ function: { arguments: {} } }, { function: { name: "ok" } }],
    }),
  ]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(provider.chat({ transcript: [], tools: [] }));
  const requests = events.filter((e) => e.kind === "tool_request");
  assert.equal(requests.length, 1);
  assert.equal(requests[0]?.kind === "tool_request" ? requests[0].name : "", "ok");
});

test("missing tool arguments become an empty object", async () => {
  const { client } = fakeGpu([
    textResult({ content: "", tool_calls: [{ function: { name: "echo" } }] }),
  ]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(provider.chat({ transcript: [], tools: [] }));
  const request = events.find((e) => e.kind === "tool_request");
  assert.deepEqual(request?.kind === "tool_request" ? request.input : undefined, {});
});

test("cache buckets stay zero — Ollama reports no cache accounting", async () => {
  // prompt_eval_count already excludes a reused prefix, so a cached turn shows
  // a small `in`. Filling cacheRead with a guess would make the Ledger's
  // four-bucket split lie about a provider that cannot fill it.
  const { client } = fakeGpu([textResult({ prompt_tokens: 8 })]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(provider.chat({ transcript: [], tools: [] }));
  assert.deepEqual(
    events.find((e) => e.kind === "usage"),
    { kind: "usage", in: 8, cacheWrite: 0, cacheRead: 0, out: 10 },
  );
});

test("options reach the GPU service only when set", async () => {
  const { client, sent } = fakeGpu([textResult()]);
  const bare = createOllamaProvider({ gpu: client });
  await collect(bare.chat({ transcript: [], tools: [] }));
  assert.deepEqual(Object.keys(sent[0] ?? {}), ["messages"]);

  const tuned = createOllamaProvider({
    gpu: client,
    model: "qwen3:8b",
    numCtx: 8192,
    temperature: 0.1,
    maxTokens: 256,
  });
  await collect(
    tuned.chat({
      transcript: [],
      tools: [{ name: "echo", description: "e", schema: {} }],
    }),
  );
  assert.deepEqual(sent[1], {
    messages: [],
    tools: [
      { type: "function", function: { name: "echo", description: "e", parameters: {} } },
    ],
    model: "qwen3:8b",
    num_ctx: 8192,
    temperature: 0.1,
    max_tokens: 256,
  });
});

test("the model name is learned from the first result, not guessed", async () => {
  const { client } = fakeGpu([textResult({ model: "llama3.1:8b" })]);
  const provider = createOllamaProvider({ gpu: client });

  // Before any turn the registry's choice is genuinely unknown.
  assert.equal(provider.model, "ollama:role=code");
  await collect(provider.chat({ transcript: [], tools: [] }));
  // The Loop reads `model` after the turn, so the Ledger gets the real name
  // from the very first cost row.
  assert.equal(provider.model, "llama3.1:8b");
});

test("an explicit model is reported before the first turn", () => {
  const { client } = fakeGpu([textResult()]);
  const provider = createOllamaProvider({ gpu: client, model: "qwen3:8b" });
  assert.equal(provider.model, "qwen3:8b");
});

// -- through the loop ---------------------------------------------------

test("a local model drives the loop: tool call, result, second turn", async () => {
  const ledger = openLedger(":memory:");
  const tool: ToolDefinition = {
    name: "gpu_status",
    description: "look at the card",
    schema: { type: "object" },
    tier: "auto",
    handler: async () => ({ free_mb: 3000 }),
  };

  const { client, sent } = fakeGpu([
    textResult({
      content: "checking the card",
      tool_calls: [{ function: { name: "gpu_status", arguments: {} } }],
    }),
    textResult({ content: "3000 MB free", tool_calls: [] }),
  ]);

  const result = await runLoop(
    {
      provider: createOllamaProvider({ gpu: client }),
      registry: createRegistry([tool]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "check the card", budgetUsd: 2 },
  );

  assert.equal(result.status, "done");
  assert.equal(result.turns, 2);
  assert.equal(result.text, "3000 MB free");

  // The second request carries the tool result back, which is the whole point
  // of the design rule that otherwise the free provider becomes a text toy.
  const second = sent[1]?.messages ?? [];
  assert.deepEqual(
    second.map((m) => m["role"]),
    ["user", "assistant", "tool"],
  );
  assert.equal(second[2]?.["content"], '{"free_mb":3000}');

  // Local inference is free, and that is recorded as a real row with real
  // token counts against no spend rather than skipped.
  assert.equal(result.spentUsd, 0);
  const costs = ledger.listCosts(result.runId);
  assert.equal(costs.length, 2);
  assert.equal(costs[0]?.model, "llama3.1:8b");
  assert.equal(costs[0]?.provider, "ollama");
  assert.equal(costs[0]?.in_tokens, 100);
  assert.equal(costs[0]?.usd, 0);

  ledger.close();
});

test("tool results are matched to the synthesized ids, not dropped", async () => {
  const ledger = openLedger(":memory:");
  const { client } = fakeGpu([
    textResult({
      content: "",
      tool_calls: [
        { function: { name: "echo", arguments: { n: 1 } } },
        { function: { name: "echo", arguments: { n: 2 } } },
      ],
    }),
    textResult({ content: "both back", tool_calls: [] }),
  ]);

  const result = await runLoop(
    {
      provider: createOllamaProvider({ gpu: client }),
      registry: createRegistry([
        {
          name: "echo",
          description: "e",
          schema: {},
          tier: "auto",
          handler: async (args) => args,
        },
      ]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "echo twice", budgetUsd: 2 },
  );

  assert.equal(result.status, "done");
  // Both parallel calls ran and both results were recorded.
  assert.equal(ledger.listToolCalls(result.runId).length, 2);
  ledger.close();
});

// -- tool-call normalization ---------------------------------------------
//
// A small local model drifts from Ollama's structured `tool_calls`: it hands
// arguments over as a JSON string, or writes the call into its reply as JSON.
// Each drift is normalized here so the Loop sees a call it can dispatch -- or
// validate and refuse back to the model -- rather than prose it ignores.

const KNOWN = new Set(["dispatch", "gpu_status", "publish"]);

test("string arguments are parsed", () => {
  const out = normalizeToolCalls(
    "",
    [{ function: { name: "dispatch", arguments: '{"agent":"scribe","task":"x"}' } }],
    KNOWN,
  );
  assert.deepEqual(out, {
    content: "",
    calls: [{ name: "dispatch", input: { agent: "scribe", task: "x" } }],
  });
});

test("unparseable string arguments are passed through for validation", () => {
  const out = normalizeToolCalls(
    "",
    [{ function: { name: "dispatch", arguments: "{agent: scribe" } }],
    KNOWN,
  );
  assert.deepEqual(out.calls, [{ name: "dispatch", input: { _raw: "{agent: scribe" } }]);
});

test("a bare JSON tool call in content becomes a tool request", () => {
  const out = normalizeToolCalls(
    '  {"name": "dispatch", "parameters": {"agent": "scribe", "task": "write"}}\n',
    [],
    KNOWN,
  );
  assert.deepEqual(out, {
    content: "",
    calls: [{ name: "dispatch", input: { agent: "scribe", task: "write" } }],
  });
});

test("a fenced JSON tool call in content becomes a tool request", () => {
  const out = normalizeToolCalls(
    '```json\n{"name": "gpu_status", "arguments": {}}\n```',
    [],
    KNOWN,
  );
  assert.deepEqual(out, { content: "", calls: [{ name: "gpu_status", input: {} }] });
});

test("JSON naming an unknown tool stays text", () => {
  const content = '{"name": "shell", "arguments": {"cmd": "ls"}}';
  assert.deepEqual(normalizeToolCalls(content, [], KNOWN), { content, calls: [] });
});

test("ordinary text is untouched", () => {
  const content = "The scribe wrote the copy; the page is ready.";
  assert.deepEqual(normalizeToolCalls(content, [], KNOWN), { content, calls: [] });
  // Structured calls alongside text keep both, unchanged.
  const both = normalizeToolCalls(
    content,
    [{ function: { name: "gpu_status", arguments: {} } }],
    KNOWN,
  );
  assert.deepEqual(both, { content, calls: [{ name: "gpu_status", input: {} }] });
});

test("a JSON tool call written as text is dispatched as a tool request", async () => {
  const { client } = fakeGpu([
    textResult({ content: '{"name": "gpu_status", "parameters": {}}', tool_calls: [] }),
  ]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(
    provider.chat({
      transcript: [],
      tools: [{ name: "gpu_status", description: "d", schema: { type: "object" } }],
    }),
  );
  assert.equal(events.some((e) => e.kind === "text"), false);
  const request = events.find((e) => e.kind === "tool_request");
  assert.equal(request?.kind === "tool_request" ? request.name : "", "gpu_status");
  assert.deepEqual(events.at(-1), { kind: "end", reason: "tool_use" });
});

test("string scalars are coerced to the types the schema asks for", () => {
  const schema = {
    type: "object",
    properties: {
      budget_usd: { type: "number" },
      steps: { type: "integer" },
      draft: { type: "boolean" },
      agent: { type: "string" },
    },
  };
  assert.deepEqual(
    coerceToSchema(
      { budget_usd: " 0.5 ", steps: "4", draft: "false", agent: "42" },
      schema,
    ),
    { budget_usd: 0.5, steps: 4, draft: false, agent: "42" },
  );
  // Not a clean number, or not an integer where one is asked: left for the tool.
  assert.deepEqual(coerceToSchema({ budget_usd: "lots", steps: "1.5" }, schema), {
    budget_usd: "lots",
    steps: "1.5",
  });
});

test("a local call's string-typed number reaches the tool as a number", async () => {
  const { client } = fakeGpu([
    textResult({
      content: "",
      tool_calls: [{ function: { name: "dispatch", arguments: { budget_usd: "1" } } }],
    }),
  ]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(
    provider.chat({
      transcript: [],
      tools: [
        {
          name: "dispatch",
          description: "d",
          schema: { type: "object", properties: { budget_usd: { type: "number" } } },
        },
      ],
    }),
  );
  const request = events.find((e) => e.kind === "tool_request");
  assert.deepEqual(request?.kind === "tool_request" ? request.input : undefined, {
    budget_usd: 1,
  });
});

test("a turn whose every call is nameless ends the turn rather than asking for tools", async () => {
  // The raw result had tool_calls, but none is dispatchable: claiming
  // tool_use would send the Loop to run nothing.
  const { client } = fakeGpu([
    textResult({ content: "", tool_calls: [{ function: { arguments: {} } }] }),
  ]);
  const provider = createOllamaProvider({ gpu: client });
  const events = await collect(provider.chat({ transcript: [], tools: [] }));
  assert.equal(events.some((e) => e.kind === "tool_request"), false);
  assert.deepEqual(events.at(-1), { kind: "end", reason: "end_turn" });
});

test("a malformed JSON tool call written as text is passed through for validation", () => {
  // Seen live: an unescaped quote inside a string value. The reply is plainly
  // a call to an offered tool, so it goes to the Loop as one, and the Loop
  // tells the model its arguments were not JSON -- rather than the run
  // ending with that text as its answer.
  const content =
    '{"name": "dispatch", "parameters": {"agent": "scribe", "context": "the bakery"s bread"}}';
  assert.deepEqual(normalizeToolCalls(content, [], KNOWN), {
    content: "",
    calls: [{ name: "dispatch", input: { _raw: content } }],
  });
  // Malformed and naming no offered tool: still text.
  const other = '{"name": "shell", "parameters": {"cmd": "a"b"}}';
  assert.deepEqual(normalizeToolCalls(other, [], KNOWN), { content: other, calls: [] });
});
