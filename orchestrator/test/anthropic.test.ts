/**
 * Tests for the `anthropic` adapter.
 *
 * Everything here runs against recorded SSE frames rather than a fake server.
 * The distinction matters: a fake server would be our own guess about the API
 * asserted against our own translation of it, which proves only that the two
 * halves of one assumption agree. Frames copied from the published event
 * vocabulary at least pin the translation to a documented shape.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  assistantContent,
  buildBody,
  createAnthropicProvider,
  GatewayError,
  toMessages,
  translateStream,
} from "../src/gateway/anthropic.ts";
import type { TranscriptEntry, TurnEvent } from "../src/gateway/types.ts";
import type { HttpResponse } from "../src/gpu/http.ts";

// -- helpers ------------------------------------------------------------

type Frame = { event: string; data: string };

function frame(event: string, payload: unknown): Frame {
  return { event, data: JSON.stringify(payload) };
}

async function* frames(list: Frame[]): AsyncGenerator<Frame> {
  for (const item of list) {
    await Promise.resolve();
    yield item;
  }
}

async function drain(list: Frame[]): Promise<TurnEvent[]> {
  const out: TurnEvent[] = [];
  for await (const event of translateStream(frames(list))) out.push(event);
  return out;
}

/** A complete text-only turn. */
const TEXT_TURN: Frame[] = [
  frame("message_start", {
    message: {
      usage: {
        input_tokens: 12,
        cache_creation_input_tokens: 500,
        cache_read_input_tokens: 4000,
      },
    },
  }),
  frame("content_block_start", {
    index: 0,
    content_block: { type: "text", text: "" },
  }),
  frame("content_block_delta", {
    index: 0,
    delta: { type: "text_delta", text: "Hel" },
  }),
  frame("content_block_delta", {
    index: 0,
    delta: { type: "text_delta", text: "lo" },
  }),
  frame("content_block_stop", { index: 0 }),
  frame("message_delta", {
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 7 },
  }),
  frame("message_stop", {}),
];

// -- stream translation -------------------------------------------------

test("text deltas stream through, and usage lands in four buckets", async () => {
  const events = await drain(TEXT_TURN);

  assert.deepEqual(
    events.filter((e) => e.kind === "text"),
    [
      { kind: "text", text: "Hel" },
      { kind: "text", text: "lo" },
    ],
  );

  const usage = events.find((e) => e.kind === "usage");
  assert.deepEqual(usage, {
    kind: "usage",
    in: 12,
    cacheWrite: 500,
    cacheRead: 4000,
    out: 7,
  });

  assert.deepEqual(events.at(-1), { kind: "end", reason: "end_turn" });
});

test("a thinking block is emitted once, carrying its signature in opaque", async () => {
  const events = await drain([
    frame("message_start", { message: { usage: {} } }),
    frame("content_block_start", {
      index: 0,
      content_block: { type: "thinking", thinking: "" },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "thinking_delta", thinking: "weighing " },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "thinking_delta", thinking: "options" },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "signature_delta", signature: "sig-abc" },
    }),
    frame("content_block_stop", { index: 0 }),
    frame("message_delta", {
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 3 },
    }),
  ]);

  const thinking = events.filter((e) => e.kind === "thinking");
  // One event per block, not per delta: a signature only exists once the
  // block is closed, so a streamed thinking event could never carry it.
  assert.equal(thinking.length, 1);
  assert.deepEqual(thinking[0], {
    kind: "thinking",
    text: "weighing options",
    opaque: {
      type: "thinking",
      thinking: "weighing options",
      signature: "sig-abc",
    },
  });
});

test("a thinking block with text omitted still carries its signature", async () => {
  // Opus 5.5's default display returns an empty `thinking` field. A serializer
  // that skipped empty blocks would drop the signature and 400 on replay.
  const events = await drain([
    frame("content_block_start", {
      index: 0,
      content_block: { type: "thinking", thinking: "" },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "signature_delta", signature: "sig-only" },
    }),
    frame("content_block_stop", { index: 0 }),
  ]);

  assert.equal(events.length, 1);
  const [event] = events;
  assert.equal(event?.kind, "thinking");
  assert.equal(event?.kind === "thinking" ? event.text : "x", "");
  assert.deepEqual(event?.kind === "thinking" ? event.opaque : undefined, {
    type: "thinking",
    thinking: "",
    signature: "sig-only",
  });
});

test("an unrecognised block type is passed through whole, never inspected", async () => {
  const block = { type: "redacted_thinking", data: "EncRypTedBlob==" };
  const events = await drain([
    frame("content_block_start", { index: 0, content_block: block }),
    frame("content_block_stop", { index: 0 }),
  ]);

  assert.deepEqual(events, [{ kind: "thinking", text: "", opaque: block }]);
});

test("tool input is accumulated across deltas and parsed at block stop", async () => {
  const events = await drain([
    frame("content_block_start", {
      index: 0,
      content_block: { type: "tool_use", id: "toolu_1", name: "gpu_status" },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "input_json_delta", partial_json: '{"deta' },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "input_json_delta", partial_json: 'il":true}' },
    }),
    frame("content_block_stop", { index: 0 }),
    frame("message_delta", { delta: { stop_reason: "tool_use" }, usage: {} }),
  ]);

  assert.deepEqual(events[0], {
    kind: "tool_request",
    id: "toolu_1",
    name: "gpu_status",
    input: { detail: true },
  });
  assert.deepEqual(events.at(-1), { kind: "end", reason: "tool_use" });
});

test("a tool block with no input at all becomes an empty object", async () => {
  const events = await drain([
    frame("content_block_start", {
      index: 0,
      content_block: { type: "tool_use", id: "t", name: "gpu_status" },
    }),
    frame("content_block_stop", { index: 0 }),
  ]);
  assert.deepEqual(events[0], {
    kind: "tool_request",
    id: "t",
    name: "gpu_status",
    input: {},
  });
});

test("truncated tool JSON is handed back, not thrown", async () => {
  const events = await drain([
    frame("content_block_start", {
      index: 0,
      content_block: { type: "tool_use", id: "t", name: "echo" },
    }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "input_json_delta", partial_json: '{"a":' },
    }),
    frame("content_block_stop", { index: 0 }),
  ]);

  const [event] = events;
  assert.equal(event?.kind, "tool_request");
  assert.deepEqual(event?.kind === "tool_request" ? event.input : undefined, {
    _malformed_json: '{"a":',
  });
});

test("interleaved blocks keep their order", async () => {
  const events = await drain([
    frame("content_block_start", { index: 0, content_block: { type: "text" } }),
    frame("content_block_delta", {
      index: 0,
      delta: { type: "text_delta", text: "before" },
    }),
    frame("content_block_stop", { index: 0 }),
    frame("content_block_start", {
      index: 1,
      content_block: { type: "tool_use", id: "t1", name: "echo" },
    }),
    frame("content_block_stop", { index: 1 }),
    frame("content_block_start", { index: 2, content_block: { type: "text" } }),
    frame("content_block_delta", {
      index: 2,
      delta: { type: "text_delta", text: "after" },
    }),
    frame("content_block_stop", { index: 2 }),
  ]);

  assert.deepEqual(
    events.map((e) => e.kind),
    ["text", "tool_request", "text"],
  );
});

test("stop_sequence and unknown reasons degrade to end_turn", async () => {
  for (const raw of ["stop_sequence", "something_new", null]) {
    const events = await drain([
      frame("message_delta", { delta: { stop_reason: raw }, usage: {} }),
    ]);
    assert.deepEqual(events.at(-1), { kind: "end", reason: "end_turn" });
  }
});

test("refusal is a terminal reason, not an exception", async () => {
  const events = await drain([
    frame("message_delta", {
      delta: { stop_reason: "refusal" },
      usage: { output_tokens: 1 },
    }),
  ]);
  assert.deepEqual(events.at(-1), { kind: "end", reason: "refusal" });
});

test("a mid-stream error event throws a GatewayError", async () => {
  await assert.rejects(
    () => drain([frame("error", { error: { type: "overloaded_error", message: "busy" } })]),
    (error: unknown) => {
      assert.ok(error instanceof GatewayError);
      assert.equal(error.type, "overloaded_error");
      assert.equal(error.retryable, true);
      return true;
    },
  );
});

test("an unparseable frame is skipped rather than failing the turn", async () => {
  const events: TurnEvent[] = [];
  const bad: Frame[] = [
    { event: "content_block_delta", data: "not json at all" },
    ...TEXT_TURN,
  ];
  for await (const event of translateStream(frames(bad))) events.push(event);
  assert.ok(events.some((e) => e.kind === "text"));
});

// -- request translation ------------------------------------------------

test("adjacent text events coalesce back into one block", () => {
  // The round-trip invariant: we stream N deltas but must replay 1 block, or
  // the conversation prefix changes and every thinking signature breaks.
  const blocks = assistantContent([
    { kind: "text", text: "a" },
    { kind: "text", text: "b" },
    { kind: "text", text: "c" },
    { kind: "end", reason: "end_turn" },
  ]);
  assert.deepEqual(blocks, [{ type: "text", text: "abc" }]);
});

test("a thinking event replays its opaque block verbatim", () => {
  const opaque = { type: "thinking", thinking: "hm", signature: "sig" };
  const blocks = assistantContent([
    { kind: "thinking", text: "hm", opaque },
    { kind: "text", text: "answer" },
  ]);
  assert.equal(blocks[0], opaque, "the same object, not a reconstruction");
  assert.deepEqual(blocks[1], { type: "text", text: "answer" });
});

test("a thinking event with no opaque block is dropped, not reconstructed", () => {
  // Rebuilding {type:"thinking", thinking} without a signature is a 400, so a
  // provider that gave us no block simply has no thinking to replay.
  const blocks = assistantContent([
    { kind: "thinking", text: "from a local model" },
    { kind: "text", text: "answer" },
  ]);
  assert.deepEqual(blocks, [{ type: "text", text: "answer" }]);
});

test("text, thinking and tool_use keep their relative order on replay", () => {
  const opaque = { type: "thinking", thinking: "t", signature: "s" };
  const blocks = assistantContent([
    { kind: "text", text: "one" },
    { kind: "thinking", text: "t", opaque },
    { kind: "text", text: "two" },
    { kind: "tool_request", id: "t1", name: "echo", input: { a: 1 } },
  ]);
  assert.deepEqual(blocks.map((b) => b["type"]), [
    "text",
    "thinking",
    "text",
    "tool_use",
  ]);
});

test("tool results become a user message, the API having no tool role", () => {
  const transcript: TranscriptEntry[] = [
    { role: "user", text: "go" },
    {
      role: "assistant",
      events: [
        { kind: "tool_request", id: "t1", name: "echo", input: { a: 1 } },
        { kind: "end", reason: "tool_use" },
      ],
    },
    {
      role: "tool_results",
      results: [
        { toolUseId: "t1", name: "echo", result: { ok: true } },
        { toolUseId: "t2", name: "boom", result: { error: "x" }, isError: true },
      ],
    },
  ];

  const messages = toMessages(transcript);
  assert.deepEqual(messages.map((m) => m.role), ["user", "assistant", "user"]);
  assert.deepEqual(messages[2]?.content, [
    {
      type: "tool_result",
      tool_use_id: "t1",
      content: '{"ok":true}',
    },
    {
      type: "tool_result",
      tool_use_id: "t2",
      content: '{"error":"x"}',
      is_error: true,
    },
  ]);
});

test("a string tool result is sent as-is, not double-encoded", () => {
  const messages = toMessages([
    { role: "tool_results", results: [{ toolUseId: "t", name: "e", result: "plain" }] },
  ]);
  assert.equal(
    (messages[0]?.content[0] as Record<string, unknown>)["content"],
    "plain",
  );
});

test("an assistant turn with nothing replayable is omitted", () => {
  // A refusal produces usage and end and no content. An empty content array
  // is a 400, so the message must not be sent at all.
  const messages = toMessages([
    { role: "user", text: "go" },
    { role: "assistant", events: [{ kind: "end", reason: "refusal" }] },
  ]);
  assert.deepEqual(messages.map((m) => m.role), ["user"]);
});

// -- request body -------------------------------------------------------

test("the body carries adaptive thinking, streaming and a cache breakpoint", () => {
  const body = buildBody(
    { transcript: [{ role: "user", text: "hi" }], tools: [] },
    { apiKey: "k", model: "claude-opus-5-5" },
  );

  assert.equal(body["model"], "claude-opus-5-5");
  assert.equal(body["stream"], true);
  assert.deepEqual(body["thinking"], { type: "adaptive", display: "summarized" });
  // Top-level: the API moves the breakpoint to the last cacheable block, so a
  // growing transcript needs no per-turn bookkeeping.
  assert.deepEqual(body["cache_control"], { type: "ephemeral" });
  assert.equal(body["output_config"], undefined);
  assert.equal(body["tools"], undefined);
  assert.equal(body["system"], undefined);
});

test("budget_tokens is never sent — it is a 400 on every current model", () => {
  const body = buildBody(
    { transcript: [{ role: "user", text: "hi" }], tools: [] },
    { apiKey: "k", model: "claude-opus-5-5", effort: "high" },
  );
  assert.equal(JSON.stringify(body).includes("budget_tokens"), false);
  assert.deepEqual(body["output_config"], { effort: "high" });
});

test("a 1h cache ttl is sent explicitly; 5m is left implicit", () => {
  const oneHour = buildBody(
    { transcript: [], tools: [] },
    { apiKey: "k", model: "claude-opus-5-5", cacheTtl: "1h" },
  );
  assert.deepEqual(oneHour["cache_control"], { type: "ephemeral", ttl: "1h" });
});

test("tool schemas are renamed to input_schema", () => {
  const schema = { type: "object", properties: { a: { type: "string" } } };
  const body = buildBody(
    {
      transcript: [],
      tools: [{ name: "echo", description: "echoes", schema }],
    },
    { apiKey: "k", model: "claude-opus-5-5" },
  );
  assert.deepEqual(body["tools"], [
    { name: "echo", description: "echoes", input_schema: schema },
  ]);
});

// -- transport ----------------------------------------------------------

function fakeResponse(status: number, body: string, headers = {}): HttpResponse {
  async function* chunks() {
    yield Buffer.from(body);
  }
  const stream = chunks() as unknown as HttpResponse["stream"];
  return { status, contentType: "application/json", headers, stream, abort: () => {} };
}

function sseResponse(list: Frame[]): HttpResponse {
  const text = list
    .map((f) => `event: ${f.event}\ndata: ${f.data}\n\n`)
    .join("");
  const response = fakeResponse(200, text);
  return { ...response, contentType: "text/event-stream" };
}

test("a 402 surfaces as a credits spending wall with a local-fallback remedy", async () => {
  const provider = createAnthropicProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    send: async () =>
      fakeResponse(
        402,
        JSON.stringify({
          error: { type: "billing_error", message: "credit balance too low" },
        }),
      ),
  });

  await assert.rejects(
    async () => {
      for await (const _ of provider.chat({ transcript: [], tools: [] })) void _;
    },
    (error: unknown) => {
      assert.ok(error instanceof GatewayError);
      assert.equal(error.spendBlocked, true);
      assert.equal(error.spendLimit, "credits");
      assert.equal(error.retryable, false);
      assert.match(error.remedy, /llama3\.1:8b|local provider/);
      return true;
    },
  );
});

test("a 401 is not retried and names the key as the cause", async () => {
  let calls = 0;
  const provider = createAnthropicProvider({
    apiKey: "bad",
    model: "claude-opus-5-5",
    send: async () => {
      calls++;
      return fakeResponse(
        401,
        JSON.stringify({ error: { type: "authentication_error", message: "no" } }),
      );
    },
  });

  await assert.rejects(async () => {
    for await (const _ of provider.chat({ transcript: [], tools: [] })) void _;
  });
  assert.equal(calls, 1, "a bad key must not be retried");
});

test("a 529 is retried, then succeeds", async () => {
  let calls = 0;
  const slept: number[] = [];
  const provider = createAnthropicProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    sleep: async (ms) => void slept.push(ms),
    send: async () => {
      calls++;
      if (calls === 1) {
        return fakeResponse(
          529,
          JSON.stringify({ error: { type: "overloaded_error", message: "busy" } }),
        );
      }
      return sseResponse(TEXT_TURN);
    },
  });

  const events: TurnEvent[] = [];
  for await (const event of provider.chat({ transcript: [], tools: [] })) {
    events.push(event);
  }

  assert.equal(calls, 2);
  assert.deepEqual(slept, [500]);
  assert.ok(events.some((e) => e.kind === "text"));
});

test("retry-after is honoured over the exponential floor", async () => {
  let calls = 0;
  const slept: number[] = [];
  const provider = createAnthropicProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    sleep: async (ms) => void slept.push(ms),
    send: async () => {
      calls++;
      if (calls === 1) {
        return {
          ...fakeResponse(
            429,
            JSON.stringify({ error: { type: "rate_limit_error", message: "slow" } }),
          ),
          headers: { "retry-after": "3" },
        };
      }
      return sseResponse(TEXT_TURN);
    },
  });

  for await (const _ of provider.chat({ transcript: [], tools: [] })) void _;
  assert.deepEqual(slept, [3000]);
});

test("retries are bounded and the last error is thrown", async () => {
  let calls = 0;
  const provider = createAnthropicProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    maxRetries: 2,
    sleep: async () => {},
    send: async () => {
      calls++;
      return fakeResponse(
        529,
        JSON.stringify({ error: { type: "overloaded_error", message: "busy" } }),
      );
    },
  });

  await assert.rejects(async () => {
    for await (const _ of provider.chat({ transcript: [], tools: [] })) void _;
  });
  assert.equal(calls, 3, "one attempt plus two retries");
});

test("the provider reports its name and model for the cost row", () => {
  const provider = createAnthropicProvider({
    apiKey: "k",
    model: "claude-haiku-5-5",
  });
  assert.equal(provider.name, "anthropic");
  assert.equal(provider.model, "claude-haiku-5-5");
});

test("onCost fires once per turn with the priced usage", async () => {
  const seen: Array<{ usd: number }> = [];
  const provider = createAnthropicProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    onCost: (cost) => seen.push(cost),
    send: async () => sseResponse(TEXT_TURN),
  });

  for await (const _ of provider.chat({ transcript: [], tools: [] })) void _;

  assert.equal(seen.length, 1);
  // 12 fresh in, 500 written, 4000 read, 7 out, on Opus 5.5.
  const expected =
    (12 * 4 + 500 * 4 * 1.25 + 4000 * 0.2 + 7 * 20) / 1_000_000;
  assert.ok(Math.abs((seen[0]?.usd ?? 0) - expected) < 1e-12);
});
