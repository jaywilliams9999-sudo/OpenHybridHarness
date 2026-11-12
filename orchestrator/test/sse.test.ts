import { test } from "node:test";
import assert from "node:assert/strict";

import { readSse, type SseFrame } from "../src/gpu/sse.ts";

/** Feed the reader an exact sequence of byte chunks. */
async function* chunks(...parts: string[]): AsyncGenerator<Uint8Array> {
  const encoder = new TextEncoder();
  for (const part of parts) yield encoder.encode(part);
}

async function collect(
  source: AsyncIterable<Uint8Array>,
): Promise<SseFrame[]> {
  const frames: SseFrame[] = [];
  for await (const frame of readSse(source)) frames.push(frame);
  return frames;
}

test("parses a single frame", async () => {
  const frames = await collect(
    chunks('event: op.done\ndata: {"op_id":"x"}\n\n'),
  );
  assert.deepEqual(frames, [{ event: "op.done", data: '{"op_id":"x"}' }]);
});

test("parses several frames in one chunk", async () => {
  const frames = await collect(
    chunks(
      'event: op.progress\ndata: {"step":1}\n\n' +
        'event: op.progress\ndata: {"step":2}\n\n',
    ),
  );
  assert.equal(frames.length, 2);
  assert.equal(frames[1]?.data, '{"step":2}');
});

test("reassembles a frame split across chunks", async () => {
  // The case the stub server never produces and the network always will: a
  // frame arriving in pieces, including a split inside the blank-line
  // separator itself.
  const frames = await collect(
    chunks("event: op.d", 'one\ndata: {"op_i', 'd":"x"}\n', "\n"),
  );
  assert.deepEqual(frames, [{ event: "op.done", data: '{"op_id":"x"}' }]);
});

test("a byte-at-a-time stream still yields exactly one frame", async () => {
  const text = 'event: op.done\ndata: {"a":1}\n\n';
  const frames = await collect(chunks(...text.split("")));
  assert.deepEqual(frames, [{ event: "op.done", data: '{"a":1}' }]);
});

test("tolerates CRLF line endings", async () => {
  const frames = await collect(
    chunks('event: op.done\r\ndata: {"a":1}\r\n\r\n'),
  );
  assert.deepEqual(frames, [{ event: "op.done", data: '{"a":1}' }]);
});

test("ignores keepalive comments", async () => {
  const frames = await collect(
    chunks(": keepalive\n\n", 'event: op.done\ndata: {"a":1}\n\n'),
  );
  assert.equal(frames.length, 1);
  assert.equal(frames[0]?.event, "op.done");
});

test("joins multi-line data fields", async () => {
  const frames = await collect(chunks("event: x\ndata: one\ndata: two\n\n"));
  assert.equal(frames[0]?.data, "one\ntwo");
});

test("defaults the event name when only data is sent", async () => {
  const frames = await collect(chunks("data: bare\n\n"));
  assert.equal(frames[0]?.event, "message");
});

test("an unterminated trailing frame is not emitted", async () => {
  // Half a frame is not a frame. Emitting it would hand the client truncated
  // JSON that happens to parse.
  const frames = await collect(
    chunks('event: op.done\ndata: {"a":1}\n\n', "event: op.progress\ndata: {"),
  );
  assert.equal(frames.length, 1);
});

test("a multi-byte character split across chunks survives", async () => {
  // TextDecoder with stream:true is what makes this work; without it the
  // half-character becomes U+FFFD and the JSON downstream is corrupt.
  const encoder = new TextEncoder();
  const whole = encoder.encode('event: x\ndata: {"m":"café"}\n\n');
  const cut = whole.indexOf(0xc3); // the first byte of "é"
  const frames = await collect(
    (async function* () {
      yield whole.subarray(0, cut + 1);
      yield whole.subarray(cut + 1);
    })(),
  );
  assert.equal(frames[0]?.data, '{"m":"café"}');
});
