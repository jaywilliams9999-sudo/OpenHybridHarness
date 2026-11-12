/**
 * Minimal SSE reader over `fetch`'s ReadableStream. No dependency needed: the
 * GPU service emits exactly `event: NAME\ndata: {json}\n\n` frames (its `sse_format`
 * helper, tested on that side), so a splitter on the blank-line separator is
 * the whole protocol we need.
 */

export interface SseFrame {
  readonly event: string;
  readonly data: string;
}

/**
 * Transport-agnostic on purpose: takes any `AsyncIterable` of bytes, which a
 * `node:http` `IncomingMessage` already is. That keeps this readable without a
 * reader/lock dance and lets the client choose its transport freely.
 */
export async function* readSse(
  body: AsyncIterable<Uint8Array>,
): AsyncGenerator<SseFrame> {
  const decoder = new TextDecoder();
  let buffer = "";

  for await (const chunk of body) {
    buffer += decoder.decode(chunk, { stream: true });

    // Frames are separated by a blank line; tolerate CRLF from any proxy.
    for (;;) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match) break;
      const raw = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const frame = parseFrame(raw);
      if (frame) yield frame;
    }
  }
}

function parseFrame(raw: string): SseFrame | undefined {
  let event = "message";
  const data: string[] = [];

  for (const line of raw.split(/\r?\n/)) {
    if (line === "" || line.startsWith(":")) continue; // keepalive comment
    const colon = line.indexOf(":");
    if (colon === -1) continue;
    const field = line.slice(0, colon);
    const value = line.slice(colon + 1).trimStart();
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }

  if (data.length === 0) return undefined;
  return { event, data: data.join("\n") };
}
