/**
 * The `anthropic` provider — the Messages API over raw HTTPS.
 *
 * **Why not the official SDK.** The orchestrator has zero runtime dependencies by
 * design: `node:sqlite`, `node:test`, `node:http`. That rule is
 * what lets the harness be audited in one sitting and deployed by copying a
 * directory, and one adapter is not worth breaking it for. The cost is this
 * file; the saving is that `src/gpu/sse.ts` already parses the exact frame
 * shape Anthropic emits, so the stream reader was free.
 *
 * **Why credits, not OAuth.** Claude Max includes monthly Claude Platform
 * credits spent through an ordinary API key. Routing a
 * third-party harness through Free/Pro/Max *login* credentials is prohibited;
 * an API key against those same credits is not. This adapter therefore
 * authenticates with `x-api-key` and nothing else, which is also why it can
 * never accidentally become a claude.ai client.
 *
 * **What the contract got right.** `gateway/types.ts` was shaped from this API's
 * streaming vocabulary, so this file is a translator and not a redesign. The
 * one place the contract fell short is `thinking`, which could not carry the
 * block signature that Opus 5.5 requires on replay — see the `opaque` field.
 */

import { send as defaultSend, readBody, type HttpResponse } from "../gpu/http.ts";
import { readSse } from "../gpu/sse.ts";
import { costUsd, type CacheTtl } from "./pricing.ts";
import type {
  Provider,
  StopReason,
  TranscriptEntry,
  TurnEvent,
  TurnRequest,
} from "./types.ts";

export const ANTHROPIC_BASE_URL = "https://api.anthropic.com";
const API_VERSION = "2023-06-01";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

/**
 * Which kind of spending wall a request hit.
 *
 * `credits` — the monthly Claude Platform balance is gone. HTTP 402,
 * `billing_error`, unambiguous and structural.
 *
 * `configured` — an organization or workspace spend limit set in the Console
 * refuses the call. Observed 2026-10-15 with a $1/month limit in force:
 *
 *     HTTP 400 invalid_request_error
 *     "You have reached your specified API usage limits.
 *      You will regain access on 2026-11-01 at 00:00 UTC."
 *
 * Note the status. There is no distinguishing `type` — it is
 * `invalid_request_error`, exactly like a malformed body — and
 * `x-should-retry: false` is true of both. The message is the only signal
 * the API gives, so this is the one place the adapter matches on prose.
 */
export type SpendLimit = "credits" | "configured";

/**
 * Does this failure mean "you cannot spend", rather than "your request is
 * wrong"?
 *
 * Matching an error message is normally a bad trade, and it is defensible
 * here only because the asymmetry runs the right way. A match that succeeds
 * degrades the Run to free local inference instead of killing it. A match
 * that fails — because the wording changed — leaves exactly the behaviour
 * that exists without this function. The downside of being wrong is the
 * status quo, so the only risk worth guarding is a *false positive*, and two
 * independent clauses are required to avoid it: a real malformed request does
 * not tell you when you will regain access.
 */
export function spendLimitOf(status: number, message: string): SpendLimit | undefined {
  if (status === 402) return "credits";
  if (
    status === 400 &&
    /\b(usage|spend|spending)\s+limits?\b/i.test(message) &&
    /\b(regain access|will reset|next billing)\b/i.test(message)
  ) {
    return "configured";
  }
  return undefined;
}

export class GatewayError extends Error {
  readonly status: number;
  /** Anthropic's own error type, e.g. `rate_limit_error`. */
  readonly type: string;
  readonly retryable: boolean;
  /**
   * The degradation trigger: this request hit a spending wall.
   *
   * Set for both kinds, because the orchestrator's answer to either is the same —
   * fall back to the local model. `spendLimit` says which, for the operator's
   * benefit, since the remedies differ completely.
   *
   * Named `spendBlocked` rather than `creditsExhausted` because it stopped
   * being only about credits the day a Console spend limit was observed doing
   * the same thing through a different status code.
   */
  readonly spendBlocked: boolean;
  /** Which wall, when one was hit. */
  readonly spendLimit?: SpendLimit;
  readonly remedy: string;

  constructor(
    message: string,
    options: {
      status: number;
      type: string;
      retryable: boolean;
      spendLimit?: SpendLimit;
      remedy: string;
    },
  ) {
    super(message);
    this.name = "GatewayError";
    this.status = options.status;
    this.type = options.type;
    this.retryable = options.retryable;
    this.spendBlocked = options.spendLimit !== undefined;
    if (options.spendLimit !== undefined) this.spendLimit = options.spendLimit;
    this.remedy = options.remedy;
  }
}

export interface AnthropicOptions {
  readonly apiKey: string;
  /** e.g. `claude-opus-5-5`. Priced by `pricing.ts`, recorded in `costs`. */
  readonly model: string;
  readonly maxTokens?: number;
  /**
   * Thinking depth. Opus 5.5 cannot disable thinking and bills it as output,
   * so this — not context size — is the cost lever on a Claude Run. Omitted
   * means the model's own default (`medium` on Opus 5.5, `high` on Sonnet 5.5).
   */
  readonly effort?: Effort;
  /**
   * Prompt-cache lifetime. Defaults to `5m`, the rate that needs no beta
   * header. `1h` costs 2x the base input rate to write instead of 1.25x and
   * survives human-speed gaps between turns — worth it when an operator walks
   * away mid-Run, which is the normal case here.
   */
  readonly cacheTtl?: CacheTtl;
  readonly baseUrl?: string;
  readonly maxRetries?: number;
  /** Injected for tests; defaults to the shared `node:http`/`node:https` helper. */
  readonly send?: typeof defaultSend;
  readonly sleep?: (ms: number) => Promise<void>;
  /** Called with each turn's cost in USD, after `usage` is known. */
  readonly onCost?: (cost: { usd: number; usage: UsageTotals }) => void;
}

export interface UsageTotals {
  in: number;
  cacheWrite: number;
  cacheRead: number;
  out: number;
}

// -- request translation ------------------------------------------------

interface AnthropicBlock {
  readonly type: string;
  readonly [key: string]: unknown;
}

interface AnthropicMessage {
  readonly role: "user" | "assistant";
  readonly content: AnthropicBlock[];
}

/**
 * Rebuild one assistant turn's content blocks from the events we emitted.
 *
 * Two rules are load-bearing, both for the same reason: a thinking block's
 * signature binds it to the exact conversation prefix that produced it, so the
 * replayed message must have the same *shape* the API returned, not merely the
 * same text.
 *
 * 1. Adjacent `text` events are coalesced. We stream text deltas one event at
 *    a time for a responsive operator view, so a single API text block arrives
 *    here as dozens of events; emitting one block each would turn one block
 *    into dozens and change the prefix.
 * 2. A `thinking` event is replayed from `opaque` and dropped entirely if it
 *    has none. Reconstructing `{type:"thinking", thinking: text}` without the
 *    signature is a 400, so a provider that gave us no opaque block (the
 *    scripted double, a local model) simply has no thinking to replay.
 */
export function assistantContent(events: readonly TurnEvent[]): AnthropicBlock[] {
  const blocks: AnthropicBlock[] = [];
  let pending = "";

  const flush = () => {
    if (pending !== "") {
      blocks.push({ type: "text", text: pending });
      pending = "";
    }
  };

  for (const event of events) {
    switch (event.kind) {
      case "text":
        pending += event.text;
        break;
      case "thinking":
        flush();
        if (event.opaque !== undefined) blocks.push(event.opaque as AnthropicBlock);
        break;
      case "tool_request":
        flush();
        blocks.push({
          type: "tool_use",
          id: event.id,
          name: event.name,
          input: event.input ?? {},
        });
        break;
      case "usage":
      case "end":
        break;
    }
  }
  flush();
  return blocks;
}

export function toMessages(
  transcript: readonly TranscriptEntry[],
): AnthropicMessage[] {
  const messages: AnthropicMessage[] = [];

  for (const entry of transcript) {
    if (entry.role === "user") {
      messages.push({
        role: "user",
        content: [{ type: "text", text: entry.text }],
      });
      continue;
    }

    if (entry.role === "assistant") {
      const content = assistantContent(entry.events);
      // An assistant message with no content is not representable; a turn that
      // produced only usage/end events (a refusal, a cap trip) has nothing to
      // replay and is better omitted than sent as an empty array, which 400s.
      if (content.length > 0) messages.push({ role: "assistant", content });
      continue;
    }

    // Tool results go back as a *user* message — the API has no tool role.
    messages.push({
      role: "user",
      content: entry.results.map((result) => ({
        type: "tool_result",
        tool_use_id: result.toolUseId,
        content:
          typeof result.result === "string"
            ? result.result
            : JSON.stringify(result.result ?? null),
        ...(result.isError ? { is_error: true } : {}),
      })),
    });
  }

  return messages;
}

export function buildBody(
  request: TurnRequest,
  options: AnthropicOptions,
): Record<string, unknown> {
  const ttl = options.cacheTtl ?? "5m";
  const cacheControl =
    ttl === "5m"
      ? { type: "ephemeral" as const }
      : { type: "ephemeral" as const, ttl };

  return {
    model: options.model,
    max_tokens: options.maxTokens ?? 16_384,
    stream: true,
    // `adaptive` is the only accepted form on Opus 5.5 / Sonnet 5.5 / Haiku
    // 5.5; `budget_tokens` is a 400 on all three. `summarized` display is what
    // makes the operator's view show reasoning instead of empty blocks — the
    // default (`omitted`) returns a signature and no text.
    thinking: { type: "adaptive", display: "summarized" },
    ...(options.effort ? { output_config: { effort: options.effort } } : {}),
    // Placed at the top level on purpose: the API moves the breakpoint to the
    // last cacheable block for us. In an agent loop the prefix grows by one
    // message per turn, so a hand-placed breakpoint would have to be moved
    // every turn — and a breakpoint left behind caches a prefix nobody reads.
    cache_control: cacheControl,
    ...(request.system !== undefined
      ? { system: [{ type: "text", text: request.system }] }
      : {}),
    ...(request.tools.length > 0
      ? {
          tools: request.tools.map((tool) => ({
            name: tool.name,
            description: tool.description,
            input_schema: tool.schema,
          })),
        }
      : {}),
    messages: toMessages(request.transcript),
  };
}

// -- errors -------------------------------------------------------------

const RETRYABLE = new Set([429, 500, 502, 503, 529]);

interface ApiErrorEnvelope {
  error?: { type?: string; message?: string };
}

function remedyFor(status: number, type: string, message: string): string {
  if (spendLimitOf(status, message) === "configured") {
    return (
      "A spend limit set in the Console refuses this call -- not the credit " +
      "balance. Raise it at platform.claude.com under Settings > Billing > " +
      "Spend limits, and check the workspace's own limit too: every " +
      "non-Default workspace carries one, and the lower of the two wins."
    );
  }
  if (status === 402) {
    return (
      "The Claude Platform credit balance is exhausted. Credits reset monthly " +
      "and do not roll over, so no retry will succeed before then. Re-run this " +
      "Run against the local provider (llama3.1:8b through the GPU service), or top " +
      "up the Console balance."
    );
  }
  if (status === 401) {
    return (
      "OHH_ANTHROPIC_API_KEY is missing, revoked, or belongs to another " +
      "organization. Create a key in the Console org linked from claude.ai " +
      "Settings > Billing, and put it in .env at the repo root -- not in the shell."
    );
  }
  if (status === 404) {
    return (
      `The model may not exist or may not be available to this organization. ` +
      `Check the model id and the org's model access.`
    );
  }
  if (RETRYABLE.has(status)) {
    return "Transient. The adapter already retried; try again shortly.";
  }
  return `Fix the request and retry. The API said: ${message || type}.`;
}

async function toGatewayError(response: HttpResponse): Promise<GatewayError> {
  const raw = (await readBody(response)).toString("utf8");
  let envelope: ApiErrorEnvelope = {};
  try {
    envelope = JSON.parse(raw) as ApiErrorEnvelope;
  } catch {
    // Not JSON — a proxy or gateway page. `raw` is still the best evidence.
  }
  const type = envelope.error?.type ?? "unknown_error";
  const message = envelope.error?.message ?? raw.slice(0, 400);

  const spendLimit = spendLimitOf(response.status, message);

  return new GatewayError(
    `Anthropic ${response.status} ${type}: ${message}`,
    {
      status: response.status,
      type,
      retryable: RETRYABLE.has(response.status),
      ...(spendLimit !== undefined ? { spendLimit } : {}),
      remedy: remedyFor(response.status, type, message),
    },
  );
}

function retryDelayMs(response: HttpResponse, attempt: number): number {
  const header = response.headers["retry-after"];
  const value = Array.isArray(header) ? header[0] : header;
  const seconds = value !== undefined ? Number(value) : Number.NaN;
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  // Exponential with a floor, for the 529s that carry no header.
  return 500 * 2 ** attempt;
}

// -- stream translation -------------------------------------------------

type BlockState =
  | { kind: "text" }
  | { kind: "thinking"; thinking: string; signature: string }
  | { kind: "tool_use"; id: string; name: string; json: string }
  | { kind: "passthrough"; block: AnthropicBlock }
  | { kind: "ignored" };

function stopReasonOf(raw: unknown): StopReason {
  switch (raw) {
    case "tool_use":
    case "max_tokens":
    case "refusal":
    case "pause_turn":
      return raw;
    // `stop_sequence` and anything unrecognised are a finished turn: we have
    // the text, and inventing a failure from an unknown reason is worse than
    // treating it as an ordinary end.
    default:
      return "end_turn";
  }
}

/**
 * Translate one SSE stream into `TurnEvent`s.
 *
 * Exported so the shape can be tested against recorded frames without a
 * socket — which is the only honest way to test this, since the alternative is
 * asserting against our own fake of the API.
 */
export async function* translateStream(
  frames: AsyncIterable<{ event: string; data: string }>,
  onUsage?: (usage: UsageTotals) => void,
): AsyncGenerator<TurnEvent> {
  const blocks = new Map<number, BlockState>();
  const usage: UsageTotals = { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 };
  let sawUsage = false;

  for await (const frame of frames) {
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(frame.data) as Record<string, unknown>;
    } catch {
      continue; // A keepalive or a malformed frame is not worth failing a turn.
    }

    switch (frame.event) {
      case "error": {
        const error = (payload["error"] ?? {}) as { type?: string; message?: string };
        throw new GatewayError(
          `Anthropic stream error: ${error.type ?? "unknown"}: ${error.message ?? ""}`,
          {
            status: 0,
            type: error.type ?? "stream_error",
            retryable: error.type === "overloaded_error",
            remedy:
              "The stream failed after headers. Re-run the turn; the " +
              "transcript is unchanged, so the prompt cache still applies.",
          },
        );
      }

      case "message_start": {
        const message = (payload["message"] ?? {}) as Record<string, unknown>;
        const u = (message["usage"] ?? {}) as Record<string, number>;
        usage.in = u["input_tokens"] ?? 0;
        usage.cacheWrite = u["cache_creation_input_tokens"] ?? 0;
        usage.cacheRead = u["cache_read_input_tokens"] ?? 0;
        sawUsage = true;
        break;
      }

      case "content_block_start": {
        const index = payload["index"] as number;
        const block = (payload["content_block"] ?? {}) as AnthropicBlock;
        switch (block["type"]) {
          case "text":
            blocks.set(index, { kind: "text" });
            break;
          case "thinking":
            blocks.set(index, { kind: "thinking", thinking: "", signature: "" });
            break;
          case "tool_use":
            blocks.set(index, {
              kind: "tool_use",
              id: String(block["id"] ?? ""),
              name: String(block["name"] ?? ""),
              json: "",
            });
            break;
          // `redacted_thinking` (and any future opaque reasoning block) arrives
          // whole and must be replayed byte-for-byte. We never look inside.
          default:
            blocks.set(index, { kind: "passthrough", block });
            break;
        }
        break;
      }

      case "content_block_delta": {
        const index = payload["index"] as number;
        const delta = (payload["delta"] ?? {}) as Record<string, unknown>;
        const state = blocks.get(index);
        if (!state) break;

        if (delta["type"] === "text_delta" && state.kind === "text") {
          // Streamed through immediately; coalesced again on replay.
          yield { kind: "text", text: String(delta["text"] ?? "") };
        } else if (delta["type"] === "thinking_delta" && state.kind === "thinking") {
          state.thinking += String(delta["thinking"] ?? "");
        } else if (delta["type"] === "signature_delta" && state.kind === "thinking") {
          state.signature += String(delta["signature"] ?? "");
        } else if (delta["type"] === "input_json_delta" && state.kind === "tool_use") {
          state.json += String(delta["partial_json"] ?? "");
        }
        break;
      }

      case "content_block_stop": {
        const index = payload["index"] as number;
        const state = blocks.get(index);
        blocks.delete(index);
        if (!state) break;

        if (state.kind === "thinking") {
          yield {
            kind: "thinking",
            text: state.thinking,
            // Rebuilt in the API's own shape, signature included. This is the
            // object that gets replayed verbatim next turn.
            opaque: {
              type: "thinking",
              thinking: state.thinking,
              signature: state.signature,
            },
          };
        } else if (state.kind === "passthrough") {
          yield { kind: "thinking", text: "", opaque: state.block };
        } else if (state.kind === "tool_use") {
          let input: unknown = {};
          if (state.json.trim() !== "") {
            try {
              input = JSON.parse(state.json);
            } catch {
              // A truncated tool input is a real failure mode, not a crash:
              // hand the model back something it can see is wrong rather than
              // aborting a turn that otherwise succeeded.
              input = { _malformed_json: state.json };
            }
          }
          yield { kind: "tool_request", id: state.id, name: state.name, input };
        }
        break;
      }

      case "message_delta": {
        const u = (payload["usage"] ?? {}) as Record<string, number>;
        if (u["output_tokens"] !== undefined) {
          usage.out = u["output_tokens"];
          sawUsage = true;
        }
        // Later turns can revise input counts; take the last word on each.
        if (u["input_tokens"] !== undefined) usage.in = u["input_tokens"];
        if (u["cache_creation_input_tokens"] !== undefined) {
          usage.cacheWrite = u["cache_creation_input_tokens"];
        }
        if (u["cache_read_input_tokens"] !== undefined) {
          usage.cacheRead = u["cache_read_input_tokens"];
        }

        if (sawUsage) {
          onUsage?.(usage);
          yield { kind: "usage", ...usage };
          sawUsage = false;
        }

        const delta = (payload["delta"] ?? {}) as Record<string, unknown>;
        if (delta["stop_reason"] !== undefined) {
          yield { kind: "end", reason: stopReasonOf(delta["stop_reason"]) };
        }
        break;
      }

      default:
        break; // ping, message_stop — nothing to translate.
    }
  }
}

// -- the provider -------------------------------------------------------

export function createAnthropicProvider(options: AnthropicOptions): Provider {
  const send = options.send ?? defaultSend;
  const sleep =
    options.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const baseUrl = (options.baseUrl ?? ANTHROPIC_BASE_URL).replace(/\/+$/, "");
  const maxRetries = options.maxRetries ?? 2;
  const ttl = options.cacheTtl ?? "5m";

  async function open(body: string): Promise<HttpResponse> {
    let lastError: GatewayError | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const response = await send(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": options.apiKey,
          "anthropic-version": API_VERSION,
          accept: "text/event-stream",
          "content-length": String(Buffer.byteLength(body)),
        },
        body,
        // No headers timeout: a long prompt legitimately takes a while before
        // the first frame, and a cold 100k-token prefix can take minutes.
      });

      if (response.status === 200) return response;

      const delayMs = retryDelayMs(response, attempt);
      lastError = await toGatewayError(response);
      if (!lastError.retryable || attempt === maxRetries) throw lastError;
      await sleep(delayMs);
    }

    // Unreachable: the loop either returns or throws.
    throw lastError ?? new Error("anthropic: no response");
  }

  return {
    name: "anthropic",
    model: options.model,

    chat(request: TurnRequest): AsyncIterable<TurnEvent> {
      const body = JSON.stringify(buildBody(request, options));

      return {
        async *[Symbol.asyncIterator]() {
          const response = await open(body);
          try {
            yield* translateStream(
              readSse(response.stream),
              (usage) =>
                options.onCost?.({
                  usd: costUsd(options.model, usage, ttl),
                  usage,
                }),
            );
          } finally {
            // A consumer that breaks early (a budget trip) must not leave
            // the socket open — the response is already being billed.
            response.abort();
          }
        },
      };
    },
  };
}
