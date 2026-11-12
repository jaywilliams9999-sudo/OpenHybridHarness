/**
 * The `ollama` provider — local inference, routed through the GPU service.
 *
 * **It never touches port 11434.** This is settled design: `OllamaBackend.load`
 * *is* Ollama's on-demand auto-load, so anything that posts a prompt to Ollama
 * loads whatever model it names with no fit calculation and no eviction. An
 * orchestrator inferencing directly would make the GPU service irrelevant rather than
 * merely blind — and one subsystem is the single source of truth for VRAM, or
 * none is. So this adapter speaks `POST /generate/text`, which reserves the
 * model and answers inside one hold of the GPU lock.
 *
 * **The whole turn arrives at once.** The GPU service streams `op.token` frames for
 * live observation, but its event bus drops frames under backpressure, so the
 * reply is read from the op's terminal `content`. This provider therefore
 * yields one `text` event per turn rather than a token stream. That is a
 * deliberate trade of liveness for correctness: a dropped token would silently
 * corrupt the transcript, and the transcript is what every later turn is built
 * on. Operators still see tokens live through the client's `onProgress`.
 *
 * **Cost is zero and that is not a special case.** `pricing.ts` has no entry
 * for a local model, so `costUsd` returns 0 and the Ledger records a row with
 * real token counts against no spend — which is exactly true.
 */

import type { GpuClient, TextResult } from "../gpu/client.ts";
import type {
  Provider,
  StopReason,
  ToolResultMessage,
  TranscriptEntry,
  TurnEvent,
  TurnRequest,
} from "./types.ts";

export interface OllamaOptions {
  readonly gpu: GpuClient;
  /** Omitted means the GPU service picks its `role = "code"` model from the registry. */
  readonly model?: string;
  /** Omitted means the model's own registry window. */
  readonly numCtx?: number;
  readonly temperature?: number;
  readonly maxTokens?: number;
}

type OllamaMessage = Record<string, unknown>;

// -- request translation ------------------------------------------------

/**
 * Flatten one assistant turn's events into Ollama's single-string `content`.
 *
 * Ollama has no block model: an assistant message is `{role, content,
 * tool_calls?}`. Thinking is dropped rather than concatenated into the text —
 * a local model has no signed thinking to preserve (so nothing breaks), and
 * folding reasoning into `content` would teach it that its own scratchpad is
 * part of the answer.
 */
export function assistantMessage(events: readonly TurnEvent[]): OllamaMessage {
  let content = "";
  const toolCalls: Array<{ function: { name: string; arguments: unknown } }> = [];

  for (const event of events) {
    if (event.kind === "text") content += event.text;
    else if (event.kind === "tool_request") {
      toolCalls.push({
        function: { name: event.name, arguments: event.input ?? {} },
      });
    }
  }

  return {
    role: "assistant",
    content,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
}

function toolResultMessages(
  results: readonly ToolResultMessage[],
): OllamaMessage[] {
  // One message per result, where Anthropic takes one message holding many
  // blocks. Ollama's `tool` role has no id field, so `tool_name` is the only
  // thing tying a result back to the call that asked for it.
  return results.map((result) => ({
    role: "tool",
    tool_name: result.name,
    content:
      typeof result.result === "string"
        ? result.result
        : JSON.stringify(result.result ?? null),
  }));
}

export function toOllamaMessages(
  transcript: readonly TranscriptEntry[],
  system?: string,
): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  if (system !== undefined) messages.push({ role: "system", content: system });

  for (const entry of transcript) {
    if (entry.role === "user") {
      messages.push({ role: "user", content: entry.text });
    } else if (entry.role === "assistant") {
      const message = assistantMessage(entry.events);
      // A turn that produced neither text nor a tool call has nothing to say;
      // an empty assistant message only wastes context.
      if (message["content"] !== "" || message["tool_calls"] !== undefined) {
        messages.push(message);
      }
    } else {
      messages.push(...toolResultMessages(entry.results));
    }
  }

  return messages;
}

export function toOllamaTools(
  tools: TurnRequest["tools"],
): Array<Record<string, unknown>> {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      // Ollama calls it `parameters` where Anthropic calls it `input_schema`.
      // Same JSON Schema either side.
      parameters: tool.schema,
    },
  }));
}

// -- response translation -----------------------------------------------

/**
 * Ollama reports no `stop_reason`, so it is inferred.
 *
 * The presence of tool calls is the signal: `done_reason` is `"stop"` whether
 * the model finished talking or finished asking, and the Loop branches on
 * `"tool_use"` to decide whether to dispatch tools at all.
 */
export function stopReasonFor(result: TextResult): StopReason {
  if (result.tool_calls.length > 0) return "tool_use";
  if (result.done_reason === "length") return "max_tokens";
  return "end_turn";
}

export interface NormalizedTurn {
  content: string;
  calls: Array<{ name: string; input: unknown }>;
}

const FENCE = /^```(?:json)?\s*\n?([\s\S]*?)\n?\s*```$/;

/**
 * The one JSON object a reply consists of, or undefined. A ```json fence
 * around it is tolerated; anything before or after it is not -- a reply that
 * merely *mentions* JSON is prose.
 */
/** The start of a `{"name": "<tool>", ...}` call, read without parsing. */
const MALFORMED_CALL = /^\{\s*"name"\s*:\s*"([^"]+)"/;

/** The reply trimmed, with a ```json fence around the whole of it removed. */
function unfenced(content: string): string {
  const text = content.trim();
  const fenced = FENCE.exec(text);
  return fenced ? (fenced[1] ?? "").trim() : text;
}

function soleJsonObject(content: string): Record<string, unknown> | undefined {
  const text = unfenced(content);
  if (!text.startsWith("{") || !text.endsWith("}")) return undefined;
  try {
    const parsed: unknown = JSON.parse(text);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

/**
 * A string of arguments, parsed. Unparseable is kept as `{ _raw }` rather
 * than dropped: the Loop's input validation then refuses it back to the
 * model, which learns its call was malformed instead of silently not made.
 */
function parseArguments(args: unknown): unknown {
  if (typeof args !== "string") return args ?? {};
  if (args.trim() === "") return {};
  try {
    return JSON.parse(args);
  } catch {
    return { _raw: args };
  }
}

/**
 * Bring a small local model's tool calls back to the structured shape.
 *
 * `llama3.1:8b` mostly emits proper `tool_calls`, but it drifts in two ways
 * that would otherwise end a run: it hands `arguments` over as a JSON string,
 * and it sometimes writes the call into its reply as `{"name": ..,
 * "parameters": ..}` with no `tool_calls` at all -- which the Loop would read
 * as a final answer. The second is recovered only when the whole reply is
 * that one object and it names a tool this turn offered, so ordinary prose,
 * or JSON about something else, is never mistaken for a call. A call with no
 * name is dropped: there is nothing to dispatch it to.
 */
export function normalizeToolCalls(
  content: string,
  toolCalls: unknown[],
  known: ReadonlySet<string>,
): NormalizedTurn {
  const calls: NormalizedTurn["calls"] = [];
  for (const raw of toolCalls) {
    const fn =
      typeof raw === "object" && raw !== null
        ? (raw as { function?: { name?: unknown; arguments?: unknown } }).function
        : undefined;
    const name = fn?.name;
    if (typeof name !== "string" || name === "") continue;
    calls.push({ name, input: parseArguments(fn?.arguments) });
  }
  if (toolCalls.length > 0) return { content, calls };

  const json = soleJsonObject(content);
  if (json === undefined) {
    // Plainly a call to an offered tool that is not valid JSON (seen live:
    // an unescaped quote in a value). Passed on as `{ _raw }` so the Loop
    // tells the model, rather than the run ending with it as the answer.
    const text = unfenced(content);
    const named = MALFORMED_CALL.exec(text)?.[1];
    if (named !== undefined && known.has(named) && text.endsWith("}")) {
      return { content: "", calls: [{ name: named, input: { _raw: text } }] };
    }
    return { content, calls };
  }
  const name = json["name"];
  if (typeof name !== "string" || !known.has(name)) {
    return { content, calls };
  }
  const args = json["arguments"] ?? json["parameters"] ?? {};
  return { content: "", calls: [{ name, input: parseArguments(args) }] };
}

const NUMERIC = /^-?\d+(\.\d+)?$/;

/**
 * Scalars given as strings, turned back into what the tool's schema asks for.
 *
 * `llama3.1:8b` routinely sends every argument as a string -- `"budget_usd":
 * "1"` for a `number` -- and a handler that checks its types then refuses the
 * call however often the model retries, because the model cannot see why
 * `"1"` is not a number. Only a top-level property the schema types as
 * `number`/`integer`/`boolean`, holding a string that reads cleanly as one,
 * is converted; anything else is left for the tool to judge.
 */
export function coerceToSchema(input: unknown, schema: unknown): unknown {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return input;
  const properties = (schema as { properties?: unknown } | null)?.properties;
  if (typeof properties !== "object" || properties === null) return input;

  const out: Record<string, unknown> = { ...(input as Record<string, unknown>) };
  for (const [key, value] of Object.entries(out)) {
    if (typeof value !== "string") continue;
    const type = (properties as Record<string, { type?: unknown } | undefined>)[key]?.type;
    const text = value.trim();
    if ((type === "number" || type === "integer") && NUMERIC.test(text)) {
      const n = Number(text);
      if (type === "number" || Number.isInteger(n)) out[key] = n;
    } else if (type === "boolean" && (text === "true" || text === "false")) {
      out[key] = text === "true";
    }
  }
  return out;
}

/**
 * Synthesize a call id.
 *
 * Ollama's tool calls carry no identifier, but `ToolResultMessage.toolUseId`
 * needs one and the Loop keys results by it. Ids are therefore minted here,
 * unique within a Run by turn and position. They never travel back to Ollama
 * — `toolResultMessages` matches on `tool_name` instead — so their only job is
 * to stay distinct inside the orchestrator.
 */
function callId(turn: number, index: number): string {
  return `local-${turn}-${index}`;
}

export function createOllamaProvider(options: OllamaOptions): Provider {
  let turn = 0;
  let resolvedModel = options.model;

  return {
    name: "ollama",

    /**
     * A getter, not a field, because when the model is left to the registry the
     * Orchestrator does not know its name until a turn comes back. The Loop reads
     * this *after* collecting the turn, so the Ledger records the real model
     * from the first row onward rather than a placeholder that would have to be
     * reconciled later.
     */
    get model(): string {
      return resolvedModel ?? "ollama:role=code";
    },

    chat(request: TurnRequest): AsyncIterable<TurnEvent> {
      turn++;
      const thisTurn = turn;

      return {
        async *[Symbol.asyncIterator]() {
          const result = await options.gpu.generateText({
            messages: toOllamaMessages(request.transcript, request.system),
            ...(request.tools.length > 0
              ? { tools: toOllamaTools(request.tools) }
              : {}),
            ...(options.model !== undefined ? { model: options.model } : {}),
            ...(options.numCtx !== undefined ? { num_ctx: options.numCtx } : {}),
            ...(options.temperature !== undefined
              ? { temperature: options.temperature }
              : {}),
            ...(options.maxTokens !== undefined
              ? { max_tokens: options.maxTokens }
              : {}),
          });

          resolvedModel = result.model;

          const schemas = new Map(request.tools.map((tool) => [tool.name, tool.schema]));
          const known = new Set(schemas.keys());
          const { content, calls } = normalizeToolCalls(
            result.content,
            result.tool_calls,
            known,
          );

          if (content !== "") {
            yield { kind: "text", text: content };
          }

          for (const [index, call] of calls.entries()) {
            yield {
              kind: "tool_request",
              id: callId(thisTurn, index),
              name: call.name,
              input: coerceToSchema(call.input, schemas.get(call.name)),
            };
          }

          // Three buckets of four: Ollama reports no cache accounting, and
          // `prompt_eval_count` already excludes a reused prefix, so a cached
          // turn simply shows a small `in` rather than a populated cacheRead.
          // Recording a guess in the cache buckets would make the Ledger's
          // four-bucket split lie about a provider that cannot fill it.
          yield {
            kind: "usage",
            in: result.prompt_tokens,
            cacheWrite: 0,
            cacheRead: 0,
            out: result.completion_tokens,
          };

          // The normalized calls decide, not the raw result: a call recovered
          // from the reply text makes this a tool turn, and raw calls that were
          // all dropped (no name) do not.
          yield {
            kind: "end",
            reason:
              calls.length > 0 ? "tool_use" : stopReasonFor({ ...result, tool_calls: [] }),
          };
        },
      };
    },
  };
}
