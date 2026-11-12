/**
 * The normalized turn event shape — the contract the real adapters
 * (`anthropic`, `google`, `ollama`) must satisfy.
 *
 * It is deliberately shaped from the real Anthropic streaming vocabulary rather
 * than from whatever the scripted double finds convenient, so each adapter is filling in
 * an interface rather than retrofitting one:
 *
 *   content_block_delta / text_delta      -> { kind: "text" }
 *   content_block_delta / thinking_delta  -> { kind: "thinking" }
 *   content_block_delta / input_json_delta-> accumulated into "tool_request"
 *   message_delta       / usage           -> { kind: "usage" }
 *   message_delta       / stop_reason     -> { kind: "end" }
 *
 * Two choices are load-bearing:
 *
 * - `refusal` is a terminal *reason*, not an exception. A safety classifier
 *   declining a request returns HTTP 200 with `stop_reason: "refusal"`.
 *   Throwing would make an ordinary response look like a transport failure.
 * - `thinking` is a first-class event, because thinking blocks must be replayed
 *   back unchanged on the same model over an append-only history. That is what
 *   constrains the checkpoint blob: it may append to a transcript, never
 *   rewrite an earlier turn.
 */

export type StopReason =
  | "end_turn"
  | "tool_use"
  | "max_tokens"
  | "refusal"
  | "pause_turn";

export type TurnEvent =
  | { kind: "text"; text: string }
  /**
   * `opaque` is the provider's own block, kept verbatim for replay.
   *
   * `text` alone cannot satisfy the replay rule stated above.
   * Claude Opus 5.5 returns thinking blocks whose `thinking` field is **empty
   * by default** (`display: "omitted"`) and whose `signature` cryptographically
   * binds the block to the conversation prefix that produced it — the system
   * prompt, the tool set, and every preceding message. Replaying the text
   * without the signature is a 400; dropping the block instead is an ordering
   * 400. Neither is recoverable from `text`.
   *
   * So the adapter round-trips whatever it was handed and the Gateway contract
   * stays provider-agnostic: Ollama has no signatures, Google has its own
   * scheme, and neither needs to be modelled here. `unknown`, not a typed
   * block, precisely because the orchestrator must never interpret it.
   */
  | { kind: "thinking"; text: string; opaque?: unknown }
  | { kind: "tool_request"; id: string; name: string; input: unknown }
  /**
   * Four buckets, not two. A cache read costs ~0.1x a fresh input token, so
   * collapsing them overstates a cache-heavy turn ~10x — and the budget spends that
   * figure. The scripted provider has no spend, so only the real adapters
   * produce it.
   */
  | {
      kind: "usage";
      in: number;
      cacheWrite: number;
      cacheRead: number;
      out: number;
    }
  | { kind: "end"; reason: StopReason };

export interface ToolResultMessage {
  readonly toolUseId: string;
  readonly name: string;
  readonly result: unknown;
  readonly isError?: boolean;
}

export type TranscriptEntry =
  | { role: "user"; text: string }
  | { role: "assistant"; events: TurnEvent[] }
  | { role: "tool_results"; results: ToolResultMessage[] };

export interface TurnRequest {
  readonly system?: string;
  readonly transcript: readonly TranscriptEntry[];
  readonly tools: readonly { name: string; description: string; schema: unknown }[];
}

export interface Provider {
  /** The adapter: "anthropic" | "ollama" | "scripted". Goes in `costs.provider`. */
  readonly name: string;
  /**
   * The specific model being driven, e.g. `claude-opus-5-5`. Separate from
   * `name` because pricing is per model, not per provider — Haiku and Opus
   * differ 40x through the same adapter, so a cost row keyed only by provider
   * could not be re-priced later.
   */
  readonly model: string;
  chat(request: TurnRequest): AsyncIterable<TurnEvent>;
}
