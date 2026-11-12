/**
 * The `scripted` provider: a canned sequence of turns.
 *
 * It sits alongside the three real adapters and is the only Gateway
 * implementation that spends nothing. It exists so the Loop can be driven
 * deterministically — and so the slice proves the GPU service boundary without
 * spending money on a cloud API.
 */

import type { Provider, TurnEvent, TurnRequest } from "./types.ts";

export interface ScriptedOptions {
  /** One array of events per turn, consumed in order. */
  readonly turns: readonly TurnEvent[][];
  /** Observe what the Loop actually sent, for assertions. */
  readonly onRequest?: (request: TurnRequest) => void;
}

export function createScriptedProvider(options: ScriptedOptions): Provider {
  let turn = 0;

  return {
    name: "scripted",
    model: "scripted",

    chat(request: TurnRequest): AsyncIterable<TurnEvent> {
      options.onRequest?.(request);
      const events = options.turns[turn] ?? [
        { kind: "text", text: "(script exhausted)" },
        { kind: "end", reason: "end_turn" },
      ];
      turn++;

      return {
        async *[Symbol.asyncIterator]() {
          for (const event of events) {
            // Yield on the microtask queue so consumers cannot accidentally
            // depend on synchronous delivery, which no real provider offers.
            await Promise.resolve();
            yield event;
          }
        },
      };
    },
  };
}
