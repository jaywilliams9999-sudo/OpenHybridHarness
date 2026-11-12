/**
 * A provider that degrades to a second provider when the credits run out.
 *
 * Reaching the credit ceiling should **degrade rather than stop**. The
 * `ollama` provider has zero marginal cost, so a Run that would overrun the
 * remaining credit is a candidate for routing to local inference instead of
 * `blocked`. This is that routing, and it makes the local model a **tier**
 * rather than a redundancy.
 *
 * Three restraints, each the reason this is a separate component rather than a
 * `catch` in the Loop:
 *
 * 1. **Only a spending wall triggers it.** A 401, an ordinary 400, a
 *    malformed request — those are defects to fix, and silently downgrading to
 *    a weaker model would hide them behind degraded output. A 429 is already
 *    retried inside the adapter. Being unable to spend is the one failure where
 *    no retry can ever help and a worse answer genuinely beats no answer.
 *
 *    That covers two conditions, not one: an exhausted credit balance (402)
 *    and a Console spend limit (400, recognised by its message). The second
 *    was found the hard way — a $1/month limit produced a 400 that this
 *    provider would have re-thrown, stopping a Run that could have finished
 *    for free on the local model.
 * 2. **The switch is permanent for the rest of the Run.** Credits do not come
 *    back before the month rolls over, so retrying the primary on every turn
 *    would spend a round trip to learn the same thing.
 * 3. **It switches only before the first event of a turn.** See `chat`.
 */

import { GatewayError, type SpendLimit } from "./anthropic.ts";
import type { Provider, TurnEvent, TurnRequest } from "./types.ts";

export interface FailoverOptions {
  readonly primary: Provider;
  readonly fallback: Provider;
  /** Called once, when the switch happens, so an operator learns about it. */
  readonly onDegrade?: (info: {
    from: string;
    to: string;
    reason: string;
    /** Which wall was hit; the operator's remedy differs completely. */
    limit?: SpendLimit;
  }) => void;
}

function isSpendBlocked(error: unknown): boolean {
  return error instanceof GatewayError && error.spendBlocked;
}

export function createFailoverProvider(options: FailoverOptions): Provider {
  const { primary, fallback } = options;
  let degraded = false;

  return {
    /** Whichever provider is actually answering, so `costs.provider` is true. */
    get name(): string {
      return degraded ? fallback.name : primary.name;
    },

    get model(): string {
      return degraded ? fallback.model : primary.model;
    },

    chat(request: TurnRequest): AsyncIterable<TurnEvent> {
      return {
        async *[Symbol.asyncIterator]() {
          if (degraded) {
            yield* fallback.chat(request);
            return;
          }

          /**
           * The switch is only safe before the first event.
           *
           * A spend rejection — 402 for credits, 400 for a configured limit —
           * is a rejection of the request itself, so it arrives before any
           * content: nothing was streamed and nothing was billed, which makes
           * re-running the turn on the fallback clean. If
           * a stream instead dies halfway, some text is already in the
           * caller's hands and on the operator's screen; re-running would
           * duplicate it into the transcript, and every later turn is built on
           * that transcript. So a mid-stream failure propagates even if it
           * somehow claimed credit exhaustion.
           */
          let started = false;
          try {
            for await (const event of primary.chat(request)) {
              started = true;
              yield event;
            }
            return;
          } catch (error) {
            if (started || !isSpendBlocked(error)) throw error;

            degraded = true;
            options.onDegrade?.({
              from: primary.name,
              to: fallback.name,
              reason: error instanceof Error ? error.message : String(error),
              ...(error instanceof GatewayError && error.spendLimit !== undefined
                ? { limit: error.spendLimit }
                : {}),
            });
          }

          // Outside the catch: an error thrown by the fallback is its own
          // failure, not something to re-interpret as another degrade.
          yield* fallback.chat(request);
        },
      };
    },
  };
}
