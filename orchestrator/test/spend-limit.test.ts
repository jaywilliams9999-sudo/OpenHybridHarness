/**
 * The two ways the money can stop, and the one way it must not be
 * confused with a bug.
 *
 * The configured-limit arm exists because of a real failure. On 2026-10-15,
 * with a $1/month Console spend limit in force, the live API answered:
 *
 *     HTTP 400 invalid_request_error
 *     "You have reached your specified API usage limits.
 *      You will regain access on 2026-11-01 at 00:00 UTC."
 *
 * The degradation trigger tested `status === 402`, so it would not have
 * fired. These tests pin both arms, and — more importantly — pin that an
 * ordinary 400 still fails loudly rather than quietly costing the operator a
 * weaker model.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  createAnthropicProvider,
  GatewayError,
  spendLimitOf,
} from "../src/gateway/anthropic.ts";
import { createFailoverProvider } from "../src/gateway/failover.ts";
import type { HttpResponse } from "../src/gpu/http.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";

/** The exact body the live API returned, kept verbatim. */
const LIMIT_MESSAGE =
  "You have reached your specified API usage limits. " +
  "You will regain access on 2026-11-01 at 00:00 UTC.";

function jsonResponse(status: number, body: string): HttpResponse {
  async function* chunks() {
    yield Buffer.from(body);
  }
  return {
    status,
    contentType: "application/json",
    headers: {},
    stream: chunks() as unknown as HttpResponse["stream"],
    abort: () => {},
  };
}

function failing(status: number, type: string, message: string): Provider {
  return createAnthropicProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    maxRetries: 0,
    send: async () =>
      jsonResponse(status, JSON.stringify({ error: { type, message } })),
  });
}

async function drain(provider: Provider): Promise<TurnEvent[]> {
  const events: TurnEvent[] = [];
  for await (const event of provider.chat({ transcript: [], tools: [] })) {
    events.push(event);
  }
  return events;
}

// -- the predicate ------------------------------------------------------

test("a 402 is the credits wall, whatever it says", () => {
  assert.equal(spendLimitOf(402, "credit balance too low"), "credits");
  assert.equal(spendLimitOf(402, ""), "credits");
});

test("the live 400 limit message is recognised as a configured wall", () => {
  assert.equal(spendLimitOf(400, LIMIT_MESSAGE), "configured");
});

test("an ordinary 400 is not a spending wall", () => {
  // Every one of these is a defect to fix. Degrading on any of them would
  // hide a bug behind a weaker model's output.
  for (const message of [
    "messages: at least one message is required",
    "max_tokens: must be greater than 0",
    "thinking.budget_tokens: unsupported parameter",
    "model: claude-opus-5-6 not found",
  ]) {
    assert.equal(spendLimitOf(400, message), undefined, message);
  }
});

test("a limit phrase alone is not enough to degrade", () => {
  // Both clauses are required. A malformed request does not tell you when
  // you will regain access, and that is what keeps a false positive -- the
  // only expensive direction to be wrong in -- from being cheap to trigger.
  assert.equal(spendLimitOf(400, "usage limits are described in the docs"), undefined);
  assert.equal(spendLimitOf(400, "you will regain access shortly"), undefined);
});

// -- the error ----------------------------------------------------------

test("the configured-limit error points at the Console, not at the balance", async () => {
  await assert.rejects(
    () => drain(failing(400, "invalid_request_error", LIMIT_MESSAGE)),
    (error: unknown) => {
      assert.ok(error instanceof GatewayError);
      assert.equal(error.spendBlocked, true);
      assert.equal(error.spendLimit, "configured");
      assert.equal(error.retryable, false);
      // The remedies are not interchangeable: one is a setting to raise, the
      // other is a month to wait out.
      assert.match(error.remedy, /Spend limits/i);
      assert.match(error.remedy, /workspace/i);
      assert.doesNotMatch(error.remedy, /balance is exhausted/i);
      return true;
    },
  );
});

test("an ordinary 400 carries no spending wall", async () => {
  await assert.rejects(
    () => drain(failing(400, "invalid_request_error", "max_tokens: must be > 0")),
    (error: unknown) => {
      assert.ok(error instanceof GatewayError);
      assert.equal(error.spendBlocked, false);
      assert.equal(error.spendLimit, undefined);
      return true;
    },
  );
});

// -- the failover -------------------------------------------------------

function localFallback(): Provider {
  return {
    name: "ollama",
    model: "llama3.1:8b",
    chat: async function* () {
      yield { kind: "text", text: "local answer" } as TurnEvent;
      yield { kind: "end", reason: "end_turn" } as TurnEvent;
    },
  };
}

test("a configured spend limit degrades to local rather than stopping", async () => {
  const degrades: Array<{ to: string; limit?: string }> = [];
  const provider = createFailoverProvider({
    primary: failing(400, "invalid_request_error", LIMIT_MESSAGE),
    fallback: localFallback(),
    onDegrade: ({ to, limit }) => degrades.push({ to, ...(limit ? { limit } : {}) }),
  });

  const events = await drain(provider);

  assert.deepEqual(
    events.filter((e) => e.kind === "text").map((e) => (e as { text: string }).text),
    ["local answer"],
  );
  // The operator is told which wall, because the two remedies differ: one is
  // a Console setting, the other is waiting for the month to roll.
  assert.deepEqual(degrades, [{ to: "ollama", limit: "configured" }]);
  assert.equal(provider.model, "llama3.1:8b");
});

test("the credits wall still degrades, and says so differently", async () => {
  const degrades: Array<string | undefined> = [];
  const provider = createFailoverProvider({
    primary: failing(402, "billing_error", "credit balance too low"),
    fallback: localFallback(),
    onDegrade: ({ limit }) => degrades.push(limit),
  });

  await drain(provider);
  assert.deepEqual(degrades, ["credits"]);
});

test("an ordinary 400 is not quietly answered by a weaker model", async () => {
  let degraded = false;
  const provider = createFailoverProvider({
    primary: failing(400, "invalid_request_error", "max_tokens: must be > 0"),
    fallback: localFallback(),
    onDegrade: () => {
      degraded = true;
    },
  });

  // The whole restraint of the failover: a defect must surface as a
  // defect. Silently downgrading would leave a broken request producing
  // plausible output from the wrong model, indefinitely.
  await assert.rejects(() => drain(provider), GatewayError);
  assert.equal(degraded, false);
});
