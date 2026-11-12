/**
 * Budgets and the degradation path.
 *
 * Two mechanisms, deliberately separate: the Loop refuses to *start* a turn it
 * cannot pay for, and the failover provider handles the case where the money
 * is gone at the account level rather than the Run level.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { createFailoverProvider } from "../src/gateway/failover.ts";
import { GatewayError } from "../src/gateway/anthropic.ts";
import { runLoop } from "../src/loop/loop.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";

function priced(turns: TurnEvent[][], model = "claude-opus-5-5"): Provider {
  return { ...createScriptedProvider({ turns }), model };
}

const echo: ToolDefinition = {
  name: "echo",
  description: "echo",
  schema: { type: "object" },
  tier: "auto",
  handler: async () => ({ ok: true }),
};

/** A turn that asks for a tool and bills 1000 output tokens = $0.02 on Opus. */
const EXPENSIVE: TurnEvent[] = [
  { kind: "tool_request", id: "t", name: "echo", input: {} },
  { kind: "usage", in: 0, cacheWrite: 0, cacheRead: 0, out: 1_000 },
  { kind: "end", reason: "tool_use" },
];

// -- the Loop's cap -----------------------------------------------------

test("a Run within budget runs to completion", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: priced([
        [
          { kind: "text", text: "cheap" },
          { kind: "usage", in: 0, cacheWrite: 0, cacheRead: 0, out: 10 },
          { kind: "end", reason: "end_turn" },
        ],
      ]),
      registry: createRegistry([]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 1 },
  );
  assert.equal(result.status, "done");
  assert.equal(result.reason, "end_turn");
  ledger.close();
});

test("exceeding the budget blocks rather than truncating silently", async () => {
  const ledger = openLedger(":memory:");
  // $0.02 per turn against a $0.05 cap: turns 1 and 2 spend $0.04, turn 3
  // would cross it... but the check is "already over", so it runs turn 3 to
  // $0.06 and blocks before turn 4.
  const result = await runLoop(
    {
      provider: priced([EXPENSIVE, EXPENSIVE, EXPENSIVE, EXPENSIVE, EXPENSIVE]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.05 },
  );

  assert.equal(result.status, "blocked");
  assert.equal(result.reason, "budget");
  assert.ok(result.spentUsd > 0.05, `spent ${result.spentUsd}`);
  // `blocked`, not `failed`: the distinction is the whole point of the budget design.
  assert.equal(ledger.getRun(result.runId)?.status, "blocked");
  ledger.close();
});

test("the cap stops turns rather than capping mid-turn", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: priced([EXPENSIVE, EXPENSIVE, EXPENSIVE, EXPENSIVE]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 10,
    },
    { agent: "site", goal: "g", budgetUsd: 0.03 },
  );

  // Two turns: the first spends $0.02, the second takes it to $0.04, then the
  // third is refused. A turn's cost is unknowable before it runs -- output is
  // the larger half -- so a turn is never started once the money is gone and
  // never interrupted once it has.
  assert.equal(result.turns, 2);
  assert.equal(ledger.listCosts(result.runId).length, 2);
  ledger.close();
});

test("a zero budget blocks before the first turn and spends nothing", async () => {
  const ledger = openLedger(":memory:");
  const result = await runLoop(
    {
      provider: priced([EXPENSIVE]),
      registry: createRegistry([echo]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 0 },
  );

  assert.equal(result.status, "blocked");
  assert.equal(result.reason, "budget");
  assert.equal(result.turns, 0);
  assert.equal(result.spentUsd, 0);
  assert.equal(ledger.listCosts(result.runId).length, 0);
  ledger.close();
});

test("a free provider never trips the budget", async () => {
  const ledger = openLedger(":memory:");
  // Local inference prices at zero, so a tiny cap is not a constraint on it.
  const result = await runLoop(
    {
      provider: priced(
        [
          [
            { kind: "text", text: "free" },
            { kind: "usage", in: 50_000, cacheWrite: 0, cacheRead: 0, out: 5_000 },
            { kind: "end", reason: "end_turn" },
          ],
        ],
        "llama3.1:8b",
      ),
      registry: createRegistry([]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 0.000001 },
  );
  assert.equal(result.status, "done");
  assert.equal(result.spentUsd, 0);
  ledger.close();
});

// -- the failover provider ----------------------------------------------

function creditsGone(): Provider {
  return {
    name: "anthropic",
    model: "claude-opus-5-5",
    chat() {
      return {
        async *[Symbol.asyncIterator]() {
          await Promise.resolve();
          throw new GatewayError("Anthropic 402 billing_error: balance too low", {
            status: 402,
            type: "billing_error",
            retryable: false,
            spendLimit: "credits" as const,
            remedy: "use the local provider",
          });
          // eslint-disable-next-line no-unreachable
          yield { kind: "text", text: "never" } as TurnEvent;
        },
      };
    },
  };
}

function localProvider(text = "answered locally"): Provider {
  return {
    ...createScriptedProvider({
      turns: [[{ kind: "text", text }, { kind: "end", reason: "end_turn" }]],
    }),
    name: "ollama",
    model: "llama3.1:8b",
  };
}

async function drain(iterable: AsyncIterable<TurnEvent>): Promise<TurnEvent[]> {
  const out: TurnEvent[] = [];
  for await (const event of iterable) out.push(event);
  return out;
}

test("credit exhaustion degrades to the fallback and answers anyway", async () => {
  const degrades: Array<{ from: string; to: string; reason: string }> = [];
  const provider = createFailoverProvider({
    primary: creditsGone(),
    fallback: localProvider(),
    onDegrade: (info) => degrades.push(info),
  });

  const events = await drain(provider.chat({ transcript: [], tools: [] }));

  assert.deepEqual(events, [
    { kind: "text", text: "answered locally" },
    { kind: "end", reason: "end_turn" },
  ]);
  assert.equal(degrades.length, 1);
  assert.equal(degrades[0]?.from, "anthropic");
  assert.equal(degrades[0]?.to, "ollama");
  // The reason carries the API's own message, so the log says *why*.
  assert.match(degrades[0]?.reason ?? "", /402|billing/);
});

test("the active provider is reported, so cost rows name what actually ran", async () => {
  const provider = createFailoverProvider({
    primary: creditsGone(),
    fallback: localProvider(),
  });

  assert.equal(provider.name, "anthropic");
  assert.equal(provider.model, "claude-opus-5-5");

  await drain(provider.chat({ transcript: [], tools: [] }));

  assert.equal(provider.name, "ollama");
  assert.equal(provider.model, "llama3.1:8b");
});

test("the switch is permanent — the primary is not retried next turn", async () => {
  let primaryCalls = 0;
  const primary: Provider = {
    name: "anthropic",
    model: "claude-opus-5-5",
    chat() {
      primaryCalls++;
      return creditsGone().chat({ transcript: [], tools: [] });
    },
  };

  const provider = createFailoverProvider({ primary, fallback: localProvider() });
  await drain(provider.chat({ transcript: [], tools: [] }));
  await drain(provider.chat({ transcript: [], tools: [] }));
  await drain(provider.chat({ transcript: [], tools: [] }));

  // Credits do not return before the month rolls over, so re-asking would
  // spend a round trip to learn the same thing.
  assert.equal(primaryCalls, 1);
});

test("a non-credit failure propagates instead of hiding behind a weaker model", async () => {
  const badKey: Provider = {
    name: "anthropic",
    model: "claude-opus-5-5",
    chat() {
      return {
        async *[Symbol.asyncIterator]() {
          await Promise.resolve();
          throw new GatewayError("Anthropic 401 authentication_error", {
            status: 401,
            type: "authentication_error",
            retryable: false,
            remedy: "fix the key",
          });
        },
      };
    },
  };

  const provider = createFailoverProvider({
    primary: badKey,
    fallback: localProvider(),
  });

  // A broken key is a defect to fix. Degrading would turn it into mysteriously
  // worse output that nobody investigates.
  await assert.rejects(() => drain(provider.chat({ transcript: [], tools: [] })), {
    name: "GatewayError",
  });
});

test("a mid-stream failure propagates rather than duplicating output", async () => {
  const diesHalfway: Provider = {
    name: "anthropic",
    model: "claude-opus-5-5",
    chat() {
      return {
        async *[Symbol.asyncIterator]() {
          yield { kind: "text", text: "half an answer" } as TurnEvent;
          throw new GatewayError("died after streaming", {
            status: 402,
            type: "billing_error",
            retryable: false,
            spendLimit: "credits" as const,
            remedy: "x",
          });
        },
      };
    },
  };

  const provider = createFailoverProvider({
    primary: diesHalfway,
    fallback: localProvider(),
  });

  // Even claiming credit exhaustion: text is already in the caller's hands and
  // on the operator's screen, so re-running would write it twice into the
  // transcript every later turn is built on.
  const seen: TurnEvent[] = [];
  await assert.rejects(async () => {
    for await (const event of provider.chat({ transcript: [], tools: [] })) {
      seen.push(event);
    }
  });
  assert.deepEqual(seen, [{ kind: "text", text: "half an answer" }]);
});

test("a healthy primary is never diverted", async () => {
  const healthy: Provider = {
    ...createScriptedProvider({
      turns: [[{ kind: "text", text: "from opus" }, { kind: "end", reason: "end_turn" }]],
    }),
    name: "anthropic",
    model: "claude-opus-5-5",
  };
  let fallbackCalls = 0;
  const fallback: Provider = {
    name: "ollama",
    model: "llama3.1:8b",
    chat() {
      fallbackCalls++;
      return localProvider().chat({ transcript: [], tools: [] });
    },
  };

  const provider = createFailoverProvider({ primary: healthy, fallback });
  const events = await drain(provider.chat({ transcript: [], tools: [] }));

  assert.equal(fallbackCalls, 0);
  assert.equal(provider.model, "claude-opus-5-5");
  assert.deepEqual(events[0], { kind: "text", text: "from opus" });
});

// -- both together ------------------------------------------------------

test("a degraded Run records rows against the model that answered", async () => {
  const ledger = openLedger(":memory:");
  const fallback: Provider = {
    ...createScriptedProvider({
      turns: [
        [
          { kind: "text", text: "local answer" },
          { kind: "usage", in: 100, cacheWrite: 0, cacheRead: 0, out: 20 },
          { kind: "end", reason: "end_turn" },
        ],
      ],
    }),
    name: "ollama",
    model: "llama3.1:8b",
  };

  const result = await runLoop(
    {
      provider: createFailoverProvider({ primary: creditsGone(), fallback }),
      registry: createRegistry([]),
      ledger,
      maxTurns: 5,
    },
    { agent: "site", goal: "g", budgetUsd: 1 },
  );

  assert.equal(result.status, "done");
  const costs = ledger.listCosts(result.runId);
  assert.equal(costs.length, 1);
  // Not "anthropic"/"claude-opus-5-5": the row has to name what ran, or the
  // Ledger misprices the turn by a factor of forty.
  assert.equal(costs[0]?.provider, "ollama");
  assert.equal(costs[0]?.model, "llama3.1:8b");
  assert.equal(costs[0]?.usd, 0);
  ledger.close();
});
