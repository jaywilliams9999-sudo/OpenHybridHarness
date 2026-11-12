import { test } from "node:test";
import assert from "node:assert/strict";

import { costUsd, priceFor, PRICES } from "../src/gateway/pricing.ts";

const MTOK = 1_000_000;

test("a pure-output million tokens costs the published output rate", () => {
  const usd = costUsd("claude-opus-5-5", {
    in: 0,
    cacheWrite: 0,
    cacheRead: 0,
    out: MTOK,
  });
  assert.equal(usd, 20);
});

test("a cache read is a twentieth of a fresh input token on Opus 5.5", () => {
  const fresh = costUsd("claude-opus-5-5", {
    in: MTOK,
    cacheWrite: 0,
    cacheRead: 0,
    out: 0,
  });
  const cached = costUsd("claude-opus-5-5", {
    in: 0,
    cacheWrite: 0,
    cacheRead: MTOK,
    out: 0,
  });
  assert.equal(fresh, 4);
  assert.equal(cached, 0.2);
  assert.equal(fresh / cached, 20);
});

test("a cache write bills at 1.25x input on 5m and 2x on 1h", () => {
  const usage = { in: 0, cacheWrite: MTOK, cacheRead: 0, out: 0 };
  assert.equal(costUsd("claude-opus-5-5", usage, "5m"), 5);
  assert.equal(costUsd("claude-opus-5-5", usage, "1h"), 8);
});

test("Sonnet 5.5 is NOT uniformly half of Opus — cache reads match", () => {
  // The correction that matters for budgeting: input and output halve, cache
  // reads do not, so a cache-heavy Run is ~1.7x cheaper rather than 2x.
  assert.equal(PRICES["claude-sonnet-5-5"]?.inPerMTok, 2);
  assert.equal(PRICES["claude-opus-5-5"]?.inPerMTok, 4);
  assert.equal(
    PRICES["claude-sonnet-5-5"]?.cacheReadPerMTok,
    PRICES["claude-opus-5-5"]?.cacheReadPerMTok,
  );

  // A realistic 20-turn shape: a large reused prefix, a small write, some output.
  const usage = { in: 0, cacheWrite: 30_000, cacheRead: 325_000, out: 10_000 };
  const opus = costUsd("claude-opus-5-5", usage);
  const sonnet = costUsd("claude-sonnet-5-5", usage);
  const ratio = opus / sonnet;
  assert.ok(ratio > 1.6 && ratio < 1.8, `expected ~1.7x, got ${ratio}`);
});

test("Haiku 5.5 crosses a 5x price cliff above 100K prompt tokens", () => {
  const under = costUsd("claude-haiku-5-5", {
    in: 50_000,
    cacheWrite: 0,
    cacheRead: 0,
    out: 1_000,
  });
  const over = costUsd("claude-haiku-5-5", {
    in: 150_000,
    cacheWrite: 0,
    cacheRead: 0,
    out: 1_000,
  });
  // 3x the tokens but far more than 3x the cost: the rate itself changed.
  assert.ok(over / under > 10, `expected a cliff, got ${over / under}x`);
});

test("the cliff is measured against the whole prompt, cached tokens included", () => {
  // A long cached prefix still puts the request over the threshold — the cliff
  // is a property of the prompt, not of what was billed fresh.
  const usage = { in: 1_000, cacheWrite: 0, cacheRead: 120_000, out: 100 };
  const price = priceFor("claude-haiku-5-5");
  assert.ok(price?.cliffTokens === 100_000);
  const usd = costUsd("claude-haiku-5-5", usage);
  // Input is billed at the long rate (0.5), not the short one (0.1).
  const expected =
    (1_000 * 0.5 + 120_000 * 0.01 + 100 * 2.5) / MTOK;
  assert.ok(Math.abs(usd - expected) < 1e-12, `${usd} vs ${expected}`);
});

test("an unknown model prices at zero rather than guessing or throwing", () => {
  // Zero is visibly wrong next to a populated usage_json, which is the one
  // state an operator can correct. A guess would look like evidence.
  assert.equal(
    costUsd("some-model-that-does-not-exist", {
      in: MTOK,
      cacheWrite: MTOK,
      cacheRead: MTOK,
      out: MTOK,
    }),
    0,
  );
  assert.equal(priceFor("some-model-that-does-not-exist"), undefined);
});
