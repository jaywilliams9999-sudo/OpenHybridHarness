/**
 * What a turn cost. The budget spends this number, so it lives in one
 * table with its provenance attached rather than inline in the adapter.
 *
 * Every figure is per million tokens, quoted from Anthropic's model reference
 * as of **2026-10-15**. Prices move; `costs.usage_json` keeps the raw token
 * counts so a wrong table here can be re-derived later instead of having
 * silently corrupted the Ledger.
 *
 * Two shapes in this table are easy to get wrong and both change the maths:
 *
 * - **Cache reads are $0.20 on both Opus 5.5 and Sonnet 5.5.** Sonnet is *not*
 *   uniformly half of Opus: input and output halve, cache reads do not. On a
 *   cache-heavy agent loop — which is every long Run — that pushes the real
 *   Opus:Sonnet ratio to about 1.7x, not 2x.
 * - **Haiku 5.5 has a price cliff**, not a flat rate: $0.10/$0.50 while the
 *   prompt is <=100K tokens and $0.50/$2.50 above it, a 5x step. A Haiku
 *   sub-task that grows past 100K tokens stops being cheap, which is the
 *   quantitative reason to keep delegated work on short transcripts.
 */

export type CacheTtl = "5m" | "1h";

export interface Usage {
  readonly in: number;
  readonly cacheWrite: number;
  readonly cacheRead: number;
  readonly out: number;
}

export interface ModelPrice {
  readonly inPerMTok: number;
  readonly outPerMTok: number;
  readonly cacheReadPerMTok: number;
  /**
   * Above this many *prompt* tokens the long-context rates apply. Only Haiku
   * 5.5 publishes one; `undefined` means a single flat rate at any length.
   */
  readonly cliffTokens?: number;
  readonly longInPerMTok?: number;
  readonly longOutPerMTok?: number;
}

/** A cache write bills at a multiple of the model's base input rate. */
const CACHE_WRITE_MULTIPLIER: Record<CacheTtl, number> = {
  "5m": 1.25,
  "1h": 2.0,
};

export const PRICES: Readonly<Record<string, ModelPrice>> = {
  "claude-opus-5-5": {
    inPerMTok: 4,
    outPerMTok: 20,
    cacheReadPerMTok: 0.2,
  },
  "claude-sonnet-5-5": {
    inPerMTok: 2,
    outPerMTok: 10,
    cacheReadPerMTok: 0.2,
  },
  "claude-haiku-5-5": {
    inPerMTok: 0.1,
    outPerMTok: 0.5,
    // Inferred at 0.1x input, not quoted: Anthropic's reference states Haiku
    // 5.5's token rates but not its cache-read rate. The ratio is not constant
    // across the line (Opus 5.5 is 0.05x, Sonnet 5.5 is 0.1x), so this is the
    // one number here that is an assumption. It is also the smallest: at
    // $0.01/MTok it cannot move a budget decision.
    cacheReadPerMTok: 0.01,
    cliffTokens: 100_000,
    longInPerMTok: 0.5,
    longOutPerMTok: 2.5,
  },
};

/**
 * Zero, deliberately, for a model we have no price for.
 *
 * The alternative — guessing, or throwing — are both worse. A guess writes
 * fiction into the Ledger that later looks like evidence; throwing fails a Run
 * that already succeeded, after the money was spent. A zero is visibly wrong
 * in `costs` next to a non-null `usage_json`, which is the one state an
 * operator can actually correct.
 */
export function priceFor(model: string): ModelPrice | undefined {
  return PRICES[model];
}

export function costUsd(
  model: string,
  usage: Usage,
  ttl: CacheTtl = "5m",
): number {
  const price = priceFor(model);
  if (!price) return 0;

  // The cliff is measured against the whole prompt, which is every input token
  // the request carried however it was billed — fresh, written to cache, or
  // read from it.
  const promptTokens = usage.in + usage.cacheWrite + usage.cacheRead;
  const overCliff =
    price.cliffTokens !== undefined && promptTokens > price.cliffTokens;

  const inRate = overCliff ? (price.longInPerMTok ?? price.inPerMTok) : price.inPerMTok;
  const outRate = overCliff
    ? (price.longOutPerMTok ?? price.outPerMTok)
    : price.outPerMTok;

  const perMTok =
    usage.in * inRate +
    usage.cacheWrite * inRate * CACHE_WRITE_MULTIPLIER[ttl] +
    usage.cacheRead * price.cacheReadPerMTok +
    usage.out * outRate;

  return perMTok / 1_000_000;
}
