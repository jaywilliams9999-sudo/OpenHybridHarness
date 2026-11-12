/**
 * ToolRegistry — name, schema, risk tier, handler.
 *
 * Every call writes its tier into `tool_calls`, so the data exists from the first
 * turn. The Loop enforces `gated` by stopping the Run before the call runs;
 * the registry itself only records.
 */

import type { RiskTier } from "../ledger/ledger.ts";

/** What a handler knows about the run it is part of. */
export interface ToolContext {
  readonly runId: string;
}

export interface ToolDefinition {
  readonly name: string;
  readonly description: string;
  readonly schema: unknown;
  readonly tier: RiskTier;
  /**
   * Checked for `gated` calls only, before an approval is requested. A message
   * answers the call as an error with no approval row: asking a human to
   * approve a call that cannot succeed wastes their attention and teaches them
   * to approve without reading.
   */
  validate?(args: Record<string, unknown>, ctx: ToolContext): string | undefined;
  /**
   * Runs alone in its turn. Every other request beside it is answered with an
   * error and never runs, so a call that blocks the Run (a dispatched child
   * waiting on a human) never strands siblings half-applied behind it.
   */
  readonly exclusive?: boolean;
  handler(args: Record<string, unknown>, ctx: ToolContext): Promise<unknown>;
}

export interface ToolRegistry {
  get(name: string): ToolDefinition | undefined;
  list(): ToolDefinition[];
  /** The shape a provider needs to advertise the tools. */
  forProvider(): Array<{ name: string; description: string; schema: unknown }>;
}

export function createRegistry(tools: readonly ToolDefinition[]): ToolRegistry {
  const byName = new Map<string, ToolDefinition>();
  for (const tool of tools) {
    if (byName.has(tool.name)) {
      throw new Error(`duplicate tool registered: ${tool.name}`);
    }
    byName.set(tool.name, tool);
  }

  return {
    get: (name) => byName.get(name),
    list: () => [...byName.values()],
    forProvider: () =>
      [...byName.values()].map(({ name, description, schema }) => ({
        name,
        description,
        schema,
      })),
  };
}

const TOOL_ERROR = Symbol("tool error");

/**
 * Mark a handler's returned value as an error result without throwing.
 *
 * A thrown error loses the value -- the model would see only the message --
 * and some failures carry more than that: a failed child's report says what
 * it did and spent before it failed. The mark is a non-enumerable symbol, so
 * the value records, serialises and compares exactly as it would unmarked.
 */
export function errorResult<T extends object>(value: T): T {
  Object.defineProperty(value, TOOL_ERROR, { value: true, enumerable: false });
  return value;
}

export function isErrorResult(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { [TOOL_ERROR]?: unknown })[TOOL_ERROR] === true
  );
}
