/**
 * Agents — the agents themselves, declared in TOML rather than in `main.ts`.
 *
 * An agent is a name, a job, the tools it may use and what it may spend. Before
 * this file those four things were constants in `main.ts`, which meant the
 * harness could run exactly one agent and changing it meant editing code.
 *
 * **Why the tool list is a filter and not a suggestion.** The registry knows
 * every tool the orchestrator can perform; an agent declares the subset *this* agent
 * is permitted. The model is only ever shown the subset, so a tool absent from
 * the list is not refused at call time — it does not exist. That is the
 * difference between a capability boundary and a reminder, and it is the seam
 * the risk tiers eventually hang off: an agent that cannot see `shell.git`
 * cannot be talked into reaching for it.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { parseToml, TomlSubsetError } from "../config/toml.ts";
import type { ToolRegistry } from "../tools/registry.ts";

export class AgentError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentError";
  }
}

export interface AgentLimits {
  /** Per-Run ceiling in dollars, or undefined to take the config default. */
  readonly maxUsd?: number;
  readonly maxTurns?: number;
}

export interface AgentDefinition {
  readonly name: string;
  /** Prepended to the harness system prompt: what *this* agent is for. */
  readonly role: string;
  /**
   * Decides placement (`gateway/route.ts`): a `models.toml` name runs locally,
   * a `claude-*` name runs on Claude. Absent means the local default model.
   */
  readonly model?: string;
  /**
   * Declared in the agent file and carried, but nothing consumes it yet: no tool in the
   * orchestrator touches the filesystem. It becomes load-bearing the day `fs.write`
   * exists, and is parsed now so an agent written today does not need editing
   * then.
   */
  readonly workdir?: string;
  readonly tools: readonly string[];
  readonly limits: AgentLimits;
}

function table(doc: Record<string, unknown>, key: string, path: string): Record<string, unknown> {
  const value = doc[key];
  if (value === undefined) {
    throw new AgentError(`${path} has no [${key}] table.`);
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new AgentError(`${path}: [${key}] is not a table.`);
  }
  return value as Record<string, unknown>;
}

function requiredString(
  from: Record<string, unknown>,
  key: string,
  where: string,
  path: string,
): string {
  const value = from[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new AgentError(`${path}: [${where}] ${key} must be a non-empty string.`);
  }
  return value;
}

function optionalString(
  from: Record<string, unknown>,
  key: string,
  where: string,
  path: string,
): string | undefined {
  const value = from[key];
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.trim() === "") {
    throw new AgentError(`${path}: [${where}] ${key} must be a non-empty string.`);
  }
  return value;
}

function optionalCount(
  from: Record<string, unknown>,
  key: string,
  path: string,
): number | undefined {
  const value = from[key];
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new AgentError(`${path}: [limits] ${key} must be a positive whole number.`);
  }
  return value;
}

/**
 * Read one agent from disk.
 *
 * Every failure names the file and the field. An agent is operator-authored
 * configuration that decides what an autonomous agent may do and spend, so a
 * malformed one must stop the run rather than fall back to a default: the
 * default would be *more* permissive than what the operator wrote, which is
 * the wrong direction to guess in.
 */
export function loadAgent(dir: string, name: string): AgentDefinition {
  if (!/^[a-z0-9][a-z0-9_-]*$/i.test(name)) {
    throw new AgentError(
      `"${name}" is not a usable agent name — letters, digits, dash and ` +
        `underscore only. The name becomes a filename.`,
    );
  }

  const path = join(dir, `${name}.toml`);
  if (!existsSync(path)) {
    throw new AgentError(`No agent at ${path}.`);
  }

  let doc: Record<string, unknown>;
  try {
    doc = parseToml(readFileSync(path, "utf8"));
  } catch (error) {
    if (error instanceof TomlSubsetError) {
      throw new AgentError(`${path} is not readable: ${error.message}`);
    }
    throw error;
  }

  const agent = table(doc, "agent", path);
  const declaredName = requiredString(agent, "name", "agent", path);
  if (declaredName !== name) {
    // The Ledger keys Runs by this name, so a file whose contents disagree
    // with its filename would file a run under an agent nobody asked for.
    throw new AgentError(
      `${path}: [agent] name is "${declaredName}" but the file is ${name}.toml.`,
    );
  }

  const rawTools = agent["tools"];
  if (!Array.isArray(rawTools) || rawTools.some((t) => typeof t !== "string")) {
    throw new AgentError(`${path}: [agent] tools must be an array of strings.`);
  }
  const tools = rawTools as string[];
  const duplicate = tools.find((t, i) => tools.indexOf(t) !== i);
  if (duplicate !== undefined) {
    throw new AgentError(`${path}: [agent] tools lists ${duplicate} twice.`);
  }

  // `[limits]` is optional as a whole; an absent table means config defaults.
  const limitsTable = doc["limits"];
  if (
    limitsTable !== undefined &&
    (typeof limitsTable !== "object" || limitsTable === null || Array.isArray(limitsTable))
  ) {
    throw new AgentError(`${path}: [limits] is not a table.`);
  }
  const limitsRaw = (limitsTable ?? {}) as Record<string, unknown>;

  // Money arrives as whole cents because the TOML subset refuses floats, and
  // that refusal is deliberate: it guards the registry's VRAM figures, where a
  // silently-coerced number is the exact bug class that cost this project a
  // debugging session. Cents are also simply the honest unit for a cap -- an
  // authored "0.1" is not 0.1, and a budget is a number you compare against.
  const cents = optionalCount(limitsRaw, "max_usd_cents", path);
  const maxTurns = optionalCount(limitsRaw, "max_turns", path);

  const model = optionalString(agent, "model", "agent", path);
  const workdir = optionalString(agent, "workdir", "agent", path);

  return {
    name: declaredName,
    role: requiredString(agent, "role", "agent", path),
    ...(model !== undefined ? { model } : {}),
    ...(workdir !== undefined ? { workdir } : {}),
    tools,
    limits: {
      ...(cents !== undefined ? { maxUsd: cents / 100 } : {}),
      ...(maxTurns !== undefined ? { maxTurns } : {}),
    },
  };
}

/**
 * Narrow a registry to the tools an agent declares.
 *
 * An unknown name is fatal rather than skipped. Dropping it silently would
 * leave an agent missing a capability its author believed it had, and the
 * symptom -- a model that inexplicably refuses to do its job -- points nowhere
 * near the typo that caused it.
 */
export function restrictRegistry(
  registry: ToolRegistry,
  agent: AgentDefinition,
): ToolRegistry {
  const available = new Set(registry.list().map((tool) => tool.name));
  const unknown = agent.tools.filter((name) => !available.has(name));
  if (unknown.length > 0) {
    throw new AgentError(
      `agent "${agent.name}" lists ${unknown.length === 1 ? "a tool" : "tools"} ` +
        `that do not exist: ${unknown.join(", ")}. ` +
        `Available: ${[...available].sort().join(", ")}.`,
    );
  }

  const permitted = new Set(agent.tools);
  const kept = registry.list().filter((tool) => permitted.has(tool.name));

  return {
    get: (name) => (permitted.has(name) ? registry.get(name) : undefined),
    list: () => [...kept],
    forProvider: () =>
      kept.map(({ name, description, schema }) => ({ name, description, schema })),
  };
}

/**
 * The system prompt for a run: what the harness is, then what this agent is.
 *
 * The first half describes machinery every agent shares and would be wasteful
 * to repeat in each file; the second is the agent's own. Keeping them separate
 * means an operator writing a new agent states only what differs.
 */
export function buildSystem(harness: string, agent: AgentDefinition): string {
  return `${harness}\n\nYour role: ${agent.role}`;
}
