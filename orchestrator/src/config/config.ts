/**
 * Configuration. `models.toml` is the single source of truth for the GPU service's
 * address — the orchestrator reads it rather than keeping a second copy, so there is
 * no way for the two to disagree. `OHH_GPU_URL` overrides it for the cases
 * the registry cannot describe (a tunnel, a local stub, a test server).
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { parseToml, TomlSubsetError } from "./toml.ts";

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

/**
 * `routed` is the real thing: each agent's `model` line sends its runs to
 * Claude or to the local GPU (see `gateway/route.ts`). With no Anthropic key
 * every run goes local -- local-only mode, real models, not a script.
 * `scripted` is a fixed script for tests and demos, chosen only explicitly.
 */
export type ProviderMode = "routed" | "scripted";

export interface AnthropicConfig {
  readonly apiKey: string;
  readonly model: string;
  readonly maxTokens: number;
  readonly effort?: "low" | "medium" | "high" | "xhigh" | "max";
  readonly cacheTtl: "5m" | "1h";
}

export interface OllamaProviderConfig {
  /** Unset lets the GPU service pick its `role = "code"` model from the registry. */
  readonly model?: string;
  /** Unset uses the model's own registry window. */
  readonly numCtx?: number;
  readonly temperature?: number;
  readonly maxTokens?: number;
}

export interface Config {
  readonly gpuUrl: string;
  readonly gpuToken: string;
  readonly ledgerPath: string;
  readonly artifactsDir: string;
  /** Where `publish` copies approved artifacts. */
  readonly publishedDir: string;
  /** Where `agents/<name>.toml` live. */
  readonly agentsDir: string;
  /** One pid file per running root: who owns it. */
  readonly runningDir: string;
  readonly maxTurns: number;
  /** Per-Run ceiling in USD. Reaching it blocks at a turn boundary. */
  readonly budgetUsd: number;
  readonly mode: ProviderMode;
  /** Present iff `OHH_ANTHROPIC_API_KEY` is set, whatever the mode. */
  readonly anthropic?: AnthropicConfig;
  /** Always present: every local run, and every fallback, uses it. */
  readonly ollama: OllamaProviderConfig;
  /** The `[models."…"]` names in `models.toml`: what an agent may name to run locally. */
  readonly localModels: ReadonlySet<string>;
}

const EFFORTS = new Set(["low", "medium", "high", "xhigh", "max"]);

/**
 * The orchestrator's own name for its key, deliberately not `ANTHROPIC_API_KEY`.
 *
 * That name is also read by Claude Code, where it takes precedence over a
 * subscription login. Exported for the orchestrator as a User environment variable,
 * it moved every Claude Code window on the machine onto metered API billing
 * and drained the credit (2026-10-18). A key only this process reads, loaded
 * from the repo-root `.env` rather than the shell, cannot leak that way.
 */
export const API_KEY_VAR = "OHH_ANTHROPIC_API_KEY";

/**
 * Routed unless the operator explicitly asks for the script. A missing key is
 * not a reason to fall back to the script: it means local-only mode.
 */
function resolveMode(env: Env): ProviderMode {
  const explicit = env["OHH_PROVIDER"];
  if (explicit === undefined || explicit === "") return "routed";
  if (explicit !== "scripted") {
    throw new ConfigError(`OHH_PROVIDER must be scripted or unset; got "${explicit}".`);
  }
  return "scripted";
}

/** The one line the startup log and the hub print about where runs go. */
export function describeMode(config: Pick<Config, "mode" | "anthropic">): string {
  if (config.mode === "scripted") return "mode: scripted";
  return config.anthropic
    ? "mode: routed (Claude + local)"
    : "mode: local-only (no Anthropic key configured)";
}

function resolveAnthropic(env: Env): AnthropicConfig {
  const effort = env["OHH_EFFORT"];
  if (effort !== undefined && !EFFORTS.has(effort)) {
    throw new ConfigError(
      `OHH_EFFORT must be one of ${[...EFFORTS].join(", ")}; got "${effort}".`,
    );
  }

  const ttl = env["OHH_CACHE_TTL"] ?? "5m";
  if (ttl !== "5m" && ttl !== "1h") {
    throw new ConfigError(`OHH_CACHE_TTL must be 5m or 1h; got "${ttl}".`);
  }

  const maxTokensRaw = env["OHH_MAX_TOKENS"];

  return {
    apiKey: env[API_KEY_VAR] as string,
    model: env["OHH_MODEL"] ?? "claude-opus-5-5",
    maxTokens: maxTokensRaw ? positiveInt(maxTokensRaw, "OHH_MAX_TOKENS") : 16_384,
    ...(effort !== undefined
      ? { effort: effort as NonNullable<AnthropicConfig["effort"]> }
      : {}),
    cacheTtl: ttl,
  };
}

export type Env = Record<string, string | undefined>;

/** `orchestrator/` — resolved from this module, so cwd cannot change the answer. */
const ORCHESTRATOR_DIR = join(import.meta.dirname, "..", "..");

/**
 * `OHH_MODELS` if set, else the repo-root `models.toml` -- the same
 * precedence as the GPU service, so the two processes read one registry and
 * agree on which model names are local.
 */
export function defaultModelsPath(env: Env = process.env): string {
  const override = env["OHH_MODELS"];
  if (override) return override;
  return join(ORCHESTRATOR_DIR, "..", "models.toml");
}

function readRegistry(path: string): Record<string, unknown> {
  try {
    return parseToml(readFileSync(path, "utf8"));
  } catch (error) {
    if (error instanceof TomlSubsetError) {
      throw new ConfigError(`${path} is not readable: ${error.message}`);
    }
    throw error;
  }
}

function gpuUrlFromRegistry(doc: Record<string, unknown>, path: string): string {
  const gpu = doc["gpu_service"];
  if (typeof gpu !== "object" || gpu === null) {
    throw new ConfigError(`${path} has no [gpu_service] table.`);
  }
  const { host, port } = gpu as Record<string, unknown>;

  if (typeof host !== "string" || host === "") {
    throw new ConfigError(`${path} has no [gpu_service] host.`);
  }
  if (typeof port !== "number") {
    throw new ConfigError(`${path} has no numeric [gpu_service] port.`);
  }
  return `http://${host}:${port}`;
}

/** The `[models."…"]` table names: the models an agent may name to run locally. */
function localModelsFromRegistry(doc: Record<string, unknown>): Set<string> {
  const models = doc["models"];
  if (typeof models !== "object" || models === null || Array.isArray(models)) {
    return new Set();
  }
  return new Set(Object.keys(models));
}

function positiveNumber(raw: string, name: string): number {
  // Not positiveInt: a budget is dollars and cents, and $0.05 is a realistic
  // cap for a single Haiku sub-task.
  if (!/^\d+(\.\d+)?$/.test(raw)) {
    throw new ConfigError(
      `${name} must be a non-negative number of dollars, got "${raw}".`,
    );
  }
  return Number(raw);
}

function positiveInt(raw: string, name: string): number {
  if (!/^\d+$/.test(raw)) {
    throw new ConfigError(`${name} must be a whole number, got "${raw}".`);
  }
  return Number(raw);
}

export function loadConfig(
  env: Env = process.env,
  registryPath: string = defaultModelsPath(env),
): Config {
  const gpuToken = env["OHH_GPU_TOKEN"];
  if (!gpuToken) {
    throw new ConfigError(
      "OHH_GPU_TOKEN is not set. The GPU service loads and executes models on " +
        "request and must never be dialled unauthenticated, even on a private network.",
    );
  }

  const override = env["OHH_GPU_URL"];
  if (!override && !existsSync(registryPath)) {
    throw new ConfigError(
      `No registry at ${registryPath}, and OHH_GPU_URL is not set. ` +
        `The orchestrator takes the GPU service's address from the registry; ` +
        `set OHH_GPU_URL to override it.`,
    );
  }
  // With an override the registry is optional: it is read for model names
  // when it exists, and an absent one simply means no named local models.
  const registry = existsSync(registryPath) ? readRegistry(registryPath) : undefined;
  const gpuUrl = override
    ? override.replace(/\/+$/, "")
    : gpuUrlFromRegistry(registry ?? {}, registryPath);

  const maxTurnsRaw = env["OHH_MAX_TURNS"];
  const budgetRaw = env["OHH_BUDGET_USD"];
  const mode = resolveMode(env);

  const ledgerPath = env["OHH_LEDGER"] ?? join(ORCHESTRATOR_DIR, "ledger.db");

  return {
    gpuUrl,
    gpuToken,
    ledgerPath,
    // Beside the Ledger, so every process sharing a Ledger shares this too.
    runningDir: env["OHH_RUNNING_DIR"] ?? join(dirname(ledgerPath), "running"),
    artifactsDir: env["OHH_ARTIFACTS"] ?? join(ORCHESTRATOR_DIR, "artifacts"),
    publishedDir: env["OHH_PUBLISHED"] ?? join(ORCHESTRATOR_DIR, "published"),
    agentsDir: env["OHH_AGENTS"] ?? join(ORCHESTRATOR_DIR, "agents"),
    maxTurns: maxTurnsRaw ? positiveInt(maxTurnsRaw, "OHH_MAX_TURNS") : 20,
    // 2.00 is roughly five 20-turn Opus Runs at
    // the measured ~$0.42 each, so a runaway loop costs a few dollars rather
    // than a month's credit.
    budgetUsd: budgetRaw ? positiveNumber(budgetRaw, "OHH_BUDGET_USD") : 2,
    mode,
    ...(env[API_KEY_VAR] ? { anthropic: resolveAnthropic(env) } : {}),
    ollama: resolveOllama(env),
    localModels: registry ? localModelsFromRegistry(registry) : new Set(),
  };
}

function resolveOllama(env: Env): OllamaProviderConfig {
  const numCtx = env["OHH_NUM_CTX"];
  const maxTokens = env["OHH_MAX_TOKENS"];
  const temperature = env["OHH_TEMPERATURE"];

  if (temperature !== undefined && !/^\d+(\.\d+)?$/.test(temperature)) {
    throw new ConfigError(
      `OHH_TEMPERATURE must be a non-negative number, got "${temperature}".`,
    );
  }

  // No `model`: which local model a run uses is the agent file's business
  // (its `model` line, or the registry's `role = "code"` default).
  return {
    ...(numCtx !== undefined
      ? { numCtx: positiveInt(numCtx, "OHH_NUM_CTX") }
      : {}),
    ...(maxTokens !== undefined
      ? { maxTokens: positiveInt(maxTokens, "OHH_MAX_TOKENS") }
      : {}),
    ...(temperature !== undefined ? { temperature: Number(temperature) } : {}),
  };
}
