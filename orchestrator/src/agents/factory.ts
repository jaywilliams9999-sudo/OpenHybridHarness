/**
 * The agent factory: an agent name and a depth in, a runnable agent out.
 *
 * One agent used to be assembled inline in `main.ts`. A tree needs the same
 * assembly in three places -- the root of a fresh run, a child that `dispatch`
 * starts, and any Run a resume wakes from a fresh process -- and the three
 * must agree, or a resumed child would come back as whatever agent the command
 * line happened to default to. So it lives here, once.
 */

import type { Config } from "../config/config.ts";
import type { GpuClient } from "../gpu/client.ts";
import type { Ledger, RunRow } from "../ledger/ledger.ts";
import type { LoopDeps } from "../loop/loop.ts";
import type { Provider } from "../gateway/types.ts";
import { createRegistry } from "../tools/registry.ts";
import { createGpuTools } from "../tools/gpu-tools.ts";
import { createPublishTool } from "../tools/publish.ts";
import { createDispatchTool, type ChildAgent } from "../tools/dispatch.ts";
import { buildSystem, loadAgent, restrictRegistry } from "./definition.ts";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { createScriptedProvider } from "../gateway/scripted.ts";
import { createAnthropicProvider } from "../gateway/anthropic.ts";
import { createOllamaProvider } from "../gateway/ollama.ts";
import { createFailoverProvider } from "../gateway/failover.ts";
import {
  resumeRoute,
  routeAgent,
  type Placement,
  type Route,
} from "../gateway/route.ts";

/**
 * Appended to the system prompt of every agent that runs on the local model.
 *
 * `llama3.1:8b` drives tools well when told plainly how; left alone it writes
 * calls into its reply as JSON, fires several at once, or narrates instead of
 * finishing. `normalizeToolCalls` recovers the first; this asks for none of it.
 * A Claude-placed run does not get it -- it needs no such coaching -- and
 * neither does a local agent with no tools.
 */
export const LOCAL_ADDENDUM =
  "Use the provided tools by calling them, never by writing JSON in your reply. Call one tool at a time and wait for its result. When the task is done, reply with a short plain-text summary.";

/**
 * Appended to an agent's system prompt when the card cannot fit the image
 * model. The dispatch tool's description says it too, but a small model reads
 * its own instructions more reliably than a tool's, and the Lead's role is
 * kept generic: what this card cannot do is a fact of the run, not the agent.
 */
export const IMAGE_UNAVAILABLE =
  "Image generation is unavailable on this card: the image model doesn't fit. Ask no agent for an image, and tell the operator the image was skipped.";

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;

/**
 * The provider for one run, from the agent's route.
 *
 * The scripted arm is not dead code: it is the deterministic path the tests
 * and demos drive, selected only by `OHH_PROVIDER=scripted`. In routed mode a
 * Claude-placed agent gets Claude with the local model behind it, and a local
 * one -- including a Claude-placed agent on a machine with no key -- gets the
 * local model through the GPU service.
 */
export function buildProvider(
  config: Config,
  route: Route,
  gpu: GpuClient,
  prompt: string,
): Provider {
  if (config.mode === "routed" && route.placement === "claude" && config.anthropic) {
    const { apiKey, model, maxTokens, effort, cacheTtl } = config.anthropic;
    const primary = createAnthropicProvider({
      apiKey,
      // The model-tiering seam. An agent doing cheap, bounded work names a cheap
      // model here, and the expensive one is spent only where judgement is
      // actually needed. Switching *within* a transcript would be the wrong
      // lever -- it throws away the cached prefix, which costs far more than
      // the rate difference saves -- so the choice belongs to the agent, once.
      model: route.model ?? model,
      maxTokens,
      cacheTtl,
      ...(effort !== undefined ? { effort } : {}),
    });
    // The monthly credit is a hard ceiling that does not roll over, and
    // reaching it should degrade rather than stop. The local model costs
    // nothing, so an exhausted balance means a worse answer rather than none.
    // The GPU service's health check has already passed by the time we get here.
    return createFailoverProvider({
      primary,
      fallback: createOllamaProvider({ gpu, ...config.ollama }),
      onDegrade: ({ to, reason, limit }) => {
        // Named apart on purpose: a configured limit is a setting someone can
        // raise in a minute, and an exhausted balance is a wait for the month
        // to roll. Printing both as "credits exhausted" would send an operator
        // to the wrong place.
        const headline =
          limit === "configured" ? "spend limit reached" : "credits exhausted";
        console.log(`
  ${red(headline)} ${dim(reason)}`);
        console.log(`  ${cyan("degraded")} to ${to} ${dim("(credit ceiling)")}`);
      },
    });
  }

  if (config.mode === "routed") {
    // Reaches local inference through the GPU service, never port 11434, so
    // the GPU service stays the one owner of VRAM. No model means the
    // registry's `role = "code"` model.
    return createOllamaProvider({
      gpu,
      ...config.ollama,
      ...(route.model ? { model: route.model } : {}),
    });
  }

  // A fixed three-turn script standing in for a model: look at the card, decide
  // an asset is needed, then report.
  return createScriptedProvider({
    turns: [
      [
        { kind: "thinking", text: "checking what the card is holding first" },
        { kind: "tool_request", id: "t1", name: "gpu_status", input: {} },
        { kind: "end", reason: "tool_use" },
      ],
      [
        { kind: "text", text: "the card has room; rendering the asset" },
        {
          kind: "tool_request",
          id: "t2",
          name: "gpu_generate_image",
          input: { prompt },
        },
        { kind: "end", reason: "tool_use" },
      ],
      [
        { kind: "text", text: "asset rendered and recorded in the Ledger" },
        { kind: "end", reason: "end_turn" },
      ],
    ],
  });
}

/**
 * The agents an agent may dispatch to: every loadable one in the directory
 * that cannot itself dispatch (nesting is refused). Unloadable files are
 * skipped, not fatal.
 */
export function dispatchableAgents(
  agentsDir: string,
  imageAvailable = true,
): Array<{ name: string; role: string }> {
  let files: string[];
  try {
    files = readdirSync(agentsDir).filter((f) => f.endsWith(".toml")).sort();
  } catch {
    return [];
  }
  const found: Array<{ name: string; role: string }> = [];
  for (const file of files) {
    try {
      const agent = loadAgent(agentsDir, file.slice(0, -".toml".length));
      if (!imageAvailable && agent.tools.includes("gpu_generate_image")) continue;
      if (!agent.tools.includes("dispatch")) {
        found.push({ name: agent.name, role: agent.role });
      }
    } catch {
      // A broken agent file is not a reason to take lead down.
    }
  }
  return found;
}

/** Names of loadable agents that carry the image tool. */
export function imageAgentNames(agentsDir: string): string[] {
  const all = dispatchableAgents(agentsDir, true).map((a) => a.name);
  const shown = new Set(dispatchableAgents(agentsDir, false).map((a) => a.name));
  return all.filter((n) => !shown.has(n));
}

/** The agent's system prompt with what this run adds; the addendum stays last. */
function systemFor(base: string, addendum: boolean, noImages: boolean): string {
  const parts = [base];
  if (noImages) parts.push(IMAGE_UNAVAILABLE);
  if (addendum) parts.push(LOCAL_ADDENDUM);
  return parts.join("\n\n");
}

export interface AgentFactoryOptions {
  readonly config: Config;
  readonly gpu: GpuClient;
  readonly ledger: Ledger;
  /** The shared half of every system prompt; the agent adds its role. */
  readonly harness: string;
  /** Display callbacks for an agent at this depth. */
  readonly output: (
    depth: number,
  ) => Pick<LoopDeps, "onEvent" | "onToolCall" | "onCost">;
  /** Told when an agent at `depth - 1` dispatches a child at `depth`. */
  readonly onChildCreated?: (
    depth: number,
    child: { id: string; agent: string; allocationUsd: number },
  ) => void;
  /** The operator's prompt; only the scripted provider reads it. */
  readonly prompt?: string;
  /** False when the image model doesn't fit the card (default true). */
  readonly imageAvailable?: boolean;
}

export interface AgentFactory {
  /**
   * `resumedFrom` is the placement on the row of a run being resumed: the
   * agent is then placed where that run started (`resumeRoute`), not where a
   * fresh run would go today.
   */
  agent(agentName: string, depth: number, resumedFrom?: Placement): ChildAgent;
}

export function createAgentFactory(opts: AgentFactoryOptions): AgentFactory {
  const { config, gpu, ledger, harness, output, onChildCreated } = opts;
  // Scripted mode only: one agent per (agent, depth), so a resume within this
  // process reuses the scripted provider that already ran and it keeps its
  // place. In routed mode every call builds a fresh provider, because
  // fallback is per run -- one run reaching the spending wall must not move
  // the next run of the same agent onto the local model.
  const cache = new Map<string, ChildAgent>();

  function buildAgent(
    agentName: string,
    depth: number,
    resumedFrom?: Placement,
  ): ChildAgent {
    const key = `${agentName}\u0000${depth}`;
    const cached = config.mode === "scripted" ? cache.get(key) : undefined;
    if (cached) return cached;

    const agent = loadAgent(config.agentsDir, agentName);
    // At load time, before any run starts: an unknown model names the file.
    const hasKey = config.anthropic !== undefined;
    const today = routeAgent(join(config.agentsDir, `${agentName}.toml`), agent.model, {
      localModels: config.localModels,
      hasKey,
    });
    const route =
      resumedFrom === undefined ? today : resumeRoute(resumedFrom, today, hasKey);
    const registry = restrictRegistry(
      createRegistry([
        ...createGpuTools({ gpu, ledger, artifactsDir: config.artifactsDir }),
        createPublishTool({ ledger, publishedDir: config.publishedDir }),
        createDispatchTool({
          ledger,
          agents: dispatchableAgents(config.agentsDir, opts.imageAvailable ?? true),
          imageAvailable: opts.imageAvailable ?? true,
          hiddenImageAgents:
            opts.imageAvailable === false ? imageAgentNames(config.agentsDir) : [],
          // Lazy: the child is built when dispatch runs, not now, so building
          // an agent never recurses through every agent it could reach.
          spawn: (name) => buildAgent(name, depth + 1),
          ...(onChildCreated
            ? {
                onChildCreated: (id: string, agent: string, allocationUsd: number) =>
                  onChildCreated(depth + 1, { id, agent, allocationUsd }),
              }
            : {}),
        }),
      ]),
      agent,
    );

    // Precedence at the root: an explicitly-set environment variable beats
    // the agent file, which beats the built-in default. Below the root the
    // variable does not apply -- the operator overrode the run they started,
    // not every agent it happens to dispatch.
    const maxTurns =
      depth === 0 && process.env["OHH_MAX_TURNS"] !== undefined
        ? config.maxTurns
        : (agent.limits.maxTurns ?? config.maxTurns);

    const built: ChildAgent = {
      agent,
      system: systemFor(
        buildSystem(harness, agent),
        // Only where there are tools to coach: told to "use the provided
        // tools", a tool-less local agent (scribe) narrated tool calls it
        // cannot make.
        route.placement !== "claude" && registry.list().length > 0,
        // Only for an agent that can dispatch: it is the one deciding who
        // to ask for what.
        opts.imageAvailable === false && agent.tools.includes("dispatch"),
      ),
      placement: route.placement,
      deps: {
        provider: buildProvider(config, route, gpu, opts.prompt ?? ""),
        placement: route.placement,
        registry,
        ledger,
        maxTurns,
        ...output(depth),
      },
    };
    if (config.mode === "scripted") cache.set(key, built);
    return built;
  }

  return { agent: buildAgent };
}

/** How many `parent_id` hops separate a Run from the root of its tree. */
export function depthOf(ledger: Ledger, run: RunRow): number {
  let depth = 0;
  let parentId = run.parent_id;
  const seen = new Set<string>([run.id]);
  while (parentId) {
    if (seen.has(parentId)) {
      throw new Error(`Run ${run.id} has a cycle in its parent chain`);
    }
    seen.add(parentId);
    depth += 1;
    parentId = ledger.getRun(parentId)?.parent_id ?? null;
  }
  return depth;
}

/**
 * The deps a resume uses for one Run: its own agent, at its own depth, placed
 * where the Run started.
 *
 * The Run row names its agent, so a child resumed from a fresh process comes
 * back with its own provider and tools, whatever `--agent` said. It also
 * names the placement, so adding a key never moves a local run onto Claude.
 * A Claude run resumed with no key runs locally, and its row is updated to
 * `local-no-key` so the Ledger says why.
 */
export function runDeps(
  factory: AgentFactory,
  ledger: Ledger,
  run: RunRow,
): LoopDeps {
  const built = factory.agent(run.agent, depthOf(ledger, run), run.placement);
  if (run.placement === "claude" && built.placement === "local-no-key") {
    ledger.setRunPlacement(run.id, "local-no-key");
  }
  return built.deps;
}
