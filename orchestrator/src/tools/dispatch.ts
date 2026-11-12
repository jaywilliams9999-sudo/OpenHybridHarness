/**
 * `dispatch` -- a parent Run starts a child Run for another agent.
 *
 * The child runs in-process inside the handler. It gets a slice of the
 * parent's remaining budget and the parent keeps a reserve, so a child cannot
 * spend the parent into a corner it cannot finish from.
 */

import { RouteError, type Placement } from "../gateway/route.ts";
import { runLoop, ChildBlocked, type LoopDeps } from "../loop/loop.ts";
import { readReport, writeReport } from "../loop/report.ts";
import { AgentError, type AgentDefinition } from "../agents/definition.ts";
import type { Ledger } from "../ledger/ledger.ts";
import { errorResult, type ToolDefinition } from "./registry.ts";

export interface ChildAgent {
  readonly deps: LoopDeps;
  readonly agent: AgentDefinition;
  readonly system: string;
  /** Where this agent's runs go: its `model` line, routed. */
  readonly placement: Placement;
}

export interface DispatchToolDeps {
  readonly ledger: Ledger;
  readonly spawn: (agentName: string) => ChildAgent;
  /** The agents the model may dispatch to, so it is told rather than left to guess. */
  readonly agents?: ReadonlyArray<{ name: string; role: string }>;
  /** False when the card can't fit the image model; the model is told. */
  readonly imageAvailable?: boolean;
  /** Agents hidden because they render images; named in the refusal. */
  readonly hiddenImageAgents?: ReadonlyArray<string>;
  /** Told once the child Run exists, before its first turn. For display. */
  readonly onChildCreated?: (
    childId: string,
    agentName: string,
    allocationUsd: number,
  ) => void;
}

const MAX_ROLE_CHARS = 160;

function oneLine(role: string): string {
  const flat = role.replace(/[ \t\r\n]+/g, " ").trim();
  return flat.length > MAX_ROLE_CHARS ? `${flat.slice(0, MAX_ROLE_CHARS - 1)}…` : flat;
}

const MIN_RESERVE_USD = 0.25;
const RESERVE_FRACTION = 0.2;

/**
 * A failed child's report is an error result, here as on resume, where
 * `settlePending` marks it so: the model gets the same signal either way.
 */
function answer<T extends { status?: string }>(report: T): T {
  return report.status === "failed" ? errorResult(report) : report;
}

export function createDispatchTool(deps: DispatchToolDeps): ToolDefinition {
  const { ledger, spawn, onChildCreated, agents } = deps;
  const listed = agents && agents.length > 0 ? agents : undefined;

  return {
    name: "dispatch",
    description:
      "Start another agent on a goal and wait for its report. " +
      "The child gets a slice of your remaining budget, capped so you keep a " +
      "reserve; the report says what it did, what it spent, and what it " +
      "could not do. Runs alone in its turn. An agent can publish only what " +
      "it rendered itself, so send rendering and publishing to one child as " +
      "one task." +
      (listed
        ? ` Available agents: ${listed.map((s) => `${s.name} — ${oneLine(s.role)}`).join("; ")}.`
        : "") +
      (deps.imageAvailable === false
        ? " Image generation is unavailable on this card: the image model doesn't fit."
        : ""),
    schema: {
      type: "object",
      properties: {
        agent: {
          type: "string",
          description: "The name of the agent to dispatch.",
          ...(listed ? { enum: listed.map((s) => s.name) } : {}),
        },
        goal: { type: "string", description: "What the child should accomplish." },
        budget_usd: {
          type: "number",
          description: "The most you want the child to spend, in USD.",
        },
        context: {
          type: "string",
          description: "Anything the child needs to know that the goal does not say.",
        },
      },
      required: ["agent", "goal", "budget_usd"],
      additionalProperties: false,
    },
    tier: "budgeted",
    exclusive: true,
    handler: async (args, ctx) => {
      const agentName = String(args.agent);
      const goal = String(args.goal);
      const requested = args.budget_usd;
      const context =
        typeof args.context === "string" && args.context !== ""
          ? args.context
          : undefined;

      if (typeof requested !== "number" || !Number.isFinite(requested) || requested <= 0) {
        return { error: "budget_usd must be a number greater than zero" };
      }

      // The schema enum only advises; a model that ignores it must not get a
      // child built and budgeted for an agent it was told is not on offer.
      if (listed && !listed.some((a) => a.name === agentName)) {
        const why =
          deps.imageAvailable === false &&
          deps.hiddenImageAgents?.includes(agentName)
            ? ` Image generation is unavailable on this card: the image model doesn't fit.`
            : "";
        return {
          error:
            `"${agentName}" isn't available here.${why} Available agents: ` +
            listed.map((a) => a.name).join(", ") +
            ".",
        };
      }

      let child: ChildAgent;
      try {
        child = spawn(agentName);
      } catch (error) {
        if (error instanceof AgentError || error instanceof RouteError) {
          return { error: error.message };
        }
        throw error;
      }
      if (child.agent.tools.includes("dispatch")) {
        return {
          error: `agent ${agentName} can itself dispatch; nested dispatch is not allowed`,
        };
      }

      const parent = ledger.getRun(ctx.runId);
      if (!parent) throw new Error(`Run ${ctx.runId} does not exist`);
      const remaining = parent.budget_usd - ledger.treeSpentUsd(parent.id);
      const reserve = Math.max(MIN_RESERVE_USD, RESERVE_FRACTION * remaining);
      const allocation = Math.min(
        requested,
        remaining - reserve,
        child.agent.limits.maxUsd ?? Number.POSITIVE_INFINITY,
      );
      if (allocation <= 0) {
        return { error: "no budget left to dispatch with; finish with what you have" };
      }
      const note =
        allocation < requested
          ? {
              note: `budget clamped from $${requested.toFixed(2)} to $${allocation.toFixed(2)}`,
            }
          : {};

      let childId: string | undefined;
      try {
        const result = await runLoop(child.deps, {
          agent: child.agent.name,
          goal: context ? `${goal}\n\nContext: ${context}` : goal,
          budgetUsd: allocation,
          system: child.system,
          parentId: parent.id,
          // Recorded here, not after the run, so the dispatch message always
          // precedes the report in the child's log.
          onCreated: (id) => {
            childId = id;
            ledger.recordMessage({
              runId: id,
              from: parent.agent,
              to: child.agent.name,
              kind: "dispatch",
              body: { goal, budget_usd: allocation, context },
            });
            onChildCreated?.(id, child.agent.name, allocation);
          },
        });
        if (result.status === "blocked") throw new ChildBlocked(result.runId);
        return answer({ ...readReport(ledger, result.runId), ...note });
      } catch (error) {
        if (error instanceof ChildBlocked || childId === undefined) throw error;
        // A child that blew up is a result for the parent to handle, not a
        // reason to take the parent down with it.
        // Report first, as `finish` does: a terminal child is never without one.
        const report = writeReport(ledger, childId, {
          status: "failed",
          text: "",
          error: error instanceof Error ? error.message : String(error),
        });
        ledger.setRunStatus(childId, "failed");
        return answer({ ...report, ...note });
      }
    },
  };
}
