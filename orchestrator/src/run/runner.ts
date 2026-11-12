/**
 * The shared runner: the one way to start a goal, decide an approval and
 * resume a tree.
 *
 * The terminal and the Hub both call these, so a goal ends the same way
 * whichever surface started it: its root owns a pid file while it runs, and
 * when it ends -- done, failed, or thrown -- it leaves a `result` for the
 * operator. Printing stays with the caller; this module only reports through
 * the callbacks it is given and the errors it throws.
 */

import type { Config } from "../config/config.ts";
import { imageAvailable, type GpuClient } from "../gpu/client.ts";
import type { ApprovalRow, Ledger, RunRow } from "../ledger/ledger.ts";
import type { AgentDefinition } from "../agents/definition.ts";
import { runLoop, type RunResult } from "../loop/loop.ts";
import { resumeTree } from "../loop/tree.ts";
import {
  createAgentFactory,
  runDeps,
  depthOf,
  type AgentFactory,
  type AgentFactoryOptions,
} from "../agents/factory.ts";
import { claimRoot, releaseRoot, rootOwner } from "./owners.ts";
import { failRunning, recordResult } from "./result.ts";

/**
 * What every agent shares: the machinery, not the job.
 *
 * The job half lives in the agent's own `role`, so an operator writing a new
 * one states only what differs. `buildSystem` joins them.
 */
export const HARNESS =
  "You are an agent in OpenHybridHarness, running on this machine. A local GPU service holds the models on the graphics card; your tools are the only way to reach it.";

export interface RunnerOptions {
  readonly config: Config;
  readonly gpu: GpuClient;
  readonly ledger: Ledger;
  readonly harness: string;
  readonly output: AgentFactoryOptions["output"];
  readonly onChildCreated?: AgentFactoryOptions["onChildCreated"];
  /**
   * Told which agent is about to run, and on what budget, just before its
   * first turn: the root of a new goal, or the Run a resume wakes first.
   */
  readonly onReady?: (agent: { agent: AgentDefinition; maxTurns: number; budgetUsd: number }) => void;
  /**
   * Told the orchestrator's own line about a goal that has ended, printed
   * after the root's answer: today, that its image was skipped.
   */
  readonly onNotice?: (line: string) => void;
  /** Builds the agent factory for one call. Tests substitute scripted agents. */
  readonly factory?: (prompt: string, imageAvailable: boolean) => AgentFactory;
}

/**
 * A refusal meant for the operator, not a fault.
 *
 * `head` is the part a terminal shows in red and `detail` the rest, so the
 * terminal's wording survives the move here unchanged.
 */
export class RunRefused extends Error {
  readonly head: string;
  readonly detail: string;

  constructor(head: string, detail: string) {
    super(`${head} ${detail}`);
    this.name = "RunRefused";
    this.head = head;
    this.detail = detail;
  }
}

/**
 * The root's budget.
 *
 * Precedence: a figure the operator gave for this goal, then an explicitly
 * set environment variable, then the agent file, then the built-in default.
 * The file is the agent's declared policy; the other two are this operator,
 * this run, overriding it on purpose.
 */
export function rootBudget(
  config: Config,
  agent: AgentDefinition,
  explicitUsd?: number,
  env: Record<string, string | undefined> = process.env,
): number {
  if (explicitUsd !== undefined) return explicitUsd;
  if (env["OHH_BUDGET_USD"] !== undefined) return config.budgetUsd;
  return agent.limits.maxUsd ?? config.budgetUsd;
}

/**
 * One factory per call. In scripted mode the factory caches an agent per
 * (agent, depth), so within one call a resume reuses the provider that
 * already ran -- and two goals never share one, because the scripted
 * provider's prompt is the goal. In routed mode every run gets its own.
 */
function factoryFor(
  opts: RunnerOptions,
  prompt: string,
  images: boolean,
): AgentFactory {
  if (opts.factory) return opts.factory(prompt, images);
  return createAgentFactory({
    config: opts.config,
    gpu: opts.gpu,
    ledger: opts.ledger,
    harness: opts.harness,
    prompt,
    imageAvailable: images,
    output: opts.output,
    ...(opts.onChildCreated ? { onChildCreated: opts.onChildCreated } : {}),
  });
}

/**
 * Whether the card can run the image model. One fetch per call; a failure
 * (or a stub with no status) is "unknown", which counts as available -- the
 * GPU service still refuses an unavailable model plainly.
 */
async function imagesAvailable(gpu: GpuClient): Promise<boolean> {
  try {
    return imageAvailable(await gpu.status());
  } catch {
    return true;
  }
}

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

/** Recorded on a root started on a card that cannot fit the image model. */
export const IMAGE_SKIPPED = "image skipped: model doesn't fit this card";

/** How the orchestrator says it in a summary, in its own voice. */
export const IMAGE_SKIPPED_LINE = `[orchestrator] ${IMAGE_SKIPPED}`;

/** Whether this goal ran without images: its root carries the notice. */
function imageSkipped(ledger: Ledger, rootId: string): boolean {
  return ledger
    .listMessages(rootId)
    .some((m) => m.kind === "notice" && JSON.parse(m.body) === IMAGE_SKIPPED);
}

/**
 * Write the root's result if it has ended; a blocked root has none yet.
 *
 * A goal that ran without images gets the skip notice appended by the
 * orchestrator, not left to the root's own prose: on the local model the
 * Lead reported hero images nobody made, and a key-less run must never
 * silently fake work.
 */
function recordIfEnded(
  opts: Pick<RunnerOptions, "ledger" | "onNotice">,
  rootId: string,
  result: RunResult,
): void {
  const { ledger } = opts;
  if (result.runId !== rootId) return;
  if (result.status !== "done" && result.status !== "failed") return;
  const skipped = imageSkipped(ledger, rootId);
  const summary = skipped
    ? result.text.trim() === ""
      ? IMAGE_SKIPPED_LINE
      : `${result.text}\n\n${IMAGE_SKIPPED_LINE}`
    : result.text;
  recordResult(ledger, rootId, {
    status: result.status,
    ...(result.reason !== undefined ? { reason: result.reason } : {}),
    summary,
  });
  if (skipped) opts.onNotice?.(IMAGE_SKIPPED_LINE);
}

/**
 * A root that threw has no parent to report to: fail it and anything of its
 * still running, and leave the error as its result.
 */
function failRoot(ledger: Ledger, rootId: string, error: unknown): void {
  failRunning(ledger, rootId);
  recordResult(ledger, rootId, {
    status: "failed",
    reason: errorMessage(error),
    summary: "",
  });
}

export async function startGoal(
  opts: RunnerOptions,
  req: {
    agent: string;
    goal: string;
    budgetUsd?: number;
    onCreated?: (rootId: string) => void;
  },
): Promise<RunResult> {
  const { ledger, config } = opts;
  // The agent is loaded here, after whatever checks the caller ran first, so
  // a malformed file is reported against a Orchestrator otherwise known to work.
  const images = await imagesAvailable(opts.gpu);
  const root = factoryFor(opts, req.goal, images).agent(req.agent, 0);
  const budgetUsd = rootBudget(config, root.agent, req.budgetUsd);
  opts.onReady?.({ agent: root.agent, maxTurns: root.deps.maxTurns, budgetUsd });

  let rootId: string | undefined;
  try {
    const result = await runLoop(root.deps, {
      agent: root.agent.name,
      goal: req.goal,
      budgetUsd,
      system: root.system,
      onCreated: (id) => {
        rootId = id;
        claimRoot(config.runningDir, id);
        if (!images) {
          ledger.recordMessage({
            runId: id,
            from: "orchestrator",
            to: root.agent.name,
            kind: "notice",
            body: IMAGE_SKIPPED,
          });
        }
        req.onCreated?.(id);
      },
    });
    recordIfEnded(opts, result.runId, result);
    return result;
  } catch (error) {
    if (rootId !== undefined) failRoot(ledger, rootId, error);
    throw error;
  } finally {
    if (rootId !== undefined) releaseRoot(config.runningDir, rootId);
  }
}

/** The root of the tree a Run belongs to. */
export function rootOf(ledger: Ledger, run: RunRow): string {
  let current = run;
  const seen = new Set<string>([run.id]);
  while (current.parent_id) {
    const parent = ledger.getRun(current.parent_id);
    if (!parent || seen.has(parent.id)) break;
    seen.add(parent.id);
    current = parent;
  }
  return current.id;
}

/**
 * Refuse a tree whose root another live process is running.
 *
 * Neither `resumeTree` nor `resumeRun` refuses a `running` Run, so without
 * this a second process -- the Hub beside a terminal -- would resume the same
 * tree, take over its claim, and on a throw fail a goal that is still alive.
 */
function refuseIfOwnedElsewhere(runningDir: string, rootId: string): void {
  const owner = rootOwner(runningDir, rootId);
  if (owner !== undefined && owner !== process.pid) {
    throw new RunRefused("not resumable", `${rootId} is running in another process`);
  }
}

/**
 * Record a decision on a pending approval, without resuming anything.
 *
 * Refused before anything is written if the tree is running elsewhere: an
 * approval marked approved that nobody resumes is exactly the state
 * `decideApproval` exists to prevent. Exported for the terminal, which prints
 * the decision before it dials the GPU service; everyone else wants
 * `decideApproval`.
 */
export function recordDecision(
  ledger: Ledger,
  runningDir: string,
  approvalId: string,
  decision: "approved" | "denied",
): ApprovalRow {
  const approval = ledger.readApproval(approvalId);
  if (!approval) throw new RunRefused("unknown approval", approvalId);
  const run = ledger.getRun(approval.run_id);
  if (approval.status === "pending" && run) {
    refuseIfOwnedElsewhere(runningDir, rootOf(ledger, run));
  }
  if (!ledger.decideApproval(approval.id, decision)) {
    throw new RunRefused("already decided", `${approval.tool} is ${approval.status}`);
  }
  return approval;
}

/**
 * The Run a resume would wake, or a refusal.
 *
 * Resuming a Run that already ended would run it again: a second report from
 * a child, a fresh dispatch from a finished lead root. `resumeTree` refuses
 * too; this is the operator's message, given before the GPU service is dialled. A
 * tree another live process is running is refused as well.
 */
export function resumable(ledger: Ledger, runningDir: string, runId: string): RunRow {
  const run = ledger.getRun(runId);
  if (!run) throw new RunRefused("unknown run", runId);
  if (run.status === "done" || run.status === "failed") {
    throw new RunRefused(
      "not resumable",
      `${run.id} is already ${run.status}; only a blocked Run can be resumed`,
    );
  }
  refuseIfOwnedElsewhere(runningDir, rootOf(ledger, run));
  return run;
}

/**
 * A decision and its resume, as one call on purpose. Splitting them leaves a
 * Run approved-but-stopped, which from the outside looks exactly like one
 * nobody has answered -- and the approval design exists so those cannot be confused.
 */
export async function decideApproval(
  opts: RunnerOptions,
  approvalId: string,
  decision: "approved" | "denied",
): Promise<{ approval: ApprovalRow; result: RunResult }> {
  const approval = recordDecision(opts.ledger, opts.config.runningDir, approvalId, decision);
  const result = await resumeGoal(opts, approval.run_id);
  return { approval, result };
}

/**
 * Resume a blocked Run and every parent waiting on it.
 *
 * A resume rebuilds every Run it wakes from the Run's own row -- its agent
 * and its depth -- and keeps the budget it was created with. The root is
 * claimed for the duration, because that is what the sweep looks at, and gets
 * its result only if the walk reached it and it ended: a resume can stop at a
 * child that blocks again.
 */
export async function resumeGoal(opts: RunnerOptions, runId: string): Promise<RunResult> {
  const { ledger, config } = opts;
  const run = resumable(ledger, config.runningDir, runId);
  const rootId = rootOf(ledger, run);
  const factory = factoryFor(opts, "", await imagesAvailable(opts.gpu));
  const first = factory.agent(run.agent, depthOf(ledger, run), run.placement);
  opts.onReady?.({
    agent: first.agent,
    maxTurns: first.deps.maxTurns,
    budgetUsd: run.budget_usd,
  });

  claimRoot(config.runningDir, rootId);
  try {
    const result = await resumeTree(
      ledger,
      (p) => runDeps(factory, ledger, p),
      runId,
    );
    recordIfEnded(opts, rootId, result);
    return result;
  } catch (error) {
    // `resumeTree` turns a child's throw into a failed report, so a throw
    // that reaches here is the root's own.
    failRoot(ledger, rootId, error);
    throw error;
  } finally {
    releaseRoot(config.runningDir, rootId);
  }
}
