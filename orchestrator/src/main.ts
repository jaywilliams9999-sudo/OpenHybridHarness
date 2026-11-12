/**
 * The entry point: run one Run end to end against the live GPU service.
 *
 *   node src/main.ts "a brass agent on black"
 *
 * Each agent's `model` line decides whether its runs go to Claude or the
 * local GPU (`OHH_PROVIDER=scripted` swaps in a fixed script instead). Every
 * tool call, the network hop, the eviction and the artifact are real. This is
 * the whole thread end to end: config, a turn, a tool call, the wire, a file,
 * a Ledger row.
 */

import { applyDotEnv } from "./env.ts";
import { join } from "node:path";

import { loadConfig, ConfigError, describeMode } from "./config/config.ts";
import { openLedger, type Ledger } from "./ledger/ledger.ts";
import { createGpuClient, GpuError } from "./gpu/client.ts";
import { AgentError } from "./agents/definition.ts";
import { RouteError } from "./gateway/route.ts";
import { GatewayError } from "./gateway/anthropic.ts";
import {
  parseCheckpoint,
  CheckpointError,
  type LoopDeps,
  type RunResult,
} from "./loop/loop.ts";
import { ResumeError } from "./loop/tree.ts";
import {
  HARNESS,
  RunRefused,
  recordDecision,
  resumable,
  resumeGoal,
  startGoal,
  type RunnerOptions,
} from "./run/runner.ts";
import { printLedgerSummary } from "./cli/summary.ts";
import type { TurnEvent } from "./gateway/types.ts";
import { createTextStream } from "./output/stream-text.ts";

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;

function rule(label: string): void {
  console.log(`\n${bold(label)} ${dim("─".repeat(Math.max(0, 58 - label.length)))}`);
}

const USAGE = `
  node src/main.ts "<prompt>"          run a Run
  node src/main.ts --agent <name> "<prompt>"
                                       run it as a named agent (default: site)
  node src/main.ts --pending           list Runs waiting on a decision
  node src/main.ts --approve <id>      approve a gated call, then resume
  node src/main.ts --deny <id>         deny it, then resume
  node src/main.ts --resume <runId>   continue a blocked Run, and any parents
                                       waiting on it (--agent is ignored)
`.trimEnd();

/**
 * Print the operator's inbox.
 *
 * The approval ids are printed in full and ready to paste: the terminal is the
 * operator surface here, and the Ledger *is* the queue. A mechanism you
 * cannot answer from the terminal is implemented but not operable.
 */
function listPending(ledger: Ledger): number {
  const blocked = ledger.listBlockedRuns();
  if (blocked.length === 0) {
    console.log(dim("  nothing is waiting"));
    return 0;
  }
  for (const run of blocked) {
    console.log(`  ${run.id.slice(0, 8)} ${run.agent} ${dim(run.goal)}`);
    // A parent blocked behind its child has nothing of its own to decide: the
    // decision is the child's, listed under the child, and approving it
    // resumes this Run too. Pointing at the child keeps the operator from
    // resuming a parent that would only block again. The id is printed in
    // full: it is the one a stuck child's `--resume` needs.
    const waitingOn = waitingOnChildren(ledger, run.id);
    for (const childId of waitingOn) {
      const child = ledger.getRun(childId);
      console.log(
        `    ${cyan("⤷")} waiting on ${childId} ${dim(`(${child?.agent ?? "unknown"})`)}`,
      );
    }
    if (waitingOn.length > 0) continue;
    const pending = ledger
      .listApprovals(run.id)
      .filter((approval) => approval.status === "pending");
    for (const approval of pending) {
      console.log(`    ${cyan("?")} ${approval.tool} ${dim(approval.args_json)}`);
      console.log(`      ${dim(`--approve ${approval.id}`)}`);
    }
    if (pending.length === 0) {
      console.log(`    ${dim(`--resume ${run.id}`)}`);
    }
  }
  return 0;
}

/** The children a blocked Run's checkpoint says it is waiting on. */
function waitingOnChildren(ledger: Ledger, runId: string): string[] {
  const blob = ledger.readCheckpoint(runId);
  if (blob === undefined) return [];
  try {
    return (parseCheckpoint(blob).pending ?? []).flatMap((call) =>
      call.childRunId !== undefined ? [call.childRunId] : [],
    );
  } catch (error) {
    // The inbox must still list everything else; a resume of this Run will
    // report the unreadable checkpoint properly.
    if (error instanceof CheckpointError) return [];
    throw error;
  }
}

/** A runner refusal, in the terminal's words: the head in red, exit 1. */
function refused(ledger: Ledger, error: RunRefused): number {
  console.error(`${red(error.head)} ${error.detail}`);
  ledger.close();
  return 1;
}

async function main(): Promise<number> {
  // Orchestrator-only secrets live in the repo-root .env, loaded into this process alone.
  // Never the shell: see API_KEY_VAR for what an exported key cost. Variables
  // already set win, so a one-off override on the command line still works.
  const dotEnv = join(import.meta.dirname, "..", "..", ".env");
  applyDotEnv(dotEnv);

  // `--agent <name>` is consumed before anything else so the remaining
  // argument vector reads exactly as it did before this flag existed.
  const argv = [...process.argv.slice(2)];
  let agentName = "site";
  const agentAt = argv.indexOf("--agent");
  if (agentAt !== -1) {
    const named = argv[agentAt + 1];
    if (named === undefined || named.startsWith("--")) {
      console.error(`${red("--agent needs a name")}\n${USAGE}`);
      return 1;
    }
    agentName = named;
    argv.splice(agentAt, 2);
  }

  const flag = argv[0];
  const value = argv[1];
  const isCommand = flag !== undefined && flag.startsWith("--");
  const prompt = isCommand
    ? ""
    : (flag ?? "a brass agent on black, occult, high contrast");

  if (flag === "--help" || flag === "-h") {
    console.log(USAGE);
    return 0;
  }
  if (isCommand && flag !== "--pending" && value === undefined) {
    console.error(`${red("missing argument")}\n${USAGE}`);
    return 1;
  }

  const config = loadConfig();
  rule("orchestrator");
  console.log(`  gpu      ${config.gpuUrl}`);
  console.log(`  ledger     ${config.ledgerPath}`);
  console.log(`  artifacts  ${config.artifactsDir}`);
  console.log(
    `  ${describeMode(config)}`,
  );

  const ledger = openLedger(config.ledgerPath);

  // Answered without dialling the GPU service: an operator checking the queue should
  // not need the GPU machine to be awake.
  if (flag === "--pending") {
    rule("pending");
    const code = listPending(ledger);
    ledger.close();
    return code;
  }

  // A decision and its resume are one command on purpose (see
  // `decideApproval`). The decision is printed, and any refusal given, before
  // the GPU service is dialled: an unknown Run, or one that already ended and would
  // run again if resumed.
  let resumeId: string | undefined;
  try {
    if (flag === "--approve" || flag === "--deny") {
      const decision = flag === "--approve" ? "approved" : "denied";
      const approval = recordDecision(ledger, config.runningDir, value as string, decision);
      rule("decision");
      console.log(
        `  ${decision === "approved" ? green("approved") : red("denied")} ${approval.tool} ${dim(approval.args_json)}`,
      );
      resumeId = approval.run_id;
    } else if (flag === "--resume") {
      resumeId = value;
    }
    if (resumeId !== undefined) resumable(ledger, config.runningDir, resumeId);
  } catch (error) {
    if (!(error instanceof RunRefused)) throw error;
    return refused(ledger, error);
  }

  let lastPct = -1;
  // Set by `op.token`, cleared when the matching text event arrives. A local
  // turn streams its tokens live *and* yields one text event carrying the
  // authoritative content, so printing both would show the reply twice.
  let streamed = false;
  const gpu = createGpuClient({
    baseUrl: config.gpuUrl,
    token: config.gpuToken,
    onProgress: (event) => {
      if (event.event === "op.progress" && event.total) {
        const pct = Math.round(((event.step ?? 0) / event.total) * 100);
        if (pct !== lastPct) {
          lastPct = pct;
          process.stdout.write(`\r  ${dim(`render ${pct}%`)}   `);
        }
        return;
      }
      // Live tokens from a local turn. Written straight to stdout with no
      // newline so the reply appears as it is generated — the provider still
      // reads the authoritative text from the op's terminal result, because
      // these frames are droppable.
      if (event.event === "op.token") {
        streamed = true;
        process.stdout.write(String(event["text"] ?? ""));
        return;
      }
      if (event.event === "model.loaded") {
        console.log(
          `\r  ${cyan("loaded")} ${event["model"]} ${dim(`${event["vram_mb"]}MB in ${event["load_ms"]}ms`)}`,
        );
      }
      if (event.event === "model.unloaded") {
        console.log(
          `\r  ${cyan("unloaded")} ${event["model"]} ${dim(`freed ${event["freed_mb"]}MB (${event["reason"]})`)}`,
        );
      }
    },
  });

  rule("gpu health");
  const health = await gpu.health();
  console.log(
    `  ${green(health.status)}  ${health.gpu}  ${dim(`${health.vram_free_mb}/${health.vram_total_mb} MB free · v${health.version}`)}`,
  );

  // Every line an agent prints is indented by its depth, so a tree reads as
  // one: the root at the margin, each child two spaces further in.
  const output = (
    depth: number,
  ): Pick<LoopDeps, "onEvent" | "onToolCall" | "onCost"> => {
    const pad = "  ".repeat(depth);
    // Fragments of a turn's text are written as they arrive, indented once per
    // line; anything else (or the turn's end) closes the line first.
    const text = createTextStream(`${pad}  `);
    const closeLine = () => process.stdout.write(text.end());
    return {
      onEvent: (event: TurnEvent) => {
        if (event.kind !== "text") closeLine();
        if (event.kind === "thinking") console.log(`${pad}  ${dim(`· ${event.text}`)}`);
        if (event.kind === "text") {
          // Already on screen token by token; just close the line.
          if (streamed) {
            streamed = false;
            console.log();
          } else {
            process.stdout.write(text.push(event.text));
          }
        }
        if (event.kind === "tool_request") {
          console.log(`${pad}  ${cyan("→")} ${event.name}`);
        }
      },
      onToolCall: ({ tool, tier, ms, isError }) => {
        closeLine();
        const mark = isError ? red("✗") : green("✓");
        console.log(`${pad}  ${mark} ${tool} ${dim(`[${tier}] ${ms}ms`)}`);
      },
      onCost: ({ usd, cumulativeUsd, usage }) => {
        closeLine();
        // Cache reads are printed because they are the whole economics of a
        // multi-turn Run: a turn that re-reads its prefix costs a twentieth
        // of one that rebuilds it, and that is visible here or nowhere.
        console.log(
          dim(
            `${pad}  $${usd.toFixed(4)} (${cumulativeUsd.toFixed(4)} total) · ` +
              `in ${usage.in} · cache-w ${usage.cacheWrite} · ` +
              `cache-r ${usage.cacheRead} · out ${usage.out}`,
          ),
        );
      },
    };
  };

  const opts: RunnerOptions = {
    config,
    gpu,
    ledger,
    harness: HARNESS,
    output,
    // Printed at the parent's indent, just before the child's own lines.
    onChildCreated: (depth, child) => {
      console.log(
        `${"  ".repeat(depth)}${cyan("⤷")} ${child.agent} ${child.id.slice(0, 8)} ${dim(`$${child.allocationUsd.toFixed(2)}`)}`,
      );
    },
    // After the root's answer, in the orchestrator's voice, so a summary
    // that forgot to say it cannot hide it.
    onNotice: (line) => console.log(`\n  ${cyan(line)}`),
    // The agent decides what this agent is, what it may touch and what it may
    // spend; the runner loads it only now, after the GPU service health check. A
    // resume rebuilds every Run it wakes from the Run's own row -- its
    // agent, depth and budget -- so `--agent` plays no part in it.
    onReady: ({ agent, maxTurns, budgetUsd }) => {
      rule("agent");
      console.log(`  name    ${agent.name}`);
      console.log(`  tools   ${agent.tools.join(", ") || dim("none")}`);
      if (agent.model) console.log(`  model   ${agent.model}`);
      console.log(`  limits  $${budgetUsd.toFixed(2)} · ${maxTurns} turns`);
      rule("run");
    },
  };

  let result: RunResult;
  try {
    result =
      resumeId !== undefined
        ? await resumeGoal(opts, resumeId)
        : await startGoal(opts, { agent: agentName, goal: prompt });
  } catch (error) {
    // Checked above; only a Run that changed in between lands here.
    if (!(error instanceof RunRefused)) throw error;
    return refused(ledger, error);
  }

  rule("ledger");
  printLedgerSummary(ledger, result);

  ledger.close();
  console.log();
  return result.status === "done" ? 0 : 1;
}

main().then(
  (code) => process.exit(code),
  (error: unknown) => {
    if (
      error instanceof ConfigError ||
      error instanceof GpuError ||
      error instanceof GatewayError ||
      error instanceof CheckpointError ||
      error instanceof AgentError ||
      error instanceof RouteError ||
      error instanceof ResumeError
    ) {
      console.error(`\n${red(error.name)}: ${error.message}`);
      const remedy =
        error instanceof ConfigError ||
        error instanceof CheckpointError ||
        error instanceof AgentError ||
        error instanceof RouteError ||
        error instanceof ResumeError
          ? undefined
          : error.remedy;
      if (remedy) console.error(dim(`  remedy: ${remedy}`));
    } else {
      console.error(error);
    }
    process.exit(1);
  },
);
