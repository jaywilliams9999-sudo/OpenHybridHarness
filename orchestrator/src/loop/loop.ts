/**
 * The Loop — runs one Run to completion.
 *
 * Model turn, tool dispatch, repeat, until the provider says it is done, the
 * provider is refused, or the turn cap trips. Budget debiting, approval gating
 * and checkpoint/resume live in the helpers below; this is the turn cycle.
 */

import type { Placement } from "../gateway/route.ts";
import { costUsd } from "../gateway/pricing.ts";
import { readReport, writeReport } from "./report.ts";
import type { Ledger, RunStatus } from "../ledger/ledger.ts";
import { isErrorResult, type ToolRegistry } from "../tools/registry.ts";
import type {
  Provider,
  StopReason,
  ToolResultMessage,
  TranscriptEntry,
  TurnEvent,
} from "../gateway/types.ts";

export interface LoopDeps {
  readonly provider: Provider;
  readonly registry: ToolRegistry;
  readonly ledger: Ledger;
  readonly maxTurns: number;
  /**
   * Where this agent's runs were placed (`gateway/route.ts`), recorded on the
   * run row. The factory always sets it; unset records `local`, the default.
   */
  readonly placement?: Placement;
  readonly onEvent?: (event: TurnEvent) => void;
  readonly onToolCall?: (call: {
    tool: string;
    tier: string;
    ms: number;
    isError: boolean;
  }) => void;
  readonly onCost?: (cost: {
    usd: number;
    cumulativeUsd: number;
    usage: { in: number; cacheWrite: number; cacheRead: number; out: number };
  }) => void;
}

export interface RunSpec {
  readonly agent: string;
  readonly goal: string;
  readonly budgetUsd: number;
  readonly system?: string;
  /** Set when this Run is a child dispatched by another. */
  readonly parentId?: string;
  /** Called with the new Run's id right after it is created, before turn one. */
  readonly onCreated?: (runId: string) => void;
}

export type EndReason =
  | StopReason
  | "max_turns"
  | "budget"
  | "approval"
  | "child";

export interface RunResult {
  readonly runId: string;
  readonly status: RunStatus;
  readonly turns: number;
  readonly text: string;
  readonly thinking: string;
  readonly reason?: EndReason;
  /** What this Run actually spent, summed from the `costs` rows it wrote. */
  readonly spentUsd: number;
}

interface CollectedTurn {
  text: string;
  thinking: string;
  toolRequests: Array<{ id: string; name: string; input: unknown }>;
  reason: StopReason;
  events: TurnEvent[];
  usage?: { in: number; cacheWrite: number; cacheRead: number; out: number };
}

async function collectTurn(
  provider: Provider,
  request: Parameters<Provider["chat"]>[0],
  onEvent?: (event: TurnEvent) => void,
): Promise<CollectedTurn> {
  const turn: CollectedTurn = {
    text: "",
    thinking: "",
    toolRequests: [],
    // If a provider's stream ends without a stop reason, treat it as a finished
    // turn rather than inventing a failure — the text we got is still the answer.
    reason: "end_turn",
    events: [],
  };

  for await (const event of provider.chat(request)) {
    onEvent?.(event);
    turn.events.push(event);
    switch (event.kind) {
      case "text":
        turn.text += event.text;
        break;
      case "thinking":
        turn.thinking += event.text;
        break;
      case "tool_request":
        turn.toolRequests.push({
          id: event.id,
          name: event.name,
          input: event.input,
        });
        break;
      case "end":
        turn.reason = event.reason;
        break;
      case "usage":
        // Last write wins: a provider may revise its counts in `message_delta`
        // after an initial estimate in `message_start`.
        turn.usage = {
          in: event.in,
          cacheWrite: event.cacheWrite,
          cacheRead: event.cacheRead,
          out: event.out,
        };
        break;
    }
  }
  return turn;
}

/**
 * The resume point, stored as JSON in `runs.checkpoint_blob`.
 *
 * `version` exists because the Ledger has no migration runner: a
 * blob written by an older Loop has to be *recognisable* as unreadable rather
 * than silently misparsed into a transcript that no longer means what it says.
 */
/**
 * A tool call the Run stopped in front of, waiting on a human.
 *
 * `approvalId` is present only on the `gated` calls; the ungated ones from the
 * same turn ride along so the turn can be executed whole on resume. This is
 * also where the call-to-approval mapping lives, because the `approvals`
 * table has no tool_use_id column and adding one needs a migration runner.
 */
export interface PendingCall {
  readonly id: string;
  readonly name: string;
  readonly input: unknown;
  readonly approvalId?: string;
  /** The child this call is waiting on; its report becomes the result. */
  readonly childRunId?: string;
  /**
   * Set when the call is **already answered** -- refused beside an exclusive
   * tool, failed validation, or ran beside a call that blocked. It was
   * recorded when it was answered and is never dispatched again.
   */
  readonly result?: unknown;
  readonly isError?: boolean;
}

/** An answer decided before dispatch; the handler is skipped. */
interface Preset {
  readonly result: unknown;
  readonly isError?: boolean;
  /**
   * Whether dispatch records it in `tool_calls`. A child's report is recorded
   * when it settles the call (ruling R1); a stored answer was recorded when
   * it was produced, and recording it again would double-count the call.
   */
  readonly record?: boolean;
}

type ToolRequest = { id: string; name: string; input: unknown };

export interface Checkpoint {
  readonly version: 1;
  readonly agent: string;
  readonly goal: string;
  readonly system?: string;
  readonly budgetUsd: number;
  readonly transcript: readonly TranscriptEntry[];
  readonly turns: number;
  readonly spentUsd: number;
  /** Set only while blocked on an approval or a child; absent is the ordinary case. */
  readonly pending?: readonly PendingCall[];
}

/**
 * Thrown by a handler to say "my child is waiting on a human".
 *
 * Not a failure: the parent blocks behind the child, and on resume the child's
 * report becomes this call's result. The handler is never run a second time.
 */
export class ChildBlocked extends Error {
  // A declared field, not a parameter property: Node's type stripping cannot
  // erase a parameter property, because it emits code.
  readonly childRunId: string;

  constructor(childRunId: string) {
    super(`child Run ${childRunId} is waiting on a human`);
    this.childRunId = childRunId;
    this.name = "ChildBlocked";
  }
}

/**
 * Why a call's arguments cannot be used, when they never were JSON.
 *
 * A local provider keeps arguments it could not parse as `{ _raw }` rather
 * than dropping the call. Running a handler on that would fail on whichever
 * field it happens to check first, which tells the model nothing; this
 * names the real problem so its retry can fix it.
 */
function unparsedArguments(name: string, input: unknown): string | undefined {
  if (typeof input !== "object" || input === null) return undefined;
  const keys = Object.keys(input);
  const raw = (input as { _raw?: unknown })._raw;
  if (keys.length !== 1 || typeof raw !== "string") return undefined;
  return (
    `the arguments for ${name} were not valid JSON: ${raw.slice(0, 300)}. ` +
    `Call ${name} again with its arguments as a JSON object.`
  );
}

export class CheckpointError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CheckpointError";
  }
}

export function parseCheckpoint(blob: string): Checkpoint {
  let parsed: unknown;
  try {
    parsed = JSON.parse(blob);
  } catch {
    throw new CheckpointError("The checkpoint is not valid JSON.");
  }
  const candidate = parsed as Partial<Checkpoint>;
  if (candidate?.version !== 1) {
    throw new CheckpointError(
      `Unsupported checkpoint version ${String(candidate?.version)}; this ` +
        `build reads version 1. Start a new Run rather than resuming.`,
    );
  }
  if (!Array.isArray(candidate.transcript)) {
    throw new CheckpointError("The checkpoint has no transcript.");
  }
  return candidate as Checkpoint;
}

interface LoopState {
  readonly runId: string;
  readonly spec: RunSpec;
  readonly transcript: TranscriptEntry[];
  turns: number;
  spentUsd: number;
  readonly pending?: readonly PendingCall[];
}

export async function runLoop(
  deps: LoopDeps,
  spec: RunSpec,
): Promise<RunResult> {
  const runId = deps.ledger.createRun({
    agent: spec.agent,
    goal: spec.goal,
    parentId: spec.parentId ?? null,
    budgetUsd: spec.budgetUsd,
    ...(deps.placement !== undefined ? { placement: deps.placement } : {}),
  });
  spec.onCreated?.(runId);

  return drive(deps, {
    runId,
    spec,
    transcript: [{ role: "user", text: spec.goal }],
    turns: 0,
    spentUsd: 0,
  });
}

/**
 * Continue a Run from its checkpoint.
 *
 * The transcript is restored **verbatim and appended to, never rewritten**.
 * That is not tidiness: a thinking block's signature binds it to the exact
 * conversation prefix that produced it (system prompt, tool set, every
 * preceding message), so editing any earlier turn invalidates every thinking
 * block after it. A resume that "cleaned up" the history would 400 on the
 * first turn, and the prompt cache would miss on every turn after that.
 */
export async function resumeRun(
  deps: LoopDeps,
  runId: string,
): Promise<RunResult> {
  const blob = deps.ledger.readCheckpoint(runId);
  if (blob === undefined) {
    throw new CheckpointError(
      `Run ${runId} has no checkpoint. It either never reached a turn ` +
        `boundary or does not exist.`,
    );
  }
  const checkpoint = parseCheckpoint(blob);

  deps.ledger.setRunStatus(runId, "running");
  return drive(deps, {
    runId,
    spec: {
      agent: checkpoint.agent,
      goal: checkpoint.goal,
      budgetUsd: checkpoint.budgetUsd,
      ...(checkpoint.system !== undefined ? { system: checkpoint.system } : {}),
    },
    transcript: [...checkpoint.transcript],
    turns: checkpoint.turns,
    spentUsd: checkpoint.spentUsd,
    ...(checkpoint.pending !== undefined ? { pending: checkpoint.pending } : {}),
  });
}

async function drive(deps: LoopDeps, state: LoopState): Promise<RunResult> {
  const { provider, registry, ledger, maxTurns } = deps;
  const { runId, spec, transcript } = state;
  const tools = registry.forProvider();

  let turns = state.turns;
  let text = "";
  let thinking = "";
  let spentUsd = state.spentUsd;

  /**
   * Written only at a **completed cycle** -- after any tool results are in the
   * transcript -- because that is the one point where resuming cannot re-run a
   * side effect. A checkpoint taken after the model asked for a tool but
   * before the tool ran would, on resume, dispatch it again: and these tools
   * write files, render images and occupy a GPU.
   *
   * Crashing *during* tool execution therefore replays from the previous
   * boundary and the model asks again, so tool dispatch is at-least-once. That
   * is the honest cost of not requiring every tool to be idempotent, and it is
   * stated here rather than discovered later.
   */
  const checkpoint = (pending?: readonly PendingCall[]): void => {
    const blob: Checkpoint = {
      version: 1,
      agent: spec.agent,
      goal: spec.goal,
      ...(spec.system !== undefined ? { system: spec.system } : {}),
      budgetUsd: spec.budgetUsd,
      transcript,
      turns,
      spentUsd,
      ...(pending !== undefined && pending.length > 0 ? { pending } : {}),
    };
    ledger.saveCheckpoint(runId, JSON.stringify(blob));
  };
  /**
   * Record a call answered without running its handler, and say so.
   *
   * Every refusal the Loop makes on a tool's behalf -- unknown name, denial,
   * failed validation, a sibling of an exclusive tool -- goes through here, so
   * `tool_calls` shows the model asked even though nothing ran.
   */
  const answer = (
    requested: ToolRequest,
    result: unknown,
    isError: boolean,
  ): ToolResultMessage => {
    const definition = registry.get(requested.name);
    const tier = definition?.tier ?? "auto";
    ledger.recordToolCall({
      runId,
      tool: requested.name,
      args: requested.input,
      result,
      tier,
      ms: 0,
    });
    deps.onToolCall?.({ tool: requested.name, tier, ms: 0, isError });
    return {
      toolUseId: requested.id,
      name: requested.name,
      result,
      ...(isError ? { isError: true } : {}),
    };
  };

  /**
   * Run a turn's tool calls and record each one.
   *
   * `denied` carries the ids whose approval a human refused. A denial is a
   * *result*, not a failure: the tool returns a denial the agent must
   * handle, so the model learns it was refused and can choose another route.
   * Failing the Run instead would make every refusal look like a defect.
   *
   * `preset` carries ids already answered; their handlers never run. A
   * handler that throws `ChildBlocked` is not answered at all: it lands in
   * `blockedOn` unrecorded, because its real result is the child's report.
   */
  const dispatch = async (
    requests: readonly ToolRequest[],
    denied: ReadonlySet<string> = new Set(),
    preset: ReadonlyMap<string, Preset> = new Map(),
  ): Promise<{ results: ToolResultMessage[]; blockedOn: Map<string, string> }> => {
    const blockedOn = new Map<string, string>();
    // All requests in one turn run together and their results go back in one
    // message. Splitting them teaches the model to stop asking in parallel.
    const results = await Promise.all(
      requests.map(async (requested): Promise<ToolResultMessage> => {
        const answered = preset.get(requested.id);
        if (answered !== undefined) {
          if (answered.record) {
            return answer(requested, answered.result, answered.isError === true);
          }
          return {
            toolUseId: requested.id,
            name: requested.name,
            result: answered.result,
            ...(answered.isError ? { isError: true } : {}),
          };
        }

        const definition = registry.get(requested.name);
        const started = Date.now();

        if (!definition) {
          return answer(
            requested,
            { error: `no tool named ${requested.name} is registered` },
            true,
          );
        }

        if (denied.has(requested.id)) {
          return answer(
            requested,
            {
              denied: true,
              error: `the operator denied this ${definition.name} call`,
            },
            true,
          );
        }

        const unparsed = unparsedArguments(requested.name, requested.input);
        if (unparsed !== undefined) {
          return answer(requested, { error: unparsed }, true);
        }

        let result: unknown;
        let isError = false;
        try {
          result = await definition.handler(
            (requested.input ?? {}) as Record<string, unknown>,
            { runId },
          );
          // A handler can fail without throwing when the failure has a body
          // worth keeping, such as a failed child's report.
          isError = isErrorResult(result);
        } catch (error) {
          if (error instanceof ChildBlocked) {
            blockedOn.set(requested.id, error.childRunId);
            // A placeholder the caller discards: this turn's results are not
            // sent while any call in it is still waiting.
            return { toolUseId: requested.id, name: requested.name, result: null };
          }
          isError = true;
          result = {
            error: error instanceof Error ? error.message : String(error),
          };
        }

        const ms = Date.now() - started;
        ledger.recordToolCall({
          runId,
          tool: definition.name,
          args: requested.input,
          result,
          tier: definition.tier,
          ms,
        });
        deps.onToolCall?.({
          tool: definition.name,
          tier: definition.tier,
          ms,
          isError,
        });

        return {
          toolUseId: requested.id,
          name: definition.name,
          result,
          ...(isError ? { isError: true } : {}),
        };
      }),
    );
    return { results, blockedOn };
  };

  const finish = (
    status: RunStatus,
    reason: EndReason,
    pending?: readonly PendingCall[],
  ): RunResult => {
    // Checkpointed before the status is set, so a `blocked` Run is always
    // resumable: a status that promises a resume with no checkpoint behind it
    // is the lie the budget design exists to avoid.
    const parentId = ledger.getRun(runId)?.parent_id ?? null;
    // A child that runs out of budget cannot be resumed by anyone -- its parent
    // is a model, not an operator -- so `blocked` would be a state nobody is
    // watching. It fails and reports up instead.
    if (parentId !== null && status === "blocked" && reason === "budget") {
      status = "failed";
    }
    checkpoint(pending);
    // Before the status is set: a terminal child with no report yet is one
    // `settlePending` would answer with "ended without a report". `blocked` is
    // not an end: the child will resume and report then.
    if (parentId !== null && (status === "done" || status === "failed")) {
      writeReport(ledger, runId, { status, text, reason });
    }
    ledger.setRunStatus(runId, status);
    return { runId, status, turns, text, thinking, reason, spentUsd };
  };
  /**
   * Dispatch a whole turn and append its results -- or, if a handler said its
   * child is waiting on a human, block the Run behind that child.
   *
   * On a block nothing joins the transcript: the results go back in one
   * message or not at all. The calls that did run are carried with their
   * results so they are never run again; the waiting one carries only its
   * child, whose report answers it on resume.
   */
  const runTools = async (
    requests: readonly ToolRequest[],
    denied?: ReadonlySet<string>,
    preset?: ReadonlyMap<string, Preset>,
  ): Promise<RunResult | undefined> => {
    const { results, blockedOn } = await dispatch(requests, denied, preset);
    if (blockedOn.size > 0) {
      const pending: PendingCall[] = requests.map((requested, index) => {
        const base = {
          id: requested.id,
          name: requested.name,
          input: requested.input,
        };
        const childRunId = blockedOn.get(requested.id);
        if (childRunId !== undefined) return { ...base, childRunId };
        const done = results[index];
        return {
          ...base,
          // `null`, never `undefined`: JSON drops an undefined key, and a call
          // without `result` would be dispatched again on resume.
          result: done?.result ?? null,
          ...(done?.isError ? { isError: true } : {}),
        };
      });
      return finish("blocked", "child", pending);
    }
    transcript.push({ role: "tool_results", results });
    checkpoint();
    return undefined;
  };

  /**
   * Finish the turn the Run was blocked in front of.
   *
   * Runs once, on entry, when the checkpoint carried pending calls. An
   * approval still undecided blocks again rather than guessing: an operator
   * who has not answered is not the same as one who said yes, and the whole
   * purpose of approvals is that nobody can mistake the two. A child that has not reported
   * is undecided in the same way.
   */
  const settlePending = async (
    pending: readonly PendingCall[],
  ): Promise<RunResult | undefined> => {
    const undecided: PendingCall[] = [];
    const denied = new Set<string>();
    const preset = new Map<string, Preset>();

    for (const call of pending) {
      if ("result" in call) {
        preset.set(call.id, {
          result: call.result,
          ...(call.isError ? { isError: true } : {}),
        });
      } else if (call.childRunId !== undefined) {
        const report = readReport(ledger, call.childRunId);
        const status = ledger.getRun(call.childRunId)?.status;
        if (report !== undefined) {
          preset.set(call.id, {
            result: report,
            ...(report.status === "failed" ? { isError: true } : {}),
            record: true,
          });
        } else if (status === "blocked" || status === "running") {
          // Still waiting on its human, or being resumed right now.
          undecided.push(call);
        } else {
          // Ended without reporting: say so rather than wait forever on a
          // report that is never coming.
          preset.set(call.id, {
            result: {
              error: `child Run ${call.childRunId} ended (${status ?? "missing"}) without a report`,
            },
            isError: true,
            record: true,
          });
        }
      } else if (call.approvalId !== undefined) {
        const approval = ledger.readApproval(call.approvalId);
        if (approval === undefined || approval.status === "pending") {
          undecided.push(call);
        } else if (approval.status === "denied") {
          denied.add(call.id);
        }
      }
      // Otherwise ungated; rides along.
    }

    if (undecided.length > 0) {
      const onApproval = undecided.some((call) => call.approvalId !== undefined);
      return finish("blocked", onApproval ? "approval" : "child", pending);
    }

    return runTools(pending, denied, preset);
  };

  if (state.pending !== undefined && state.pending.length > 0) {
    const blocked = await settlePending(state.pending);
    if (blocked) return blocked;
  }

  while (turns < maxTurns) {
    // The tree's spend, not just this Run's: a Run that dispatches children
    // is bounded by what the whole subtree has cost.
    // Checked before spending, not after. A turn's cost cannot be known until
    // it has run -- output tokens are the larger half and nobody knows how
    // many there will be -- so the only honest cap is "do not start a turn
    // once the budget is already gone". The status is `blocked`, never silent
    // truncation, because a resumable half-finished task beats one that
    // quietly gave up. The checkpoint written with it makes the resume real.
    if (ledger.treeSpentUsd(runId) >= spec.budgetUsd) {
      return finish("blocked", "budget");
    }

    turns++;
    ledger.bumpTurn(runId);

    const turn = await collectTurn(
      provider,
      {
        ...(spec.system !== undefined ? { system: spec.system } : {}),
        // A snapshot, not the live array. `TurnRequest.transcript` is declared
        // readonly, and a provider that iterates lazily (every streaming
        // adapter does) must not see turns appended after its request began.
        transcript: [...transcript],
        tools,
      },
      deps.onEvent,
    );

    if (turn.text) text = turn.text;
    if (turn.thinking) thinking = turn.thinking;
    transcript.push({ role: "assistant", events: turn.events });

    // Recorded before the turn can end for any reason — a refusal and a cap
    // trip both cost money, and a Ledger that only prices successful turns
    // under-reports exactly the runs an operator most wants to account for.
    if (turn.usage) {
      const usd = costUsd(provider.model, turn.usage);
      spentUsd += usd;
      ledger.recordCost({
        runId,
        provider: provider.name,
        model: provider.model,
        usage: turn.usage,
        usd,
      });
      deps.onCost?.({ usd, cumulativeUsd: spentUsd, usage: turn.usage });
    }

    if (turn.reason === "refusal") {
      return finish("failed", "refusal");
    }

    if (turn.reason !== "tool_use" || turn.toolRequests.length === 0) {
      return finish("done", turn.reason);
    }

    // Answers decided before anything runs. Each is recorded as it is made and
    // carried as a stored `result`, so neither a block nor a resume can run it.
    const preset = new Map<string, Preset>();
    const settle = (requested: ToolRequest, error: string): void => {
      const { result, isError } = answer(requested, { error }, true);
      preset.set(requested.id, { result, ...(isError ? { isError } : {}) });
    };

    // An exclusive tool runs alone, and that is decided first -- before gating
    // -- so an exclusive turn never asks a human to approve a sibling that was
    // never going to run.
    const exclusive = turn.toolRequests.find(
      (requested) => registry.get(requested.name)?.exclusive === true,
    );
    if (exclusive !== undefined) {
      for (const requested of turn.toolRequests) {
        if (requested.id === exclusive.id) continue;
        settle(
          requested,
          `${exclusive.name} runs alone in its turn; ask for ${requested.name} again after it returns`,
        );
      }
    }

    // A `gated` call stops the Run *before anything runs*.
    //
    // All-or-nothing per turn, deliberately. The requests in one turn run
    // together, so executing the ungated ones and holding the gated one would
    // half-apply a turn the operator has not approved -- and `gpu_*` tools
    // occupy a GPU and write files. Blocking the whole turn also keeps the
    // checkpoint invariant intact: no side effect has happened yet, so the
    // resume point is still a clean boundary.
    const gated = turn.toolRequests.filter((requested) => {
      if (preset.has(requested.id)) return false;
      const definition = registry.get(requested.name);
      if (definition?.tier !== "gated") return false;
      // Validated before the approval is requested: a human should only be
      // asked about a call that could succeed if they said yes.
      const problem =
        unparsedArguments(requested.name, requested.input) ??
        definition.validate?.(
          (requested.input ?? {}) as Record<string, unknown>,
          { runId },
        );
      if (problem !== undefined) {
        settle(requested, problem);
        return false;
      }
      return true;
    });
    if (gated.length > 0) {
      const pending: PendingCall[] = turn.toolRequests.map((requested) => {
        const base = {
          id: requested.id,
          name: requested.name,
          input: requested.input,
        };
        const answered = preset.get(requested.id);
        if (answered !== undefined) {
          return {
            ...base,
            result: answered.result,
            ...(answered.isError ? { isError: true } : {}),
          };
        }
        if (!gated.some((g) => g.id === requested.id)) return base;
        return {
          ...base,
          approvalId: ledger.requestApproval({
            runId,
            tool: requested.name,
            args: requested.input,
          }),
        };
      });
      return finish("blocked", "approval", pending);
    }

    const blocked = await runTools(turn.toolRequests, undefined, preset);
    if (blocked) return blocked;
  }

  // Deliberately `failed`, not `blocked`: blocked promises that a resume will
  // pick the work back up, and a Run that spent its turn cap would only hit
  // the cap again. A status that lies about being recoverable is worse than
  // one that is blunt.
  return finish("failed", "max_turns");
}
