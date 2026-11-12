/**
 * A goal's result: what a root Run leaves for the operator when it ends.
 *
 * A child reports to its parent with a `report` message; a root has
 * no parent, so it writes a `result` message to the operator instead. Same
 * table, no schema change, and the Hub reads it after any restart.
 */

import type { Ledger } from "../ledger/ledger.ts";
import { CheckpointError, parseCheckpoint } from "../loop/loop.ts";
import { descendants } from "../cli/summary.ts";
import { forgetRoot, isRootOwned } from "./owners.ts";

/** The `to_agent` of a root's result: nobody in the tree, the human. */
export const OPERATOR = "operator";

export interface GoalResult {
  status: "done" | "failed";
  reason: string | null;
  summary: string;
  spent_usd: number;
}

export function recordResult(
  ledger: Ledger,
  rootId: string,
  r: { status: "done" | "failed"; reason?: string; summary: string },
): void {
  const root = ledger.getRun(rootId);
  if (!root) throw new Error(`Run ${rootId} does not exist`);
  const body: GoalResult = {
    status: r.status,
    reason: r.reason ?? null,
    summary: r.summary,
    // The whole tree: under lead the money is mostly spent by children.
    spent_usd: ledger.treeSpentUsd(rootId),
  };
  ledger.recordMessage({
    runId: rootId,
    from: root.agent,
    to: OPERATOR,
    kind: "result",
    body,
  });
}

export function readResult(ledger: Ledger, rootId: string): GoalResult | undefined {
  const results = ledger.listMessages(rootId).filter((m) => m.kind === "result");
  const last = results[results.length - 1];
  return last ? (JSON.parse(last.body) as GoalResult) : undefined;
}

/**
 * What a Run last said, read from its checkpoint: the text of its latest
 * assistant turn. For a goal still in flight, which has no result yet.
 */
export function summaryFromCheckpoint(ledger: Ledger, runId: string): string | undefined {
  const blob = ledger.readCheckpoint(runId);
  if (blob === undefined) return undefined;
  let transcript;
  try {
    transcript = parseCheckpoint(blob).transcript;
  } catch (error) {
    if (error instanceof CheckpointError) return undefined;
    throw error;
  }
  for (let i = transcript.length - 1; i >= 0; i--) {
    const entry = transcript[i]!;
    if (entry.role !== "assistant") continue;
    return entry.events.flatMap((e) => (e.kind === "text" ? [e.text] : [])).join("");
  }
  return undefined;
}

/**
 * Mark a root and every descendant still `running` as failed.
 *
 * A descendant left `running` under a failed root would look alive forever;
 * one already `done`, `failed` or `blocked` keeps the status it earned.
 */
export function failRunning(ledger: Ledger, rootId: string): void {
  ledger.setRunStatus(rootId, "failed");
  for (const { run } of descendants(ledger, rootId)) {
    if (run.status === "running") ledger.setRunStatus(run.id, "failed");
  }
}

/**
 * Fail every `running` root no live process owns, and give it a result.
 *
 * Run once when the Hub starts: a goal whose process died mid-turn would
 * otherwise show as running forever. A `blocked` root is waiting on a human,
 * not on a process, and is never touched.
 */
export function sweepInterrupted(ledger: Ledger, runningDir: string): string[] {
  const swept: string[] = [];
  for (const root of ledger.listRootRuns()) {
    if (root.status !== "running" || isRootOwned(runningDir, root.id)) continue;
    failRunning(ledger, root.id);
    recordResult(ledger, root.id, { status: "failed", reason: "interrupted", summary: "" });
    // The dead owner's file, if it left one.
    forgetRoot(runningDir, root.id);
    swept.push(root.id);
  }
  return swept;
}
