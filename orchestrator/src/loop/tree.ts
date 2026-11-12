/**
 * Resume a Run and walk upward through any parents waiting on it.
 *
 * One approval unblocks a leaf; the leaf finishing is what unblocks its
 * parent, and so on to the root. The ledger holds all of that state, so this
 * is only the walk.
 */

import type { Ledger, RunRow } from "../ledger/ledger.ts";
import { writeReport } from "./report.ts";
import { resumeRun, type LoopDeps, type RunResult } from "./loop.ts";

/** The Run named cannot be resumed: it has already ended. */
export class ResumeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ResumeError";
  }
}

export async function resumeTree(
  ledger: Ledger,
  depsFor: (run: RunRow) => LoopDeps,
  runId: string,
): Promise<RunResult> {
  const resume = async (id: string): Promise<RunResult> => {
    const run = ledger.getRun(id);
    if (!run) throw new Error(`Run ${id} does not exist`);
    try {
      return await resumeRun(depsFor(run), id);
    } catch (error) {
      // A root that throws has nobody to report to; the operator hears it.
      if (!run.parent_id) throw error;
      // A child that blew up is a result for its parent to handle, exactly as
      // when it throws inside `dispatch`. Left `running`, it would
      // hold its parent `blocked` on a report that is never coming.
      writeReport(ledger, id, {
        status: "failed",
        text: "",
        error: error instanceof Error ? error.message : String(error),
      });
      ledger.setRunStatus(id, "failed");
      const after = ledger.getRun(id);
      return {
        runId: id,
        status: "failed",
        turns: after?.turns ?? run.turns,
        text: "",
        thinking: "",
        spentUsd: after?.spent_usd ?? run.spent_usd,
      };
    }
  };

  // Resuming a Run that already ended would run it again: a second report
  // from a child, a fresh dispatch from a finished lead root.
  const named = ledger.getRun(runId);
  if (named && (named.status === "done" || named.status === "failed")) {
    throw new ResumeError(
      `Run ${runId} is already ${named.status}; only a blocked Run can be resumed`,
    );
  }

  let result = await resume(runId);
  while (result.status === "done" || result.status === "failed") {
    const parentId = ledger.getRun(result.runId)?.parent_id;
    if (!parentId || ledger.getRun(parentId)?.status !== "blocked") break;
    result = await resume(parentId);
  }
  return result;
}
