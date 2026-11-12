/**
 * The Ledger summary printed at the end of a run.
 *
 * Kept apart from `main.ts` so it can be tested: `main.ts` runs on import.
 * In a tree the work usually happened in a child -- under lead, the image
 * belongs to the site Run -- so each descendant's artifacts are printed under
 * it, or the summary would show a finished run with nothing to show for it.
 */

import type { Ledger, RunRow } from "../ledger/ledger.ts";
import type { RunResult } from "../loop/loop.ts";

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;

/** Every descendant of a Run, depth first, with its depth below it. */
export function descendants(
  ledger: Ledger,
  runId: string,
  depth = 1,
): Array<{ run: RunRow; depth: number }> {
  return ledger.listChildren(runId).flatMap((run) => [
    { run, depth },
    ...descendants(ledger, run.id, depth + 1),
  ]);
}

function printArtifacts(
  ledger: Ledger,
  runId: string,
  pad: string,
  log: (line: string) => void,
): void {
  for (const artifact of ledger.listArtifacts(runId)) {
    const meta = JSON.parse(artifact.meta_json ?? "{}") as {
      bytes?: number;
      evicted?: Array<{ model: string; freed_mb: number }>;
    };
    log(`${pad}  art     ${artifact.path} ${dim(`${meta.bytes} bytes`)}`);
    for (const eviction of meta.evicted ?? []) {
      log(`${pad}  evicted ${eviction.model} ${dim(`freed ${eviction.freed_mb}MB`)}`);
    }
  }
}

export function printLedgerSummary(
  ledger: Ledger,
  result: RunResult,
  log: (line: string) => void = (line) => console.log(line),
): void {
  const run = ledger.getRun(result.runId);
  log(
    `  run    ${result.runId.slice(0, 8)} ${result.status === "done" ? green(result.status) : red(result.status)} ${dim(`${run?.turns} turns${result.reason ? ` · ${result.reason}` : ""}`)}`,
  );
  if (result.reason === "approval") {
    // The Ledger *is* the approval queue, and printing the pending rows is
    // what makes approvals usable rather than merely implemented.
    log(`  ${red("awaiting approval")} ${dim("nothing ran; the Run is checkpointed")}`);
    for (const approval of ledger.listApprovals(result.runId)) {
      if (approval.status !== "pending") continue;
      log(`  ${cyan("?")} ${approval.tool} ${dim(approval.args_json)}`);
      log(`    ${dim(`--approve ${approval.id}`)}`);
    }
  }
  if (result.reason === "budget") {
    log(
      `  ${red("over budget")} ${dim("stopped at a turn boundary; spend is recorded and the Run is checkpointed and can be resumed")}`,
    );
  }
  log(
    `  spent   $${result.spentUsd.toFixed(4)} ${dim(`of $${(run?.budget_usd ?? 0).toFixed(2)} budget`)}`,
  );
  for (const call of ledger.listToolCalls(result.runId)) {
    log(`  call    ${call.tool} ${dim(`[${call.tier}] ${call.ms}ms`)}`);
  }
  printArtifacts(ledger, result.runId, "", log);

  // The tree beneath it: one line per descendant with what it made, then
  // what the whole tree cost -- the figure the root's budget is checked
  // against, so it comes last.
  for (const { run: child, depth } of descendants(ledger, result.runId)) {
    const pad = "  ".repeat(depth);
    const status = child.status === "done" ? green(child.status) : red(child.status);
    log(
      `${pad}  ${cyan("⤷")} ${child.agent} ${child.id.slice(0, 8)} ${status} ${dim(`$${child.spent_usd.toFixed(4)}`)}`,
    );
    for (const approval of ledger.listApprovals(child.id)) {
      if (approval.status !== "pending") continue;
      log(`${pad}    ${cyan("?")} ${approval.tool} ${dim(approval.args_json)}`);
      log(`${pad}      ${dim(`--approve ${approval.id}`)}`);
    }
    printArtifacts(ledger, child.id, pad, log);
  }
  log(`  tree    $${ledger.treeSpentUsd(result.runId).toFixed(4)}`);
}
