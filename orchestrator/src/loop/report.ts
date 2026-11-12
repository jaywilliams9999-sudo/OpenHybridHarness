/**
 * A child Run's report to its parent.
 *
 * The report is a `messages` row, not a return value, so the parent can read it
 * after a restart and an operator can see it in the tree.
 */

import type { Ledger } from "../ledger/ledger.ts";

export interface Report {
  status: "done" | "failed";
  summary: string;
  artifacts: Array<{ path: string; kind: string }>;
  spent_usd: number;
  gaps: string[];
}

export function writeReport(
  ledger: Ledger,
  runId: string,
  end: {
    status: "done" | "failed";
    text: string;
    reason?: string;
    error?: string;
  },
): Report {
  const run = ledger.getRun(runId);
  if (!run) throw new Error(`Run ${runId} does not exist`);
  const parent = run.parent_id ? ledger.getRun(run.parent_id) : undefined;
  if (!parent) throw new Error(`Run ${runId} has no parent to report to`);

  const gaps: string[] = [];
  if (end.status !== "done" && end.reason) gaps.push(end.reason);
  // `done` on a cut-off turn (max_tokens, pause_turn) is not a finished
  // answer, and the summary alone would not say so.
  if (end.status === "done" && end.reason && end.reason !== "end_turn") {
    gaps.push(end.reason);
  }
  if (end.error) gaps.push(end.error);
  // A denial is a result the child handled, not a failure -- but the parent
  // needs to know the work it asked for was not fully carried out.
  for (const call of ledger.listToolCalls(runId)) {
    try {
      const result = JSON.parse(call.result_json ?? "null") as {
        denied?: unknown;
      } | null;
      if (result?.denied === true) gaps.push(`${call.tool} denied`);
    } catch {
      // A result that is not JSON cannot be a denial.
    }
  }

  const report: Report = {
    status: end.status,
    summary: end.text,
    artifacts: ledger
      .listArtifacts(runId)
      .map((a) => ({ path: a.path, kind: a.kind })),
    spent_usd: run.spent_usd,
    gaps,
  };
  ledger.recordMessage({
    runId,
    from: run.agent,
    to: parent.agent,
    kind: "report",
    body: report,
  });
  return report;
}

export function readReport(
  ledger: Ledger,
  runId: string,
): Report | undefined {
  const reports = ledger
    .listMessages(runId)
    .filter((m) => m.kind === "report");
  const last = reports[reports.length - 1];
  return last ? (JSON.parse(last.body) as Report) : undefined;
}
