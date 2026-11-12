/** One goal's full story: its card, then every Run in its tree. */

import { descendants } from "../cli/summary.ts";
import type { ApprovalRow, Ledger, RunStatus, ToolCallRow } from "../ledger/ledger.ts";
import { imagesOf, toCard, type BoardCard } from "./board.ts";

export interface RunDetail {
  id: string;
  agent: string;
  depth: number;
  status: RunStatus;
  spentUsd: number;
  toolCalls: ToolCallRow[];
  approvals: ApprovalRow[];
  images: string[];
}

export function goalDetail(
  ledger: Ledger,
  rootId: string,
  runningDir?: string,
): { card: BoardCard; runs: RunDetail[] } | undefined {
  const root = ledger.getRun(rootId);
  if (!root || root.parent_id !== null) return undefined;
  const tree = [{ run: root, depth: 0 }, ...descendants(ledger, rootId)];
  return {
    card: toCard(ledger, root, runningDir),
    runs: tree.map(({ run, depth }) => ({
      id: run.id,
      agent: run.agent,
      depth,
      status: run.status,
      spentUsd: run.spent_usd,
      toolCalls: ledger.listToolCalls(run.id),
      approvals: ledger.listApprovals(run.id),
      images: imagesOf(ledger, [run.id]),
    })),
  };
}
