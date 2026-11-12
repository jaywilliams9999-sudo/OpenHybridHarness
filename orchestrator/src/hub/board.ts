/**
 * The Hub's board: root goals sorted into three columns, read from the Ledger.
 *
 * Pure functions over the Ledger -- no state of their own, so the board is
 * right after any restart. Only roots are cards; a goal's agents hang off it.
 */

import { descendants } from "../cli/summary.ts";
import { parseCheckpoint } from "../loop/loop.ts";
import { isRootOwned } from "../run/owners.ts";
import { readResult, summaryFromCheckpoint } from "../run/result.ts";
import type { Ledger, RunRow, RunStatus } from "../ledger/ledger.ts";

export type Column = "working" | "needs_you" | "done";

export interface BoardCard {
  id: string;
  goal: string;
  status: RunStatus;
  column: Column;
  createdAt: string;
  updatedAt: string;
  budgetUsd: number;
  /** The whole tree's spend, the figure the budget is checked against. */
  spentUsd: number;
  /** The root's own agent: the one the goal was given to. */
  root: { id: string; agent: string; status: RunStatus };
  /** The agents the root dispatched, anywhere in its tree. */
  agents: Array<{ id: string; agent: string; status: RunStatus }>;
  /** Artifact ids of kind "image" across the tree, newest first. */
  images: string[];
  summary: string | null;
  reason: string | null;
  approval: { id: string; tool: string; args: unknown; imageId: string | null } | null;
}

/** The image artifact ids of some Runs, newest first. */
export function imagesOf(ledger: Ledger, runIds: string[]): string[] {
  return runIds
    .flatMap((id) => ledger.listArtifacts(id))
    .filter((a) => a.kind === "image")
    .reverse()
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .map((a) => a.id);
}

/** The first pending approval in a tree, depth first, root before children. */
export function firstPending(ledger: Ledger, root: RunRow) {
  for (const id of [root.id, ...descendants(ledger, root.id).map((d) => d.run.id)]) {
    const pending = ledger.listApprovals(id).find((a) => a.status === "pending");
    if (pending) return pending;
  }
  return undefined;
}

/**
 * Why a blocked root has nothing for the operator to decide. The Ledger keeps
 * no reason column, so this reads what it does keep: a checkpoint waiting on a
 * child, or a tree that has spent its budget. Otherwise just "blocked".
 */
function blockReason(ledger: Ledger, root: RunRow): string {
  const blob = ledger.readCheckpoint(root.id);
  if (blob !== undefined) {
    try {
      if (parseCheckpoint(blob).pending?.some((p) => p.childRunId !== undefined)) return "child";
    } catch {
      // An unreadable checkpoint falls through to the spend check.
    }
  }
  return ledger.treeSpentUsd(root.id) >= root.budget_usd ? "stopped: budget spent" : "blocked";
}

/**
 * Why Run again is refused for a root, or null when it may run. A failed
 * root may; so may one stopped blocked with nothing in its tree to decide --
 * nothing will resume it. One waiting on an approval is the operator's to
 * answer, not to start over.
 */
export function againRefusal(ledger: Ledger, root: RunRow): string | null {
  if (root.status === "failed") return null;
  if (root.status === "blocked") return firstPending(ledger, root) ? "the goal is waiting on an approval" : null;
  return `the goal is ${root.status}, not failed or stopped`;
}

/**
 * `runningDir`, when given, keeps a blocked root that a live process is
 * resuming in Working: its approval is already decided, but the resume has not
 * yet woken it.
 */
export function toCard(ledger: Ledger, root: RunRow, runningDir?: string): BoardCard {
  const kids = descendants(ledger, root.id).map((d) => d.run);
  const result = readResult(ledger, root.id);
  const pending = root.status === "blocked" ? firstPending(ledger, root) : undefined;

  let approval: BoardCard["approval"] = null;
  if (pending) {
    let args: unknown = null;
    try {
      args = JSON.parse(pending.args_json);
    } catch {
      // A bad row shows the approval without its args rather than sinking the board.
    }
    const path = (args as { artifact_path?: unknown } | null)?.artifact_path;
    let imageId: string | null = null;
    if (typeof path === "string") {
      for (const p of [root, ...kids]) {
        imageId = ledger.listArtifacts(p.id).find((a) => a.kind === "image" && a.path === path)?.id ?? null;
        if (imageId) break;
      }
    }
    approval = { id: pending.id, tool: pending.tool, args, imageId };
  }

  // A blocked root with nothing to decide and nobody resuming it has stopped.
  const stopped =
    root.status === "blocked" && !approval && !(runningDir !== undefined && isRootOwned(runningDir, root.id));
  const column: Column =
    root.status === "done" || root.status === "failed" || stopped
      ? "done"
      : approval
        ? "needs_you"
        : "working";

  let reason: string | null = result?.reason ?? null;
  if (root.status === "blocked" && !approval) reason = blockReason(ledger, root);
  if (root.status === "failed" && reason === null) reason = "no reason recorded";

  return {
    id: root.id,
    goal: root.goal,
    status: root.status,
    column,
    createdAt: root.created_at,
    updatedAt: root.updated_at,
    budgetUsd: root.budget_usd,
    spentUsd: ledger.treeSpentUsd(root.id),
    root: { id: root.id, agent: root.agent, status: root.status },
    agents: kids.map((k) => ({ id: k.id, agent: k.agent, status: k.status })),
    images: imagesOf(ledger, [root.id, ...kids.map((k) => k.id)]),
    summary: result?.summary || summaryFromCheckpoint(ledger, root.id) || null,
    reason,
    approval,
  };
}

export function buildBoard(ledger: Ledger, runningDir?: string): Record<Column, BoardCard[]> {
  const board: Record<Column, BoardCard[]> = { working: [], needs_you: [], done: [] };
  for (const root of ledger.listRootRuns()) {
    const card = toCard(ledger, root, runningDir);
    board[card.column].push(card);
  }
  return board;
}
