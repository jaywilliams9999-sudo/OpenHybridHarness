/**
 * The Hub's runner: the shared runner, shaped for a server.
 *
 * Every method decides synchronously whether it will run -- a refusal is a
 * `RunRefused` thrown before any promise exists -- so the server can answer
 * 404 or 409 before it answers 202. What it then returns is the background
 * drive.
 *
 * The ownership files stop a second *process* driving a tree; nothing in them
 * stops this process driving one tree twice, because the owner they name is
 * us. So the roots being driven here are also held in memory, and a decision
 * or a retry on one of them is refused.
 */

import type { Ledger } from "../ledger/ledger.ts";
import { isRootOwned } from "../run/owners.ts";
import {
  RunRefused,
  recordDecision,
  resumeGoal,
  rootOf,
  startGoal,
  type RunnerOptions,
} from "../run/runner.ts";
import { againRefusal } from "./board.ts";

/** The agent every new Hub goal starts under: lead plans, and dispatches the rest. */
const HUB_AGENT = "lead";

/** The most a goal started from the Hub may spend. */
export const HUB_MAX_BUDGET_USD = 10;

export interface HubRunner {
  startGoal(req: { goal: string; budgetUsd: number }): Promise<unknown>;
  decide(approvalId: string, d: "approved" | "denied"): Promise<unknown>;
  again(rootId: string): Promise<unknown>;
}

/** `settled` is told whenever a drive ends, however it ends. */
export function createHubRunner(opts: RunnerOptions, settled: () => void): HubRunner {
  const { ledger } = opts;
  const driving = new Set<string>();

  const refuseIfDriving = (rootId: string): void => {
    if (driving.has(rootId)) throw new RunRefused("already running", "");
  };

  /** Hold `rootId` until `work` settles, and say so when it does. */
  const hold = (rootId: string, work: () => Promise<unknown>): Promise<unknown> => {
    driving.add(rootId);
    return work().finally(() => {
      driving.delete(rootId);
      settled();
    });
  };

  /** A new goal; its root is held from the moment it exists. */
  const begin = (agent: string, goal: string, budgetUsd: number): Promise<unknown> => {
    let rootId: string | undefined;
    return startGoal(opts, {
      agent,
      goal,
      budgetUsd,
      onCreated: (id) => {
        rootId = id;
        driving.add(id);
      },
    }).finally(() => {
      if (rootId !== undefined) driving.delete(rootId);
      settled();
    });
  };

  return {
    startGoal: (req) => begin(HUB_AGENT, req.goal, req.budgetUsd),

    decide(approvalId, d) {
      const rootId = approvalRoot(ledger, approvalId);
      if (rootId !== undefined) refuseIfDriving(rootId);
      // Throws for an unknown or decided approval, or a tree another process
      // owns -- all before anything is written.
      const approval = recordDecision(ledger, opts.config.runningDir, approvalId, d);
      return hold(rootId ?? approval.run_id, () => resumeGoal(opts, approval.run_id));
    },

    again(rootId) {
      const old = ledger.getRun(rootId);
      if (!old) throw new RunRefused("unknown run", rootId);
      const refusal = againRefusal(ledger, old);
      if (refusal !== null) throw new RunRefused(refusal, "");
      refuseIfDriving(rootId);
      // A stopped root another process is resuming is not ours to start over.
      if (isRootOwned(opts.config.runningDir, rootId)) throw new RunRefused("already running", "");
      // A terminal goal may have been given more than the Hub allows.
      if (old.budget_usd > HUB_MAX_BUDGET_USD) {
        throw new RunRefused(
          `budget $${old.budget_usd.toFixed(2)} is over the Hub's $${HUB_MAX_BUDGET_USD} limit;`,
          "start it again from the goal box",
        );
      }
      // The same agent, text and budget; the old root is left as it is.
      // Held under the old root's id too, so a double click is one retry.
      return hold(rootId, () => begin(old.agent, old.goal, old.budget_usd));
    },
  };
}

function approvalRoot(ledger: Ledger, approvalId: string): string | undefined {
  const approval = ledger.readApproval(approvalId);
  const run = approval ? ledger.getRun(approval.run_id) : undefined;
  return run ? rootOf(ledger, run) : undefined;
}
