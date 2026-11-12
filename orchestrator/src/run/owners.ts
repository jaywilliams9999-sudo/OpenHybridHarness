/**
 * Which process is running a goal.
 *
 * The Hub and a terminal can share one Ledger, and a `running` root looks the
 * same whether its process is alive or was killed mid-turn. Each running root
 * therefore gets a file named for it holding the owner's pid: a live pid means
 * leave it alone, anything else means nobody is coming back for it.
 */

import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export function claimRoot(dir: string, rootId: string): void {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, rootId), String(process.pid));
}

/**
 * Give up this process's claim. Only ours: deleting another live process's
 * file would let the next sweep fail a goal that is still running.
 */
export function releaseRoot(dir: string, rootId: string): void {
  if (claimedPid(dir, rootId) === process.pid) forgetRoot(dir, rootId);
}

/** Remove the claim whoever made it. For a dead owner's file only. */
export function forgetRoot(dir: string, rootId: string): void {
  rmSync(join(dir, rootId), { force: true });
}

export function isRootOwned(dir: string, rootId: string): boolean {
  return rootOwner(dir, rootId) !== undefined;
}

/** The pid that owns a root, if the file names one that is alive. */
export function rootOwner(dir: string, rootId: string): number | undefined {
  const pid = claimedPid(dir, rootId);
  if (pid === undefined) return undefined;
  try {
    // Signal 0 delivers nothing; it only asks whether the pid exists.
    process.kill(pid, 0);
    return pid;
  } catch (error) {
    // Alive, but owned by someone we may not signal.
    return (error as { code?: string }).code === "EPERM" ? pid : undefined;
  }
}

/** The pid in a root's file, alive or not. */
function claimedPid(dir: string, rootId: string): number | undefined {
  let pid: number;
  try {
    pid = Number(readFileSync(join(dir, rootId), "utf8").trim());
  } catch {
    return undefined;
  }
  return Number.isInteger(pid) && pid > 0 ? pid : undefined;
}
