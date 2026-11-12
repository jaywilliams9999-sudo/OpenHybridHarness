/**
 * Which process is running a goal: one pid file per root.
 *
 * The Hub and a terminal may share one Ledger, so "running" alone cannot say
 * whether a goal is alive or was interrupted. A file holding a live pid can.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { claimRoot, isRootOwned, releaseRoot, rootOwner } from "../src/run/owners.ts";

function tempDir(): { dir: string; cleanup: () => void } {
  const base = mkdtempSync(join(tmpdir(), "ohh-owners-"));
  // A subdirectory that does not exist yet: claiming must create it.
  return { dir: join(base, "running"), cleanup: () => rmSync(base, { recursive: true, force: true }) };
}

test("a claimed root is owned", () => {
  const { dir, cleanup } = tempDir();
  try {
    claimRoot(dir, "root-1");
    assert.equal(isRootOwned(dir, "root-1"), true);
    assert.equal(isRootOwned(dir, "root-2"), false);
  } finally {
    cleanup();
  }
});

test("a released root is not", () => {
  const { dir, cleanup } = tempDir();
  try {
    claimRoot(dir, "root-1");
    releaseRoot(dir, "root-1");
    assert.equal(isRootOwned(dir, "root-1"), false);
    // Releasing what was never claimed is not an error.
    releaseRoot(dir, "root-1");
  } finally {
    cleanup();
  }
});

test("a pid that is not alive is not owned", () => {
  const { dir, cleanup } = tempDir();
  try {
    claimRoot(dir, "root-1");
    writeFileSync(join(dir, "root-1"), "2147483646");
    assert.equal(isRootOwned(dir, "root-1"), false);
  } finally {
    cleanup();
  }
});

test("releasing leaves another process's claim alone", () => {
  const { dir, cleanup } = tempDir();
  try {
    claimRoot(dir, "root-1");
    writeFileSync(join(dir, "root-1"), String(process.ppid));
    releaseRoot(dir, "root-1");
    assert.equal(readFileSync(join(dir, "root-1"), "utf8"), String(process.ppid));
    assert.equal(rootOwner(dir, "root-1"), process.ppid);
  } finally {
    cleanup();
  }
});
