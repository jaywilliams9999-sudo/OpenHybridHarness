/** The Hub's read side: the board and a goal's story, from a seeded Ledger. */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

import { openLedger, type Ledger } from "../src/ledger/ledger.ts";
import { recordResult } from "../src/run/result.ts";
import { buildBoard } from "../src/hub/board.ts";
import { goalDetail } from "../src/hub/detail.ts";
import { claimRoot } from "../src/run/owners.ts";

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "ohh-hub-board-"));
  const ledger = openLedger(join(dir, "ledger.db"));
  return {
    dir,
    ledger,
    cleanup: () => {
      ledger.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

const root = (l: Ledger, goal = "make a logo", budgetUsd = 1.5) =>
  l.createRun({ agent: "lead", goal, parentId: null, budgetUsd });
const child = (l: Ledger, parentId: string) =>
  l.createRun({ agent: "site", goal: "sub", parentId, budgetUsd: 0.5 });

function spend(l: Ledger, runId: string, usd: number): void {
  l.recordCost({
    runId,
    provider: "scripted",
    model: "claude-opus-5-5",
    usage: { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 },
    usd,
  });
}

test("a running root is working", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const board = buildBoard(ledger);
    assert.deepEqual(board.working.map((c) => c.id), [id]);
    assert.equal(board.working[0]!.column, "working");
    assert.equal(board.needs_you.length + board.done.length, 0);
  } finally {
    cleanup();
  }
});

test("a done root is done with its result summary", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    spend(ledger, id, 0.25);
    recordResult(ledger, id, { status: "done", summary: "logo made" });
    ledger.setRunStatus(id, "done");
    const card = buildBoard(ledger).done[0]!;
    assert.equal(card.summary, "logo made");
    assert.equal(card.spentUsd, 0.25);
    assert.equal(card.reason, null);
  } finally {
    cleanup();
  }
});

test("a failed root is done and carries its reason", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    recordResult(ledger, id, { status: "failed", reason: "interrupted", summary: "" });
    ledger.setRunStatus(id, "failed");
    const card = buildBoard(ledger).done[0]!;
    assert.equal(card.status, "failed");
    assert.equal(card.reason, "interrupted");
    assert.equal(card.summary, null);
  } finally {
    cleanup();
  }
});

test("a child's pending publish puts the root in needs_you with the image id", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const kid = child(ledger, id);
    ledger.recordArtifact({ runId: kid, kind: "image", path: "/a/one.png", meta: {} });
    const wanted = ledger.listArtifacts(kid)[0]!.id;
    ledger.requestApproval({ runId: kid, tool: "publish", args: { artifact_path: "/a/one.png" } });
    ledger.setRunStatus(kid, "blocked");
    ledger.setRunStatus(id, "blocked");
    const board = buildBoard(ledger);
    assert.equal(board.working.length, 0);
    const card = board.needs_you[0]!;
    assert.equal(card.approval?.tool, "publish");
    assert.equal(card.approval?.imageId, wanted);
    assert.deepEqual(card.approval?.args, { artifact_path: "/a/one.png" });
  } finally {
    cleanup();
  }
});

test("a root stopped on budget is done, saying so", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger, "x", 0.1);
    spend(ledger, id, 0.2);
    ledger.setRunStatus(id, "blocked");
    const board = buildBoard(ledger);
    assert.equal(board.working.length + board.needs_you.length, 0);
    const card = board.done[0]!;
    assert.equal(card.id, id);
    assert.equal(card.status, "blocked");
    assert.equal(card.reason, "stopped: budget spent");
    assert.equal(card.approval, null);
  } finally {
    cleanup();
  }
});

test("a blocked root with nothing pending anywhere is done with its reason", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const kid = child(ledger, id);
    // A decided approval is not pending.
    const decided = ledger.requestApproval({ runId: kid, tool: "publish", args: {} });
    ledger.decideApproval(decided, "denied");
    ledger.setRunStatus(kid, "blocked");
    ledger.setRunStatus(id, "blocked");
    const card = buildBoard(ledger).done[0]!;
    assert.equal(card.id, id);
    assert.equal(card.reason, "blocked");
  } finally {
    cleanup();
  }
});

test("a blocked root a live process is resuming stays working", () => {
  const { dir, ledger, cleanup } = setup();
  try {
    const runningDir = join(dir, "running");
    const id = root(ledger, "x", 0.1);
    spend(ledger, id, 0.2);
    ledger.setRunStatus(id, "blocked");
    claimRoot(runningDir, id);
    const board = buildBoard(ledger, runningDir);
    assert.equal(board.working[0]?.id, id);
    assert.equal(board.done.length, 0);
    assert.equal(goalDetail(ledger, id, runningDir)!.card.column, "working");
  } finally {
    cleanup();
  }
});

test("children are not cards", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const kid = child(ledger, id);
    const board = buildBoard(ledger);
    const all = [...board.working, ...board.needs_you, ...board.done];
    assert.deepEqual(all.map((c) => c.id), [id]);
    assert.deepEqual(all[0]!.agents, [{ id: kid, agent: "site", status: "running" }]);
  } finally {
    cleanup();
  }
});

test("a card names its root's own agent, children or not", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const card = buildBoard(ledger).working[0]!;
    assert.deepEqual(card.root, { id, agent: "lead", status: "running" });
    assert.deepEqual(card.agents, []);
  } finally {
    cleanup();
  }
});

test("images come from the whole tree", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const kid = child(ledger, id);
    ledger.recordArtifact({ runId: id, kind: "image", path: "/a/root.png", meta: {} });
    ledger.recordArtifact({ runId: kid, kind: "image", path: "/a/kid.png", meta: {} });
    ledger.recordArtifact({ runId: kid, kind: "file", path: "/a/notes.txt", meta: {} });
    const card = buildBoard(ledger).working[0]!;
    const images = [...ledger.listArtifacts(id), ...ledger.listArtifacts(kid)]
      .filter((a) => a.kind === "image")
      .map((a) => a.id);
    assert.equal(card.images.length, 2);
    assert.deepEqual(new Set(card.images), new Set(images));
  } finally {
    cleanup();
  }
});

test("an old root with no result uses the checkpoint's last text", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    ledger.saveCheckpoint(
      id,
      JSON.stringify({
        version: 1,
        agent: "lead",
        goal: "g",
        budgetUsd: 1,
        turns: 1,
        spentUsd: 0,
        transcript: [{ role: "assistant", events: [{ kind: "text", text: "nearly there" }] }],
      }),
    );
    ledger.setRunStatus(id, "done");
    assert.equal(buildBoard(ledger).done[0]!.summary, "nearly there");
  } finally {
    cleanup();
  }
});

test("an old root with neither says null", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    ledger.setRunStatus(id, "done");
    assert.equal(buildBoard(ledger).done[0]!.summary, null);
  } finally {
    cleanup();
  }
});

test("goalDetail lists every run in the tree with depth", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    const kid = child(ledger, id);
    const grandkid = child(ledger, kid);
    ledger.recordArtifact({ runId: grandkid, kind: "image", path: "/a/g.png", meta: {} });
    const detail = goalDetail(ledger, id)!;
    assert.equal(detail.card.id, id);
    assert.deepEqual(
      detail.runs.map((p) => [p.id, p.depth]),
      [[id, 0], [kid, 1], [grandkid, 2]],
    );
    assert.equal(detail.runs[2]!.images.length, 1);
    assert.deepEqual(detail.runs[0]!.toolCalls, []);
  } finally {
    cleanup();
  }
});

test("goalDetail of an unknown id is undefined", () => {
  const { ledger, cleanup } = setup();
  try {
    assert.equal(goalDetail(ledger, "nope"), undefined);
    const id = root(ledger);
    assert.equal(goalDetail(ledger, child(ledger, id)), undefined);
  } finally {
    cleanup();
  }
});

test("a failed root with no result still says why", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    ledger.setRunStatus(id, "failed");
    assert.equal(buildBoard(ledger).done[0]!.reason, "no reason recorded");
  } finally {
    cleanup();
  }
});

test("a failed root keeps its reason and the checkpoint's last text", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    ledger.saveCheckpoint(
      id,
      JSON.stringify({
        version: 1,
        agent: "lead",
        goal: "g",
        budgetUsd: 1,
        turns: 1,
        spentUsd: 0,
        transcript: [{ role: "assistant", events: [{ kind: "text", text: "got this far" }] }],
      }),
    );
    recordResult(ledger, id, { status: "failed", reason: "interrupted", summary: "" });
    ledger.setRunStatus(id, "failed");
    const card = buildBoard(ledger).done[0]!;
    assert.equal(card.reason, "interrupted");
    assert.equal(card.summary, "got this far");
  } finally {
    cleanup();
  }
});

test("an approval's imageId ignores a non-image artifact at that path", () => {
  const { ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    ledger.recordArtifact({ runId: id, kind: "file", path: "/a/x.txt", meta: {} });
    ledger.requestApproval({ runId: id, tool: "publish", args: { artifact_path: "/a/x.txt" } });
    ledger.setRunStatus(id, "blocked");
    assert.equal(buildBoard(ledger).needs_you[0]!.approval?.imageId, null);
  } finally {
    cleanup();
  }
});

test("an approval with unparseable args does not sink the board", () => {
  const { dir, ledger, cleanup } = setup();
  try {
    const id = root(ledger);
    ledger.requestApproval({ runId: id, tool: "publish", args: {} });
    const row = ledger.listApprovals(id)[0]!;
    ledger.setRunStatus(id, "blocked");
    // Corrupt the row behind the Ledger's back.
    const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite") as typeof import("node:sqlite");
    const db = new DatabaseSync(join(dir, "ledger.db"));
    db.prepare("UPDATE approvals SET args_json = ? WHERE id = ?").run("{nope", row.id);
    db.close();
    const card = buildBoard(ledger).needs_you[0]!;
    assert.equal(card.approval?.args, null);
    assert.equal(card.approval?.imageId, null);
  } finally {
    cleanup();
  }
});
