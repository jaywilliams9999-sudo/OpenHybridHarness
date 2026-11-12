/**
 * One approval resumes the tree to its root.
 *
 * Real `dispatch` and `publish` tools, scripted providers, a fake renderer.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger, type Ledger, type RunRow } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { runLoop, type LoopDeps } from "../src/loop/loop.ts";
import { readReport } from "../src/loop/report.ts";
import { resumeTree } from "../src/loop/tree.ts";
import { createDispatchTool } from "../src/tools/dispatch.ts";
import { createPublishTool } from "../src/tools/publish.ts";
import type { AgentDefinition } from "../src/agents/definition.ts";
import type { TurnEvent } from "../src/gateway/types.ts";

const USE_TOOL: TurnEvent = { kind: "end", reason: "tool_use" };

function text(body: string): TurnEvent[] {
  return [
    { kind: "text", text: body },
    { kind: "end", reason: "end_turn" },
  ];
}

function call(id: string, name: string, input: Record<string, unknown>): TurnEvent[] {
  return [{ kind: "tool_request", id, name, input }, USE_TOOL];
}

function scripted(turns: TurnEvent[][]) {
  return { ...createScriptedProvider({ turns }), model: "claude-opus-5-5" };
}

function agent(name: string): AgentDefinition {
  return { name, role: name, tools: [], limits: {} };
}

function fakeRender(ledger: Ledger, file: string): ToolDefinition {
  return {
    name: "gpu_generate_image",
    description: "fake render",
    schema: { type: "object" },
    tier: "budgeted",
    handler: async (_args, ctx) => {
      writeFileSync(file, "png");
      ledger.recordArtifact({ runId: ctx.runId, kind: "image", path: file, meta: {} });
      return { path: file };
    },
  };
}

/**
 * One lead agent that dispatches a site agent. Each agent gets exactly one
 * provider instance, so a script continues across a resume.
 */
function build(
  opsTurns: TurnEvent[][],
  siteTurns: (image: string) => TurnEvent[][],
  extraSiteTools: (ledger: Ledger) => ToolDefinition[] = () => [],
) {
  const dir = mkdtempSync(join(tmpdir(), "ohh-tree-"));
  const ledger = openLedger(":memory:");
  const image = join(dir, "hero.png");
  const publishedDir = join(dir, "published");

  const siteDeps: LoopDeps = {
    provider: scripted(siteTurns(image)),
    registry: createRegistry([
      fakeRender(ledger, image),
      createPublishTool({ ledger, publishedDir }),
      ...extraSiteTools(ledger),
    ]),
    ledger,
    maxTurns: 6,
  };
  const dispatch = createDispatchTool({
    ledger,
    spawn: () => ({ deps: siteDeps, agent: agent("site"), system: "site", placement: "local" }),
  });
  const opsDeps: LoopDeps = {
    provider: scripted(opsTurns),
    registry: createRegistry([dispatch]),
    ledger,
    maxTurns: 6,
  };
  return {
    ledger,
    publishedDir,
    opsDeps,
    depsFor: (run: RunRow): LoopDeps => (run.agent === "lead" ? opsDeps : siteDeps),
    cleanup: () => {
      ledger.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

const SPEC = { agent: "lead", goal: "ship the site", budgetUsd: 1.5 };

function publishedFiles(dir: string): string[] {
  return existsSync(dir) ? readdirSync(dir) : [];
}

function publishTree() {
  return build(
    [
      call("d1", "dispatch", { agent: "site", goal: "hero", budget_usd: 0.5 }),
      text("site published the hero"),
    ],
    (image) => [
      call("r1", "gpu_generate_image", {}),
      call("p1", "publish", { artifact_path: image }),
      text("published"),
    ],
  );
}

async function blockedTree() {
  const t = publishTree();
  const first = await runLoop(t.opsDeps, SPEC);
  const [child] = t.ledger.listChildren(first.runId);
  const [approval] = t.ledger.listApprovals(child!.id);
  return { t, first, childId: child!.id, approvalId: approval!.id };
}

test("lead → site → publish: approve resumes the child and then the parent", async () => {
  const { t, first, childId, approvalId } = await blockedTree();
  try {
    assert.equal(first.status, "blocked");
    assert.equal(first.reason, "child");
    assert.equal(t.ledger.listApprovals(first.runId).length, 0);
    assert.deepEqual(publishedFiles(t.publishedDir), []);

    t.ledger.decideApproval(approvalId, "approved");
    const result = await resumeTree(t.ledger, t.depsFor, childId);

    assert.equal(result.runId, first.runId);
    assert.equal(result.status, "done");
    assert.equal(result.text, "site published the hero");
    assert.deepEqual(publishedFiles(t.publishedDir), ["hero.png"]);
    assert.equal(t.ledger.listChildren(first.runId).length, 1);
    assert.deepEqual(readReport(t.ledger, childId)?.gaps, []);
  } finally {
    t.cleanup();
  }
});

test("deny: the child reports the denial and the parent still finishes", async () => {
  const { t, first, childId, approvalId } = await blockedTree();
  try {
    t.ledger.decideApproval(approvalId, "denied");
    const result = await resumeTree(t.ledger, t.depsFor, childId);

    assert.equal(result.runId, first.runId);
    assert.equal(result.status, "done");
    assert.deepEqual(readReport(t.ledger, childId)?.gaps, ["publish denied"]);
    assert.deepEqual(publishedFiles(t.publishedDir), []);
  } finally {
    t.cleanup();
  }
});

test("a second dispatch is allocated from what the tree has left", async () => {
  const spent = 0.4;
  const t = build(
    [
      call("d1", "dispatch", { agent: "site", goal: "a", budget_usd: 0.6 }),
      call("d2", "dispatch", { agent: "site", goal: "b", budget_usd: 5 }),
      text("both done"),
    ],
    () => [call("s1", "spend", {}), text("first"), text("second")],
    (ledger) => [
      {
        name: "spend",
        description: "records a known cost",
        schema: { type: "object" },
        tier: "budgeted",
        handler: async (_args, ctx) => {
          ledger.recordCost({
            runId: ctx.runId,
            provider: "scripted",
            model: "claude-opus-5-5",
            usage: { in: 0, cacheWrite: 0, cacheRead: 0, out: 0 },
            usd: spent,
          });
          return { spent };
        },
      },
    ],
  );
  try {
    const result = await runLoop(t.opsDeps, SPEC);
    assert.equal(result.status, "done");

    const [first, second] = t.ledger.listChildren(result.runId);
    assert.ok(Math.abs((t.ledger.getRun(first!.id)?.budget_usd ?? 0) - 0.6) < 1e-9);
    const remaining = 1.5 - spent;
    const expected = Math.min(5, remaining - Math.max(0.25, 0.2 * remaining));
    assert.ok(
      Math.abs((t.ledger.getRun(second!.id)?.budget_usd ?? 0) - expected) < 1e-9,
    );
  } finally {
    t.cleanup();
  }
});

test("a child that throws on resume fails, reports up, and the parent still finishes", async () => {
  const t = build(
    [
      call("d1", "dispatch", { agent: "site", goal: "hero", budget_usd: 0.5 }),
      text("site could not finish"),
    ],
    (image) => [
      call("r1", "gpu_generate_image", {}),
      call("p1", "publish", { artifact_path: image }),
      // Never reached: the turn after the approved publish throws instead.
      text("published"),
    ],
  );
  // The site's third turn -- the first one after the approval -- throws, as a
  // GatewayError or GpuError would.
  const siteProvider = (t.depsFor({ agent: "site" } as RunRow)).provider;
  let calls = 0;
  const chat = siteProvider.chat.bind(siteProvider);
  (siteProvider as { chat: typeof chat }).chat = (request) => {
    calls++;
    if (calls === 3) throw new Error("gateway down");
    return chat(request);
  };
  try {
    const first = await runLoop(t.opsDeps, SPEC);
    const [child] = t.ledger.listChildren(first.runId);
    const [approval] = t.ledger.listApprovals(child!.id);
    t.ledger.decideApproval(approval!.id, "approved");

    const result = await resumeTree(t.ledger, t.depsFor, child!.id);

    assert.equal(result.runId, first.runId);
    assert.equal(result.status, "done");
    assert.equal(result.text, "site could not finish");
    assert.equal(t.ledger.getRun(child!.id)?.status, "failed");
    const report = readReport(t.ledger, child!.id);
    assert.equal(report?.status, "failed");
    assert.deepEqual(report?.gaps, ["gateway down"]);
    assert.equal(t.ledger.getRun(first.runId)?.status, "done");
  } finally {
    t.cleanup();
  }
});

test("a root that throws on resume still throws", async () => {
  const t = build([call("d1", "dispatch", { agent: "site", goal: "hero", budget_usd: 0.5 })], () => [
    call("r1", "gpu_generate_image", {}),
    call("p1", "publish", { artifact_path: "nope" }),
  ]);
  try {
    const rootId = t.ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1 });
    t.ledger.saveCheckpoint(rootId, "not json");
    t.ledger.setRunStatus(rootId, "blocked");
    await assert.rejects(resumeTree(t.ledger, t.depsFor, rootId), /not valid JSON/);
  } finally {
    t.cleanup();
  }
});

test("a Run that already ended is not resumed", async () => {
  const t = publishTree();
  try {
    const first = await runLoop(t.opsDeps, SPEC);
    const [child] = t.ledger.listChildren(first.runId);
    t.ledger.decideApproval(t.ledger.listApprovals(child!.id)[0]!.id, "approved");
    await resumeTree(t.ledger, t.depsFor, child!.id);
    const reports = () =>
      t.ledger.listMessages(child!.id).filter((m) => m.kind === "report").length;
    assert.equal(reports(), 1);

    for (const id of [child!.id, first.runId]) {
      await assert.rejects(resumeTree(t.ledger, t.depsFor, id), (error: Error) => {
        assert.equal(error.name, "ResumeError");
        assert.match(error.message, /already done/);
        return true;
      });
    }
    assert.equal(reports(), 1);
    assert.equal(t.ledger.listChildren(first.runId).length, 1);
  } finally {
    t.cleanup();
  }
});
