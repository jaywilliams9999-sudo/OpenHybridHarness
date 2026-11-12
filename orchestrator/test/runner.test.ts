/**
 * The shared runner: the one way the terminal and the Hub start a goal,
 * decide an approval and resume a tree.
 *
 * Scripted providers and real `dispatch`/`publish` tools, as in
 * `tree.test.ts`; the GPU service is a stub nothing here calls.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger, type Ledger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { createDispatchTool, type ChildAgent } from "../src/tools/dispatch.ts";
import { createPublishTool } from "../src/tools/publish.ts";
import type { Config } from "../src/config/config.ts";
import type { GpuClient } from "../src/gpu/client.ts";
import type { AgentDefinition } from "../src/agents/definition.ts";
import type { Provider, TurnEvent } from "../src/gateway/types.ts";
import { claimRoot, isRootOwned } from "../src/run/owners.ts";
import { readResult } from "../src/run/result.ts";
import {
  decideApproval,
  resumeGoal,
  rootBudget,
  startGoal,
  type RunnerOptions,
} from "../src/run/runner.ts";

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

function scripted(turns: TurnEvent[][]): Provider {
  return { ...createScriptedProvider({ turns }), model: "claude-opus-5-5" };
}

function agent(name: string, limits: AgentDefinition["limits"] = {}): AgentDefinition {
  return { name, role: name, tools: [], limits };
}

function config(runningDir: string): Config {
  return {
    gpuUrl: "http://gpu.invalid:1",
    gpuToken: "x",
    ledgerPath: join(runningDir, "..", "ledger.db"),
    runningDir,
    artifactsDir: join(runningDir, "..", "artifacts"),
    publishedDir: join(runningDir, "..", "published"),
    agentsDir: join(import.meta.dirname, "..", "agents"),
    maxTurns: 6,
    budgetUsd: 2,
    mode: "scripted",
    ollama: {},
    localModels: new Set(),
  };
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
 * A lead agent that may dispatch a site agent, behind the runner's factory
 * seam. Each agent has one provider for the whole test, so a script continues
 * across a resume exactly as the real factory's cache makes it.
 */
function build(
  opsProvider: Provider,
  siteTurns: (image: string) => TurnEvent[][] = () => [],
  gpu: GpuClient = {} as GpuClient,
) {
  const dir = mkdtempSync(join(tmpdir(), "ohh-runner-"));
  const ledger = openLedger(join(dir, "ledger.db"));
  const runningDir = join(dir, "running");
  const image = join(dir, "hero.png");

  const site: ChildAgent = {
    agent: agent("site"),
    system: "site",
    placement: "local",
    deps: {
      provider: scripted(siteTurns(image)),
      registry: createRegistry([
        fakeRender(ledger, image),
        createPublishTool({ ledger, publishedDir: join(dir, "published") }),
      ]),
      ledger,
      maxTurns: 6,
    },
  };
  const lead: ChildAgent = {
    agent: agent("lead"),
    system: "lead",
    placement: "local",
    deps: {
      provider: opsProvider,
      registry: createRegistry([createDispatchTool({ ledger, spawn: () => site })]),
      ledger,
      maxTurns: 6,
    },
  };

  const prompts: string[] = [];
  const opts: RunnerOptions = {
    config: config(runningDir),
    // Default cast on purpose: a call into it would mean a test dialled the GPU service.
    gpu,
    ledger,
    harness: "HARNESS",
    output: () => ({}),
    factory: (prompt) => {
      prompts.push(prompt);
      return { agent: (name) => (name === "lead" ? lead : site) };
    },
  };
  return {
    ledger,
    runningDir,
    opts,
    prompts,
    cleanup: () => {
      ledger.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

/** The lead → site → publish script: the site blocks on `publish`. */
function publishTree() {
  return build(
    scripted([
      call("d1", "dispatch", { agent: "site", goal: "hero", budget_usd: 0.5 }),
      text("site published the hero"),
    ]),
    (image) => [
      call("r1", "gpu_generate_image", {}),
      call("p1", "publish", { artifact_path: image }),
      text("published"),
    ],
  );
}

const GOAL = { agent: "lead", goal: "ship the site", budgetUsd: 1.5 };

function throwing(message: string): Provider {
  return {
    name: "scripted",
    model: "claude-opus-5-5",
    chat: () => {
      throw new Error(message);
    },
  };
}

function isRefused(pattern: RegExp) {
  return (error: Error) => {
    assert.equal(error.name, "RunRefused");
    assert.match(error.message, pattern);
    return true;
  };
}

test("startGoal writes a done result with the root's final text", async () => {
  const t = build(scripted([text("all done")]));
  try {
    const result = await startGoal(t.opts, GOAL);

    assert.equal(result.status, "done");
    assert.deepEqual(t.prompts, ["ship the site"]);
    const recorded = readResult(t.ledger, result.runId);
    assert.equal(recorded?.status, "done");
    assert.equal(recorded?.summary, "all done");
    assert.equal(recorded?.reason, result.reason ?? null);
    assert.equal(t.ledger.getRun(result.runId)?.budget_usd, 1.5);
  } finally {
    t.cleanup();
  }
});

test("startGoal releases ownership on done, blocked and throw", async () => {
  for (const make of [
    () => build(scripted([text("all done")])),
    publishTree,
    () => build(throwing("boom")),
  ]) {
    const t = make();
    try {
      let rootId: string | undefined;
      let ownedWhileRunning = false;
      await startGoal(t.opts, {
        ...GOAL,
        onCreated: (id) => {
          rootId = id;
          ownedWhileRunning = isRootOwned(t.runningDir, id);
        },
      }).catch(() => undefined);

      assert.ok(rootId);
      assert.equal(ownedWhileRunning, true);
      assert.equal(isRootOwned(t.runningDir, rootId), false);
    } finally {
      t.cleanup();
    }
  }
});

test("a blocked root leaves no result", async () => {
  const t = publishTree();
  try {
    const result = await startGoal(t.opts, GOAL);
    assert.equal(result.status, "blocked");
    assert.equal(readResult(t.ledger, result.runId), undefined);
  } finally {
    t.cleanup();
  }
});

test("a root that throws is failed with the error as reason", async () => {
  const t = build(throwing("boom"));
  try {
    let rootId: string | undefined;
    await assert.rejects(
      startGoal(t.opts, { ...GOAL, onCreated: (id) => (rootId = id) }),
      /boom/,
    );

    assert.ok(rootId);
    assert.equal(t.ledger.getRun(rootId)?.status, "failed");
    const recorded = readResult(t.ledger, rootId);
    assert.equal(recorded?.status, "failed");
    assert.equal(recorded?.reason, "boom");
  } finally {
    t.cleanup();
  }
});

test("decideApproval on a decided approval is RunRefused", async () => {
  const t = publishTree();
  try {
    const first = await startGoal(t.opts, GOAL);
    const [child] = t.ledger.listChildren(first.runId);
    const [approval] = t.ledger.listApprovals(child!.id);
    t.ledger.decideApproval(approval!.id, "approved");

    await assert.rejects(
      decideApproval(t.opts, approval!.id, "denied"),
      isRefused(/^already decided publish is approved$/),
    );
    await assert.rejects(
      decideApproval(t.opts, "nope", "approved"),
      isRefused(/^unknown approval nope$/),
    );
  } finally {
    t.cleanup();
  }
});

test("decideApproval resumes the tree and records the root's result", async () => {
  const t = publishTree();
  try {
    const first = await startGoal(t.opts, GOAL);
    assert.equal(first.status, "blocked");
    const [child] = t.ledger.listChildren(first.runId);
    const [pending] = t.ledger.listApprovals(child!.id);

    const { approval, result } = await decideApproval(t.opts, pending!.id, "approved");

    assert.equal(approval.id, pending!.id);
    assert.equal(approval.tool, "publish");
    assert.equal(result.runId, first.runId);
    assert.equal(result.status, "done");
    assert.equal(t.ledger.getRun(child!.id)?.status, "done");
    const recorded = readResult(t.ledger, first.runId);
    assert.equal(recorded?.status, "done");
    assert.equal(recorded?.summary, "site published the hero");
    // The root's result only: a child reports to its parent, not the operator.
    assert.equal(readResult(t.ledger, child!.id), undefined);
    assert.equal(isRootOwned(t.runningDir, first.runId), false);
  } finally {
    t.cleanup();
  }
});

test("resumeGoal refuses a done run", async () => {
  const t = build(scripted([text("all done")]));
  try {
    const done = await startGoal(t.opts, GOAL);

    await assert.rejects(
      resumeGoal(t.opts, done.runId),
      isRefused(
        new RegExp(
          `^not resumable ${done.runId} is already done; only a blocked Run can be resumed$`,
        ),
      ),
    );
    await assert.rejects(resumeGoal(t.opts, "nope"), isRefused(/^unknown run nope$/));
  } finally {
    t.cleanup();
  }
});

test("rootBudget prefers the explicit figure, then OHH_BUDGET_USD, then the agent", () => {
  const cfg = { ...config(tmpdir()), budgetUsd: 2 };
  const capped = agent("lead", { maxUsd: 1.5 });

  assert.equal(rootBudget(cfg, capped, 0.75, { OHH_BUDGET_USD: "2" }), 0.75);
  assert.equal(rootBudget(cfg, capped, undefined, { OHH_BUDGET_USD: "2" }), 2);
  assert.equal(rootBudget(cfg, capped, undefined, {}), 1.5);
  assert.equal(rootBudget(cfg, agent("site"), undefined, {}), 2);
});

/** A blocked tree whose root another live process claims. */
async function claimedElsewhere() {
  const t = publishTree();
  const first = await startGoal(t.opts, GOAL);
  const [child] = t.ledger.listChildren(first.runId);
  const [approval] = t.ledger.listApprovals(child!.id);
  claimRoot(t.runningDir, first.runId);
  const file = join(t.runningDir, first.runId);
  // The parent process: alive for as long as this test runs.
  writeFileSync(file, String(process.ppid));
  return { t, rootId: first.runId, childId: child!.id, approvalId: approval!.id, file };
}

test("resumeGoal refuses a tree another live process owns, and leaves its claim", async () => {
  const { t, rootId, childId, file } = await claimedElsewhere();
  try {
    await assert.rejects(
      resumeGoal(t.opts, childId),
      isRefused(new RegExp(`^not resumable ${rootId} is running in another process$`)),
    );
    assert.equal(readFileSync(file, "utf8"), String(process.ppid));
    assert.equal(t.ledger.getRun(rootId)?.status, "blocked");
  } finally {
    t.cleanup();
  }
});

test("decideApproval refuses before deciding when another process owns the tree", async () => {
  const { t, rootId, approvalId, file } = await claimedElsewhere();
  try {
    await assert.rejects(
      decideApproval(t.opts, approvalId, "approved"),
      isRefused(new RegExp(`^not resumable ${rootId} is running in another process$`)),
    );
    assert.equal(t.ledger.readApproval(approvalId)?.status, "pending");
    assert.equal(readFileSync(file, "utf8"), String(process.ppid));
  } finally {
    t.cleanup();
  }
});

function statusGpu(models: Array<{ role: string; available: boolean }>): GpuClient {
  return {
    status: async () => ({
      vram: { total_mb: 1, free_mb: 1, usable_free_mb: 1 },
      resident: [],
      busy: false,
      queue_depth: 0,
      kv_cache_type: "f16",
      models: models.map((m, i) => ({
        name: `m${i}`,
        role: m.role,
        available: m.available,
        num_ctx: null,
        reduced_from: null,
        reason: m.available ? null : "too big",
      })),
    }),
  } as unknown as GpuClient;
}

test("runner records the image-skipped notice when the card can't fit the image model", async () => {
  const t = build(
    scripted([text("done")]),
    undefined,
    statusGpu([{ role: "code", available: true }, { role: "image", available: false }]),
  );
  try {
    const result = await startGoal(t.opts, GOAL);
    const notice = t.ledger.listMessages(result.runId).find((m) => m.kind === "notice");
    assert.ok(notice);
    assert.equal(notice.from_agent, "orchestrator");
    assert.equal(notice.to_agent, "lead");
    assert.equal(JSON.parse(notice.body), "image skipped: model doesn't fit this card");
  } finally {
    t.cleanup();
  }
});

test("runner records no notice when it can", async () => {
  const t = build(
    scripted([text("done")]),
    undefined,
    statusGpu([{ role: "image", available: true }]),
  );
  try {
    const result = await startGoal(t.opts, GOAL);
    assert.equal(t.ledger.listMessages(result.runId).filter((m) => m.kind === "notice").length, 0);
  } finally {
    t.cleanup();
  }
});

test("runner starts the goal with no notice when gpu.status() rejects", async () => {
  const seen: boolean[] = [];
  const t = build(scripted([text("done")]), undefined, {
    status: async () => {
      throw new Error("gpu down");
    },
  } as unknown as GpuClient);
  const inner = t.opts.factory!;
  t.opts = { ...t.opts, factory: (p, images) => (seen.push(images), inner(p, images)) };
  try {
    const result = await startGoal(t.opts, GOAL);
    assert.equal(result.status, "done");
    assert.deepEqual(seen, [true]);
    assert.equal(t.ledger.listMessages(result.runId).filter((m) => m.kind === "notice").length, 0);
  } finally {
    t.cleanup();
  }
});

test("with no image model, the recorded summary ends with the skip notice whatever the root said", async () => {
  // The root's own prose is not trusted to say it: on the local model it
  // claimed a hero image nobody made.
  const notices: string[] = [];
  const t = build(
    scripted([text("The landing page is built, hero image included.")]),
    undefined,
    statusGpu([{ role: "image", available: false }]),
  );
  t.opts = { ...t.opts, onNotice: (line) => notices.push(line) };
  try {
    const result = await startGoal(t.opts, GOAL);
    const summary = readResult(t.ledger, result.runId)?.summary;
    assert.equal(
      summary,
      "The landing page is built, hero image included.\n\n" +
        "[orchestrator] image skipped: model doesn't fit this card",
    );
    assert.deepEqual(notices, ["[orchestrator] image skipped: model doesn't fit this card"]);
  } finally {
    t.cleanup();
  }
});

test("with an image model, the summary is the root's text alone and nothing is noticed", async () => {
  const notices: string[] = [];
  const t = build(scripted([text("done")]), undefined, statusGpu([{ role: "image", available: true }]));
  t.opts = { ...t.opts, onNotice: (line) => notices.push(line) };
  try {
    const result = await startGoal(t.opts, GOAL);
    assert.equal(readResult(t.ledger, result.runId)?.summary, "done");
    assert.deepEqual(notices, []);
  } finally {
    t.cleanup();
  }
});

test("a resumed goal that started without images gets the skip notice in its summary", async () => {
  // Decided by the root's recorded notice, not today's card: the goal ran
  // without images, whatever a later status says.
  const t = build(
    scripted([
      call("d1", "dispatch", { agent: "site", goal: "hero", budget_usd: 0.5 }),
      text("all published"),
    ]),
    // The scripted tree only needs a gated call to block on; the fake render
    // stands in for any work before it.
    (image) => [
      call("r1", "gpu_generate_image", {}),
      call("s1", "publish", { artifact_path: image }),
      text("published"),
    ],
    statusGpu([{ role: "image", available: false }]),
  );
  try {
    const started = await startGoal(t.opts, GOAL);
    assert.equal(started.status, "blocked");
    const approval = t.ledger
      .listBlockedRuns()
      .flatMap((run) => t.ledger.listApprovals(run.id))
      .find((a) => a.status === "pending");
    assert.ok(approval);
    const { result } = await decideApproval(t.opts, approval.id, "approved");
    assert.equal(result.status, "done");
    assert.equal(
      readResult(t.ledger, result.runId)?.summary,
      "all published\n\n[orchestrator] image skipped: model doesn't fit this card",
    );
  } finally {
    t.cleanup();
  }
});
