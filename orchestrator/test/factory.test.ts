/**
 * The agent factory: one place that turns an agent name and a depth into
 * a runnable agent, for a fresh run, a dispatched child, and a resume alike.
 *
 * Nothing here dials the GPU service or the API: the GPU service client is a stub and the
 * provider is scripted.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openLedger } from "../src/ledger/ledger.ts";
import {
  IMAGE_UNAVAILABLE,
  LOCAL_ADDENDUM,
  createAgentFactory,
  depthOf,
  runDeps,
} from "../src/agents/factory.ts";
import { RouteError } from "../src/gateway/route.ts";
import { openLedger as openFreshLedger, ranAs } from "../src/ledger/ledger.ts";
import type { Config } from "../src/config/config.ts";
import type { GpuClient } from "../src/gpu/client.ts";

const AGENTS = join(import.meta.dirname, "..", "agents");

function config(extra: Partial<Config> = {}): Config {
  return {
    gpuUrl: "http://gpu.invalid:1",
    gpuToken: "x",
    ledgerPath: ":memory:",
    artifactsDir: join(import.meta.dirname, "..", "artifacts-test"),
    publishedDir: join(import.meta.dirname, "..", "published-test"),
    agentsDir: AGENTS,
    runningDir: join(import.meta.dirname, "..", "running-test"),
    maxTurns: 7,
    budgetUsd: 1,
    mode: "scripted",
    ollama: {},
    localModels: new Set(["llama3.1:8b", "qwen2.5-coder:7b"]),
    ...extra,
  };
}

function setup(extra: Partial<Config> = {}) {
  const ledger = openLedger(":memory:");
  // Cast on purpose: a call into it would mean a test dialled the GPU service.
  const gpu = {} as GpuClient;
  const factory = createAgentFactory({
    config: config(extra),
    gpu,
    ledger,
    harness: "HARNESS",
    output: () => ({}),
  });
  return { ledger, factory };
}

const names = (deps: { registry: { list(): { name: string }[] } }) =>
  deps.registry.list().map((tool) => tool.name).sort();

function withEnv(name: string, value: string | undefined, fn: () => void): void {
  const before = process.env[name];
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
  try {
    fn();
  } finally {
    if (before === undefined) delete process.env[name];
    else process.env[name] = before;
  }
}

test("the same (agent, depth) is the same agent, so a scripted provider keeps its place", () => {
  const { factory } = setup();

  const first = factory.agent("site", 0);
  assert.equal(factory.agent("site", 0), first);
  assert.notEqual(factory.agent("site", 1), first);
  assert.notEqual(factory.agent("lead", 0), first);
});

test("lead sees exactly dispatch and gpu_status", () => {
  const { factory } = setup();
  const lead = factory.agent("lead", 0);

  assert.deepEqual(names(lead.deps), ["dispatch", "gpu_status"]);
  assert.equal(lead.agent.name, "lead");
  assert.match(lead.system, /HARNESS/);
  assert.match(lead.system, /You coordinate other agents\./);
});

test("a local agent's system prompt ends with the local addendum", () => {
  const { factory } = setup({ mode: "routed" });
  // No key: site is local and lead (a claude-* model) is local-no-key.
  for (const name of ["site", "lead"]) {
    const built = factory.agent(name, 0);
    assert.notEqual(built.placement, "claude");
    assert.ok(built.system.endsWith(`\n\n${LOCAL_ADDENDUM}`), name);
  }
});

test("a local agent with no tools gets no local addendum", () => {
  const { factory } = setup({ mode: "routed" });
  const scribe = factory.agent("scribe", 1);
  assert.equal(scribe.placement, "local");
  assert.ok(!scribe.system.includes(LOCAL_ADDENDUM));
});

test("on a card with no image model the lead's own system prompt says so", () => {
  const ledger = openLedger(":memory:");
  const make = (imageAvailable: boolean) =>
    createAgentFactory({
      config: config({ mode: "routed" }),
      gpu: {} as GpuClient,
      ledger,
      harness: "HARNESS",
      output: () => ({}),
      imageAvailable,
    });
  const without = make(false).agent("lead", 0).system;
  assert.ok(without.includes(IMAGE_UNAVAILABLE));
  // Before the local addendum, which stays last.
  assert.ok(without.endsWith(LOCAL_ADDENDUM));
  assert.ok(!make(true).agent("lead", 0).system.includes(IMAGE_UNAVAILABLE));
});

test("a Claude-placed agent gets no local addendum", () => {
  const { factory } = setup({ mode: "routed", anthropic: ANTHROPIC });
  const lead = factory.agent("lead", 0);
  assert.equal(lead.placement, "claude");
  assert.ok(!lead.system.includes(LOCAL_ADDENDUM));
});

test("site can publish", () => {
  const { factory } = setup();
  const site = factory.agent("site", 1);

  assert.deepEqual(names(site.deps), [
    "gpu_generate_image",
    "gpu_status",
    "publish",
  ]);
});

test("OHH_MAX_TURNS overrides the agent at the root only", () => {
  withEnv("OHH_MAX_TURNS", "7", () => {
    const { factory } = setup();
    // The operator's override is for the run they started, not for every
    // agent it happens to dispatch.
    assert.equal(factory.agent("lead", 0).deps.maxTurns, 7);
    assert.equal(factory.agent("site", 1).deps.maxTurns, 20);
  });
  withEnv("OHH_MAX_TURNS", undefined, () => {
    const { factory } = setup();
    assert.equal(factory.agent("lead", 0).deps.maxTurns, 10);
  });
});

test("an unknown agent fails as an AgentError", () => {
  const { factory } = setup();
  assert.throws(() => factory.agent("nobody", 0), { name: "AgentError" });
});

test("a resume from a fresh process builds the child's own agent, at its own depth", () => {
  const { ledger, factory } = setup();
  const root = ledger.createRun({
    agent: "lead",
    goal: "ship",
    parentId: null,
    budgetUsd: 1.5,
  });
  const childId = ledger.createRun({
    agent: "site",
    goal: "render",
    parentId: root,
    budgetUsd: 1,
  });
  const child = ledger.getRun(childId)!;

  assert.equal(depthOf(ledger, ledger.getRun(root)!), 0);
  assert.equal(depthOf(ledger, child), 1);

  // Whatever `--agent` defaulted to plays no part: the Run names its agent.
  const deps = runDeps(factory, ledger, child);
  assert.equal(deps, factory.agent("site", 1).deps);
  assert.deepEqual(names(deps), [
    "gpu_generate_image",
    "gpu_status",
    "publish",
  ]);
  assert.deepEqual(names(runDeps(factory, ledger, ledger.getRun(root)!)), [
    "dispatch",
    "gpu_status",
  ]);
});

test("dispatching from lead spawns the child one level down", async () => {
  const created: Array<{ depth: number; agent: string; allocationUsd: number }> = [];
  const ledger = openLedger(":memory:");
  const factory = createAgentFactory({
    config: config(),
    gpu: {} as GpuClient,
    ledger,
    harness: "HARNESS",
    output: () => ({}),
    onChildCreated: (depth, info) =>
      created.push({ depth, agent: info.agent, allocationUsd: info.allocationUsd }),
  });
  const lead = factory.agent("lead", 0);
  const parent = ledger.createRun({
    agent: "lead",
    goal: "ship",
    parentId: null,
    budgetUsd: 1.5,
  });

  // `scribe` cannot dispatch and has no tools, so the scripted run's tool
  // requests fail as results and the child still finishes.
  const dispatch = lead.deps.registry.get("dispatch")!;
  await dispatch.handler(
    { agent: "scribe", goal: "write", budget_usd: 0.1 },
    { runId: parent },
  );

  assert.deepEqual(created, [{ depth: 1, agent: "scribe", allocationUsd: 0.1 }]);
  const [child] = ledger.listChildren(parent);
  assert.equal(child?.agent, "scribe");
});

test("lead's dispatch tool names the agents it can reach, and only those", () => {
  const { factory } = setup();
  const dispatch = factory.agent("lead", 0).deps.registry.get("dispatch")!;
  const names = (
    dispatch.schema as { properties: { agent: { enum?: string[] } } }
  ).properties.agent.enum;

  assert.ok(names, "the agent property has an enum");
  assert.ok(names.includes("site"));
  assert.ok(names.includes("scribe"));
  assert.ok(!names.includes("lead"), "lead can itself dispatch, so it is not offered");
  assert.match(dispatch.description, /Available agents: .*site —/);
});

const ANTHROPIC = {
  apiKey: "sk-test",
  model: "claude-opus-5-5",
  maxTokens: 1024,
  cacheTtl: "5m" as const,
};

test("a claude agent without a key runs on the ollama provider with placement local-no-key", () => {
  const { factory } = setup({ mode: "routed" });
  const lead = factory.agent("lead", 0);

  assert.equal(lead.placement, "local-no-key");
  assert.equal(lead.deps.placement, "local-no-key");
  assert.equal(lead.deps.provider.name, "ollama");
});

test("a claude agent with a key gets a failover provider whose name is anthropic", () => {
  const { factory } = setup({ mode: "routed", anthropic: ANTHROPIC });
  const lead = factory.agent("lead", 0);

  assert.equal(lead.placement, "claude");
  assert.equal(lead.deps.placement, "claude");
  assert.equal(lead.deps.provider.name, "anthropic");
  // The agent file's model, not the configured default.
  assert.equal(lead.deps.provider.model, "claude-sonnet-5-5");
});

test("an agent with no model line runs locally on the default model", () => {
  const { factory } = setup({ mode: "routed", anthropic: ANTHROPIC });
  const site = factory.agent("site", 1);

  assert.equal(site.placement, "local");
  assert.equal(site.deps.provider.name, "ollama");
  // No override: the GPU service picks the registry's role = "code" model.
  assert.equal(site.deps.provider.model, "ollama:role=code");
});

test("an unknown model fails agent construction naming the file", () => {
  const dir = mkdtempSync(join(tmpdir(), "ohh-agents-"));
  writeFileSync(
    join(dir, "odd.toml"),
    '[agent]\nname = "odd"\nrole = "r"\nmodel = "gpt-9"\ntools = []\n',
    "utf8",
  );
  const { factory } = setup({ mode: "routed", agentsDir: dir });

  assert.throws(
    () => factory.agent("odd", 0),
    (error: unknown) => {
      assert.ok(error instanceof RouteError);
      assert.ok(error.message.includes(join(dir, "odd.toml")), error.message);
      assert.match(error.message, /model "gpt-9" is not in models\.toml/);
      return true;
    },
  );
});

test("two runs of the same claude agent get separate providers in routed mode", () => {
  const { factory } = setup({ mode: "routed", anthropic: ANTHROPIC });

  const first = factory.agent("lead", 0);
  const second = factory.agent("lead", 0);
  // Fallback is per run: one run reaching the spending wall must not move
  // the next run of the same agent onto the local model.
  assert.notEqual(second.deps.provider, first.deps.provider);
});

function routedSetup(withKey: boolean) {
  const ledger = openFreshLedger(":memory:");
  const factory = createAgentFactory({
    config: config({ mode: "routed", ...(withKey ? { anthropic: ANTHROPIC } : {}) }),
    gpu: {} as GpuClient,
    ledger,
    harness: "HARNESS",
    output: () => ({}),
  });
  return { ledger, factory };
}

test("a local-no-key run resumed after a key is added stays local", () => {
  const { ledger, factory } = routedSetup(true);
  const id = ledger.createRun({
    agent: "lead", goal: "g", parentId: null, budgetUsd: 1, placement: "local-no-key",
  });

  const deps = runDeps(factory, ledger, ledger.getRun(id)!);

  // No surprise spend: the run began local and finishes local.
  assert.equal(deps.provider.name, "ollama");
  assert.equal(deps.placement, "local-no-key");
  assert.equal(ledger.getRun(id)?.placement, "local-no-key");
  // A fresh run of the same agent does go to Claude now.
  assert.equal(factory.agent("lead", 0).deps.provider.name, "anthropic");
});

test("a claude run resumed without a key runs locally and is recorded as local-no-key", () => {
  const { ledger, factory } = routedSetup(false);
  const id = ledger.createRun({
    agent: "lead", goal: "g", parentId: null, budgetUsd: 1, placement: "claude",
  });

  const deps = runDeps(factory, ledger, ledger.getRun(id)!);

  assert.equal(deps.provider.name, "ollama");
  assert.equal(deps.placement, "local-no-key");
  const run = ledger.getRun(id)!;
  assert.equal(run.placement, "local-no-key");
  assert.equal(ranAs(ledger, run), "local-no-key");
});

test("a claude run resumed with a key stays on claude", () => {
  const { ledger, factory } = routedSetup(true);
  const id = ledger.createRun({
    agent: "lead", goal: "g", parentId: null, budgetUsd: 1, placement: "claude",
  });

  const deps = runDeps(factory, ledger, ledger.getRun(id)!);

  assert.equal(deps.provider.name, "anthropic");
  assert.equal(ledger.getRun(id)?.placement, "claude");
});

test("imageAvailable: false hides site from the dispatch tool the factory builds", async () => {
  const ledger = openLedger(":memory:");
  const factory = createAgentFactory({
    config: config(),
    gpu: {} as GpuClient,
    ledger,
    harness: "HARNESS",
    output: () => ({}),
    imageAvailable: false,
  });
  const dispatch = factory.agent("lead", 0).deps.registry.get("dispatch")!;
  const names = (dispatch.schema as { properties: { agent: { enum?: string[] } } })
    .properties.agent.enum;
  assert.ok(names && !names.includes("site") && names.includes("scribe"));
  assert.match(dispatch.description, /image model doesn't fit/);

  const parent = ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1.5 });
  const result = (await dispatch.handler(
    { agent: "site", goal: "x", budget_usd: 0.5 },
    { runId: parent },
  )) as { error: string };
  assert.match(result.error, /image model doesn't fit/);
  assert.equal(ledger.listChildren(parent).length, 0);
});
