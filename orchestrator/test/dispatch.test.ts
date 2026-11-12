/**
 * `dispatch`: a parent Run starts a child Run for another agent.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { openLedger, type Ledger } from "../src/ledger/ledger.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";
import { createScriptedProvider } from "../src/gateway/scripted.ts";
import { ChildBlocked, runLoop } from "../src/loop/loop.ts";
import { AgentError, type AgentDefinition } from "../src/agents/definition.ts";
import { RouteError } from "../src/gateway/route.ts";
import {
  createDispatchTool,
  type ChildAgent,
} from "../src/tools/dispatch.ts";
import type { Provider, TurnEvent, TurnRequest } from "../src/gateway/types.ts";
import { join } from "node:path";
import { dispatchableAgents } from "../src/agents/factory.ts";

const TEXT: TurnEvent[] = [
  { kind: "text", text: "child finished" },
  { kind: "end", reason: "end_turn" },
];

function agent(extra: Partial<AgentDefinition> = {}): AgentDefinition {
  return {
    name: "site",
    role: "builds sites",
    tools: [],
    limits: {},
    ...extra,
  };
}

function child(
  ledger: Ledger,
  options: {
    agent?: AgentDefinition;
    turns?: TurnEvent[][];
    tools?: ToolDefinition[];
    provider?: Provider;
  } = {},
): ChildAgent {
  const provider =
    options.provider ?? {
      ...createScriptedProvider({ turns: options.turns ?? [TEXT] }),
      model: "claude-opus-5-5",
    };
  return {
    deps: {
      provider,
      registry: createRegistry(options.tools ?? []),
      ledger,
      maxTurns: 5,
    },
    agent: options.agent ?? agent(),
    system: "you are the site agent",
    placement: "local",
  };
}

function setup(parentBudget: number, make: (ledger: Ledger) => ChildAgent) {
  const ledger = openLedger(":memory:");
  const parent = ledger.createRun({
    agent: "lead",
    goal: "ship",
    parentId: null,
    budgetUsd: parentBudget,
  });
  const tool = createDispatchTool({ ledger, spawn: () => make(ledger) });
  const run = (args: Record<string, unknown>) =>
    tool.handler(args, { runId: parent }) as Promise<Record<string, unknown>>;
  return { ledger, parent, tool, run };
}

test("dispatch runs the child with a parent id and returns its report", async () => {
  const { ledger, parent, run } = setup(1.5, (l) => child(l));
  const result = await run({ agent: "site", goal: "build", budget_usd: 0.5 });

  assert.equal(result.status, "done");
  assert.equal(result.summary, "child finished");
  const [childId] = ledger.listChildren(parent).map((c) => c.id);
  assert.ok(childId);
  assert.equal(ledger.getRun(childId)?.parent_id, parent);
  assert.deepEqual(
    ledger.listMessages(childId).map((m) => m.kind),
    ["dispatch", "report"],
  );
  ledger.close();
});

test("the allocation keeps a reserve for the parent", async () => {
  const { ledger, parent, run } = setup(1.5, (l) => child(l));
  const clamped = await run({ agent: "site", goal: "a", budget_usd: 5 });
  const small = await run({ agent: "site", goal: "b", budget_usd: 0.4 });

  const [first, second] = ledger.listChildren(parent);
  assert.equal(ledger.getRun(first!.id)?.budget_usd, 1.2);
  assert.equal(clamped.note, "budget clamped from $5.00 to $1.20");
  assert.equal(ledger.getRun(second!.id)?.budget_usd, 0.4);
  assert.equal("note" in small, false);
  ledger.close();
});

test("the child agent's cap wins when lower", async () => {
  const { ledger, parent, run } = setup(1.5, (l) =>
    child(l, { agent: agent({ limits: { maxUsd: 0.5 } }) }),
  );
  await run({ agent: "site", goal: "a", budget_usd: 1 });
  const [only] = ledger.listChildren(parent);
  assert.equal(ledger.getRun(only!.id)?.budget_usd, 0.5);
  ledger.close();
});

test("with nothing left after the reserve, no child is created", async () => {
  const { ledger, parent, run } = setup(0.25, (l) => child(l));
  const result = await run({ agent: "site", goal: "a", budget_usd: 1 });
  assert.equal(typeof result.error, "string");
  assert.equal(ledger.listChildren(parent).length, 0);
  ledger.close();
});

test("an agent that can itself dispatch is refused", async () => {
  const { ledger, parent, run } = setup(1.5, (l) =>
    child(l, { agent: agent({ tools: ["dispatch"] }) }),
  );
  const result = await run({ agent: "site", goal: "a", budget_usd: 1 });
  assert.equal(typeof result.error, "string");
  assert.equal(ledger.listChildren(parent).length, 0);
  ledger.close();
});

test("an unknown agent, or a bad budget, is an error result", async () => {
  const ledger = openLedger(":memory:");
  const parent = ledger.createRun({
    agent: "lead",
    goal: "ship",
    parentId: null,
    budgetUsd: 1.5,
  });
  const tool = createDispatchTool({
    ledger,
    spawn: (name) => {
      if (name === "ghost") throw new AgentError("no agent named ghost");
      return child(ledger);
    },
  });
  const ask = (args: Record<string, unknown>) =>
    tool.handler(args, { runId: parent }) as Promise<{ error?: string }>;

  assert.equal((await ask({ agent: "ghost", goal: "a", budget_usd: 1 })).error, "no agent named ghost");
  for (const bad of [0, -1, "1", Number.NaN]) {
    const result = await ask({ agent: "site", goal: "a", budget_usd: bad });
    assert.equal(typeof result.error, "string");
  }
  assert.equal(ledger.listChildren(parent).length, 0);
  ledger.close();
});

test("a child blocked on approval throws ChildBlocked", async () => {
  let ran = 0;
  const gated: ToolDefinition = {
    name: "publish",
    description: "publish",
    schema: { type: "object" },
    tier: "gated",
    handler: async () => {
      ran++;
      return {};
    },
  };
  const { ledger, parent, run } = setup(1.5, (l) =>
    child(l, {
      tools: [gated],
      turns: [
        [
          { kind: "tool_request", id: "p1", name: "publish", input: {} },
          { kind: "end", reason: "tool_use" },
        ],
      ],
    }),
  );

  await assert.rejects(
    run({ agent: "site", goal: "a", budget_usd: 1 }),
    (error: unknown) => {
      assert.ok(error instanceof ChildBlocked);
      const [childId] = ledger.listChildren(parent).map((c) => c.id);
      assert.equal(error.childRunId, childId);
      return true;
    },
  );
  assert.equal(ran, 0);
  ledger.close();
});

test("a child that throws is marked failed and reported with the error as a gap", async () => {
  for (const provider of [
    {
      name: "boom",
      model: "claude-opus-5-5",
      chat(): AsyncIterable<TurnEvent> {
        throw new Error("gpu down");
      },
    },
    {
      name: "boom",
      model: "claude-opus-5-5",
      chat(): AsyncIterable<TurnEvent> {
        return {
          // eslint-disable-next-line require-yield
          async *[Symbol.asyncIterator]() {
            throw new Error("gpu down");
          },
        };
      },
    },
  ] as Provider[]) {
    const { ledger, parent, run } = setup(1.5, (l) => child(l, { provider }));
    const result = await run({ agent: "site", goal: "a", budget_usd: 1 });

    assert.equal(result.status, "failed");
    assert.deepEqual(result.gaps, ["gpu down"]);
    const [childId] = ledger.listChildren(parent).map((c) => c.id);
    assert.equal(ledger.getRun(childId!)?.status, "failed");
    assert.deepEqual(
      ledger.listMessages(childId!).map((m) => m.kind),
      ["dispatch", "report"],
    );
    ledger.close();
  }
});

test("a failed child comes back to the model as an error result, as it does on resume", async () => {
  const boom: Provider = {
    name: "boom",
    model: "claude-opus-5-5",
    chat(): AsyncIterable<TurnEvent> {
      throw new Error("gpu down");
    },
  };
  const outOfTurns = [
    [
      { kind: "tool_request", id: "l1", name: "look", input: {} },
      { kind: "end", reason: "tool_use" },
    ],
  ] as TurnEvent[][];
  const look: ToolDefinition = {
    name: "look",
    description: "look",
    schema: { type: "object" },
    tier: "auto",
    handler: async () => ({}),
  };
  const cases: Array<[string, (l: Ledger) => ChildAgent, string]> = [
    ["threw", (l) => child(l, { provider: boom }), "failed"],
    [
      "ran out of turns",
      (l) => {
        const agent = child(l, { turns: outOfTurns, tools: [look] });
        return { ...agent, deps: { ...agent.deps, maxTurns: 1 } };
      },
      "failed",
    ],
    ["finished", (l) => child(l), "done"],
  ];

  for (const [label, make, status] of cases) {
    const ledger = openLedger(":memory:");
    const requests: TurnRequest[] = [];
    const tool = createDispatchTool({ ledger, spawn: () => make(ledger) });
    const result = await runLoop(
      {
        provider: {
          ...createScriptedProvider({
            turns: [
              [
                { kind: "tool_request", id: "d1", name: "dispatch", input: { agent: "site", goal: "a", budget_usd: 0.5 } },
                { kind: "end", reason: "tool_use" },
              ],
              TEXT,
            ],
            onRequest: (r) => requests.push(r),
          }),
          model: "claude-opus-5-5",
        },
        registry: createRegistry([tool]),
        ledger,
        maxTurns: 5,
      },
      { agent: "lead", goal: "ship", budgetUsd: 1.5 },
    );
    assert.equal(result.status, "done", label);
    const transcript = requests.at(-1)?.transcript ?? [];
    const last = transcript.at(-1);
    assert.equal(last?.role, "tool_results", label);
    const answer = last?.role === "tool_results" ? last.results[0] : undefined;
    assert.equal((answer?.result as { status?: string }).status, status, label);
    assert.equal(answer?.isError, status === "failed" ? true : undefined, label);
    ledger.close();
  }
});

test("agents in deps are listed in the description and pinned as an enum", () => {
  const ledger = openLedger(":memory:");
  const long = "x".repeat(300);
  const tool = createDispatchTool({
    ledger,
    spawn: () => child(ledger),
    agents: [
      { name: "site", role: "builds sites" },
      { name: "scribe", role: long },
    ],
  });
  const props = (tool.schema as { properties: Record<string, { enum?: string[] }> })
    .properties;
  assert.deepEqual(props["agent"]?.enum, ["site", "scribe"]);
  assert.match(tool.description, /Available agents: site — builds sites; scribe — x+…/);
  assert.ok(!tool.description.includes(long));
});

test("without agents there is no enum and no list", () => {
  const ledger = openLedger(":memory:");
  const tool = createDispatchTool({ ledger, spawn: () => child(ledger) });
  const props = (tool.schema as { properties: Record<string, { enum?: string[] }> })
    .properties;
  assert.equal(props["agent"]?.enum, undefined);
  assert.doesNotMatch(tool.description, /Available agents/);
});

test("a bad model line in the child's file is an error result, not a crash", async () => {
  const ledger = openLedger(":memory:");
  const parent = ledger.createRun({ agent: "lead", goal: "ship", parentId: null, budgetUsd: 1.5 });
  const message = 'agents/odd.toml: model "gpt-9" is not in models.toml and is not a claude-* model.';
  const tool = createDispatchTool({
    ledger,
    spawn: () => {
      throw new RouteError(message);
    },
  });

  const result = (await tool.handler(
    { agent: "odd", goal: "a", budget_usd: 0.5 },
    { runId: parent },
  )) as { error?: string };

  assert.equal(result.error, message);
  assert.equal(ledger.listChildren(parent).length, 0);
  ledger.close();
});

const AGENTS_DIR = join(import.meta.dirname, "..", "agents");

test("dispatch omits image agents and says why when images are unavailable", () => {
  const ledger = openLedger(":memory:");
  const agents = dispatchableAgents(AGENTS_DIR, false);
  const tool = createDispatchTool({
    ledger,
    spawn: () => child(ledger),
    agents,
    imageAvailable: false,
  });

  assert.ok(agents.length > 0);
  assert.ok(!agents.some((a) => a.name === "site"));
  assert.ok(!tool.description.includes("site —"));
  assert.ok(
    tool.description.endsWith(
      " Image generation is unavailable on this card: the image model doesn't fit.",
    ),
  );
  ledger.close();
});

test("dispatch lists every agent when images are available", () => {
  const ledger = openLedger(":memory:");
  const agents = dispatchableAgents(AGENTS_DIR);
  const tool = createDispatchTool({ ledger, spawn: () => child(ledger), agents });

  assert.ok(agents.some((a) => a.name === "site"));
  assert.ok(tool.description.includes("site —"));
  assert.ok(!tool.description.includes("Image generation is unavailable"));
  ledger.close();
});

test("dispatch rejects an unlisted agent before spawning anything", async () => {
  const ledger = openLedger(":memory:");
  const parent = ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1.5 });
  let spawned = 0;
  const tool = createDispatchTool({
    ledger,
    spawn: () => {
      spawned += 1;
      return child(ledger);
    },
    agents: [{ name: "scribe", role: "writes" }],
  });
  const result = (await tool.handler(
    { agent: "ghost", goal: "x", budget_usd: 0.5 },
    { runId: parent },
  )) as { error: string };

  assert.match(result.error, /"ghost" isn't available here\. Available agents: scribe\./);
  assert.equal(spawned, 0);
  assert.equal(ledger.listChildren(parent).length, 0);
  ledger.close();
});

test("dispatch explains a hidden image agent on a small card", async () => {
  const ledger = openLedger(":memory:");
  const parent = ledger.createRun({ agent: "lead", goal: "g", parentId: null, budgetUsd: 1.5 });
  const tool = createDispatchTool({
    ledger,
    spawn: () => child(ledger),
    agents: [{ name: "scribe", role: "writes" }],
    imageAvailable: false,
    hiddenImageAgents: ["site"],
  });
  const result = (await tool.handler(
    { agent: "site", goal: "x", budget_usd: 0.5 },
    { runId: parent },
  )) as { error: string };

  assert.match(result.error, /image model doesn't fit/);
  assert.equal(ledger.listChildren(parent).length, 0);
  ledger.close();
});
