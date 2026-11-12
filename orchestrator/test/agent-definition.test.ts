/**
 * Agents — the agents themselves, declared rather than coded.
 *
 * What these tests mostly protect is the *capability boundary*: an agent's
 * tool list decides what an autonomous agent can reach, so "the list is
 * wrong" has to be loud and "the list is short" has to be real.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  buildSystem,
  loadAgent,
  restrictRegistry,
  AgentError,
} from "../src/agents/definition.ts";
import { createRegistry, type ToolDefinition } from "../src/tools/registry.ts";

function dirWith(name: string, body: string): string {
  const dir = mkdtempSync(join(tmpdir(), "ohh-agent-"));
  writeFileSync(join(dir, `${name}.toml`), body, "utf8");
  return dir;
}

const GOOD = `
[agent]
name = "site"
role = "Produce hero assets."
workdir = "./work/site"
tools = ["gpu_status", "gpu_generate_image"]

[limits]
max_usd_cents = 250
max_turns = 12
`;

function tool(name: string): ToolDefinition {
  return {
    name,
    description: name,
    schema: { type: "object" },
    tier: "auto",
    handler: async () => ({ ran: name }),
  };
}

const REGISTRY = () =>
  createRegistry([
    tool("gpu_status"),
    tool("gpu_generate_image"),
    tool("shell_git"),
  ]);

// -- loading ------------------------------------------------------------

test("a well-formed agent loads with its limits in dollars", () => {
  const agent = loadAgent(dirWith("site", GOOD), "site");

  assert.equal(agent.name, "site");
  assert.equal(agent.role, "Produce hero assets.");
  assert.equal(agent.workdir, "./work/site");
  assert.deepEqual(agent.tools, ["gpu_status", "gpu_generate_image"]);
  // Authored as whole cents; consumed as dollars.
  assert.equal(agent.limits.maxUsd, 2.5);
  assert.equal(agent.limits.maxTurns, 12);
  assert.equal(agent.model, undefined);
});

test("an absent [limits] table means the config defaults stand", () => {
  const dir = dirWith(
    "bare",
    `[agent]\nname = "bare"\nrole = "r"\ntools = []\n`,
  );
  const agent = loadAgent(dir, "bare");

  assert.deepEqual(agent.limits, {});
  assert.deepEqual(agent.tools, []);
});

test("a missing agent names the path it looked for", () => {
  const dir = mkdtempSync(join(tmpdir(), "ohh-agent-"));
  assert.throws(() => loadAgent(dir, "nope"), (error: Error) => {
    assert.ok(error instanceof AgentError);
    assert.match(error.message, /nope\.toml/);
    return true;
  });
});

test("a filename that disagrees with the declared name is refused", () => {
  // The Ledger keys Runs by this name, so the disagreement would file a run
  // under an agent nobody asked for.
  const dir = dirWith("alpha", `[agent]\nname = "beta"\nrole = "r"\ntools = []\n`);
  assert.throws(() => loadAgent(dir, "alpha"), /name is "beta"/);
});

test("an agent name cannot escape the agents directory", () => {
  const dir = dirWith("site", GOOD);
  // The name becomes a filename, so it is validated before it is joined.
  assert.throws(() => loadAgent(dir, "../../../etc/passwd"), AgentError);
  assert.throws(() => loadAgent(dir, "a/b"), AgentError);
});

test("money must be whole cents, because the TOML subset refuses floats", () => {
  const dir = dirWith(
    "site",
    `[agent]\nname = "site"\nrole = "r"\ntools = []\n\n[limits]\nmax_usd_cents = 2.50\n`,
  );
  assert.throws(() => loadAgent(dir, "site"), AgentError);
});

test("a missing role is refused rather than defaulted", () => {
  const dir = dirWith("site", `[agent]\nname = "site"\ntools = []\n`);
  assert.throws(() => loadAgent(dir, "site"), /role/);
});

test("tools listed twice is an error, not a silently deduplicated list", () => {
  const dir = dirWith(
    "site",
    `[agent]\nname = "site"\nrole = "r"\ntools = ["a", "a"]\n`,
  );
  assert.throws(() => loadAgent(dir, "site"), /twice/);
});

// -- the capability boundary -------------------------------------------

test("the registry is narrowed to exactly what the agent declares", () => {
  const agent = loadAgent(dirWith("site", GOOD), "site");
  const restricted = restrictRegistry(REGISTRY(), agent);

  assert.deepEqual(
    restricted.list().map((t) => t.name).sort(),
    ["gpu_generate_image", "gpu_status"],
  );
  // Not merely hidden from the listing: unreachable by name too, so a model
  // that guesses the name of an undeclared tool still cannot call it.
  assert.equal(restricted.get("shell_git"), undefined);
  assert.ok(restricted.get("gpu_status"));
  assert.equal(
    restricted.forProvider().some((t) => t.name === "shell_git"),
    false,
    "an undeclared tool was advertised to the model",
  );
});

test("an agent listing a tool that does not exist fails loudly", () => {
  const dir = dirWith(
    "typo",
    `[agent]\nname = "typo"\nrole = "r"\ntools = ["gpu_statu"]\n`,
  );
  const agent = loadAgent(dir, "typo");

  // Dropping it silently would leave the agent missing a capability its
  // author believed it had, and the symptom points nowhere near the typo.
  assert.throws(() => restrictRegistry(REGISTRY(), agent), (error: Error) => {
    assert.ok(error instanceof AgentError);
    assert.match(error.message, /gpu_statu/);
    assert.match(error.message, /Available:/);
    return true;
  });
});

test("an agent with no tools gets a registry with no tools", () => {
  const dir = dirWith("mute", `[agent]\nname = "mute"\nrole = "r"\ntools = []\n`);
  const restricted = restrictRegistry(REGISTRY(), loadAgent(dir, "mute"));

  assert.deepEqual(restricted.list(), []);
  assert.deepEqual(restricted.forProvider(), []);
  assert.equal(restricted.get("gpu_status"), undefined);
});

// -- the system prompt --------------------------------------------------

test("the system prompt is the harness plus this agent's role", () => {
  const agent = loadAgent(dirWith("site", GOOD), "site");
  const system = buildSystem("HARNESS TEXT", agent);

  assert.match(system, /HARNESS TEXT/);
  assert.match(system, /Produce hero assets\./);
});

// -- the shipped agents -------------------------------------------------

const SHIPPED = join(import.meta.dirname, "..", "agents");

test("the shipped lead agent loads: it dispatches and checks the card, nothing else", () => {
  const lead = loadAgent(SHIPPED, "lead");

  assert.deepEqual(lead.tools, ["dispatch", "gpu_status"]);
  assert.equal(lead.limits.maxUsd, 1.5);
  assert.equal(lead.limits.maxTurns, 10);
  assert.match(lead.role, /^You coordinate other agents\./);
  assert.match(lead.role, /Never claim work a report does not show\./);
  // Publish accepts only the calling Run's own artifacts, so a split
  // render-then-publish would render twice and pay twice.
  assert.match(
    lead.role,
    /An agent can publish only what it rendered itself, so send rendering and publishing to one child as one task\./,
  );
  // For the local model, which invented agent names. Generic on purpose: what
  // this card cannot do is said at run time, not in the agent file.
  assert.match(lead.role, /Dispatch only to an agent the `dispatch` tool lists\.$/);
  assert.doesNotMatch(lead.role, /image/i);
});

test("the shipped site agent loads and may publish", () => {
  const site = loadAgent(SHIPPED, "site");

  assert.deepEqual(site.tools, ["gpu_status", "gpu_generate_image", "publish"]);
  assert.match(site.role, /publish it with `publish`/);
});
