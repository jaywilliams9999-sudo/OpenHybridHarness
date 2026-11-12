/**
 * Per-agent routing: the agent file's `model` line decides where its runs go.
 * A registry name runs locally, a `claude-*` name runs on Claude when a key is
 * configured and locally (marked) when not, and anything else is an error that
 * names the file.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { resumeRoute, routeAgent, RouteError } from "../src/gateway/route.ts";

const localModels = new Set(["llama3.1:8b", "qwen2.5-coder:7b"]);
const FILE = "agents/site.toml";

test("no model line runs locally on the default model", () => {
  assert.deepEqual(routeAgent(FILE, undefined, { localModels, hasKey: true }), {
    placement: "local",
  });
});

test("a registry model runs locally on that model", () => {
  assert.deepEqual(
    routeAgent(FILE, "qwen2.5-coder:7b", { localModels, hasKey: true }),
    { placement: "local", model: "qwen2.5-coder:7b" },
  );
});

test("a claude model with a key runs on claude", () => {
  assert.deepEqual(
    routeAgent(FILE, "claude-sonnet-5-5", { localModels, hasKey: true }),
    { placement: "claude", model: "claude-sonnet-5-5" },
  );
});

test("a claude model without a key runs locally, marked local-no-key", () => {
  assert.deepEqual(
    routeAgent(FILE, "claude-sonnet-5-5", { localModels, hasKey: false }),
    { placement: "local-no-key" },
  );
});

test("an unknown model names the agent file", () => {
  assert.throws(
    () => routeAgent(FILE, "gpt-9", { localModels, hasKey: true }),
    (error: unknown) => {
      assert.ok(error instanceof RouteError);
      assert.equal(
        error.message,
        `${FILE}: model "gpt-9" is not in models.toml and is not a claude-* model.`,
      );
      return true;
    },
  );
});

// A resume keeps the placement its run row records; the environment today only
// decides whether a Claude placement can still be honoured.
test("a resume keeps a local-no-key run local even once a key exists", () => {
  const today = { placement: "claude", model: "claude-sonnet-5-5" } as const;
  assert.deepEqual(resumeRoute("local-no-key", today, true), { placement: "local-no-key" });
});

test("a resume of a claude run without a key runs as local-no-key", () => {
  const today = { placement: "local-no-key" } as const;
  assert.deepEqual(resumeRoute("claude", today, false), { placement: "local-no-key" });
});

test("a resume of a claude run with a key stays on claude with the agent's model", () => {
  const today = { placement: "claude", model: "claude-sonnet-5-5" } as const;
  assert.deepEqual(resumeRoute("claude", today, true), today);
});

test("a resume of a local run stays local", () => {
  assert.deepEqual(
    resumeRoute("local", { placement: "local", model: "qwen2.5-coder:7b" }, true),
    { placement: "local", model: "qwen2.5-coder:7b" },
  );
  // The agent file moved to Claude since: the run started local, so it stays local.
  assert.deepEqual(
    resumeRoute("local", { placement: "claude", model: "claude-sonnet-5-5" }, true),
    { placement: "local" },
  );
});
