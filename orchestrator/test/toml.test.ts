import { test } from "node:test";
import assert from "node:assert/strict";

import { parseToml, TomlSubsetError } from "../src/config/toml.ts";

// The reader covers the subset models.toml and agents/*.toml actually use and
// refuses everything else by name. Guessing is worse than refusing here: the
// registry holds the VRAM figures the GPU service schedules against, so a parser that
// improvised would hand back a plausible wrong number instead of an error.

test("parses a table of scalars", () => {
  const doc = parseToml(`
[gpu_service]
host = "gpu.example.test"
port = 6666
vram_headroom_mb = 768
enabled = true
`);
  assert.deepEqual(doc, {
    gpu_service: {
      host: "gpu.example.test",
      port: 6666,
      vram_headroom_mb: 768,
      enabled: true,
    },
  });
});

test("parses a quoted table key containing dots and colons", () => {
  const doc = parseToml(`
[models."qwen2.5-coder:7b"]
backend = "ollama"
vram_mb = 4870
`);
  const models = doc["models"] as Record<string, unknown>;
  assert.deepEqual(models["qwen2.5-coder:7b"], {
    backend: "ollama",
    vram_mb: 4870,
  });
});

test("parses a string array", () => {
  const doc = parseToml(`
[agent]
tools = ["fs.read", "fs.write", "gpu_generate_image"]
`);
  const agent = doc["agent"] as Record<string, unknown>;
  assert.deepEqual(agent["tools"], [
    "fs.read",
    "fs.write",
    "gpu_generate_image",
  ]);
});

test("ignores comments, trailing comments and blank lines", () => {
  const doc = parseToml(`
# leading comment

[gpu_service]
port = 6666   # trailing comment

# another
host = "h"
`);
  assert.deepEqual(doc, { gpu_service: { port: 6666, host: "h" } });
});

test("accepts a negative integer", () => {
  const doc = parseToml(`[a]\nn = -12\n`);
  assert.deepEqual(doc, { a: { n: -12 } });
});

test("a key before any table header is rejected", () => {
  assert.throws(() => parseToml(`orphan = 1\n`), TomlSubsetError);
});

for (const [label, body] of [
  ["inline table", `[a]\nb = { c = 1 }\n`],
  ["array of tables", `[[a]]\nb = 1\n`],
  ["multi-line string", `[a]\nb = """\nhello\n"""\n`],
  ["float", `[a]\nb = 1.5\n`],
  ["datetime", `[a]\nb = 1979-05-27T07:32:00Z\n`],
  ["nested array", `[a]\nb = [["x"]]\n`],
  ["mixed-type array", `[a]\nb = ["x", 1]\n`],
  ["dotted bare key", `[a]\nb.c = 1\n`],
  ["bare unquoted value", `[a]\nb = bare\n`],
] as const) {
  test(`rejects ${label}`, () => {
    assert.throws(
      () => parseToml(body),
      (error: unknown) => {
        assert.ok(error instanceof TomlSubsetError, `${label}: wrong type`);
        // The operator has to be able to find it: the message names the line.
        assert.match(error.message, /line \d+/, `${label}: no line number`);
        return true;
      },
    );
  });
}

test("the error names the line the problem is on, not the first line", () => {
  let caught: TomlSubsetError | undefined;
  try {
    parseToml(`[a]\nok = 1\nbad = 1.5\n`);
  } catch (error) {
    caught = error as TomlSubsetError;
  }
  assert.ok(caught, "expected a throw");
  assert.equal(caught.line, 3);
  assert.match(caught.message, /bad/);
});

test("a duplicate key is rejected rather than silently overwritten", () => {
  assert.throws(() => parseToml(`[a]\nb = 1\nb = 2\n`), TomlSubsetError);
});
