/**
 * The assembler that turns a turn's streamed text fragments into terminal
 * writes: one indent per line, no line break per fragment.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { createTextStream } from "../src/output/stream-text.ts";

test("fragments of one line are written without breaks, indented once", () => {
  const s = createTextStream("    ");
  const out = [s.push("Hel"), s.push("lo "), s.push("there"), s.end()].join("");
  assert.equal(out, "    Hello there\n");
});

test("an embedded newline continues the indent on the next line", () => {
  const s = createTextStream("  ");
  const out = [s.push("one\ntw"), s.push("o\n\nthree"), s.end()].join("");
  assert.equal(out, "  one\n  two\n\n  three\n");
});

test("end writes nothing when the line is already closed or nothing was said", () => {
  const s = createTextStream("  ");
  assert.equal(s.end(), "");
  assert.equal(s.push("done\n"), "  done\n");
  assert.equal(s.end(), "");
});

test("a new turn starts with a fresh indent", () => {
  const s = createTextStream("> ");
  const out = [s.push("a"), s.end(), s.push("b"), s.end()].join("");
  assert.equal(out, "> a\n> b\n");
});
