import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ensureGpuToken, readDotEnv } from "../src/env.ts";

const tmp = (): string => join(mkdtempSync(join(tmpdir(), "ohh-env-")), ".env");

test("reads a CRLF .env with a UTF-8 BOM", () => {
  const p = tmp();
  writeFileSync(p, "﻿OHH_GPU_TOKEN=abc\r\n# c\r\n\r\nOHH_ANTHROPIC_API_KEY=\"sk-x\"\r\n");
  assert.deepEqual(readDotEnv(p), { OHH_GPU_TOKEN: "abc", OHH_ANTHROPIC_API_KEY: "sk-x" });
});

test("a missing file reads as empty", () => {
  assert.deepEqual(readDotEnv(tmp()), {});
});

test("ensureGpuToken keeps an existing token", () => {
  const p = tmp();
  writeFileSync(p, "OHH_GPU_TOKEN=abc\n");
  assert.equal(ensureGpuToken(p), "abc");
  assert.equal(readFileSync(p, "utf8"), "OHH_GPU_TOKEN=abc\n");
});

test("ensureGpuToken creates the file and a 64-hex-char token", () => {
  const p = tmp();
  const token = ensureGpuToken(p);
  assert.match(token, /^[0-9a-f]{64}$/);
  assert.match(readFileSync(p, "utf8"), new RegExp(`OHH_GPU_TOKEN=${token}`));
});

test("ensureGpuToken appends on its own line to a file without a trailing newline", () => {
  const p = tmp();
  writeFileSync(p, "OTHER=1");
  const token = ensureGpuToken(p);
  assert.deepEqual(readDotEnv(p), { OTHER: "1", OHH_GPU_TOKEN: token });
});

test("ensureGpuToken creates the file owner-only (POSIX)", { skip: process.platform === "win32" }, () => {
  const p = tmp();
  ensureGpuToken(p);
  assert.equal(statSync(p).mode & 0o777, 0o600);
});

test("ensureGpuToken tightens an existing world-readable file (POSIX)", { skip: process.platform === "win32" }, () => {
  const p = tmp();
  writeFileSync(p, "OHH_GPU_TOKEN=abc\n", { mode: 0o644 });
  ensureGpuToken(p);
  assert.equal(statSync(p).mode & 0o777, 0o600);
});
