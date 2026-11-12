/** The repo-root `.env`: read it, and make sure it holds a GPU service token. */

import { randomBytes } from "node:crypto";
import { appendFileSync, chmodSync, existsSync, readFileSync } from "node:fs";

/** Parse a `.env` file. Tolerates a BOM, CRLF, `#` comments, blank lines and quotes. */
export function readDotEnv(path: string): Record<string, string> {
  if (!existsSync(path)) return {};
  const text = readFileSync(path, "utf8").replace(/^﻿/, "");
  const out: Record<string, string> = {};
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim().replace(/^export\s+/, "");
    let value = line.slice(eq + 1).trim();
    const quote = value[0];
    if (value.length >= 2 && (quote === '"' || quote === "'") && value.endsWith(quote)) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

/** Copy `.env` values into `env`, never overriding a variable already set. */
export function applyDotEnv(path: string, env: NodeJS.ProcessEnv = process.env): void {
  for (const [key, value] of Object.entries(readDotEnv(path))) {
    if (env[key] === undefined) env[key] = value;
  }
}

function ownerOnly(path: string): void {
  try {
    chmodSync(path, 0o600); // a no-op on Windows
  } catch {
    // Best effort: the file is still usable.
  }
}

/** The existing `OHH_GPU_TOKEN`, or a fresh 32-byte hex one appended to the file (owner-only). */
export function ensureGpuToken(path: string): string {
  const existing = readDotEnv(path)["OHH_GPU_TOKEN"];
  if (existing) {
    ownerOnly(path);
    return existing;
  }
  const token = randomBytes(32).toString("hex");
  const current = existsSync(path) ? readFileSync(path, "utf8") : "";
  const needsBreak = current.length > 0 && !current.endsWith("\n");
  appendFileSync(path, `${needsBreak ? "\n" : ""}OHH_GPU_TOKEN=${token}\n`, { mode: 0o600 });
  ownerOnly(path);
  return token;
}
