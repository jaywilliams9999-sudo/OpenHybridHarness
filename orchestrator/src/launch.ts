/**
 * One command: `npm start` at the repo root.
 *
 * Makes sure a GPU service token exists, checks Ollama is up, starts the GPU
 * service as a child process, waits for it to be healthy, then runs the Hub in
 * this process. Ctrl+C stops the Hub, then the GPU service.
 */

import { spawn as nodeSpawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { ConfigError, loadConfig } from "./config/config.ts";
import { ensureGpuToken, readDotEnv } from "./env.ts";
import { hubPort, startHub as startHubInProcess } from "./hub.ts";

export class LaunchError extends Error {
  override name = "LaunchError";
}

export interface Child {
  kill(): boolean;
  exitCode?: number | null;
  /** A ChildProcess emits "error" when it cannot be started (e.g. no such program). */
  once?(event: "error", listener: (error: Error) => void): unknown;
}

export type SpawnFn = (
  command: string,
  args: string[],
  options: { cwd?: string; env?: NodeJS.ProcessEnv; stdio?: "inherit" },
) => Child;

export interface LaunchOptions {
  repoRoot: string;
  /** Default: `OHH_OLLAMA_URL`, else `http://127.0.0.1:11434`. */
  ollamaUrl?: string;
  /** The environment to start from; default `process.env`. */
  env?: NodeJS.ProcessEnv;
  /** Stop everything on SIGINT/SIGTERM, including while still starting up. */
  handleSignals?: boolean;
  fetch?: typeof fetch;
  spawn?: SpawnFn;
  startHub?: (env: NodeJS.ProcessEnv) => Promise<{ close(): Promise<void> }>;
  /** Where the running line goes; default `console.log`. */
  log?: (line: string) => void;
  /** Default 60000. */
  healthTimeoutMs?: number;
  /** Default 500. */
  healthIntervalMs?: number;
}

export interface LaunchHandle {
  stop(): Promise<void>;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function pythonFor(repoRoot: string): string {
  const win = join(repoRoot, "gpu-service", ".venv", "Scripts", "python.exe");
  if (existsSync(win)) return win;
  const posix = join(repoRoot, "gpu-service", ".venv", "bin", "python");
  if (existsSync(posix)) return posix;
  return "python";
}

async function ollamaUp(url: string, fetchFn: typeof fetch): Promise<boolean> {
  try {
    const res = await fetchFn(`${url}/api/version`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function healthy(url: string, token: string, fetchFn: typeof fetch): Promise<boolean> {
  try {
    const res = await fetchFn(`${url}/health`, {
      headers: { authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(3000),
    });
    return res.status === 200;
  } catch {
    return false;
  }
}

export async function launch(opts: LaunchOptions): Promise<LaunchHandle> {
  const { repoRoot } = opts;
  const fetchFn = opts.fetch ?? fetch;
  const spawnFn: SpawnFn = opts.spawn ?? ((c, a, o) => nodeSpawn(c, a, o));
  const startHub = opts.startHub ?? startHubInProcess;
  const log = opts.log ?? ((line: string) => console.log(line));

  const envFile = join(repoRoot, ".env");
  ensureGpuToken(envFile);
  // The .env fills in what the shell left unset; the shell always wins.
  const env: NodeJS.ProcessEnv = { ...readDotEnv(envFile), ...(opts.env ?? process.env) };

  const ollamaUrl = opts.ollamaUrl ?? env["OHH_OLLAMA_URL"] ?? "http://127.0.0.1:11434";
  env["OHH_OLLAMA_URL"] = ollamaUrl;

  if (!(await ollamaUp(ollamaUrl, fetchFn))) {
    throw new LaunchError(
      `Ollama isn't reachable at ${ollamaUrl}. Install it from https://ollama.com and start it (\`ollama serve\`), then run this again.`,
    );
  }

  // OHH_MODELS first, as the GPU service started below reads it.
  const config = loadConfig(env, env["OHH_MODELS"] || join(repoRoot, "models.toml"));

  // The GPU service has no use for the orchestrator's provider key.
  const childEnv: NodeJS.ProcessEnv = { ...env };
  delete childEnv["OHH_ANTHROPIC_API_KEY"];

  const python = pythonFor(repoRoot);
  let spawnError: Error | undefined;
  const child = spawnFn(python, ["-m", "ohh_gpu.main"], {
    cwd: join(repoRoot, "gpu-service"),
    env: childEnv,
    stdio: "inherit",
  });
  child.once?.("error", (error) => {
    spawnError = error;
  });

  let hub: { close(): Promise<void> } | undefined;
  let stopping: Promise<void> | undefined;
  const stop = (): Promise<void> =>
    (stopping ??= (async () => {
      try {
        await hub?.close();
      } finally {
        child.kill();
      }
    })());

  const onSignal = (): void => void stop().finally(() => process.exit(0));
  const signals = opts.handleSignals ? (["SIGINT", "SIGTERM"] as const) : [];
  for (const sig of signals) process.on(sig, onSignal);
  const dropSignals = (): void => {
    for (const sig of signals) process.off(sig, onSignal);
  };

  try {
    const timeoutMs = opts.healthTimeoutMs ?? 60_000;
    const intervalMs = opts.healthIntervalMs ?? 500;
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      if (spawnError) {
        throw new LaunchError(
          `Couldn't start the GPU service with ${python}: ${spawnError.message}. Run scripts/install.ps1 first.`,
        );
      }
      if (await healthy(config.gpuUrl, config.gpuToken, fetchFn)) break;
      if (child.exitCode != null) {
        throw new LaunchError(
          `The GPU service exited with code ${child.exitCode} before it was healthy. Its output is above.`,
        );
      }
      if (Date.now() >= deadline) {
        throw new LaunchError("The GPU service didn't become healthy within 60s. Its output is above.");
      }
      await sleep(intervalMs);
    }
    hub = await startHub(env);
  } catch (error) {
    dropSignals();
    await stop();
    throw error;
  }
  log(`OpenHybridHarness is running at http://127.0.0.1:${hubPort(env)}`);

  return { stop };
}

if (import.meta.main) {
  const repoRoot = join(import.meta.dirname, "..", "..");
  try {
    await launch({ repoRoot, handleSignals: true });
  } catch (error) {
    if (!(error instanceof LaunchError || error instanceof ConfigError)) throw error;
    console.error(error.message);
    process.exit(1);
  }
}
