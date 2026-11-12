/**
 * The Hub's entry point: the board, the GPU service card and the model list, served
 * to a browser on this machine.
 *
 *   npm run hub        then open http://127.0.0.1:6680/
 *
 * One process holds the Ledger, a GPU service client, a watch on the GPU service's event
 * stream and the shared runner; the browser gets a page and the routes in
 * `hub/server.ts`, never the GPU service's token.
 */

import { randomBytes } from "node:crypto";
import { join } from "node:path";

import { applyDotEnv } from "./env.ts";
import { ConfigError, defaultModelsPath, describeMode, loadConfig } from "./config/config.ts";
import { openLedger } from "./ledger/ledger.ts";
import { createGpuClient } from "./gpu/client.ts";
import { watchGpu } from "./gpu/watch.ts";
import { HARNESS, type RunnerOptions } from "./run/runner.ts";
import { sweepInterrupted } from "./run/result.ts";
import { createHubRunner } from "./hub/drive.ts";
import { createHubServer, type HubEvent } from "./hub/server.ts";

/** The board may be re-read at most this often while agents are busy. */
const BOARD_EVERY_MS = 250;

/** Call `fn` soon, and at most once per `ms`: a burst becomes one call. */
function throttle(fn: () => void, ms: number): () => void {
  let last = 0;
  let timer: NodeJS.Timeout | undefined;
  return () => {
    if (timer) return;
    timer = setTimeout(
      () => {
        timer = undefined;
        last = Date.now();
        fn();
      },
      Math.max(0, last + ms - Date.now()),
    );
  };
}

/** The port the Hub listens on: `OHH_HUB_PORT`, default 6680. */
export function hubPort(env: NodeJS.ProcessEnv = process.env): number {
  const portRaw = env["OHH_HUB_PORT"] ?? "6680";
  if (!/^\d+$/.test(portRaw) || Number(portRaw) > 65535) {
    throw new ConfigError(`OHH_HUB_PORT must be a port number, got "${portRaw}".`);
  }
  return Number(portRaw);
}

/** Start the Hub in this process; resolves once it is listening. */
export async function startHub(
  env: NodeJS.ProcessEnv = process.env,
): Promise<{ close(): Promise<void> }> {
  const config = loadConfig(env);
  const port = hubPort(env);
  const ledger = openLedger(config.ledgerPath);
  sweepInterrupted(ledger, config.runningDir);
  const gpu = createGpuClient({ baseUrl: config.gpuUrl, token: config.gpuToken });

  const listeners = new Set<(e: HubEvent) => void>();
  const emit = (e: HubEvent): void => {
    // A throwing listener must not take the Hub down from inside a timer.
    for (const listener of listeners) {
      try {
        listener(e);
      } catch (error) {
        console.error(`hub: an event listener failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  };

  // Down until the watch says otherwise: it reports a drop only after a link
  // that was up, so a GPU service that is off at start is never announced.
  let link = false;
  const watch = watchGpu({
    baseUrl: config.gpuUrl,
    token: config.gpuToken,
    onEvent: (e) => emit({ ...e, source: "gpu", event: e.event }),
    onLink: (up) => {
      link = up;
      emit({ source: "gpu", event: "gpu.link", up });
    },
  });

  const boardChanged = throttle(
    () => emit({ source: "hub", event: "board.changed" }),
    BOARD_EVERY_MS,
  );
  const opts: RunnerOptions = {
    config,
    gpu,
    ledger,
    harness: HARNESS,
    output: () => ({
      onEvent: boardChanged,
      onToolCall: boardChanged,
      onCost: boardChanged,
    }),
    onChildCreated: boardChanged,
  };
  const runner = createHubRunner(opts, boardChanged);

  const server = createHubServer({
    ledger,
    gpu,
    runner,
    registryPath: defaultModelsPath(env),
    staticDir: join(import.meta.dirname, "..", "hub"),
    artifactsDir: config.artifactsDir,
    secret: randomBytes(24).toString("hex"),
    runningDir: config.runningDir,
    emit,
    subscribe: (listener) => {
      listeners.add(listener);
      // A new page learns the link's state now, not at its next change.
      listener({ source: "gpu", event: "gpu.link", up: link });
      return () => listeners.delete(listener);
    },
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  server.on("error", (error) => console.error(`hub: ${error.message}`));
  console.log(`hub  http://127.0.0.1:${port}/`);
  // A missing key means every run is local; say so rather than leave it implied.
  console.log(`     ${describeMode(config)}`);

  return {
    close: () =>
      new Promise<void>((resolve) => {
        watch.close();
        server.closeAllConnections();
        server.close(() => {
          ledger.close();
          resolve();
        });
      }),
  };
}

if (import.meta.main) {
  // Orchestrator-only secrets live in the repo-root .env; variables already set win.
  applyDotEnv(join(import.meta.dirname, "..", "..", ".env"));
  try {
    const hub = await startHub();
    const stop = (): void => void hub.close().then(() => process.exit(0));
    process.on("SIGINT", stop);
    process.on("SIGTERM", stop);
  } catch (error) {
    if (!(error instanceof ConfigError)) throw error;
    console.error(`${error.name}: ${error.message}`);
    process.exit(1);
  }
}
