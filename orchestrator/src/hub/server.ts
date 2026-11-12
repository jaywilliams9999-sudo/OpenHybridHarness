/**
 * The Hub's HTTP server: the page, the board, the GPU service card and model list,
 * the event relay, and the few requests that change anything.
 *
 * `node:http` alone. It binds 127.0.0.1 and trusts nothing else about the
 * caller: a page on any other site can still make a browser send a request
 * here, and a goal spends real money. So every request must name this server
 * as its Host (a rebinding page names its own), and every POST must also come
 * from this origin, carry the per-process secret only the served page holds,
 * and be JSON -- all checked before a body is read. The GPU service's token stays
 * in this process; the browser only ever sees what these routes return.
 */

import { timingSafeEqual } from "node:crypto";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { extname, isAbsolute, join, relative } from "node:path";

import type { GpuClient, LoadResult } from "../gpu/client.ts";
import type { Ledger } from "../ledger/ledger.ts";
import { RunRefused } from "../run/runner.ts";
import { sweepInterrupted } from "../run/result.ts";
import { againRefusal, buildBoard } from "./board.ts";
import { goalDetail } from "./detail.ts";
import { mergeModels, readRegistryModels } from "./models.ts";
import { HUB_MAX_BUDGET_USD, type HubRunner } from "./drive.ts";

export interface HubEvent {
  source: "gpu" | "hub";
  event: string;
  [k: string]: unknown;
}

export interface HubDeps {
  ledger: Ledger;
  gpu: Pick<GpuClient, "status" | "health" | "load" | "unload">;
  /**
   * May throw `RunRefused` synchronously; that is the 404 or 409. The promise
   * it returns otherwise runs in the background.
   */
  runner: HubRunner;
  registryPath: string;
  staticDir: string;
  artifactsDir: string;
  secret: string;
  runningDir: string;
  subscribe(listener: (e: HubEvent) => void): () => void;
  /** An event from the Hub itself, to every subscriber. */
  emit(e: HubEvent): void;
  /** How long a GPU service reading may take before it counts as unreachable. */
  gpuTimeoutMs?: number;
}

const MAX_BODY = 16 * 1024;
const DEFAULT_BUDGET_USD = 1.5;
const MAX_BUDGET_USD = HUB_MAX_BUDGET_USD;
const MAX_GOAL = 2000;
const PING_MS = 15_000;
/** A sleeping GPU machine never answers; the strip must not show its last reading as live. */
const GPU_TIMEOUT_MS = 4_000;
const ASSET = /^[a-z-]+\.(js|css)$/;

const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
};

/**
 * What an artifact may be served as: inert images only. Anything else -- an
 * `.html` or `.svg` an agent wrote -- would run on the Hub's own origin, where
 * it could read the page's secret, so it is a download instead.
 */
const ARTIFACT_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

/** A reply the handler gives up with: status, and the error the page shows. */
class HttpError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** A path segment, decoded; a malformed one names nothing here. */
function segment(raw: string): string {
  try {
    return decodeURIComponent(raw);
  } catch (error) {
    if (error instanceof URIError) throw new HttpError(404, "not found");
    throw error;
  }
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  res.end(JSON.stringify(body));
}

function plain(res: ServerResponse, status: number, text: string): void {
  res.writeHead(status, { "content-type": "text/plain; charset=utf-8" });
  res.end(text);
}

/** `work`, or a rejection once `ms` has passed without it. */
function within<T>(ms: number, work: Promise<T>): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`the GPU service did not answer within ${ms} ms`)), ms);
  });
  return Promise.race([work, late]).finally(() => clearTimeout(timer));
}

function firstLine(error: unknown): string {
  return (error instanceof Error ? error.message : String(error)).split("\n")[0]!;
}

function sameSecret(given: string, secret: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(secret);
  return a.length === b.length && timingSafeEqual(a, b);
}

function readBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY) {
        reject(new HttpError(413, "body too large"));
        req.pause();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch {
        reject(new HttpError(400, "body is not JSON"));
      }
    });
    req.on("error", reject);
  });
}

/** `goal` and `budget_usd` as the runner wants them, or a 400. */
function goalRequest(body: unknown): { goal: string; budgetUsd: number } {
  const { goal, budget_usd } = (body ?? {}) as Record<string, unknown>;
  if (typeof goal !== "string") throw new HttpError(400, "goal must be text");
  const trimmed = goal.trim();
  if (trimmed.length < 1 || trimmed.length > MAX_GOAL) {
    throw new HttpError(400, `goal must be 1 to ${MAX_GOAL} characters`);
  }
  if (budget_usd === undefined) return { goal: trimmed, budgetUsd: DEFAULT_BUDGET_USD };
  if (
    typeof budget_usd !== "number" ||
    !Number.isFinite(budget_usd) ||
    budget_usd <= 0 ||
    budget_usd > MAX_BUDGET_USD
  ) {
    throw new HttpError(400, `budget_usd must be more than 0 and at most ${MAX_BUDGET_USD}`);
  }
  return { goal: trimmed, budgetUsd: budget_usd };
}

export function createHubServer(deps: HubDeps): Server {
  const { ledger, gpu, runner } = deps;
  const gpuTimeoutMs = deps.gpuTimeoutMs ?? GPU_TIMEOUT_MS;

  /** Background work: its outcome reaches the page another way, so log only. */
  const background = (what: string, work: Promise<unknown>): void => {
    work.catch((error: unknown) => {
      console.error(`hub: ${what} failed: ${firstLine(error)}`);
    });
  };

  /**
   * A load or unload, told to the page when it settles: the GPU service's own
   * events say nothing of a refusal, and only the load's result names what
   * it actually unloaded.
   */
  const modelAction = (model: string, action: "load" | "unload"): void => {
    const work: Promise<unknown> = action === "load" ? gpu.load(model) : gpu.unload(model);
    const told = { source: "hub" as const, event: "model.outcome", model, action };
    work.then(
      (result) => {
        const evicted =
          action === "load"
            ? ((result as LoadResult).evicted ?? []).map(({ model, freed_mb, reason }) => ({ model, freed_mb, reason }))
            : undefined;
        deps.emit({ ...told, ok: true, ...(evicted ? { evicted } : {}) });
      },
      (error: unknown) => {
        console.error(`hub: ${action} ${model} failed: ${firstLine(error)}`);
        deps.emit({ ...told, ok: false, error: firstLine(error) });
      },
    );
  };

  /** Run a runner call, turning its synchronous refusal into an HTTP one. */
  const accept = (what: string, start: () => Promise<unknown>): void => {
    let work: Promise<unknown>;
    try {
      work = start();
    } catch (error) {
      if (!(error instanceof RunRefused)) throw error;
      const status = error.head.startsWith("unknown") ? 404 : 409;
      throw new HttpError(status, `${error.head} ${error.detail}`.trim());
    }
    background(what, work);
  };

  const page = (res: ServerResponse, name: string): void => {
    const file = join(deps.staticDir, name);
    if (!existsSync(file)) return plain(res, 503, `the Hub page is not built: ${name} is missing`);
    const html = readFileSync(file, "utf8").replaceAll("{{HUB_SECRET}}", deps.secret);
    res.writeHead(200, { "content-type": TYPES[".html"]!, "cache-control": "no-store" });
    res.end(html);
  };

  const asset = (res: ServerResponse, name: string): void => {
    const file = join(deps.staticDir, name);
    if (!ASSET.test(name) || !existsSync(file)) return plain(res, 404, "not found");
    res.writeHead(200, { "content-type": TYPES[extname(name)]! });
    res.end(readFileSync(file));
  };

  /** An artifact by its Ledger id, and only one inside the artifacts directory. */
  const artifact = (res: ServerResponse, id: string): void => {
    const row = ledger.getArtifact(id);
    let real: string | undefined;
    try {
      if (row) real = realpathSync(row.path);
    } catch {
      // A row whose file is gone is not found.
    }
    if (real !== undefined && existsSync(deps.artifactsDir)) {
      const inside = relative(realpathSync(deps.artifactsDir), real);
      if (inside !== "" && !inside.startsWith("..") && !isAbsolute(inside)) {
        const type = ARTIFACT_TYPES[extname(real).toLowerCase()];
        res.writeHead(200, {
          "content-type": type ?? "application/octet-stream",
          ...(type ? {} : { "content-disposition": "attachment" }),
          "x-content-type-options": "nosniff",
          "content-security-policy": "sandbox; default-src 'none'",
        });
        res.end(readFileSync(real));
        return;
      }
    }
    plain(res, 404, "not found");
  };

  const card = async (res: ServerResponse): Promise<void> => {
    try {
      const [health, status] = await within(gpuTimeoutMs, Promise.all([gpu.health(), gpu.status()]));
      json(res, 200, { reachable: true, health, status });
    } catch (error) {
      json(res, 200, { reachable: false, error: error instanceof Error ? error.message : String(error) });
    }
  };

  const models = async (res: ServerResponse): Promise<void> => {
    const registry = readRegistryModels(deps.registryPath);
    const status = await within(gpuTimeoutMs, gpu.status()).catch(() => null);
    json(res, 200, mergeModels(registry, status));
  };

  const events = (req: IncomingMessage, res: ServerResponse): void => {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-store",
      connection: "keep-alive",
    });
    res.flushHeaders();
    const unsubscribe = deps.subscribe((e) => {
      res.write(`event: ${e.event}\ndata: ${JSON.stringify(e)}\n\n`);
    });
    const ping = setInterval(() => res.write(": ping\n\n"), PING_MS);
    req.on("close", () => {
      clearInterval(ping);
      unsubscribe();
    });
  };

  async function get(req: IncomingMessage, res: ServerResponse, path: string): Promise<void> {
    if (path === "/") return page(res, "index.html");
    if (path === "/models") return page(res, "models.html");
    if (path === "/api/board") {
      // A goal whose process died shows as failed, not running forever.
      sweepInterrupted(ledger, deps.runningDir);
      return json(res, 200, buildBoard(ledger, deps.runningDir));
    }
    if (path === "/api/card") return card(res);
    if (path === "/api/models") return models(res);
    if (path === "/api/events") return events(req, res);

    let m: RegExpMatchArray | null;
    if ((m = path.match(/^\/api\/goals\/([^/]+)$/))) {
      const detail = goalDetail(ledger, segment(m[1]!), deps.runningDir);
      return detail ? json(res, 200, detail) : json(res, 404, { error: "unknown goal" });
    }
    if ((m = path.match(/^\/artifacts\/([^/]+)$/))) return artifact(res, segment(m[1]!));
    if ((m = path.match(/^\/([^/]+)$/))) return asset(res, segment(m[1]!));
    plain(res, 404, "not found");
  }

  async function post(req: IncomingMessage, res: ServerResponse, path: string, port: number): Promise<void> {
    const origin = req.headers.origin;
    const secret = req.headers["x-hub-secret"];
    if (
      (origin !== `http://127.0.0.1:${port}` && origin !== `http://localhost:${port}`) ||
      typeof secret !== "string" ||
      !sameSecret(secret, deps.secret)
    ) {
      throw new HttpError(403, "forbidden");
    }
    if (!/^application\/json\b/i.test(req.headers["content-type"] ?? "")) {
      throw new HttpError(415, "body must be JSON");
    }
    const body = await readBody(req);

    let m: RegExpMatchArray | null;
    if (path === "/api/goals") {
      const goal = goalRequest(body);
      accept("goal", () => runner.startGoal(goal));
      return json(res, 202, { accepted: true });
    }
    if ((m = path.match(/^\/api\/approvals\/([^/]+)$/))) {
      const id = segment(m[1]!);
      const { decision } = (body ?? {}) as Record<string, unknown>;
      if (decision !== "approved" && decision !== "denied") {
        throw new HttpError(400, "decision must be approved or denied");
      }
      const approval = ledger.readApproval(id);
      if (!approval) throw new HttpError(404, "unknown approval");
      if (approval.status !== "pending") throw new HttpError(409, `already ${approval.status}`);
      accept("decision", () => runner.decide(id, decision));
      return json(res, 202, { accepted: true });
    }
    if ((m = path.match(/^\/api\/goals\/([^/]+)\/again$/))) {
      const id = segment(m[1]!);
      const root = ledger.getRun(id);
      if (!root || root.parent_id !== null) throw new HttpError(404, "unknown goal");
      const refusal = againRefusal(ledger, root);
      if (refusal !== null) throw new HttpError(409, refusal);
      accept("retry", () => runner.again(id));
      return json(res, 202, { accepted: true });
    }
    if ((m = path.match(/^\/api\/models\/([^/]+)\/(load|unload)$/))) {
      const name = segment(m[1]!);
      if (!readRegistryModels(deps.registryPath).some((model) => model.name === name)) {
        throw new HttpError(404, "the registry has no such model");
      }
      modelAction(name, m[2] as "load" | "unload");
      return json(res, 202, { accepted: true });
    }
    throw new HttpError(404, "not found");
  }

  const server = createServer((req, res) => {
    // Read at request time, so a server listening on port 0 knows its port.
    const port = (server.address() as AddressInfo).port;
    // No other site may frame the Hub and steer a click onto Run again or Load.
    res.setHeader("x-frame-options", "DENY");
    res.setHeader("content-security-policy", "frame-ancestors 'none'");
    const handle = async (): Promise<void> => {
      const host = req.headers.host;
      if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) {
        throw new HttpError(403, "forbidden");
      }
      const path = new URL(req.url ?? "/", "http://hub").pathname;
      if (req.method === "GET") return get(req, res, path);
      if (req.method === "POST") return post(req, res, path, port);
      throw new HttpError(405, "method not allowed");
    };
    handle().catch((error: unknown) => {
      if (res.headersSent) {
        res.destroy();
        return;
      }
      if (error instanceof HttpError) {
        // A refused body is not read; close rather than drain it.
        if (error.status === 413) res.setHeader("connection", "close");
        return json(res, error.status, { error: error.message });
      }
      const message = error instanceof Error ? error.message : String(error);
      console.error(`hub: ${req.method} ${req.url} failed: ${message.split("\n")[0]}`);
      json(res, 500, { error: "internal error" });
    });
  });
  return server;
}
