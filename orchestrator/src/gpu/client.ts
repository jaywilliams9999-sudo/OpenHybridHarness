/**
 * GpuClient — the sole wire-aware module.
 *
 * It owns the wire and nothing else: no filesystem, no Ledger. `fetchArtifact`
 * hands back bytes and the tool handler decides where they live, which is what
 * keeps this testable against a stub server with no disk and no database.
 *
 * The op protocol subscribes to `/events` BEFORE posting, then races
 * the terminal SSE event against a slow poll of `/ops/{id}`. The ordering and
 * the backstop are both load-bearing: the GPU service's own suite asserts that events
 * emitted before a subscriber exists are never replayed, and that a full queue
 * drops the oldest event. POST-then-subscribe can therefore lose `op.done` to a
 * bus with nobody listening, and a client trusting the stream alone would wait
 * forever on an op that already finished.
 */

import { readSse } from "./sse.ts";
import { readBody, readJson, send, type HttpResponse } from "./http.ts";

export class GpuError extends Error {
  readonly code: string | undefined;
  readonly remedy: string | undefined;
  readonly status: number | undefined;

  constructor(
    message: string,
    detail: { code?: string; remedy?: string; status?: number } = {},
  ) {
    super(message);
    this.name = "GpuError";
    this.code = detail.code;
    this.remedy = detail.remedy;
    this.status = detail.status;
  }
}

export interface GpuEvent {
  readonly event: string;
  readonly op_id?: string;
  readonly step?: number;
  readonly total?: number;
  readonly [key: string]: unknown;
}

export interface Health {
  status: string;
  gpu: string;
  vram_total_mb: number;
  vram_free_mb: number;
  version: string;
}

export interface Vram {
  total_mb: number;
  free_mb: number;
  headroom_mb?: number;
  usable_free_mb: number;
}

export interface Status {
  vram: Vram;
  resident: Array<{
    model: string;
    vram_mb: number;
    pinned: boolean;
    role: string;
    idle_s?: number;
    load_ms?: number | null;
  }>;
  busy: boolean;
  queue_depth: number;
  busy_model?: string | null;
  recent?: Array<{ event: string; at: string; [key: string]: unknown }>;
  /** Registry models as the card can run them. Absent from an older service. */
  models?: Array<{
    name: string;
    role: string;
    available: boolean;
    num_ctx: number | null;
    reduced_from: number | null;
    reason: string | null;
  }>;
  kv_cache_type?: string;
}

export type GpuStatus = Status;

/**
 * True unless the service reports image models and none of them is available.
 * A status without a model list (older service) is "unknown", so available.
 */
export function imageAvailable(status: Status): boolean {
  const images = (status.models ?? []).filter((m) => m.role === "image");
  return images.length === 0 || images.some((m) => m.available);
}

export interface EvictionRecord {
  model: string;
  freed_mb: number;
  reason: string;
}

export interface ImageResult {
  model: string;
  steps: number;
  artifact_id: string;
  artifact_url: string;
  evicted: EvictionRecord[];
}

export interface LoadResult {
  model: string;
  vram_mb: number;
  load_ms: number;
  already_resident?: boolean;
  evicted?: EvictionRecord[];
}

/**
 * One local text turn.
 *
 * `content` is the whole reply. The GPU service also streams `op.token` frames, which
 * reach `onProgress` like any other event — but those are advisory: the GPU service's
 * event bus drops frames under backpressure, so assembling the reply from them
 * would silently lose words. Read `content`; watch the tokens.
 */
export interface TextResult {
  model: string;
  content: string;
  /** Ollama's shape: `{function: {name, arguments}}`. No call ids, so the ollama adapter mints its own. */
  tool_calls: Array<{ function?: { name?: string; arguments?: unknown } }>;
  prompt_tokens: number;
  completion_tokens: number;
  done_reason: string;
  num_ctx: number | null;
  evicted?: EvictionRecord[];
}

export interface TextRequest {
  readonly messages: Array<Record<string, unknown>>;
  readonly tools?: Array<Record<string, unknown>>;
  readonly num_ctx?: number;
  readonly temperature?: number;
  readonly max_tokens?: number;
  readonly model?: string;
}

export interface GpuClientOptions {
  readonly baseUrl: string;
  readonly token: string;
  /** Reconciliation backstop interval. Deliberately slow; not the mechanism. */
  readonly pollIntervalMs?: number;
  /**
   * Grace period after `/events` returns headers, before posting. The server's
   * generator — and therefore its bus subscription — starts as the response
   * begins streaming, which we cannot observe directly because the GPU service sends
   * no greeting frame. The poll covers the residual race; this just shrinks it.
   */
  readonly settleMs?: number;
  readonly onProgress?: (event: GpuEvent) => void;
}

export interface GpuClient {
  health(): Promise<Health>;
  status(): Promise<Status>;
  load(model: string): Promise<LoadResult>;
  unload(model: string): Promise<{ freed_mb: number }>;
  generateImage(req: {
    prompt: string;
    steps?: number;
    width?: number;
    height?: number;
    seed?: number;
  }): Promise<ImageResult>;
  generateText(req: TextRequest): Promise<TextResult>;
  fetchArtifact(url: string): Promise<Uint8Array>;
}

type Terminal = { status: "done" | "failed"; result: Record<string, unknown> };

const sleep = (ms: number) => new Promise((done) => setTimeout(done, ms));

/**
 * Turn a socket-level failure into a GpuError that names the target. The
 * commonest failure by far is simply that the GPU service is not running, and a bare
 * ECONNREFUSED object tells the operator nothing about what to start.
 */
function asGpuError(error: unknown, base: string, what: string): GpuError {
  if (error instanceof GpuError) return error;
  const code = (error as { code?: string } | undefined)?.code;
  const reachability = ["ECONNREFUSED", "ETIMEDOUT", "ENOTFOUND", "EHOSTUNREACH", "ECONNRESET"];
  if (code && reachability.includes(code)) {
    return new GpuError(`The GPU service at ${base} did not answer (${code}) for ${what}.`, {
      code,
      remedy:
        "Start the GPU service (python -m ohh_gpu.main) and check " +
        "OHH_GPU_URL, or the registry's [gpu_service] host and port.",
    });
  }
  return new GpuError(
    `${what} against ${base} failed: ${error instanceof Error ? error.message : String(error)}`,
    ...(code !== undefined ? [{ code }] : []),
  );
}

export function createGpuClient(options: GpuClientOptions): GpuClient {
  const base = options.baseUrl.replace(/\/+$/, "");
  const pollIntervalMs = options.pollIntervalMs ?? 2000;
  const settleMs = options.settleMs ?? 100;
  const headers = { authorization: `Bearer ${options.token}` };

  // `init`, not `options` — the outer `options` is the client's own config, and
  // shadowing it here made it ambiguous which one a reader was looking at.
  async function request(
    path: string,
    init: { method?: string; body?: unknown } = {},
  ): Promise<HttpResponse> {
    let response: HttpResponse;
    try {
      response = await send(`${base}${path}`, {
        method: init.method ?? "GET",
        headers:
          init.body === undefined
            ? headers
            : { ...headers, "content-type": "application/json" },
        ...(init.body === undefined ? {} : { body: JSON.stringify(init.body) }),
      });
    } catch (error) {
      throw asGpuError(error, base, path);
    }

    if (response.status >= 200 && response.status < 300) return response;

    if (response.status === 401) {
      response.abort();
      throw new GpuError(
        `The GPU service rejected the bearer token (401). Check that OHH_GPU_TOKEN ` +
          `matches the value the GPU service was started with.`,
        { status: 401, code: "UNAUTHORIZED" },
      );
    }

    let code: string | undefined;
    let remedy: string | undefined;
    let error = `HTTP ${response.status}`;
    try {
      const body = await readJson<Record<string, unknown>>(response);
      // The GPU service wraps errors in one envelope: {code, error, remedy}.
      const detail = (body["detail"] ?? body) as Record<string, unknown>;
      code = typeof detail["code"] === "string" ? detail["code"] : undefined;
      remedy = typeof detail["remedy"] === "string" ? detail["remedy"] : undefined;
      if (typeof detail["error"] === "string") error = detail["error"];
    } catch {
      // a non-JSON error body; the status line is all we have
    }
    throw new GpuError(`${path}: ${error}`, {
      ...(code !== undefined ? { code } : {}),
      ...(remedy !== undefined ? { remedy } : {}),
      status: response.status,
    });
  }

  async function getJson<T>(path: string): Promise<T> {
    return readJson<T>(await request(path));
  }

  async function postJson<T>(path: string, body: unknown): Promise<T> {
    return readJson<T>(await request(path, { method: "POST", body }));
  }

  /** A live `/events` subscription that records terminals as they arrive. */
  async function subscribe() {
    // No headers timeout: an idle /events stream legitimately sends nothing.
    let response: HttpResponse;
    try {
      response = await send(`${base}/events`, { headers });
    } catch (error) {
      throw asGpuError(error, base, "/events");
    }
    if (response.status < 200 || response.status >= 300) {
      response.abort();
      throw new GpuError(`Could not open /events (HTTP ${response.status}).`, {
        status: response.status,
      });
    }

    const seen = new Map<string, Terminal>();
    const waiters = new Map<string, (terminal: Terminal) => void>();

    const pump = (async () => {
      try {
        for await (const frame of readSse(response.stream)) {
          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(frame.data) as Record<string, unknown>;
          } catch {
            continue;
          }
          const event: GpuEvent = { event: frame.event, ...payload };

          if (frame.event === "op.progress") {
            options.onProgress?.(event);
            continue;
          }
          if (frame.event !== "op.done" && frame.event !== "op.failed") {
            options.onProgress?.(event);
            continue;
          }

          const opId = typeof payload["op_id"] === "string" ? payload["op_id"] : "";
          if (!opId) continue;
          const { op_id: _ignored, ...rest } = payload;
          const terminal: Terminal = {
            status: frame.event === "op.done" ? "done" : "failed",
            result: rest,
          };
          seen.set(opId, terminal);
          waiters.get(opId)?.(terminal);
          waiters.delete(opId);
        }
      } catch {
        // The stream ends when we abort, or when the GPU service goes away. Either
        // way the reconciliation poll is still running and will finish the op.
      }
    })();

    // Give the server's generator a moment to reach its bus subscription.
    await sleep(settleMs);

    return {
      waitFor(opId: string): Promise<Terminal> {
        const already = seen.get(opId);
        if (already) return Promise.resolve(already);
        return new Promise<Terminal>((resolve) => waiters.set(opId, resolve));
      },
      async close(): Promise<void> {
        response.abort();
        await pump;
      },
    };
  }

  async function pollUntilTerminal(
    opId: string,
    stop: { aborted: boolean },
  ): Promise<Terminal> {
    for (;;) {
      await sleep(pollIntervalMs);
      if (stop.aborted) return await new Promise<Terminal>(() => {}); // never settles
      const body = await getJson<{ status: string; result?: unknown }>(
        `/ops/${opId}`,
      );
      if (body.status !== "running") {
        return {
          status: body.status === "done" ? "done" : "failed",
          result: (body.result ?? {}) as Record<string, unknown>,
        };
      }
    }
  }

  async function runOp<T>(path: string, body: unknown): Promise<T> {
    const subscription = await subscribe();
    const stop = { aborted: false };
    try {
      const { op_id: opId } = await postJson<{ op_id: string }>(path, body);
      const terminal = await Promise.race([
        subscription.waitFor(opId),
        pollUntilTerminal(opId, stop),
      ]);

      if (terminal.status === "failed") {
        const result = terminal.result;
        throw new GpuError(
          typeof result["error"] === "string"
            ? result["error"]
            : `op ${opId} failed`,
          {
            ...(typeof result["code"] === "string" ? { code: result["code"] } : {}),
            ...(typeof result["remedy"] === "string"
              ? { remedy: result["remedy"] }
              : {}),
          },
        );
      }
      return terminal.result as T;
    } finally {
      stop.aborted = true;
      await subscription.close();
    }
  }

  return {
    health: () => getJson<Health>("/health"),
    status: () => getJson<Status>("/status"),
    load: (model) => runOp<LoadResult>("/load", { model }),
    unload: (model) => postJson<{ freed_mb: number }>("/unload", { model }),
    generateImage: (req) => runOp<ImageResult>("/generate/image", req),
    generateText: (req) => runOp<TextResult>("/generate/text", req),

    async fetchArtifact(url) {
      return new Uint8Array(await readBody(await request(url)));
    },
  };
}
