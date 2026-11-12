/**
 * A thin request helper over `node:http`.
 *
 * Why not `fetch`: one `send` serves both the plain-HTTP GPU service and the
 * HTTPS provider APIs, so the SSE reader and the error envelope are shared.
 * `fetch` also refuses the WHATWG blocked-port list (the IRC range 6665–6669):
 * undici enforces it unconditionally and rejects the request with `bad port`
 * before a socket is opened — there is no opt-out. `node:http` does not apply
 * it, and neither does Python's httpx, which is why the GPU service's own suite
 * never saw it when the service was first given a port in that range.
 *
 * Verified on this machine: fetch to 127.0.0.1:6666 fails with `bad port` while
 * node:http succeeds, and both succeed on 6670.
 *
 * Still zero-dependency, and an `IncomingMessage` is already an
 * `AsyncIterable<Buffer>`, which the SSE reader consumes directly.
 */

import { request as httpRequest, type IncomingMessage } from "node:http";
import { request as httpsRequest } from "node:https";

export interface HttpResponse {
  readonly status: number;
  readonly contentType: string;
  /** Lower-cased response headers. The Gateway needs `retry-after` on a 429. */
  readonly headers: Record<string, string | string[] | undefined>;
  /** Async-iterable body. Consume it, or call `abort()` to stop early. */
  readonly stream: IncomingMessage;
  abort(): void;
}

export interface SendOptions {
  readonly method?: string;
  readonly headers?: Record<string, string>;
  readonly body?: string;
  /** Milliseconds to wait for response headers. Omit for no limit (SSE). */
  readonly headersTimeoutMs?: number;
  /** Aborts the request, including a connect still in flight. */
  readonly signal?: AbortSignal;
  /**
   * Enable TCP keepalive with this initial delay on the socket. For a stream
   * that carries no heartbeat: lets the OS notice a dead peer.
   */
  readonly keepAliveMs?: number;
}

export function send(url: string, options: SendOptions = {}): Promise<HttpResponse> {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    // One transport for both halves of the harness: the GPU service is plain HTTP on
    // loopback, the Gateway's providers are HTTPS. Picking by protocol keeps
    // a single `send`, so the SSE reader and the error envelope are shared.
    const transport = target.protocol === "https:" ? httpsRequest : httpRequest;
    const req = transport(
      {
        protocol: target.protocol,
        hostname: target.hostname,
        // Empty for a default-port URL (https://api.anthropic.com); `undefined`
        // lets the transport apply 443/80 rather than coercing "" to a port.
        port: target.port === "" ? undefined : target.port,
        path: `${target.pathname}${target.search}`,
        method: options.method ?? "GET",
        headers: options.headers ?? {},
        signal: options.signal,
      },
      (res) => {
        resolve({
          status: res.statusCode ?? 0,
          contentType: String(res.headers["content-type"] ?? ""),
          headers: res.headers,
          stream: res,
          abort: () => {
            res.destroy();
            req.destroy();
          },
        });
      },
    );

    req.on("error", reject);
    if (options.keepAliveMs !== undefined) {
      const keepAliveMs = options.keepAliveMs;
      req.on("socket", (socket) => socket.setKeepAlive(true, keepAliveMs));
    }
    if (options.headersTimeoutMs !== undefined) {
      req.setTimeout(options.headersTimeoutMs, () => {
        req.destroy(new Error(`no response headers within ${options.headersTimeoutMs}ms`));
      });
    }
    if (options.body !== undefined) req.write(options.body);
    req.end();
  });
}

export async function readBody(response: HttpResponse): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of response.stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array));
  }
  return Buffer.concat(chunks);
}

export async function readJson<T>(response: HttpResponse): Promise<T> {
  const text = (await readBody(response)).toString("utf8");
  return JSON.parse(text) as T;
}
