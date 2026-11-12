/**
 * A long-lived watcher of the GPU service's `/events` stream.
 *
 * The per-op subscription in `client.ts` lives for one op; the Hub wants the
 * stream for as long as it runs, and the GPU machine may sleep. So this holds the
 * stream open, tells the caller when the link goes up or down, and reconnects
 * after a pause. The GPU service sends no greeting frame: "up" means the response
 * headers arrived with a 200.
 */

import { send, type HttpResponse } from "./http.ts";
import { readSse } from "./sse.ts";
import type { GpuEvent } from "./client.ts";

export interface WatchOptions {
  readonly baseUrl: string;
  readonly token: string;
  readonly onEvent: (event: GpuEvent) => void;
  readonly onLink: (up: boolean) => void;
  /** Pause before reconnecting. Default 3000. */
  readonly retryMs?: number;
}

export function watchGpu(options: WatchOptions): { close(): void } {
  const base = options.baseUrl.replace(/\/+$/, "");
  const retryMs = options.retryMs ?? 3000;
  const headers = { authorization: `Bearer ${options.token}` };

  let closed = false;
  let timer: NodeJS.Timeout | undefined;
  let current: HttpResponse | undefined;
  const controller = new AbortController();

  // A throwing consumer must not stop the reconnect loop.
  const safely = (fn: () => void): void => {
    try {
      fn();
    } catch {
      // ignored on purpose
    }
  };

  async function connect(): Promise<void> {
    let up = false;
    try {
      // No headers timeout and no idle timer: the GPU service sends no heartbeat, so
      // an idle stream is normal. TCP keepalive is what notices a half-open
      // connection after the GPU machine sleeps; the signal lets close() cancel a
      // connect that is still hanging.
      const response = await send(`${base}/events`, {
        headers,
        signal: controller.signal,
        keepAliveMs: 15_000,
      });
      current = response;
      if (closed) {
        response.abort();
        return;
      }
      if (response.status !== 200) {
        response.abort();
      } else {
        up = true;
        safely(() => options.onLink(true));
        for await (const frame of readSse(response.stream)) {
          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(frame.data) as Record<string, unknown>;
          } catch {
            continue;
          }
          safely(() => options.onEvent({ event: frame.event, ...payload }));
        }
      }
    } catch {
      // A refused connection or a dropped stream is the same thing here: the
      // link is down. Fall through to the retry.
    } finally {
      current = undefined;
    }
    if (up && !closed) safely(() => options.onLink(false));
    if (!closed) timer = setTimeout(() => void connect(), retryMs);
  }

  void connect();

  return {
    close() {
      closed = true;
      controller.abort();
      if (timer) clearTimeout(timer);
      timer = undefined;
      current?.abort();
    },
  };
}
