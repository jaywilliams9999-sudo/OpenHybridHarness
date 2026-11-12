import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

import { watchGpu } from "../src/gpu/watch.ts";
import type { GpuEvent } from "../src/gpu/client.ts";

interface Stub {
  url: string;
  connections: IncomingMessage[];
  close(): Promise<void>;
}

/** A stub /events: `onConnect` decides what each connection receives. */
function startStub(
  onConnect: (res: ServerResponse, n: number) => void,
): Promise<Stub> {
  const connections: IncomingMessage[] = [];
  const server = createServer((req, res) => {
    connections.push(req);
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.flushHeaders();
    onConnect(res, connections.length);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        url: `http://127.0.0.1:${port}`,
        connections,
        close: () =>
          new Promise<void>((done) => {
            server.closeAllConnections();
            server.close(() => done());
          }),
      });
    });
  });
}

const until = async (check: () => boolean, ms = 2000): Promise<void> => {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error("timed out waiting");
    await new Promise((r) => setTimeout(r, 5));
  }
};

const frame = (event: string, data: string) => `event: ${event}\ndata: ${data}\n\n`;

test("relays events with their kind", async () => {
  const stub = await startStub((res) => res.write(frame("op.done", '{"op_id":"a","x":1}')));
  const seen: GpuEvent[] = [];
  const watch = watchGpu({ baseUrl: stub.url, token: "t", onEvent: (e) => seen.push(e), onLink: () => {} });
  try {
    await until(() => seen.length === 1);
    assert.deepEqual(seen[0], { event: "op.done", op_id: "a", x: 1 });
  } finally {
    watch.close();
    await stub.close();
  }
});

test("skips a frame with bad JSON", async () => {
  const stub = await startStub((res) => {
    res.write(frame("op.progress", "not json"));
    res.write(frame("op.done", '{"op_id":"b"}'));
  });
  const seen: GpuEvent[] = [];
  const watch = watchGpu({ baseUrl: stub.url, token: "t", onEvent: (e) => seen.push(e), onLink: () => {} });
  try {
    await until(() => seen.length >= 1);
    assert.equal(seen.length, 1);
    assert.equal(seen[0]!.op_id, "b");
  } finally {
    watch.close();
    await stub.close();
  }
});

test("reports the link down and reconnects", async () => {
  const stub = await startStub((res, n) => {
    if (n === 1) res.end();
    else res.write(frame("op.done", '{"op_id":"c"}'));
  });
  const links: boolean[] = [];
  const seen: GpuEvent[] = [];
  const watch = watchGpu({
    baseUrl: stub.url,
    token: "t",
    retryMs: 20,
    onEvent: (e) => seen.push(e),
    onLink: (up) => links.push(up),
  });
  try {
    await until(() => seen.length === 1);
    assert.deepEqual(links, [true, false, true]);
    assert.equal(seen[0]!.op_id, "c");
  } finally {
    watch.close();
    await stub.close();
  }
});

test("sends the bearer token", async () => {
  const stub = await startStub(() => {});
  const watch = watchGpu({ baseUrl: stub.url, token: "secret-1", onEvent: () => {}, onLink: () => {} });
  try {
    await until(() => stub.connections.length === 1);
    assert.equal(stub.connections[0]!.headers.authorization, "Bearer secret-1");
  } finally {
    watch.close();
    await stub.close();
  }
});

test("close stops reconnecting", async () => {
  const stub = await startStub((res) => res.end());
  const watch = watchGpu({ baseUrl: stub.url, token: "t", retryMs: 20, onEvent: () => {}, onLink: () => {} });
  try {
    await until(() => stub.connections.length >= 2);
    watch.close();
    await new Promise((r) => setTimeout(r, 30));
    const count = stub.connections.length;
    await new Promise((r) => setTimeout(r, 60));
    assert.equal(stub.connections.length, count);
  } finally {
    watch.close();
    await stub.close();
  }
});

test("close cancels a connect that never gets a reply", async () => {
  const { createServer: tcp } = await import("node:net");
  let accepted = 0;
  let closedSockets = 0;
  const server = tcp((socket) => {
    accepted++;
    socket.on("close", () => closedSockets++);
    socket.on("error", () => {});
    socket.resume(); // unread sockets never see the peer close
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as AddressInfo;
  const watch = watchGpu({ baseUrl: `http://127.0.0.1:${port}`, token: "t", onEvent: () => {}, onLink: () => {} });
  try {
    await until(() => accepted === 1);
    watch.close();
    await until(() => closedSockets === 1);
  } finally {
    watch.close();
    await new Promise<void>((r) => server.close(() => r()));
  }
});

test("a throwing onLink does not stop the reconnect", async () => {
  const stub = await startStub((res, n) => {
    if (n === 1) res.end();
  });
  const watch = watchGpu({
    baseUrl: stub.url,
    token: "t",
    retryMs: 20,
    onEvent: () => {},
    onLink: () => {
      throw new Error("consumer bug");
    },
  });
  try {
    await until(() => stub.connections.length >= 2);
  } finally {
    watch.close();
    await stub.close();
  }
});
