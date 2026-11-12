/** The model list: registry entries merged with the GPU service's status. */

import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

import { mergeModels, readRegistryModels } from "../src/hub/models.ts";
import type { Status } from "../src/gpu/client.ts";

const registryPath = fileURLToPath(new URL("../../models.toml", import.meta.url));

const status = (over: Partial<Status>): Status => ({
  vram: { total_mb: 12288, free_mb: 12288, usable_free_mb: 12288 },
  resident: [],
  busy: false,
  queue_depth: 0,
  ...over,
});

test("readRegistryModels reads the repo registry", () => {
  const models = readRegistryModels(registryPath);
  assert.deepEqual(models.find((m) => m.name === "sdxl-turbo"), {
    name: "sdxl-turbo",
    backend: "diffusers",
    role: "image",
    vramMb: 7515,
  });
  assert.ok(models.length >= 3);
});

test("mergeModels marks residents loaded with measured size and idle", () => {
  const registry = readRegistryModels(registryPath);
  const merged = mergeModels(
    registry,
    status({
      resident: [
        { model: "sdxl-turbo", vram_mb: 7100, pinned: true, role: "image", idle_s: 12, load_ms: 4000 },
      ],
      recent: [
        { event: "model.loaded", at: "2026-10-22T10:00:00Z", model: "llama3.1:8b", load_ms: 900 },
      ],
    }),
  );
  assert.deepEqual(merged.map((m) => m.name), registry.map((m) => m.name));
  const sdxl = merged.find((m) => m.name === "sdxl-turbo")!;
  assert.deepEqual(
    [sdxl.loaded, sdxl.vramMb, sdxl.idleS, sdxl.loadMs, sdxl.pinned, sdxl.registryMb],
    [true, 7100, 12, 4000, true, 7515],
  );
  // Evicted since: not loaded, but its last load time is still known.
  const llama = merged.find((m) => m.name === "llama3.1:8b")!;
  assert.deepEqual([llama.loaded, llama.vramMb, llama.loadMs], [false, null, 900]);
});

test("mergeModels with no status knows nothing loaded or not", () => {
  const merged = mergeModels(readRegistryModels(registryPath), null);
  assert.ok(
    merged.every((m) => m.loaded === null && m.vramMb === null && m.idleS === null && m.loadMs === null),
  );
});
