/** The card's pure parts: the memory bar, the swap line and the eviction guess. */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  applyEvent,
  barSegments,
  describeSwap,
  initialState,
  lastSwapFrom,
  predictEvictions,
  settleLoading,
} from "../hub/card.js";
import type { Status } from "../src/gpu/client.ts";
import type { ModelEntry } from "../src/hub/models.ts";

const status = (over: Partial<Status>): Status => ({
  vram: { total_mb: 12288, free_mb: 12288, headroom_mb: 512, usable_free_mb: 11776 },
  resident: [],
  busy: false,
  queue_depth: 0,
  ...over,
});

const coder = { model: "qwen2.5-coder", vram_mb: 4900, pinned: false, role: "coder", idle_s: 30, load_ms: 4000 };

const sum = (xs: Array<{ pct: number }>): number => xs.reduce((a, s) => a + s.pct, 0);

const entry = (name: string, registryMb: number): ModelEntry => ({
  name,
  backend: "ollama",
  role: null,
  registryMb,
  loaded: false,
  vramMb: null,
  idleS: null,
  loadMs: null,
  pinned: false,
});

test("barSegments for one resident model sums to 100", () => {
  const segs = barSegments(
    status({
      vram: { total_mb: 12288, free_mb: 7388, headroom_mb: 512, usable_free_mb: 6876 },
      resident: [coder],
    }),
  );
  assert.deepEqual(
    segs.map((s) => [s.kind, s.label, s.mb]),
    [
      ["model", "qwen2.5-coder", 4900],
      ["free", "free", 6876],
      ["headroom", "headroom", 512],
    ],
  );
  assert.ok(Math.abs(sum(segs) - 100) <= 0.5);
});

test("the busy model is marked", () => {
  const segs = barSegments(
    status({
      vram: { total_mb: 12288, free_mb: 7388, headroom_mb: 512, usable_free_mb: 6876 },
      resident: [coder],
      busy: true,
      busy_model: "qwen2.5-coder",
    }),
  );
  assert.equal(segs[0]?.busy, true);
});

test("not-models appears only when there is unowned memory", () => {
  const clean = barSegments(
    status({
      vram: { total_mb: 12288, free_mb: 7388, headroom_mb: 512, usable_free_mb: 6876 },
      resident: [coder],
    }),
  );
  assert.equal(clean.some((s) => s.kind === "not_models"), false);

  const game = barSegments(
    status({
      vram: { total_mb: 12288, free_mb: 5388, headroom_mb: 512, usable_free_mb: 4876 },
      resident: [coder],
    }),
  );
  const other = game.find((s) => s.kind === "not_models");
  assert.equal(other?.mb, 2000);
  assert.equal(other?.label, "not models");
  assert.ok(Math.abs(sum(game) - 100) <= 0.5);
});

test("describeSwap copy for loaded, idle, evicted, explicit", () => {
  assert.equal(describeSwap({ event: "model.loaded", model: "qwen2.5-coder", load_ms: 4200 }), "qwen2.5-coder loaded · 4 s");
  assert.equal(describeSwap({ event: "model.loaded", model: "coder", load_ms: 300 }), "coder loaded · <1 s");
  assert.equal(
    describeSwap({ event: "model.unloaded", model: "sdxl-turbo", reason: "idle", idle_s: 1800 }),
    "sdxl-turbo unloaded · idle 30 min",
  );
  assert.equal(
    describeSwap({ event: "model.unloaded", model: "coder", reason: "evicted", for_model: "sdxl-turbo" }),
    "coder unloaded to make room for sdxl-turbo",
  );
  assert.equal(describeSwap({ event: "model.unloaded", model: "coder", reason: "explicit" }), "coder unloaded");
});

test("applyEvent turns a loading event into a loading block", () => {
  const state = applyEvent(initialState(), { event: "model.loading", model: "sdxl-turbo", pct: 40 });
  assert.equal(state.stale, false);
  const segs = barSegments(
    status({
      vram: { total_mb: 12288, free_mb: 4388, headroom_mb: 512, usable_free_mb: 3876 },
      resident: [coder],
    }),
    state.loading,
  );
  const loading = segs.find((s) => s.label === "sdxl-turbo");
  assert.equal(loading?.kind, "model");
  assert.equal(loading?.loadingPct, 40);
  // The memory the load has taken so far is its block, not "not models".
  assert.equal(loading?.mb, 3000);
  assert.equal(segs.some((s) => s.kind === "not_models"), false);

  const done = applyEvent(state, { event: "model.loaded", model: "sdxl-turbo", vram_mb: 7100, load_ms: 9000 });
  assert.deepEqual(done.loading, {});
  assert.equal(done.stale, true);
  assert.equal(done.lastSwap?.model, "sdxl-turbo");
});

test("predictEvictions picks the idlest unpinned model first", () => {
  const s = status({
    vram: { total_mb: 12288, free_mb: 1500, headroom_mb: 512, usable_free_mb: 988 },
    resident: [
      { model: "llama3.1:8b", vram_mb: 5000, pinned: false, role: "chat", idle_s: 60 },
      { model: "qwen2.5-coder", vram_mb: 4900, pinned: false, role: "coder", idle_s: 900 },
      { model: "embed", vram_mb: 800, pinned: true, role: "embed", idle_s: 5000 },
    ],
  });
  const models = [entry("sdxl-turbo", 7515), entry("llama3.1:8b", 5000), entry("qwen2.5-coder", 4900)];
  assert.deepEqual(predictEvictions(s, models, entry("sdxl-turbo", 5000)), ["qwen2.5-coder"]);
  assert.deepEqual(predictEvictions(s, models, entry("sdxl-turbo", 7515)), ["qwen2.5-coder", "llama3.1:8b"]);
});

test("predictEvictions is empty when it fits", () => {
  const s = status({ resident: [coder], vram: { total_mb: 12288, free_mb: 7388, headroom_mb: 512, usable_free_mb: 6876 } });
  assert.deepEqual(predictEvictions(s, [], entry("llama3.1:8b", 5000)), []);
});

test("lastSwapFrom reads the newest recent event", () => {
  const s = status({
    recent: [
      { event: "model.loaded", at: "2026-10-22T10:00:00+00:00", model: "coder", load_ms: 4000 },
      { event: "model.unloaded", at: "2026-10-22T10:30:00+00:00", model: "coder", reason: "idle" },
    ],
  });
  assert.equal(lastSwapFrom(s)?.event, "model.unloaded");
  assert.equal(lastSwapFrom(status({})), null);
});

test("a load the GPU service has gone quiet on is dropped once a reading shows it absent", () => {
  const t0 = 1_000_000;
  const state = applyEvent(initialState(), { event: "model.loading", model: "sdxl-turbo", pct: 40 }, t0);
  const absent = status({ resident: [coder] });

  // Still within the window: the load may just be slow.
  assert.deepEqual(settleLoading(state, absent, t0 + 10_000).loading, { "sdxl-turbo": 40 });
  // Quiet for longer than the window, and not resident: it failed.
  assert.deepEqual(settleLoading(state, absent, t0 + 31_000).loading, {});
  // Resident: the block is the model itself now, whatever the clock says.
  const present = status({ resident: [{ ...coder, model: "sdxl-turbo" }] });
  assert.deepEqual(settleLoading(state, present, t0 + 1_000).loading, {});
});
