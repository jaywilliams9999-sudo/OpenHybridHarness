/**
 * The card's arithmetic: what the memory bar shows, what the swap line says,
 * and which models a load would probably push out.
 *
 * Pure functions over the GPU service's `/status` and its events -- no DOM here, so
 * the tests run them in Node and the page only draws what they return.
 */

/** What the strip knows between reads of `/api/card`. */
export function initialState() {
  return { loading: {}, heard: {}, stale: false, lastSwap: null };
}

/** How long a load may go without a `model.loading` before it is presumed failed. */
export const QUIET_MS = 30_000;

/**
 * The bar, left to right: loaded models, the used memory no model owns, free,
 * then the safety headroom. `loading` maps a model being loaded to its
 * percentage; until it is resident, the memory it has taken so far is its
 * block rather than "not models".
 */
export function barSegments(status, loading = {}) {
  const { total_mb, free_mb } = status.vram;
  const segments = status.resident.map((r) => ({
    kind: "model",
    label: r.model,
    mb: r.vram_mb,
    pct: 0,
    ...(r.model === status.busy_model ? { busy: true } : {}),
  }));

  let unowned = total_mb - free_mb - status.resident.reduce((a, r) => a + r.vram_mb, 0);
  for (const [model, pct] of Object.entries(loading)) {
    if (status.resident.some((r) => r.model === model)) continue;
    const mb = Math.max(0, unowned);
    unowned -= mb;
    segments.push({ kind: "model", label: model, mb, pct: 0, loadingPct: pct });
  }
  if (unowned > 0) segments.push({ kind: "not_models", label: "not models", mb: unowned, pct: 0 });

  const headroom = Math.min(status.vram.headroom_mb ?? 0, free_mb);
  if (free_mb - headroom > 0) segments.push({ kind: "free", label: "free", mb: free_mb - headroom, pct: 0 });
  if (headroom > 0) segments.push({ kind: "headroom", label: "headroom", mb: headroom, pct: 0 });

  // Against what the segments add up to, not `total_mb`: a reading taken
  // mid-change can disagree with itself, and the bar must still fill.
  const whole = segments.reduce((a, s) => a + s.mb, 0);
  for (const s of segments) s.pct = whole > 0 ? (s.mb * 100) / whole : 0;
  return segments;
}

/** Which events mean the last `/status` read is out of date. */
const STALE = new Set(["model.loaded", "model.unloaded", "vram.changed", "gpu.busy", "gpu.idle"]);

/** The strip's state after one GPU service event, heard at `now`. */
export function applyEvent(state, event, now = Date.now()) {
  const next = { ...state, loading: { ...state.loading }, heard: { ...state.heard } };
  const model = typeof event.model === "string" ? event.model : null;
  if (event.event === "model.loading" && model !== null) {
    next.loading[model] = typeof event.pct === "number" ? event.pct : 0;
    next.heard[model] = now;
  }
  if (event.event === "model.loaded" || event.event === "model.unloaded") {
    if (model !== null) {
      delete next.loading[model];
      delete next.heard[model];
    }
    next.lastSwap = event;
  }
  if (STALE.has(event.event)) next.stale = true;
  return next;
}

/**
 * The state after a fresh `/status` reading: a load whose model is now
 * resident is finished, and one the GPU service has said nothing about for
 * `QUIET_MS` and that is still not resident has failed -- a failed load
 * sends no `model.loaded`, so without this its block would stay forever.
 */
export function settleLoading(state, status, now = Date.now()) {
  const next = { ...state, loading: { ...state.loading }, heard: { ...state.heard } };
  for (const model of Object.keys(next.loading)) {
    const resident = status.resident.some((r) => r.model === model);
    if (resident || now - (next.heard[model] ?? 0) > QUIET_MS) {
      delete next.loading[model];
      delete next.heard[model];
    }
  }
  return next;
}

/** A load or unload, in words. */
export function describeSwap(e) {
  if (e.event === "model.loaded") {
    if (typeof e.load_ms !== "number") return `${e.model} loaded`;
    const s = e.load_ms < 500 ? "<1" : String(Math.round(e.load_ms / 1000));
    return `${e.model} loaded · ${s} s`;
  }
  if (e.reason === "idle") {
    // The GPU service's idle unload does not say how long the model sat unused.
    if (typeof e.idle_s !== "number") return `${e.model} unloaded · idle`;
    return `${e.model} unloaded · idle ${Math.round(e.idle_s / 60)} min`;
  }
  if (e.reason === "evicted" && typeof e.for_model === "string") {
    return `${e.model} unloaded to make room for ${e.for_model}`;
  }
  return `${e.model} unloaded`;
}

/** The GPU service's own newest load or unload, for a strip that has not seen one. */
export function lastSwapFrom(status) {
  let newest = null;
  for (const e of status.recent ?? []) {
    if (newest === null || e.at >= newest.at) newest = e;
  }
  return newest;
}

/**
 * What loading `target` would probably unload: the GPU service evicts unpinned
 * models least recently used first until the load fits. A guess -- the GPU service
 * does not publish its plan -- so the page says "probably".
 */
export function predictEvictions(status, models, target) {
  const need = target.registryMb;
  let room = status.vram.usable_free_mb;
  if (room >= need) return [];
  const idle = (r) => r.idle_s ?? models.find((m) => m.name === r.model)?.idleS ?? 0;
  const candidates = status.resident
    .filter((r) => !r.pinned && r.model !== target.name)
    .sort((a, b) => idle(b) - idle(a));
  const out = [];
  for (const r of candidates) {
    out.push(r.model);
    room += r.vram_mb;
    if (room >= need) break;
  }
  return out;
}
