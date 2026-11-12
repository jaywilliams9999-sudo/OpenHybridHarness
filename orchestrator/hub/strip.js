/**
 * The top strip: this machine's GPU, drawn from `/api/card`, kept live by the
 * GPU service's events. Both pages carry it.
 *
 * Events move the bar between reads (a loading block fills, the swap line
 * changes); any model event marks the reading stale and the strip reads
 * `/api/card` again, as it does on every link change and every 10 s, so a
 * dropped event never leaves the bar wrong for long.
 */

import { applyEvent, barSegments, describeSwap, initialState, lastSwapFrom, settleLoading } from "./card.js";
import { el, fill } from "./dom.js";
import { debounce, getJson, onEvent, size } from "./api.js";

const EVERY_MS = 10_000;

export function mountStrip(root) {
  let state = initialState();
  let reading = null; // { health, status, at } -- the last good one
  let down = null; // the error, while the GPU service cannot be reached
  let inFlight = false;
  let again = false;

  const head = el("h2", { class: "eyebrow" }, "GPU");
  const gpu = el("span", { class: "strip-gpu" });
  const swap = el("p", { class: "strip-swap", "aria-live": "polite" });
  const bar = el("div", { class: "bar", role: "img" });
  const note = el("p", { class: "strip-note", role: "status" });
  fill(root, el("header", { class: "strip-head" }, el("div", { class: "strip-title" }, head, gpu), swap), bar, note);

  const draw = () => {
    root.classList.toggle("strip--down", down !== null);
    const health = reading?.health;
    fill(gpu, health ? `${health.gpu} · ${health.vram_total_mb.toLocaleString("en-US")} MB` : "");

    const last = state.lastSwap ?? (reading ? lastSwapFrom(reading.status) : null);
    fill(swap, last ? `${last.event === "model.loaded" ? "⤒" : "⤓"} ${describeSwap(last)}` : "");

    if (!reading) {
      bar.setAttribute("aria-label", "no reading yet");
      fill(bar, el("div", { class: "seg seg--void", style: "width:100%" }, down ? "The GPU service can't be reached" : "Reading the GPU…"));
    } else {
      const segments = barSegments(reading.status, state.loading);
      bar.setAttribute("style", `--gb:${(1024 * 100) / reading.status.vram.total_mb}%`);
      bar.setAttribute("aria-label", segments.map(words).join(", "));
      let nth = 0;
      fill(bar, segments.map((s) => segment(s, s.kind === "model" ? nth++ : 0)));
    }

    fill(
      note,
      down
        ? [
            "The GPU service can't be reached",
            reading ? ` · last good reading ${reading.at.toLocaleTimeString()}` : "",
            el("span", { class: "strip-error" }, ` · ${down}`),
          ]
        : "",
    );
  };

  const refresh = async () => {
    if (inFlight) {
      again = true;
      return;
    }
    inFlight = true;
    try {
      const card = await getJson("/api/card");
      if (card.reachable) {
        reading = { health: card.health, status: card.status, at: new Date() };
        down = null;
        // The reading now includes what any loading block was waiting on,
        // and shows which loads finished or quietly failed.
        state = settleLoading({ ...state, stale: false }, card.status);
      } else {
        down = card.error;
      }
    } catch (error) {
      down = error instanceof Error ? error.message : String(error);
    } finally {
      inFlight = false;
    }
    draw();
    if (again) {
      again = false;
      void refresh();
    }
  };
  const soon = debounce(refresh, 150);

  onEvent((e) => {
    if (e.event === "gpu.link") return void refresh();
    if (e.source !== "gpu") return;
    state = applyEvent(state, e);
    draw();
    if (state.stale) soon();
  });
  setInterval(refresh, EVERY_MS);
  draw();
  void refresh();
}

function words(s) {
  if (s.kind === "model") {
    if (s.loadingPct !== undefined) return `${s.label} loading ${s.loadingPct}%`;
    return `${s.label} ${size(s.mb)}${s.busy ? " in use" : ""}`;
  }
  if (s.kind === "headroom") return `safety headroom ${size(s.mb)}`;
  return `${s.label} ${size(s.mb)}`;
}

function segment(s, nth) {
  const style = `width:${s.pct}%`;
  const title = words(s);
  if (s.kind === "model" && s.loadingPct !== undefined) {
    return el(
      "div",
      { class: "seg seg--model seg--loading", style, title },
      el("div", { class: "seg-fill", style: `width:${s.loadingPct}%` }),
      el("span", { class: "seg-label" }, `${s.label} · loading ${s.loadingPct}%`),
    );
  }
  if (s.kind === "model") {
    return el(
      "div",
      { class: `seg seg--model seg--m${nth % 3}${s.busy ? " seg--busy" : ""}`, style, title },
      el("span", { class: "seg-label" }, s.label, el("span", { class: "seg-mb" }, ` · ${size(s.mb)}`), s.busy ? el("em", null, " · in use") : null),
    );
  }
  if (s.kind === "headroom") return el("div", { class: "seg seg--headroom", style, title });
  return el(
    "div",
    { class: `seg seg--${s.kind === "free" ? "free" : "other"}`, style, title },
    el("span", { class: "seg-label" }, `${s.label} ${size(s.mb)}`),
  );
}
