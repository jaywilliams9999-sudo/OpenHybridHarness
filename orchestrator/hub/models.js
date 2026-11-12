/**
 * The Models page: every registry model, what the GPU service holds, Load and Unload.
 *
 * Before a load that would push something out, the page reads the card
 * afresh, says what the load probably unloads, and asks; the Hub then says
 * how the load or unload ended -- an error, or what it actually unloaded.
 * The table is re-read every 10 s as well, so "last used" keeps moving. With
 * no GPU service reading a model's state is unknown, and nothing can be loaded.
 */

import { applyEvent, initialState, predictEvictions, settleLoading } from "./card.js";
import { el, fill } from "./dom.js";
import { debounce, duration, getJson, onEvent, post, size } from "./api.js";
import { mountStrip } from "./strip.js";

mountStrip(document.getElementById("card"));

const body = document.getElementById("models-body");
const pageError = document.getElementById("models-error");

let models = [];
let status = null; // the GPU service's /status, when it can be reached
let card = initialState(); // loads in progress, from the GPU service's events
const confirming = new Map(); // model -> the models a load probably unloads
const errors = new Map(); // model -> the error its last action got
const unloaded = new Map(); // model -> what its last load actually unloaded
const asked = new Set(); // models with a request in flight

async function refresh() {
  try {
    const [list, reading] = await Promise.all([getJson("/api/models"), getJson("/api/card")]);
    models = list;
    status = reading.reachable ? reading.status : null;
    if (status) card = settleLoading(card, status);
    fill(pageError);
  } catch (error) {
    fill(pageError, error instanceof Error ? error.message : String(error));
  }
  draw();
}

async function act(model, action) {
  confirming.delete(model.name);
  errors.delete(model.name);
  unloaded.delete(model.name);
  asked.add(model.name);
  draw();
  const problem = await post(`/api/models/${encodeURIComponent(model.name)}/${action}`);
  asked.delete(model.name);
  if (problem) errors.set(model.name, problem);
  draw();
}

/** Load, after a fresh reading says whether it would unload anything. */
async function load(model) {
  errors.delete(model.name);
  unloaded.delete(model.name);
  asked.add(model.name);
  draw();
  let out = [];
  try {
    const reading = await getJson("/api/card");
    if (reading.reachable) {
      status = reading.status;
      out = predictEvictions(status, models, model);
    }
  } catch {
    // No reading, no prediction: the load itself will say what it can.
  }
  asked.delete(model.name);
  if (out.length) {
    confirming.set(model.name, out);
    draw();
    return;
  }
  void act(model, "load");
}

function stateCell(m) {
  if (m.name in card.loading) {
    const pct = card.loading[m.name];
    return el(
      "td",
      null,
      el("span", { class: "progress", role: "progressbar", "aria-valuenow": pct, "aria-valuemin": 0, "aria-valuemax": 100 }, el("span", { class: "progress-fill", style: `width:${pct}%` })),
      ` loading ${pct}%`,
    );
  }
  if (m.loaded === null) return el("td", { class: "muted" }, "unknown");
  return el(
    "td",
    null,
    el("span", { class: `dot ${m.loaded ? "dot--on" : "dot--off"}`, "aria-hidden": "true" }),
    m.loaded ? (m.pinned ? "loaded · pinned" : "loaded") : "not loaded",
  );
}

function actionCell(m) {
  // Unknown state: neither Load nor Unload is a safe guess.
  const busy = m.loaded === null || asked.has(m.name) || m.name in card.loading;
  const out = confirming.get(m.name);
  if (out) {
    return el(
      "td",
      { class: "act" },
      el("p", { class: "confirm" }, `Probably unloads: ${out.join(", ")}`),
      el("button", { class: "btn btn--small btn--start", type: "button", onclick: () => act(m, "load") }, "Load anyway"),
      el("button", { class: "btn btn--small btn--ghost", type: "button", onclick: () => (confirming.delete(m.name), draw()) }, "Cancel"),
    );
  }
  return el(
    "td",
    { class: "act" },
    m.loaded
      ? el("button", { class: "btn btn--small btn--ghost", type: "button", disabled: busy, onclick: () => act(m, "unload") }, "Unload")
      : el("button", { class: "btn btn--small", type: "button", disabled: busy, onclick: () => load(m) }, "Load"),
    errors.has(m.name) ? el("p", { class: "inline-error", role: "alert" }, errors.get(m.name)) : null,
    unloaded.has(m.name) ? el("p", { class: "meta" }, `unloaded: ${unloaded.get(m.name).join(", ")}`) : null,
  );
}

function draw() {
  if (!models.length) {
    fill(body, el("tr", null, el("td", { colspan: 8, class: "empty" }, "Reading the registry…")));
    return;
  }
  fill(
    body,
    models.map((m) =>
      el(
        "tr",
        { class: m.loaded ? "is-loaded" : "" },
        el("th", { scope: "row", class: "model-name" }, m.name),
        el("td", null, m.backend),
        el("td", null, m.role ?? "—"),
        el(
          "td",
          { class: "num", title: m.vramMb !== null ? "measured on the card" : "the registry's figure" },
          size(m.vramMb ?? m.registryMb),
          m.vramMb === null ? el("span", { class: "muted" }, " est.") : null,
        ),
        stateCell(m),
        el("td", { class: "num" }, m.loadMs === null ? "—" : `${m.loadMs < 500 ? "<1" : Math.round(m.loadMs / 1000)} s`),
        el("td", { class: "num" }, m.idleS === null ? "—" : `${duration(m.idleS)} ago`),
        actionCell(m),
      ),
    ),
  );
}

const soon = debounce(refresh, 150);
onEvent((e) => {
  if (e.source === "hub" && e.event === "model.outcome" && typeof e.model === "string") {
    if (!e.ok) errors.set(e.model, e.error ?? `the ${e.action} failed`);
    else if (e.evicted?.length) unloaded.set(e.model, e.evicted.map((x) => x.model));
    draw();
    soon();
    return;
  }
  if (e.source !== "gpu" || typeof e.event !== "string") return;
  card = applyEvent(card, e);
  draw();
  if (e.event === "model.loaded" || e.event === "model.unloaded" || e.event === "gpu.link") soon();
});

setInterval(refresh, 10_000);
draw();
void refresh();
