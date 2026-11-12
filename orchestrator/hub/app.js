/**
 * The main page: the strip, the goal box, the board and a goal's story.
 *
 * The board is re-read whole from `/api/board` -- on load, at most every
 * 250 ms while the Hub says it is changing, whenever the event stream
 * (re)connects, and every 30 s -- and redrawn; it keeps no state of its own,
 * so it is right after anything, a dropped stream included.
 */

import { el, fill } from "./dom.js";
import { duration, getJson, onEvent, post, usd } from "./api.js";
import { throttle } from "./pace.js";
import { mountStrip } from "./strip.js";

const SUMMARY_CHARS = 240;

mountStrip(document.getElementById("card"));

// -- the goal box -----------------------------------------------------------

const form = document.getElementById("start");
const goalInput = document.getElementById("goal");
const budgetInput = document.getElementById("budget");
const startButton = document.getElementById("start-go");
const startError = document.getElementById("start-error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  // Disabled for the whole round trip: a double click starts one goal.
  if (startButton.disabled) return;
  startButton.disabled = true;
  fill(startError);
  const error = await post("/api/goals", { goal: goalInput.value, budget_usd: Number(budgetInput.value) });
  startButton.disabled = false;
  if (error) {
    fill(startError, error);
    return;
  }
  goalInput.value = "";
  void refreshBoard();
});

goalInput.addEventListener("keydown", (event) => {
  // Enter starts; Shift+Enter is a new line.
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

// -- the board --------------------------------------------------------------

const columns = {
  working: document.getElementById("col-working"),
  needs_you: document.getElementById("col-needs"),
  done: document.getElementById("col-done"),
};
const EMPTY = {
  working: "Nothing is running. Start a goal above.",
  needs_you: "No goal is waiting on you.",
  done: "Finished goals land here.",
};

let board = null;
let boardError = null;
let boardInFlight = false;
let boardAgain = false;

/** Re-read the board; a call while one read is out becomes one more read after it. */
async function refreshBoard() {
  if (boardInFlight) {
    boardAgain = true;
    return;
  }
  boardInFlight = true;
  try {
    board = await getJson("/api/board");
    boardError = null;
  } catch (error) {
    boardError = error instanceof Error ? error.message : String(error);
  } finally {
    boardInFlight = false;
  }
  drawBoard();
  if (boardAgain) {
    boardAgain = false;
    void refreshBoard();
  }
}

function drawBoard() {
  for (const [column, list] of Object.entries(columns)) {
    const cards = board?.[column] ?? [];
    const section = list.closest(".column");
    fill(section.querySelector(".count"), cards.length ? String(cards.length) : "");
    if (column === "done") cards.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    fill(
      list,
      boardError && column === "working"
        ? el("p", { class: "empty error" }, `The board cannot be read: ${boardError}`)
        : cards.length
          ? cards.map((c) => cardFor(column, c))
          : el("p", { class: "empty" }, EMPTY[column]),
    );
  }
}

function since(iso) {
  return duration((Date.now() - Date.parse(iso)) / 1000);
}

function spend(card) {
  return el("span", { class: "spend" }, `${usd(card.spentUsd)} of ${usd(card.budgetUsd)}`);
}

/** Who is on the goal: the root's agent, then those it dispatched. */
function agents(card) {
  const names = [...new Set(card.agents.map((a) => a.agent))];
  return el("span", { class: "agents" }, card.root.agent, names.length ? ` → ${names.join(" · ")}` : "");
}

/** A card's frame: the goal (which opens the story), then its body. */
function frame(card, extra, ...body) {
  const open = () => openDetail(card.id);
  return el(
    "article",
    { class: `goal-card ${extra}`, onclick: (e) => !e.target.closest("button, a") && open() },
    el("button", { class: "goal-text", type: "button", onclick: open, title: "Open the goal's story" }, card.goal),
    ...body,
  );
}

function cardFor(column, card) {
  if (column === "working") {
    // A blocked root here is being resumed: its approval was just decided.
    const waiting = card.status === "blocked" ? el("p", { class: "meta warn" }, "resuming") : null;
    return frame(card, "", el("p", { class: "meta" }, agents(card), ` · ${since(card.createdAt)}`), waiting, el("p", { class: "meta" }, spend(card)));
  }
  if (column === "needs_you") return needsYou(card);
  return done(card);
}

function needsYou(card) {
  const a = card.approval;
  const error = el("p", { class: "inline-error", role: "alert" });
  const decide = (decision) => async (event) => {
    const buttons = event.target.closest(".decide").querySelectorAll("button");
    for (const b of buttons) b.disabled = true;
    fill(error);
    const problem = await post(`/api/approvals/${encodeURIComponent(a.id)}`, { decision });
    if (problem) {
      for (const b of buttons) b.disabled = false;
      fill(error, problem);
      return;
    }
    void refreshBoard();
  };
  return frame(
    card,
    "goal-card--needs",
    a.imageId ? el("img", { class: "proof", src: `/artifacts/${encodeURIComponent(a.imageId)}`, alt: `The image ${a.tool} would use` }) : null,
    el("p", { class: "asks" }, el("strong", null, a.tool), " asks", argLine(a.args)),
    el(
      "div",
      { class: "decide" },
      el("button", { class: "btn btn--yes", type: "button", onclick: decide("approved") }, "Approve"),
      el("button", { class: "btn btn--no", type: "button", onclick: decide("denied") }, "Deny"),
    ),
    error,
    el("p", { class: "meta" }, spend(card)),
  );
}

/** A tool's arguments in one short line, for a person deciding. */
function argLine(args) {
  if (args === null || typeof args !== "object") return "";
  const parts = Object.entries(args).map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  const line = parts.join(" · ");
  return el("span", { class: "args" }, ` — ${line.length > 160 ? `${line.slice(0, 159)}…` : line}`);
}

/** Why a stopped goal stopped, as one line. */
function stopLine(card) {
  if (card.reason?.startsWith("stopped:")) return `S${card.reason.slice(1)}`;
  if (card.reason === "child") return "Stopped: an agent it waited on stopped";
  return `Stopped: ${card.reason ?? "blocked"}`;
}

function done(card) {
  // A goal stopped blocked with nothing to decide is offered Run again too.
  const stopped = card.status === "blocked";
  const failed = card.status === "failed" || stopped;
  let result;
  if (card.images.length) {
    result = el(
      "div",
      { class: "proofs" },
      card.images.slice(0, 1).map((id) => el("img", { class: "proof", src: `/artifacts/${encodeURIComponent(id)}`, alt: "An image this goal made", loading: "lazy" })),
      card.images.length > 1 ? el("span", { class: "more" }, `+${card.images.length - 1} more`) : null,
    );
  } else if (card.summary) {
    const s = card.summary;
    result = el("p", { class: "summary" }, s.length > SUMMARY_CHARS ? `${s.slice(0, SUMMARY_CHARS)}…` : s);
  } else {
    result = el("p", { class: "summary summary--none" }, "no summary recorded");
  }

  let retry = null;
  if (failed) {
    const error = el("p", { class: "inline-error", role: "alert" });
    const again = async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      fill(error);
      const problem = await post(`/api/goals/${encodeURIComponent(card.id)}/again`);
      if (problem) {
        button.disabled = false;
        fill(error, problem);
        return;
      }
      void refreshBoard();
    };
    retry = [
      el("p", { class: "reason" }, stopped ? stopLine(card) : `Failed: ${card.reason ?? "no reason recorded"}`),
      el("button", { class: "btn btn--again", type: "button", onclick: again }, `Run again · ${usd(card.budgetUsd)}`),
      error,
    ];
  }
  return frame(card, failed ? "goal-card--failed" : "", result, retry, el("p", { class: "meta" }, agents(card), " · ", el("span", { class: "spend" }, usd(card.spentUsd))));
}

// -- a goal's story ---------------------------------------------------------

const dialog = document.getElementById("detail");
const dialogBody = document.getElementById("detail-body");
document.getElementById("detail-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  // A click on the backdrop, outside the sheet, closes it.
  if (event.target === dialog) dialog.close();
});

async function openDetail(id) {
  fill(dialogBody, el("p", { class: "empty" }, "Reading the goal…"));
  if (!dialog.open) dialog.showModal();
  try {
    drawDetail(await getJson(`/api/goals/${encodeURIComponent(id)}`));
  } catch (error) {
    fill(dialogBody, el("p", { class: "empty error" }, error instanceof Error ? error.message : String(error)));
  }
}

function parse(json) {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

function drawDetail({ card, runs }) {
  fill(
    dialogBody,
    el("p", { class: "eyebrow" }, `${card.status} · ${usd(card.spentUsd)} of ${usd(card.budgetUsd)} · started ${new Date(card.createdAt).toLocaleString()}`),
    el("h2", { class: "detail-goal" }, card.goal),
    card.reason ? el("p", { class: "reason" }, `Reason: ${card.reason}`) : null,
    el("h3", { class: "eyebrow" }, "Summary"),
    el("p", { class: card.summary ? "summary" : "summary summary--none" }, card.summary ?? "no summary recorded"),
    card.images.length
      ? [el("h3", { class: "eyebrow" }, "Images"), card.images.map((id) => el("img", { class: "proof proof--full", src: `/artifacts/${encodeURIComponent(id)}`, alt: "An image this goal made" }))]
      : null,
    el("h3", { class: "eyebrow" }, "Agents"),
    runs.map(runBlock),
  );
}

function runBlock(p) {
  return el(
    "section",
    { class: "run", style: `--depth:${p.depth}` },
    el("h4", { class: "run-head" }, el("span", { class: "agent" }, p.agent), ` · ${p.status} · ${usd(p.spentUsd)}`),
    p.toolCalls.length
      ? el(
          "ol",
          { class: "calls" },
          p.toolCalls.map((c) => {
            const evicted = parse(c.result_json)?.evicted;
            return el(
              "li",
              null,
              el("span", { class: "call-tool" }, c.tool),
              el("span", { class: `tier tier--${c.tier}` }, c.tier),
              el("span", { class: "call-ms" }, `${(c.ms / 1000).toFixed(1)} s`),
              Array.isArray(evicted) && evicted.length
                ? el("span", { class: "call-evicted" }, `unloaded ${evicted.map((e) => e.model).join(", ")}`)
                : null,
            );
          }),
        )
      : el("p", { class: "meta" }, "no tool calls"),
    p.approvals.map((a) =>
      el("p", { class: `approval approval--${a.status}` }, `${a.tool}: ${a.status}${a.decided_at ? ` at ${new Date(a.decided_at).toLocaleString()}` : ""}`),
    ),
    p.images.length ? el("p", { class: "meta" }, `${p.images.length} image${p.images.length === 1 ? "" : "s"}`) : null,
  );
}

// -- live -------------------------------------------------------------------

const boardSoon = throttle(refreshBoard, 250);
onEvent((e) => {
  // `gpu.link` is also the first thing a (re)connected stream says, so a
  // board that missed changes while the stream was down is read again.
  if (e.event === "board.changed" || e.event === "gpu.link") boardSoon();
});
// The slow poll also moves the Working column's clocks.
setInterval(refreshBoard, 30_000);
void refreshBoard();
