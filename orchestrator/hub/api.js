/**
 * The page's side of the Hub's routes. Reads are plain GETs; every request
 * that changes something is a JSON POST carrying the secret the Hub put in
 * this page, and the browser adds the Origin the Hub checks.
 */

function secret() {
  return document.querySelector('meta[name="hub-secret"]')?.getAttribute("content") ?? "";
}

/** A GET route's JSON; throws with the Hub's own error text. */
export async function getJson(path) {
  const res = await fetch(path, { headers: { accept: "application/json" } });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.error ?? `the Hub answered ${res.status}`);
  return body;
}

/** A state-changing request. Resolves to null when accepted, else the error to show. */
export async function post(path, body = {}) {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json", "x-hub-secret": secret() },
      body: JSON.stringify(body),
    });
    if (res.ok) return null;
    // The secret is per process: a page from before a restart holds a dead one.
    if (res.status === 403) return "The Hub restarted — reload this page.";
    const answer = await res.json().catch(() => null);
    return answer?.error ?? `the Hub answered ${res.status}`;
  } catch {
    return "the Hub cannot be reached";
  }
}

/** The events the page listens for; `op.*` progress is not drawn. */
const EVENTS = [
  "model.loading",
  "model.loaded",
  "model.unloaded",
  "vram.changed",
  "gpu.busy",
  "gpu.idle",
  "gpu.link",
  "board.changed",
  "model.outcome",
];

let source = null;
const listeners = new Set();

/** Every relayed event, parsed, to `listener`. One stream per page. */
export function onEvent(listener) {
  listeners.add(listener);
  if (source === null) {
    source = new EventSource("/api/events");
    for (const name of EVENTS) {
      source.addEventListener(name, (message) => {
        let data;
        try {
          data = JSON.parse(message.data);
        } catch {
          return;
        }
        for (const l of listeners) l(data);
      });
    }
  }
}

/** Call `fn` once things have been quiet for `ms`. */
export function debounce(fn, ms) {
  let timer;
  return () => {
    clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
}

/** "4.9 GB" from megabytes; under a gigabyte stays in MB. */
export function size(mb) {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

export function usd(n) {
  return `$${n.toFixed(2)}`;
}

/** A duration in seconds, in words: "40 s", "12 min", "3.5 h". */
export function duration(seconds) {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds / 360) / 10} h`;
}
