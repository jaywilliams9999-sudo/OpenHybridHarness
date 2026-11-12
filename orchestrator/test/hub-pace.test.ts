/** The page's pacing: a burst of events refetches promptly and never starves. */

import { test, mock } from "node:test";
import assert from "node:assert/strict";

import { throttle } from "../hub/pace.js";

test("throttle runs at once, then at most once per window, trailing the last call", () => {
  mock.timers.enable({ apis: ["setTimeout", "Date"] });
  try {
    let runs = 0;
    const call = throttle(() => runs++, 250);
    call();
    assert.equal(runs, 1);
    call();
    call();
    assert.equal(runs, 1);
    mock.timers.tick(250);
    assert.equal(runs, 2);
    mock.timers.tick(1000);
    assert.equal(runs, 2);
  } finally {
    mock.timers.reset();
  }
});

test("throttle keeps running while calls never stop", () => {
  mock.timers.enable({ apis: ["setTimeout", "Date"] });
  try {
    let runs = 0;
    const call = throttle(() => runs++, 250);
    // A call every 100 ms for 2 s: a resetting debounce would not run until
    // they stop. (The mock clock fires a timer at the end of the tick that
    // passes it, so here a run lands every 300 ms rather than every 250.)
    for (let t = 0; t < 2000; t += 100) {
      call();
      mock.timers.tick(100);
    }
    assert.ok(runs >= 6, `ran ${runs} times`);
  } finally {
    mock.timers.reset();
  }
});
