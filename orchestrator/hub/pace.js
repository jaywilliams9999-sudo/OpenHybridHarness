/**
 * Pacing for refetches driven by events. No DOM: the tests run it in Node.
 */

/**
 * Call `fn` now, then at most once per `ms` while calls keep coming, always
 * once more after the last of them. Unlike a debounce that restarts on every
 * call, a steady stream of calls cannot hold `fn` off.
 */
export function throttle(fn, ms) {
  let last = -Infinity;
  let timer;
  const run = () => {
    timer = undefined;
    last = Date.now();
    fn();
  };
  return () => {
    if (timer !== undefined) return;
    const wait = last + ms - Date.now();
    if (wait <= 0) run();
    else timer = setTimeout(run, wait);
  };
}
