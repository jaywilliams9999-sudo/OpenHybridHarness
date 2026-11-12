/**
 * Per-agent routing: the agent file's `model` line decides where its runs go.
 *
 * There is no separate provider field. A name defined in `models.toml` runs on
 * the local GPU; a `claude-*` name runs on Claude. An agent with no `model`
 * line runs locally on the registry's `role = "code"` model. A Claude-placed
 * agent on a machine with no Anthropic key still runs -- locally -- and the
 * run is marked `local-no-key` so that substitution is never silent.
 */

/** Where a run was placed when it started. Stored on the run row. */
export type Placement = "local" | "claude" | "local-no-key";

/**
 * Where a run actually ran: its placement, or `local-fallback` when a
 * Claude-placed run hit the spending wall and finished on the local model.
 */
export type RanAs = Placement | "local-fallback";

export interface Route {
  readonly placement: Placement;
  /** Absent means the local default model (`role = "code"`). */
  readonly model?: string;
}

export class RouteError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RouteError";
  }
}

export function routeAgent(
  agentFile: string,
  model: string | undefined,
  opts: { localModels: ReadonlySet<string>; hasKey: boolean },
): Route {
  if (model === undefined) return { placement: "local" };
  if (opts.localModels.has(model)) return { placement: "local", model };
  if (/^claude-/.test(model)) {
    // Without a key the Claude model name means nothing to the local
    // provider, so it is dropped and the registry default answers instead.
    return opts.hasKey ? { placement: "claude", model } : { placement: "local-no-key" };
  }
  throw new RouteError(
    `${agentFile}: model "${model}" is not in models.toml and is not a claude-* model.`,
  );
}

/**
 * The route for a run being resumed. The run row's placement wins over where
 * the agent file would send a fresh run today, so a resume never moves a run
 * somewhere it did not start:
 *
 * - `local` and `local-no-key` stay local, even if a key has been added since
 *   (no surprise spend). `local` keeps the agent's local model if it still has one.
 * - `claude` stays on Claude while a key exists. With no key it runs locally
 *   and becomes `local-no-key`: the caller records that on the run row, so the
 *   Ledger says why the rest of the run was local.
 */
export function resumeRoute(stored: Placement, today: Route, hasKey: boolean): Route {
  if (stored === "local-no-key") return { placement: "local-no-key" };
  if (stored === "local") {
    return today.placement === "local" ? today : { placement: "local" };
  }
  if (!hasKey) return { placement: "local-no-key" };
  return today.placement === "claude" ? today : { placement: "claude" };
}
