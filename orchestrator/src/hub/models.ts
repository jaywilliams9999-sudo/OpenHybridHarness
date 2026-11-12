/** The model list: the registry's models, marked with what the GPU service holds. */

import { readFileSync } from "node:fs";

import { parseToml } from "../config/toml.ts";
import type { Status } from "../gpu/client.ts";

export interface ModelEntry {
  name: string;
  backend: string;
  role: string | null;
  registryMb: number;
  /** Null when the GPU service could not be read: nobody knows. */
  loaded: boolean | null;
  vramMb: number | null;
  idleS: number | null;
  loadMs: number | null;
  pinned: boolean;
}

export function readRegistryModels(
  path: string,
): Array<{ name: string; backend: string; role: string | null; vramMb: number }> {
  const doc = parseToml(readFileSync(path, "utf8"));
  const models = (doc.models ?? {}) as Record<string, Record<string, unknown>>;
  return Object.entries(models).map(([name, table]) => ({
    name,
    backend: String(table.backend),
    role: typeof table.role === "string" ? table.role : null,
    vramMb: Number(table.vram_mb),
  }));
}

export function mergeModels(
  registry: ReturnType<typeof readRegistryModels>,
  status: Status | null,
): ModelEntry[] {
  return registry.map((model) => {
    const resident = status?.resident.find((r) => r.model === model.name);
    let loadMs = resident?.load_ms ?? null;
    if (loadMs === null && status?.recent) {
      // The newest load for this model, for one the GPU service has since evicted.
      let newest: { at: string; load_ms: unknown } | undefined;
      for (const e of status.recent) {
        if (e.event !== "model.loaded" || e.model !== model.name) continue;
        if (!newest || e.at >= newest.at) newest = { at: e.at, load_ms: e.load_ms };
      }
      if (typeof newest?.load_ms === "number") loadMs = newest.load_ms;
    }
    return {
      name: model.name,
      backend: model.backend,
      role: model.role,
      registryMb: model.vramMb,
      loaded: status === null ? null : resident !== undefined,
      vramMb: resident?.vram_mb ?? null,
      idleS: resident?.idle_s ?? null,
      loadMs,
      pinned: resident?.pinned ?? false,
    };
  });
}
