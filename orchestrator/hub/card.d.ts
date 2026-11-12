/**
 * Types for `card.js`, which the browser loads as plain JavaScript. The
 * tsconfig does not check `.js`, so the tests import it through these.
 */

import type { Status } from "../src/gpu/client.ts";
import type { ModelEntry } from "../src/hub/models.ts";

export interface Segment {
  kind: "model" | "not_models" | "free" | "headroom";
  label: string;
  mb: number;
  pct: number;
  busy?: boolean;
  loadingPct?: number;
}

export interface SwapEvent {
  event: string;
  model?: string;
  [key: string]: unknown;
}

export interface CardState {
  loading: Record<string, number>;
  /** When each loading model's last `model.loading` arrived, epoch ms. */
  heard: Record<string, number>;
  stale: boolean;
  lastSwap: SwapEvent | null;
}

export function initialState(): CardState;
export function barSegments(status: Status, loading?: Record<string, number>): Segment[];
export const QUIET_MS: number;
export function applyEvent(state: CardState, event: SwapEvent, now?: number): CardState;
export function settleLoading(state: CardState, status: Status, now?: number): CardState;
export function describeSwap(e: SwapEvent): string;
export function lastSwapFrom(status: Status): SwapEvent | null;
export function predictEvictions(
  status: Status,
  models: ModelEntry[],
  target: Pick<ModelEntry, "name" | "registryMb">,
): string[];
