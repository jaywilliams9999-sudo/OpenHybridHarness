/**
 * Types for `dom.js`; see `card.d.ts` for why these exist. The tsconfig has
 * no DOM library, so nodes are opaque here.
 */

type Child = object | string | number | null | undefined | false | Child[];

export function el(tag: string, attrs: Record<string, unknown> | null, ...children: Child[]): object;
export function fill<T extends object>(node: T, ...children: Child[]): T;
