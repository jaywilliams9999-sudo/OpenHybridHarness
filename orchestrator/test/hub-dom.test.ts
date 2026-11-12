/** The page builds its DOM from text, never from markup. */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

import { el } from "../hub/dom.js";

const hubDir = fileURLToPath(new URL("../hub/", import.meta.url));

interface FakeNode {
  tag?: string;
  text?: string;
  attrs: Record<string, string>;
  children: FakeNode[];
  listeners: string[];
}

/** Just enough `document` for `el`, recording what it was asked to do. */
function fakeDocument() {
  const calls: string[] = [];
  const node = (over: Partial<FakeNode>): FakeNode & Record<string, unknown> => ({
    attrs: {},
    children: [],
    listeners: [],
    ...over,
    setAttribute(name: string, value: string) {
      this.attrs[name] = value;
    },
    appendChild(child: FakeNode) {
      this.children.push(child);
      return child;
    },
    addEventListener(name: string) {
      this.listeners.push(name);
    },
  });
  return {
    calls,
    document: {
      createElement(tag: string) {
        calls.push(`createElement ${tag}`);
        return node({ tag });
      },
      createTextNode(text: string) {
        calls.push(`createTextNode ${text}`);
        return node({ text });
      },
    },
  };
}

test("el puts a goal containing markup in as text", () => {
  const fake = fakeDocument();
  const saved = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = fake.document;
  try {
    const goal = `<img src=x onerror="alert(1)">make a header`;
    const node = el("p", { class: "goal", onclick: () => {} }, goal, null, ["!"]) as unknown as FakeNode;
    assert.equal(node.tag, "p");
    assert.equal(node.attrs["class"], "goal");
    assert.deepEqual(node.listeners, ["click"]);
    assert.deepEqual(
      node.children.map((c) => c.text),
      [goal, "!"],
    );
    assert.deepEqual(fake.calls, ["createElement p", `createTextNode ${goal}`, "createTextNode !"]);
  } finally {
    (globalThis as { document?: unknown }).document = saved;
  }
});

test("no file in orchestrator/hub uses innerHTML", () => {
  for (const name of readdirSync(hubDir)) {
    const text = readFileSync(join(hubDir, name), "utf8");
    assert.equal(/innerHTML|outerHTML|insertAdjacentHTML/.test(text), false, `${name} builds markup from strings`);
  }
});
