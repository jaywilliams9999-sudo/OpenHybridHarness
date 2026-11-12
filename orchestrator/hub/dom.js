/**
 * Building the page's DOM. Everything a goal, a summary or the GPU service says
 * reaches the page as a text node -- never parsed as markup -- because those
 * strings come from agents and an agent's output is not trusted on this
 * origin, where the page's secret lives.
 *
 * No DOM access at import time: the tests load this in Node.
 */

/**
 * `el("p", { class: "goal", onclick: fn }, "text", child, [more])`: attributes
 * are set as attributes, `on*` functions become listeners, strings become text
 * nodes, arrays are flattened, and null, undefined and false are skipped.
 */
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (name.startsWith("on") && typeof value === "function") {
      node.addEventListener(name.slice(2), value);
    } else {
      node.setAttribute(name, value === true ? "" : String(value));
    }
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) append(node, child);
    else if (typeof child === "string" || typeof child === "number") node.appendChild(document.createTextNode(String(child)));
    else node.appendChild(child);
  }
}

/** Empty `node` and fill it with `children`. */
export function fill(node, ...children) {
  node.replaceChildren();
  append(node, children);
  return node;
}
