/**
 * A strict reader for the TOML subset OpenHybridHarness actually uses.
 *
 * Node has no built-in TOML parser, and the GPU service's registry parser carries a
 * comment forbidding a TOML dependency, so this is hand-rolled. It covers
 * `[table]`, `[table."quoted"]`, string/integer/boolean scalars and string
 * arrays — and refuses everything else **by name and line number**.
 *
 * Refusing beats guessing here. `models.toml` holds the VRAM figures the
 * GPU service schedules against; a parser that improvised on an unsupported
 * construct would hand back a plausible wrong number instead of an error, and
 * a wrong VRAM number is exactly the class of bug that cost this project a
 * whole debugging session already.
 */

export class TomlSubsetError extends Error {
  readonly line: number;

  constructor(message: string, line: number) {
    super(`${message} (line ${line})`);
    this.name = "TomlSubsetError";
    this.line = line;
  }
}

type Scalar = string | number | boolean | string[];
type Table = Record<string, unknown>;

/** Remove a trailing `#` comment without touching a `#` inside a string. */
function stripComment(line: string): string {
  let inString = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === "\\" && inString) {
      i++; // skip the escaped character
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (ch === "#" && !inString) return line.slice(0, i);
  }
  return line;
}

/** Split a table header path on dots that are not inside quotes. */
function splitPath(header: string, line: number): string[] {
  const parts: string[] = [];
  let current = "";
  let inQuote = false;

  for (let i = 0; i < header.length; i++) {
    const ch = header[i];
    if (ch === '"') {
      inQuote = !inQuote;
      continue;
    }
    if (ch === "." && !inQuote) {
      parts.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  if (inQuote) throw new TomlSubsetError(`unterminated quoted table key`, line);
  parts.push(current);

  for (const part of parts) {
    if (part.trim() === "") {
      throw new TomlSubsetError(`empty segment in table header`, line);
    }
  }
  return parts.map((p) => p.trim());
}

function parseString(raw: string, line: number): string {
  // The subset has basic strings only; `"""` is multi-line and unsupported.
  if (raw.startsWith('"""')) {
    throw new TomlSubsetError(`multi-line strings are not supported`, line);
  }
  if (raw.length < 2 || !raw.endsWith('"')) {
    throw new TomlSubsetError(`unterminated string ${raw}`, line);
  }
  return raw
    .slice(1, -1)
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, "\\");
}

function parseArray(raw: string, key: string, line: number): string[] {
  const inner = raw.slice(1, -1).trim();
  if (inner === "") return [];
  if (inner.includes("[")) {
    throw new TomlSubsetError(`nested arrays are not supported in ${key}`, line);
  }

  const items: string[] = [];
  for (const piece of inner.split(",")) {
    const item = piece.trim();
    if (item === "") continue;
    if (!item.startsWith('"')) {
      throw new TomlSubsetError(
        `arrays may only hold strings in this subset; ${key} has ${item}`,
        line,
      );
    }
    items.push(parseString(item, line));
  }
  return items;
}

function parseValue(raw: string, key: string, line: number): Scalar {
  const value = raw.trim();

  if (value === "") throw new TomlSubsetError(`${key} has no value`, line);
  if (value.startsWith("{")) {
    throw new TomlSubsetError(`inline tables are not supported in ${key}`, line);
  }
  if (value.startsWith("[")) {
    if (!value.endsWith("]")) {
      throw new TomlSubsetError(`unterminated array in ${key}`, line);
    }
    return parseArray(value, key, line);
  }
  if (value.startsWith('"')) return parseString(value, line);
  if (value === "true") return true;
  if (value === "false") return false;
  if (/^-?\d+$/.test(value)) return Number(value);

  // Everything left over is a construct the subset deliberately excludes:
  // floats, datetimes, hex/octal, bare words. Name it rather than coerce it.
  throw new TomlSubsetError(
    `unsupported value for ${key}: ${value} — this subset allows quoted strings, integers, booleans and string arrays`,
    line,
  );
}

export function parseToml(text: string): Table {
  const root: Table = {};
  let table: Table | undefined;
  const seen = new Set<string>();
  let tablePath = "";

  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const lineNumber = index + 1;
    const line = stripComment(lines[index] ?? "").trim();
    if (line === "") continue;

    if (line.startsWith("[[")) {
      throw new TomlSubsetError(`arrays of tables are not supported`, lineNumber);
    }

    if (line.startsWith("[")) {
      if (!line.endsWith("]")) {
        throw new TomlSubsetError(`unterminated table header`, lineNumber);
      }
      const path = splitPath(line.slice(1, -1).trim(), lineNumber);
      let cursor: Table = root;
      for (const segment of path) {
        const next = cursor[segment];
        if (next === undefined) {
          const created: Table = {};
          cursor[segment] = created;
          cursor = created;
        } else if (typeof next === "object" && next !== null) {
          cursor = next as Table;
        } else {
          throw new TomlSubsetError(
            `table header ${segment} collides with a value of the same name`,
            lineNumber,
          );
        }
      }
      table = cursor;
      tablePath = path.join(".");
      continue;
    }

    const equals = line.indexOf("=");
    if (equals === -1) {
      throw new TomlSubsetError(`expected key = value, got ${line}`, lineNumber);
    }

    const key = line.slice(0, equals).trim();
    if (table === undefined) {
      throw new TomlSubsetError(
        `${key} appears before any [table] header`,
        lineNumber,
      );
    }
    if (key === "") throw new TomlSubsetError(`missing key name`, lineNumber);
    if (key.includes(".")) {
      throw new TomlSubsetError(
        `dotted keys are not supported: ${key} — use a [table."quoted"] header`,
        lineNumber,
      );
    }
    if (key.startsWith('"')) {
      throw new TomlSubsetError(
        `quoted keys are only supported in table headers, not here: ${key}`,
        lineNumber,
      );
    }

    const fullKey = `${tablePath}.${key}`;
    if (seen.has(fullKey)) {
      throw new TomlSubsetError(`duplicate key ${key}`, lineNumber);
    }
    seen.add(fullKey);

    table[key] = parseValue(line.slice(equals + 1), key, lineNumber);
  }

  return root;
}
