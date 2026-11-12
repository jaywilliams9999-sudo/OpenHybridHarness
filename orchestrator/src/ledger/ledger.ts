/**
 * The Ledger — `node:sqlite`, one file, six tables.
 *
 * All six tables are created, but accessors exist only for the three that are
 * exercised. A migration runner arrives when a column first *changes*,
 * not when a table is first *used*.
 */

import { createRequire } from "node:module";
import { randomUUID } from "node:crypto";

import type { Placement, RanAs } from "../gateway/route.ts";

const require_ = createRequire(import.meta.url);

/**
 * `node:sqlite` is experimental and announces itself on first load. The warning
 * is suppressed around this one require and immediately restored, rather than
 * silenced process-wide, so an unrelated future warning still reaches the
 * operator.
 */
function loadSqlite(): typeof import("node:sqlite") {
  const original = process.emitWarning;
  process.emitWarning = ((warning: unknown, ...rest: unknown[]) => {
    const text =
      typeof warning === "string"
        ? warning
        : ((warning as Error | undefined)?.message ?? "");
    if (text.includes("SQLite is an experimental feature")) return;
    return (original as (...args: unknown[]) => void).call(
      process,
      warning,
      ...rest,
    );
  }) as typeof process.emitWarning;
  try {
    return require_("node:sqlite") as typeof import("node:sqlite");
  } finally {
    process.emitWarning = original;
  }
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS runs (
  id              TEXT PRIMARY KEY,
  agent           TEXT NOT NULL,
  goal            TEXT NOT NULL,
  parent_id       TEXT REFERENCES runs(id),
  status          TEXT NOT NULL CHECK (status IN ('running','blocked','done','failed')),
  budget_usd      REAL NOT NULL,
  spent_usd       REAL NOT NULL DEFAULT 0,
  turns           INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  checkpoint_blob TEXT,
  placement       TEXT NOT NULL DEFAULT 'local'
                  CHECK (placement IN ('local','claude','local-no-key'))
);

CREATE TABLE IF NOT EXISTS messages (
  id         TEXT PRIMARY KEY,
  run_id    TEXT NOT NULL REFERENCES runs(id),
  from_agent TEXT NOT NULL,
  to_agent   TEXT NOT NULL,
  kind       TEXT NOT NULL,
  body       TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id          TEXT PRIMARY KEY,
  run_id     TEXT NOT NULL REFERENCES runs(id),
  tool        TEXT NOT NULL,
  args_json   TEXT NOT NULL,
  result_json TEXT NOT NULL,
  tier        TEXT NOT NULL CHECK (tier IN ('auto','budgeted','gated')),
  ms          INTEGER NOT NULL,
  created_at  TEXT NOT NULL
);

-- Usage arrives in four buckets at
-- three prices, and a cache read costs ~0.1x a fresh input token. Collapsing
-- them overstates a cache-heavy turn ~10x, and the budget spends that number.
-- usage_json keeps the raw evidence so a wrong price table can be
-- re-derived instead of having been discarded.
CREATE TABLE IF NOT EXISTS costs (
  id                 TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL REFERENCES runs(id),
  provider           TEXT NOT NULL,
  model              TEXT NOT NULL,
  in_tokens          INTEGER NOT NULL,
  cache_write_tokens INTEGER NOT NULL,
  cache_read_tokens  INTEGER NOT NULL,
  out_tokens         INTEGER NOT NULL,
  usd                REAL NOT NULL,
  usage_json         TEXT,
  created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  id         TEXT PRIMARY KEY,
  run_id    TEXT NOT NULL REFERENCES runs(id),
  kind       TEXT NOT NULL,
  path       TEXT NOT NULL,
  meta_json  TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
  id           TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL REFERENCES runs(id),
  tool         TEXT NOT NULL,
  args_json    TEXT NOT NULL,
  status       TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  decided_at   TEXT
);
`;

export type RunStatus = "running" | "blocked" | "done" | "failed";
export type RiskTier = "auto" | "budgeted" | "gated";

export interface RunRow {
  id: string;
  agent: string;
  goal: string;
  parent_id: string | null;
  status: RunStatus;
  budget_usd: number;
  spent_usd: number;
  turns: number;
  created_at: string;
  updated_at: string;
  /** Where the run was placed when it started; see `ranAs` for where it ran. */
  placement: Placement;
}

export interface ToolCallRow {
  id: string;
  run_id: string;
  tool: string;
  args_json: string;
  result_json: string;
  tier: RiskTier;
  ms: number;
  created_at: string;
}

export interface ArtifactRow {
  id: string;
  run_id: string;
  kind: string;
  path: string;
  meta_json: string | null;
  created_at: string;
}

export type ApprovalStatus = "pending" | "approved" | "denied";

export interface ApprovalRow {
  id: string;
  run_id: string;
  tool: string;
  args_json: string;
  status: ApprovalStatus;
  requested_at: string;
  decided_at: string | null;
}

export interface CostRow {
  id: string;
  run_id: string;
  provider: string;
  model: string;
  in_tokens: number;
  cache_write_tokens: number;
  cache_read_tokens: number;
  out_tokens: number;
  usd: number;
  usage_json: string | null;
  created_at: string;
}

export interface MessageRow {
  id: string;
  run_id: string;
  from_agent: string;
  to_agent: string;
  kind: string;
  body: string;
  created_at: string;
}

export interface Ledger {
  createRun(run: {
    agent: string;
    goal: string;
    parentId: string | null;
    budgetUsd: number;
    /** Defaults to `local`, the placement of an agent with no `model` line. */
    placement?: Placement;
  }): string;
  getRun(id: string): RunRow | undefined;
  /** Runs with no parent, newest first. The Hub's list of goals. */
  listRootRuns(): RunRow[];
  setRunStatus(id: string, status: RunStatus): void;
  /** Only for a Claude run resumed with no key, which finishes as `local-no-key`. */
  setRunPlacement(id: string, placement: Placement): void;
  bumpTurn(id: string): void;
  recordToolCall(call: {
    runId: string;
    tool: string;
    args: unknown;
    result: unknown;
    tier: RiskTier;
    ms: number;
  }): string;
  listToolCalls(runId: string): ToolCallRow[];
  recordArtifact(artifact: {
    runId: string;
    kind: string;
    path: string;
    meta: unknown;
  }): string;
  listArtifacts(runId: string): ArtifactRow[];
  getArtifact(id: string): ArtifactRow | undefined;
  /**
   * One row per model turn, plus the running total on the Run.
   *
   * Both in one call and one transaction on purpose: `costs` is the evidence
   * and `runs.spent_usd` is what the budget check reads. Letting a
   * caller write one without the other is how a budget silently stops
   * matching the rows that justify it.
   */
  recordCost(cost: {
    runId: string;
    provider: string;
    model: string;
    usage: {
      in: number;
      cacheWrite: number;
      cacheRead: number;
      out: number;
    };
    usd: number;
  }): string;
  listCosts(runId: string): CostRow[];
  /** One row of Run-to-Run traffic; `body` is stored as JSON. */
  recordMessage(message: {
    runId: string;
    from: string;
    to: string;
    kind: string;
    body: unknown;
  }): string;
  listMessages(runId: string): MessageRow[];
  /**
   * `spent_usd` of the Run plus every descendant.
   *
   * Summed from the column `recordCost` already maintains rather than from
   * `costs`, so the number a parent's budget is checked against is the same
   * number each child's own budget was.
   */
  treeSpentUsd(runId: string): number;
  /** Direct children only, oldest first. */
  listChildren(runId: string): RunRow[];
  /**
   * Store the resume point. Opaque JSON to the Ledger on purpose — the shape
   * is the Loop's business, and a schema here would need a migration runner
   * the first time the Loop learned a new field.
   */
  saveCheckpoint(runId: string, blob: string): void;
  readCheckpoint(runId: string): string | undefined;
  /**
   * Record that a `gated` call is waiting on a human.
   *
   * The row deliberately carries no tool_use_id: the approvals table has no
   * such column, and adding one is exactly the schema change a migration
   * runner would be needed for. The call-to-approval mapping lives in the Run's
   * checkpoint instead, which is the only place that needs it — an operator
   * surface needs the tool and its arguments, and both are here.
   */
  requestApproval(request: {
    runId: string;
    tool: string;
    args: unknown;
  }): string;
  listApprovals(runId: string): ApprovalRow[];
  /** Every Run waiting on something, newest first. The operator's inbox. */
  listBlockedRuns(): RunRow[];
  readApproval(id: string): ApprovalRow | undefined;
  /** False if it was already decided, so a race cannot flip a live decision. */
  decideApproval(id: string, status: "approved" | "denied"): boolean;
  tableNames(): string[];
  columnNames(table: string): string[];
  close(): void;
}

/**
 * Where a run actually ran. A Claude-placed run that hit the spending wall
 * finished on the local model, and its cost rows say so: the failover
 * provider reports whichever provider answered, so an `ollama` row on a
 * `claude` run is the fallback.
 */
export function ranAs(ledger: Pick<Ledger, "listCosts">, run: RunRow): RanAs {
  if (
    run.placement === "claude" &&
    ledger.listCosts(run.id).some((row) => row.provider === "ollama")
  ) {
    return "local-fallback";
  }
  return run.placement;
}

export function openLedger(path: string): Ledger {
  const { DatabaseSync } = loadSqlite();
  const db = new DatabaseSync(path);

  // Per-connection, and off by default. Without it sqlite accepts an orphaned
  // tool_call silently and the Ledger stops being a faithful record.
  db.exec("PRAGMA foreign_keys = ON");
  // Two writers share this file: the Loop and the Hub. busy_timeout comes
  // first so that switching to WAL, which takes a lock, waits instead of
  // failing with "database is locked" when two opens race. WAL then lets a
  // reader run beside a writer.
  db.exec("PRAGMA busy_timeout = 5000");
  db.exec("PRAGMA journal_mode = WAL");
  db.exec(SCHEMA);
  // A ledger written before runs carried a placement: `CREATE TABLE IF NOT
  // EXISTS` leaves its runs table alone, so the column is added here. Every
  // earlier run was a single-provider run, recorded as the default.
  const runColumns = (
    db.prepare(`PRAGMA table_info(runs)`).all() as Array<{ name: string }>
  ).map((row) => row.name);
  if (!runColumns.includes("placement")) {
    db.exec(
      `ALTER TABLE runs ADD COLUMN placement TEXT NOT NULL DEFAULT 'local'
         CHECK (placement IN ('local','claude','local-no-key'))`,
    );
  }

  const now = () => new Date().toISOString();

  return {
    createRun({ agent, goal, parentId, budgetUsd, placement = "local" }) {
      const id = randomUUID();
      const stamp = now();
      db.prepare(
        `INSERT INTO runs
           (id, agent, goal, parent_id, status, budget_usd, spent_usd, turns,
            created_at, updated_at, placement)
         VALUES (?, ?, ?, ?, 'running', ?, 0, 0, ?, ?, ?)`,
      ).run(id, agent, goal, parentId, budgetUsd, stamp, stamp, placement);
      return id;
    },

    getRun(id) {
      return db.prepare(`SELECT * FROM runs WHERE id = ?`).get(id) as
        | RunRow
        | undefined;
    },

    listRootRuns() {
      return db
        .prepare(
          `SELECT * FROM runs WHERE parent_id IS NULL
           ORDER BY created_at DESC, rowid DESC`,
        )
        .all() as unknown as RunRow[];
    },

    setRunPlacement(id, placement) {
      db.prepare(`UPDATE runs SET placement = ?, updated_at = ? WHERE id = ?`).run(
        placement,
        now(),
        id,
      );
    },

    setRunStatus(id, status) {
      db.prepare(`UPDATE runs SET status = ?, updated_at = ? WHERE id = ?`).run(
        status,
        now(),
        id,
      );
    },

    bumpTurn(id) {
      db.prepare(
        `UPDATE runs SET turns = turns + 1, updated_at = ? WHERE id = ?`,
      ).run(now(), id);
    },

    recordToolCall({ runId, tool, args, result, tier, ms }) {
      const id = randomUUID();
      db.prepare(
        `INSERT INTO tool_calls
           (id, run_id, tool, args_json, result_json, tier, ms, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        id,
        runId,
        tool,
        JSON.stringify(args ?? null),
        JSON.stringify(result ?? null),
        tier,
        ms,
        now(),
      );
      return id;
    },

    listToolCalls(runId) {
      return db
        .prepare(
          `SELECT * FROM tool_calls WHERE run_id = ? ORDER BY created_at, id`,
        )
        .all(runId) as unknown as ToolCallRow[];
    },

    recordArtifact({ runId, kind, path: filePath, meta }) {
      const id = randomUUID();
      db.prepare(
        `INSERT INTO artifacts (id, run_id, kind, path, meta_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(id, runId, kind, filePath, JSON.stringify(meta ?? null), now());
      return id;
    },

    listArtifacts(runId) {
      return db
        .prepare(
          `SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at, id`,
        )
        .all(runId) as unknown as ArtifactRow[];
    },

    getArtifact(id) {
      return db.prepare(`SELECT * FROM artifacts WHERE id = ?`).get(id) as
        | ArtifactRow
        | undefined;
    },

    recordCost({ runId, provider, model, usage, usd }) {
      const id = randomUUID();
      const stamp = now();
      // One transaction: a cost row without its matching `spent_usd` bump, or
      // a bump without its row, both leave the budget unauditable.
      db.exec("BEGIN");
      try {
        db.prepare(
          `INSERT INTO costs
             (id, run_id, provider, model, in_tokens, cache_write_tokens,
              cache_read_tokens, out_tokens, usd, usage_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          id,
          runId,
          provider,
          model,
          usage.in,
          usage.cacheWrite,
          usage.cacheRead,
          usage.out,
          usd,
          JSON.stringify(usage),
          stamp,
        );
        db.prepare(
          `UPDATE runs SET spent_usd = spent_usd + ?, updated_at = ?
           WHERE id = ?`,
        ).run(usd, stamp, runId);
        db.exec("COMMIT");
      } catch (error) {
        db.exec("ROLLBACK");
        throw error;
      }
      return id;
    },

    listCosts(runId) {
      return db
        .prepare(`SELECT * FROM costs WHERE run_id = ? ORDER BY created_at, id`)
        .all(runId) as unknown as CostRow[];
    },

    recordMessage({ runId, from, to, kind, body }) {
      const id = randomUUID();
      db.prepare(
        `INSERT INTO messages
           (id, run_id, from_agent, to_agent, kind, body, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(id, runId, from, to, kind, JSON.stringify(body ?? null), now());
      return id;
    },

    listMessages(runId) {
      return db
        .prepare(
          `SELECT * FROM messages WHERE run_id = ? ORDER BY created_at, rowid`,
        )
        .all(runId) as unknown as MessageRow[];
    },

    treeSpentUsd(runId) {
      const row = db
        .prepare(
          `WITH RECURSIVE tree(id) AS (
             SELECT id FROM runs WHERE id = ?
             UNION ALL
             SELECT p.id FROM runs p JOIN tree t ON p.parent_id = t.id
           )
           SELECT COALESCE(SUM(p.spent_usd), 0) AS total
           FROM runs p JOIN tree t ON p.id = t.id`,
        )
        .get(runId) as { total: number };
      return row.total;
    },

    listChildren(runId) {
      return db
        .prepare(
          `SELECT * FROM runs WHERE parent_id = ? ORDER BY created_at, rowid`,
        )
        .all(runId) as unknown as RunRow[];
    },

    saveCheckpoint(runId, blob) {
      db.prepare(
        `UPDATE runs SET checkpoint_blob = ?, updated_at = ? WHERE id = ?`,
      ).run(blob, now(), runId);
    },

    readCheckpoint(runId) {
      const row = db
        .prepare(`SELECT checkpoint_blob FROM runs WHERE id = ?`)
        .get(runId) as { checkpoint_blob: string | null } | undefined;
      return row?.checkpoint_blob ?? undefined;
    },

    requestApproval({ runId, tool, args }) {
      const id = randomUUID();
      db.prepare(
        `INSERT INTO approvals
           (id, run_id, tool, args_json, status, requested_at, decided_at)
         VALUES (?, ?, ?, ?, 'pending', ?, NULL)`,
      ).run(id, runId, tool, JSON.stringify(args ?? null), now());
      return id;
    },

    listBlockedRuns() {
      return db
        .prepare(
          `SELECT * FROM runs WHERE status = 'blocked' ORDER BY updated_at DESC`,
        )
        .all() as unknown as RunRow[];
    },

    listApprovals(runId) {
      return db
        .prepare(
          `SELECT * FROM approvals WHERE run_id = ? ORDER BY requested_at, id`,
        )
        .all(runId) as unknown as ApprovalRow[];
    },

    readApproval(id) {
      return db.prepare(`SELECT * FROM approvals WHERE id = ?`).get(id) as
        | ApprovalRow
        | undefined;
    },

    decideApproval(id, status) {
      // Guarded on `status = 'pending'` inside the UPDATE rather than
      // read-then-write: two operators deciding at once must not have the
      // later one silently overwrite a decision the agent already acted on.
      const result = db
        .prepare(
          `UPDATE approvals SET status = ?, decided_at = ?
           WHERE id = ? AND status = 'pending'`,
        )
        .run(status, now(), id);
      return Number(result.changes) > 0;
    },

    tableNames() {
      const rows = db
        .prepare(`SELECT name FROM sqlite_master WHERE type = 'table'`)
        .all() as Array<{ name: string }>;
      return rows.map((row) => row.name);
    },

    columnNames(table) {
      const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
        name: string;
      }>;
      return rows.map((row) => row.name);
    },

    close() {
      db.close();
    },
  };
}
