/**
 * `publish` — the harmless gated tool.
 *
 * It exists so the approval queue is exercised on something that cannot hurt
 * anything: it copies an image the Run already rendered into `published/`.
 * Gated means the loop blocks for the operator before this handler runs.
 */

import { copyFileSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import type { Ledger } from "../ledger/ledger.ts";
import type { ToolDefinition } from "./registry.ts";

export interface PublishToolDeps {
  readonly ledger: Ledger;
  readonly publishedDir: string;
}

export function createPublishTool(deps: PublishToolDeps): ToolDefinition {
  return {
    name: "publish",
    description:
      "Publish an image you already rendered by copying it into the " +
      "published folder. This needs operator approval: the call waits until " +
      "the operator approves or denies it. Only a path returned by " +
      "gpu_generate_image in this Run can be published.",
    schema: {
      type: "object",
      properties: {
        artifact_path: {
          type: "string",
          description:
            "The exact `path` returned by gpu_generate_image for the image to publish.",
        },
      },
      required: ["artifact_path"],
      additionalProperties: false,
    },
    tier: "gated",

    validate: (args, ctx) => {
      const path = args["artifact_path"];
      if (typeof path !== "string" || path === "") {
        return "publish needs an artifact_path string.";
      }
      const known = deps.ledger
        .listArtifacts(ctx.runId)
        .some((artifact) => artifact.path === path);
      return known
        ? undefined
        : `${path} is not an artifact this Run rendered; publish only accepts a path returned by gpu_generate_image.`;
    },

    handler: async (args) => {
      const path = String(args["artifact_path"] ?? "");
      mkdirSync(deps.publishedDir, { recursive: true });
      const dest = join(deps.publishedDir, basename(path));
      copyFileSync(path, dest);
      return { published: dest };
    },
  };
}
