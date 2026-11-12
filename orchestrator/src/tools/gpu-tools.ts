/**
 * The two GPU service tools exposed to a model.
 *
 * This is the layer that owns the disk and the Ledger, deliberately — not
 * `GpuClient`, which stays purely wire-aware so it can be tested against a
 * stub server with no filesystem and no database in play.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { GpuClient } from "../gpu/client.ts";
import type { Ledger } from "../ledger/ledger.ts";
import type { ToolDefinition } from "./registry.ts";

export interface GpuToolDeps {
  readonly gpu: GpuClient;
  readonly ledger: Ledger;
  readonly artifactsDir: string;
}

export function createGpuTools(deps: GpuToolDeps): ToolDefinition[] {
  return [
    {
      name: "gpu_status",
      description:
        "Report the GPU service's resident models and free VRAM. Free and reversible.",
      schema: { type: "object", properties: {}, additionalProperties: false },
      tier: "auto",
      handler: async () => deps.gpu.status(),
    },

    {
      name: "gpu_generate_image",
      description:
        "Render an image on the GPU service's GPU, auto-loading the image model and " +
        "evicting another model if the card is full. Returns the local path and " +
        "any evictions the request caused.",
      // No `steps`: the registry owns step count, because it is a property of
      // the model and not of the request. sdxl-turbo is distilled to converge
      // in 1-4 steps, and a model offered an unbounded integer picked 30 --
      // 36.8s against 2.8s for the same image, for no gain. `additionalProperties:
      // false` makes that unaskable rather than merely discouraged.
      schema: {
        type: "object",
        properties: {
          prompt: {
            type: "string",
            description: "What to draw. Describe subject, style and lighting.",
          },
          width: {
            type: "integer",
            description:
              "Pixels wide, a multiple of 64 (default 512). The model is " +
              "trained at 512x512; 1024x576 is a good 16:9 banner. Larger " +
              "costs render time roughly in proportion to pixel count.",
          },
          height: {
            type: "integer",
            description: "Pixels tall, a multiple of 64 (default 512).",
          },
          seed: {
            type: "integer",
            description:
              "Fixes the result, so the same prompt and seed redraw the same " +
              "image. Omit for variety; set it to iterate on one composition.",
          },
        },
        required: ["prompt"],
        additionalProperties: false,
      },
      tier: "budgeted",

      handler: async (args, ctx) => {
        const prompt = String(args["prompt"] ?? "");
        if (!prompt) throw new Error("gpu_generate_image needs a prompt");

        const request: Parameters<GpuClient["generateImage"]>[0] = { prompt };
        for (const key of ["width", "height", "seed"] as const) {
          const value = args[key];
          if (typeof value === "number") request[key] = value;
        }

        const result = await deps.gpu.generateImage(request);
        const bytes = await deps.gpu.fetchArtifact(result.artifact_url);

        mkdirSync(deps.artifactsDir, { recursive: true });
        const path = join(deps.artifactsDir, `${result.artifact_id}.png`);
        writeFileSync(path, bytes);

        deps.ledger.recordArtifact({
          runId: ctx.runId,
          kind: "image",
          path,
          meta: {
            bytes: bytes.length,
            model: result.model,
            steps: result.steps,
            evicted: result.evicted,
          },
        });

        // The eviction records are part of what the MODEL sees,
        // not merely something the operator can read in a log. A model whose
        // request unloaded another model learns that it happened, and may
        // choose to reload.
        return {
          path,
          bytes: bytes.length,
          model: result.model,
          steps: result.steps,
          evicted: result.evicted,
        };
      },
    },
  ];
}
