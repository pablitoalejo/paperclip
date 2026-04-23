import { Router } from "express";
import type { Db } from "@paperclipai/db";
import { agents as agentsTable } from "@paperclipai/db";
import { eq } from "drizzle-orm";
import { conversationService } from "../services/conversation.js";
import { badRequest, notFound } from "../errors.js";

export function conversationRoutes(db: Db) {
  const router = Router();
  const convSvc = conversationService(db);

  async function resolveAgent(agentId: string) {
    const rows = await db
      .select()
      .from(agentsTable)
      .where(eq(agentsTable.id, agentId));
    return rows[0] ?? null;
  }

  /**
   * POST /agents/:id/conversation
   *
   * Send a message to an agent and get a response.
   * Sessions are persisted across calls using the sessionKey.
   */
  router.post("/agents/:id/conversation", async (req, res, next) => {
    try {
      const agentId = req.params.id;
      const agent = await resolveAgent(agentId);
      if (!agent) throw notFound("Agent not found");

      const { message, sessionKey, source, allowedTools, maxTurns } = req.body as {
        message?: unknown;
        sessionKey?: unknown;
        source?: unknown;
        allowedTools?: unknown;
        maxTurns?: unknown;
      };

      if (typeof message !== "string" || !message.trim()) {
        throw badRequest("message is required and must be a non-empty string");
      }

      const result = await convSvc.converse(agentId, {
        message,
        sessionKey: typeof sessionKey === "string" ? sessionKey : undefined,
        source: typeof source === "string" ? source : undefined,
        allowedTools: Array.isArray(allowedTools) ? allowedTools.filter((t): t is string => typeof t === "string") : undefined,
        maxTurns: typeof maxTurns === "number" && maxTurns > 0 ? Math.min(maxTurns, 20) : undefined,
      });

      res.json(result);
    } catch (err) {
      next(err);
    }
  });

  /**
   * POST /agents/:id/chain
   *
   * Execute a chain of agent steps sequentially.
   * Each step's output can be forwarded as context to the next step.
   */
  router.post("/agents/:id/chain", async (req, res, next) => {
    try {
      const initiatingAgentId = req.params.id;
      const agent = await resolveAgent(initiatingAgentId);
      if (!agent) throw notFound("Agent not found");

      const body = req.body as {
        steps?: unknown;
        context?: unknown;
        passOutputForward?: unknown;
      };

      if (!Array.isArray(body.steps) || body.steps.length === 0) {
        throw badRequest("steps is required and must be a non-empty array");
      }

      for (const step of body.steps) {
        if (
          typeof step !== "object" ||
          step === null ||
          typeof (step as Record<string, unknown>).agentId !== "string" ||
          typeof (step as Record<string, unknown>).task !== "string"
        ) {
          throw badRequest("each step must have agentId (string) and task (string)");
        }
      }

      const result = await convSvc.chain(initiatingAgentId, {
        steps: body.steps as Array<{ agentId: string; task: string }>,
        context: typeof body.context === "string" ? body.context : undefined,
        passOutputForward: body.passOutputForward !== false,
      });

      res.json(result);
    } catch (err) {
      next(err);
    }
  });

  return router;
}
