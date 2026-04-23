import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { and, eq } from "drizzle-orm";
import type { Db } from "@paperclipai/db";
import { agents as agentsTable, agentTaskSessions } from "@paperclipai/db";
import { getServerAdapter } from "../adapters/index.js";
import { secretService } from "./secrets.js";
import { createLocalAgentJwt } from "../agent-auth-jwt.js";
import { notFound } from "../errors.js";
import { parseObject } from "../adapters/utils.js";
import { logger } from "../middleware/logger.js";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ConversationRequest {
  message: string;
  sessionKey?: string;
  source?: string;
  allowedTools?: string[];
  maxTurns?: number;
}

export interface ConversationResult {
  response: string;
  sessionId: string;
  turnsUsed: number;
  rawResult: Record<string, unknown> | null;
}

export interface ChainStep {
  agentId: string;
  task: string;
}

export interface ChainRequest {
  steps: ChainStep[];
  context?: string;
  passOutputForward?: boolean;
}

export interface ChainStepResult {
  agentId: string;
  agentName: string;
  output: string;
  turnsUsed: number;
  error?: string;
}

export interface ChainResult {
  results: ChainStepResult[];
  totalTurnsUsed: number;
}

// ─── Service ─────────────────────────────────────────────────────────────────

export function conversationService(db: Db) {
  const secrets = secretService(db);

  async function getAgent(agentId: string) {
    const rows = await db
      .select()
      .from(agentsTable)
      .where(eq(agentsTable.id, agentId));
    return rows[0] ?? null;
  }

  async function getConvSession(
    companyId: string,
    agentId: string,
    adapterType: string,
    sessionKey: string,
  ) {
    const taskKey = `conv:${sessionKey}`;
    const rows = await db
      .select()
      .from(agentTaskSessions)
      .where(
        and(
          eq(agentTaskSessions.companyId, companyId),
          eq(agentTaskSessions.agentId, agentId),
          eq(agentTaskSessions.adapterType, adapterType),
          eq(agentTaskSessions.taskKey, taskKey),
        ),
      );
    return rows[0] ?? null;
  }

  async function upsertConvSession(input: {
    companyId: string;
    agentId: string;
    adapterType: string;
    sessionKey: string;
    sessionParamsJson: Record<string, unknown> | null;
    sessionDisplayId: string | null;
  }) {
    const taskKey = `conv:${input.sessionKey}`;
    const existing = await getConvSession(
      input.companyId,
      input.agentId,
      input.adapterType,
      input.sessionKey,
    );

    if (existing) {
      await db
        .update(agentTaskSessions)
        .set({
          sessionParamsJson: input.sessionParamsJson,
          sessionDisplayId: input.sessionDisplayId,
          updatedAt: new Date(),
        })
        .where(eq(agentTaskSessions.id, existing.id));
      return;
    }

    await db.insert(agentTaskSessions).values({
      id: randomUUID(),
      companyId: input.companyId,
      agentId: input.agentId,
      adapterType: input.adapterType,
      taskKey,
      sessionParamsJson: input.sessionParamsJson,
      sessionDisplayId: input.sessionDisplayId,
      createdAt: new Date(),
      updatedAt: new Date(),
    });
  }

  async function converse(
    agentId: string,
    request: ConversationRequest,
  ): Promise<ConversationResult> {
    const agent = await getAgent(agentId);
    if (!agent) throw notFound("Agent not found");

    const sessionKey = request.sessionKey ?? "conversation";
    const runId = randomUUID();

    const rawConfig = parseObject(agent.adapterConfig) ?? {};
    const { config: runtimeConfig } = await secrets.resolveAdapterConfigForRuntime(
      agent.companyId,
      rawConfig,
    );

    // Build conversation directive based on tool access level
    const hasTools = request.allowedTools && request.allowedTools.length > 0;
    const toolDirective = hasTools
      ? `You MAY use these tools when needed: ${request.allowedTools!.join(", ")}. Use tools when the user asks you to DO something, recall something, or when you need context. For casual chat, just talk — no tools needed.`
      : "Do NOT use tools.";

    // Inject current date/time and HEARTBEAT.md into the conversation prompt so the
    // agent has accurate context without needing to make file-read tool calls.
    const nowET = new Date().toLocaleString("en-US", {
      timeZone: "America/New_York",
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    });

    let heartbeatSection = "";
    const instructionsFilePath = typeof rawConfig.instructionsFilePath === "string" ? rawConfig.instructionsFilePath : "";
    if (instructionsFilePath) {
      const heartbeatPath = path.join(path.dirname(instructionsFilePath), "HEARTBEAT.md");
      try {
        const heartbeatContent = await fs.readFile(heartbeatPath, "utf-8");
        heartbeatSection = [
          "",
          "---",
          "## Active Protocols (HEARTBEAT.md — loaded at session start, do not re-read)",
          "",
          heartbeatContent.trim(),
          "---",
          "",
        ].join("\n");
      } catch {
        // Agent has no HEARTBEAT.md — skip injection
      }
    }

    const conversationDirective = [
      "You are in a LIVE CONVERSATION via messaging. This is NOT a heartbeat or task.",
      "",
      `Current date and time: ${nowET}`,
      "",
      "RULES:",
      `- ${toolDirective}`,
      "- Do NOT run heartbeat procedures. Do NOT check task queues.",
      "- Keep responses under 3 sentences unless asked to elaborate.",
      "- If a task will take multiple steps, acknowledge first, then work.",
      heartbeatSection,
      request.message,
    ].join("\n");
    runtimeConfig.promptTemplate = conversationDirective;

    // Use caller-specified maxTurns, fall back to config, cap at 20
    const requestedTurns = request.maxTurns ?? (hasTools ? 15 : 3);
    runtimeConfig.maxTurnsPerRun = runtimeConfig.maxTurnsPerRun
      ? Math.min(Number(runtimeConfig.maxTurnsPerRun), requestedTurns)
      : requestedTurns;

    // Pass allowed tools to the adapter so it can restrict Claude CLI
    if (request.allowedTools) {
      runtimeConfig.allowedTools = request.allowedTools;
    }

    const existingSession = await getConvSession(
      agent.companyId,
      agentId,
      agent.adapterType,
      sessionKey,
    );

    const adapter = getServerAdapter(agent.adapterType);
    const authToken = adapter.supportsLocalAgentJwt
      ? createLocalAgentJwt(agent.id, agent.companyId, agent.adapterType, runId)
      : null;

    const logLines: string[] = [];
    const onLog = async (_stream: "stdout" | "stderr", chunk: string) => {
      logLines.push(chunk);
    };

    const runtime = {
      sessionId: existingSession?.sessionDisplayId ?? null,
      sessionParams: existingSession?.sessionParamsJson ?? null,
      sessionDisplayId: existingSession?.sessionDisplayId ?? null,
      taskKey: `conv:${sessionKey}`,
    };

    logger.info(
      { agentId, sessionKey, runId, hasExistingSession: !!existingSession },
      "conversation: invoking adapter",
    );

    const adapterResult = await adapter.execute({
      runId,
      agent: {
        id: agent.id,
        companyId: agent.companyId,
        name: agent.name,
        adapterType: agent.adapterType,
        adapterConfig: agent.adapterConfig as Record<string, unknown>,
      },
      runtime,
      config: runtimeConfig,
      context: {
        source: request.source ?? "conversation_api",
        conversationSessionKey: sessionKey,
      },
      onLog,
      authToken: authToken ?? undefined,
    });

    // Persist session for resume
    const sessionParams = adapterResult.sessionParams ?? (
      adapterResult.sessionId ? { sessionId: adapterResult.sessionId } : null
    );
    if (sessionParams) {
      await upsertConvSession({
        companyId: agent.companyId,
        agentId,
        adapterType: agent.adapterType,
        sessionKey,
        sessionParamsJson: sessionParams,
        sessionDisplayId: adapterResult.sessionDisplayId ?? adapterResult.sessionId ?? null,
      });
    }

    const response = adapterResult.summary ?? adapterResult.errorMessage ?? "";
    const turnsUsed = adapterResult.usage?.outputTokens
      ? Math.ceil(adapterResult.usage.outputTokens / 1000)
      : 1;

    return {
      response,
      sessionId: sessionKey,
      turnsUsed,
      rawResult: adapterResult.resultJson ?? null,
    };
  }

  async function chain(
    _initiatingAgentId: string,
    request: ChainRequest,
  ): Promise<ChainResult> {
    const passForward = request.passOutputForward !== false;
    const results: ChainStepResult[] = [];
    let runningContext = request.context ?? "";
    let totalTurnsUsed = 0;
    const chainRunId = randomUUID();

    for (const step of request.steps) {
      const stepAgent = await getAgent(step.agentId);
      const agentName = stepAgent?.name ?? step.agentId;

      try {
        const message = passForward && runningContext
          ? `${runningContext}\n\n---\n\nTask: ${step.task}`
          : step.task;

        const result = await converse(step.agentId, {
          message,
          sessionKey: `chain:${chainRunId}`,
          source: "chain",
        });

        results.push({
          agentId: step.agentId,
          agentName,
          output: result.response,
          turnsUsed: result.turnsUsed,
        });

        totalTurnsUsed += result.turnsUsed;

        if (passForward) {
          runningContext = result.response;
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        logger.error({ agentId: step.agentId, chainRunId, error: errorMessage }, "chain step failed");
        results.push({
          agentId: step.agentId,
          agentName,
          output: "",
          turnsUsed: 0,
          error: errorMessage,
        });
      }
    }

    return { results, totalTurnsUsed };
  }

  return { converse, chain };
}
