"""
RiTa API Server — FastAPI microservice for DPDP compliance interception.

Endpoints:
  POST /intercept         — Synchronous: run the agent and return the full result.
  POST /intercept/stream  — SSE: stream the agent's Thought/Action/Observation loop.
  GET  /health            — Liveness probe.

Run:
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Self
from pydantic import BaseModel, Field, model_validator

from langgraph_agent import get_agent, run_agent_stream, run_agent_sync

# ──────────────────────────────────────────────────────────────
# Lifespan — warm up the agent on startup
# ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialize the LangGraph agent so the first request isn't slow."""
    try:
        get_agent()
        print("✅ RiTa agent initialized")
    except EnvironmentError as exc:
        print(f"⚠️  Agent init skipped: {exc}")
    yield


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RiTa",
    description="DPDP Compliance Agent — intercepts Redshift SQL, queries DataHub, rewrites with proactive masking.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────


class SQLFileInput(BaseModel):
    """A SQL file provided by optional path and content."""

    path: str | None = Field(
        default=None,
        description="Optional file path or name (e.g. 'models/analytics/vw_credit.sql').",
    )
    content: str = Field(
        ...,
        description="The SQL file content.",
    )


class InterceptRequest(BaseModel):
    """Payload for the /intercept endpoint."""

    pr_id: str | None = Field(
        default=None,
        description="The Pull Request ID to scan for DPDP compliance.",
        json_schema_extra={"example": "442"},
    )
    sql_files: list[str | SQLFileInput] | None = Field(
        default=None,
        description="List of SQL files to scan (either raw SQL strings or objects with path and content).",
    )
    context: str | None = Field(
        default=None,
        description="Optional additional context to pass to the agent.",
    )

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        if not self.pr_id and not self.sql_files:
            raise ValueError("Either 'pr_id' or 'sql_files' must be provided.")
        return self

    def to_agent_input(self) -> str:
        """Build the agent's user message from the request fields."""
        parts = []
        if self.pr_id:
            parts.append(f"Please review PR #{self.pr_id} for DPDP compliance.")
        else:
            parts.append("Please review the provided SQL files for DPDP compliance.")

        if self.sql_files:
            parts.append("\nSQL files content to evaluate:")
            for idx, f in enumerate(self.sql_files, 1):
                if isinstance(f, str):
                    path = f"file_{idx}.sql"
                    content = f
                else:
                    path = f.path or f"file_{idx}.sql"
                    content = f.content
                parts.append(f"\n--- File: {path} ---\n{content}\n--- End File ---")

        if self.context:
            parts.append(f"\nAdditional context: {self.context}")

        return "\n".join(parts)


class StepTrace(BaseModel):
    """A single step in the agent's execution trace."""

    type: str = Field(description="One of: thought, tool_call, tool_result")
    tool: str | None = Field(default=None, description="Tool name (for tool_call/tool_result)")
    args: dict | None = Field(default=None, description="Tool arguments (for tool_call)")
    content: str | None = Field(default=None, description="Text content (for thought/tool_result)")


class InterceptResponse(BaseModel):
    """Response from the /intercept endpoint."""

    status: str = "completed"
    pr_id: str | None = None
    final_response: str
    steps: list[StepTrace]
    elapsed_seconds: float


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "service": "rita"}


@app.post("/intercept", response_model=InterceptResponse)
async def intercept(req: InterceptRequest):
    """Run the RiTa compliance agent synchronously and return the full result.

    Accepts a PR ID or a list of SQL files content, runs the compliance
    agent to completion, and returns the execution trace with the final response.
    """
    try:
        t0 = time.perf_counter()
        result = run_agent_sync(req.to_agent_input())
        elapsed = round(time.perf_counter() - t0, 2)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return InterceptResponse(
        status="completed",
        pr_id=req.pr_id,
        final_response=result["final_response"],
        steps=[StepTrace(**s) for s in result["steps"]],
        elapsed_seconds=elapsed,
    )


@app.post("/intercept/stream")
async def intercept_stream(req: InterceptRequest):
    """Stream the agent's execution via Server-Sent Events (SSE).

    Each event is a JSON object representing one step in the agent's
    Thought → Action → Observation loop. The stream ends with a
    `{"type": "done"}` sentinel.

    Usage with curl:
        curl -N -X POST http://localhost:8000/intercept/stream \
          -H 'Content-Type: application/json' \
          -d '{"pr_id": "442"}'

        # Or with raw SQL files content:
        curl -N -X POST http://localhost:8000/intercept/stream \
          -H 'Content-Type: application/json' \
          -d '{"sql_files": ["CREATE VIEW v AS SELECT customer_id, pan_no FROM stg_lake.raw_customers;"]}'
    """

    def event_generator():
        try:
            for step in run_agent_stream(req.to_agent_input()):
                yield f"data: {json.dumps(step)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except EnvironmentError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Agent error: {exc}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
