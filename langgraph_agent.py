"""
RiTa — LangGraph ReAct Agent for Dataverse DPDP Compliance.

Intercepts Redshift SQL from Pull Request descriptions, queries DataHub for
compliance tags, rewrites the SQL with proactive masking via sqlglot, and
blocks the PR in Azure DevOps if violations are found.

Usage:
  - CLI:  python langgraph_agent.py
  - API:  import and call get_agent() / run_agent_sync() / run_agent_stream()
"""

from __future__ import annotations

import os
import sys
from typing import Any, Generator

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from agent_tools import (
    block_azure_devops_pr,
    extract_sql_columns_and_tables,
    fetch_pr_sql_files,
    get_datahub_column_tags,
    rewrite_sql_ast,
)

load_dotenv()

console = Console()

# ──────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are RiTa, a strict DPDP (Digital Personal Data Protection) compliance agent.

Your job is to scan SQL files provided directly, parse and check every column
against Dataverse for compliance tags, and proactively inject masking into the SQL.

─── WORKFLOW ───────────────────────────────────────────────────
1. **Determine input mode**:
   • If SQL files content is directly provided in the user's message, use those SQL files directly. Do NOT call `fetch_pr_sql_files`.
2. **For each SQL file**, call `extract_sql_columns_and_tables(sql)` to extract all columns along with the table in which each column is present.
3. **For each column**, call `get_datahub_column_tags(table_name, column_name)` to fetch
   its compliance tags from Dataverse. If tag is not available for any column, use lineage of the column to fetch upstream tables and columns.
   You can recursively traverse till the top of the lineage to fetch that particular column tag.
4. **Apply masking rules** based on tags:
   • If a tag equals PII  → mandate: `SHA2({col}, 256)`
5. **If any column requires masking**, call `rewrite_sql_ast(raw_sql, mask_instructions)`
   for each affected SQL file, where `mask_instructions` is a dict mapping column names
   to their masking expressions with `{col}` as placeholder.
   Example: `{"pan_no": "SHA2({col}, 256)", "phone_num": "'XXXXXX' || RIGHT({col}, 4)"}`
6. **Report / Action**:
   • If SQL files were provided directly: do NOT call `block_azure_devops_pr`. Detail the compliance findings and provide the rewritten masked SQL directly in your response.
   • If reviewing a PR via `fetch_pr_sql_files`: call `block_azure_devops_pr(pr_id, violation_summary, suggested_sql)` to block the PR and post the compliant SQL as a comment. Include ALL rewritten files in the suggested_sql.
7. **If NO columns require masking** across any file, reply that the SQL is compliant.

─── RULES ──────────────────────────────────────────────────────
• When SQL files are directly provided, do NOT call `fetch_pr_sql_files` or `block_azure_devops_pr`.
• When reviewing by PR ID alone, you MUST call `fetch_pr_sql_files` first.
• You MUST scan EVERY SQL file, not just the first one.
• You MUST check EVERY column in every SELECT clause.
• You MUST NOT approve or pass SQL that selects unmasked PII columns.
• Always use the Redshift dialect for SQL operations.
• Be concise in your reasoning but thorough in your checks.
"""


TOOLS = [
    extract_sql_columns_and_tables,
    fetch_pr_sql_files,
    get_datahub_column_tags,
    rewrite_sql_ast,
    block_azure_devops_pr,
]


# ──────────────────────────────────────────────────────────────
# Agent Factory
# ──────────────────────────────────────────────────────────────

_agent_instance = None


def _get_llm():
    """Return the configured Chat LLM (Bedrock or Gemini)."""
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "bedrock":
        from bedrock import get_bedrock_chat_llm

        return get_bedrock_chat_llm()

    # Default to Gemini if GOOGLE_API_KEY is set or provider is gemini
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0,
        )

    # If neither explicit provider nor Gemini key, try Bedrock before failing
    try:
        from bedrock import get_bedrock_chat_llm

        return get_bedrock_chat_llm()
    except Exception:
        pass

    raise EnvironmentError(
        "No LLM provider configured. Either set GOOGLE_API_KEY for Gemini, "
        "or set LLM_PROVIDER=bedrock with valid AWS credentials."
    )


def get_agent():
    """Lazily create and cache the LangGraph ReAct agent."""
    global _agent_instance
    if _agent_instance is None:
        llm = _get_llm()
        _agent_instance = create_react_agent(
            model=llm,
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
        )
    return _agent_instance


# ──────────────────────────────────────────────────────────────
# Programmatic API (used by server.py)
# ──────────────────────────────────────────────────────────────


def run_agent_sync(user_input: str) -> dict[str, Any]:
    """Run the agent to completion and return structured results.

    Returns:
        A dict with keys: steps (list of trace dicts) and final_response (str).
    """
    agent = get_agent()
    steps: list[dict[str, Any]] = []
    final_response = ""

    for step in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
    ):
        for node_name, node_output in step.items():
            if node_name == "agent":
                for msg in node_output.get("messages", []):
                    content = getattr(msg, "content", "")
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            steps.append(
                                {
                                    "type": "tool_call",
                                    "tool": tc["name"],
                                    "args": tc["args"],
                                }
                            )
                    if content:
                        steps.append({"type": "thought", "content": content})
                        final_response = content  # last thought is the final answer
            elif node_name == "tools":
                for msg in node_output.get("messages", []):
                    content = getattr(msg, "content", "")
                    tool_name = getattr(msg, "name", node_name)
                    if content:
                        steps.append(
                            {
                                "type": "tool_result",
                                "tool": tool_name,
                                "content": content,
                            }
                        )

    return {"steps": steps, "final_response": final_response}


def run_agent_stream(user_input: str) -> Generator[dict[str, Any], None, None]:
    """Stream agent steps as individual dicts (for SSE)."""
    agent = get_agent()

    for step in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
    ):
        for node_name, node_output in step.items():
            if node_name == "agent":
                for msg in node_output.get("messages", []):
                    content = getattr(msg, "content", "")
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            yield {
                                "type": "tool_call",
                                "tool": tc["name"],
                                "args": tc["args"],
                            }
                    if content:
                        yield {"type": "thought", "content": content}
            elif node_name == "tools":
                for msg in node_output.get("messages", []):
                    content = getattr(msg, "content", "")
                    tool_name = getattr(msg, "name", node_name)
                    if content:
                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "content": content,
                        }

# ──────────────────────────────────────────────────────────────
# Main — Run with a mock PR context
# # ──────────────────────────────────────────────────────────────

# MOCK_PR_INPUT = "Please review PR #442 for DPDP compliance."


# def main() -> None:
#     try:
#         agent = get_agent()
#     except EnvironmentError:
#         console.print(
#             "[bold red]ERROR:[/bold red] GOOGLE_API_KEY is not set.\n"
#             "  1. Copy [cyan].env.example[/cyan] → [cyan].env[/cyan]\n"
#             "  2. Paste your Gemini API key from https://aistudio.google.com/apikey\n"
#             "  3. Re-run the script.\n"
#         )
#         sys.exit(1)

#     console.print()
#     console.print(
#         Panel(
#             "[bold]RiTa[/bold] — DPDP Compliance Agent\n"
#             "[dim]LangGraph ReAct · DataHub · sqlglot · Azure DevOps[/dim]",
#             border_style="bright_magenta",
#             padding=(1, 2),
#         )
#     )
#     console.print()
#     console.print(Rule("[bold cyan]Agent Input[/bold cyan]"))
#     console.print(f"\n  [italic]{MOCK_PR_INPUT}[/italic]\n")
#     console.print(Rule("[bold cyan]Agent Execution Trace[/bold cyan]"))
#     console.print()

#     for step in agent.stream(
#         {"messages": [{"role": "user", "content": MOCK_PR_INPUT}]},
#         stream_mode="updates",
#     ):
#         for node_name, node_output in step.items():
#             if node_name == "agent":
#                 for msg in node_output.get("messages", []):
#                     content = getattr(msg, "content", "")
#                     tool_calls = getattr(msg, "tool_calls", [])

#                     if content:
#                         console.print(
#                             Panel(
#                                 content,
#                                 title="🤖 Agent Thought",
#                                 border_style="yellow",
#                                 padding=(0, 1),
#                             )
#                         )
#                     if tool_calls:
#                         for tc in tool_calls:
#                             console.print(
#                                 f"  [bold magenta]⚡ Tool Call:[/bold magenta] "
#                                 f"[cyan]{tc['name']}[/cyan]"
#                                 f"({tc['args']})"
#                             )
#                         console.print()
#             elif node_name == "tools":
#                 for msg in node_output.get("messages", []):
#                     content = getattr(msg, "content", "")
#                     tool_name = getattr(msg, "name", node_name)
#                     if content:
#                         console.print(
#                             Panel(
#                                 content,
#                                 title=f"🔧 {tool_name} → Result",
#                                 border_style="blue",
#                                 padding=(0, 1),
#                             )
#                         )

#     console.print()
#     console.print(Rule("[bold green]Agent Complete[/bold green]"))
#     console.print()


# if __name__ == "__main__":
#     main()
