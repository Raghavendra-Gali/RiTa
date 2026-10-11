"""
RiTa Bedrock Utility — Amazon Bedrock integration for LangGraph and direct LLM calls.

Provides:
  1. get_bedrock_runtime_client()  — boto3 client for Bedrock Runtime with credentials & region resolution.
  2. get_bedrock_chat_llm()        — LangChain ChatBedrockConverse instance for agent reasoning.
  3. invoke_bedrock()              — Direct Converse API caller for single-turn or multi-turn prompts.
  4. invoke_bedrock_raw()          — Low-level invoke_model helper with JSON payload.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse

# Default foundation model
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_AWS_REGION = "ap-south-1"


def get_bedrock_region() -> str:
    """Resolve AWS region from environment variables or default."""
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or DEFAULT_AWS_REGION
    )


def get_bedrock_model_id() -> str:
    """Resolve default Bedrock model ID from environment or default."""
    return os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)


def get_bedrock_runtime_client(
    region_name: str | None = None,
    profile_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    max_retries: int = 3,
) -> Any:
    """Create and return a boto3 bedrock-runtime client.

    Resolves credentials in standard AWS priority:
      1. Explicit arguments
      2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
      3. AWS profile / SSO / IAM role / credentials file
    """
    region = region_name or get_bedrock_region()
    profile = profile_name or os.getenv("AWS_PROFILE")

    session_kwargs: dict[str, Any] = {}
    if profile:
        session_kwargs["profile_name"] = profile
    if aws_access_key_id and aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = aws_access_key_id
        session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

    session = boto3.Session(**session_kwargs)
    config = Config(
        retries={"max_attempts": max_retries, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=60,
    )
    return session.client("bedrock-runtime", region_name=region, config=config)


def get_bedrock_chat_llm(
    model_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    region_name: str | None = None,
    client: Any | None = None,
) -> ChatBedrockConverse:
    """Return a LangChain ChatBedrockConverse model for use in LangGraph agents.

    Args:
        model_id: Bedrock model ARN or ID (defaults to BEDROCK_MODEL_ID or Claude 3.5 Sonnet v2).
        temperature: Sampling temperature (0.0 for deterministic compliance tasks).
        max_tokens: Maximum tokens in response.
        region_name: AWS region.
        client: Optional pre-configured boto3 bedrock-runtime client.

    Returns:
        A ChatBedrockConverse instance compatible with LangChain & LangGraph.
    """
    model = model_id or get_bedrock_model_id()
    region = region_name or get_bedrock_region()
    bedrock_client = client or get_bedrock_runtime_client(region_name=region)

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "region_name": region,
        "client": bedrock_client,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatBedrockConverse(**kwargs)


def invoke_bedrock(
    prompt: str,
    system_prompt: str | None = None,
    model_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    region_name: str | None = None,
    client: Any | None = None,
) -> str:
    """Directly invoke a Bedrock foundation model using the Converse API.

    The Converse API provides a unified interface across all Bedrock foundation
    models (Anthropic Claude, Amazon Nova, Meta Llama, etc.).

    Args:
        prompt: User input message.
        system_prompt: Optional system instructions.
        model_id: Bedrock model ID (defaults to Claude 3.5 Sonnet v2).
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum output tokens.
        region_name: AWS region.
        client: Optional pre-configured boto3 bedrock-runtime client.

    Returns:
        The generated text response.
    """
    model = model_id or get_bedrock_model_id()
    bedrock_client = client or get_bedrock_runtime_client(region_name=region_name)

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    converse_kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": {
            "temperature": temperature,
            "maxTokens": max_tokens,
        },
    }

    if system_prompt:
        converse_kwargs["system"] = [{"text": system_prompt}]

    response = bedrock_client.converse(**converse_kwargs)
    output_message = response.get("output", {}).get("message", {})
    contents = output_message.get("content", [])

    text_parts = [part["text"] for part in contents if "text" in part]
    return "".join(text_parts)


def invoke_bedrock_raw(
    body: dict[str, Any],
    model_id: str | None = None,
    region_name: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Low-level invoke_model call on Bedrock Runtime with arbitrary JSON payload.

    Args:
        body: Python dict payload to be JSON-encoded.
        model_id: Bedrock model ID.
        region_name: AWS region.
        client: Optional pre-configured boto3 client.

    Returns:
        Parsed JSON response dictionary.
    """
    model = model_id or get_bedrock_model_id()
    bedrock_client = client or get_bedrock_runtime_client(region_name=region_name)

    response = bedrock_client.invoke_model(
        modelId=model,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    return json.loads(response["body"].read().decode("utf-8"))
