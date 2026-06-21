import json
import os
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message
from openai import OpenAI


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, tool_id: str, name: str, tool_input: dict[str, Any]):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = tool_input


class _OpenRouterResponse:
    def __init__(self, completion: Any):
        choice = completion.choices[0]
        message = choice.message
        self.stop_reason = self._map_stop_reason(choice.finish_reason)
        self.content = []

        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            self.content.append(_TextBlock(content))
        elif isinstance(content, list):
            for item in content:
                if getattr(item, "type", None) == "text":
                    self.content.append(_TextBlock(item.text))

        for tool_call in getattr(message, "tool_calls", []) or []:
            try:
                tool_input = json.loads(tool_call.function.arguments)
            except (TypeError, json.JSONDecodeError):
                tool_input = {
                    "arguments": tool_call.function.arguments,
                }
            self.content.append(
                _ToolUseBlock(
                    tool_call.id,
                    tool_call.function.name,
                    tool_input,
                )
            )

    @staticmethod
    def _map_stop_reason(reason: str | None) -> str:
        if reason == "tool_calls":
            return "tool_use"
        if reason == "stop":
            return "end_turn"
        return reason or "end_turn"


class Claude:
    def __init__(self, model: str):
        self.model = model
        self.provider = (
            "openrouter"
            if os.getenv("OPENROUTER_API_KEY")
            else "anthropic"
        )
        self.client: Any

        if self.provider == "openrouter":
            self.client = OpenAI(
                base_url=os.getenv(
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1",
                ),
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        else:
            self.client = Anthropic()

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Any):
        return "\n".join(
            [
                block.text
                for block in getattr(message, "content", [])
                if getattr(block, "type", None) == "text"
            ]
        )

    def _normalize_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)

            if role == "tool":
                normalized.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id")
                        if isinstance(message, dict)
                        else getattr(message, "tool_call_id", None),
                        "content": message.get("content")
                        if isinstance(message, dict)
                        else getattr(message, "content", None),
                    }
                )
                continue

            if role == "assistant" and isinstance(content, list):
                tool_calls = []
                text_parts = []
                for block in content:
                    block_type = (
                        block.get("type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )
                    if block_type == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id")
                                if isinstance(block, dict)
                                else getattr(block, "id", None),
                                "type": "function",
                                "function": {
                                    "name": block.get("name")
                                    if isinstance(block, dict)
                                    else getattr(block, "name", None),
                                    "arguments": json.dumps(
                                        block.get("input")
                                        if isinstance(block, dict)
                                        else getattr(block, "input", {})
                                    ),
                                },
                            }
                        )
                    elif block_type == "text":
                        text_parts.append(
                            block.get("text")
                            if isinstance(block, dict)
                            else getattr(block, "text", "")
                        )

                message_payload: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                }
                if tool_calls:
                    message_payload["tool_calls"] = tool_calls
                normalized.append(message_payload)
                continue

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    block_type = (
                        block.get("type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )
                    if block_type == "text":
                        text_parts.append(
                            block.get("text")
                            if isinstance(block, dict)
                            else getattr(block, "text", "")
                        )
                normalized.append(
                    {
                        "role": role or "user",
                        "content": "\n".join(text_parts)
                        if text_parts
                        else content,
                    }
                )
            else:
                normalized.append(
                    {
                        "role": role or "user",
                        "content": content if content is not None else "",
                    }
                )

        return normalized

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Any:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
        }

        if stop_sequences:
            params["stop"] = stop_sequences
        else:
            params["stop"] = []

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        if self.provider == "openrouter":
            response = self.client.chat.completions.create(**params)
            return _OpenRouterResponse(response)

        params["max_tokens"] = 8000
        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        message = self.client.messages.create(**params)
        return message
