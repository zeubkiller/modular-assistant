"""Conversation loop, RAG context, tool calling and self-correction."""

from collections.abc import Callable
import json
from typing import Any

import ollama


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return vars(value)


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class ConversationEngine:
    """Orchestrate an Ollama conversation with RAG and Python tools."""

    def __init__(
        self,
        config: dict[str, Any],
        tools: dict[str, Callable[..., str]],
        tool_schemas: list[dict[str, Any]] | None = None,
        rag_index: Any | None = None,
        ollama_client: Any | None = None,
        max_corrections: int = 2,
    ) -> None:
        self.config = config
        self.tools = tools
        self.tool_schemas = tool_schemas or []
        self.rag_index = rag_index
        self.ollama = ollama_client or ollama
        self.max_corrections = max_corrections
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": config["system_prompt"]}
        ]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.config["system_prompt"]}]

    def _rag_context(self, prompt: str) -> str:
        if self.rag_index is None:
            return ""
        chunks = self.rag_index.query(prompt)
        if not chunks:
            return ""
        return "\n\n".join(chunks)

    @staticmethod
    def _extract_embedded_tool_calls(content: str) -> list[dict[str, Any]]:
        """Extract tool calls serialized as JSON objects inside model text."""
        decoder = json.JSONDecoder()
        calls: list[dict[str, Any]] = []
        position = 0

        while position < len(content):
            start = content.find("{", position)
            if start == -1:
                break
            try:
                value, end = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                position = start + 1
                continue

            position = start + end
            if not isinstance(value, dict):
                continue
            function = value.get("function")
            if isinstance(function, dict):
                calls.append(value)
            elif value.get("name") and isinstance(value.get("arguments"), (dict, str)):
                calls.append({
                    "function": {
                        "name": value["name"],
                        "arguments": value["arguments"],
                    }
                })

        return calls

    def chat(self, prompt: str) -> str:
        """Send a user prompt and handle tool calls and retryable tool errors."""
        context = self._rag_context(prompt)
        content = prompt
        if context:
            content = (
                f"Relevant local knowledge:\n{context}\n\n"
                f"User request:\n{prompt}"
            )

        self.messages.append({"role": "user", "content": content})
        correction_count = 0

        while True:
            kwargs: dict[str, Any] = {
                "model": self.config["model"],
                "messages": self.messages,
            }
            if self.tool_schemas:
                kwargs["tools"] = self.tool_schemas

            response = self.ollama.chat(**kwargs)
            message = _get_field(response, "message")
            if message is None:
                return ""

            tool_calls = _get_field(message, "tool_calls", None) or []
            assistant_content = _get_field(message, "content", "") or ""

            embedded_tool_calls = False
            if not tool_calls and assistant_content:
                tool_calls = self._extract_embedded_tool_calls(assistant_content)
                embedded_tool_calls = bool(tool_calls)

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": assistant_content})
                return assistant_content
            if not embedded_tool_calls:
                self.messages.append(_as_dict(message))
            tool_error_found = False

            for call in tool_calls:
                function = _get_field(call, "function")
                name = _get_field(function, "name", "")
                arguments = _get_field(function, "arguments", {}) or {}
                if isinstance(arguments, str):
                    import json

                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        arguments = {}
                        tool_error = f"Invalid tool arguments JSON: {exc}"
                        tool_error_found = True
                        self.messages.append(
                            {"role": "tool", "name": name, "content": tool_error}
                        )
                        continue

                tool = self.tools.get(name)
                if tool is None:
                    tool_error = f"Unknown tool: {name}"
                    tool_error_found = True
                else:
                    try:
                        tool_error = tool(**arguments)
                    except Exception as exc:
                        tool_error = f"{type(exc).__name__}: {exc}"
                        tool_error_found = True

                self.messages.append(
                    {"role": "tool", "name": name, "content": str(tool_error)}
                )

            if tool_error_found and correction_count < self.max_corrections:
                correction_count += 1
                continue
            if tool_error_found:
                return str(tool_error)
