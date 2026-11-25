"""Conversation loop, RAG context, tool calling and self-correction."""

from collections.abc import Callable
import json
from typing import Any

import ollama

from core.prompt_budget import estimate_message_tokens

DEFAULT_MODEL_KEEP_ALIVE = -1
OLLAMA_CPU_OPTIONS = {"num_ctx": 2048, "num_thread": 4}
MAX_PROMPT_TOKENS = 550
MAX_RAG_CONTEXT_TOKENS = 250
MAX_RAG_RESULTS = 2
RAG_CONTEXT_PREFIX = "Exemples culinaires (RAG):\n"


def warmup_ollama_model(
    model: str,
    keep_alive: float | str,
    ollama_client: Any | None = None,
) -> None:
    """Load an Ollama model with an empty prompt before latency-sensitive work."""
    client = ollama_client or ollama
    client.generate(
        model=model,
        prompt="",
        keep_alive=keep_alive,
        options={
            **OLLAMA_CPU_OPTIONS,
            "num_predict": 1,
            "temperature": 0.0,
            "seed": 42,
        },
    )


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
        max_tool_rounds: int = 3,
    ) -> None:
        self.config = config
        self.tools = tools
        self.tool_schemas = tool_schemas or []
        self.rag_index = rag_index
        self.ollama = ollama_client or ollama
        self.max_corrections = max_corrections
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": config["system_prompt"]}
        ]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.config["system_prompt"]}]

    def _rag_context(self, prompt: str) -> list[str]:
        if self.rag_index is None:
            return []
        chunks = self.rag_index.query(prompt, n_results=MAX_RAG_RESULTS)
        selected = list(chunks[:MAX_RAG_RESULTS])
        while selected and estimate_message_tokens(
            [
                {
                    "role": "system",
                    "content": RAG_CONTEXT_PREFIX + "\n".join(selected),
                }
            ]
        ) > MAX_RAG_CONTEXT_TOKENS:
            selected.pop()
        return selected

    def prepare_messages(
        self,
        prompt: str,
        rag_query: str | None = None,
    ) -> list[dict[str, str]]:
        """Build a bounded request with the fixed system prompt before RAG."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.config["system_prompt"]}
        ]
        rag_chunks = self._rag_context(rag_query if rag_query is not None else prompt)
        if rag_chunks:
            messages.append(
                {
                    "role": "system",
                    "content": RAG_CONTEXT_PREFIX + "\n".join(rag_chunks),
                }
            )
        messages.append({"role": "user", "content": prompt})

        while (
            len(messages) > 2
            and estimate_message_tokens(messages) > MAX_PROMPT_TOKENS
        ):
            messages.pop(-2)
        if estimate_message_tokens(messages) > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"Request exceeds the {MAX_PROMPT_TOKENS}-token prompt budget"
            )

        self.messages = messages
        return messages

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
        self.prepare_messages(prompt)
        correction_count = 0
        tool_rounds = 0
        completed_calls: dict[str, str] = {}
        last_tool_result = ""

        while True:
            if tool_rounds >= self.max_tool_rounds:
                return last_tool_result
            kwargs: dict[str, Any] = {
                "model": self.config["model"],
                "messages": self.messages,
                "keep_alive": self.config.get(
                    "model_keep_alive",
                    DEFAULT_MODEL_KEEP_ALIVE,
                ),
                "options": OLLAMA_CPU_OPTIONS,
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
            tool_rounds += 1
            tool_error_found = False

            for call in tool_calls:
                function = _get_field(call, "function")
                name = _get_field(function, "name", "")
                arguments = _get_field(function, "arguments", {}) or {}
                if isinstance(arguments, str):
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

                signature = json.dumps(
                    {"name": name, "arguments": arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                if signature in completed_calls:
                    tool_result = completed_calls[signature]
                else:
                    tool = self.tools.get(name)
                    if tool is None:
                        tool_result = f"Unknown tool: {name}"
                        tool_error_found = True
                    else:
                        try:
                            tool_result = str(tool(**arguments))
                            completed_calls[signature] = tool_result
                        except Exception as exc:
                            tool_result = f"{type(exc).__name__}: {exc}"
                            tool_error_found = True
                last_tool_result = str(tool_result)

                self.messages.append(
                    {"role": "tool", "name": name, "content": str(tool_result)}
                )

            if tool_error_found and correction_count < self.max_corrections:
                correction_count += 1
                continue
            if tool_error_found:
                return last_tool_result
