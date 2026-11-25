"""Conservative lexical estimate for local prompt-token budgets."""

import re
from collections.abc import Mapping, Sequence
from typing import Any


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Estimate tokens by counting words and punctuation separately."""
    return len(_TOKEN_PATTERN.findall(text))


def estimate_message_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimate message tokens including a small per-message framing allowance."""
    return sum(
        estimate_tokens(str(message.get("content", ""))) + 4
        for message in messages
    )
