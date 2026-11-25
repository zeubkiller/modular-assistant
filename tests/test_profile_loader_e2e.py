import ollama
import pytest

from core.config_loader import load_config
from core.engine import (
    MAX_PROMPT_TOKENS,
    OLLAMA_CPU_OPTIONS,
    ConversationEngine,
)
from core.prompt_budget import estimate_message_tokens


@pytest.mark.e2e
def test_ollama_uses_food_profile_as_system_prompt():
    config = load_config("chef")
    engine = ConversationEngine(config, tools={})
    messages = engine.prepare_messages(
        "Réponds uniquement avec les allergies strictes."
    )
    assert all(
        allergen in messages[0]["content"]
        for allergen in ("arachides", "fruits à coque", "crustacés")
    )
    assert "sauce soja" in messages[0]["content"]
    assert estimate_message_tokens(messages) <= MAX_PROMPT_TOKENS

    response = ollama.chat(
        model=config["model"],
        messages=messages,
        options={
            **OLLAMA_CPU_OPTIONS,
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 32,
        },
        keep_alive=config["model_keep_alive"],
    )

    answer = response.message.content.casefold()
    assert all(
        allergen in answer
        for allergen in ("arachides", "fruits à coque", "crustacés")
    )
