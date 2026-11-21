from types import SimpleNamespace

import pytest

from core.engine import ConversationEngine


CONFIG = {
    "model": "qwen2.5-coder:3b",
    "system_prompt": "Tu es un chef.",
}


def test_chat_returns_plain_response(mocker):
    client = mocker.Mock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="Bonjour!", tool_calls=None)
    )
    engine = ConversationEngine(CONFIG, tools={}, ollama_client=client)

    assert engine.chat("Bonjour") == "Bonjour!"
    assert engine.messages[-1] == {"role": "assistant", "content": "Bonjour!"}


def test_rag_context_is_injected(mocker):
    rag = mocker.Mock()
    rag.query.return_value = ["[Source: preferences.md]\nPas d'arachides."]
    client = mocker.Mock()
    client.chat.return_value = {"message": {"content": "Compris"}}
    engine = ConversationEngine(CONFIG, tools={}, rag_index=rag, ollama_client=client)

    engine.chat("Peux-tu cuisiner?")

    sent_messages = client.chat.call_args.kwargs["messages"]
    user_message = next(message for message in sent_messages if message["role"] == "user")
    assert "Pas d'arachides" in user_message["content"]


def test_tool_call_is_executed(mocker):
    tool = mocker.Mock(return_value="Recette sauvegardée")
    client = mocker.Mock()
    client.chat.side_effect = [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "save_recipe_markdown",
                                  "arguments": {"title": "Soupe", "content": "# Soupe"}}}
                ],
            }
        },
        {"message": {"content": "La recette est enregistrée.", "tool_calls": None}},
    ]
    engine = ConversationEngine(
        CONFIG,
        tools={"save_recipe_markdown": tool},
        tool_schemas=[{"type": "function", "function": {"name": "save_recipe_markdown"}}],
        ollama_client=client,
    )

    result = engine.chat("Sauvegarde une soupe")

    assert result == "La recette est enregistrée."
    tool.assert_called_once_with(title="Soupe", content="# Soupe")
    assert {"role": "tool", "name": "save_recipe_markdown",
            "content": "Recette sauvegardée"} in engine.messages


def test_tool_exception_triggers_self_correction(mocker):
    tool = mocker.Mock(side_effect=[RuntimeError("disk full"), "saved"])
    client = mocker.Mock()
    client.chat.side_effect = [
        {"message": {"tool_calls": [{"function": {"name": "save_recipe_markdown",
                                                  "arguments": {"title": "T", "content": "C"}}}],
                     "content": ""}},
        {"message": {"tool_calls": [{"function": {"name": "save_recipe_markdown",
                                                  "arguments": {"title": "T", "content": "C"}}}],
                     "content": ""}},
        {"message": {"tool_calls": None, "content": "Erreur résolue"}},
    ]
    engine = ConversationEngine(
        CONFIG,
        tools={"save_recipe_markdown": tool},
        ollama_client=client,
    )

    result = engine.chat("Sauvegarde")

    assert result == "Erreur résolue"
    assert tool.call_count == 2


def test_embedded_json_tool_call_is_executed(mocker):
    tool = mocker.Mock(return_value="Recette enregistrée")
    client = mocker.Mock()
    client.chat.side_effect = [
        {
            "message": {
                "content": (
                    '{"name": "save_recipe_markdown", '
                    '"arguments": {"title": "Riz aux tomates", "content": "# Recette"}}'
                ),
                "tool_calls": None,
            }
        },
        {"message": {"content": "Recette enregistrée.", "tool_calls": None}},
    ]
    engine = ConversationEngine(
        CONFIG,
        tools={"save_recipe_markdown": tool},
        ollama_client=client,
    )

    result = engine.chat("Tomates, riz")

    assert result == "Recette enregistrée."
    tool.assert_called_once_with(title="Riz aux tomates", content="# Recette")
