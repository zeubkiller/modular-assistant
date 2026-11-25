import json
from pathlib import Path

from core.config_loader import load_config
from core.engine import MAX_PROMPT_TOKENS, ConversationEngine
from core.prompt_budget import estimate_message_tokens
from main import (
    _include_photo_ingredients,
    create_approved_recipe,
    generate_recipe_proposals,
    parse_recipe_selection,
)
from profiles.chef.tools.chef_tools import RecipeProposalSet


def proposal_payload():
    return {
        "c": [
            ("Poulet riz", "Simple familial"),
            ("Courgette", "Douce et rôtie"),
        ],
        "e": [
            ("Riz sauté", "Simple et bon"),
        ],
    }


def test_generate_recipe_proposals_uses_fixed_profile_and_short_context(
    monkeypatch,
    mocker,
):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    mock_chat = mocker.patch(
        "main.ollama.chat",
        return_value={"message": {"content": json.dumps(proposal_payload())}},
    )
    rag = mocker.Mock()
    rag.query.return_value = [
        "TITRE: Poulet citron | TAGS: rapide",
        "TITRE: Lentilles | TAGS: mijoté",
        "TITRE: Hors budget | TAGS: à exclure",
    ]
    engine = ConversationEngine(
        load_config("chef"),
        tools={},
        rag_index=rag,
    )

    proposals = generate_recipe_proposals(
        engine,
        "Poulet, courgettes, envie d'un dîner rapide",
    )

    assert isinstance(proposals, RecipeProposalSet)
    assert len(proposals.proposals) == 3
    kwargs = mock_chat.call_args.kwargs
    prompt_messages = kwargs["messages"][:3]
    prompt_content = prompt_messages[-1]["content"]
    assert "dîner rapide" in prompt_content
    system_prompt = prompt_messages[0]["content"]
    assert "arachides" in system_prompt
    assert "sauce soja" in system_prompt
    assert "TITRE: Poulet citron" in prompt_messages[1]["content"]
    assert "TITRE: Lentilles" in prompt_messages[1]["content"]
    assert "Hors budget" not in "\n".join(
        message["content"] for message in prompt_messages
    )
    assert len(prompt_messages) == 3
    rag.query.assert_called_once_with(
        "Poulet, courgettes, envie d'un dîner rapide",
        n_results=2,
    )
    assert kwargs["keep_alive"] == -1
    assert kwargs["options"]["num_ctx"] == 2048
    assert kwargs["options"]["num_thread"] == 4
    assert estimate_message_tokens(prompt_messages) < 600
    assert kwargs["options"]["num_predict"] == 80
    assert kwargs["options"]["temperature"] == 0.0
    assert kwargs["options"]["seed"] == 42


def test_generate_recipe_proposals_retries_invalid_pitch(mocker, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    invalid = proposal_payload()
    invalid["c"][0] = ("Titre", " ".join(["mot"] * 11))
    valid = proposal_payload()
    mock_chat = mocker.patch(
        "main.ollama.chat",
        side_effect=[
            {"message": {"content": json.dumps(invalid)}},
            {"message": {"content": json.dumps(valid)}},
        ],
    )
    engine = ConversationEngine(
        load_config("chef"),
        tools={},
    )

    proposals = generate_recipe_proposals(engine, "Légumes")

    assert len(proposals.proposals) == 3
    assert mock_chat.call_count == 2
    assert all(
        estimate_message_tokens(call.kwargs["messages"]) <= MAX_PROMPT_TOKENS
        for call in mock_chat.call_args_list
    )
    assert "Légumes" in engine.messages[-2]["content"]
    assert "Corrige la réponse précédente" not in engine.messages[-2]["content"]


def test_parse_recipe_selection_accepts_only_one_to_three():
    assert [parse_recipe_selection(str(number)) for number in range(1, 4)] == [
        1, 2, 3
    ]
    assert parse_recipe_selection("4") is None
    assert parse_recipe_selection("je choisis la 2") is None


def test_photo_ingredients_merge_with_typed_ingredients(tmp_path, mocker):
    image = tmp_path / "fridge.jpg"
    image.touch()
    analyze_mock = mocker.patch(
        "main.analyze_fridge_photo",
        return_value="courgettes, yaourt",
    )

    prompt = _include_photo_ingredients(f"{image} et du riz")

    assert str(image) in prompt
    assert "courgettes, yaourt" in prompt
    assert "et du riz" in prompt
    analyze_mock.assert_called_once_with(str(image))


def test_create_approved_recipe_generates_only_selected_proposal(mocker):
    mock_chat = mocker.patch(
        "main.ollama.chat",
        return_value={
            "message": {
                "content": (
                    '{"title":"Wrong title","category":"Classique",'
                    '"prep_time":"15 minutes","cook_time":"20 minutes",'
                    '"servings":4,"ingredients":["riz"],'
                    '"steps":["Rincer le riz.","Faire revenir oignon.",'
                    '"Cuire le riz.","Saler et poivrer."],'
                    '"chef_tip":"Servir chaud."}'
                )
            }
        },
    )
    save_mock = mocker.patch(
        "main.save_recipe_markdown",
        return_value="/tmp/poulet-au-riz.md",
    )
    proposals = RecipeProposalSet.model_validate(
        {
            **proposal_payload()
        }
    )
    engine = ConversationEngine(
        {"model": "qwen2.5-coder:3b", "system_prompt": "Chef."},
        tools={},
    )
    engine.messages = [
        {"role": "system", "content": "Chef."},
        {"role": "user", "content": "Famille et ingrédients"},
        {"role": "assistant", "content": proposals.model_dump_json()},
    ]

    saved = create_approved_recipe(engine, proposals, 3, "qwen2.5-coder:3b")

    assert saved == Path("/tmp/poulet-au-riz.md")
    assert mock_chat.call_args.kwargs["keep_alive"] == -1
    assert mock_chat.call_args.kwargs["options"]["num_ctx"] == 2048
    assert mock_chat.call_args.kwargs["options"]["num_thread"] == 4
    assert mock_chat.call_args.kwargs["options"]["num_predict"] == 260
    assert mock_chat.call_args.kwargs["options"]["temperature"] == 0.0
    assert mock_chat.call_args.kwargs["options"]["seed"] == 42
    save_mock.assert_called_once_with(
        title="Riz sauté",
        category="Exploration",
        prep_time="15 minutes",
        cook_time="20 minutes",
        servings=4,
        ingredients=["riz"],
        steps=[
            "Rincer le riz.",
            "Faire revenir oignon.",
            "Cuire le riz.",
            "Saler et poivrer.",
        ],
        chef_tip="Servir chaud.",
    )
    assert "Riz sauté" in engine.messages[1]["content"]
