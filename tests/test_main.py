import json
from pathlib import Path

from core.engine import ConversationEngine
from main import (
    _include_photo_ingredients,
    create_approved_recipe,
    generate_recipe_proposals,
    parse_recipe_selection,
)
from profiles.chef.tools.chef_tools import RecipeProposalSet


def proposal_payload():
    return {
        "classiques": [
            {"title": "Poulet au riz", "pitch": "Simple et familial."},
            {"title": "Courgettes rôties", "pitch": "Douces et méditerranéennes."},
            {"title": "Lentilles mijotées", "pitch": "Nourrissant et doux."},
        ],
        "explorations": [
            {"title": "Riz sauté", "pitch": "Inspiré des saveurs asiatiques."},
            {"title": "Curry de légumes", "pitch": "Indien doux et parfumé."},
        ],
    }


def test_generate_recipe_proposals_uses_profile_knowledge(tmp_path, mocker):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "family_preferences.md").write_text(
        "Allergie: arachides", encoding="utf-8"
    )
    (knowledge / "pantry_staples.md").write_text(
        "Disponible: sauce soja", encoding="utf-8"
    )
    mock_chat = mocker.patch(
        "main.ollama.chat",
        return_value={"message": {"content": json.dumps(proposal_payload())}},
    )
    engine = ConversationEngine(
        {"model": "qwen2.5-coder:3b", "system_prompt": "Chef."},
        tools={},
    )

    proposals = generate_recipe_proposals(
        engine,
        "Poulet, courgettes, envie d'un dîner rapide",
        tmp_path,
    )

    assert isinstance(proposals, RecipeProposalSet)
    assert len(proposals.proposals) == 5
    kwargs = mock_chat.call_args.kwargs
    prompt_content = kwargs["messages"][-1]["content"]
    assert "Allergie: arachides" in prompt_content
    assert "Disponible: sauce soja" in prompt_content
    assert "dîner rapide" in prompt_content
    assert kwargs["options"]["temperature"] == 0.0
    assert kwargs["options"]["seed"] == 42


def test_generate_recipe_proposals_retries_invalid_pitch(tmp_path, mocker):
    family = tmp_path / "knowledge"
    family.mkdir()
    (family / "family_preferences.md").write_text("Aucune allergie.", encoding="utf-8")
    (family / "pantry_staples.md").write_text("Riz disponible.", encoding="utf-8")
    invalid = proposal_payload()
    invalid["classiques"][0]["pitch"] = " ".join(["mot"] * 11)
    valid = proposal_payload()
    mock_chat = mocker.patch(
        "main.ollama.chat",
        side_effect=[
            {"message": {"content": json.dumps(invalid)}},
            {"message": {"content": json.dumps(valid)}},
        ],
    )
    engine = ConversationEngine(
        {"model": "qwen2.5-coder:3b", "system_prompt": "Chef."},
        tools={},
    )

    proposals = generate_recipe_proposals(engine, "Légumes", tmp_path)

    assert len(proposals.proposals) == 5
    assert mock_chat.call_count == 2
    assert "Légumes" in engine.messages[-2]["content"]
    assert "Corrige la réponse précédente" not in engine.messages[-2]["content"]


def test_parse_recipe_selection_accepts_only_one_to_five():
    assert [parse_recipe_selection(str(number)) for number in range(1, 6)] == [
        1, 2, 3, 4, 5
    ]
    assert parse_recipe_selection("6") is None
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
                    '"steps":["Cuire le riz."],"chef_tip":"Servir chaud."}'
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

    saved = create_approved_recipe(engine, proposals, 4, "qwen2.5-coder:3b")

    assert saved == Path("/tmp/poulet-au-riz.md")
    assert mock_chat.call_args.kwargs["options"]["temperature"] == 0.0
    assert mock_chat.call_args.kwargs["options"]["seed"] == 42
    save_mock.assert_called_once_with(
        title="Riz sauté",
        category="Exploration",
        prep_time="15 minutes",
        cook_time="20 minutes",
        servings=4,
        ingredients=["riz"],
        steps=["Cuire le riz."],
        chef_tip="Servir chaud.",
    )
    assert "Riz sauté" in engine.messages[1]["content"]
