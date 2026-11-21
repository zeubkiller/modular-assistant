from main import (
    available_chef_tool_schemas,
    create_approved_recipe,
    extract_recipe_title,
    has_three_recipe_ideas,
    is_recipe_approval,
    save_chef_recipe_if_present,
)


def test_extract_recipe_title():
    answer = "**Titre :** Riz aux tomates\n\n**Ingrédients:**"
    assert extract_recipe_title(answer, "tomates riz") == "Riz aux tomates"


def test_save_chef_recipe_if_present():
    assert save_chef_recipe_if_present("3 idées courtes", "lentilles riz") is None


def test_chef_tools_are_gated_by_user_approval():
    schemas = [
        {"type": "function", "function": {"name": "analyze_fridge_photo"}},
        {"type": "function", "function": {"name": "save_recipe_markdown"}},
    ]

    assert available_chef_tool_schemas(schemas, "tomates et riz", False) == []
    selected = available_chef_tool_schemas(
        schemas,
        "Je choisis la première idée, rédige la recette.",
        True,
    )
    assert [schema["function"]["name"] for schema in selected] == [
        "save_recipe_markdown"
    ]


def test_detects_three_numbered_recipe_ideas():
    answer = "1. Idée un\n2. Idée deux\n3. Idée trois"
    assert has_three_recipe_ideas(answer)


def test_is_recipe_approval_requires_pending_ideas():
    assert is_recipe_approval("Je choisis la première idée.", True)
    assert not is_recipe_approval("Je choisis la première idée.", False)


def test_create_approved_recipe_validates_and_saves(tmp_path, mocker):
    recipe_json = (
        '{"title":"Riz aux tomates","prep_time":"25 minutes","servings":4,'
        '"ingredients":["riz","tomates"],"steps":["Cuire le riz."],'
        '"chef_tip":"Ajouter du basilic."}'
    )
    mocker.patch(
        "main.ollama.chat",
        return_value={"message": {"content": recipe_json}},
    )
    saved_path = tmp_path / "risotto.md"
    save_mock = mocker.patch("main.save_recipe_markdown", return_value=str(saved_path))
    engine = mocker.Mock()
    engine.messages = [{"role": "system", "content": "chef"}]

    result = create_approved_recipe(
        engine,
        "Je choisis la première idée.",
        "qwen2.5-coder:3b",
    )

    assert result == saved_path
    save_mock.assert_called_once_with(
        title="Riz aux tomates",
        prep_time="25 minutes",
        servings=4,
        ingredients=["riz", "tomates"],
        steps=["Cuire le riz."],
        chef_tip="Ajouter du basilic.",
    )
    assert engine.messages[-1]["role"] == "assistant"
