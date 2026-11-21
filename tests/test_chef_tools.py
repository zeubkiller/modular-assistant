from pathlib import Path

import pytest
from pydantic import ValidationError

from profiles.chef.tools.chef_tools import (
    RecipeProposalSet,
    analyze_fridge_photo,
    save_recipe_markdown,
)


def test_analyze_fridge_photo_missing_file():
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        analyze_fridge_photo("/does/not/exist.jpg")


def test_analyze_fridge_photo_uses_moondream(tmp_path, mocker):
    image = tmp_path / "fridge.jpg"
    image.write_bytes(b"fake-image")
    client = mocker.Mock()
    client.chat.return_value = {"message": {"content": "tomates, oeufs"}}

    result = analyze_fridge_photo(str(image), ollama_client=client)

    assert result == "tomates, oeufs"
    client.chat.assert_called_once()
    call = client.chat.call_args.kwargs
    assert call["model"] == "moondream"
    assert call["keep_alive"] == 0
    assert str(image.resolve()) in call["messages"][0]["images"]


def test_save_recipe_markdown(tmp_path, mocker):
    path = Path(save_recipe_markdown(
        title="Soupe aux lentilles",
        category="Classique",
        prep_time="30 minutes",
        cook_time="25 minutes",
        servings=4,
        ingredients=["200 g de lentilles", "1 oignon", "1 L de bouillon"],
        steps=["1. Couper l'oignon.", "2. Cuire les lentilles dans le bouillon."],
        chef_tip="Ajouter un filet d'huile d'olive.",
    ))
    try:
        assert path.is_file()
        assert path.name == "soupe-aux-lentilles.md"
        rendered = path.read_text(encoding="utf-8")
        assert "# Soupe aux lentilles" in rendered
        assert "**Catégorie:** Classique" in rendered
        assert "**Temps de préparation:** 30 minutes" in rendered
        assert "**Temps de cuisson:** 25 minutes" in rendered
        assert "**Nombre de personnes:** 4" in rendered
        assert "- 200 g de lentilles" in rendered
        assert "1. Couper l'oignon." in rendered
        assert "1. 1. Couper" not in rendered
        assert "Ajouter un filet d'huile d'olive." in rendered
    finally:
        path.unlink(missing_ok=True)


def test_save_recipe_markdown_requires_title():
    with pytest.raises(ValidationError):
        save_recipe_markdown(
            title="",
            category="Classique",
            prep_time="20 minutes",
            cook_time="20 minutes",
            servings=4,
            ingredients=["riz"],
            steps=["Cuire le riz."],
            chef_tip="Servir chaud.",
        )


def test_save_recipe_markdown_rejects_empty_servings():
    with pytest.raises(ValidationError):
        save_recipe_markdown(
            title="Riz",
            category="Classique",
            prep_time="20 minutes",
            cook_time="20 minutes",
            servings=0,
            ingredients=["riz"],
            steps=["Cuire le riz."],
            chef_tip="Servir chaud.",
        )


def test_save_recipe_markdown_requires_numeric_prep_time():
    with pytest.raises(ValidationError):
        save_recipe_markdown(
            title="Riz",
            category="Classique",
            prep_time="Facile",
            cook_time="20 minutes",
            servings=2,
            ingredients=["riz"],
            steps=["Cuire le riz."],
            chef_tip="Servir chaud.",
        )


def test_proposal_set_requires_three_classics_and_two_explorations():
    validated = RecipeProposalSet.model_validate(
        {
            "classiques": [
                {"title": f"Idée {index}", "pitch": "Repas simple et familial."}
                for index in range(1, 4)
            ],
            "explorations": [
                {"title": f"Idée {index}", "pitch": "Saveurs nouvelles et douces."}
                for index in range(4, 6)
            ],
        }
    )

    assert len(validated.proposals) == 5
    assert [idea.category for idea in validated.proposals].count("Classique") == 3
    assert [idea.category for idea in validated.proposals].count("Exploration") == 2


def test_proposal_set_rejects_wrong_category_distribution():
    with pytest.raises(ValidationError):
        RecipeProposalSet.model_validate(
            {
                "classiques": [
                    {"title": "Une", "pitch": "Simple et familial."},
                    {"title": "Deux", "pitch": "Facile et gourmand."},
                ],
                "explorations": [
                    {"title": "Trois", "pitch": "Saveurs du monde."},
                    {"title": "Quatre", "pitch": "Une recette originale."},
                ],
            }
        )


def test_proposal_pitch_cannot_exceed_ten_words():
    with pytest.raises(ValidationError, match="at most 10 words"):
        RecipeProposalSet.model_validate(
            {
                "classiques": [
                    {"title": "Idée", "pitch": " ".join(["mot"] * 11)},
                    {"title": "Deux", "pitch": "Simple et familial."},
                    {"title": "Trois", "pitch": "Facile et gourmand."},
                ],
                "explorations": [
                    {"title": "Quatre", "pitch": "Saveurs du monde."},
                    {"title": "Cinq", "pitch": "Une recette originale."},
                ],
            }
        )
