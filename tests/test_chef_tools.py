from pathlib import Path
import json

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
    metrics_path = tmp_path / "perf_metrics.jsonl"
    client = mocker.Mock()
    client.chat.return_value = {"message": {"content": "tomates, oeufs"}}
    client.ps.return_value = {"models": []}

    result = analyze_fridge_photo(
        str(image),
        ollama_client=client,
        metrics_path=metrics_path,
    )

    assert result == "tomates, oeufs"
    client.chat.assert_called_once()
    call = client.chat.call_args.kwargs
    assert call["model"] == "moondream"
    assert call["keep_alive"] == 0
    assert call["options"] == {"temperature": 0.0, "seed": 42}
    assert str(image.resolve()) in call["messages"][0]["images"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["vision_inference_ms"] < 12_000
    assert metrics["ollama_model_ram_mb"] == 0


def test_save_recipe_markdown(tmp_path, mocker):
    path = Path(save_recipe_markdown(
        title="Soupe aux lentilles",
        category="Classique",
        prep_time="30 minutes",
        cook_time="25 minutes",
        servings=4,
        ingredients=["200 g de lentilles", "1 oignon", "1 L de bouillon"],
        steps=[
            "1. Couper l'oignon.",
            "2. Faire revenir l'oignon.",
            "3. Ajouter les lentilles et le bouillon.",
            "4. Cuire jusqu'à tendreté.",
        ],
        chef_tip="Ajouter un filet d'huile d'olive.",
        output_dir=tmp_path / "recipes",
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


def test_proposal_set_requires_two_classics_and_one_exploration():
    validated = RecipeProposalSet.model_validate(
        {
            "c": [
                (f"Idée {index}", "Repas simple.")
                for index in range(1, 3)
            ],
            "e": [("Idée 3", "Inédit")],
        }
    )

    assert len(validated.proposals) == 3
    assert [idea.category for idea in validated.proposals].count("Classique") == 2
    assert [idea.category for idea in validated.proposals].count("Exploration") == 1


def test_proposal_set_rejects_wrong_category_distribution():
    with pytest.raises(ValidationError):
        RecipeProposalSet.model_validate(
            {
                "c": [
                    ("Une", "Simple familial"),
                    ("Deux", "Facile gourmand."),
                ],
                "e": [
                    ("Trois", "Saveurs du monde"),
                    ("Quatre", "Recette originale."),
                ],
            }
        )


def test_proposal_pitch_cannot_exceed_four_words():
    with pytest.raises(ValidationError, match="at most 4 words"):
        RecipeProposalSet.model_validate(
            {
                "c": [
                    ("Idée", "a b c d e"),
                    ("Deux", "Simple familial"),
                ],
                "e": [("Trois", "Saveurs du monde")],
            }
        )
