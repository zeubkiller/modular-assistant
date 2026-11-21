from pathlib import Path

import pytest
from pydantic import ValidationError

from profiles.chef.tools.chef_tools import analyze_fridge_photo, save_recipe_markdown


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
    assert str(image.resolve()) in call["messages"][0]["images"]


def test_save_recipe_markdown(tmp_path, mocker):
    path = Path(save_recipe_markdown(
        title="Soupe aux lentilles",
        prep_time="30 minutes",
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
        assert "**Temps de préparation:** 30 minutes" in rendered
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
            prep_time="20 minutes",
            servings=4,
            ingredients=["riz"],
            steps=["Cuire le riz."],
            chef_tip="Servir chaud.",
        )


def test_save_recipe_markdown_rejects_empty_servings():
    with pytest.raises(ValidationError):
        save_recipe_markdown(
            title="Riz",
            prep_time="20 minutes",
            servings=0,
            ingredients=["riz"],
            steps=["Cuire le riz."],
            chef_tip="Servir chaud.",
        )


def test_save_recipe_markdown_requires_numeric_prep_time():
    with pytest.raises(ValidationError):
        save_recipe_markdown(
            title="Riz",
            prep_time="Facile",
            servings=2,
            ingredients=["riz"],
            steps=["Cuire le riz."],
            chef_tip="Servir chaud.",
        )
