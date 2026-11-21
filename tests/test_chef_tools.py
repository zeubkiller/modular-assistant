from pathlib import Path

import pytest

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
    title = "Test Soupe aux Lentilles"
    path = Path(save_recipe_markdown(title, "# Recette\n\n1. Cuire les lentilles."))
    try:
        assert path.is_file()
        assert path.name == "test-soupe-aux-lentilles.md"
        assert "lentilles" in path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


def test_save_recipe_markdown_requires_title():
    with pytest.raises(ValueError, match="title is required"):
        save_recipe_markdown("", "contenu")
