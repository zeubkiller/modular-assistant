"""Chef profile tools."""

import re
from pathlib import Path
from typing import Any

import ollama


def analyze_fridge_photo(
    image_path: str,
    ollama_client: Any | None = None,
) -> str:
    """Analyze a fridge photo with the local Moondream vision model."""
    if not image_path or not isinstance(image_path, str):
        raise FileNotFoundError("Image path is required")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    client = ollama_client or ollama
    response = client.chat(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": (
                    "Liste uniquement les ingrédients alimentaires visibles dans cette "
                    "photo, avec une liste simple séparée par des virgules."
                ),
                "images": [str(path.resolve())],
            }
        ],
    )
    message = getattr(response, "message", None)
    if isinstance(response, dict):
        message = response.get("message")
    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content")
    if not content:
        return "Aucun ingrédient n'a pu être identifié."
    return str(content).strip()


def save_recipe_markdown(title: str, content: str) -> str:
    """Save a Markdown recipe under output/recipes and return its path."""
    if not title or not isinstance(title, str):
        raise ValueError("Recipe title is required")
    if content is None:
        raise ValueError("Recipe content is required")

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "output" / "recipes"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    stem = stem[:80] or "recette"
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)
