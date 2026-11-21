"""Chef profile tools."""

import re
from pathlib import Path
from typing import Any

import ollama
from pydantic import BaseModel, Field, field_validator


class RecipeData(BaseModel):
    """Validated structured recipe supplied by the assistant."""

    title: str = Field(min_length=1)
    prep_time: str = Field(
        min_length=1,
        pattern=r"^\d+\s*(?:min(?:ute)?s?|h(?:eure)?s?)(?:\s+\d+\s*(?:min(?:ute)?s?))?$",
        description="Durée réelle, par exemple '25 minutes' ou '1 h 15 min'.",
    )
    servings: int = Field(gt=0)
    ingredients: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    chef_tip: str = Field(min_length=1)

    @field_validator("title", "prep_time", "chef_tip")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text fields cannot be blank")
        return value

    @field_validator("ingredients")
    @classmethod
    def clean_ingredients(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError("Recipe ingredients cannot be empty")
        return cleaned

    @field_validator("steps")
    @classmethod
    def clean_steps(cls, values: list[str]) -> list[str]:
        cleaned = [
            re.sub(r"^\s*(?:\d+[.)]\s*)+", "", value.strip())
            for value in values
            if value and value.strip()
        ]
        if not cleaned:
            raise ValueError("Recipe steps cannot be empty")
        return cleaned


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


def save_recipe_markdown(
    title: str,
    prep_time: str,
    servings: int,
    ingredients: list[str],
    steps: list[str],
    chef_tip: str,
) -> str:
    """Validate a recipe, render the chef template, and save it as Markdown."""
    recipe = RecipeData(
        title=title,
        prep_time=prep_time,
        servings=servings,
        ingredients=ingredients,
        steps=steps,
        chef_tip=chef_tip,
    )

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "output" / "recipes"
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = (
        repo_root
        / "profiles"
        / "chef"
        / "templates"
        / "recipe_template.md"
    )
    template = template_path.read_text(encoding="utf-8")
    rendered = template
    replacements = {
        "{title}": recipe.title,
        "{prep_time}": recipe.prep_time,
        "{servings}": str(recipe.servings),
        "{ingredients}": "\n".join(f"- {item}" for item in recipe.ingredients),
        "{steps}": "\n".join(
            f"{index}. {step}" for index, step in enumerate(recipe.steps, start=1)
        ),
        "{chef_tip}": recipe.chef_tip,
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)

    stem = re.sub(r"[^a-zA-Z0-9]+", "-", recipe.title.lower()).strip("-")
    stem = stem[:80] or "recette"
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(rendered, encoding="utf-8")
    return str(output_path)
