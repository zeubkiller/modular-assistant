"""Chef profile tools."""

import re
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import ollama
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.telemetry import record_performance_metric


class RecipeIdea(BaseModel):
    """A compact idea containing only its title and short pitch."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=36)
    pitch: str = Field(min_length=1, max_length=32)

    @field_validator("title")
    @classmethod
    def validate_short_title(cls, value: str) -> str:
        value = value.strip()
        if len(value.split()) > 4:
            raise ValueError("Proposal titles must contain at most 4 words")
        return value

    @field_validator("pitch")
    @classmethod
    def validate_short_pitch(cls, value: str) -> str:
        value = value.strip()
        if len(value.split()) > 4:
            raise ValueError("Proposal pitches must contain at most 4 words")
        return value


class RecipeProposalSet(BaseModel):
    """Exactly three familiar and two exploratory recipe ideas."""

    model_config = ConfigDict(extra="forbid")

    classiques: list[RecipeIdea] = Field(min_length=3, max_length=3)
    explorations: list[RecipeIdea] = Field(min_length=2, max_length=2)

    @property
    def proposals(self) -> list["RecipeProposal"]:
        return [
            RecipeProposal(title=idea.title, pitch=idea.pitch, category="Classique")
            for idea in self.classiques
        ] + [
            RecipeProposal(title=idea.title, pitch=idea.pitch, category="Exploration")
            for idea in self.explorations
        ]


class RecipeProposal(BaseModel):
    """A compact classified idea ready for display and selection."""

    title: str
    category: Literal["Classique", "Exploration"]
    pitch: str


class RecipeData(BaseModel):
    """Validated structured recipe supplied by the assistant."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=60)
    category: Literal["Classique", "Exploration"]
    prep_time: str = Field(
        min_length=1,
        pattern=r"^\d+\s*(?:min(?:ute)?s?|h(?:eure)?s?)(?:\s+\d+\s*(?:min(?:ute)?s?))?$",
        description="Durée réelle, par exemple '25 minutes' ou '1 h 15 min'.",
    )
    cook_time: str = Field(
        min_length=1,
        pattern=r"^\d+\s*(?:min(?:ute)?s?|h(?:eure)?s?)(?:\s+\d+\s*(?:min(?:ute)?s?))?$",
        description="Durée réelle, par exemple '20 minutes' ou '1 h'.",
    )
    servings: int = Field(gt=0)
    ingredients: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        min_length=1,
        max_length=5,
    )
    steps: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        min_length=1,
        max_length=3,
    )
    chef_tip: str = Field(min_length=1, max_length=120)

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
    metrics_path: str | Path | None = None,
) -> str:
    """Analyze a fridge photo with the local Moondream vision model."""
    if not image_path or not isinstance(image_path, str):
        raise FileNotFoundError("Image path is required")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    client = ollama_client or ollama
    started_at = time.perf_counter()
    response = None
    try:
        response = client.chat(
            model="moondream",
            keep_alive=0,
            options={"temperature": 0.0, "seed": 42},
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
    finally:
        record_performance_metric(
            "vision_inference_ms",
            (time.perf_counter() - started_at) * 1000,
            response=response,
            ollama_client=client,
            metrics_path=metrics_path,
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
    category: str,
    prep_time: str,
    cook_time: str,
    servings: int,
    ingredients: list[str],
    steps: list[str],
    chef_tip: str,
    output_dir: str | Path | None = None,
) -> str:
    """Validate a recipe, render the chef template, and save it as Markdown."""
    recipe = RecipeData(
        title=title,
        category=category,
        prep_time=prep_time,
        cook_time=cook_time,
        servings=servings,
        ingredients=ingredients,
        steps=steps,
        chef_tip=chef_tip,
    )

    repo_root = Path(__file__).resolve().parents[3]
    recipes_dir = Path(output_dir) if output_dir is not None else repo_root / "output" / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)

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
        "{category}": recipe.category,
        "{prep_time}": recipe.prep_time,
        "{cook_time}": recipe.cook_time,
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
    output_path = recipes_dir / f"{stem}.md"
    output_path.write_text(rendered, encoding="utf-8")
    return str(output_path)
