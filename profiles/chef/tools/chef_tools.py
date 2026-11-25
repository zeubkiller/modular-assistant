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

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(
        min_length=1,
        max_length=16,
        alias="t",
        description="Short dish title, at most four words",
    )
    pitch: str = Field(
        min_length=1,
        max_length=16,
        alias="p",
        description="Short pitch, at most four words",
    )

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
    """Two familiar and one exploratory recipe idea."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    classiques: list[tuple[str, str]] = Field(
        min_length=2,
        max_length=2,
        alias="c",
        description="Two familiar ideas, each as [title, pitch]",
    )
    explorations: list[tuple[str, str]] = Field(
        min_length=1,
        max_length=1,
        alias="e",
        description="One exploratory idea as [title, pitch]",
    )

    @field_validator("classiques", "explorations")
    @classmethod
    def validate_ideas(cls, values: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for title, pitch in values:
            RecipeIdea(title=title.strip(), pitch=pitch.strip())
        return values

    @property
    def proposals(self) -> list["RecipeProposal"]:
        return [
            RecipeProposal(title=title.strip(), pitch=pitch.strip(), category="Classique")
            for title, pitch in self.classiques
        ] + [
            RecipeProposal(
                title=title.strip(),
                pitch=pitch.strip(),
                category="Exploration",
            )
            for title, pitch in self.explorations
        ]


RECIPE_PROPOSAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "c": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 16},
                "minItems": 2,
                "maxItems": 2,
            },
            "minItems": 2,
            "maxItems": 2,
        },
        "e": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 16},
                "minItems": 2,
                "maxItems": 2,
            },
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": ["c", "e"],
    "additionalProperties": False,
}


class RecipeProposal(BaseModel):
    """A compact classified idea ready for display and selection."""

    title: str
    category: Literal["Classique", "Exploration"]
    pitch: str


class RecipeData(BaseModel):
    """Validated structured recipe supplied by the assistant."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1, max_length=40, alias="t")
    category: Literal["Classique", "Exploration"] = Field(alias="c")
    prep_time: str = Field(
        min_length=1,
        pattern=r"^\d+\s*(?:min(?:ute)?s?|h(?:eure)?s?)(?:\s+\d+\s*(?:min(?:ute)?s?))?$",
        alias="p",
    )
    cook_time: str = Field(
        min_length=1,
        pattern=r"^\d+\s*(?:min(?:ute)?s?|h(?:eure)?s?)(?:\s+\d+\s*(?:min(?:ute)?s?))?$",
        alias="k",
    )
    servings: int = Field(gt=0, alias="n")
    ingredients: list[Annotated[str, Field(min_length=1, max_length=48)]] = Field(
        min_length=1,
        max_length=5,
        alias="i",
    )
    steps: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        min_length=4,
        max_length=4,
        alias="s",
    )
    chef_tip: str = Field(min_length=1, max_length=40, alias="h")

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


RECIPE_DATA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "t": {"type": "string", "minLength": 1, "maxLength": 40},
        "c": {"type": "string", "enum": ["Classique", "Exploration"]},
        "p": {"type": "string"},
        "k": {"type": "string"},
        "n": {"type": "integer", "minimum": 1},
        "i": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 48},
            "minItems": 1,
            "maxItems": 5,
        },
        "s": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 64},
            "minItems": 4,
            "maxItems": 4,
        },
        "h": {"type": "string", "minLength": 1, "maxLength": 40},
    },
    "required": ["t", "c", "p", "k", "n", "i", "s", "h"],
    "additionalProperties": False,
}


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
