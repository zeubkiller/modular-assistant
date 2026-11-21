"""Command-line entry point for Portable AI Core."""

import argparse
import importlib
import re
from pathlib import Path

import ollama
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from core.config_loader import ConfigError, load_config
from core.engine import ConversationEngine
from core.rag import RAGIndex
from profiles.chef.tools.chef_tools import (
    RecipeData,
    RecipeProposalSet,
    analyze_fridge_photo,
    save_recipe_markdown,
)


console = Console()
OLLAMA_DETERMINISTIC_OPTIONS = {"temperature": 0.0, "seed": 42}


def _response_content(response: object) -> str:
    message = getattr(response, "message", None)
    if isinstance(response, dict):
        message = response.get("message")
    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content")
    if not content:
        raise RuntimeError("Ollama n'a pas retourné de contenu")
    return str(content)


def _chef_reference(profile_dir: Path) -> str:
    knowledge_dir = profile_dir / "knowledge"
    family_preferences = (knowledge_dir / "family_preferences.md").read_text(
        encoding="utf-8"
    )
    pantry_staples = (knowledge_dir / "pantry_staples.md").read_text(
        encoding="utf-8"
    )
    return (
        f"PRÉFÉRENCES FAMILIALES:\n{family_preferences}\n\n"
        f"FOND DE PLACARD:\n{pantry_staples}"
    )


def _include_photo_ingredients(prompt: str) -> str:
    photo_match = re.search(
        r"""(?:^|\s)([^\s'"]+\.(?:png|jpe?g|webp))\b""",
        prompt,
        flags=re.IGNORECASE,
    )
    if not photo_match:
        return prompt
    image_path = photo_match.group(1)
    ingredients = analyze_fridge_photo(image_path)
    return (
        f"{prompt}\n\nINGRÉDIENTS DÉTECTÉS SUR LA PHOTO "
        f"({image_path}): {ingredients}"
    )


def generate_recipe_proposals(
    engine: ConversationEngine,
    prompt: str,
    profile_dir: Path,
) -> RecipeProposalSet:
    """Generate five validated, compact ideas grounded in profile knowledge."""
    engine.reset()
    request = (
        f"{_chef_reference(profile_dir)}\n\n"
        f"INGRÉDIENTS, ENVIE ET DIRECTION DE L'UTILISATEUR:\n{prompt}\n\n"
        "Retourne exactement trois idées dans le tableau 'classiques' et deux "
        "dans 'explorations'. Chaque idée contient uniquement title et pitch. "
        "Le pitch ne dépasse jamais dix mots."
    )
    messages = engine.messages + [{"role": "user", "content": request}]
    for attempt in range(2):
        response = engine.ollama.chat(
            model=engine.config["model"],
            messages=messages,
            format=RecipeProposalSet.model_json_schema(),
            options={**OLLAMA_DETERMINISTIC_OPTIONS, "num_predict": 500},
        )
        content = _response_content(response)
        try:
            proposals = RecipeProposalSet.model_validate_json(content)
            break
        except ValidationError as exc:
            if attempt == 1:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Corrige la réponse précédente. Elle doit contenir "
                            "exactement trois idées dans 'classiques' et deux "
                            "dans 'explorations', chacune avec un pitch de dix "
                            f"mots maximum. Erreur: {exc}"
                        ),
                    },
                ]
            )
    engine.messages = engine.messages + [
        {"role": "user", "content": request},
        {"role": "assistant", "content": content},
    ]
    return proposals


def parse_recipe_selection(prompt: str) -> int | None:
    match = re.fullmatch(r"\s*(?:[Nn]°?\s*)?([1-5])\s*[.!]?\s*", prompt)
    return int(match.group(1)) if match else None


def create_approved_recipe(
    engine: ConversationEngine,
    proposals: RecipeProposalSet,
    selected_number: int,
    model: str,
    output_dir: Path | None = None,
) -> Path:
    """Generate only the selected idea and render it using the recipe template."""
    selected = proposals.proposals[selected_number - 1]
    original_request = engine.messages[-2]["content"]
    request = (
        f"{original_request}\n\n"
        f"PROPOSITION CHOISIE ({selected_number}/5):\n"
        f"Catégorie: {selected.category}\n"
        f"Titre: {selected.title}\n"
        f"Pitch: {selected.pitch}\n\n"
        "Rédige uniquement cette recette, pour quatre personnes, avec des "
        "quantités réalistes et des étapes concises."
    )
    messages = [
        engine.messages[0],
        {"role": "user", "content": request},
    ]
    for attempt in range(2):
        response = engine.ollama.chat(
            model=model,
            messages=messages,
            format=RecipeData.model_json_schema(),
            options={**OLLAMA_DETERMINISTIC_OPTIONS, "num_predict": 1200},
        )
        content = _response_content(response)
        try:
            generated = RecipeData.model_validate_json(content)
            break
        except ValidationError as exc:
            if attempt == 1:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Corrige la recette en respectant strictement le "
                            "schéma JSON, les durées chiffrées et tous les champs. "
                            f"Erreur: {exc}"
                        ),
                    },
                ]
            )
    recipe = generated.model_copy(
        update={"title": selected.title, "category": selected.category}
    )
    save_arguments = recipe.model_dump()
    if output_dir is not None:
        save_arguments["output_dir"] = output_dir
    saved_path = Path(save_recipe_markdown(**save_arguments))
    engine.messages = messages + [{"role": "assistant", "content": content}]
    return saved_path


def get_profile_tools(profile_name: str, profile_dir: Path) -> tuple[dict, list[dict]]:
    tools_module = importlib.import_module(f"profiles.{profile_name}.tools.{profile_name}_tools")
    tools: dict = {}
    schemas: list[dict] = []

    if profile_name == "chef":
        tools = {
            "analyze_fridge_photo": tools_module.analyze_fridge_photo,
            "save_recipe_markdown": tools_module.save_recipe_markdown,
        }
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_fridge_photo",
                    "description": "Analyse les ingrédients visibles sur une photo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"image_path": {"type": "string"}},
                        "required": ["image_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_recipe_markdown",
                    "description": "Sauvegarde une recette au format Markdown.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": ["Classique", "Exploration"],
                            },
                            "prep_time": {"type": "string"},
                            "cook_time": {"type": "string"},
                            "servings": {"type": "integer", "minimum": 1},
                            "ingredients": {"type": "array", "items": {"type": "string"}},
                            "steps": {"type": "array", "items": {"type": "string"}},
                            "chef_tip": {"type": "string"},
                        },
                        "required": [
                            "title",
                            "category",
                            "prep_time",
                            "cook_time",
                            "servings",
                            "ingredients",
                            "steps",
                            "chef_tip",
                        ],
                    },
                },
            },
        ]
    return tools, schemas


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable AI Core")
    parser.add_argument("--profile", default="chef", help="Profil à charger")
    args = parser.parse_args()

    try:
        config = load_config(args.profile)
    except ConfigError as exc:
        console.print(f"[red]Erreur de configuration:[/red] {exc}")
        return

    profile_dir = Path(config["_profile_dir"])
    rag = RAGIndex(
        profile_dir / "knowledge",
        Path(".chroma") / args.profile,
        collection_name=f"{args.profile}_knowledge",
        embed_model=config.get("embedding_model", "nomic-embed-text"),
    )
    if args.profile != "chef":
        rag.build()
    tools, _schemas = get_profile_tools(args.profile, profile_dir)
    engine = ConversationEngine(
        config,
        tools=tools,
        tool_schemas=[],
        rag_index=rag if args.profile != "chef" else None,
    )

    console.print(f"[bold green]Profil {args.profile} chargé[/bold green]")
    console.print("Tapez 'exit' ou 'quit' pour quitter.")
    pending_recipe_ideas: RecipeProposalSet | None = None
    while True:
        try:
            prompt = console.input("[bold cyan]Vous >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Au revoir[/dim]")
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        try:
            if args.profile == "chef":
                if pending_recipe_ideas is not None:
                    selected_number = parse_recipe_selection(prompt)
                    if selected_number is None:
                        console.print("[yellow]Choisissez un numéro de 1 à 5.[/yellow]")
                        continue
                    saved_path = create_approved_recipe(
                        engine,
                        pending_recipe_ideas,
                        selected_number,
                        config["model"],
                    )
                    console.print(Markdown(saved_path.read_text(encoding="utf-8")))
                    console.print(
                        f"[bold green]Recette enregistrée:[/bold green] {saved_path}"
                    )
                    pending_recipe_ideas = None
                    continue
                request = _include_photo_ingredients(prompt)
                pending_recipe_ideas = generate_recipe_proposals(
                    engine,
                    request,
                    profile_dir,
                )
                for index, proposal in enumerate(
                    pending_recipe_ideas.proposals,
                    start=1,
                ):
                    console.print(
                        f"{index}. [{proposal.category}] {proposal.title} — "
                        f"{proposal.pitch}",
                        no_wrap=True,
                        overflow="ellipsis",
                    )
                console.print("[dim]Choisissez une idée avec son numéro (1-5).[/dim]")
            else:
                answer = engine.chat(prompt)
                console.print(Markdown(answer))
        except Exception as exc:
            console.print(f"[red]Erreur:[/red] {exc}")


if __name__ == "__main__":
    main()
