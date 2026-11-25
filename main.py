"""Command-line entry point for Portable AI Core."""

import argparse
import importlib
import re
import time
from pathlib import Path

import ollama
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from core.config_loader import ConfigError, load_config
from core.engine import (
    DEFAULT_MODEL_KEEP_ALIVE,
    OLLAMA_CPU_OPTIONS,
    ConversationEngine,
    warmup_ollama_model,
)
from core.rag import RAGIndex
from core.telemetry import record_performance_metric
from profiles.chef.tools.chef_tools import (
    RECIPE_DATA_JSON_SCHEMA,
    RECIPE_PROPOSAL_JSON_SCHEMA,
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


def _retry_messages(
    engine: ConversationEngine,
    request: str,
    rag_query: str,
    error: ValidationError,
) -> list[dict[str, str]]:
    error_summary = "; ".join(
        f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
        for issue in error.errors(include_input=False)
    )
    retry_prompt = f"{request}\nErreurs à corriger: {error_summary}"
    return list(engine.prepare_messages(retry_prompt, rag_query=rag_query))


def generate_recipe_proposals(
    engine: ConversationEngine,
    prompt: str,
    metrics_path: Path | None = None,
) -> RecipeProposalSet:
    """Generate three validated, compact ideas grounded in profile knowledge."""
    request = (
        f"{prompt}\nJSON: c=2 classiques, e=1 exploration; idées [titre, pitch], "
        "très courtes."
    )
    messages = list(engine.prepare_messages(request, rag_query=prompt))
    started_at = time.perf_counter()
    response = None
    try:
        for attempt in range(2):
            response = engine.ollama.chat(
                model=engine.config["model"],
                messages=messages,
                format=RECIPE_PROPOSAL_JSON_SCHEMA,
                keep_alive=engine.config.get(
                    "model_keep_alive",
                    DEFAULT_MODEL_KEEP_ALIVE,
                ),
                options={
                    **OLLAMA_CPU_OPTIONS,
                    **OLLAMA_DETERMINISTIC_OPTIONS,
                    "num_predict": 80,
                    "num_batch": 256,
                },
            )
            content = _response_content(response)
            try:
                proposals = RecipeProposalSet.model_validate_json(content)
                break
            except ValidationError as exc:
                if attempt == 1:
                    raise
                messages = _retry_messages(engine, request, prompt, exc)
        engine.messages.append({"role": "assistant", "content": content})
        return proposals
    finally:
        record_performance_metric(
            "step1_total_ms",
            (time.perf_counter() - started_at) * 1000,
            response=response,
            ollama_client=engine.ollama,
            metrics_path=metrics_path,
        )


def parse_recipe_selection(prompt: str) -> int | None:
    match = re.fullmatch(r"\s*(?:[Nn]°?\s*)?([1-3])\s*[.!]?\s*", prompt)
    return int(match.group(1)) if match else None


def create_approved_recipe(
    engine: ConversationEngine,
    proposals: RecipeProposalSet,
    selected_number: int,
    model: str,
    output_dir: Path | None = None,
    metrics_path: Path | None = None,
) -> Path:
    """Generate only the selected idea and render it using the recipe template."""
    selected = proposals.proposals[selected_number - 1]
    original_request = next(
        message["content"]
        for message in reversed(engine.messages)
        if message["role"] == "user"
    )
    request = (
        f"{original_request}\n\n"
        f"PROPOSITION CHOISIE ({selected_number}/3):\n"
        f"Catégorie: {selected.category}\n"
        f"Titre: {selected.title}\n"
        f"Pitch: {selected.pitch}\n"
        "JSON court: t=titre, c=catégorie, p/k=durées avec unité "
        "(ex. 10 min), n=portions, "
        "i=ingrédients, s=4 étapes, h=conseil. Pour 4, 5 ingrédients max."
    )
    messages = list(engine.prepare_messages(request, rag_query=original_request))
    started_at = time.perf_counter()
    response = None
    try:
        for attempt in range(2):
            response = engine.ollama.chat(
                model=model,
                messages=messages,
                format=RECIPE_DATA_JSON_SCHEMA,
                keep_alive=engine.config.get(
                    "model_keep_alive",
                    DEFAULT_MODEL_KEEP_ALIVE,
                ),
                options={
                    **OLLAMA_CPU_OPTIONS,
                    **OLLAMA_DETERMINISTIC_OPTIONS,
                    "num_predict": 260,
                    "num_batch": 256,
                },
            )
            content = _response_content(response)
            try:
                generated = RecipeData.model_validate_json(content)
                break
            except ValidationError as exc:
                if attempt == 1:
                    raise
                messages = _retry_messages(
                    engine,
                    request,
                    original_request,
                    exc,
                )
        recipe = generated.model_copy(
            update={"title": selected.title, "category": selected.category}
        )
        save_arguments = recipe.model_dump()
        if output_dir is not None:
            save_arguments["output_dir"] = output_dir
        saved_path = Path(save_recipe_markdown(**save_arguments))
        engine.messages.append({"role": "assistant", "content": content})
        return saved_path
    finally:
        record_performance_metric(
            "step2_total_ms",
            (time.perf_counter() - started_at) * 1000,
            response=response,
            ollama_client=engine.ollama,
            metrics_path=metrics_path,
        )


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
    recipe_cards_only = args.profile == "chef"
    knowledge_dir = profile_dir / "knowledge"
    rag = RAGIndex(
        knowledge_dir / "recipes" if recipe_cards_only else knowledge_dir,
        Path(".chroma") / args.profile,
        collection_name=f"{args.profile}_knowledge",
        embed_model=config.get("embedding_model", "nomic-embed-text"),
        recipe_cards_only=recipe_cards_only,
    )
    rag.build()
    tools, _schemas = get_profile_tools(args.profile, profile_dir)
    engine = ConversationEngine(
        config,
        tools=tools,
        tool_schemas=[],
        rag_index=rag,
    )

    with console.status(f"Chargement du modèle {config['model']}..."):
        warmup_ollama_model(
            config["model"],
            config.get("model_keep_alive", DEFAULT_MODEL_KEEP_ALIVE),
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
                        console.print("[yellow]Choisissez un numéro de 1 à 3.[/yellow]")
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
                console.print("[dim]Choisissez une idée avec son numéro (1-3).[/dim]")
            else:
                answer = engine.chat(prompt)
                console.print(Markdown(answer))
        except Exception as exc:
            console.print(f"[red]Erreur:[/red] {exc}")


if __name__ == "__main__":
    main()
