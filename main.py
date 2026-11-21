"""Command-line entry point for Portable AI Core."""

import argparse
import importlib
import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from core.config_loader import ConfigError, load_config
from core.engine import ConversationEngine
from core.rag import RAGIndex


console = Console()


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
                            "content": {"type": "string"},
                        },
                        "required": ["title", "content"],
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
    rag.build()
    tools, schemas = get_profile_tools(args.profile, profile_dir)
    engine = ConversationEngine(
        config,
        tools=tools,
        tool_schemas=schemas,
        rag_index=rag,
    )

    console.print(f"[bold green]Profil {args.profile} chargé[/bold green]")
    console.print("Tapez 'exit' ou 'quit' pour quitter.")
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
            answer = engine.chat(prompt)
            console.print(Markdown(answer))
        except Exception as exc:
            console.print(f"[red]Erreur:[/red] {exc}")


if __name__ == "__main__":
    main()
