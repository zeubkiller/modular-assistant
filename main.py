"""Command-line entry point for Portable AI Core."""

import argparse
import importlib
import re
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
import ollama

from core.config_loader import ConfigError, load_config
from core.engine import ConversationEngine
from core.rag import RAGIndex
from profiles.chef.tools.chef_tools import RecipeData, save_recipe_markdown


console = Console()


def extract_recipe_title(answer: str, prompt: str) -> str:
    title_match = re.search(
        r"(?:\*\*Titre\s*:\*\*|^#\s*|Titre\s*:)\s*(.+)",
        answer,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if title_match:
        return re.sub(r"[*#]+", "", title_match.group(1)).strip()

    stopwords = {
        "avec", "des", "du", "le", "les", "une", "un", "pour",
        "dans", "et", "jai", "propose", "recette", "personnes",
    }
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-zÀ-ÿ0-9]+", prompt)
        if word.lower() not in stopwords and len(word) > 2
    ]
    selected = words[:5]
    return "recette-" + "-".join(selected) if selected else "recette-maison"


def save_chef_recipe_if_present(answer: str, prompt: str) -> Path | None:
    return None


def available_chef_tool_schemas(
    schemas: list[dict],
    prompt: str,
    pending_recipe_ideas: bool,
) -> list[dict]:
    """Expose tools only when the user's turn explicitly authorizes their use."""
    allowed_names: set[str] = set()
    if pending_recipe_ideas and re.search(
        r"\b(je choisis|je prends|je valide|je confirme|"
        r"rédige|redige|prépare|prepare|"
        r"(?:premier|première|premiere|deuxième|deuxieme|troisième|troisieme)"
        r"(?: idée| option| recette)?|option [123]|recette [123])\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        allowed_names.add("save_recipe_markdown")

    image_path = re.search(
        r"""(?:^|\s)([~/\w.-]+\.(?:png|jpe?g|webp))\b""",
        prompt,
        flags=re.IGNORECASE,
    )
    if image_path and Path(image_path.group(1)).expanduser().is_file():
        allowed_names.add("analyze_fridge_photo")

    return [
        schema
        for schema in schemas
        if schema.get("function", {}).get("name") in allowed_names
    ]


def has_three_recipe_ideas(answer: str) -> bool:
    numbered_items = re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S", answer)
    return len(numbered_items) >= 3


def is_recipe_approval(prompt: str, pending_recipe_ideas: bool) -> bool:
    return pending_recipe_ideas and bool(
        re.search(
            r"\b(je choisis|je prends|je valide|je confirme|"
            r"rédige|redige|prépare|prepare|"
            r"(?:premier|première|premiere|deuxième|deuxieme|troisième|troisieme)"
            r"(?: idée| option| recette)?|option [123]|recette [123])\b",
            prompt,
            flags=re.IGNORECASE,
        )
    )


def create_approved_recipe(
    engine: ConversationEngine,
    prompt: str,
    model: str,
) -> Path:
    """Generate validated recipe JSON and render it through the profile template."""
    engine.messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=model,
        messages=engine.messages
        + [
            {
                "role": "user",
                "content": (
                    "Rédige la recette correspondant à l'idée choisie. Réponds "
                    "uniquement avec les données correspondant au schéma JSON fourni."
                ),
            }
        ],
        format=RecipeData.model_json_schema(),
    )
    message = getattr(response, "message", None)
    if isinstance(response, dict):
        message = response.get("message")
    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content")
    if not content:
        raise RuntimeError("Ollama n'a pas retourné de recette structurée")

    recipe = RecipeData.model_validate_json(content)
    saved_path = Path(save_recipe_markdown(**recipe.model_dump()))
    engine.messages.append({"role": "assistant", "content": content})
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
                            "prep_time": {"type": "string"},
                            "servings": {"type": "integer", "minimum": 1},
                            "ingredients": {"type": "array", "items": {"type": "string"}},
                            "steps": {"type": "array", "items": {"type": "string"}},
                            "chef_tip": {"type": "string"},
                        },
                        "required": [
                            "title",
                            "prep_time",
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
    rag.build()
    tools, schemas = get_profile_tools(args.profile, profile_dir)
    engine = ConversationEngine(
        config,
        tools=tools,
        tool_schemas=[],
        rag_index=rag,
    )

    console.print(f"[bold green]Profil {args.profile} chargé[/bold green]")
    console.print("Tapez 'exit' ou 'quit' pour quitter.")
    pending_recipe_ideas = False
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
                if is_recipe_approval(prompt, pending_recipe_ideas):
                    saved_path = create_approved_recipe(
                        engine,
                        prompt,
                        config["model"],
                    )
                    console.print(Markdown(saved_path.read_text(encoding="utf-8")))
                    console.print(
                        f"[bold green]Recette enregistrée:[/bold green] {saved_path}"
                    )
                    pending_recipe_ideas = False
                    continue
                engine.tool_schemas = available_chef_tool_schemas(
                    schemas, prompt, pending_recipe_ideas=False
                )
            answer = engine.chat(prompt)
            console.print(Markdown(answer))
            if args.profile == "chef":
                saved = any(
                    message.get("role") == "tool"
                    and message.get("name") == "save_recipe_markdown"
                    and "/output/recipes/" in str(message.get("content", ""))
                    for message in engine.messages
                )
                if saved:
                    pending_recipe_ideas = False
                elif has_three_recipe_ideas(answer):
                    pending_recipe_ideas = True
        except Exception as exc:
            console.print(f"[red]Erreur:[/red] {exc}")


if __name__ == "__main__":
    main()
