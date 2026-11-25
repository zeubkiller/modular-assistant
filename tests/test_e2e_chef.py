import json
from pathlib import Path

import pytest
from PIL import Image

from core.config_loader import load_config
from core.engine import (
    MAX_PROMPT_TOKENS,
    MAX_RAG_CONTEXT_TOKENS,
    ConversationEngine,
)
from core.prompt_budget import estimate_message_tokens
from core.rag import RAGIndex
from main import (
    create_approved_recipe,
    generate_recipe_proposals,
)
from profiles.chef.tools.chef_tools import analyze_fridge_photo


STEP1_HARD_LIMIT_MS = 15_000
STEP2_HARD_LIMIT_MS = 30_000
VISION_HARD_LIMIT_MS = 12_000
RAG_HARD_LIMIT_MS = 1_000
MIN_TOKENS_PER_SECOND = 4.0
MAX_RAM_MB = 5_500
FORBIDDEN_ALLERGENS = (
    "arachide",
    "cacahuète",
    "cacahuetes",
    "peanut",
    "cashew",
    "almond",
)


def _show_debug(request, capsys, label: str, content: str) -> None:
    if request.config.getoption("--e2e-debug"):
        with capsys.disabled():
            print(f"\n--- {label} ---\n{content}\n", flush=True)


@pytest.fixture(scope="module")
def recipe_rag(tmp_path_factory):
    config = load_config("chef")
    profile_dir = Path(config["_profile_dir"])
    rag = RAGIndex(
        profile_dir / "knowledge" / "recipes",
        tmp_path_factory.mktemp("chef-rag"),
        collection_name="chef_e2e_recipes",
        embed_model=config.get("embedding_model", "nomic-embed-text"),
        recipe_cards_only=True,
    )
    assert rag.build() == 3
    return rag


def _create_engine(rag_index: RAGIndex, system_context: str = "") -> ConversationEngine:
    config = load_config("chef")
    if system_context:
        config["system_prompt"] = f"{config['system_prompt']}\n{system_context}"
    return ConversationEngine(config, tools={}, rag_index=rag_index)


def _assert_proposal_counts(proposals) -> None:
    assert len(proposals.proposals) == 3
    assert [idea.category for idea in proposals.proposals].count("Classique") == 2
    assert [idea.category for idea in proposals.proposals].count("Exploration") == 1
    assert all(len(idea.pitch.split()) <= 4 for idea in proposals.proposals)


def _assert_prompt_features(engine: ConversationEngine) -> None:
    prompt_messages = engine.messages[:-1]
    system_prompt = prompt_messages[0]["content"]
    assert all(
        allergen in system_prompt
        for allergen in ("arachides", "fruits à coque", "crustacés")
    )
    assert "sauce soja" in system_prompt

    rag_messages = [
        message
        for message in prompt_messages[1:]
        if message["role"] == "system"
    ]
    assert len(rag_messages) == 1
    rag_content = rag_messages[0]["content"]
    assert rag_content.startswith("Exemples culinaires (RAG):")
    assert 1 <= rag_content.count("TITRE:") <= 2
    assert all(
        field in rag_content
        for field in ("TAGS:", "INGRÉDIENTS:", "ÉTAPES:")
    )
    assert estimate_message_tokens(rag_messages) <= MAX_RAG_CONTEXT_TOKENS
    assert estimate_message_tokens(prompt_messages) <= MAX_PROMPT_TOKENS


def _assert_performance(metrics_path: Path) -> None:
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    by_stage = {
        name: entry
        for entry in metrics
        for name in (
            "rag_retrieval_ms",
            "step1_total_ms",
            "step2_total_ms",
        )
        if name in entry
    }
    step1 = by_stage["step1_total_ms"]
    step2 = by_stage["step2_total_ms"]
    assert step1["step1_total_ms"] < STEP1_HARD_LIMIT_MS
    assert step2["step2_total_ms"] < STEP2_HARD_LIMIT_MS
    if "rag_retrieval_ms" in by_stage:
        assert by_stage["rag_retrieval_ms"]["rag_retrieval_ms"] < RAG_HARD_LIMIT_MS
    assert step1["tokens_per_second"] > MIN_TOKENS_PER_SECOND
    assert step2["tokens_per_second"] > MIN_TOKENS_PER_SECOND
    assert max(step1["ram_usage_mb"], step2["ram_usage_mb"]) < MAX_RAM_MB


@pytest.mark.e2e
def test_e2e_chef_two_stage_pipeline(tmp_path, request, capsys, recipe_rag):
    engine = _create_engine(recipe_rag)
    user_input = "poulet, courgettes, riz; dîner rapide"
    metrics_path = tmp_path / "output" / "logs" / "perf_metrics.jsonl"

    proposals = generate_recipe_proposals(
        engine,
        user_input,
        metrics_path=metrics_path,
    )
    _assert_prompt_features(engine)
    _show_debug(
        request, capsys, "STEP 1 INPUT", engine.messages[-2]["content"]
    )
    _show_debug(
        request,
        capsys,
        "STEP 1 OLLAMA RESPONSE",
        proposals.model_dump_json(indent=2),
    )
    _assert_proposal_counts(proposals)

    output_dir = tmp_path / "output" / "recipes"
    _show_debug(request, capsys, "STEP 2 SELECTION", "3")
    recipe_path = create_approved_recipe(
        engine,
        proposals,
        selected_number=3,
        model=engine.config["model"],
        output_dir=output_dir,
        metrics_path=metrics_path,
    )
    _assert_prompt_features(engine)

    assert recipe_path.is_file()
    assert recipe_path.parent == output_dir
    markdown = recipe_path.read_text(encoding="utf-8")
    for section in (
        "# ",
        "**Catégorie:**",
        "**Temps de préparation:**",
        "**Temps de cuisson:**",
        "**Nombre de personnes:**",
        "## Ingrédients",
        "## Étapes",
        "## Astuce du chef",
    ):
        assert section in markdown
    _assert_performance(metrics_path)
    _show_debug(request, capsys, "STEP 2 INPUT", engine.messages[-2]["content"])
    _show_debug(request, capsys, "STEP 2 OLLAMA RESPONSE", engine.messages[-1]["content"])
    _show_debug(request, capsys, "FINAL MARKDOWN", markdown)


@pytest.mark.e2e
def test_e2e_strict_allergy_enforcement(tmp_path, request, capsys, recipe_rag):
    engine = _create_engine(
        recipe_rag,
        "STRICT ALLERGY: arachides. Ne jamais utiliser d'arachides."
    )
    metrics_path = tmp_path / "output" / "logs" / "perf_metrics.jsonl"
    proposals = generate_recipe_proposals(
        engine,
        "poulet, courgettes, riz; exploration asiatique",
        metrics_path=metrics_path,
    )
    _assert_prompt_features(engine)
    _show_debug(
        request, capsys, "ALLERGY STEP 1 INPUT", engine.messages[-2]["content"]
    )
    _show_debug(
        request,
        capsys,
        "ALLERGY STEP 1 OLLAMA RESPONSE",
        proposals.model_dump_json(indent=2),
    )
    _assert_proposal_counts(proposals)

    proposal_text = " ".join(
        f"{idea.title} {idea.pitch}" for idea in proposals.proposals
    ).casefold()
    assert not any(allergen in proposal_text for allergen in FORBIDDEN_ALLERGENS)
    assert proposals.proposals[2].category == "Exploration"

    output_dir = tmp_path / "output" / "recipes"
    recipe_path = create_approved_recipe(
        engine,
        proposals,
        selected_number=3,
        model=engine.config["model"],
        output_dir=output_dir,
        metrics_path=metrics_path,
    )
    _assert_prompt_features(engine)
    markdown = recipe_path.read_text(encoding="utf-8")
    recipe_text = markdown.casefold()
    assert not any(allergen in recipe_text for allergen in FORBIDDEN_ALLERGENS)
    _assert_performance(metrics_path)
    _show_debug(
        request,
        capsys,
        "ALLERGY STEP 2 INPUT",
        engine.messages[-2]["content"],
    )
    _show_debug(
        request,
        capsys,
        "ALLERGY STEP 2 OLLAMA RESPONSE",
        engine.messages[-1]["content"],
    )
    _show_debug(request, capsys, "ALLERGY FINAL MARKDOWN", markdown)


@pytest.mark.e2e
def test_e2e_vision_to_recipe_flow(tmp_path, request, capsys):
    image_path = tmp_path / "sample_fridge.png"
    Image.new("RGB", (64, 64), "#d8b59e").save(image_path, format="PNG")
    metrics_path = tmp_path / "output" / "logs" / "perf_metrics.jsonl"

    detected = analyze_fridge_photo(
        str(image_path),
        metrics_path=metrics_path,
    )
    merged_prompt = (
        f"{image_path} plus carotte et crème fraîche\n"
        f"INGRÉDIENTS DÉTECTÉS SUR LA PHOTO ({image_path}): {detected}"
    )
    _show_debug(request, capsys, "VISION MERGED PROMPT", merged_prompt)

    assert str(image_path) in merged_prompt
    assert "INGRÉDIENTS DÉTECTÉS SUR LA PHOTO" in merged_prompt
    assert "carotte et crème fraîche" in merged_prompt

    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    vision = next(
        entry for entry in metrics if "vision_inference_ms" in entry
    )
    assert vision["vision_inference_ms"] < VISION_HARD_LIMIT_MS
