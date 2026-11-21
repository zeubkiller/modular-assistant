import json
from pathlib import Path

import pytest

from core.config_loader import load_config
from core.engine import ConversationEngine
from main import (
    create_approved_recipe,
    generate_recipe_proposals,
)


MODEL = "qwen2.5-coder:3b"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STEP1_HARD_LIMIT_MS = 15_000
STEP2_HARD_LIMIT_MS = 30_000
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


def _create_profile(tmp_path: Path, family_preferences: str) -> Path:
    profile_dir = tmp_path / "profile"
    knowledge_dir = profile_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "family_preferences.md").write_text(
        family_preferences,
        encoding="utf-8",
    )
    pantry_source = (
        REPOSITORY_ROOT / "profiles" / "chef" / "knowledge" / "pantry_staples.md"
    )
    (knowledge_dir / "pantry_staples.md").write_text(
        pantry_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return profile_dir


def _create_engine() -> ConversationEngine:
    config = load_config("chef")
    config["model"] = MODEL
    return ConversationEngine(config, tools={})


def _assert_proposal_counts(proposals) -> None:
    assert len(proposals.proposals) == 5
    assert [idea.category for idea in proposals.proposals].count("Classique") == 3
    assert [idea.category for idea in proposals.proposals].count("Exploration") == 2
    assert all(len(idea.pitch.split()) <= 4 for idea in proposals.proposals)


def _assert_performance(metrics_path: Path) -> None:
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    step1, step2 = metrics
    assert step1["step1_total_ms"] < STEP1_HARD_LIMIT_MS
    assert step2["step2_total_ms"] < STEP2_HARD_LIMIT_MS
    assert step1["tokens_per_second"] > MIN_TOKENS_PER_SECOND
    assert step2["tokens_per_second"] > MIN_TOKENS_PER_SECOND
    assert max(step1["ram_usage_mb"], step2["ram_usage_mb"]) < MAX_RAM_MB


@pytest.mark.e2e
def test_e2e_full_recipe_workflow(tmp_path, request, capsys):
    profile_dir = _create_profile(
        tmp_path,
        (REPOSITORY_ROOT / "profiles" / "chef" / "knowledge" /
         "family_preferences.md").read_text(encoding="utf-8"),
    )
    engine = _create_engine()
    user_input = "poulet, courgettes, riz; dîner rapide"
    metrics_path = tmp_path / "output" / "logs" / "perf_metrics.jsonl"

    proposals = generate_recipe_proposals(
        engine,
        user_input,
        profile_dir,
        metrics_path=metrics_path,
    )
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
    _show_debug(request, capsys, "STEP 2 SELECTION", "4")
    recipe_path = create_approved_recipe(
        engine,
        proposals,
        selected_number=4,
        model=MODEL,
        output_dir=output_dir,
        metrics_path=metrics_path,
    )

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
def test_e2e_allergy_constraint(tmp_path, request, capsys):
    profile_dir = _create_profile(
        tmp_path,
        "STRICT ALLERGY: arachides. Ne jamais utiliser d'arachides.",
    )
    engine = _create_engine()
    metrics_path = tmp_path / "output" / "logs" / "perf_metrics.jsonl"
    proposals = generate_recipe_proposals(
        engine,
        "poulet, courgettes, riz; exploration asiatique",
        profile_dir,
        metrics_path=metrics_path,
    )
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
    assert proposals.proposals[3].category == "Exploration"

    output_dir = tmp_path / "output" / "recipes"
    recipe_path = create_approved_recipe(
        engine,
        proposals,
        selected_number=4,
        model=MODEL,
        output_dir=output_dir,
        metrics_path=metrics_path,
    )
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
