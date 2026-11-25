"""Load and compact the fixed food profile for model prompts."""

from pathlib import Path
from typing import Any

import yaml


FOOD_PROFILE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "profil_alimentaire.yaml"
)


def load_food_profile(path: str | Path = FOOD_PROFILE_PATH) -> dict[str, Any]:
    """Load the food profile YAML as a mapping."""
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as profile_file:
        profile = yaml.safe_load(profile_file)
    if not isinstance(profile, dict):
        raise ValueError(f"Food profile must be a YAML mapping: {profile_path}")
    return profile


def format_food_profile(profile: dict[str, Any]) -> str:
    """Format food preferences and pantry contents as compact prompt text."""
    preferences = profile["preferences"]
    quantities = profile["reperes_par_personne"]
    pantry = profile["placard_de_base"]

    return (
        f"Allergies strictes: {', '.join(profile['allergies_strictes'])}. "
        f"À éviter: {', '.join(preferences['a_eviter'])}; "
        f"apprécié: {', '.join(preferences['apprecie'])}. "
        "Par personne: "
        f"protéines {quantities['proteines']}, "
        f"féculents {quantities['feculents']}, "
        f"légumes {quantities['legumes']}, "
        f"matières grasses {quantities['matieres_grasses']}. "
        "Placard de base: "
        f"assaisonnements ({', '.join(pantry['assaisonnements'])}); "
        f"aromates ({', '.join(pantry['aromates'])}); "
        f"basiques secs ({', '.join(pantry['basiques_secs'])}). "
        f"Règles: {' '.join(profile['regles'])}"
    )
