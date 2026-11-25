"""Profile YAML loading and validation."""

import os
from pathlib import Path
from typing import Any

import yaml

from core.profile_loader import format_food_profile, load_food_profile


class ConfigurationError(ValueError):
    """Raised when a profile configuration is missing or invalid."""


ConfigError = ConfigurationError


def load_config(profile: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a profile config.yaml file."""
    profile_path = Path(profile)
    if not profile_path.is_absolute():
        root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
        profile_path = root / "profiles" / profile_path

    config_path = profile_path / "config.yaml" if profile_path.is_dir() else profile_path
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("The profile configuration must be a YAML mapping")

    model_override = os.environ.get("OLLAMA_MODEL")
    if model_override is not None:
        model_override = model_override.strip()
        if not model_override:
            raise ConfigError("Environment variable OLLAMA_MODEL must not be empty")
        config["model"] = model_override

    if "profile_name" not in config and config.get("name"):
        config["profile_name"] = config["name"]

    required = ("profile_name", "model", "system_prompt", "tools")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ConfigError(f"Missing required configuration fields: {', '.join(missing)}")

    tools = config["tools"]
    if not isinstance(tools, list):
        raise ConfigError("Configuration field 'tools' must be a list")

    if config["profile_name"] == "chef":
        fixed_profile = format_food_profile(load_food_profile())
        config["system_prompt"] = f"{fixed_profile}\n\n{config['system_prompt']}"

    config["_profile_dir"] = str(config_path.parent)
    config["_config_path"] = str(config_path)
    return config
