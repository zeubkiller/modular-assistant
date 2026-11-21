"""Profile YAML loading and validation."""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a profile configuration is missing or invalid."""


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

    required = ("name", "model", "system_prompt")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ConfigError(f"Missing required configuration fields: {', '.join(missing)}")

    tools = config.get("tools", [])
    if not isinstance(tools, list):
        raise ConfigError("Configuration field 'tools' must be a list")

    config["_profile_dir"] = str(config_path.parent)
    config["_config_path"] = str(config_path)
    return config
