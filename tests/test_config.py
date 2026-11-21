import pytest
import yaml

from core.config_loader import ConfigError, ConfigurationError, load_config


def test_load_chef_profile():
    config = load_config("chef")
    assert config["profile_name"] == "chef"
    assert config["model"]
    assert config["system_prompt"]
    assert config["tools"] == ["analyze_fridge_photo", "save_recipe_markdown"]
    assert "exactement cinq propositions" in config["system_prompt"]


def test_missing_profile(tmp_path):
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config("missing", base_dir=tmp_path)


def test_invalid_yaml(tmp_path):
    profile = tmp_path / "profiles" / "broken"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("name: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config("broken", base_dir=tmp_path)


def test_missing_required_field(tmp_path):
    profile = tmp_path / "profiles" / "broken"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(yaml.safe_dump({"name": "broken"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing required"):
        load_config("broken", base_dir=tmp_path)


def test_tools_are_required(tmp_path):
    profile = tmp_path / "profiles" / "broken"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "profile_name": "broken",
                "model": "local-model",
                "system_prompt": "Prompt",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="tools"):
        load_config("broken", base_dir=tmp_path)
