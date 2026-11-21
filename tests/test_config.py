import pytest
import yaml

from core.config_loader import ConfigError, load_config


def test_load_chef_profile():
    config = load_config("chef")
    assert config["name"] == "chef"
    assert config["model"]
    assert config["system_prompt"]
    assert config["tools"] == ["analyze_fridge_photo", "save_recipe_markdown"]
    assert "fond de placard" not in config["system_prompt"].lower() or "3 idées" in config["system_prompt"].lower()


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
