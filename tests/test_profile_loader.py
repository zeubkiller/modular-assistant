import re

from core.profile_loader import format_food_profile, load_food_profile


def test_loads_valid_food_profile_yaml():
    profile = load_food_profile()

    assert profile["allergies_strictes"] == [
        "arachides",
        "fruits à coque",
        "crustacés",
    ]
    assert profile["preferences"]["a_eviter"] == [
        "plats très sucrés",
        "fritures",
        "coriandre",
    ]
    assert profile["preferences"]["apprecie"] == [
        "cuisine méditerranéenne",
        "plats mijotés",
        "recettes rapides",
    ]


def test_food_profile_prompt_contains_allergies_pantry_and_portions():
    formatted = format_food_profile(load_food_profile())

    for required in (
        "arachides",
        "fruits à coque",
        "crustacés",
        "sauce soja",
        "tomates concassées",
        "130g cru",
        "150g cuit",
        "200g min",
        "10g huile",
    ):
        assert required in formatted


def test_food_profile_prompt_stays_compact():
    formatted = format_food_profile(load_food_profile())
    approximate_tokens = re.findall(r"\w+|[^\w\s]", formatted)

    assert len(formatted.split()) < 180
    assert len(approximate_tokens) < 150
