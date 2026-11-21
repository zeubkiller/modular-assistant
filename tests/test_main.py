from main import extract_recipe_title, save_chef_recipe_if_present


def test_extract_recipe_title():
    answer = "**Titre :** Riz aux tomates\n\n**Ingrédients:**"
    assert extract_recipe_title(answer, "tomates riz") == "Riz aux tomates"


def test_save_chef_recipe_if_present():
    answer = (
        "**Titre :** Test Lentilles\n\n"
        "**Ingrédients:** lentilles, riz\n\n"
        "**Instructions:** mélanger et cuire."
    )
    saved = save_chef_recipe_if_present(answer, "lentilles riz")
    try:
        assert saved is not None
        assert saved.is_file()
        assert saved.name == "test-lentilles.md"
    finally:
        if saved is not None:
            saved.unlink(missing_ok=True)
