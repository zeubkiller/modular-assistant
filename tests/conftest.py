import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-debug",
        action="store_true",
        default=False,
        help="Print prompts, Ollama responses, and generated recipe Markdown",
    )


@pytest.fixture(autouse=True)
def isolate_unit_tests_from_ollama_process_api(request, mocker):
    if request.node.get_closest_marker("e2e") is None:
        mocker.patch("ollama.ps", return_value={"models": []})
