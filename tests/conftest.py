import pytest
import ollama

from core.config_loader import load_config
from core.engine import warmup_ollama_model


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


@pytest.fixture(scope="session", autouse=True)
def warmup_ollama_for_e2e(request):
    has_e2e_tests = any(
        item.get_closest_marker("e2e") is not None
        for item in request.session.items
    )
    if not has_e2e_tests:
        yield
        return

    model = load_config("chef")["model"]
    warmup_ollama_model(model, keep_alive=-1)
    ollama.generate(
        model="moondream",
        prompt="",
        keep_alive=0,
        options={"temperature": 0.0, "seed": 42, "num_predict": 1},
    )
    yield
    warmup_ollama_model(model, keep_alive=0)
