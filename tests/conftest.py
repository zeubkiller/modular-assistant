def pytest_addoption(parser):
    parser.addoption(
        "--e2e-debug",
        action="store_true",
        default=False,
        help="Print prompts, Ollama responses, and generated recipe Markdown",
    )
