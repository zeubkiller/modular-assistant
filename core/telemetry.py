"""Lightweight JSONL telemetry for local inference pipeline stages."""

import json
import resource
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _ollama_model_memory_mb(client: Any | None) -> float:
    if client is None:
        return 0.0
    process_response = client.ps()
    models = _field(process_response, "models", []) or []
    return round(
        sum(float(_field(model, "size", 0) or 0) for model in models)
        / (1024 * 1024),
        2,
    )


def _tokens_per_second(response: Any | None) -> float | None:
    if response is None:
        return None
    token_count = _field(response, "eval_count")
    duration_ns = _field(response, "eval_duration")
    if not token_count or not duration_ns:
        return None
    return round(float(token_count) / (float(duration_ns) / 1_000_000_000), 2)


def _response_timings(response: Any | None) -> dict[str, int]:
    if response is None:
        return {}
    fields = {
        "prompt_eval_count": "prompt_eval_count",
        "eval_count": "eval_count",
        "prompt_eval_ms": "prompt_eval_duration",
        "eval_ms": "eval_duration",
        "model_load_ms": "load_duration",
    }
    return {
        output_name: round(float(value) / 1_000_000)
        if source_name.endswith("_duration")
        else int(value)
        for output_name, source_name in fields.items()
        if (value := _field(response, source_name)) is not None
    }


def record_performance_metric(
    metric_name: str,
    duration_ms: float,
    response: Any | None = None,
    ollama_client: Any | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one stage record to the configured performance JSONL file."""
    process_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    model_memory_mb = _ollama_model_memory_mb(ollama_client)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        metric_name: round(duration_ms, 2),
        "ram_usage_mb": round(process_peak_mb + model_memory_mb, 2),
        "process_peak_ram_mb": round(process_peak_mb, 2),
        "ollama_model_ram_mb": model_memory_mb,
    }

    token_rate = _tokens_per_second(response)
    if token_rate is not None:
        entry["tokens_per_second"] = token_rate
    entry.update(_response_timings(response))

    if metrics_path is None:
        metrics_path = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "logs"
            / "perf_metrics.jsonl"
        )
    output_path = Path(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
