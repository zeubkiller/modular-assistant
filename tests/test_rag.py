import json
from unittest.mock import Mock

from core.rag import MAX_QUERY_RESULTS, RAGIndex, _chunks


def test_rag_chunks_are_bounded():
    chunks = _chunks(" ".join(f"w{index}" for index in range(450)))

    assert len(chunks) == 3
    assert all(len(chunk.split()) <= 200 for chunk in chunks)


def test_rag_query_caps_results_and_context_size(tmp_path):
    collection = Mock()
    collection.count.return_value = 10
    collection.query.return_value = {
        "documents": [["word " * 300, "word " * 300, "word " * 300]],
        "metadatas": [[{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}]],
    }
    rag = RAGIndex(
        tmp_path / "knowledge",
        tmp_path / "chroma",
        metrics_path=tmp_path / "logs" / "perf_metrics.jsonl",
    )
    rag._collection = collection
    rag._embed = Mock(return_value=[0.1, 0.2])

    results = rag.query("query text", n_results=20)

    assert len(results) <= MAX_QUERY_RESULTS
    assert sum(len(result.split()) for result in results) <= 600
    assert collection.query.call_args.kwargs["n_results"] == MAX_QUERY_RESULTS
    metrics = json.loads(
        (tmp_path / "logs" / "perf_metrics.jsonl").read_text(encoding="utf-8")
    )
    assert metrics["rag_retrieval_ms"] < 1000


def test_performance_telemetry_records_generation_rate(tmp_path):
    from core.telemetry import record_performance_metric

    metrics_path = tmp_path / "perf_metrics.jsonl"
    metric = record_performance_metric(
        "step1_total_ms",
        123.4,
        response={"eval_count": 16, "eval_duration": 2_000_000_000},
        metrics_path=metrics_path,
    )

    assert metric["tokens_per_second"] == 8.0
    assert metric["eval_count"] == 16
    assert metric["eval_ms"] == 2000
    assert metric["ram_usage_mb"] < 5500
    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert written["step1_total_ms"] == 123.4
