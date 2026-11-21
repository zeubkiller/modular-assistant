"""Local RAG indexing and retrieval backed by ChromaDB and Ollama."""

from pathlib import Path
import time

import pymupdf
import ollama

from core.telemetry import record_performance_metric


MAX_CHUNK_WORDS = 200
MAX_CONTEXT_WORDS = 600
MAX_QUERY_RESULTS = 3


def _chunks(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[index:index + max_words])
        for index in range(0, len(words), max_words)
        if words[index:index + max_words]
    ]

def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        with pymupdf.open(path) as document:
            return "\n".join(page.get_text() for page in document)
    return ""


class RAGIndex:
    """Indexes profile knowledge files and retrieves relevant chunks."""

    def __init__(
        self,
        knowledge_dir: str | Path,
        persist_dir: str | Path,
        collection_name: str = "profile_knowledge",
        embed_model: str = "nomic-embed-text",
        metrics_path: str | Path | None = None,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self._client = None
        self._collection = None

    def _embed(self, text: str) -> list[float]:
        result = ollama.embed(model=self.embed_model, input=text)
        embeddings = getattr(result, "embeddings", None)
        if embeddings is None and isinstance(result, dict):
            embeddings = result.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama did not return an embedding")
        return list(embeddings[0])

    def build(self) -> int:
        """Build or refresh a Chroma collection. Returns the chunk count."""
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed") from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        chunks: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, str]] = []
        if self.knowledge_dir.is_dir():
            for path in sorted(self.knowledge_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                    text = _extract_text(path)
                    for index, chunk_text in enumerate(_chunks(text)):
                        chunks.append(chunk_text)
                        ids.append(f"{path.name}-{index}")
                        metadatas.append({"source": str(path)})

        if chunks:
            embeddings = [self._embed(chunk) for chunk in chunks]
            self._collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        return len(chunks)

    def query(self, text: str, n_results: int = 4) -> list[str]:
        """Return the most relevant indexed document chunks."""
        started_at = time.perf_counter()
        try:
            if self._collection is None:
                try:
                    import chromadb

                    self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                    self._collection = self._client.get_or_create_collection(
                        name=self.collection_name
                    )
                except ImportError as exc:
                    raise RuntimeError("ChromaDB is not installed") from exc

            collection_count = self._collection.count()
            if collection_count == 0:
                return []
            embedding = self._embed(text)
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(max(1, n_results), MAX_QUERY_RESULTS, collection_count),
                include=["documents", "metadatas"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            context: list[str] = []
            words_left = MAX_CONTEXT_WORDS
            for document, metadata in zip(
                documents[:MAX_QUERY_RESULTS],
                metadatas[:MAX_QUERY_RESULTS],
            ):
                source = metadata.get("source", "unknown")
                chunk = f"[Source: {source}]\n{document}"
                chunk_words = chunk.split()
                if len(chunk_words) > words_left:
                    chunk = " ".join(chunk_words[:words_left])
                if chunk:
                    context.append(chunk)
                    words_left -= len(chunk.split())
                if words_left <= 0:
                    break
            return context
        finally:
            record_performance_metric(
                "rag_retrieval_ms",
                (time.perf_counter() - started_at) * 1000,
                ollama_client=ollama,
                metrics_path=self.metrics_path,
            )
