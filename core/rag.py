"""Local RAG indexing and retrieval backed by ChromaDB and Ollama."""

from pathlib import Path
from typing import Iterable

import pymupdf
import ollama


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
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embed_model = embed_model
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
                    for index, paragraph in enumerate(
                        chunk for chunk in text.split("\n\n") if chunk.strip()
                    ):
                        chunk_text = paragraph.strip()
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
        if self._collection is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            except ImportError as exc:
                raise RuntimeError("ChromaDB is not installed") from exc

        if self._collection.count() == 0:
            return []
        embedding = self._embed(text)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            f"[Source: {meta.get('source', 'unknown')}]\n{doc}"
            for doc, meta in zip(documents, metadatas)
        ]
