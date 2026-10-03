"""Small deterministic TF-IDF retriever for fixture documents and citations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


class FixtureRetriever:
    """In-memory retriever; fixture corpora keep tests and demos offline."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("retrieval corpus must contain at least one chunk")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("chunk IDs must be unique")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, str | float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))[:top_k]
        return [
            {"chunk_id": self.chunks[i].chunk_id, "source": self.chunks[i].source,
             "text": self.chunks[i].text, "score": float(scores[i])}
            for i in ranked if scores[i] > 0
        ]
