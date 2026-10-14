"""Paragraph retriever: TF-IDF (default), dense (MiniLM + FAISS/numpy) or hybrid."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .schemas import Chunk


class Retriever:
    def __init__(self, docs: list[dict], kind: str = "tfidf"):
        self.docs = docs
        self.kind = kind
        self.texts = [d["title"] + ". " + " ".join(d["sentences"]) for d in docs]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.texts)
        names = self.vec.get_feature_names_out()
        self._idf = dict(zip(names, self.vec.idf_))
        self._max_idf = float(self.vec.idf_.max())
        self.qdrant = None
        if kind == "qdrant":
            from .qdrant_store import QdrantStore
            self.qdrant = QdrantStore()
            info = self.qdrant.get_info()
            if info.get("status") != "active" or info.get("points_count", 0) == 0:
                self.qdrant.index_corpus(self.docs)
        elif kind in ("dense", "hybrid"):
            self._init_dense()

    def _init_dense(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emb = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb.astype("float32"))
        except Exception:
            self.index = None

    def idf(self, term: str) -> float:
        return self._idf.get(term, self._max_idf)

    @staticmethod
    def _mm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def search(self, query: str, k: int) -> list[Chunk]:
        if self.kind == "qdrant" and self.qdrant is not None:
            return self.qdrant.search(query, k)
        sp = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        if self.emb is not None:
            qe = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            de = self.emb @ qe
            scores = de if self.kind == "dense" else 0.5 * self._mm(sp) + 0.5 * self._mm(de)
        else:
            scores = sp
        idx = np.argsort(-scores)[:k]
        return [Chunk(title=self.docs[i]["title"], sentences=self.docs[i]["sentences"],
                      score=float(scores[i]), query=query) for i in idx]
