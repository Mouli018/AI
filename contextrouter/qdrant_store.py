"""Qdrant Vector DB integration for ContextRouter document collections.

Supports:
1. Local embedded storage (default: data/qdrant_storage, zero external services needed)
2. Remote Qdrant server/cloud via QDRANT_URL and optional QDRANT_API_KEY
"""
from __future__ import annotations
import collections
import math
import os
import re
import uuid
from typing import Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from .schemas import Chunk

DEFAULT_COLLECTION = "hotpotqa_corpus"
EMBEDDED_PATH = "data/qdrant_storage"
VECTOR_DIM = 256


def text_to_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Lightweight term-frequency sign-hashing vectorizer.
    
    Fast, memory-efficient, and requires zero heavy external C-libraries/DLLs.
    Maps arbitrary text directly to a normalized cosine vector.
    """
    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
    if not words:
        return [0.0] * dim
    counts = collections.Counter(words)
    v = np.zeros(dim, dtype=np.float32)
    for word, cnt in counts.items():
        h = hash(word)
        idx = abs(h) % dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        v[idx] += sign * (1.0 + math.log(cnt))
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v.tolist()


class QdrantStore:
    def __init__(self, collection_name: str = DEFAULT_COLLECTION, path: str = EMBEDDED_PATH):
        self.collection_name = collection_name
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)

    def init_collection(self, vector_dim: int = VECTOR_DIM, force_recreate: bool = False):
        """Ensure the collection exists in Qdrant."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if force_recreate:
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    def index_corpus(self, docs: list[dict], batch_size: int = 250) -> int:
        """Store documents in the Qdrant collection with rich metadata payloads."""
        self.init_collection(vector_dim=VECTOR_DIM, force_recreate=True)

        points = []
        for i, d in enumerate(docs):
            full_text = d["title"] + ". " + " ".join(d["sentences"])
            vec = text_to_vector(full_text, dim=VECTOR_DIM)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.collection_name}:{d['title']}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "title": d["title"],
                        "sentences": d["sentences"],
                        "text": full_text,
                        "num_sentences": len(d["sentences"]),
                    },
                )
            )

        # Batch upload to Qdrant
        for start in range(0, len(points), batch_size):
            end = start + batch_size
            self.client.upsert(collection_name=self.collection_name, points=points[start:end])

        return len(points)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Query Qdrant collection and return typed Chunks."""
        qv = text_to_vector(query, dim=VECTOR_DIM)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=qv,
            limit=k,
        ).points

        chunks = []
        for r in results:
            payload = r.payload or {}
            chunks.append(
                Chunk(
                    title=payload.get("title", ""),
                    sentences=payload.get("sentences", []),
                    score=float(r.score),
                    query=query,
                )
            )
        return chunks

    def get_info(self) -> dict[str, Any]:
        """Return collection telemetry for diagnostics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "status": "active",
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.name,
                "mode": "remote" if os.getenv("QDRANT_URL") else "embedded (local disk)",
            }
        except Exception as e:
            return {
                "collection": self.collection_name,
                "status": "not_initialized",
                "error": str(e),
            }
