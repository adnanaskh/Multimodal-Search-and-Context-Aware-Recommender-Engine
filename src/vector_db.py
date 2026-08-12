import faiss
import numpy as np


class VectorDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def build_index(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings shape (n, {self.dim})")
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 10):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.dim:
            raise ValueError(f"Expected query dimension {self.dim}")
        distances, indices = self.index.search(query_embedding, k)
        return distances[0].tolist(), indices[0].tolist()

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
