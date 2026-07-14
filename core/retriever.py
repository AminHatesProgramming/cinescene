import faiss
import pickle
import numpy as np
from pathlib import Path
from core.embedder import Embedder

INDEX_DIR = Path("data/index")

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index    = faiss.read_index(str(INDEX_DIR / "movies.index"))
        with open(INDEX_DIR / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Index loaded: {self.index.ntotal} movies")

    def search(self, query: str, top_k: int = 10) -> list:
        vec = self.embedder.embed(query)
        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            movie = self.metadata[idx].copy()
            movie["score"] = float(score)
            results.append(movie)

        return results
