from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

MODEL_BASE_DIR = Path("model/base")

class Embedder:
    def __init__(self):
        if not (MODEL_BASE_DIR / "config.json").exists():
            raise FileNotFoundError(
                f"Model files not found in {MODEL_BASE_DIR}. "
                "Please download all-MiniLM-L6-v2 files and place them there."
            )
        
        self.model = SentenceTransformer(
            str(MODEL_BASE_DIR),
            local_files_only=True
        )

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vec.astype(np.float32)

    def embed_batch(self, texts: list) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vecs.astype(np.float32)
