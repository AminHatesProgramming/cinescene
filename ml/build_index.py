import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ast

PROCESSED = Path("data/processed")
INDEX_DIR = Path("data/index")
MODEL_DIR = Path("model")
MODEL_BASE_DIR = Path("model/base")
INDEX_DIR.mkdir(exist_ok=True)

def load_model():
    # اول چک می‌کنیم مدل fine-tuned وجود داره
    if (MODEL_DIR / "config.json").exists():
        print("Loading fine-tuned model...")
        return SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    
    # اگه نه، از مدل base استفاده می‌کنیم
    if not (MODEL_BASE_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Base model not found in {MODEL_BASE_DIR}. "
            "Please download all-MiniLM-L6-v2 files first."
        )
    
    print("Loading base model...")
    return SentenceTransformer(str(MODEL_BASE_DIR), local_files_only=True)

def main():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED / "movies_clean.csv")

    for col in ["genres", "keywords", "cast"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    texts = df["rich_text"].tolist()
    print(f"Encoding {len(texts)} movies...")

    model = load_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}")

    # FAISS index با Inner Product
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"Index size: {index.ntotal}")

    # ذخیره index
    faiss.write_index(index, str(INDEX_DIR / "movies.index"))

    # ذخیره metadata
    metadata = df[["id", "title", "overview", "tagline", "genres",
                    "cast", "director", "vote_average", "vote_count",
                    "popularity", "rich_text"]].to_dict(orient="records")

    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved index and metadata to {INDEX_DIR}")

if __name__ == "__main__":
    main()
