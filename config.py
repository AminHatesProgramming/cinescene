from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    base_model: str = "BAAI/bge-large-en-v1.5"
    local_base_model: Path = Path("models/bge-large-en-v1.5")
    finetuned_model: Path = Path("models/cinescene-v2/final")
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")
    offline_video_dir: Path = Path("data/offline_videos")
    video_ingestion_dir: Path = Path("data/processed/video_ingestion")
    app_memory_db: Path = Path("data/app_memory.sqlite")

    tmdb_processed: Path = processed_dir / "tmdb_processed.json"
    movies_enriched: Path = processed_dir / "movies_enriched.json"
    triplets: Path = processed_dir / "triplets_v2.json"
    faiss_index: Path = index_dir / "faiss_index_v2.bin"
    metadata: Path = index_dir / "metadata_v2.pkl"


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    use_curriculum: bool = True
    triplets_per_movie: int = 3
    hard_negative_ratio: float = 0.7


@dataclass
class SearchConfig:
    vector_k: int = 60
    lexical_k: int = 60
    rrf_k: int = 60
    rerank_top_k: int = 20
    default_top_k: int = 8
    use_reranking: bool = True


@dataclass
class IndexConfig:
    use_hnsw: bool = True
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.search = SearchConfig()
        self.index = IndexConfig()

    def create_directories(self):
        for path in [
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.index_dir,
            self.data.offline_video_dir,
            self.data.video_ingestion_dir,
            Path("models"),
            Path("logs"),
        ]:
            path.mkdir(parents=True, exist_ok=True)
        print("Project directories are ready")


config = Config()
