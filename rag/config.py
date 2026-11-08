from pathlib import Path

# --- RAG model configuration ---
LLM_MODEL: str = "openai/gpt-oss-120b"

EMBEDDING_MODEL_NAME: str = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Retrieval settings ---
SIMILARITY_TOP_K: int = 4
CHUNK_SIZE: int = 256
CHUNK_OVERLAP: int = 50

# --- Paths ---
ROOT_PATH: Path = Path(__file__).parent.parent

RAG_DATA_PATH: Path = ROOT_PATH / "rag" / "knowledge"

VECTOR_STORE_PATH: Path = (
    ROOT_PATH / "rag" / "vector_store"
)

EMBEDDING_CACHE_PATH: Path = (
    ROOT_PATH / "rag" / "embedding_cache"
)