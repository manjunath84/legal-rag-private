# src/raglab/config.py
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config in one place, overridable via `.env` (env_prefix RAGLAB_).

    Defaults point at the user's local mlx_vlm.server so the app is private by
    default — no query or document data leaves the machine. Same provider-agnostic
    `openai`-SDK pattern as Week 1's narrate.py, just pointed local.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAGLAB_", extra="ignore")

    # --- Local LLM (OpenAI-compatible endpoint) ---
    llm_base_url: str = "http://localhost:8085/v1"  # env: RAGLAB_LLM_BASE_URL
    llm_api_key: str = "local-no-key-needed"  # local server ignores it; any non-empty string
    llm_model: str = "mlx-community/gemma-4-12B-it-4bit"  # env: RAGLAB_LLM_MODEL
    llm_temperature: float = 0.2
    llm_max_tokens: int = 800

    # --- Embeddings (local, in-process) ---
    embed_model: str = "BAAI/bge-small-en-v1.5"  # env: RAGLAB_EMBED_MODEL

    # --- Retrieval / chunking ---
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 150
    top_k: int = 5
    # "hybrid" = BM25 + vector fused with RRF (default); "vector" = dense only.
    retrieval_mode: str = "hybrid"  # env: RAGLAB_RETRIEVAL_MODE
    candidate_k: int = 10  # how many each retriever contributes before fusion
    rrf_k: int = 60  # Reciprocal Rank Fusion constant (standard default)

    # --- Storage paths ---
    data_dir: Path = Path("data/legal")
    chroma_dir: Path = Path(".chroma")
    collection: str = "legal"


settings = Settings()
