"""Central settings. Every value can be overridden in a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env(name: str, default):
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # where parsed chunks, figures and the local vector DB live (git-ignored)
    data_dir: Path = Path(_env("DATA_DIR", ROOT / "data"))

    # Zotero (local API: Zotero 7 → Settings → Advanced → "Allow other applications...")
    zotero_storage: Path = Path(_env("ZOTERO_STORAGE", Path.home() / "Zotero" / "storage"))

    # Vector store: embedded Qdrant on disk by default; set QDRANT_URL to use a server
    qdrant_url: str | None = _env("QDRANT_URL", None)
    collection: str = _env("QDRANT_COLLECTION", "mg_battery")
    domain: str = _env("DOMAIN", "magnesium-battery")

    # Models (all run on CPU; see README for GPU upgrades)
    dense_model: str = _env("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model: str = _env("SPARSE_MODEL", "Qdrant/bm25")
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # LLMs through LiteLLM: "ollama/qwen3:8b", "anthropic/<model>", "openai/<model>", ...
    llm_model: str = _env("LLM_MODEL", "ollama/qwen3:8b")
    vision_model: str = _env("VISION_MODEL", _env("LLM_MODEL", "ollama/qwen3:8b"))
    judge_model: str = _env("JUDGE_MODEL", _env("LLM_MODEL", "ollama/qwen3:8b"))

    # Retrieval knobs
    max_chunk_tokens: int = int(_env("MAX_CHUNK_TOKENS", 400))
    top_k_retrieve: int = int(_env("TOP_K_RETRIEVE", 30))
    top_k_context: int = int(_env("TOP_K_CONTEXT", 6))
    # Guardrail: abstain when the best reranked chunk scores below this (0..1)
    min_rerank_score: float = float(_env("MIN_RERANK_SCORE", 0.25))

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"


settings = Settings()
