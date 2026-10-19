"""Centralised configuration, loaded once from environment / .env.

Everything else imports :func:`get_settings`; no module reads ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Repository root. Configured paths are relative to it, not to the CWD."""


class Settings(BaseSettings):
    """Runtime settings. Override via ``.env`` or ``APEX_``-prefixed env vars."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APEX_", extra="ignore")

    # --- PostgreSQL -------------------------------------------------------
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "apex"
    pg_password: str = "apex"
    pg_database: str = "apex_rag"
    pg_connect_timeout: int = 10

    # A complete connection string, for CI and for reviewers pointing the
    # benchmark at a database they did not create. It wins over the individual
    # fields above. Setting ``validation_alias`` opts this one field out of the
    # ``APEX_`` prefix, so both ``DATABASE_URL`` and ``APEX_DATABASE_URL`` work.
    #
    # Caveat when running inside compose: the app reaches the database at host
    # ``db``, so a DATABASE_URL holding ``localhost`` breaks the container path.
    # Leave it unset unless you mean it.
    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("APEX_DATABASE_URL", "DATABASE_URL"),
    )

    # --- Embedding --------------------------------------------------------
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(
        default=32,
        description="Texts per forward pass. Bounds peak memory on a CPU-only host.",
    )
    embedding_max_retries: int = Field(
        default=3,
        description="Attempts per batch before the ingest aborts. Backoff is 1s, 2s, 4s.",
    )
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- Reranker ---------------------------------------------------------
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    reranker_device: str = "cpu"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512
    reranker_local_files_only: bool = True
    reranker_cache_folder: Path = Path(".cache/huggingface/hub")

    # --- Generation provider (Phase 8) -----------------------------------
    # OpenAI-compatible endpoint. The key is read from the environment or .env
    # and is never printed. APEX_LLM_API_KEY is the project name; the DeepSeek
    # alias lets an existing environment variable be reused without copying it.
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("APEX_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_connect_timeout: float = Field(default=5.0, gt=0.0, le=120.0)
    llm_read_timeout: float = Field(default=60.0, gt=0.0, le=600.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1, le=32768)

    # --- RAG answer chain (Phase 8) --------------------------------------
    rag_candidates: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Candidates retrieved per arm and passed to the reranker.",
    )
    rag_top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Final evidence pool size after reranking.",
    )
    rag_max_context_chars: int = Field(default=8000, ge=200)
    rag_max_chunk_chars: int = Field(default=2000, ge=1)

    # --- Ingestion --------------------------------------------------------
    expected_document_count: int = Field(
        default=150,
        description="The loader fails the import when the corpus is not exactly this size.",
    )

    # --- Paths ------------------------------------------------------------
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")

    def resolve(self, path: Path) -> Path:
        """Anchor a configured relative path at the project root.

        Settings are read from ``.env`` and from the environment, so a path in
        them is written relative to the repository -- not to wherever the
        process happens to have been started.
        """
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def dsn(self) -> str:
        """Return a libpq connection string; ``database_url`` wins when it is set."""
        if self.database_url:
            return self.database_url
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_database} "
            f"user={self.pg_user} password={self.pg_password} "
            f"connect_timeout={self.pg_connect_timeout}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
