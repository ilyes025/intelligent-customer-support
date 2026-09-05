"""
Application configuration settings using Pydantic.
"""

import os
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings class."""

    PROJECT_NAME: str = "Intelligent Support Agent API"
    PROJECT_DESCRIPTION: str = (
        "Support agent API with confidence-based response validation, "
        "context-aware routing and multi-source retrieval."
    )
    VERSION: str = "0.1.0"
    API_PREFIX: str = ""
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    DEV_MODE: bool = DEBUG

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    @field_validator("CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string to list if needed."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        # covers JSON-style list string, e.g. '["http://a.com", "http://b.com"]'
        elif isinstance(v, str):
            import json

            try:
                parsed = json.loads(v)
                if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
                    return parsed
            except Exception:
                pass
        raise ValueError(
            f"CORS_ORIGINS must be a list of strings or a comma-separated string. cors value{v}"
        )

    # Database
    DB_ENGINE: str = os.getenv("DB_ENGINE", "sqlite")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: str = os.getenv("DB_PORT", "")
    DB_NAME: str = os.getenv("DB_NAME", "db.sqlite3")

    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL based on configuration."""
        if self.DB_ENGINE == "sqlite":
            return f"sqlite+aiosqlite:///{self.DB_NAME}"
        elif self.DB_ENGINE == "postgresql":
            return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        return f"{self.DB_ENGINE}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def TEST_DATABASE_URL(self) -> str:
        """Construct database URL based on configuration."""
        if self.DB_ENGINE == "sqlite":
            return f"sqlite+aiosqlite:///{self.DB_NAME}-test"
        elif self.DB_ENGINE == "postgresql":
            return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}-test"
        return f"{self.DB_ENGINE}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}-test"

    # --- Support Agent ---------------------------------------------------

    # External data source (mock support/user API)
    EXTERNAL_API_BASE_URL: str = os.getenv(
        "EXTERNAL_API_BASE_URL", "https://jsonplaceholder.typicode.com"
    )
    EXTERNAL_API_TIMEOUT_SECONDS: float = float(
        os.getenv("EXTERNAL_API_TIMEOUT_SECONDS", "3.0")
    )
    EXTERNAL_API_CACHE_SIZE: int = int(os.getenv("EXTERNAL_API_CACHE_SIZE", "256"))
    EXTERNAL_API_FAILURE_THRESHOLD: int = int(
        os.getenv("EXTERNAL_API_FAILURE_THRESHOLD", "3")
    )
    EXTERNAL_API_COOLDOWN_SECONDS: float = float(
        os.getenv("EXTERNAL_API_COOLDOWN_SECONDS", "30.0")
    )

    # Hugging Face (optional enhancement, never a hard dependency)
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_ZERO_SHOT_MODEL: str = os.getenv(
        "HF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli"
    )
    HF_API_TIMEOUT_SECONDS: float = float(os.getenv("HF_API_TIMEOUT_SECONDS", "5.0"))

    # Hugging Face generative model (used for RAG answer composition and,
    # for internal drafts only, ungrounded generation - see app/services/agent.py)
    #
    # NOTE: HF's Inference Providers catalog changes over time - models get
    # dropped by providers with no warning (zephyr-7b-beta, the old default,
    # was fully discontinued). If generation starts silently returning None,
    # check https://router.huggingface.co/v1/models for currently-served
    # models before assuming the code is broken - see diagnose_hf.py.
    HF_GENERATION_MODEL: str = os.getenv(
        "HF_GENERATION_MODEL", "Qwen/Qwen2.5-7B-Instruct"
    )
    # Optional: pin a specific inference provider (e.g. "novita", "together",
    # "fireworks-ai") by appending ":<provider>" to the model id. Leave blank
    # to let the router auto-select among providers enabled on this token.
    HF_GENERATION_PROVIDER: str = os.getenv("HF_GENERATION_PROVIDER", "")
    HF_GENERATION_MAX_TOKENS: int = int(os.getenv("HF_GENERATION_MAX_TOKENS", "300"))
    HF_GENERATION_TEMPERATURE: float = float(
        os.getenv("HF_GENERATION_TEMPERATURE", "0.3")
    )

    @property
    def HF_ENABLED(self) -> bool:
        return bool(self.HF_API_TOKEN)

    # Knowledge base / retrieval
    KB_MIN_RELEVANCE_SCORE: float = float(os.getenv("KB_MIN_RELEVANCE_SCORE", "0.12"))
    KB_TOP_K: int = int(os.getenv("KB_TOP_K", "3"))

    # "tfidf"  - original dependency-free TF-IDF cosine (always available)
    # "bm25"   - BM25Okapi lexical ranking (pure Python, rank_bm25 package)
    # "hybrid" - BM25 + dense embeddings, fused via reciprocal-rank fusion.
    #            Falls back to BM25-only automatically if HF embeddings are
    #            unavailable/disabled, so "hybrid" is always safe to set.
    KB_RETRIEVAL_MODE: str = os.getenv("KB_RETRIEVAL_MODE", "hybrid")
    KB_RRF_K: int = int(os.getenv("KB_RRF_K", "60"))  # standard RRF constant

    # Embedding model for the dense half of hybrid retrieval. Run LOCALLY via
    # the `sentence-transformers` package rather than HF's hosted Inference
    # API - HF has been progressively dropping free-tier serverless hosting
    # for sentence-transformer embedding models, so the API path became
    # unreliable. Running locally also removes a network round-trip from the
    # hot retrieval path and has no dependency on HF_API_TOKEN at all. See
    # app/clients/huggingface_client.py:embed_text.
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Faithfulness checking (Theme 1: verify, don't just prompt for it) --
    # Below this score, a generated answer is treated as insufficiently
    # grounded in its retrieved context and the routing decision is
    # downgraded a step (auto_send -> pending_validation -> escalated).
    FAITHFULNESS_THRESHOLD: float = float(
        os.getenv("FAITHFULNESS_THRESHOLD", "0.55")
    )
    # NLI model used to score entailment between answer and context when HF
    # is enabled. When HF is disabled, faithfulness falls back to a lexical
    # coverage heuristic (see app/services/faithfulness.py) rather than being
    # skipped entirely - there is always a check, just a weaker one.
    HF_NLI_MODEL: str = os.getenv("HF_NLI_MODEL", "facebook/bart-large-mnli")

    # Confidence-based routing thresholds (Theme 1)
    CONFIDENCE_AUTO_THRESHOLD: float = float(
        os.getenv("CONFIDENCE_AUTO_THRESHOLD", "0.75")
    )
    CONFIDENCE_VALIDATION_THRESHOLD: float = float(
        os.getenv("CONFIDENCE_VALIDATION_THRESHOLD", "0.40")
    )

    # Below this intent-classification confidence, ask a clarifying question
    # instead of guessing (Theme 3).
    INTENT_CLARIFICATION_THRESHOLD: float = float(
        os.getenv("INTENT_CLARIFICATION_THRESHOLD", "0.35")
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()